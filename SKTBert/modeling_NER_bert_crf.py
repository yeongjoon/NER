from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
from pytorchcrf import CRF

class BertCRFForTokenClassification(BertPreTrainedModel):
    """ New class for BERT + CRF NER Tagging based on huggingface bert model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # CRF Module
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # CRF 수정
        #TODO CRF 형태로 꼴 바꾸기
        # CRF같은 경우에는 의미 있는 단어 토큰들끼리만 전파가 되어야 하므로 전에 하던것처럼 BERT OUTPUT 토큰
        # 맨 처음과 맨 끝 ([CLS], [SEP]) 제외하고, 중간중간 token이 원래 토큰보다 하나 더 이상 잘라진거 처리 해야함
        # 현재로서는 이 내부에서 잘라서 붙이는거 말고는 방법이 없을듯?
        # 코드 수정을 덜 하려면 CRF는 붙여서 통과시키고, 나온 결과는 다시 원래대로 흐트러 놓는게 가장 좋음
        # index 뽑아서 valid한 index만 따로 만들고, 기억해 두었다가 다시 붙이는 방식 예정
        # or 맨 앞에꺼만 떼서 CRF 넣고 다시 붙이는 방식

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # CRF 함수 문제로 label의 ignore_index -100을 다른 수로 바꿔 주어야 함

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # CRF 위해서 mask 생성 ( -100 index 처리를 위해서 )
            mask_tensor = torch.where(
                labels >= 0, torch.ones(labels.shape).to(logits.device),
                torch.zeros(labels.shape).to(logits.device)
            )
            mask_tensor = mask_tensor.to(torch.uint8)

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )


                # active 변수들 원래대로 꼴 원상복귀 [batch_size, seq_length]

                active_logits = active_logits.view(input_ids.shape[0], input_ids.shape[-1], -1)
                active_labels = active_labels.view(input_ids.shape[0], -1)

                # CRF  꼴을 맞춰주기 위해서 맨 앞의 [CLS] 토큰 부분을 강제로 제거하고 학습 시작
                a = active_logits[:, 1:]
                b = active_labels[:, 1:]
                c = mask_tensor[:, 1:]
                b = torch.where(c > 0, b, c.type(torch.long))
                loss, sequence_of_tags = -1 * self.crf(a, b, mask=c), self.crf.decode(a)

                # loss, sequence_of_tags = -1 * self.crf(active_logits, active_labels, mask=mask_tensor), self.crf.decode(active_logits)
                # loss = loss_fct(active_logits, active_labels)
            else:
                loss, sequence_of_tags = -1 * self.crf(logits, labels), self.crf.decode(logits)
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        else:   # inference
            # 이 부분도 CRF 꼴을 위해 맨 앞의 [CLS] 토큰 부분 제거
            # 학습도 제거하고 했으니까 decode도 우선은 빼고 한 후에 갖다 붙이는게 맞을 듯?
            sequence_of_tags = self.crf.decode(logits[:, 1:])

        # Sequence_of_tags의 맨 앞 [CLS] 토큰 삭제한 부분을 다시 채워넣어야 하기 때문에 Dummy 토큰을 사용
        # 이 때 어차피 f1 score 계산시에 이 부분은 빠지기 때문에 값에 대한 상관은 없음
        dummy_tag = torch.zeros([input_ids.shape[0], 1], dtype=torch.long).to(sequence_of_tags.device)
        sequence_of_tags = torch.cat((dummy_tag, sequence_of_tags.squeeze(0)), dim=-1) # [batch, seq]

        outputs = outputs + (sequence_of_tags,)
        return outputs  # (loss), scores, (hidden_states), (attentions)

if __name__ == '__main__':
    import logging

    print("Why not logging")