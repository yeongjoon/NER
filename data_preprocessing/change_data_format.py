import random
whole_data = []
seed = 42

with open('/home/yeongjoon/data/Naver_NER/raw.txt', 'r') as f:
    for line in f.readlines():
        if len(line) > 1:
            whole_data.append(line.split('\t')[1] + ' ' + line.split('\t')[2].replace('-','O'))
        else:
            whole_data.append(line)

with open('/home/yeongjoon/data/Naver_NER/changed_train.txt', 'w') as f:
    for data in whole_data:
        f.write(data)
