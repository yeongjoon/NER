import random

whole_data = []
seed = 42

data_dir = '/home/yeongjoon/data/Naver_NER/'
with open(data_dir + 'changed_train.txt', 'r') as f:
    for line in f.read().split('\n\n'):
        whole_data.append(line.strip())

random.Random(seed).shuffle(whole_data)

splited = [None]*3
splited[0], splited[1], splited[2] = whole_data[:70000], whole_data[70000:80000], whole_data[80000:-1]

name = ['train', 'dev', 'test']

for n, s in zip(name, splited):
    with open(data_dir + n + '.txt', 'w') as f:
        for paragraph in s:
            for line in paragraph.split('\n'):
                f.write(line)
                f.write('\n')
            f.write('\n')
