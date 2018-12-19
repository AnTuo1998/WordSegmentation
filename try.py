from config import MAX_SEQ_LEN
from test import division
from collections import Counter

# line1 ='sf---fsf。'
# line = division(line1)
# print(line,type(line))

file = open("data/train.txt",'r',encoding='utf-8')
num=[]
for x in file:
    one_list = ''.join(x.split())
    num.append(len(one_list))
    if len(one_list)>550:
        print(x)
print(max(num))

