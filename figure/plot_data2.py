import matplotlib.pyplot as plt
import numpy as np
from config import FIGURE_PATH


def division(line :str) -> list:
    if len(line) < 45:
        return [line]
    sentenceList = []
    SplitWordStr = "，。！？、／：；《》（）、"
    sentence = ""

    for i in range(len(line)):
        sentence += line[i]
        if line[i] in SplitWordStr and i + 1 < len(line) and line[i + 1] not in SplitWordStr:
            sentenceList.append(sentence)
            sentence = ""

    if sentence != "":
        sentenceList.append(sentence)
    return sentenceList

infile = open("data/train.txt", 'r',encoding = 'utf-8')

num =[0]*21
for line in infile:
	mylist = division(line[:-1])
	for ele in mylist:
		if len(ele) == 1:
			print(ele)
		if len(ele) <= 100:
			pos = int(len(ele)/5)
			num[pos] += 1
		else:
			num[20] += 1
num_col = ["[0,5)","[5,10)","[10,15)","[15,20)","[20,25)","[25,30)","[30,35)","[35,40)","[40,45)",
"[45,50)","[50,55)","[55,60)","[60,65)","[65,70)","[70,75)","[75,80)","[80,85)","[85,90)","[90,95)","[95,100)",">=100"]

plt.xticks(np.arange(1,28),labels=num_col,rotation=45)
print(len(num))
position = np.arange(1,22)
plt.bar(position,num,0.8)
plt.legend()

plt.xlabel('length of sentences')
plt.ylabel('frequency')
plt.title('Distribution of the divided sentences in train data')
plt.show()
plt.savefig(FIGURE_PATH + "train_div_data.png")