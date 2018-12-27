import matplotlib.pyplot as plt
import numpy as np
from config import FIGURE_PATH
infile = open("data/train.txt", 'r',encoding = 'utf-8')
num = [0] * 25
for line in infile:
	myline = ''.join(line.split())
	if len(myline) > 200:
		num[24]+=1
	elif len(myline) >= 100:
		if len(myline) >=150:
			num[23] += 1
		else:
			pos = int((len(myline)- 100)/10)
			num[pos+18] += 1 
	else:
		if len(myline) < 15:
			num[0] += 1
		else:
			pos = int(len(myline) / 5)
			num[pos - 2] += 1
print(len(num))
num_col = ["[0,15)","[15,20)","[20,25)","[25,30)","[30,35)","[35,40)","[40,45)",
"[45,50)","[50,55)","[55,60)","[60,65)","[65,70)","[70,75)","[75,80)","[80,85)","[85,90)","[90,95)",
"[95,100)","[100,110)","[110,120)","[120,130)","[130,140)","[140,150)","[150,200)",">=200"]
plt.xticks(np.arange(1,26),labels=num_col,rotation=45)
position = np.arange(1,26)
plt.bar(position,num,0.8)
plt.legend()

plt.xlabel('length of sentences')
plt.ylabel('frequency')

plt.title('Distribution of the sentences in train data')
plt.show()
plt.savefig(FIGURE_PATH + "train data.png")