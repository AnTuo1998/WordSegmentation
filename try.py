<<<<<<< HEAD
<<<<<<< HEAD
# from config import MAX_SEQ_LEN
# from test import division
# from collections import Counter

# # line1 ='sf---fsf。'
# # line = division(line1)
# # print(line,type(line))

# file = open("data/train.txt",'r',encoding='utf-8')
# num=[]
# for x in file:
#     one_list = ''.join(x.split())
#     num.append(len(one_list))
#     if len(one_list)>550:
#         print(x)
# print(max(num))
# from train import myplot
# class H(object):
#         def __init__(self):
#                self.history = {'acc':[1,2,3],'val_acc':[4,5,6],'loss':[7,8,9],'val_loss':[10,11,12]}
# from config import EPOCH_SIZE
# EPOCH_SIZE = 3
# history = H()
# myplot(history)
import matplotlib.pyplot as plt
import numpy as np
infile = open("data/train.txt", 'r',encoding = 'utf-8')
count1 = 0
count2 = 0 
num = []
num1 = [0] * 20
num5 = [0] * 5
num2 = 0
num3 = 0
num4 = 0
for line in infile:
	myline = ''.join(line.split())
	num.append(len(myline))
	if len(myline) > 200:
		num4+=1
	elif len(myline) >= 100:
		if len(myline) >=150:
			num3 += 1
		else:
			pos = int((len(myline)- 100)/10)
			num5[pos] += 1 
	else:
		pos = int(len(myline) / 5)
		num1[pos] += 1
	count1 += 1
	# if count % 5 == 0:
		#print(myline)
		# print('process {}%\r'.format(count / 3985 * 100))
print(count1, count2)
num1 += num5
num1.append(num3)
num1.append(num4)
num_col = ["[0,5)","[5,10)","[10,15)","[15,20)","[20,25)","[25,30)","[30,35)","[35,40)","[40,45)",
"[45,50)","[50,55)","[55,60)","[60,65)","[65,70)","[70,75)","[75,80)","[80,85)","[85,90)","[90,95)",
"[95,100)","[100,110)","[110,120)","[120,130)","[130,140)","[140,150)","[150,200)",">=200"]
#num1.loc[0,num_col].values
plt.xticks(np.arange(1,28),labels=num_col,rotation=45)
position = np.arange(1,28)
plt.bar(position,num1,0.8)
plt.legend()

plt.xlabel('length of sentences')
plt.ylabel('frequency')

plt.title('Distribution of the sentences in train data')
plt.show()
=======
=======
>>>>>>> parent of 4b254b9... complement model loading
from config import MAX_SEQ_LEN
from test import division

line1 ='sf---fsf。'
line = division(line1)
<<<<<<< HEAD
print(line,type(line))
>>>>>>> parent of 4b254b9... complement model loading
=======
print(line,type(line))
>>>>>>> parent of 4b254b9... complement model loading
