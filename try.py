from config import MAX_SEQ_LEN
from test import division

line2 = '第一条根据中华人民共和国宪法、中华人民共和国香港特别行政区基本法以及中华人民共和国全国人民代表大会和地方各级人民代表大会选举法第十五条第三款的规定，结合香港特别行政区的实际情况，制定本办法。'
line1 ='sf---fsf。'
line = division(line1)
print(line,type(line))