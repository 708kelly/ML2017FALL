
import csv
import os
import numpy as np
import sys
import _pickle as pk
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,load_model
path_1 = sys.argv[1]
text = open(path_1, 'r', encoding='UTF-8') 
test_data = []
lines=text.readlines()
content = [x.strip() for x in lines]
#x=[]
n_line = 0
for l in content:
    if n_line != 0:
        short = l.split(',',1)
        a = short[1].strip()
        test_data.append(a)
    n_line = n_line+1
text.close()

path = 'token.pkl'
tokenizer = pk.load(open(path, 'rb'))

maxlen = 30

sequences = tokenizer.texts_to_sequences(test_data)
#x_data = np.array(sequences)
test_x = np.array(pad_sequences(sequences, maxlen=maxlen))

model = load_model('hw4_2_1.h5')
p_result = model.predict(test_x, verbose=1)
result = np.around(p_result)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])

for i, v in  enumerate(result):
        s.writerow([(i),int(v)])
text.close()
