from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import numpy as np

np.random.seed(3)

data="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c, i) for i, c in enumerate(data))

int_to_char=dict((i,c)for i,c in enumerate(data))

length=1
X=[]
Y=[]

for i in range(0,len(data)-length)
    inp=int_to_char[i]
    oup=int_to_char[i+1]
    X.append(char_to_int[inp])
    Y.append(char_to_int[oup])

X=np.reshape(X,(len(X),1,1))

X=X/len(data)

Y=np_utils.to_categorical(Y)

model=Sequential()
model.add(LSTM(32,input_shape=(1,1)))
model.add(Dense(32,activation="relu"))
model.add(Dense(26,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X,Y,batch_size=len(X),verbose=1,epochs=100)

pred=model.evaluate(X,Y)

#print("Accuracy is : ",pred[1]*100)

try:
    f=open("/newdir/o.txt","w")
    f.write(str(int(pred[1]*100)))
except:
    print(end="")
finally:
    f.close()
