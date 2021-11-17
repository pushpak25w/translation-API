import pandas as pd
import numpy as np
import string
from string import digits
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
data=pd.read_table('french.txt',names=['lang','trans','useless'])
data.drop('useless',axis=1,inplace=True)
data1=data
data=data1
spchrs=set(string.punctuation)
lowerCase=lambda x:x.lower()
quotes=lambda x:re.sub("'",'',x)
specialCh=lambda x:''.join(ch for ch in x if ch not in spchrs)
rmDigits=lambda x:x.translate(str.maketrans('','',digits))
spaces=lambda x:x.strip()
unwanted=lambda x:re.sub(" +"," ",x)
startEnd=lambda x:'START_'+x+'_END'
data.lang=data.lang.apply(lowerCase)
data.lang=data.lang.apply(quotes)
data.lang=data.lang.apply(specialCh)
data.lang=data.lang.apply(rmDigits)
data.lang=data.lang.apply(spaces)
data.lang=data.lang.apply(unwanted)
data.trans=data.trans.apply(lowerCase)
data.trans=data.trans.apply(quotes)
data.trans=data.trans.apply(specialCh)
data.trans=data.trans.apply(rmDigits)
data.trans=data.trans.apply(spaces)
data.trans=data.trans.apply(unwanted)
data.trans=data.trans.apply(startEnd)
langVocab=set()
for line in data.lang:
    for word in line.split():
        langVocab.add(word)
transVocab=set()
for line in data.trans:
    for word in line.split():
        transVocab.add(word)
maxSrcLen=0
for line in data.lang:
    maxSrcLen=max(maxSrcLen,len(line.split()))
print(maxSrcLen)
maxTarLen=0
for line in data.trans:
    maxTarLen=max(maxTarLen,len(line.split()))
print(maxTarLen)
inputWords=sorted(list(langVocab))
targetWords=sorted(list(transVocab))
lenOfEncoderTokens=len(langVocab)
lenOfDecoderTokens=len(transVocab)
print(lenOfDecoderTokens,lenOfEncoderTokens)
lenOfDecoderTokens+=1
lenOfDecoderTokens
tarTokenInd,inpRevIndMap,tarRevIndMap,inpTokenInd={},{},{},{}
for i,word in enumerate(inputWords):
    inpTokenInd[word]=i+1
    inpRevIndMap[i]=word
for i,word in enumerate(targetWords):
    tarTokenInd[word]=i+1
    tarRevIndMap[i]=word
data=shuffle(data)
x,y=data.lang,data.trans
x_train,x_test,y_train,y_test=train=train_test_split(x,y,test_size=0.1)
def encode(x=x_train,y=y_train,size=128):
    while True:
        for i1 in range(0,len(x),size):
            encInp=np.zeros((size,maxSrcLen),dtype='float32')
            decInp=np.zeros((size,maxTarLen),dtype='float32')
            decTar=np.zeros((size,maxTarLen,lenOfDecoderTokens),dtype='float32')
            for i2,(inpText,tarText) in enumerate(zip(x[i1:i1+size],y[i1:i1+size])):
                for i3,word in enumerate(inpText.split()):
                    encInp[i2,i3]=inpTokenInd[word]
                tarTextSplit=tarText.split()
                for i3,word in enumerate(tarTextSplit):
                    if i3<len(tarTextSplit)-1:
                        decInp[i2,i3]=tarTokenInd[word]=tarTokenInd[word]
                    if i3>0:
                        decTar[i2,i3-1,tarTokenInd[word]]=1
            yield([encInp,decInp],decTar)
dims=50
encInp=Input(shape=(None,))
encEmb=Embedding(lenOfEncoderTokens,dims,mask_zero=True)(encInp)
encLSTM=LSTM(dims,return_state=True)
encOut,stateH,stateC=encLSTM(encEmb)
encStates=[stateH,stateC]
decInp=Input(shape=(None,))
decEmbLayer=Embedding(lenOfDecoderTokens,dims,mask_zero=True)
decEmb=decEmbLayer(decInp)
decLSTM=LSTM(dims,return_sequences=True,return_state=True)
decOut,_,_=decLSTM(decEmb,initial_state=encStates)
decDense=Dense(lenOfDecoderTokens,activation='softmax')
decOut=decDense(decOut)
model=Model([encInp,decInp],decOut)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
trainLen=len(x_train)
testLen=len(x_test)
size=128
epochs=50
model.fit(encode(x_train,y_train,size),steps_per_epoch = trainLen//size,
                    epochs=epochs,
                    validation_data = encode(x_test, y_test,size),
                    validation_steps = testLen//size)
model.save_weights('model_weights.h5')