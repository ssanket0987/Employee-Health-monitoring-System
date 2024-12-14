import pandas as pd
import numpy as np

from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, GlobalMaxPool1D

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import MaxPool1D

df_train = pd.read_csv("h_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
X = X.reshape((X.shape[0], X.shape[1]))

trainFeat, testFeat, trainClass, testClass = train_test_split(X, Y, test_size=0.2)

nclass = 5
inp = Input(shape=(187, 1))

#first layer
layers = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(inp)
layers = MaxPool1D(pool_size=2)(layers)
layers = Dropout(rate=0.1)(layers)

layers = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(layers) #as its working on pervious layers n not directly on inp
layers = MaxPool1D(pool_size=2)(layers)
layers = Dropout(rate=0.1)(layers)

layers = Convolution1D(128, kernel_size=5, activation=activations.relu, padding="valid")(layers)
layers = MaxPool1D(pool_size=2)(layers)
layers = Dropout(rate=0.1)(layers)

layers = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(layers)
layers = GlobalMaxPool1D()(layers)
layers = Dropout(rate=0.1)(layers)
dense_1 = Dense(128, activation=activations.relu, name="dense_2")(layers)
dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_h")(dense_1)
model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam(0.001)

model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
model.summary()

model.fit(trainFeat, trainClass, epochs=5, validation_split=0.001)

#Now classify the test entries
pout = model.predict(testFeat)
cout = []

for i in range(0, len(pout)) :
    prob = pout[i]
    maxp = max(prob)
    
    for j in range(0, len(prob)) :
        if(prob[j] == maxp) :
            cout.append(j)
print(cout)
pre = precision_score(cout, testClass, average="macro")
print("Test Precision score : %s "% pre)

acc = accuracy_score(cout, testClass)
print("Test accuracy score : %s "% acc)

rec = recall_score(cout, testClass, average="macro")
print("Test Recall score : %s "% rec)

conf_matrix = confusion_matrix(cout, testClass);
print('Confusion matrix')
print(conf_matrix)