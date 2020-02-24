import pandas as pd
import matplotlib.pyplot as plt
# create a sample folder with download train and test.csv dataset on malaria

train = pd.read_csv("sample/train.csv")
test = pd.read_csv("sample/test.csv")

# test.describe()

x_train = train.drop(['label'] , axis=1).values
y_train = train['label'].values

x_test = test.drop(['label'] , axis =1).values
y_test = test['label'].values

#plt.imshow(x_train[24].reshape(50,50) , cmap='gray')
#print(y_train[24])

x_train = x_train.reshape(x_train.shape[0] , 50,50,1).astype('float32')
x_train = x_train/255

x_test = x_test.reshape(x_test.shape[0] , 50,50,1).astype('float32')
x_test = x_test/255

from sklearn import preprocessing
binalize = preprocessing.LabelBinarizer()
y_train = binalize.fit_transform(y_train)
y_test = binalize.fit_transform(y_test)
# made categories into 1 and 0 rather than infected and uninfected
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation ,Flatten

from tensorflow.keras.layers import Conv2D , MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=3,padding="same",activation="relu",input_shape=(50,50,1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
          
model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
          
model.add(Dropout(0.2))
model.add(Flatten())
          
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)

predict = model.evaluate(x_test , y_test)

index = 400
plt.imshow(x_test[index].reshape(50,50),cmap='gray')
print("actual :",y_test[index])

a = model.predict([[x_test[index]]])[0][0]
b = model.predict([[x_test[index]]])[0][1]

if( a > b):
    print("infected ")
    
else:
    print("good ")
    
    

