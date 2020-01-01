from keras.layers import *
from keras import Model
import config
import pickle
import numpy
from keras.callbacks import ModelCheckpoint
import model
from keras.optimizers import SGD, Adam, RMSprop
# train_x,train_y,test_x,test_y = pickle.load(open("./data.pk","rb"))

with open("./data.pk","rb") as f:
    train_x = pickle.load(f)
    test_x = pickle.load(f)
    train_y = pickle.load(f)
    test_y = pickle.load(f)

model = model.lstm()
model.load_weights(config.model_path) # 读取权重

result_x = model.predict(test_x)
result_x = numpy.argmax(result_x,axis=1) # 返回沿轴axis最大值的索引,1代表行,返回每一行中的最大值，即最有可能是哪一种情况
result_y = numpy.argmax(test_y,axis=1)
count=0
for i,j in zip(result_x,result_y):
    if i==j:
        count += 1
result = count/len(result_y)
print(result)

# 编译用来配置模型的学习过程???
model.compile(optimizer=RMSprop(),loss="categorical_crossentropy", metrics=['accuracy'])
score = model.evaluate(test_x, test_y,verbose=1)
print(score)
# print(result)

# 0.7222546161321671