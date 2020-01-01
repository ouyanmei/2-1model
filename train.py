from keras.layers import *
from keras import Model
import config
import pickle
from keras.callbacks import ModelCheckpoint
# train_x,test_x,train_y,test_y = pickle.load(open("./data.pk","rb"))
import model
from keras.optimizers import SGD, Adam, RMSprop
# from keras.losses import CategoricalCrossentropy
from keras.losses import categorical_crossentropy
# train_x,train_y,test_x,test_y = pickle.load(open("./data.pk","rb"))


with open("./data.pk","rb") as f:
    train_x = pickle.load(f)
    test_x = pickle.load(f)
    train_y = pickle.load(f)
    test_y = pickle.load(f)

model = model.lstm()
model.summary() # 通过model.summary()输出模型各层的参数状况

# 编译用来配置模型的学习过程
# model.compile(optimizer=Adam(),loss="categorical_crossentropy", metrics=['accuracy'])
# optimizer：优化器，loss：损失函数，metrics：评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy'],以精确度作为指标
model.compile(optimizer=RMSprop(),loss="categorical_crossentropy", metrics=['accuracy'])

# keras ModelCheckpoint 实现断点续训功能，在训练阶段的model.fit之前加载先前保存的参数
# filename：字符串，保存模型的路径; save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
# monitor：需要监视的值，val_acc或这val_loss;
# mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
# verbose：信息展示模式，0为不打印输出信息，1打印
cp = ModelCheckpoint(config.model_path, save_best_only=True,
                monitor="val_acc",mode="max",verbose=1)

# 训练
# x：输入数据。如果模型只有一个输入，那么x的类型是numpyarray，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
# y：标签，numpy array
# batch_size：整数，指定进行梯度下降时每个batch包含的样本数。
# validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。???
# validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
# epochs：整数，训练的轮数，每个epoch会把训练集轮一遍
# 在model.fit添加callbacks=[checkpoint]实现回调
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# validation_data=(test_x,test_y) 改，在model.fit添加callbacks=[cp]实现回调，指定训练集的0.1比例数据作为验证集，2为每个epoch输出一行记录
model.fit(train_x,train_y,batch_size=128,validation_data=(test_x,test_y),epochs=3,callbacks=[cp],verbose=1)


embed_weight = model.predict(test_x,batch_size=1) # 为输入样本生成输出预测。
# score = model.evaluate(test_x, test_y, verbose=1)


# score = model.evaluate(test_x, test_y,verbose=1)
#
# x = test_x[:2]
# # print(x)
# x = x.reshape(2, config.max_len)
# t = model.predict(x, batch_size=1, verbose=1)
# print(t)
# print("答案：",test_y[:2])
#
# print('loss:', score[0])
# print('Test accuracy:', score[1])


print(embed_weight)
print(embed_weight.shape)