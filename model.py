from keras.layers import *
from keras import Model
import config
import pickle
from keras.callbacks import ModelCheckpoint

def lstm():
    # shape=(config.max_len,)表示了预期的输入将是一批config.max_len维的向量。
    # 返回一个张量
    input = Input(shape=(config.max_len,))

    # keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform',embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False,input_length=None)
    # 输入shape形如（samples，sequence_length）的2D张量,输出shape形如(samples, sequence_length, output_dim)的3D张量，即给每个词一个向量,词向量的维度为output_dim
    # Embedding（input_dim：字典长度，output_dim：代表全连接嵌入的维度，trainable=False：意味着不对该层权重进行更新，mask_zero=True，True代表将输入中的‘0’看作是应该被忽略的‘填充’（padding）值
    embed = Embedding(config.vocab_num+1,config.embed_dim, trainable=True, mask_zero=True)(input) # 给每个词一个向量

    # 当一个复杂的前馈神经网络被训练在小的数据集时，容易造成过拟合。Dropout解决过拟合问题和训练模型费时，训练时使(0.5)一半神经元失活，减少过拟合，所以一次只有一半的神经元的参数得到更新
    embed = Dropout(0.5)(embed)

    # 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    # 规范化处理，去量纲，使初始化权重就已经落在数据内部，加快收敛，避免过拟合
    embed = BatchNormalization()(embed)

    # repre = Bidirectional(LSTM(units=100, return_sequences=False))(embed)
    # units：输出维度为100，把词的维度由 embed_dim 转变成了 units
    # False:仅返回输出序列的最后一个输出，每个词输入都有一个units维的向量输出，只取最后生成的那个向量
    # 双向LSTM就把最后把最后得到的两个输出连接起来
    repre = Bidirectional(LSTM(units=100, return_sequences=False))(embed)

    #全连接层，线性变换+激活函数
    # units:输出的维度。units=3，该层有3个神经元（因为三分类）
    # activation:该层使用的激活函数,当不指定激活函数时（即 activation(x) = x ），这个全连接层就等价于我线性变换。区别？
    output = Dense(units=3, activation="softmax")(repre)
    model = Model(input=input, output=output) # 通过model.summary()输出模型各层的参数状况
    # model.summary() # 通过model.summary()输出模型各层的参数状况
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy']) # metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
    return model