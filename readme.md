### 数据处理

#### data_reader.py

读取数据的一些方法，包括读取三元组、实体名和实体id的映射等

#### data_helper.DataHelper

处理训练数据，把实体转换成id, 迭代生成一个batch的训练数据



### 模型

#### models.model_base.GCNModel

GCN模型

#### models.model_base.GATModel

GAT模型

#### models.models_crossKG.CrossGAT

我们提出的CrossGAT模型



### 参数配置

#### params.Params

模型参数



### 训练模型

#### trainer.BaseTrainer

训练器，负责把DataHelper生成的数据传给模型，进行训练

#### maiin_train_CrosssGAT.py

训练CrossGAT模型



### 评估模型

#### evaluate.py

