#coding:utf-8
from data_helper import DataHelper
from models.model_base import BaseModel,GCNModel
from params import Params
from trainer import BaseTrainer
import tensorflow as tf

params={
    #data
    "src_triple_num":105998,
    "tgt_triple num": 115722}

if __name__=="__main__":
    #超参数
    params=Params(data_name="fr_en")
    #data
    data_helper=DataHelper(params)

    #model
    # model=BaseModel(params)
    model=GCNModel(params)

    #trainer
    trainer=BaseTrainer(model,params)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #train
    # trainer.restore_last_session(sess)
    trainer.train(sess,data_helper,iter_num=5000)

    #test
    trainer.evaluate(sess,data_helper)
