#coding:utf-8
from collections import defaultdict
import numpy as np

def read_name2ids(path):
    name2ids={}
    id2names={}
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split("\t")
            id,ent=line
            id=int(id)
            name2ids[ent]=id
            id2names[id]=ent
    return name2ids,id2names



def read_aligned_entities(path):
    entities1=[]
    entities2=[]
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split("\t")
            ent1,ent2=line
            entities1.append(ent1)
            entities2.append(ent2)
    return entities1,entities2

def read_triples(path):
    triples=[]
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split("\t")
            h,r,t=line
            triples.append((h,r,t))
    return triples
def read_attributes(path):
    entity2attrs={}
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split()
            ent=line[0]
            attrs=line[1:]
            entity2attrs[ent]=attrs
    return entity2attrs

def convert_triple_to_ids(triples,entity2id,relatioin2id):
    triples_new=[]
    for h,r,t in triples:
        hid=entity2id[h]
        tid=entity2id[t]
        rid=relatioin2id[r]
        triples_new.append([hid,rid,tid])
    return triples_new

def convert_names_to_ids(names,name2id):
    names_new=[]
    for ent in names:
        eid=name2id[ent]
        names_new.append(eid)
    return names_new

def read_embedding(path,vocab_size,emb_dim=300):
    embedding=np.zeros(shape=[vocab_size,emb_dim])
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            id,vec=line.strip().split("\t")
            id=int(id)
            vec=[float(v) for v in vec.split()]
            embedding[id,:]=vec
    return embedding.astype("float32")


