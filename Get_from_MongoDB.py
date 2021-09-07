# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:00:33 2021

@author: PengTao
"""
import pymongo
import pprint as pp
import numpy as np
# Connect to sever. Example:"mongodb://localhost:27017/"
conn=pymongo.MongoClient("mongodb://localhost:27017/")
# Connect to database. Example:"Learn"
db=conn["Learn"]
# Pretty print all the documents in a collection.
i=0
price=np.zeros(db.orders.count_documents({}))
for doc in db.orders.find().sort("cust_id",pymongo.DESCENDING):
    pp.pprint(doc)
    price[i]=doc["price"]
    i+=1
# Read concerned content from documents in a collection.
