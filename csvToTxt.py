#!/usr/bin/env python
# coding: utf-8


# In[2]:

############Currently, this script converts whole .csv file into 'train' file as well as 'test' file. Edit the code according to your needs.######

import os
import numpy as np


# In[3]:


srcPaths = [f"./data/data_csv_integers/{i}" for i in range(8)] #Change this to where your dataset is kept
for i in range(len(srcPaths)):
    srcPaths[i] = [os.path.join(srcPaths[i], p) for p in os.listdir(srcPaths[i])]


targetPaths = [f"./data/datasets/{i}_no_val_" for i in range(8)] #Target paths of the .txt datasets
for i in range(len(targetPaths)):
    try:
        os.mkdir(targetPaths[i])
    except:
        pass
    targetPaths[i] = [os.path.join(targetPaths[i], p) for p in ["train", "test", "val"]]
    for path in targetPaths[i]:
        try:
            os.mkdir(path)
        except:
            pass


# In[4]:


print(srcPaths, targetPaths)
#order = test(testPercent %), train(remaining %), val(valPercent %)


# In[6]:


def csvToTxt(path):
    # reading csv file
    _text = open(path, "r")
    text = np.array([i.split("\n")[0].split(",") for i in _text])[:, 1:]
    _text.close()
    #print(text)
    l = len(text[0])
    assert l == len(text[1]) and len(text[1]) == len(text[2]), "Unequal lengths!"
    trainLen = l

    # for i in range(len(text)):
    #     text[i] =
    # joining with space content of text
    trainTxt = ""
    for _l in range(l):
        trainTxt += f"{text[0][_l]}\t{text[1][_l]}\t{text[2][_l]}\t{text[3][_l]}\n"

    #displaying result
    return trainTxt


# In[7]:


for ind in range(len(srcPaths)):
    path = srcPaths[ind]
    trainTxt = csvToTxt(path[0])
    #print(testTxt, "\n\n\n", trainTxt, "\n\n\n", valTxt)
    with open(os.path.join(targetPaths[ind][0], "data.txt"), "w") as f:
        f.write(trainTxt)
    with open(os.path.join(targetPaths[ind][1], "data.txt"), "w") as f:
        f.write(trainTxt)


# In[28]:




