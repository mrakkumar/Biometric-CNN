# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:34:47 2017

@author: Mukund Bharadwaj
"""

import pickle
import numpy as np
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def load_labels(start=0,stop=50,repeat=1):
    data=np.arange(start,stop,1,np.int16)
    data=np.repeat(data,repeat)
    return data

def getAccuracy(data,targets):
    assert len(data)==len(targets)
    acc = 0
    for i,d in zip(data,targets):
        if d==i:
            acc += 1
        else:
            continue
    return acc*100.0/len(data)

def getClasses(data):
    lt = list()
    for p in data:
        if p is None:
            lt.append(-1)
        else:
            lt.append(np.argmax(p))
    return lt

def identify(data,targets):
    fn = 0
    tp = 0
    fp = 0
    for f,g in zip(data,targets):
        if f==-1:
            fn += 1
        elif f==g:
            tp += 1
        else:
            fp += 1
    num = len(data)
    return (tp*100.0/num,fp*100.0/num,fn*100.0/num)

def precision_recall(data,targets):
    return (recall_score(targets,data,average='macro'),precision_score(targets,data,average='macro'))

listCNN = []
listRF = []
listLR1 = []
listLR2 = []
listEnsemble = []
#for n_out in [50,100,200,403]:
n_out=403
with open('D:\\Documents\\Biometric\\Final Code\\Analysis\\Ensemble\\ansCNN_{}.pkl'.format(n_out),'rb') as f:
    ansCNN = pickle.load(f,encoding='latin1')
with open('D:\\Documents\\Biometric\\Final Code\\Analysis\\Ensemble\\ansRF_{}.pkl'.format(n_out),'rb') as f:
    ansRF = pickle.load(f,encoding='latin1')

lr1 = normalize(ansCNN[0])
lr2 = normalize(ansCNN[1])
cnn = normalize(ansCNN[2])
cnn_e = normalize(ansCNN[3])
rf = normalize(ansRF[1])

RF = ansRF[0]
LR1 = getClasses(ansCNN[0])
LR2 = getClasses(ansCNN[1])
CNN = getClasses(ansCNN[2])
CNN_e = getClasses(ansCNN[3])
Y = load_labels(0,n_out,2)



result=[]
for f,g in zip(cnn_e,rf):
    result.append([(x*0.55+y*0.45) for x,y in zip(f,g)])
result = np.asarray(result)
resultE2=[]
for a,b,c,d in zip(cnn,lr1,lr2,rf):
    resultE2.append([(w*0.1+x*0.4+y*0.15+z*0.35) for w,x,y,z in zip(a,b,c,d)])
resultE2 = np.asarray(resultE2)
#temp = np.split(result,n_out)

thresh = 0

out = list()
for f in lr1:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultLR1 = identify(getClasses(out),Y)
prLR1 = precision_recall(getClasses(out),Y)

out = list()
for f in lr2:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultLR2 = identify(getClasses(out),Y)
prLR2 = precision_recall(getClasses(out),Y)

out = list()
for f in cnn:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultCNN = identify(getClasses(out),Y)
prCNN = precision_recall(getClasses(out),Y)

out = list()
for f in cnn_e:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultCNN_e = identify(getClasses(out),Y)
prCNN_e = precision_recall(getClasses(out),Y)

out = list()
for f in rf:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultRF = identify(getClasses(out),Y)
prRF = precision_recall(getClasses(out),Y)

out = list()
for f in result:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultEnsemble = identify(getClasses(out),Y)
prEnsemble = precision_recall(getClasses(out),Y)

out = list()
for f in resultE2:
    if max(f) < thresh:
        out.append(None)
    else:
        out.append(f)
resultEnsemble2 = identify(getClasses(out),Y)
prEnsemble2 = precision_recall(getClasses(out),Y)
    
#    listEnsemble.append(resultEnsemble2[0])
#    listCNN.append(resultCNN[0])
#    listRF.append(resultRF[0])
#    listLR1.append(resultLR1[0])
#    listLR2.append(resultLR2[0])
#
#fig = plt.figure()
#plt.plot([50,100,200,403],listCNN,'-o')
#plt.plot([50,100,200,403],listRF,'-o')
#plt.plot([50,100,200,403],listLR1,'-o')
#plt.plot([50,100,200,403],listLR2,'-o')
#plt.plot([50,100,200,403],listEnsemble,'-o')
#plt.title('Variation of accuracy with\nnumber of labels')
#plt.ylabel('% Accuracy')
#plt.xlabel('No. of Labels')
#plt.legend(['CNN','RF','LR1','LR2','Ensemble'])

#classes = getClasses(out)
#for i,j,k,z in zip(classes,Y,out,range(806)):
#    if i!=j:
#        print (j,i,out[z][j],out[z][i])

#print (out[(140*2)+1][125])

#print 'Ensemble Accuracy: {}'.format(getAccuracy(resultFinal,Y))
#print 'CNN Accuracy: {}'.format(getAccuracy(CNN,Y))
#print 'RF Accuracy: {}'.format(getAccuracy(RF,Y))
#print 'After threshold...'
#print 'CNN'
#analysisCNN = identify(resultCNN,Y)
#print analysisCNN
#print 'RF'
#analysisRF = identify(resultRF,Y)
#print analysisRF
#print 'Ensemble'
#analysisEnsemble = identify(resultEnsemble,Y)
#print analysisEnsemble