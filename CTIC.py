

from __future__ import print_function
import subprocess
import random
import argparse
import numpy as np
import pickle as pickle
from collections import Set
import re
#import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.backends.cudnn as cudnn
from operator import itemgetter, attrgetter, methodcaller
import math
#from rnnDiffusionModel import rnnDiffusionModel
from Episode import Episode
import scipy.misc
import itertools as it
import functools as ft

#from torchvision import datasets, transforms
#from torch.autograd import Variable
#import torch.autograd.profiler as profiler
from utils import *

import sys
from sortedcollections.recipes import OrderedSet
#from blaze.expr.math import isnan
sys.path.append("utils")
#from kbhit import KBHit
from tqdm import tqdm
import json

#
# Training settings
parser = argparse.ArgumentParser(description='CTIC')

parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--inTrainFile', '-itr', type=str, metavar='S',
                    help='input train file',required=True)
parser.add_argument('--noTrain', type=bool, metavar='B',
                    help='ignore train on train data',default=False)
parser.add_argument('--inTestFile', '-ite', type=str, metavar='S',
                    help='input test file',default="")
parser.add_argument('--outDir', '-o', type=str, metavar='S',
                    help='out directory',required=True)
parser.add_argument('--fromFile', '-fr', type=str, metavar='S', default="",
                    help='model file to load (default: empty)')
parser.add_argument('--fromFileT', '-frt', type=str, metavar='S', default="",
                    help='model text file to load (default: empty)')
parser.add_argument('--T', type=int, default=-1, metavar='N',
                    help='last step (default: -1 : max from train)')

#parser.add_argument('--testT', type=int, metavar='N', default=-1,
#                    help='lastTime inference in test (default: -1)')
#parser.add_argument('--trainT', type=int, metavar='N', default=-1,
#                    help='lastTime in train (default: -1 => max train time)')
parser.add_argument('--freqTest', type=int, metavar='N', default=1,
                    help='frequence for tests (default: 1)')
#parser.add_argument('-world', action='store_true', default=False,
#                    help='world node or not (default False)')
parser.add_argument('--fromTests', nargs='+', type=int, default=[0],
                    help='fromTests (always prepend with 0)')
parser.add_argument('--computellLinks', '-cl', action='store_true', default=False,
                    help='compute likelihood of known links or not. only possible if fromWho known in episodes (default False)')
parser.add_argument('--nbsimu', '-ns', type=int, metavar='N', default=10,
                    help='nb simulations to perform (default: 10)')



args = parser.parse_args()
args.commit = subprocess.check_output(['git', 'show-ref']).decode('utf-8')
args.world=True


random.seed(args.seed)
np.random.seed(args.seed)


class CTIC():
    verbose = 0
    def __init__(self, T, nbNodes,world=False):
        super(CTIC, self).__init__()
        self.T=T
        self.nbNodes=nbNodes
        #self.plus=None
        self.moins=None
        self.k=None
        self.r=None
        self.preds=None
        self.oldLL=None
        self.world=world

    def loadFromTextFile(self, file):
        with open(file, 'r') as f:
            s=f.read()
            s=s.split("k : ")[1]
            s=s.split("r : ")
            self.k=eval(s[0])
            self.r=eval(s[1])


    def prepareLearning(self,episodes):
        self.moins={}

        #self.plus={}

        self.k={}
        self.r={}
        self.preds=[None]*len(episodes)
        toTest={}
        fr=0
        if self.world:
            fr=-1
        i=0
        for e in episodes:
            pr=[]
            l=e.listN
            #prs(l)

            l=[(x,y) for x,y in l.items()]
            pr={l[j][0]:[(l[k][0],math.log(l[j][1]-l[k][1]+1)) for k in range(j) if (l[k][1]<l[j][1]  and l[k][0]<self.nbNodes and l[k][1]>fr and l[j][1]<=self.T)] for j in range(len(l))}
            self.preds[i] = pr
            ll=[(l[k][0],l[j][0]) for j in range(len(l)) for k in range(j) if (l[k][1]<l[j][1] and l[k][1]>fr and l[j][1]<=self.T)]
            for j in ll:
                #p=self.plus.get(j,[])
                #p.append(i)
                #self.plus[j]=p
                if j not in self.k:
                    self.k[j]=0.3 #random.random()*0.2+0.1
                    self.r[j] = 0.1
                p=toTest.get(j[0],set())
                p.add(j[1])
                toTest[j[0]]=p


            i+=1

        for e in episodes:
            l=e.nodes

            for x in l:
                mv = self.moins.get(x, {})
                self.moins[x]=mv
                for v in toTest.get(x,[]):
                    m = mv.get(v,0)
                    if v not in l:
                        m+=1
                    mv[v]=m




        #prs("preds",self.preds)
        #prs("moins", self.moins)

    def computeab(self, v, preds):
        c=[(u,v) for u, _ in preds]
        t=np.array([t for _,t in preds])
        k = np.array([self.k.get((u, v),1E-20) for u, _ in preds])
        r = np.array([self.r.get((u, v),1E-20) for u, _ in preds])
        rt = -r*t
        b1=k*np.exp(rt)

        a=r*b1
        b2=1.0-k
        b=b1+b2
        a=a/b
        s=np.sum(a)
        a=a/s
        #prs("a",np.exp(a))

        b=b1/b

        #prs(c,b1,b2,b)

        #b1=[math.log(k[i])+rt[i] for i in range(len(k))]
        #a = [math.log(r[i]) + b1[i] for i in range(len(k))]
        #b2=[math.log(1.0 - k[i]) for i in range(len(k))]
        #maxs=[max(b1[i],b2[i]) for i in range(len(k))]
        #b=[maxs[i]+math.log(math.exp(b1[i]-maxs[i]) + math.exp(b2[i]-maxs[i])) for i in range(len(k))]
        #a=[a[i]-b[i] for i in range(len(k))]

        #prs("k",k)
        #prs("r", r)
        ret=list(zip(c,a,b,t))
        #prs("ret",ret)
        return  ret

    def computeab2(self, v, preds):
        c=[(u,v) for u, _ in preds]
        t=np.array([t for _,t in preds])
        k = np.array([self.k[(u, v)] for u, _ in preds])
        r = np.array([self.r[(u, v)] for u, _ in preds])
        rt = -r*t
        b1=np.log(k)+rt

        a=np.log(r)+b1
        b2=np.log(1.0-k)
        maxs=np.maximum(b1,b2)
        #prs("maxs",maxs)
        b=maxs+np.log(np.exp(b1-maxs)+np.exp(b2-maxs))
        a=a-b
        m=np.max(a)
        mm=np.exp(a-m)
        s=m+np.log(np.sum(mm))
        a=a-s
        #prs("a",np.exp(a))

        b=b1-b

        #prs(c,b1,b2,b)

        #b1=[math.log(k[i])+rt[i] for i in range(len(k))]
        #a = [math.log(r[i]) + b1[i] for i in range(len(k))]
        #b2=[math.log(1.0 - k[i]) for i in range(len(k))]
        #maxs=[max(b1[i],b2[i]) for i in range(len(k))]
        #b=[maxs[i]+math.log(math.exp(b1[i]-maxs[i]) + math.exp(b2[i]-maxs[i])) for i in range(len(k))]
        #a=[a[i]-b[i] for i in range(len(k))]

        #prs("k",k)
        #prs("r", r)
        ret=list(zip(c,a,b,t))
        #prs("ret",ret)
        return  ret

    def computekr(self,x,ll):
        l=list(zip(*ll))
        #prs("l",l)
        a=np.array(list(l[1]))
        b = np.array(list(l[2]))
        t = np.array(list(l[3]))
        #prs("a ",x,a)
        #prs("b ", x, b)
        #prs("t ", x, t)
        sa=np.sum(a)
        #sa=np.sum(a)

        na=a+(1.0-a)*b
        da=na*t
        da = np.sum(da)

        #da=np.sum(na*t)
        r=np.maximum(sa/da,1e-10)

        na = np.sum(na)

        k=((1.0-2e-10)*na/(self.moins[x[0]][x[1]]+len(ll)))+(1e-10)

        return (k,r)

    def computekr2(self,x,ll):
        l=list(zip(*ll))
        #prs("l",l)
        a=np.array(list(l[1]))
        b = np.array(list(l[2]))
        t = np.array(list(l[3]))
        #prs("a ",x,a)
        #prs("b ", x, b)
        #prs("t ", x, t)
        ma=np.max(a)
        sa=ma+np.log(np.sum(np.exp(a-ma)))
        #sa=np.sum(a)

        ma = np.maximum(a,b)
        na=ma+np.log(np.exp(a-ma)+np.exp(b-ma)-np.exp(a+b-ma))
        da=na+np.log(t)
        ma = np.max(da)
        da = ma + np.log(np.sum(np.exp(da - ma)))

        #da=np.sum(na*t)
        r=np.maximum(np.exp(sa-da),1e-10)

        ma = np.max(na)
        na = ma + np.log(np.sum(np.exp(na - ma)))

        k=((1.0-2e-10)*np.exp(na)/(self.moins[x[0]][x[1]]+len(ll)))+(1e-10)

        return (k,r)


    def step(self):
        if self.k is None:
            raise RuntimeError("The model is not prepared for learning")
            #self.prepareLearning(episodes)

        #p=[{j:self.computeab(j,self.preds[i][j]) for j in self.preds[i] if len(self.preds[i][j])>0} for i in range(len(self.preds))]
        p = [c  for i in range(len(self.preds)) for j in self.preds[i] if len(self.preds[i][j]) > 0 for c in self.computeab(j, self.preds[i][j])]
        #prs("p", sorted(p))
        p={k: [x for x in v] for k, v in it.groupby(sorted(p), key=lambda x: x[0])}
        #prs("p",p)
        kr={x:self.computekr(x,l) for x,l in p.items()}
        self.k={x:e[0] for x,e in kr.items()}
        self.r = {x: e[1] for x, e in kr.items()}
        #prs("k",self.k)
        #prs("r", self.r)

        #prs("p",p)
        #nr=

    def computehg(self, v, t, preds, ninf, testFrom=0):
        #prs("preds",v,preds,ninf)
        h=0
        if len(preds)>0 and t>=testFrom:
            c=[(u[0],v) for u,_ in preds]
            dt=np.array([t for _,t in preds])
            dt0 = np.array([math.log(testFrom - u[1]) if (u[1] < (testFrom-1)) else -1 for u,_ in preds])
            k = np.array([self.k[(u[0], v)] if (u[0], v) in self.k else 1E-20 for u, _ in preds])
            r = np.array([self.r[(u[0], v)] if (u[0], v) in self.r else 1E-20 for u, _ in preds])
            rt = -r*dt
            rt0 = -r * dt0
            b1=k*np.exp(rt)
            b10 = k * np.exp(rt0)
            #r=np.maximum(r,1e-10)
            a=r*b1
            #minusk=1.0-k
            #minusk=np.maximum(minusk,1e-10)
            b2=1.0-k
            b=b1+b2
            b0 = b10 + b2
            a=a/b

            #prs("a",np.exp(a+np.sum(b)))
            s = np.sum(a)
            h=np.log(s)+np.sum(np.log(b))
            b0=b0[dt0>=0]
            h=h-np.sum(np.log(b0))
            #prs("h",np.exp(h))
        g=0
        if len(ninf)>0:
            k=np.array([self.k[(v,x)] if (v,x) in self.k else 0.0 for x in ninf])
            r = np.array([self.r[(v, x)] if (v,x) in self.r else 0.0 for x in ninf])
            #print("v"+str(v))
            #print("ninf" + str(ninf))

            dt0 = np.array([math.log(testFrom - t) if (t < (testFrom-1)) else -1 for x in ninf])
            rt0 = -r * dt0
            b10 = k * np.exp(rt0)
            g=np.log(1-k)
            b0=b10+1-k
            g=np.sum(g)
            b0 = b0[dt0 >= 0]
            g = g - np.sum(np.log(b0))
            #prs("g", g)
        #prs("h+g",h+g)
        return (h,g)



    def computell(self,episode,testFrom=0):
        l = episode.listN
        #prs(l)
        fr=0
        if self.world:
            fr=-1
        l = [(x, y) for x, y in l.items()]
        pr = {l[j][0]: [(l[k], math.log(l[j][1] - l[k][1]+1)) for k in range(j) if ((l[k][1] < l[j][1]) and (l[k][0]<self.nbNodes and l[k][1]>fr))]  for j in range(len(l)) if l[j][1]<=self.T and l[j][0]<self.nbNodes}

        l = episode.listN
        ninf={i:[v for v in self.moins[i] if v not in l] for i in l.keys()}
        #prs("ninf",ninf)
        #prs("l", l.items())

        hg=[self.computehg(v,t,pr[v],ninf[v],testFrom) for v,t in l.items() if t<=self.T and v<self.nbNodes]
        #prs("hg",hg)
        h=[x[0] for x in hg]
        g = [x[1] for x in hg]

        h=np.array(h)
        lh=np.sum(h)
        g = np.array(g)
        lg = np.sum(g)
        return (lh,lg)


    def computeLogLikelihood(self,episodes,testFrom=0):
        if self.k is None:
            self.prepareLearning(episodes)

        ll=np.array([self.computell(e,testFrom) for e in episodes])
        h = np.array([x[0] for x in ll])
        g = np.array([x[1] for x in ll])

        #if self.oldLL is not None:
        #    diff=ll-self.oldLL
        #    prs("diff",diff)
        #self.oldLL = ll
        h=h.sum()
        g=g.sum()
        s=(h+g)/len(episodes)

        return (s,h,g)


    def computellLinkse(self,episode,testFrom=0,firstStepIncluded=True):
        #print("T"+str(self.T))
        l = episode.listN
        links=episode.fromWho
        nodes=episode.nodes
        # prs(l)
        fr = 0
        if self.world:
            fr = -1
        l = [(x, y) for x, y in l.items()]
        noTest=False
        if testFrom<=0:
            noTest=True
        #if not firstStepIncluded:

        pr = {l[j][0]: [(l[k][0], math.log(l[j][1] - l[k][1] + 1)) for k in range(j) if
                       (l[k][1] < l[j][1] and l[k][0] < self.nbNodes and l[k][1] > fr)] for j in range(len(l)) if (noTest or l[j][1]>=testFrom) and l[j][1]<=self.T and l[j][0]<self.nbNodes}

        #print("pr",str(pr))

        p = {c[0]:c for j in pr if len(pr[j]) > 0 for c in self.computeab(j, pr[j])}
        #prs("p", p)

        l=[p[(y,x)][1]  for x,y in links.items() if (noTest or nodes[x]>=testFrom) and (firstStepIncluded or nodes[y]>1) and nodes[x]<=self.T and x<self.nbNodes and y<self.nbNodes]
        #print("links "+str(links))

        #s=1
        #if len(l)>0:
        #    s=sum(l)/len(l)
            #s=sum(l)

        return sum(l),len(l)

    def computellLinks(self,episodes,testFrom=0):
        l=[self.computellLinkse(episodes[i], testFrom) for i in range(len(episodes))]
        l=[x for x in l if x[1]>0]
        if len(l)==0:
            return 0,0
        ll,ss = zip(*l)
        if sum(ss)==0:
            return 0,0
        return sum(ll)/sum(ss),sum(ss)

    def simue(self,episode,testFrom,nbsimu):
        maxT=self.T
        l = episode.listN
        l = [(x, y) for x, y in l.items()]
        infectious=np.array([maxT+1]*self.nbNodes) #[x for x,y in l.items() if y<testFrom])
        #times=np.array([y for x,y in l.items() if y<testFrom])
        infectious[0]=0
        for x,y in l:
            if y>=testFrom:
                break
            infectious[x]=y

        nbInf=np.array([0]*self.nbNodes)

        for i in range(nbsimu):
            times=np.copy(infectious)
            inf=np.copy(infectious)
            while True:
                qui=np.argmin(inf)
                quand=inf[qui]
                inf[qui]=maxT+1
                if quand==maxT+1:
                    break
                times[qui]=quand
                #print(str(times)+","+str(quand))
                vers=(times>quand)
                #print(str(vers))
                versq=np.array(range(len(times)))[vers]
                k=np.array([self.k.get((qui, v),1E-20) for v in versq])
                r=np.array([self.r.get((qui, v),1E-20) for v in versq])
                x=np.random.rand(len(versq))
                ok=x<k
                t=np.random.exponential(r)
                t=((np.exp(t)-1)+quand)*ok
                wt=(t<testFrom)*ok
                tt=0
                while np.sum(wt)>0 and tt<2:
                    t2 = np.random.exponential(r)
                    t2 = ((np.exp(t2) - 1) + quand) * ok
                    t=(1-wt)*t+wt*t2
                    wt = (t < testFrom) * ok
                    tt+=1
                wt = (t >= testFrom) * ok *(t<inf[versq])
                #print(str(inf))
                #print(str(wt))
                #print(str(t))
                inf[versq[wt]]=t[wt]
            nbInf[times<maxT+1]+=1
        nbInf=nbInf/(nbsimu*1.0)
        trueInf=np.array([0]*self.nbNodes)
        trueInf[list(episode.listN.keys())]=1
        #print("nbInf "+str(nbInf))
        #print("trueInf " + str(trueInf))

        return (np.sum(np.log(np.where(trueInf>0,nbInf,1-nbInf)+1E-10)),np.sum(nbInf)) #(trueInf*np.log(nbInf)).sum()+((1-trueInf)*(np.log(1.0-nbInf))).sum()



    def simu(self,episodes,testFrom=2,nbsimu=10):
        sum=0
        nb=0
        pb = tqdm(range(len(episodes)), dynamic_ncols=True)
        for i in pb:
            c,n=self.simue(episodes[i], testFrom, nbsimu)
            sum+=c
            nb+=n*1.0
        return (sum/len(episodes),nb/len(episodes))




episodes, nodes, T = Episode.loadEpisodesFromFile(args.inTrainFile)
episodesT, nodesT, TT = (None,None,None)
testBins=None
if len(args.inTestFile)>0:
    episodesT, nodesT, TT = Episode.loadEpisodesFromFile(args.inTestFile,nodes)

args.nbNodes=len(nodes)
if args.T<0:
    args.T=T
print("world "+str(args.world))

diffMod=None
if len(args.fromFile)>0:
    diffMod = pickle.load(args.fromFile)
    #diffMod.cpu()
else:
    diffMod = CTIC(args.T, args.nbNodes, world=args.world)

     # ./arti_episodes_train.txt")
#


args.nbTrainEpisodes=0
if(episodes is not None and len(episodes)>0):
    args.nbTrainEpisodes =len(episodes)
args.nbTestEpisodes=0
if(episodesT is not None and len(episodesT)>0):
    args.nbTestEpisodes=len(episodesT)

print("Logs in: " + args.outDir)
iter_logger = LogCSV(args.outDir, filename='iter_log.csv')
iter_loggert = None
if args.nbTestEpisodes>0:
    iter_loggert = LogCSV(args.outDir, filename='iter_logt.csv', wipe=False)

args.model = str(diffMod).split("(")[0]

with open(os.path.join(args.outDir, 'info.json'), 'w') as info:
    json.dump(vars(args), info, sort_keys=True, indent=4)
with open(os.path.join(args.outDir, 'info_long.json'), 'w') as info:
    args.model = str(diffMod)
    args.nodes = str(nodes)
    json.dump(vars(args), info, sort_keys=True, indent=4)


if diffMod.k is None:
    if args.nbTrainEpisodes>0:
        diffMod.prepareLearning(episodes)
    else:
        raise RuntimeError("no train episode for learning !")

if len(args.fromFileT) > 0:
    diffMod.loadFromTextFile(args.fromFileT)

with open(os.path.join(args.outDir, "./diffMod_start"), 'wb') as pickle_file:
    pickle.dump(diffMod, pickle_file)


diffMod.verbose=0
iter_logger.new_iteration()
ll=diffMod.computeLogLikelihood(episodes)
iter_logger.log("epoch", 0)
iter_logger.log("nll", -ll[0])
#iter_logger.log("inf", ll[1])
#iter_logger.log("notinf", ll[2])
print('\n train', str(ll))
if args.computellLinks:
    st, ss = diffMod.computellLinks(episodes)
    print("links computed on " + str(ss) + " links:" + str(st))
for epoch in range(1,args.epochs):
    print('Epoch ' + str(epoch))
    iter_logger.new_iteration()
    if not args.noTrain:
        diffMod.step()
        ll=diffMod.computeLogLikelihood(episodes)
        iter_logger.log("epoch", epoch)
        iter_logger.log("nll", -ll[0])
        #iter_logger.log("inf", ll[1])
        #iter_logger.log("notinf", ll[2])
        st=0
        if args.computellLinks:
            st,ss=diffMod.computellLinks(episodes)
            print("links computed on " + str(ss) + " links:"+str(st))
        iter_logger.log("INF", st)


        print('\n train', str(ll[0]), str(ll[1]), str(ll[2]), str(st))
        #if args.nbsimu > 0 and epoch%20==1:
        #    rinf,ninf = diffMod.simu(episodes, 0, args.nbsimu)
        #    print("rinf  " + str(rinf))
        #    print("ninf  " + str(ninf))
        #    iter_logger.log("rinf ", rinf)
        #else:
        #    iter_logger.log("rinf ", 0)

    if args.nbTestEpisodes>0 and epoch % args.freqTest == 0:
        for fromTest in args.fromTests:
            ll = diffMod.computeLogLikelihood(episodesT,fromTest)
            iter_loggert.log("epoch", epoch)
            iter_loggert.log("nll_"+str(fromTest), -ll[0])
            #iter_loggert.log("inf_"+str(fromTest), ll[1])
            #iter_loggert.log("notinf_"+str(fromTest), ll[2])
            st = 0
            if args.computellLinks:
                st,ss = diffMod.computellLinks(episodesT,fromTest)
                print("links computed on "+str(ss)+" links:"+str(st))
                iter_loggert.log("INF_"+str(fromTest), st)

            print('\n test_'+str(fromTest), str(ll[0]), str(ll[1]), str(ll[2]),str(st))
            if args.nbsimu>0  and epoch%20==1:
                rinf,ninf=diffMod.simu(episodesT, fromTest, args.nbsimu)
                print("rinf  " + str(rinf))
                print("ninf  " + str(ninf))
                iter_loggert.log("CE_" + str(fromTest), rinf)
            else:
                iter_loggert.log("CE_" + str(fromTest), 0)

    if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
        iter_logger.flush()
        if args.nbTestEpisodes > 0:
            iter_loggert.flush()
        with open(os.path.join(args.outDir, "./diffMod_" + str(epoch)), 'wb') as pickle_file:
            pickle.dump(diffMod, pickle_file)
        with open(os.path.join(args.outDir, "./params_" + str(epoch)+".txt"), "w") as f:
            f.write("k" " : " + str(diffMod.k) + "\n")
            f.write("r" " : " + str(diffMod.r) + "\n")



#python CTIC.py -o xp/CTIC/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt --fromTests 0 2 5 10 -cl -ns 0