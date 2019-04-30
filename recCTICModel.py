
#from __future__ import print_function
#import argparse
import numpy as np
#import re
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import torch.backends.cudnn as cudnn
#import math

import scipy.misc

#from torchvision import datasets, transforms
#from torch.autograd import Variable
import torch.autograd.profiler as profiler

import sys
#from sortedcollections.recipes import OrderedSet
from collections import OrderedDict
#from blaze.expr.math import isnan
sys.path.append("utils")

#from kbhit import KBHit
from utils import *
from tqdm import tqdm







class EMrecCTICModel(nn.Module):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)
    verbose = 0
    def __init__(self, trainEpisodesBins, nh, nz, nbNodes, dropout=0.0,nlayersK=1, nlayersZK=0, nlayersZ=1, testEpisodesBins=None, globalR=True,historyWeight=1.0,rhistoryWeight=0.0,rnn=0,world=True,tgraph=None,graphForTrain=True):
        super(EMrecCTICModel, self).__init__()

        self.nbNodes=nbNodes

        #self.trainBins=trainEpisodesBins
        #self.T = T

        self.dropout=dropout
        self.world=world
        self.nh=nh
        self.zinit=nn.Embedding(1,nh)
        self.q= nn.ModuleList([])
        #self.qtestIndex = {}
        self.nlayersZ=nlayersZ
        #self.qtest = nn.ModuleList([])
        if nlayersK==0:
            self.knet=KNet(nh,nz,nbNodes,nlayers=nlayersZK, dropout=dropout,historyWeight=historyWeight)
        else:
            self.knet = KNet3(nh, nz, nbNodes, dropout=dropout, nlayers=nlayersK, nlayersZ=nlayersZK, historyWeight=historyWeight)
        #self.knet = KNetFake(nbNodes)

        self.rnet=None
        if globalR:
            self.rnet=RNetGlobal(nh, nbNodes,dropout=dropout,nlayers=1)
        else:
            self.rnet=RNet(nh, nbNodes,dropout=dropout,nlayers=nlayersK,historyWeight=rhistoryWeight)
        self.rnn=rnn
        if self.rnn>=0:
            self.znet=ZNetRNN(nh, nbNodes,dropout=dropout,nlayers=nlayersZ,rnn=rnn)
        else:
            self.znet = ZNetRes(nh, nbNodes,dropout=dropout,nlayers=nlayersZ)
        prs("Build q for train")
        pb = tqdm(trainEpisodesBins, dynamic_ncols=True)
        for (_, infected, times) in pb:
            #prs("times", times)
            nm=times.size()[1]
            nb=times.size()[0]
            c = nn.ModuleList([])
            self.q.append(c)
            for x in range(2,nm):
                m = times[:, x]  # if times <0 => padding
                # prs("m",m)
                m = m[m >= 0]
                sm = m.size()[0]

                # l=e.times[t]
                # for x in l:
                v = nn.Embedding(sm, x )
                v.weight.data.fill_(1.0)
                #ones=torch.ones(sm,x)
                #prs("m",m.view(-1,1))
                #prs("tt", times[:m.size()[0],:x])
                diffT=m.view(-1,1)-times[:m.size()[0],:x]
                #prs("dt",diffT.size(),diffT)
                v.weight.data.mul_((diffT>0).to(self.floatTensor))

                if not tgraph is None and graphForTrain:
                    inft = infected[:sm, x]
                    infb = infected[:sm, :x]
                    gi = tgraph[inft]
                    lg = [gi[i, infb[i]].view(1, -1) for i in range(gi.size()[0])]
                    mask = torch.cat(lg, 0)
                    #prs("xx",infb[0, 0].item())
                    #prs("xi", inft[0].item())
                    #prs("test",[1 if (infb[0, j].item(), inft[0].item()) in gt else 0 for j in range(infb.size()[1])])
                    #mask = self.floatTensor.new(
                    #    [[graph[infb[i, j].item(), inft[i].item()] for j in range(infb.size()[1])] for i in range(sm)])
                    v.weight.data.mul_(mask.to(self.floatTensor))

                c.append(v)
                #print("w "+str(x)+" :"+str(v.weight))
                #np += 1
            #
            #self.q.append(c)
            #l=e.listN
            #k = sorted(e.times.keys())
            #np = 0

        if testEpisodesBins is not None:
            #self.testBins=testEpisodesBins
            self.buildForTest(testEpisodesBins,tgraph)

        self.allNodes=None

    def setcuda(self,device):

        EMrecCTICModel.floatTensor = torch.cuda.FloatTensor(1, device=device)
        EMrecCTICModel.longTensor = torch.cuda.LongTensor(1, device=device)
        self.cuda(device=device)

        #print(DiffusionModel2.longTensor)


    #def clearQTest(self):
    #    self.qtest = nn.ModuleList([])
    #    #self.qtestIndex={}

    def prs(self,*args):
        if self.verbose>0:
            st = ""
            for s in args:
                st += str(s)
            print(st)




    def hook(self,msg):
        #if self.verbose > 1:
        #    print("grad:" + str(grad))
        return lambda grad:print(msg+": "+str(grad))





    # els a matrix with sums to be performed on dim
    def logsumexp(self, els, dim=-1, val=None, t=-1):
        #els.register_hook(lambda grad: print("els hook " + str(grad)))
        #print("els="+str(els))
        #els.register_hook(lambda grad: print("els hook " + str(grad)))

        if val is None:
            # dels=els.data.numpy()
            # c=np.argmax(dels,dim)
            # prs("els=",els)
            val = els.max(dim, True)[0]  # [range(len(dels)),c]
            # prs("val = ",val)
        val = val.detach()
        #val.register_hook(self.hook("val =" + str(val)))
        # prs("val = ", val)
        #if t==4:
        #    els.register_hook(self.hook("els before clone " + str(t) + "=" + str(els)))
        #els=els.clone()
        #if t == 4:
        #    els.register_hook(self.hook("els after clone "+str(t) + "=" + str(els)))
        ex = torch.exp(els - val)
        #if t == 4:
        #    ex.register_hook(self.hook("ex ="+str(t) + "=" + str(ex)))
        ret = val + torch.log(ex.sum(dim, True))

        return ret



    def considerGraph(self,bins,tgraph,train=False):
        q=self.qtest
        if train:
            q = self.q
        ibin=0
        prs("Consider Graph train=",train)
        pb = tqdm(bins, dynamic_ncols=True)
        for (_, infected, times) in pb:
            qb=q[ibin]
            nm = times.size()[1]
            iq=0
            for x in range(2, nm):
                m = times[:, x]  # if times <0 => padding
                m = m[m >= 0]
                sm = m.size()[0]
                diffT = m.view(-1, 1) - times[:m.size()[0], :x]
                qb[iq].weight.data.mul_((diffT > 0).to(self.floatTensor))
                inft = infected[:sm, x]
                infb = infected[:sm, :x]
                gi=tgraph[inft]
                lg=[gi[i,infb[i]].view(1,-1) for i in range(gi.size()[0])]
                mask = torch.cat(lg, 0)

                # mask = self.floatTensor.new(
                #    [[1 if (infb[i, j].item(), inft[i].item()) in graph else 0 for j in range(infb.size()[1])] for i in
                #    range(sm)])
                #prs(gi)
                #prs(infb)
                #prs("mask",mask)
                #prs("q",qb[iq].weight.data)
                #mask=gi[range(len(gi)), infb].t()
                #mask = self.floatTensor.new(
                #    [[graph[infb[i, j].item(), inft[i].item()] for j in range(infb.size()[1])] for i in range(sm)])
                qb[iq].weight.data.mul_(mask.to(self.floatTensor))
                #prs(ibin,iq,qb[iq].weight)

                iq+=1
            ibin+=1
            #prs(ibin)
        #prs(self.qinfer)
        #


    def buildForTest(self,testBins,tgraph=None):
        #print(str(fromTests))
        self.qtest = nn.ModuleList([])

        # if len(self.qtest)<=(testBin-1):
        #    raise RuntimeError("testBin "+str(testBin)+" : start inferring for previous test bins (only "+len(self.qtest)+" test bins already defined)")
        #s = str(testBin) + "_" + str(lastT)
        #if s not in self.qtestIndex.keys():

        prs("Build q for test ", 0)
        pb = tqdm(testBins, dynamic_ncols=True)
        for (_, infected, times) in pb:
            c = nn.ModuleList([])
            self.qtest.append(c)
            #self.qtestIndex[s] = len(self.qtest) - 1
            nm = times.size()[1]
            nb = times.size()[0]
            for x in range(2, nm):
                m = times[:, x]  # if times <0 => padding
                m = m[m >= 0]
                sm = m.size()[0]
                # l=e.times[t]
                # for x in l:
                v = nn.Embedding(sm, x)
                v.weight.data.fill_(1.0)

                diffT = m.view(-1, 1) - times[:m.size()[0], :x]
                v.weight.data.mul_((diffT > 0).to(torch.FloatTensor([0])))

                if not tgraph is None:

                    inft = infected[:sm, x]
                    #prs("inft",inft)
                    #inft.contiguous().view(-1)
                    infb = infected[:sm, :x]
                    #prs("infb",infb)
                    #infb=infb.contiguous().view(-1)
                    #prs("inft",inft)
                    #prs("infb",infb)
                    #prs("tgraph",tgraph)
                    gi = tgraph[inft]
                    lg = [gi[i, infb[i]].view(1, -1) for i in range(gi.size()[0])]
                    mask = torch.cat(lg, 0)

                    # prs("xx",infb[0, 0].item())
                    # prs("xi", inft[0].item())
                    # prs("test",[1 if (infb[0, j].item(), inft[0].item()) in gt else 0 for j in range(infb.size()[1])])
                    # mask = self.floatTensor.new(
                    #     [[1 if (infb[i, j].item(), inft[i].item()) in graph else 0 for j in range(infb.size()[1])] for i in
                    #      range(sm)])
                    #mask = self.floatTensor.new(
                    #    [[graph[infb[i, j].item(), inft[i].item()] for j in range(infb.size()[1])] for i in range(sm)])
                    v.weight.data.mul_(mask.to(torch.FloatTensor([0])))


                c.append(v)


        #print(str(self.qinfer))

    def simuFrom(self, infected, times, bin, train, testFrom, nbSimu, maxT, graph=None, nbMaxNodes=100):
        q = self.qtest  # infer.__getattr__(str(testFrom))[testBin]
        if train:
            q = self.q
        q=q[bin]

        sampleNotInf = True
        if nbMaxNodes < 0:
            sampleNotInf = False

        #if nbMaxNodes > self.nbNodes or
        if nbMaxNodes <= 0:
            nbMaxNodes = self.nbNodes

        nbInf=torch.zeros((nbSimu,infected.size()[0],self.nbNodes)).to(self.floatTensor)

        notinf = self.longTensor.new(infected.size()[0], self.nbNodes).fill_(1)
        c = torch.where(infected >= 0, infected, self.longTensor.new([0]))
        notinf.scatter_(1, c, torch.zeros(notinf.size()).to(notinf))
        notinf[:, 0] = 0

        if testFrom > 0:
            c = torch.where(times >= testFrom, infected, self.longTensor.new([0]))
            notinf.scatter_(1, c, torch.ones(notinf.size()).to(notinf))
            notinf[:, 0] = 0


        pb=tqdm(range(nbSimu), dynamic_ncols=True)
        for isimu in pb:
            #prs("Simu",isimu)
            qz = self.floatTensor.new(infected.size()[0], 1).fill_(0)
            px = self.floatTensor.new(infected.size()[0], 1).fill_(0)
            pnotx = self.floatTensor.new(infected.size()[0], 1).fill_(0)
            zprev = self.zinit(self.longTensor.new([0] * infected.size()[0] * self.nlayersZ)).view(self.nlayersZ,
                                                                                                   infected.size()[0], 1,
                                                                                                   self.nh)
            z0 = self.zinit(self.longTensor.new([0] * self.nlayersZ)).view(self.nlayersZ,self.nh)
            who = self.longTensor.new([0] * infected.size()[0]).view(infected.size()[0], 1, 1)

            notstopped = (infected[:, 0] >= 0)
            infectious=torch.zeros((infected.size()[0],self.nbNodes)).to(infected)
            infectious[:,0]=1

            inferedTimes=torch.ones((infected.size()[0],self.nbNodes)).to(zprev)*(float("inf"))
            inferedTimes[:,0]=0

            states=[{0:z0} for i in range(infected.size()[0])]
            fromW=[{0:0} for i in range(infected.size()[0])]
            last=torch.LongTensor([0]*infected.size()[0]).to(infected)
            tlast = torch.LongTensor([0] * infected.size()[0]).to(times)
            z0 = self.nlayersZ

            for t in range(0, infected.size()[1]):
                # prs("t",t)
                quand = times[:, t]  # if times <0 => padding

                last[notstopped] = infected[notstopped, t].view(-1)
                tlast[notstopped] = quand[notstopped]
                sel = quand[notstopped]
                sel = (sel >= 0) * (sel < testFrom)

                notstopped = (notstopped * ((quand >= 0) * (quand < testFrom)))

                # pas avant et maintenant
                qui = infected[notstopped, t].contiguous().view(-1, 1)
                if qui.size()[0] == 0:
                    #prs("not",notstopped,t)
                    break

                zprev = zprev[:, sel]
                who = who[sel]



                z = None
                a = None
                if (t > 0):

                    k = self.knet(zprev[-1], who, qui).view(qui.size()[0], -1)
                    logr = self.rnet(zprev[-1], who, qui).view(qui.size()[0], -1)
                    diffTimes = quand[notstopped].view(-1, 1) - times[notstopped, :t]

                    # pour degager les diff entre memes temps (ou pas dans le graphe)
                    mask = (diffTimes > 0).to(k)
                    if t > 1:
                        qt = q[t - 2](self.longTensor.new(range(q[t - 2].num_embeddings)))
                        qt = qt.pow(2)
                        #prs("qt", qt, "sel",sel)

                        sel = quand[quand >= 0]
                        sel = (sel >= 0) * (sel < testFrom)
                        qt = qt[sel]

                        mask.mul_((qt > 0).to(k))

                    if not self.world:
                        hasant = mask.sum(-1)
                        noant = (hasant <= 1).to(k).view(-1)
                        hasant = (noant == 0).to(k).view(-1, 1)
                        mask[range(mask.size()[0]), 0] = noant

                    k = k * mask
                    k = torch.max(k, self.floatTensor.new([1e-20]))
                    logk = torch.log(k)
                    ex = logk - torch.exp(logr) * torch.log(diffTimes + 1)
                    a = logr + ex
                    minusk = torch.max(1 - k, self.floatTensor.new([1e-10]))
                    minusk = torch.log(minusk)

                    maxs = torch.max(minusk, ex)

                    # prs("maxs=", maxs)
                    maxs = maxs.detach()
                    # print("maxs=" + str(maxs))
                    b = maxs + torch.log(torch.exp(ex - maxs) + torch.exp(minusk - maxs))

                    a = (a - b).view(qui.size()[0], t)


                    if t == 1:
                        r = self.longTensor.new(qui.size()[0]).fill_(0)
                    else:
                        c = torch.distributions.categorical.Categorical(logits=a)
                        r = c.sample()
                        pqt = c.log_prob(r)
                        pqt = pqt.view(-1, 1)
                        ze = self.floatTensor.new(qz.size()[0], 1).fill_(0)
                        # prs("ze",ze)
                        ze[notstopped] = pqt
                        pqt = ze
                        qz = qz + pqt

                    rz = r.view(1, -1).expand(zprev.size()[0], -1).contiguous().view(-1)

                    z0 = zprev.size()[0]
                    zs = zprev.size()[0] * zprev.size()[1]
                    # prs("zsize",zprev.size())
                    z = zprev.contiguous().view(zs, -1, self.nh)
                    z = z[range(zs), rz]

                    znext = self.znet(z.view(z0, -1, 1, self.nh), qui)  # .

                    # print("######## compute proba px  "+str(t)+"###############")

                    sa = a[range(qui.size()[0]), r].view(-1, 1)

                    sb = b.sum(-1).view(-1, 1)  # 0, True)
                    # prs("sb", sb.size())

                    ze = self.floatTensor.new(px.size()[0], 1).fill_(0)

                    if not self.world:
                        sa = sa * hasant

                    ze[notstopped] = sa  # *hasant
                    sa = ze

                    ze = self.floatTensor.new(px.size()[0], 1).fill_(0)

                    if not self.world:
                        sb = sb * hasant

                    ze[notstopped] = sb.view(-1, 1)  # *hasant
                    sb = ze

                    px = px + sa + sb

                    who = torch.cat((who, qui.view(-1, 1, 1)), 1)
                    # print("######## gener z "+str(t)+"########################")

                    zprev = torch.cat((zprev, znext), dim=2)
                    # prs("zprevafter", zprev)
                    z = znext[-1].view(-1, self.nh)

                if (t == 0):
                    if not self.world:
                        continue
                    z = zprev[-1, :, 0]
                    znext=zprev[:,notstopped,0]
                    r=self.longTensor.new(np.zeros(notstopped.size()[0]))

                lesquels = torch.LongTensor(range(notstopped.size()[0])).to(infected)
                lesquels = lesquels[notstopped]

                # prs("qui",qui,"lesquels",lesquels)
                for i in range(qui.size()[0]):
                    ep = lesquels[i].item()
                    dic = states[ep]
                    qi = qui[i].item()
                    dic[qi] = znext[:, i].view(z0, self.nh)
                    dic = fromW[ep]
                    # print(i,r[i],who,who[i,r[i]].item())
                    dic[qi] = who[i, r[i]].item()
                    inferedTimes[ep, qi] = quand[ep]
                    infectious[ep, qi] = 1

                # a=torch.

                iwhich = self.longTensor.new(range(infected.size()[0]))[notstopped]
                niAll = notinf[iwhich]

                quiAll = qui
                if not graph is None:
                    qg = graph[quiAll].view(quiAll.size()[0], self.nbNodes)

                    niAll = (niAll * qg).contiguous()
                    qgs = niAll.sum(-1)
                    which = (qgs > 0)
                    niAll = niAll[which]
                    if niAll.size()[0] == 0:
                        #prs("graph",graph,"niAll",niAll,"qg",qg,"quiAll",quiAll,"which",which)
                        continue
                    quiAll = quiAll[which]
                    iwhich = iwhich[which]
                    z = z[which]
                fW = quiAll.unsqueeze(1).expand(-1, nbMaxNodes, -1).contiguous().view(-1, 1, 1)
                avec = z.unsqueeze(1).expand(-1, nbMaxNodes, -1)
                avec = avec.contiguous().view(-1, self.nh)

                ze = self.floatTensor.new(pnotx.size()[0], 1).fill_(0)

                if sampleNotInf:
                    quiAll = torch.multinomial(niAll.to(self.floatTensor), nbMaxNodes, True)
                    #print("quiAll",quiAll.size())
                else:
                    quiAll = self.longTensor.new(range(self.nbNodes)).repeat(iwhich.size()[0], 1).view(-1)

                k = self.knet(avec.unsqueeze(1), fW, quiAll.view(-1, 1))  # .view(quand.size()[0],-1)

                minusk = torch.max(1 - k, self.floatTensor.new([1e-10]))
                b = torch.log(minusk)
                #if testFrom>0
                logr = self.rnet(avec.unsqueeze(1), fW, quiAll.view(-1, 1))  # .view(quand.size()[0],-1)
                diffTimes = testFrom - 1 - times[iwhich, t]

                diffTimes = diffTimes.unsqueeze(1)
                # prs("df", diffTimes)
                diffTimes = diffTimes.expand(-1, nbMaxNodes)
                diffTimes = torch.log(diffTimes.contiguous().view(-1, 1) + 1)

                ex = torch.log(k) - torch.exp(logr) * diffTimes
                maxs = torch.max(b, ex)
                # prs("maxs=", maxs)
                maxs = maxs.detach()
                # print("maxs=" + str(maxs))
                b0 = maxs + torch.log(torch.exp(ex - maxs) + torch.exp(b - maxs))

                b = b0
                    # print("b",str(b))

                b = b.view(-1, nbMaxNodes)
                # b.register_hook(self.hook("b  " + str(t) + " " + str(b)))

                if not sampleNotInf:
                    b = b * niAll.float()
                    b = b.sum(-1)
                else:
                    nbNotInf = niAll.sum(-1).view(-1)
                    b = b.sum(-1)
                    b = b * nbNotInf.to(b) / nbMaxNodes

                b = b.view(-1, 1)


                ze[iwhich] += b

                b = ze
                pnotx = pnotx + b  # .sum(0, True)


            # prs("infected", infected)
            # prs("times", times)

            step=0
            #if onlyLinks:

            #prs("onlyNext",onlyNext)
            while True:
                step+=1
                #prs("step",step)
                inft=torch.where(infectious>0,inferedTimes,self.floatTensor.new([maxT+1]))
                m=torch.min(inft,-1)
                quand=m[0]
                qui=m[1]

                infectious[range(infectious.size()[0]), qui.view(-1)] = 0
                quels=quand<maxT+1
                if quels.sum()==0:
                    break

                lstates=[states[i][qui[i].item()].view(z0,1,1,self.nh) for i in range(quels.size()[0]) if quels[i]>0]
                z=torch.cat(lstates,1)

                avec=z[-1,:, 0]

                source = qui[quels].view(-1,1,1).expand(-1, self.nbNodes, -1).contiguous().view(-1, 1, 1)
                avec = avec.unsqueeze(1).expand(-1, self.nbNodes, -1)
                avec = avec.contiguous().view(-1, self.nh)
                versqui = self.longTensor.new(range(self.nbNodes)).repeat(z.size()[1], 1).view(-1)

                k = self.knet(avec.unsqueeze(1), source, versqui.view(-1, 1)).view(-1,self.nbNodes)

                if graph is not None:

                    g=graph[qui[quels]].view(-1,self.nbNodes)
                    #prs(g)
                    #prs(k)
                    k=k*g.to(k)
                    #infected[sel,t]
                #k = k * inferedTimes

                #prs("k", k)
                r=torch.rand_like(k)

                logr = self.rnet(avec.unsqueeze(1), source, versqui.view(-1, 1))
                #prs(torch.exp(logr).view(-1,self.nbNodes))
                c=torch.distributions.Exponential(torch.exp(logr).view(-1,self.nbNodes))
                ti=c.rsample()
                ti=torch.exp(ti)-1
                ti=quand[quels].view(-1,1)+ti
                #prs("ti", ti)
                wt=(ti<testFrom)*(r<k)
                nbT=0
                while(wt.sum()>0 and nbT<2):
                    ti = ti*(1-wt).to(ti)+(quand[quels].view(-1,1)+torch.exp(c.rsample())-1)*wt.to(ti)
                    wt = (ti < testFrom) * (r<k)
                    #prs("ti_", nbT, ti)
                    nbT+=1

                ti= ti * (ti>=testFrom).to(ti) * (ti<maxT+1).to(ti)  * (r<k).to(ti)

                ti=ti.view(-1,self.nbNodes)
                #prs("ti", ti)
                newInfected=(inferedTimes[quels]>ti) * (ti>0)
                #prs("newInfected", newInfected)
                lnewInfected=newInfected.nonzero()
                if lnewInfected.size()[0]==0:
                    continue
                # prs("lnewInfected", lnewInfected)
                avec=z[:,lnewInfected[:,0]]
                versqui=lnewInfected[:,1].view(-1,1)

                znext = self.znet(avec.view(avec.size()[0], -1, 1, self.nh), versqui).view(-1,versqui.size()[0],self.nh)

                #prs("znext",znext)
                l=lnewInfected.tolist()
                quelspas0=torch.nonzero(quels.view(-1))
                for i in range(len(l)):
                    ep=quelspas0[l[i][0]]
                    qi=l[i][1]
                    dic=states[ep]
                    dic[qi]=znext[:,i].view(z0,self.nh)
                    dic=fromW[ep]
                    dic[qi]=qui[quels][l[i][0]].item()
                    inferedTimes[ep,qi]=ti[l[i][0],l[i][1]]
                    infectious[ep,qi]=1

            #prs("inferredT", inferedTimes)
            #prs("fromW", fromW)
            #prs(px,pnotx,qz,pdtau)
            ad=torch.where(inferedTimes < (maxT + 1),self.floatTensor.new([1]),self.floatTensor.new([0]))
            #prs("ad",ad)
            nbInf[isimu] +=ad #torch.where(ad<=0.0,ad,self.floatTensor.new([0]))

        #prs("nbInf", nbInf)
        #nbInf=self.logsumexp(nbInf,dim=0)
        #nbInf=nbInf.sum
        nbInf=nbInf.sum(0)
        rInf=nbInf/nbSimu
        #rInf=torch.where(rInf<=0.0,rInf,self.floatTensor.new([0.0]))
        #rInf=torch.exp(rInf)
        #prs("nbInf", rInf)
        rFin = rInf.sum()
        gt=torch.zeros((infected.size()[0],self.nbNodes)).to(infected)
        c = torch.where(infected >= 0, infected, self.longTensor.new([0]))
        gt.scatter_(1, c, torch.ones(gt.size()).to(gt))
        l=torch.where(gt>0,rInf,1-rInf)

        #prs("gt", gt.to(self.floatTensor))
        #prs("l", l[0])
        l=torch.log(l+1E-10)
        #prs("l",l.sum())
        rInf=l.sum()



        return(rInf,rFin)



        #self.qinfer = nn.ModuleList([])
    # infected is a (nepisodes,nInfectedMax) matrix with the list of infected for each episode (-1 is padding)
    # times is a (nepisodes,nInfectedMax) matrix with the list of infection timestamps
    #notinf is a (nepisodes,nNodes) of binaries where 1 indicates non infected
    def forward(self, infected, times, trainBin=-1, testBin=-1, nbMaxNodes=-1, tau=-1, graph=None, fromQui=None, firstStepLinksIncluded=True, condDist=0):
        verif=False
        #testFrom=tau
        #if fromQui is None:
        #    print("From Qui None  for "+str(trainBin))
        #torch.set_printoptions(threshold=50000)
        #prs("times",times)
        nbInfected=0
        nbLinks = torch.zeros((infected.size()[0])).to(self.floatTensor)
        if fromQui is not None:
            goodLinks = torch.zeros((infected.size()[0])).to(self.floatTensor)
            nbLinks = torch.zeros((infected.size()[0])).to(self.floatTensor)

        notinf = self.longTensor.new(infected.size()[0], self.nbNodes).fill_(1)
        c = torch.where(infected >= 0, infected, self.longTensor.new([0]))
        notinf.scatter_(1, c, torch.zeros(notinf.size()).to(notinf))
        notinf[:, 0] = 0


        if tau>0:
            notinf2=notinf
            notinf=notinf2.clone()
            c = torch.where(times >= tau, infected, self.longTensor.new([0]))
            notinf.scatter_(1, c, torch.ones(notinf.size()).to(notinf))
            notinf[:, 0] = 0

        sampleNotInf=True
        if nbMaxNodes<0:
            sampleNotInf=False

        if nbMaxNodes<=0:
            nbMaxNodes = self.nbNodes

        #if lastT<0:
        #    lastT=self.T
        if testBin>=0 and trainBin>=0:
            raise RuntimeError("test and train bin both defined!")

        #if self.allNodes is None:
        #    self.allNodes=self.longTensor.new(range(self.nbNodes))

        #proba=Variable(self.floatTensor.new(1,1).fill_(0),volatile=not self.training)
        #pz=self.floatTensor.new(infected.size()[0],1).fill_(0)
        qz=self.floatTensor.new(infected.size()[0], 1).fill_(0)
        px=self.floatTensor.new(infected.size()[0], 1).fill_(0)
        pnotx=self.floatTensor.new(infected.size()[0], 1).fill_(0)

        qz2 = self.floatTensor.new(infected.size()[0], 1).fill_(0)
        px2 = self.floatTensor.new(infected.size()[0], 1).fill_(0)
        pnotx2 = self.floatTensor.new(infected.size()[0], 1).fill_(0)

        if verif:
            qz3 = self.floatTensor.new(infected.size()[0], 1).fill_(0)
            px3 = self.floatTensor.new(infected.size()[0], 1).fill_(0)
            pnotx3 = self.floatTensor.new(infected.size()[0], 1).fill_(0)

        #pnotx=pnotx.repeat(1,1)
        #print("infected:"+str(infected))
        #prs("times=",times)
        q=None
        if trainBin>=0:
            q=self.q[trainBin]
        else:
            if testBin<0:
                raise RuntimeError("neither test or train bin defined!")

            #s = str(testBin) + "_" + str(lastT)
            q=self.qtest[testBin] #self.qtestIndex[s]]

            # if (not testMode) and (testFrom>0):
            #     notinf=notinf.clone()
            #     c=torch.where(times >= testFrom, infected, self.longTensor.new([0]))
            #     notinf.scatter_(1, c, torch.ones(notinf.size()).to(notinf))
            #     notinf[:,0]=0
            #     q = self.qinfer.__getattr__(str(testFrom))[testBin]

        #zprev = Variable(self.longTensor.new(infected.size()[0],infected.size()[1],self.nh).fill_(0),volatile=not self.training)
        #print(DiffusionModel2.longTensor)
        #print("par :"+str(self.zinit.parameters().__next__()))
        #prs("infected size=",infected.size()[0])
        if self.rnn>=0:
            zz=self.nlayersZ
        else:
            zz=1

        zprev=self.zinit(self.longTensor.new([0]*infected.size()[0]*zz)).view(zz,infected.size()[0],1,self.nh)
        who=self.longTensor.new([0]*infected.size()[0]).view(infected.size()[0],1,1)
        #zprev=who.clone()
        #print("q:" + str(q))


        for t in range(0,infected.size()[1]):
            #prs("t",t)
            quand = times[:, t]  # if times <0 => padding
            quand = quand[quand >= 0].contiguous()


            zprev = zprev[:,:quand.size()[0]]
            z=None
            a=None
            qui = infected[:quand.size()[0], t].contiguous().view(-1, 1)
            who=who[:quand.size()[0]]
            if (t>0):

                k = self.knet(zprev[-1], who[:quand.size()[0]], qui).view(quand.size()[0], -1)
                logr = self.rnet(zprev[-1], who[:quand.size()[0]], qui).view(quand.size()[0], -1)
                diffTimes = quand.view(-1, 1) - times[:quand.size()[0], :t]

                # pour degager les diff entre memes temps (ou pas dans le graphe)
                mask = (diffTimes > 0).to(k)
                if t > 1:
                    qt = q[t - 2](self.longTensor.new(range(q[t - 2].num_embeddings)))
                    qt = qt.pow(2)
                    # prs("mask", mask.size())
                    # prs("qt", qt.size())
                    mask.mul_((qt > 0).to(k))

                if not self.world:
                    hasant = mask.sum(-1)
                    noant = (hasant <= 1).to(k).view(-1)
                    hasant = (noant == 0).to(k).view(-1, 1)
                    mask[range(mask.size()[0]), 0] = noant

                k = k * mask
                k = torch.max(k, self.floatTensor.new([1e-20]))
                logk = torch.log(k)
                ex = logk - torch.exp(logr) * torch.log(diffTimes + 1)
                a = logr + ex
                minusk = torch.max(1 - k, self.floatTensor.new([1e-10]))
                minusk = torch.log(minusk)

                maxs = torch.max(minusk, ex)

                # prs("maxs=", maxs)
                maxs = maxs.detach()
                # print("maxs=" + str(maxs))
                b = maxs + torch.log(torch.exp(ex - maxs) + torch.exp(minusk - maxs))


                #b=b.detach()
                #b=torch.exp(minusk-b).detach() * minusk + torch.exp(ex-b).detach() * ex

                a = (a - b).view(quand.size()[0], t)

                if tau>0:
                        #print("b before", b)
                        if verif:
                            #averif = (a - b).view(quand.size()[0], t)
                            bverif = b.clone()
                        diffTimes0 = tau - 1 - times[:quand.size()[0], :t]
                        diffTimes0[diffTimes0<0]=0
                        ex0 = logk - torch.exp(logr) * torch.log(diffTimes0 + 1)
                        maxs = torch.max(minusk, ex0)
                        maxs = maxs.detach()
                        # print("maxs=" + str(maxs))
                        b0 = maxs + torch.log(torch.exp(ex0 - maxs) + torch.exp(minusk - maxs))
                        b0[diffTimes0<0]=0
                        #sb0 = b0.sum(-1).view(-1,1)
                        b=torch.where((quand >= tau).view(-1,1).expand(-1,b.size()[-1]),b-b0,b)
                        #print("b after", b)
                        #b2=
                        #[quand < testFrom] = 0
                        #prs("sb", sb.size())
                        #prs("sb0", sb0.size())
                        #sb=sb-sb0






                #nbInfected += a.size()[0]

                # sta = self.logsumexp(a, t=t)
                # # prs("sa", sa)
                # if (sta.data == float("inf")).sum() > 0 or (sta.data != sta.data).sum() > 0:
                #     torch.set_printoptions(threshold=5000)
                #     prs("sta", sta)
                #     prs("a", a)
                #     prs("b", b)
                #     prs("k", k)
                #     prs("logr", logr)
                #     prs("diifTimes", diffTimes)
                #     prs(quand.view(-1, 1))
                #     prs(times[:quand.size()[0], :t])
                #     exit(0)

                # prs("sa", sa.size())




                if t==1:
                    r=self.longTensor.new(quand.size()[0]).fill_(0)
                    #goodLinks[:quand.size()[0]] +=quand[:quand.size()[0]].to(nbLinks)
                    #nbLinks[:quand.size()[0]] += quand[:quand.size()[0]].to(nbLinks)
                    if (firstStepLinksIncluded or t>1) and fromQui is not None:
                        gg=goodLinks[:quand.size()[0]]
                        gg[quand>=tau] +=1
                else:
                    if condDist==1:
                        hh=a.clone()
                        hh[quand<tau]=1
                        c = torch.distributions.categorical.Categorical(logits=hh)
                        r = c.sample()

                    c=torch.distributions.categorical.Categorical(logits=a)
                    if condDist==0:
                        r = c.sample()
                    pqt = c.log_prob(r)

                    pqt = pqt.view(-1, 1)
                    if tau>0:
                        pqt2=pqt.clone()
                        pqt[quand >= tau]=0
                        pqt2[quand < tau] = 0

                        if verif:
                            #cverif = torch.distributions.categorical.Categorical(logits=a)
                            pqt3 = c.log_prob(r)
                            pqt3 = pqt3.view(-1, 1)
                            qz3[:pqt3.size()[0]] += pqt3

                    #ze = self.floatTensor.new(qz.size()[0], 1).fill_(0)
                    # prs("ze",ze)
                    #ze[:pqt.size()[0], ] = pqt
                    #pqt = ze
                    qz[:pqt.size()[0]] += pqt
                    if tau>0:
                        qz2[:pqt2.size()[0]] += pqt2
                    if (firstStepLinksIncluded or t>1) and fromQui is not None:
                        gg = goodLinks[:quand.size()[0]]
                        ff=torch.exp(c.log_prob(fromQui[:quand.size()[0], t].view(-1)))
                        gg[quand>=tau] += ff[quand>=tau]

                if (firstStepLinksIncluded or t>1) and fromQui is not None:
                    #prs(fromQui)
                    #prs(fromQui[quand.size()[0], t].view(-1))
                    #prs(r.view(-1))

                    #goodLinks[:quand.size()[0]] += (fromQui[:quand.size()[0], t].view(-1) == r.view(-1)).to(goodLinks)
                    gg = nbLinks[:quand.size()[0]]
                    gg[quand>=tau] += 1

                #else:
                #    print("From Qui None !!! ")

                rz = r.view(1, -1).expand(zprev.size()[0], -1).contiguous().view(-1)

                z0 = zprev.size()[0]
                zs = zprev.size()[0] * zprev.size()[1]
                # prs("zsize",zprev.size())
                z = zprev.contiguous().view(zs, -1, self.nh)
                z = z[range(zs), rz]

                znext = self.znet(z.view(z0, -1, 1, self.nh), qui)  # .



                #whot=who[range(who.size()[0]),r]


                #print("######## compute proba px  "+str(t)+"###############")

                #if testFrom>0 and testMode:



                sa=a[range(quand.size()[0]), r].view(-1, 1)
                if tau>0:
                    sa2=sa.clone()

                sb = b.sum(-1).view(-1,1) #0, True)
                if tau>0:
                    sb2=sb.clone()
                #prs("sb", sb.size())



                if tau>0:
                        sa[quand >= tau] = 0
                        sb[quand >= tau] = 0
                        sa2[quand < tau] = 0
                        sb2[quand < tau] = 0


                #ze = self.floatTensor.new(px.size()[0], 1).fill_(0)
                #if not self.world:
                #    sa = sa* hasant
                #prs("a",sa.size())
                #prs("ze",ze.size())

                #ze[:quand.size()[0]] = sa # *hasant
                #sa = ze


                #ze = self.floatTensor.new(px.size()[0], 1).fill_(0)

                #if not self.world:
                #    sb = sb * hasant

                #ze[:quand.size()[0]] = sb.view(-1,1)#*hasant
                #sb = ze




                px[:quand.size()[0]] += sa + sb
                if tau>0:
                    px2[:quand.size()[0]] += sa2 + sb2

                    if verif:
                        px3[:quand.size()[0]] += a[range(quand.size()[0]), r].view(-1, 1)
                        px3[:quand.size()[0]] += bverif.sum(-1).view(-1,1)




                who = torch.cat((who[:quand.size()[0]], qui.view(-1, 1, 1)), 1)
                #print("######## gener z "+str(t)+"########################")


                zprev=torch.cat((zprev,znext),dim=2)
                #prs("zprevafter", zprev)
                z=znext[-1].view(-1,self.nh)


            if (t == 0):
                if not self.world:
                    continue
                z = zprev[-1,:, 0]


            #print(qui)
            if tau>0:
                pnotx+=self.getNotInf(t,notinf,times,qui,quand,z,nbMaxNodes,graph=graph,tau=tau,before=True,sampleNotInf=sampleNotInf)
                #print(pnotx)
                #print("px",px)
                pnotx2+= self.getNotInf(t,notinf2,times,qui,quand,z,nbMaxNodes,graph=graph,tau=tau,before=False,sampleNotInf=sampleNotInf)
                #print(pnotx2)
                #print("px2",px2)
                if verif:
                    pnotx3+= self.getNotInf(t, notinf2, times, qui, quand, z, nbMaxNodes, graph=graph,sampleNotInf=sampleNotInf)
            else:
                pnotx +=self.getNotInf(t,notinf,times,qui,quand,z,nbMaxNodes,graph=graph,sampleNotInf=sampleNotInf)



        if (px.data == float("inf")).sum() > 0 or (px.data!=px.data).sum()>0:
            prs("px", px)

        if (pnotx.data == float("inf")).sum() > 0 or (pnotx.data!=pnotx.data).sum()>0:
            prs("pnotx", pnotx)



        if (qz.data == float("inf")).sum() > 0 or (qz.data!=qz.data).sum()>0:
            prs("qz", qz)
        if fromQui is not None:
            rLinks = goodLinks #torch.where(nbLinks > 0, (goodLinks / (nbLinks.to(goodLinks))), self.floatTensor.new(1).fill_(0)).sum()
            #rLinks = goodLinks #.sum()
        else:
            rLinks=self.floatTensor.new(1).fill_(0)

        if tau>0 and verif:
            print("px1*px2=",(px+px2).sum().item(),"px3=",px3.sum().item())
            print("pnotx1*pnotx2=", (pnotx + pnotx2).sum().item(), "pnotx3=", pnotx3.sum().item())
            print("qz1*qz2=", (qz + qz2).sum().item(), "qz3=", qz3.sum().item())
            print("l1*l2=", (px+px2+pnotx + pnotx2 - qz-qz2).sum().item(), "l3=", (px3+pnotx3-qz3).sum().item())

        return (px,pnotx,qz,px2,pnotx2,qz2,rLinks.sum().item(),nbLinks.sum().item())

    def getNotInf(self,t,notinf,times,qui,quand,z,nbMaxNodes,graph=None,tau=0,before=True,sampleNotInf=False):
        ret=self.floatTensor.new(times.size()[0], 1).fill_(0)
        #quand=times[:, t].view(-1)
        iwhich = (times[:, t].view(-1) >= 0)
        if tau > 0 and before:
            iwhich = iwhich * (times[:, t].view(-1) < tau)
        iwhich = self.longTensor.new(range(times.size()[0]))[iwhich]

        niAll = notinf[iwhich]

        #quand=quand[iwhich]
        which = (quand >= 0)
        if tau > 0 and before:
            which = which * (quand < tau)
            if which.sum() == 0:
                #print("not ", before, ret)
                return ret
        # print(maxT, t, times[:, t], iwhich,which)
        #print(iwhich)
        quiAll = qui[which]
        z = z[which]
        if not graph is None:
            qg = graph[quiAll].view(quiAll.size()[0], self.nbNodes)
            # prs(qui)
            #prs(qg)
            #prs(niAll)
            niAll = (niAll * qg).contiguous()
            qgs = niAll.sum(-1)
            which = (qgs > 0)
            niAll = niAll[which]
            if niAll.size()[0] == 0:
                #print("not ", before, ret)
                return ret
            quiAll = quiAll[which]
            iwhich = iwhich[which]
            z = z[which]
            # prs(niAll.size(), niAll.sum())
        # nbn=self.nbNodes

        #print(iwhich)

        fromW = quiAll.unsqueeze(1).expand(-1, nbMaxNodes, -1).contiguous().view(-1, 1, 1)
        avec = z.unsqueeze(1).expand(-1, nbMaxNodes, -1)
        avec = avec.contiguous().view(-1, self.nh)



        if sampleNotInf:
            qui = torch.multinomial(niAll.to(self.floatTensor), nbMaxNodes, True)
        else:
            qui = self.longTensor.new(range(self.nbNodes)).repeat(iwhich.size()[0], 1).view(-1)

        k = self.knet(avec.unsqueeze(1), fromW, qui.view(-1, 1))  # .view(quand.size()[0],-1)
        # print("x" + str(qui.view(-1,1).size()))
        # print("z" + str(avec.unsqueeze(1).size()))
        # print("k" + str(k.size()))
        minusk = torch.max(1 - k, self.floatTensor.new([1e-10]))
        b = torch.log(minusk)
        if (tau > 0):
            logr = self.rnet(avec.unsqueeze(1), fromW, qui.view(-1, 1))  # .view(quand.size()[0],-1)
            diffTimes = tau - 1 - times[iwhich, t]

            diffTimes = diffTimes.unsqueeze(1)
            # prs("df", diffTimes)
            diffTimes = diffTimes.expand(-1, nbMaxNodes)
            diffTimes = torch.log(diffTimes.contiguous().view(-1, 1) + 1)

            ex = torch.log(k) - torch.exp(logr) * diffTimes
            maxs = torch.max(b, ex)
            # prs("maxs=", maxs)
            maxs = maxs.detach()
            # print("maxs=" + str(maxs))
            b0 = maxs + torch.log(torch.exp(ex - maxs) + torch.exp(b - maxs))
            if before:
                b = b0
            else:
                b0 = b0.view(-1, nbMaxNodes)
                #prs(b0.size(),quand.size())
                b0[quand[which] >= tau] = 0
                b0 = b0.view(-1, 1)

                b = b - b0
            # print("b",str(b))

        b = b.view(-1, nbMaxNodes)
        # b.register_hook(self.hook("b  " + str(t) + " " + str(b)))

        if not sampleNotInf:
            b = b * niAll.float()
            #print(b.view(-1, nbMaxNodes))
            b = b.sum(-1)
        else:
            nbNotInf = niAll.sum(-1).view(-1)
            b = b.sum(-1)
            b = b * nbNotInf.to(b) / nbMaxNodes
            # if t==2:
            # prs("b ",t,b)

            # b=torch.log(b)
        b = b.view(-1, 1)
        # prs(which.size())
        # prs(iwhich.size())
        # prs(b.size())
        # prs(ze.size())

        ret[iwhich] += b

        #print("not ",before,ret)
        return ret


    def setHW(self,hw,hwr):

        if hw>=0:
            self.knet.autoWeightH=False
            self.knet.hw=hw

        else:
            if not self.knet.autoWeightH:
                self.knet.autoWeightH = True
                self.knet.hW = nn.Embedding(1, 1)
                self.knet.hW.weight.data.fill_(0.0)
        if hwr >= 0:
            self.rnet.autoWeightH = False
            self.rnet.hw = hwr
        else:
            try:
                a=self.rnet.autoWeightH
            except AttributeError:
                a=False
            if not a:
                self.rnet.autoWeightH = True
                self.rnet.hW = nn.Embedding(1, 1)
                self.rnet.hW.weight.data.fill_(0.0)


    # def setUW(self,uw):
    #     if uw>=0:
    #         self.knet.autoWeightU=False
    #         self.knet.uw=uw
    #     else:
    #         if not self.knet.autoWeightU:
    #             self.knet.autoWeightU = True
    #             self.knet.uW = nn.Embedding(1, 1)
    #             self.knet.uW.weight.data.fill_(0.0)


class KNet(nn.Module):
    def __init__(self, inSize, zSize, nbNodes, nlayers=1, dropout=0.,historyWeight=1.0):
        super(KNet, self).__init__()

        self.embedsFrom = nn.Embedding(nbNodes, inSize)

        self.embedsTo = nn.Embedding(nbNodes, inSize)

        self.autoWeightH=(historyWeight<0)
        #print(str(self.autoWeightH))
        self.hw=0
        if historyWeight<0:
            self.hW=nn.Embedding(1,1)
            self.hW.weight.data.fill_(0.0)
        else:
            self.hw = historyWeight
        self.drop = None
        if dropout > 0:
            self.drop = nn.Dropout(dropout)

        self.layersZK=None

        if nlayers>0:
            layersZK = [] + [inSize] * (nlayers - 1) + [zSize]
            self.layersZK = nn.ModuleList([])
            ins = inSize
            for x in layersZK:
                self.layersZK.append(nn.Linear(ins, x))
                ins = x

            layers = [] + [inSize] * (nlayers - 1) + [inSize]
            self.layers = nn.ModuleList([])
            ins = zSize + inSize
            for x in layers:
                self.layers.append(nn.Linear(ins, x))
                ins = x
            # self.autoWeightU = (uWeight < 0)
        # self.uw = 0
        # if uWeight < 0:
        #     self.uW = nn.Embedding(1, 1)
        #     self.uW.weight.data.fill_(0.0)
        # else:
        #     self.uw = uWeight

    def hook(self,msg):
        #if self.verbose > 1:
        #    print("grad:" + str(grad))
        return lambda grad:print(msg+": "+str(grad))


    def forward(self, z, fromWho, pourQui):

        if self.autoWeightH:
            wh=self.hW(pourQui.new([0]))
            #w=torch.pow(w,2)
            #w=1/torch.exp(w)
            wh=F.sigmoid(wh)
            #w.register_hook(self.hook("w hook " + ", w=" + str(w)))
        else:
            wh=self.hw
        #prs("pourQui",pourQui)
        x = self.embedsTo(pourQui)  # matrix (len(PourQui),d)
        #prs("u",fromWho)
        u = self.embedsFrom(fromWho.view(-1))
        #prs("u2", u)
        u=u.view(z.size()[0],z.size()[1],z.size()[2])

        try:

            if self.layersZK is not None:

                for i in range(0, len(self.layersZK)-1):
                    z= self.layersZK[i](z)
                    #prs("x ",i,x.view(-1,x.size()[-1]))
                    z = F.leaky_relu(z)
                    #prs("x ", i, x.view(-1, x.size()[-1]))
                    if self.drop is not None:
                        z = self.drop(z)
                z = self.layersZK[-1](z)
                z=torch.sigmoid(z)

                z = torch.cat((z * wh, u), -1)
                for i in range(0, len(self.layers)-1):
                    z= self.layers[i](z)
                    #prs("x ",i,x.view(-1,x.size()[-1]))
                    z = F.leaky_relu(z)
                    #prs("x ", i, x.view(-1, x.size()[-1]))
                    if self.drop is not None:
                        z = self.drop(z)
                z = self.layers[-1](z)
                z=torch.sigmoid(z)
            else:
                z = wh * z + (1 - wh) * u
        except NameError:
            z=wh * z + (1 - wh) * u



        #prs("x:", x)
        #prs("u:", u)
        x=x.expand(-1,z.size()[1],-1)






        # if self.autoWeightU:
        #     wu=self.uW(pourQui.new([0]))
        #     #w=torch.pow(w,2)
        #     #w=1/torch.exp(w)
        #     wu=F.sigmoid(wu)
        #     #w.register_hook(self.hook("w hook " + ", w=" + str(w)))
        # else:
        #     wu=self.uw
        #z

        x = F.sigmoid((x*z).sum(-1)/z.size()[2])
        #x = -torch.log(1+torch.exp(-(x*z).sum(-1)))
        return x


class KNet3(nn.Module):
    def __init__(self, inSize, zSize, nbNodes, nlayers=1, nlayersZ=0, dropout=0., historyWeight=1, **args):
        super(KNet3, self).__init__()
        outSize = 1
        d = outSize
        layers=[]+[inSize]*(nlayers-1)+[1]
        if (len(layers) > 0):
            d = layers[0]
        self.embedsFrom = nn.Embedding(nbNodes, inSize)

        self.embedsTo = nn.Embedding(nbNodes, inSize)
        self.drop=None
        if dropout>0:
            self.drop=nn.Dropout(dropout)
        ins=zSize+inSize*2
        if nlayersZ==0:
            ins = inSize * 3

        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(ins, x))
            ins = x

        self.layersZ=None
        if nlayersZ>0:
            layersZ = [] + [inSize] * (nlayersZ - 1) + [zSize]
            self.layersZ = nn.ModuleList([])
            ins = inSize
            for x in layersZ:
                self.layersZ.append(nn.Linear(ins, x))
                ins = x
            #self.layers.append(nn.Linear(inSize, outSize))
        # self.tt=nn.Linear(inSize,outSize)

        self.autoWeightH = (historyWeight < 0)
        #self.autoWeightU = (uWeight < 0)

        self.hw = 0
        #self.uw = 0
        if historyWeight < 0:
            self.hW = nn.Embedding(1, 1)
            self.hW.weight.data.fill_(0.0)
        else:
            self.hw = historyWeight

        # if uWeight < 0:
        #     self.uW = nn.Embedding(1, 1)
        #     self.uW.weight.data.fill_(0.0)
        # else:
        #     self.uw = uWeight

    def forward(self, z, fromWho, pourQui):
        #prs("pourQui",pourQui)
        if self.autoWeightH:
            wh=self.hW(pourQui.new([0]))
            #w=torch.pow(w,2)
            #w=1/torch.exp(w)
            wh=F.sigmoid(wh)
            #w.register_hook(self.hook("w hook " + ", w=" + str(w)))
        else:
            wh=self.hw
        # if self.autoWeightU:
        #     wu=self.uW(pourQui.new([0]))
        #     #w=torch.pow(w,2)
        #     #w=1/torch.exp(w)
        #     wu=F.sigmoid(wu)
        #     #w.register_hook(self.hook("w hook " + ", w=" + str(w)))
        # else:
        #     wu=self.uw


        x = self.embedsTo(pourQui)  # matrix (len(PourQui),d)
        u = self.embedsFrom(fromWho.view(-1)).view(z.size()[0],z.size()[1],z.size()[2])

        #prs("x:", x)
        #prs("u:", u)
        x=x.expand(-1,z.size()[1],-1)
        #z = self.layers[0](z)
        #prs("x:",x)
        #prs("z:",z)
        #x = torch.cat((z.repeat(1,1,1).fill_(0),u,x),-1)
        if self.layersZ is not None:
            for i in range(0, len(self.layersZ)-1):
                z = self.layersZ[i](z)
                #prs("x ",i,x.view(-1,x.size()[-1]))
                z = F.leaky_relu(z)
                #prs("x ", i, x.view(-1, x.size()[-1]))
                if self.drop is not None:
                    z = self.drop(z)
            z = self.layersZ[-1](z)
            z=torch.sigmoid(z)

        x = torch.cat((z*wh, u, x), -1)

        #prs("xx",x)
        for i in range(0, len(self.layers)-1):
            x = self.layers[i](x)
            #prs("x ",i,x.view(-1,x.size()[-1]))
            x = F.leaky_relu(x)
            #prs("x ", i, x.view(-1, x.size()[-1]))
            if self.drop is not None:
                x = self.drop(x)
        x = self.layers[-1](x)
        x = F.sigmoid(x).view(-1,1)
        #print("x"+str(x.size()))
        #print("z" + str(z.size()))

        return x

class KNetFake(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.):
        super(KNetFake, self).__init__()
        self.embeds = nn.Embedding(nbNodes, nbNodes)

    def hook(self,msg):
        #if self.verbose > 1:
        #    print("grad:" + str(grad))
        return lambda grad:print(msg+": "+str(grad))

    def forward(self, z, fromWho, pourQui):
        #prs("pourQui",pourQui)
        u = self.embeds(fromWho.view(-1)).contiguous()
        #u.register_hook(self.hook("u hook " + ", u=" + str(u)))

        #prs("u",u)
        #prs("z",z)
        x=pourQui.view(-1,1,1).expand(-1,z.size()[1],-1).contiguous()
        #prs("x:", x)
        x=u[range(u.size()[0]),x.view(-1)]
        #prs("x:", x.view(z.size()[0],z.size()[1],-1))
        #prs("x", x)
        x=F.sigmoid((x))
        #prs("x",x)
        return x

class KNet2(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0., **args):
        super(KNet2, self).__init__()
        outSize = 1
        d = outSize

        layers=[inSize]*(nlayers-1)

        if (len(layers) > 0):
            d = layers[0]
        self.embeds = nn.Embedding(nbNodes, d)
        self.drop=None
        if dropout>0:
            self.drop=nn.Dropout(dropout)

        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        # self.tt=nn.Linear(inSize,outSize)

    def forward(self, z, pourQui):

        x = self.embeds(pourQui)  # matrix (len(PourQui),d)
        z = self.layers[0](z)
        #prs("x:",x)
        #prs("z:",z)
        x = z + x

        for i in range(1, len(self.layers)):
            x = F.sigmoid(x)
            if self.drop is not None:
                x = nn.drop(x)

            x = self.layers[i](x)

        x = F.sigmoid(x)
        return x

class RNet(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.,historyWeight=-1):
        super(RNet, self).__init__()
        outSize = 1
        d = outSize

        #if (len(layers) > 0):
        #    d = layers[0]
        self.embedsFrom = nn.Embedding(nbNodes, inSize)

        self.embedsTo = nn.Embedding(nbNodes, inSize)
        self.drop = None
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        inSize *= 2
        self.layers=None
        if nlayers>0:
            layers = [] + [inSize] * (nlayers - 1) + [1]
            self.layers = nn.ModuleList([])
            for x in layers:
                self.layers.append(nn.Linear(inSize, x))
                inSize = x

        self.autoWeightH = (historyWeight < 0)
        self.hw = 0
        if historyWeight < 0:
            self.hW = nn.Embedding(1, 1)
            self.hW.weight.data.fill_(0.0)
        else:
            self.hw = historyWeight


    def forward(self, z, fromWho, pourQui):
        #prs("pourQui",pourQui)
        x = self.embedsTo(pourQui)  # matrix (len(PourQui),d)
        u = self.embedsFrom(fromWho.view(-1)).view(z.size()[0],z.size()[1],z.size()[2])

        if self.autoWeightH:
            wh=self.hW(pourQui.new([0]))
            #w=torch.pow(w,2)
            #w=1/torch.exp(w)
            wh=F.sigmoid(wh)
            #w.register_hook(self.hook("w hook " + ", w=" + str(w)))
        else:
            wh=self.hw

        z = wh * z + (1 - wh) * u

        #prs("x:", x)
        #prs("u:", u)
        x=x.expand(-1,z.size()[1],-1)
        #z = self.layers[0](z)
        #prs("x:",x)
        #prs("z:",z)
        #x = torch.cat((z.repeat(1,1,1).fill_(0),u,x),-1)
        if self.layers is not None:
            x = torch.cat((z, x), -1)
            #z = u
            for i in range(0, len(self.layers)-1):
                x = self.layers[i](x)
                x = F.sigmoid(x)
                if self.drop is not None:
                    x = self.drop(x)
            #
            x = self.layers[-1](x)
            # x = torch.pow(x, 2) * (1.0 / 100)
        else:
        #z = 0.1 * z + 0.9 * u

            x=torch.abs((x * z).sum(-1)/z.size()[2])
            #x = (x * u).sum(-1) / u.size()[2]
        return -x.view(-1,1)

class RNetGlobal(nn.Module):
    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.):
        super(RNetGlobal, self).__init__()
        outSize = 1
        self.delay = nn.Embedding(1,1)

    def forward(self, z, fromWho, pourQui):
        x=self.delay(pourQui.new([0]))
        x=-torch.pow(x,2)

        return x.expand(z.size()[0],z.size()[1],1)

class RNet2(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.):
        super(RNet2, self).__init__()
        outSize = 1
        d = outSize
        layers = [inSize] * (nlayers-1)
        if (len(layers) > 0):
            d = layers[0]
        self.embeds = nn.Embedding(nbNodes, d)
        self.drop=None
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        # self.tt=nn.Linear(inSize,outSize)

    def forward(self, z, pourQui):
        #print(str(pourQui))
        x = self.embeds(pourQui)  # matrix (len(PourQui),d)
        z = self.layers[0](z)
        x = z + x

        for i in range(1, len(self.layers)):
            x = F.sigmoid(x)
            if self.drop is not None:
                x = nn.drop(x)
            x = self.layers[i](x)

        x = torch.pow(x, 2) * (1.0 / 100)
        return x



class ZNetRNN(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.,rnn=0):
        super(ZNetRNN, self).__init__()

        self.nlayers=nlayers
        self.embeds = nn.Embedding(nbNodes, inSize)
        self.drop=None
        if dropout>0:
            self.drop=nn.Dropout(dropout)

        if rnn==0:
            self.rnn=nn.RNN(inSize,inSize,nlayers,dropout=dropout)
        else:
            self.rnn=nn.GRU(inSize,inSize,nlayers,dropout=dropout)

    def forward(self, z, pourQui):
        #prs("pourQui",pourQui)
        x = self.embeds(pourQui)  # matrix (len(PourQui),d)
        #prs("x:", x)

        x=x.expand(-1,z.size()[2],-1).contiguous()
        nbe=x.size()[0]
        nbp=z.size()[2]
        nh=x.size()[2]

        #z=z.view(-1,self.nlayers,z.size()[2]/self.nlayers)
        #z=z.transpose(0,1)
        x = x.view(1,nbe*nbp, nh)
        z=z.view(self.nlayers,nbe*nbp, nh).contiguous()
        _,w=self.rnn(x,z)
        w=w.view(self.nlayers,nbe,nbp,nh)
        #print("w"+str(w))
        #print("x" + str(x))
        #z = self.layers[0](z)
        #prs("x:",x)
        #prs("z:",z)

        #o,h=self.rnn()

        return w

class ZNetRes(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.):
        super(ZNetRes, self).__init__()

        self.nlayers=nlayers
        self.embedsTo = nn.Embedding(nbNodes, inSize)
        self.drop=None
        if dropout>0:
            self.drop=nn.Dropout(dropout)

        layers = [] + [inSize] * (nlayers - 1) + [1]
        if (len(layers) > 0):
            d = layers[0]

        inSize *= 2
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x


    def forward(self, z, pourQui):
        #prs("pourQui",pourQui)
        #prs("zsize",z.size())
        z = z.squeeze(0)
        #prs("zsize", z.size())
        x = self.embedsTo(pourQui)  # matrix (len(PourQui),d)
        x = x.expand(-1, z.size()[1], -1).contiguous()

        x = torch.cat((z, x), -1)

        # prs("xx",x)
        for i in range(0, len(self.layers) - 1):
            x = self.layers[i](x)
            # prs("x ",i,x.view(-1,x.size()[-1]))
            x = F.leaky_relu(x)
            # prs("x ", i, x.view(-1, x.size()[-1]))
            if self.drop is not None:
                x = self.drop(x)
        x = self.layers[-1](x)



        #x = F.sigmoid(x)

        return (z+x).unsqueeze(0)



class ZNet(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.,rnn=0):
        super(ZNet, self).__init__()
        outSize = inSize
        d = outSize

        layers=[]+[inSize]*(nlayers)

        if (len(layers) > 0):
            d = layers[0]
        self.embeds = nn.Embedding(nbNodes, inSize)
        self.drop=None
        if dropout>0:
            self.drop=nn.Dropout(dropout)
        inSize*=2

        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        #self.layers.append(nn.Linear(inSize, outSize))
        # self.tt=nn.Linear(inSize,outSize)

    def forward(self, z, pourQui):
        #prs("pourQui",pourQui)

        x = self.embeds(pourQui)  # matrix (len(PourQui),d)
        #prs("x:", x)

        x=x.expand(-1,z.size()[1],-1)
        #z = self.layers[0](z)
        #prs("x:",x)
        #prs("z:",z)
        if len(self.layers)>0:
            x = torch.cat((z,x),-1)

            for i in range(0, len(self.layers)-1):
                x = self.layers[i](x)
                x = F.leaky_relu(x)
                if self.drop is not None:
                    x = nn.drop(x)

            x = self.layers[-1](x)
            x = F.sigmoid(x)
        else:
            x=z+x
        return x

class ZNet2(nn.Module):

    def __init__(self, inSize, nbNodes, nlayers=1, dropout=0.):
        super(ZNet2, self).__init__()
        outSize = inSize
        d = outSize
        layers = [inSize] * (nlayers-1)
        if (len(layers) > 0):
            d = layers[0]
        self.embeds = nn.Embedding(nbNodes, d)
        self.drop = None
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        # self.tt=nn.Linear(inSize,outSize)

    def forward(self, z, pourQui):

        x = self.embeds(pourQui)  # matrix (len(PourQui),d)
        z = self.layers[0](z)
        x = z + x

        for i in range(1, len(self.layers)):
            x = F.sigmoid(x)
            if self.drop is not None:
                x = nn.drop(x)

            x = self.layers[i](x)

        return x

