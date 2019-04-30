from __future__ import print_function
import argparse
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import subprocess
import random
import argparse
import scipy.misc
from Episode import Episode
#from torchvision import datasets, transforms
#from torch.autograd import Variable
import torch.autograd.profiler as profiler

import sys
from sortedcollections.recipes import OrderedSet
from collections import OrderedDict
#from blaze.expr.math import isnan
sys.path.append("utils")

from utils import *

from tqdm import tqdm
import json


class NN(nn.Module):
    def __init__(self, nh, outSize=None, nlayers=1, dropout=0., **kwargs):
        super(NN, self).__init__()
        if outSize is None:
            outSize = nh
        self.dropout=None
        if dropout>0:
            self.dropout = nn.Dropout(dropout)
        self.activ = kwargs.get('activ', F.sigmoid)
        self.activFinal = kwargs.get('activFinal', None)
        #nlayers = kwargs.get('nlayers', 1)
        self.layers = nn.ModuleList([])
        inSize = nh
        for x in range(nlayers-1):
            self.layers.append(nn.Linear(inSize, inSize))
            #inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        # self.tt=nn.Linear(inSize,outSize)

    def init_weights(self,initrange):
        for l in self.layers:
            l.bias.data.fill_(0)
            l.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        #print("x"+str(x))
        #print("l0" + str(self.layers[0]))
        x = self.layers[0](x)
        if len(self.layers) > 1:
            if self.dropout is not None:
                x = self.dropout(x)
        for i in range(1, len(self.layers)):
            x = self.activ(x)
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                if self.dropout is not None:
                    x = self.dropout(x)

        if self.activFinal is not None:
            x = self.activFinal(x)
        return x



class DANdiffusionModel(nn.Module):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)
    verbose = 0
    def __init__(self,   T, nbNodes, nh=64,  nwe=32, nlayers=3, dropout=0.0,nbSteps=40):
        super(DANdiffusionModel, self).__init__()
        self.nh=nh
        self.nbNodes=nbNodes
        self.node_embedding = nn.Embedding(nbNodes, nwe, padding_idx=-1)
        #print("nlayers "+str(nlayers))

        self.T=T
        if self.T<100:
            self.T=100
        self.nbSteps=nbSteps
        if self.nbSteps>self.T:
            self.nbSteps=self.T
        self.encoder=NN(nwe,nh,1, dropout=dropout,activFinal=F.elu)
        self.h1 = NN(nh,nh,1, dropout=dropout)
        self.h2 = NN(nh, nh, 1, dropout=dropout)

        self.g = NN(nh*2, 1, 1, dropout=dropout, activFinal=F.sigmoid)
        self.u = NN(nh, nh, 1, dropout=dropout,activFinal=F.elu)
        self.b = NN(nh, 1, 1, dropout=dropout)

        self.nextInfected=NN(nh, nbNodes + 1, nlayers, dropout=dropout)
        self.nextTime = NN(nh, 1, 1, dropout=dropout, activFinal=F.sigmoid)

        self.lambdaT = NN(self.nbSteps+1, 1, 1, dropout=dropout)


        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.dropout = nn.Dropout(dropout)

        self.w = nn.Embedding(1, 1)

        self.init_weights()

    def setcuda(self,device):
        DANdiffusionModel.floatTensor = torch.cuda.FloatTensor(1,device=device)
        DANdiffusionModel.longTensor = torch.cuda.LongTensor(1,device=device)
        #print(DiffusionModel1.longTensor)
        self.cuda(device=device)

    def setDropout(self,dropout):
        self.nextTime.dropout = nn.Dropout(dropout)
        self.nextInfected.dropout = nn.Dropout(dropout)


    # els a matrix with sums to be performed on dim
    def logsumexp(self, els, dim=-1, val=None, t=-1):
        # els.register_hook(lambda grad: print("els hook " + str(grad)))
        # print("els="+str(els))
        # els.register_hook(lambda grad: print("els hook " + str(grad)))

        if val is None:
            # dels=els.data.numpy()
            # c=np.argmax(dels,dim)
            # prs("els=",els)
            val = els.max(dim, True)[0]  # [range(len(dels)),c]
            # prs("val = ",val)
        val = val.detach()
        # val.register_hook(self.hook("val =" + str(val)))
        # prs("val = ", val)
        # if t==4:
        #    els.register_hook(self.hook("els before clone " + str(t) + "=" + str(els)))
        # els=els.clone()
        # if t == 4:
        #    els.register_hook(self.hook("els after clone "+str(t) + "=" + str(els)))
        ex = torch.exp(els - val)
        # if t == 4:
        #    ex.register_hook(self.hook("ex ="+str(t) + "=" + str(ex)))
        ret = val + torch.log(ex.sum(dim, True))

        return ret

    def getLoss(self,quand,bprev,uprev,lambdaT,w,testFrom,discard,qui,gapT,fromWho=None,t=0):
        #prs(bprev.size(),lambdaT.size(),uprev.size())
        b = F.softmax(bprev, 1)
        c = b * lambdaT * uprev
        c = c.sum(1)
        ninf = self.nextInfected(c)
        ninf = ninf + discard
        cr = self.criterion(ninf, qui)
        mask = ((quand >= testFrom) + (quand<0)).to(cr)
        loss = (cr * mask).sum()

        goodLinks = 0
        nbLinks = 0

        if fromWho is not None:
            with torch.no_grad():
                #prs(fromWho)
                #b=b*lambdaT
                #b = F.softmax(b, 1)
                #prs((quand >= testFrom))

                q=fromWho[(quand >= testFrom),t]

                b=b[(quand >= testFrom)]
                if b.size()[0]>0:
                    nbLinks=q.size()[0]
                    #prs(quand)
                    #prs(testFrom)
                    #prs(b)
                    #prs(q)
                    goodLinks=b[range(b.size()[0]),q].sum().item()

        gp = torch.log(gapT + 1)
        nt = self.nextTime(c)
        ntt = nt + w * gp
        nt = ntt + torch.exp(nt) / w - torch.exp(ntt) / w

        tloss = -(nt * mask.view(-1, 1) * (quand > 0).view(-1, 1).to(cr)).sum()
        #print("tloss",tloss,gapT,nt)
        return (loss,tloss,goodLinks,nbLinks)

    def forward(self,infected,times,testFrom=0,fromWho=None):
        #print("tr",self.training)
        #print(times)
        #packed_input = rnn_utils.pack_padded_sequence(input, lengths.tolist())
        loss=self.floatTensor.new([0])
        tloss = self.floatTensor.new([0])
        goodLinks = 0
        nbLinks = 0

        discard = self.floatTensor.new(np.zeros(self.nbNodes+1))
        discard = discard.unsqueeze(0).expand(infected.size()[0],-1)
        hprev = None #self.floatTensor.new(self.longTensor.new([0] * infected.size()[0] * zz)).view(zz, infected.size()[0], 1, self.nh)
        bprev=None
        uprev = None
        lambdaT=None
        w=self.w(self.longTensor.new([0]))
        quand = times[:, 0]
        quand = quand[quand >= 0]
        lastT = self.floatTensor.new(np.zeros(infected.size()[0])).view(-1, 1)
        for t in range(infected.size()[1]):
            quand = times[:quand.size()[0], t]  # if times <0 => padding
            qui = infected[:quand.size()[0],t]
            qui=torch.where(quand<0,self.longTensor.new([self.nbNodes]),qui)

            if t > 0:
                tt = torch.where(quand < 0, self.floatTensor.new([self.T]), quand)
                gapT = tt.view(-1, 1) - lastT[:quand.size()[0]].view(-1, 1)
                #print("gapT",gapT,self.T-lastT[:quand.size()[0]].view(-1, 1))
                l, lt, rlinks, nblinks = self.getLoss(quand, bprev, uprev, lambdaT, w, testFrom,discard,qui,gapT,fromWho,t)
                goodLinks+=rlinks
                nbLinks+=nblinks
                loss+=l
                tloss+=lt

            quand = quand[quand >= 0]
            if quand.size()[0]==0:
                break
            lastT = quand
            qui = infected[:quand.size()[0], t]
            discard = discard[:quand.size()[0]].contiguous()
            discard[range(quand.size()[0]), qui.data] = -float("inf")
            if t>0:
                hprev=hprev[:quand.size()[0]]
                uprev = uprev[:quand.size()[0]]
                bprev = bprev[:quand.size()[0]]

            if fromWho is not None:
                fromWho=fromWho[:quand.size()[0]]


            input = self.node_embedding(qui)
            h = self.encoder(input)


            if t > 0:
                diffTimes = times[:quand.size()[0],t].view(-1, 1)-times[:quand.size()[0],:t+1]
                #prs("difft",diffTimes.size())
                diffTimes = torch.floor(diffTimes * self.nbSteps / self.T)
                tt = self.longTensor.new(diffTimes.size()[0], diffTimes.size()[1], self.nbSteps + 1).fill_(0).view(-1,
                                                                                                                   self.nbSteps + 1)
                tt[range(tt.size()[0]), diffTimes.view(-1).to(tt)] = 1
                tt = tt.view(diffTimes.size()[0], diffTimes.size()[1], self.nbSteps + 1)
                lambdaT=self.lambdaT(tt.to(input))

                h2=self.h2(hprev)
                h1=self.h1(h)
                h1=h1.view(h1.size()[0],1,h1.size()[1])
                h1=h1.expand(h1.size()[0],h2.size()[1],h1.size()[2])
                a=(h1*h2).sum(2,keepdim=True)
                a=F.softmax(a, 1)
                d=(a*hprev).sum(1)

                g=torch.cat((h,d),1)
                g=self.g(g)
                u=g*h+(1-g)*d
            else:
                u=h
                lambdaT=self.floatTensor.new(np.zeros((quand.size()[0],1,self.nbSteps+1)))
                lambdaT[:,:,0]=1
                lambdaT = self.lambdaT(lambdaT)

            b = self.u(u.view(u.size()[0], 1, -1))
            b = self.b(b)
            if hprev is None:
                hprev = h.view(h.size()[0], 1, -1)
                bprev = b.view(h.size()[0], 1, -1)
                uprev = u.view(u.size()[0], 1, -1)
            else:
                hprev=torch.cat((hprev, h.view(h.size()[0], 1, -1)), 1)
                uprev=torch.cat((uprev, u.view(u.size()[0], 1, -1)), 1)
                bprev = torch.cat((bprev, b.view(b.size()[0], 1, -1)), 1)




            #if bprev is None:
            #    bprev = b.view(h.size()[0], 1, -1)
            #else:
            #    prs("bprev",bprev.size()," ",b.view(b.size()[0], 1, -1).size())
            #    bprev=torch.cat((bprev, b.view(b.size()[0], 1, -1)), 1)

        if quand.size()[0]>0:
            gapT = self.T - lastT[:quand.size()[0]].view(-1, 1)
            l, lt, _,_ = self.getLoss(quand.new(np.ones(quand.size()[0])*(-1)), bprev, uprev, lambdaT, w, testFrom, discard, qui.new([self.nbNodes]).repeat(quand.size()[0]),gapT,None)
            loss += l
            tloss += lt


        #print("loss:"+str(loss))
        #print("tloss:" + str(tloss))

        return (loss,tloss,goodLinks,nbLinks)

    def init_weights(self):
        initrange = 0.1
        self.nextInfected.init_weights(initrange)
        self.nextTime.init_weights(initrange)
        self.node_embedding.weight.data.uniform_(-initrange, initrange)

    # def getParameters(self, **kwargs):
    #     regul = kwargs.get('regul', 0.0)
    #     regul2 = kwargs.get('regul2', 0.0)
    #     params = [{'params': self.node_embedding.parameters(), 'weight_decay': regul},
    #               {'params': self.nextInfected.parameters(), 'weight_decay': regul},
    #               {'params': self.nextTime.parameters(), 'weight_decay': regul},
    #               {'params': self.w.parameters(), 'weight_decay': regul},
    #               {'params': self.h0.parameters(), 'weight_decay': regul},
    #               {'params': self.rnn.parameters(), 'weight_decay': regul}
    #               ]
    #     return params

    def simuFrom(self, infected, times, testFrom, nbSimu):

        #prs("T",self.T)
        #prs("steps",self.nbSteps)

        discard = self.floatTensor.new(np.zeros(self.nbNodes + 1))
        discard = discard.unsqueeze(0).expand(infected.size()[0], -1)
        hprev = None  # self.floatTensor.new(self.longTensor.new([0] * infected.size()[0] * zz)).view(zz, infected.size()[0], 1, self.nh)
        bprev = None
        uprev = None
        inferredTimes=None
        lambdaT = None
        w = self.w(self.longTensor.new([0]))
        quand = times[:, 0]
        quand = quand[quand >= 0]
        lastT = self.floatTensor.new(np.zeros(infected.size()[0])).view(-1, 1)



        stop = self.longTensor.new([self.nbNodes])
        #discard = discard.unsqueeze(0).expand(infected.size()[0], -1)

        output = None
        notstopped = (infected[:, 0] >= 0)
        lh = [None] * infected.size()[0]
        lu = [None] * infected.size()[0]
        lb = [None] * infected.size()[0]
        li = [None] * infected.size()[0]

        nbInf = torch.zeros((infected.size()[0], self.nbNodes)).to(infected)
        nbAdds=torch.zeros((infected.size()[0])).to(infected)
        maxtt=0
        for t in range((infected.size()[1]+1)):
            if t==infected.size()[1]:
                newstopped = notstopped
                inewstopped = self.longTensor.new(range(newstopped.sum()))
                maxtt=t-1
            else:
                quand = times[:, t]  # if times <0 => padding
                sel = quand[notstopped]

                inewstopped=(sel < 0) + (sel > testFrom) #
                #prs("notstopped",notstopped )
                #prs("q0",(quand<0))
                #prs("qt",(quand>testFrom))
                newstopped=(notstopped * ((quand<0) + (quand>testFrom) ))
            #print(str(newstopped)+","+str(t))
            if newstopped.sum().item() > 0:
                    #print(str(newstopped)+" "+str(t))
                    nb=newstopped.sum()
                    inds=self.longTensor.new(range(infected.size()[0]))
                    inds=inds[newstopped]
                    #prs("is",inewstopped)

                    ch=list(hprev[inewstopped].chunk(nb,0))
                    cu=list(uprev[inewstopped].chunk(nb,0))
                    cb = list(bprev[inewstopped].chunk(nb, 0))
                    ci = list(inferredTimes[inewstopped].chunk(nb, 0))
                    #prs(nb,inds)
                    for i in range(nb):
                        lh[inds[i]] = ch[i]
                        lu[inds[i]] = cu[i]
                        lb[inds[i]] = cb[i]
                        li[inds[i]] = ci[i]



            if t==infected.size()[1]:
                break

            sel = (sel >= 0) * (sel <= testFrom)

            notstopped = (notstopped * ((quand >= 0) * (quand <= testFrom)))

            if notstopped.sum().item()==0:
                maxtt=t-1
                break

            quand = quand[notstopped]


            if t > 0:
                hprev = hprev[sel]
                uprev = uprev[sel]
                bprev = bprev[sel]
                inferredTimes = inferredTimes[sel]

            # prs("quand",quand)
            nbAdds[notstopped] += 1
            qui = infected[notstopped, t]
            discard[notstopped, qui] += -float("inf")
            #prs("nots",nbInf,notstopped,qui)
            nbInf[notstopped,qui]+=nbSimu

            lastT[notstopped] = quand.view(-1, 1)
            input = self.node_embedding(qui)
            h = self.encoder(input)

            if t > 0:
                h2 = self.h2(hprev)
                h1 = self.h1(h)
                h1 = h1.view(h1.size()[0], 1, h1.size()[1])
                h1 = h1.expand(h1.size()[0], h2.size()[1], h1.size()[2])
                a = (h1 * h2).sum(2, keepdim=True)
                a = F.softmax(a, 1)
                d = (a * hprev).sum(1)

                g = torch.cat((h, d), 1)
                g = self.g(g)
                u = g * h + (1 - g) * d
            else:
                u = h


            b = self.u(u.view(u.size()[0], 1, -1))
            b = self.b(b)
            if hprev is None:
                hprev = h.view(h.size()[0], 1, -1)
                bprev = b.view(b.size()[0], 1, -1)
                uprev = u.view(u.size()[0], 1, -1)
                inferredTimes=quand.view(quand.size()[0],1)
            else:
                hprev = torch.cat((hprev, h.view(h.size()[0], 1, -1)), 1)
                uprev = torch.cat((uprev, u.view(u.size()[0], 1, -1)), 1)
                bprev = torch.cat((bprev, b.view(b.size()[0], 1, -1)), 1)
                inferredTimes = torch.cat((inferredTimes, quand.view(quand.size()[0], 1)), 1)


        #prs("lo",lo)

        pb = tqdm(list(range(nbSimu)), dynamic_ncols=True)
        logsoft = torch.nn.LogSoftmax(dim=-1)
        sdisc=discard
        #prs("discard",discard)
        slastT=lastT
        snbAdds=nbAdds
        #prs("lastT before",lastT)
        for isimu in pb:
            hprev=None #torch.cat(lh,1)
            uprev=None #torch.cat(lo,0)
            bprev=None
            inferredTimes=None
            step=0
            #alive=self.longTensor.new(range(infected.size()[0]))
            now=maxtt
            discard=sdisc.clone()
            nbAdds=snbAdds.clone()
            lastT=slastT.clone()
            notstarted=(self.longTensor.new([1]*infected.size()[0])>0)
            encours=None
            while True:
                step += 1
                if notstarted.sum()==0 and encours.size()[0]==0:
                    break
                if notstarted.sum()>0:
                    m=torch.min(nbAdds)
                    #prs(nbAdds,m)
                    m=(nbAdds==m)
                    nbAdds[m]+=1

                    aajouter = notstarted * m
                    #prs("aajouter ", step, aajouter)
                    #if step==1:
                    #    while(courts.sum()==0):
                    #        now-=1
                    #        courts = times[:, now]
                    ##        courts = notstarted * ((courts > testFrom) + (courts < 0))
                    #        prs("courts ",now,courts,times[:,now])
                    if aajouter.sum()>0:

                        notstarted[aajouter]=0
                        iadd = self.longTensor.new(range(infected.size()[0]))[aajouter]
                        if encours is None:
                            encours = iadd
                        else:
                            encours=torch.cat((iadd,encours),0)
                        lic = iadd.tolist()
                        #prs("lic",lic)
                        lll = [lh[i] for i in lic]
                        #prs("lll", lll)
                        if hprev is not None: lll.append(hprev)
                        #prs("lll",lll,lic,encours)
                        hprev = torch.cat(lll,0)
                        lll = [lb[i] for i in lic]
                        if bprev is not None: lll.append(bprev)
                        bprev = torch.cat(lll, 0)
                        lll = [lu[i] for i in lic]
                        if uprev is not None: lll.append(uprev)
                        uprev = torch.cat(lll, 0)
                        lll = [li[i] for i in lic]
                        if inferredTimes is not None: lll.append(inferredTimes)
                        inferredTimes = torch.cat(lll, 0)

                    #now-=1

                #prs(lastT[encours])
                diffTimes = lastT[encours] - inferredTimes
                #prs(diffTimes)
                diffTimes = torch.floor(diffTimes * self.nbSteps / self.T)
                diffTimes=torch.where(diffTimes>(self.nbSteps),self.floatTensor.new([self.nbSteps]),diffTimes)
                tt = self.longTensor.new(diffTimes.size()[0], diffTimes.size()[1], self.nbSteps + 1).fill_(0).view(-1,self.nbSteps + 1)
                #prs("tt",tt)
                #prs(diffTimes)
                tt[range(tt.size()[0]), diffTimes.view(-1).to(tt)] = 1
                tt = tt.view(diffTimes.size()[0], diffTimes.size()[1], self.nbSteps + 1)
                #prs(tt)
                lambdaT = self.lambdaT(tt.to(input))

                b = F.softmax(bprev, 1)
                c = b * lambdaT * uprev
                c = c.sum(1)
                ninf = self.nextInfected(c)
                ninf = ninf + discard[encours]
                ninf = logsoft(ninf)
                # prs("ninf",ninf)
                ca = torch.distributions.Categorical(logits=ninf)
                next = ca.sample()
                #prs("next",next,next.size())
                #prs(bprev.size())
                #prs(inferredTimes.size())
                stopped = (next == self.nbNodes)

                nbAdds[encours[stopped]]=self.T*10
                notstopped=(next<self.nbNodes)
                encours=encours[notstopped]
                if notstopped.sum()==0:
                    bprev=None
                    uprev=None
                    hprev=None
                    inferredTimes=None
                    continue

                #alive=alive[notstopped]
                c=c[notstopped]
                qui=next[notstopped]
                #prs("nbInf before", nbInf)
                nbInf[encours,qui]+=1
                #prs("nbInf after",nbInf)
                nt = self.nextTime(c)


                r = (torch.rand(c.size()[0],10)*100).to(self.floatTensor)
                r=torch.where(lastT[encours]<testFrom,r+testFrom-lastT[encours],r)
                #prs("r",r)
                #r+=testFrom-lastT[encours]
                gp = torch.log(r + 1)

                ntt = nt + w * gp
                # print("ntt" + str(ntt))
                # print("w" + str(w))

                nt = ntt + torch.exp(nt) / w - torch.exp(ntt) / w
                # prs("add",nt,ntt,gp)
                # print("nt" + str(nt.size()))
                ct = torch.distributions.Categorical(logits=nt)
                ct=ct.sample()

                t=r[range(r.size()[0]),ct]
                #prs("t",t)
                #prs(lastT, encours, t)
                t=torch.floor(lastT[encours]+t.view(-1,1))
                #prs("time", t)
                lastT[encours]=t
                #inferredTimes=torch.cat((inferredTimes[encours],t),1)
                #lastT[notstopped]=lastT[notstopped]+t[notstopped,ct]
                #print("gp"+str(gp))
                #print("notstopped"+str(notstopped))
                #gapT = torch.log(t+1).view(-1,1) # - lastT[notstopped].view(-1, 1)

                hprev = hprev[notstopped].contiguous()
                uprev = uprev[notstopped].contiguous()
                bprev = bprev[notstopped].contiguous()
                inferredTimes=inferredTimes[notstopped].contiguous()

                input = self.node_embedding(qui)
                h = self.encoder(input)

                h2 = self.h2(hprev)
                h1 = self.h1(h)
                h1 = h1.view(h1.size()[0], 1, h1.size()[1])
                h1 = h1.expand(h1.size()[0], h2.size()[1], h1.size()[2])
                a = (h1 * h2).sum(2, keepdim=True)
                a = F.softmax(a, 1)
                d = (a * hprev).sum(1)

                g = torch.cat((h, d), 1)
                g = self.g(g)
                u = g * h + (1 - g) * d

                b = self.u(u.view(u.size()[0], 1, -1))
                b = self.b(b)
                hprev = torch.cat((hprev, h.view(h.size()[0], 1, -1)), 1)
                uprev = torch.cat((uprev, u.view(u.size()[0], 1, -1)), 1)
                bprev = torch.cat((bprev, b.view(b.size()[0], 1, -1)), 1)
                inferredTimes = torch.cat((inferredTimes,lastT[encours].view(-1,1)),1)
                #discard = discard[notstopped].contiguous()
                # prs("qui", qui)
                # prs("discard", discard)

                discard[encours, qui] = -float("inf")
                # prs("discard2", discard)

                # a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
                # b = torch.LongTensor([0, 2])
                # a[range(2),b]=-float("inf")
                # prs("a",a)



        rInf = nbInf.to(self.floatTensor) / nbSimu
        #prs("rinf",rInf[0])
        rFin = rInf.sum()
        gt = torch.zeros((infected.size()[0], self.nbNodes)).to(infected)
        c = torch.where(infected >= 0, infected, self.longTensor.new([0]))
        gt.scatter_(1, c, torch.ones(gt.size()).to(gt))
        #prs("gt",gt)
        l = torch.where(gt > 0, rInf, 1 - rInf)

        # torch.set_printoptions(threshold=5000)
        #prs("rInf", rInf[0])
        #prs("gt", gt[0].to(self.floatTensor))
        # prs("l", l[0])
        l = torch.log(l + 1E-10)
        #prs("l", l)
        # prs("l",l.sum())
        rInf = l.sum()

        return(rInf,rFin)




#
# Training settings
parser = argparse.ArgumentParser(description='PyTorch variationalCTIC')
parser.add_argument('--batchSize', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
#parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='F',
                    help='learning rate (default: 0.01)')
#parser.add_argument('--momentum', type=float, default=0.5, metavar='F', help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda-device', '-c', type=int, default=-1, metavar='N',
                    help='specifies CUDA device (default no cuda device)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--inTrainFile', '-itr', type=str, metavar='S',
                    help='input train file',required=True)
parser.add_argument('--noTrain', type=bool, metavar='B',
                    help='ignore train on train data',default=False)
parser.add_argument('--inTestFile', '-ite', type=str, metavar='S',
                    help='input test file',default="")
parser.add_argument('--outDir', '-o', type=str, metavar='S',
                    help='out directory',required=True)
#parser.add_argument('--trainRate', '-tr', type=float, metavar='F', default=1.0,
#                    help='rate of episodes from input to use for training')
#parser.add_argument('--valRate', '-vr', type=float, metavar='F', default=0.0,
#                    help='rate of episodes from input to use for validation')
parser.add_argument('--nh', type=int, metavar='N', default=64,
                    help='number of dimensions of the states')
parser.add_argument('--nw', type=int, metavar='N', default=32,
                    help='number of dimensions of the node embeddings')
parser.add_argument('--decay', '-d', type=float, metavar='F', default=0.0,
                    help='weight decay')
parser.add_argument('--dropout', '-dr', type=float, metavar='F', default=0.0,
                    help='dropout rate in train')
parser.add_argument('--nlayers', '-nl', type=int, metavar='N', default=3,
                    help='number of layers of the rnn')
parser.add_argument('--nbSteps', type=int, metavar='N', default=40,
                    help='number of time steps for time representation')
parser.add_argument('--fromFile', '-fr', type=str, metavar='S', default="",
                    help='model file to load (default: empty)')
parser.add_argument('--freqTest', type=int, metavar='N', default=100,
                    help='frequence for tests (default: 10)')
parser.add_argument('--fromTests', nargs='+', type=int, default=[0],
                    help='fromTests (always prepend with 0)')
parser.add_argument('--nbsimu', '-ns', type=int, default=10,
                    help='number of simulations to perform at each test (default 10)')
parser.add_argument('--computellLinks', '-cl', action='store_true', default=False,
                    help='compute likelihood of known links or not. only possible if fromWho known in episodes (default False)')



args = parser.parse_args()
args.commit = subprocess.check_output(['git', 'show-ref']).decode('utf-8')

print("device="+str(args.cuda_device))

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if(args.cuda_device>=0):
    #print("cuda")
    cudnn.benchmark = True
    torch.cuda.device(args.cuda_device)
    torch.cuda.manual_seed(args.seed)



episodes, nodes, T = Episode.loadEpisodesFromFile(args.inTrainFile)
trainBins=Episode.makeBins(episodes, nodes, batchSize=args.batchSize, cuda=args.cuda_device)
if args.computellLinks:
    trainLinks=Episode.makeBinsOfLinks(trainBins,nodes)

testBins=None
if len(args.inTestFile)>0:
    episodesT, nodesT, TT = Episode.loadEpisodesFromFile(args.inTestFile,nodes)
    testBins = Episode.makeBins(episodesT, nodes, batchSize=args.batchSize, cuda=args.cuda_device)
    if args.computellLinks:
        testLinks = Episode.makeBinsOfLinks(testBins,nodes)

args.nbNodes=len(nodes)
args.T=max(T,TT)

diffMod=None
if len(args.fromFile)>0:
    diffMod = torch.load(args.fromFile,map_location=lambda storage, loc: storage.cuda(args.cuda_device))
    diffMod.setDropout(args.dropout)
    #diffMod.cpu()
else:

    diffMod = DANdiffusionModel(args.T, args.nbNodes, args.nh, args.nw,  nlayers=args.nlayers, dropout=args.dropout, nbSteps=args.nbSteps)  # ./arti_episodes_train.txt")
#


#exit(0)


if(args.cuda_device>=0):
    diffMod.setcuda(args.cuda_device)


args.nbTrainEpisodes=0
if(len(trainBins)>0):
    args.nbTrainEpisodes =(len(trainBins)-1)*args.batchSize+trainBins[-1][1].size()[0]

args.nbTestEpisodes=0
if(testBins is not None):
    args.nbTestEpisodes=(len(testBins)-1)*args.batchSize+testBins[-1][1].size()[0]
args.T=T




optimizer = torch.optim.Adam(diffMod.parameters(), weight_decay=args.decay, lr=args.lr)



print("Logs in: " + args.outDir)
iter_logger = LogCSV(args.outDir, epoch_size=len(trainBins), filename='iter_log.csv')
epoch_logger = LogCSV(args.outDir, filename='epoch_log.csv', wipe=False)
iter_loggert = None
if args.nbTestEpisodes>0:
    iter_loggert = LogCSV(args.outDir, epoch_size=len(testBins), filename='iter_logt.csv', wipe=False)
# xp info
args.model = str(diffMod).split("(")[0]
args.optim = str(optimizer)
with open(os.path.join(args.outDir, 'info.json'), 'w') as info:
    json.dump(vars(args), info, sort_keys=True, indent=4)
with open(os.path.join(args.outDir, 'info_long.json'), 'w') as info:
    args.model = str(diffMod)
    args.nodes = str(nodes)
    json.dump(vars(args), info, sort_keys=True, indent=4)

torch.save(diffMod, os.path.join(args.outDir, "./diffMod_start"))


best=None
bestChanged=False
diffMod.verbose=0
for epoch in range(args.epochs):
    diffMod.train(True)
    #print(str(diffMod.rnn.dropout))
    #print(str(diffMod.nextTime.dropout))
    #print(str(diffMod.nextTime.training))
    print('Epoch ' + str(epoch))
    epoch_logger.new_iteration()
    if not args.noTrain:
        pb = tqdm(trainBins,dynamic_ncols=True)
        bin=0
        aLinks=0
        anbLinks=0
        for batch in pb:
            diffMod.zero_grad()
            iter_logger.new_iteration()
            eps,infections,times=batch
            tt = None
            if args.computellLinks:
                tt = trainLinks[bin]
            #print("ici")
            #print(str(infections))
            lx, lt,rLinks,nbLinks=diffMod(infections,times,fromWho=tt)
            #print("la")
            aLinks+=rLinks
            anbLinks+=nbLinks
            loss = lx+lt
            loss=loss/args.nbTrainEpisodes
            iter_logger.log("lx", lx.detach().cpu().numpy().item(0))
            iter_logger.log("lt", lt.detach().cpu().numpy().item(0))
            iter_logger.log('loss', loss.detach().cpu().numpy().item(0))


            s = times.new([0])
            pp = 0

            with torch.no_grad():
                for i in diffMod.parameters():
                    s = s + torch.pow(i, 2).sum()
                    nn = 1
                    for x in list(i.size()):
                        nn = nn * x
                    pp += nn
                iter_logger.log("s", s.detach().cpu().numpy().item(0))

            pb.set_postfix(loss=loss.detach().cpu().numpy().item(0))

            if (loss.data!=loss.data).sum()>0:
                print("pb recompute in verbose before exiting")
                diffMod.verbose=2
                lx,lt,_,_ = diffMod(infections, times)
                print("save the problematic model ")
                torch.save(diffMod, os.path.join(args.outDir, "./diffMod_nan"))
                print("done, exit")
                exit(0)
            else:

                #print("step")
                loss.backward()

                optimizer.step()



            bin+=1


        epoch_logger.log('epoch', epoch)
        epoch_logger.log('NLL', iter_logger.get_epoch('loss'))
        #epoch_logger.log('lx', iter_logger.get_epoch('lx'))
        #epoch_logger.log('lt', iter_logger.get_epoch('lt'))
        #epoch_logger.log('s', iter_logger.get_epoch('s'))
        if anbLinks>0:
            epoch_logger.log('INF', aLinks/anbLinks)
            #epoch_logger.log('nbLinks', anbLinks)

        #epoch_logger.log('g', iter_logger.get_epoch('g'))

        print('\n train', epoch_logger.get('NLL'))#,epoch_logger.get('lx'),epoch_logger.get('lt'),epoch_logger.get('s')) #,epoch_logger.get('g'))
        if anbLinks > 0:
            print("INF="+str(aLinks/anbLinks)+" over "+str(anbLinks))
    if args.nbTestEpisodes > 0 and epoch % args.freqTest == 0 and epoch>=0:

        diffMod.eval()
        for fromTest in args.fromTests:
            pb = tqdm(testBins, dynamic_ncols=True)
            aLinks = 0
            anbLinks = 0
            bin = 0
            #print(str(diffMod.rnn.dropout))
            #print(str(diffMod.nextTime.dropout))
            aFin = None
            aInf = None
            #print(str(diffMod.nextTime.training))
            for batch in pb:
                with torch.no_grad():
                    #diffMod.zero_grad()
                    iter_loggert.new_iteration()
                    eps,infections, times= batch
                    if args.computellLinks:
                        tt = testLinks[bin]
                    lx, lt, rLinks, nbLinks = diffMod(infections, times,fromTest,fromWho=tt)
                    aLinks += rLinks
                    anbLinks += nbLinks

                    loss = lx + lt
                    loss = loss / args.nbTestEpisodes
                    iter_loggert.log("epoch", epoch)
                    iter_loggert.log("fromTest", fromTest)
                    iter_loggert.log("it", bin + 1)
                    iter_loggert.log("lxt", lx.detach().cpu().numpy().item(0))
                    iter_loggert.log("ltt", lt.detach().cpu().numpy().item(0))
                    iter_loggert.log('losst', loss.detach().cpu().numpy().item(0))

                    if args.nbsimu > 0:
                        rInf, rFin = diffMod.simuFrom(infections, times, fromTest, args.nbsimu)
                        if aInf is None:
                            aInf=rInf/ args.nbTestEpisodes
                            aFin=rFin/ args.nbTestEpisodes

                        else:
                            aInf += rInf / args.nbTestEpisodes
                            aFin += rFin / args.nbTestEpisodes
                        #print(aInf)



                    s = times.new([0])
                    pp = 0
                    for i in diffMod.parameters():
                        s = s + torch.pow(i, 2).sum()
                        nn = 1
                        for x in list(i.size()):
                            nn = nn * x
                        pp += nn
                    iter_loggert.log("st", s.detach().cpu().numpy().item(0))

                    pb.set_postfix(loss=loss.detach().cpu().numpy().item(0))

                    if (loss.data != loss.data).sum() > 0:
                        print(lx,lt)
                        diffMod.verbose = 2
                        lx,lt = diffMod(infections, times, fromTest)
                        torch.save(diffMod, os.path.join(args.outDir, "./diffMod_nan"))
                        exit(0)

                    ee = loss.sum().detach().cpu().numpy().item(0)
                    if best is None or ee < best:
                        best = ee
                        bestChanged = True

                bin += 1

            #epoch_logger.log('epoch', epoch)
            epoch_logger.log('NLL_'+str(fromTest), iter_loggert.get_epoch('losst'))
            #epoch_logger.log('lxt_'+str(fromTest), iter_loggert.get_epoch('lxt'))
            #epoch_logger.log('ltt_'+str(fromTest), iter_loggert.get_epoch('ltt'))
            #epoch_logger.log('st_'+str(fromTest), iter_loggert.get_epoch('st'))
            if args.nbsimu > 0:
                epoch_logger.log('CE_' + str(fromTest), aInf.item())
                #epoch_logger.log('rfin_' + str(fromTest), aFin.item())
                print("CE "+str(aInf.item())) #+" nbInf "+str(aFin))
            if anbLinks > 0:
                epoch_logger.log('INF', aLinks / anbLinks)
                #epoch_logger.log('nbLinks', anbLinks)
            # epoch_logger.log('g', iter_logger.get_epoch('g'))

            print('\n test', epoch_logger.get('NLL_'+str(fromTest))) #, epoch_logger.get('lxt_'+str(fromTest)), epoch_logger.get('ltt_'+str(fromTest)),
                  #epoch_logger.get('st_'+str(fromTest)))  # ,epoch_logger.get('g'))
            if anbLinks > 0:
                print("INF=" + str(aLinks / anbLinks) + " over " + str(anbLinks))

    if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
        iter_logger.flush()
        if args.nbTestEpisodes > 0:
            iter_loggert.flush()
        epoch_logger.flush()
        torch.save(diffMod, os.path.join(args.outDir, "./diffMod_last"))
        with open(os.path.join(args.outDir, "./params_last.txt"), "w") as f:
            for name, param in diffMod.named_parameters():
                if param.requires_grad:
                    f.write(str(name)+" : "+str(param.data)+"\n")

        if bestChanged:
            shutil.copy(os.path.join(args.outDir, "diffMod_last"),os.path.join(args.outDir, "diffMod_best"))
            shutil.copy(os.path.join(args.outDir, "params_last.txt"), os.path.join(args.outDir, "params_best.txt"))
            bestChanged=False

#python DAN.py -o xp/DAN/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt -c 0 --fromTests 0 2 5 10 -cl -ns 0