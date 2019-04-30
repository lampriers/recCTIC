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

import scipy.misc

#from torchvision import datasets, transforms
#from torch.autograd import Variable
import torch.autograd.profiler as profiler

import sys
from sortedcollections.recipes import OrderedSet
from collections import OrderedDict
#from blaze.expr.math import isnan
sys.path.append("utils")
#from kbhit import KBHit
from utils import *

from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, nh, outSize=None, nlayers=1, dropout=0., **kwargs):
        super(NN, self).__init__()
        if outSize is None:
            outSize = nh
        self.dropout=None
        if dropout>0:
            self.dropout = nn.Dropout(dropout)
        self.activ = kwargs.get('activ', F.leaky_relu)
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
            #l.bias.data.fill_(0)
            #l.weight.data.uniform_(-initrange, initrange)
            torch.nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        #print("x"+str(x))
        #print("l0" + str(self.layers[0]))
        x = self.layers[0](x)
        #print("x1" + str(x))

        if len(self.layers) > 1:
            if self.dropout is not None:
                x = self.dropout(x)
                #print("x1bis" + str(x))
        for i in range(1, len(self.layers)):
            x = self.activ(x)
            #print("x",i,str(x))
            x = self.layers[i](x)
            #print("x2",i, str(x))
            if i < len(self.layers) - 1:
                if self.dropout is not None:
                    x = self.dropout(x)
                    #print("x2bis", i, str(x))

        if self.activFinal is not None:
            x = self.activFinal(x)
        #print("xx" + str(x))

        return x



class rnnDiffusionModel(nn.Module):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)
    verbose = 0
    def __init__(self, nh, nwe, T, nbNodes, dropout=0.0,rnn_type='GRU',nlayers=1,glayers=3,enclayers=1,cyanRNN=False,coverage=False):
        super(rnnDiffusionModel, self).__init__()
        self.nh=nh
        self.nbNodes=nbNodes
        self.node_embedding = nn.Embedding(nbNodes+1, nwe, padding_idx=-1)
        #print("nlayers "+str(nlayers))
        self.rnn=getattr(nn, rnn_type)(nh, nh, nlayers, dropout=dropout)
        nnh=nh
        self.T=T
        self.encoder=NN(nwe+2,nh,enclayers, dropout=dropout)
        if cyanRNN:
            nnh=nh*3
        self.nextInfected=NN(nnh, nbNodes + 1, glayers, dropout=dropout)
        self.nextTime = NN(nnh, 1, 1, dropout=dropout, activFinal=F.sigmoid)

        if cyanRNN:
            self.sigT = NN(nnh, nh, 1, activFinal=F.sigmoid)
            if coverage:
                self.a=NN(nh * 2 + 1, 1, 2, activ=F.tanh)
                self.sigV = NN(nh * 2 + 2, 1, 1, activFinal=F.sigmoid)
            else:
                self.a=NN(nh * 2, 1, 2, activ=F.tanh)

        self.nlayers=nlayers
        self.glayers = glayers
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.dropout = nn.Dropout(dropout)

        self.h0=nn.Embedding(nlayers,nh)
        self.w = nn.Embedding(1, 1)

        self.init_weights()
        self.cyanRNN=cyanRNN
        self.coverage=coverage

    def setcuda(self,device):
        rnnDiffusionModel.floatTensor = torch.cuda.FloatTensor(1,device=device)
        rnnDiffusionModel.longTensor = torch.cuda.LongTensor(1,device=device)
        #print(DiffusionModel1.longTensor)
        self.cuda(device=device)

    def setDropout(self,dropout):
        self.nextTime.dropout = nn.Dropout(dropout)
        self.nextInfected.dropout = nn.Dropout(dropout)
        self.rnn.dropout = dropout

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

    def forward(self,infected,times,testFrom=0,fromWho=None):
        #print("tr",self.training)
        #packed_input = rnn_utils.pack_padded_sequence(input, lengths.tolist())
        loss=self.floatTensor.new([0])
        tloss = self.floatTensor.new([0])
        hidden=self.dropout(self.h0(self.longTensor.new(range(self.nlayers))))
        hidden = hidden.unsqueeze(1)
        hidden=hidden.expand(self.nlayers,infected.size()[0],self.nh)
        lastT=self.floatTensor.new(np.zeros(infected.size()[0])).view(-1,1)
        stop=self.longTensor.new([self.nbNodes])
        discard = self.floatTensor.new(np.zeros(self.nbNodes+1))
        discard = discard.unsqueeze(0).expand(infected.size()[0],-1)
        if self.cyanRNN:
            #outHistory=hidden[-1].view(-1,1,self.nh)
            tk=torch.zeros((hidden[-1].size())).to(hidden)
            if self.coverage:
                vHistory = torch.zeros(hidden[-1].size()[0],1,1).to(hidden)
                aHistory = torch.zeros(hidden[-1].size()[0],1,1).to(hidden)
            #ak = torch.ones((hidden[-1].size()[0],outHistory.size()[0])).to(hidden)

        goodLinks = 0
        nbLinks = 0

        output=None
        w=self.w(self.longTensor.new([0]))
        for t in range(infected.size()[1]):
            quand = times[:, t]  # if times <0 => padding
            quand = quand[quand >= 0]


            #prs("quand",quand)

            qui = infected[:quand.size()[0],t]

            #prs("qui", qui)
            #prs("discard",discard)
            #nt=notinf[t,]
            #print("qui:"+str(qui))
            gapT = quand.view(-1, 1) - lastT[:quand.size()[0]].view(-1, 1)
            gp=torch.log(gapT+1)
            if t>0:




                #print("xout"+str(xout))
                ninf=self.nextInfected(xout[:quand.size()[0]])

                ninf=ninf+discard[:quand.size()[0]]

                #print("ninf",ninf)
                #prs("qui",qui)
                #prs("inf",ninf)
                cr=self.criterion(ninf, qui)


                #prs("cr",cr)
                mask=(quand>=testFrom).to(cr)
                #print("sm"+str(mask))
                #print("sc" + str(cr))
                #print("m" + str((cr*mask)))

                loss += (cr*mask).sum()
                #loss += cr.sum()
                #prs("loss",loss)
                nt = self.nextTime(xout[:quand.size()[0]])
                #print("gapT" + str(gapT))
                #print("nt" + str(nt))
                ntt = nt + w * gp
                #print("ntt" + str(ntt))
                #print("w" + str(w))

                nt = ntt + torch.exp(nt)/w - torch.exp(ntt)/w
                #prs("add",nt,ntt,gp)
                #print("nt" + str(nt.size()))
                tloss -=(nt*mask.view(-1,1)).sum()
                #print("m2" + str((nt * mask)))

                #tloss -= nt.sum()

                if self.cyanRNN and fromWho is not None:
                    #nmask=(times[:,t] >= testFrom)
                    q = fromWho[(times[:,t] >= testFrom), t]
                    #print("ak",ak)

                    ak=ak[:quand.size()[0]][quand>=testFrom]

                    if ak.size()[0] > 0:
                        nbLinks += q.size()[0]
                        goodLinks+=ak[range(ak.size()[0]),q].sum().item()
                        #print("akq", q, ak[range(ak.size()[0]),q])

                # for stopped episodes
                if quand.size()[0]<output.size()[1]:
                    xout = xout[quand.size()[0]:]
                    #print("xoutnotinf"+str(xout))
                    ninf = self.nextInfected(xout)
                    ninf = ninf + discard[quand.size()[0]:]
                    loss += self.criterion(ninf, stop.repeat(xout.size()[0])).sum()


            hidden=hidden[:,:quand.size()[0],:].contiguous()
            input = self.node_embedding(qui)
            #prs("qui:", qui)
            #prs("input0:", input)
            input=torch.cat((input,torch.log(gapT+1),gapT),-1)
            #prs("input1:", input)

            input=self.encoder(input)
            input=input.unsqueeze(0)
            #prs("input:",input)
            #prs("hidden:", hidden)
            output, hidden = self.rnn(input, hidden)
            if self.cyanRNN:
                if t>0:

                    outHistory = torch.cat((outHistory[:quand.size()[0]],output.view(-1,1,self.nh)),1)
                else:
                    outHistory = output.view(-1,1,self.nh)
                    #prs("output:", output)
            #prs("hidden:", hidden)
            output=self.dropout(output)
            hidden=self.dropout(hidden)
            lastT = quand
            discard = discard[:quand.size()[0]].contiguous()
            #prs("qui", qui)
            #prs("discard", discard)

            discard[range(quand.size()[0]), qui.data] = -float("inf")
            #prs("discard2", discard)

            #a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
            #b = torch.LongTensor([0, 2])
            #a[range(2),b]=-float("inf")
            #prs("a",a)

            if self.cyanRNN:

                # outHistory = outHistory[:input.size()[1]]
                if self.coverage:
                    aHistory = aHistory[:quand.size()[0]]
                    vHistory = vHistory[:quand.size()[0]]
                tk = tk[:quand.size()[0]]
                input = input[0, :quand.size()[0]]
                #print("input", str(input))
                #print("quand", str(quand))
                # TODO length dep

                #print("outhistory" + str(outHistory))
                #print("tk" + str(tk))
                ttk = tk.view(-1, 1, self.nh).expand(-1, outHistory.size()[1], -1)
                #print("ttk" + str(ttk))
                outh = torch.cat((outHistory, ttk), -1)
                #print("outh" + str(outh))
                if self.coverage:
                    outhav = torch.cat((outh, vHistory, aHistory), -1)
                    vk = self.sigV(outhav)
                    outh = torch.cat((outh, vk), -1)

                ek = self.a(outh).view(-1, outHistory.size()[1])
                #print("ek" + str(ek))
                ak = F.softmax(ek, 1)


                #print("ak" + str(ak))

                sk = (ak.view(-1, outHistory.size()[1], 1).expand(-1, -1, self.nh) * outHistory).sum(1)
                outh = torch.cat((input.view(outHistory.size()[0], -1), tk, sk), -1)
                tk = self.sigT(outh)
                if self.coverage:
                    aHistory = torch.cat(
                        (ak.view(-1, outHistory.size()[1], 1), torch.zeros(outHistory.size()[0], 1, 1).to(outHistory)),
                        1)
                    vHistory = torch.cat(
                        (vk.view(-1, outHistory.size()[1], 1), torch.zeros(outHistory.size()[0], 1, 1).to(outHistory)),
                        1)
                xout = torch.cat((input.view(outHistory.size()[0], -1), tk, sk), -1)

            else:
                xout = output[0, :quand.size()[0]]


        ninf = self.nextInfected(xout)
        loss += self.criterion(ninf, stop.repeat(ninf.size()[0])).sum()

        #print("loss:"+str(loss))
        #print("tloss:" + str(tloss))

        return (loss,tloss,goodLinks,nbLinks)

    def init_weights(self):

        initrange = 0.1
        self.nextInfected.init_weights(initrange)
        self.nextTime.init_weights(initrange)
        #self.node_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder.init_weights(initrange)
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

    def simuFrom(self, infected, times, testFrom, nbSimu, maxT):


        hidden = self.dropout(self.h0(self.longTensor.new(range(self.nlayers))))
        hidden = hidden.unsqueeze(1)
        hidden = hidden.expand(self.nlayers, infected.size()[0], self.nh)
        lastT = self.floatTensor.new(np.zeros(infected.size()[0])).view(-1, 1)
        stop = self.longTensor.new([self.nbNodes])
        discard = self.floatTensor.new(infected.size()[0],self.nbNodes + 1).fill_(0)
        #discard = discard.unsqueeze(0).expand(infected.size()[0], -1)
        if self.cyanRNN:
            # outHistory=hidden[-1].view(-1,1,self.nh)
            tk = torch.zeros((hidden[-1].size())).to(hidden)
            if self.coverage:
                vHistory = torch.zeros(hidden[-1].size()[0], 1, 1).to(hidden)
                aHistory = torch.zeros(hidden[-1].size()[0], 1, 1).to(hidden)
            # ak = torch.ones((hidden[-1].size()[0],outHistory.size()[0])).to(hidden)

        output = None
        w = self.w(self.longTensor.new([0]))
        notstopped = (infected[:, 0] >= 0)
        louth=[None]*infected.size()[0]
        ltk=[None]*infected.size()[0]
        lvh = [None] * infected.size()[0]
        lah = [None] * infected.size()[0]
        lh=[None] * infected.size()[0]
        lo=[None] * infected.size()[0]
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

                    ch=list(hidden[:,inewstopped].chunk(nb,1))
                    co=list(xout[inewstopped].chunk(nb,0))
                    if self.cyanRNN:
                        ctk=list(tk[inewstopped].chunk(nb,0))
                        couth = list(outHistory[inewstopped].chunk(nb,0))
                        if self.coverage:
                            cvh=list(vHistory[inewstopped].chunk(nb,0))
                            cah=list(aHistory[inewstopped].chunk(nb,0))
                    #prs(nb,inds)
                    for i in range(nb):
                        lh[inds[i]] = ch[i]
                        lo[inds[i]] = co[i]

                        if self.cyanRNN:
                            ltk[inds[i]] = ctk[i]
                            louth[inds[i]]=couth[i]
                            if self.coverage:
                                lvh[inds[i]]=cvh[i]
                                lah[inds[i]] = cah[i]

            if t==infected.size()[1]:
                break

            sel = (sel >= 0) * (sel <= testFrom)

            notstopped = (notstopped * ((quand >= 0) * (quand <= testFrom)))

            if notstopped.sum().item()==0:
                maxtt=t-1
                break

            quand = quand[notstopped]

            # prs("quand",quand)
            nbAdds[notstopped] += 1
            qui = infected[notstopped, t]
            #prs("nots",nbInf,notstopped,qui)
            nbInf[notstopped,qui]+=nbSimu
            #prs(nbInf)

            # prs("qui", qui)
            # prs("discard",discard)
            # nt=notinf[t,]
            # print("qui:"+str(qui))
            gapT = quand.view(-1, 1) - lastT[notstopped].view(-1, 1)

            hidden = hidden[:, sel, :].contiguous()
            input = self.node_embedding(qui)
            input = torch.cat((input, torch.log(gapT + 1), gapT), -1)
            #print(str(input))
            input = self.encoder(input)
            input = input.unsqueeze(0)
            # prs("input:",input)
            # prs("hidden:", hidden)
            output, hidden = self.rnn(input, hidden)
            if self.cyanRNN:
                if t > 0:
                    #print("outh1" + str(outHistory[sel]))
                    #print("out1" + str(output.view(-1, 1, self.nh)))
                    outHistory = torch.cat((outHistory[sel], output.view(-1, 1, self.nh)), 1)
                else:
                    outHistory = output.view(-1, 1, self.nh)
                    # prs("output:", output)
            # prs("hidden:", hidden)
            output = self.dropout(output)
            hidden = self.dropout(hidden)
            #print(str(lastT)+","+str(quand))
            lastT[notstopped] = quand.view(-1,1)
            #prs("lastT",lastT)
            #discard = discard[:quand.size()[0]].contiguous()
            #prs("qui", qui)
            #prs("discard", discard)
            #prs("notstopped",notstopped)
            #kk=self.floatTensor.new([-float("inf")]*notstopped.sum().item())
            #prs("kk",kk)
            discard[notstopped, qui] += -float("inf")
            #prs("discard2", discard)

            # a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
            # b = torch.LongTensor([0, 2])
            # a[range(2),b]=-float("inf")
            # prs("a",a)

            if self.cyanRNN:

                # outHistory = outHistory[:input.size()[1]]
                if self.coverage:
                    aHistory = aHistory[sel]
                    vHistory = vHistory[sel]
                tk = tk[sel]
                input = input[0]
                # print("input", str(input))
                # print("quand", str(quand))
                # TODO length dep

                # print("outhistory" + str(outHistory))
                # print("tk" + str(tk))
                ttk = tk.view(-1, 1, self.nh).expand(-1, outHistory.size()[1], -1)
                # print("ttk" + str(ttk))
                outh = torch.cat((outHistory, ttk), -1)
                # print("outh" + str(outh))
                if self.coverage:
                    outhav = torch.cat((outh, vHistory, aHistory), -1)
                    vk = self.sigV(outhav)
                    outh = torch.cat((outh, vk), -1)

                ek = self.a(outh).view(-1, outHistory.size()[1])
                # print("ek" + str(ek))
                ak = F.softmax(ek, 1)
                # print("ak" + str(ak))

                sk = (ak.view(-1, outHistory.size()[1], 1).expand(-1, -1, self.nh) * outHistory).sum(1)
                outh = torch.cat((input.view(outHistory.size()[0], -1), tk, sk), -1)
                tk = self.sigT(outh)
                if self.coverage:
                    aHistory = torch.cat(
                        (ak.view(-1, outHistory.size()[1], 1), torch.zeros(outHistory.size()[0], 1, 1).to(outHistory)),
                        1)
                    vHistory = torch.cat(
                        (vk.view(-1, outHistory.size()[1], 1), torch.zeros(outHistory.size()[0], 1, 1).to(outHistory)),
                        1)
                xout = torch.cat((input.view(outHistory.size()[0], -1), tk, sk), -1)

            else:
                #print("sel"+str(sel))
                #print("out"+str(output))
                xout = output[0]

        #prs("lo",lo)

        pb = tqdm(list(range(nbSimu)), dynamic_ncols=True)
        logsoft = torch.nn.LogSoftmax(dim=-1)
        sdisc=discard
        #prs("discard",discard)
        slastT=lastT
        snbAdds=nbAdds
        #prs("lastT before",lastT)
        for isimu in pb:
            hidden=None #torch.cat(lh,1)
            xout=None #torch.cat(lo,0)
            if self.cyanRNN:
                #print("louth"+str(louth))
                outHistory=None #torch.cat(louth,0)
                tk=None #torch.cat(ltk,0)
                if self.coverage:
                    aHistory = None #torch.cat(lah,0)
                    vHistory=None #torch.cat(lvh,0)
            step=0
            #alive=self.longTensor.new(range(infected.size()[0]))
            now=maxtt
            discard=sdisc.clone()
            nbAdds=snbAdds.clone()
            lastT=slastT
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
                        lll = [lo[i] for i in lic]
                        #prs("lll", lll)
                        if xout is not None: lll.append(xout)
                        #prs("lll",lll,lic,encours)
                        xout = torch.cat(lll,0)
                        lll = [lh[i] for i in lic]
                        if hidden is not None: lll.append(hidden)
                        hidden = torch.cat(lll, 1)
                        if self.cyanRNN:
                            lll = [ltk[i] for i in lic]
                            if tk is not None: lll.append(tk)
                            tk = torch.cat(lll, 0)

                            lll = [louth[i] for i in lic]
                            if outHistory is None:
                                #prs("lll",lll)
                                outHistory = torch.cat(lll,0)
                                if self.coverage:
                                    lll = [lah[i] for i in lic]
                                    aHistory = torch.cat(lll, 0)
                                    lll = [lvh[i] for i in lic]
                                    vHistory = torch.cat(lll, 0)
                            else:
                                lll.append(outHistory)
                                outHistory = torch.cat(lll, 0)
                                if self.coverage:
                                    lll = [lah[i] for i in lic]
                                    lll.append(aHistory)
                                    aHistory= torch.cat(lll, 0)
                                    lll = [lvh[i] for i in lic]
                                    lll.append(vHistory)
                                    vHistory = torch.cat(lll, 0)
                    #now-=1
                ninf = self.nextInfected(xout)
                #print("discard"+str(discard))
                #print("ninf" + str(ninf))
                #prs("encours",encours)
                ninf = ninf + discard[encours]
                ninf=logsoft(ninf)
                #prs("ninf",ninf)
                c=torch.distributions.Categorical(logits=ninf)
                c=c.sample()
                #prs("c",c)
                stopped=(c==self.nbNodes)
                nbAdds[encours[stopped]]=maxT*10
                notstopped=(c<self.nbNodes)
                encours=encours[notstopped]
                if notstopped.sum()==0:
                    hidden=None
                    xout=None
                    tk=None
                    aHistory=None
                    vHistory=None
                    outHistory=None
                    continue

                #alive=alive[notstopped]
                xout=xout[notstopped]
                qui=c[notstopped]
                #prs("nbInf before", nbInf)
                nbInf[encours,qui]+=1
                #prs("nbInf after",nbInf)
                nt = self.nextTime(xout)
                # print("gapT" + str(gapT))
                # print("nt" + str(nt))
                #gapT = now - lastT[notstopped].view(-1, 1)
                #gp = torch.log(gapT + 1)

                r = (torch.rand(xout.size()[0],10)*100).to(self.floatTensor)
                r=torch.where(lastT[encours]<testFrom,r+testFrom-lastT[encours],r)
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
                #prs(lastT, encours, t)
                lastT[encours]+=t.view(-1,1)
                #lastT[notstopped]=lastT[notstopped]+t[notstopped,ct]
                #print("gp"+str(gp))
                #print("notstopped"+str(notstopped))
                gapT = torch.log(t+1).view(-1,1) # - lastT[notstopped].view(-1, 1)

                hidden = hidden[:, notstopped, :].contiguous()
                input = self.node_embedding(qui)

                #print("input"+str(input))
                #print("gatT"+str(gapT))
                input = torch.cat((input, torch.log(gapT + 1), gapT), -1)
                input = self.encoder(input)
                input = input.unsqueeze(0)
                # prs("input:",input)
                # prs("hidden:", hidden)
                output, hidden = self.rnn(input, hidden)
                if self.cyanRNN:
                    #if t > 0:
                        #print("outh"+str(outHistory[notstopped]))
                        #print("out"+str(output.view(-1, 1, self.nh)))
                    outHistory = torch.cat((outHistory[notstopped], output.view(-1, 1, self.nh)), 1)
                    #else:
                    #    outHistory = output.view(-1, 1, self.nh)
                        # prs("output:", output)
                # prs("hidden:", hidden)
                output = self.dropout(output)
                hidden = self.dropout(hidden)

                #discard = discard[notstopped].contiguous()
                # prs("qui", qui)
                # prs("discard", discard)

                discard[encours, qui] = -float("inf")
                # prs("discard2", discard)

                # a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
                # b = torch.LongTensor([0, 2])
                # a[range(2),b]=-float("inf")
                # prs("a",a)

                if self.cyanRNN:

                    # outHistory = outHistory[:input.size()[1]]
                    if self.coverage:
                        aHistory = aHistory[notstopped]
                        vHistory = vHistory[notstopped]
                    tk = tk[notstopped]
                    input = input[0] #, notstopped]
                    # print("input", str(input))
                    # print("quand", str(quand))
                    # TODO length dep

                    # print("outhistory" + str(outHistory))
                    # print("tk" + str(tk))
                    ttk = tk.view(-1, 1, self.nh).expand(-1, outHistory.size()[1], -1)
                    # print("ttk" + str(ttk))
                    outh = torch.cat((outHistory, ttk), -1)
                    # print("outh" + str(outh))
                    if self.coverage:
                        outhav = torch.cat((outh, vHistory, aHistory), -1)
                        vk = self.sigV(outhav)
                        outh = torch.cat((outh, vk), -1)

                    ek = self.a(outh).view(-1, outHistory.size()[1])
                    # print("ek" + str(ek))
                    ak = F.softmax(ek, 1)
                    # print("ak" + str(ak))

                    sk = (ak.view(-1, outHistory.size()[1], 1).expand(-1, -1, self.nh) * outHistory).sum(1)
                    outh = torch.cat((input.view(outHistory.size()[0], -1), tk, sk), -1)
                    tk = self.sigT(outh)
                    if self.coverage:
                        aHistory = torch.cat(
                            (ak.view(-1, outHistory.size()[1], 1),
                             torch.zeros(outHistory.size()[0], 1, 1).to(outHistory)),
                            1)
                        vHistory = torch.cat(
                            (vk.view(-1, outHistory.size()[1], 1),
                             torch.zeros(outHistory.size()[0], 1, 1).to(outHistory)),
                            1)
                    xout = torch.cat((input.view(outHistory.size()[0], -1), tk, sk), -1)

                else:
                    xout = output[0]

        rInf = nbInf.to(self.floatTensor) / nbSimu
        rFin = rInf.sum()
        gt = torch.zeros((infected.size()[0], self.nbNodes)).to(infected)
        c = torch.where(infected >= 0, infected, self.longTensor.new([0]))
        gt.scatter_(1, c, torch.ones(gt.size()).to(gt))
        l = torch.where(gt > 0, rInf, 1 - rInf)
        # torch.set_printoptions(threshold=5000)
        # prs("rInf", rInf[0])
        # prs("gt", gt[0].to(self.floatTensor))
        # prs("l", l[0])
        l = torch.log(l + 1E-10)
        # prs("l",l.sum())
        rInf = l.sum()

        return(rInf,rFin)



