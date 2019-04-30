

from __future__ import print_function
import subprocess
import random
import argparse
import numpy as np
#import re
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import torch.backends.cudnn as cudnn
#from operator import itemgetter, attrgetter, methodcaller
#import math

from Episode import Episode
#import scipy.misc

# from torchvision import datasets, transforms
#from torch.autograd import Variable
#import torch.autograd.profiler as profiler
from utils import *

import sys
import torch.nn.functional as F
#from sortedcollections.recipes import OrderedSet
#from blaze.expr.math import isnan
from collections import OrderedDict

sys.path.append("utils")
#from kbhit import KBHit
from tqdm import tqdm
import json
import shutil
from sys import argv
from recCTICModel import *
#
# Training settings
parser = argparse.ArgumentParser(description='PyTorch variationalCTIC')
parser.add_argument('--batchSize', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 1024)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='F',
                    help='learning rate (default: 0.001)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='F', help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda-device', '-c', type=int, default=-1, metavar='N',
                    help='specifies CUDA device (default no cuda device)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--inTrainFile', '-itr', type=str, metavar='S',
                    help='input train file', required=True)
parser.add_argument('--noTrain', type=bool, metavar='B',
                    help='ignore train on train data', default=False)
parser.add_argument('--inTestFile', '-ite', type=str, metavar='S',
                    help='input test file', default="")
parser.add_argument('--rebuildTest', '-rt', type=bool, metavar='B',
                    help='rebuilds structures for test (has to be set to True when applying a saved model with a new test file) ',
                    default=False)
parser.add_argument('--globalR', '-gr', type=bool, metavar='B',
                    help='if true, a unique delay is used for every transmission (default:False)',
                    default=False)

#parser.add_argument('--noBaseline', action='store_true', default=False,
#                    help='if set, add no expectation baseline to reinforce')
parser.add_argument('--baselineTests', type=int, metavar='N', default=100,
                    help='size of the window for the baseline average estimation  (default: 100)')
parser.add_argument('--outDir', '-o', type=str, metavar='S',
                    help='out directory', required=True)
# parser.add_argument('--trainRate', '-tr', type=float, metavar='F', default=1.0,
#                    help='rate of episodes from input to use for training')
# parser.add_argument('--valRate', '-vr', type=float, metavar='F', default=0.0,
#                    help='rate of episodes from input to use for validation')
parser.add_argument('--nh', type=int, metavar='N', default=50,
                    help='number of dimensions of the states (default:50)')
parser.add_argument('--nz', type=int, metavar='N', default=1,
                    help='number of dimensions of the state output (default:1')
parser.add_argument('--decay', '-d', type=float, metavar='F', default=0.0,
                    help='weight decay')
parser.add_argument('--dropout', '-dr', type=float, metavar='F', default=0.0,
                    help='dropout rate in train (default: 0)')
parser.add_argument('--nlayersK', '-nlk', type=int, metavar='N', default=0,
                    help='number of layers for the Knet module (default: 0)')
parser.add_argument('--nlayersZ', '-nlz', type=int, metavar='N', default=1,
                    help='number of layers for the Znet module (default: 1)')
parser.add_argument('--nlayersZK', '-nlzk', type=int, metavar='N', default=0,
                    help='number of layers for the decoder of the state (default: 0)')
parser.add_argument('--fromFile', '-fr', type=str, metavar='S', default="",
                    help='model file to load (default: empty)')
parser.add_argument('--testT', type=int, metavar='N', default=-1,
                    help='lastTime inference in test (default: -1)')
parser.add_argument('--freqTest', type=int, metavar='N', default=100,
                    help='frequence for tests (default: 100)')
parser.add_argument('--fromTests', nargs='+', type=int, default=[0],
                    help='fromTests (always prepend with 0)')
parser.add_argument('--trainT', type=int, metavar='N', default=-1,
                    help='lastTime in train (default: -1 => max train time)')
parser.add_argument('--T', type=int, default=-1, metavar='N',
                    help='last step (default: -1 : max from train)')
parser.add_argument('--nbEst', type=int, metavar='N', default=10,
                    help='nbEstimations for likelihood in test (default: 10)')
#parser.add_argument('--temp1', type=float, metavar='F', default=2.0/3.0,
#                    help='temperature of the softmax of q (default: 2/3)')
#parser.add_argument('--temp1dec', type=float, metavar='F', default=1.0,
#                    help='factor for temperature of the softmax of q (default: 1.0)')
#parser.add_argument('--temp2', type=float, metavar='F', default=2.0/5.0,
#                    help='temperature of the softmax of p (default: 2/5)')
parser.add_argument('--historyWeight', '-hw', type=float, metavar='F', default=-1.0,
                    help='weight of the history for computing k. -1 => weight learned  (default: -1.0)')
parser.add_argument('--rhistoryWeight', '-hwr', type=float, metavar='F', default=0.0,
                    help='weight of the history for computing r. -1 => weight learned  (default: 0.0)')
#parser.add_argument('--uWeight', '-uw', type=float, metavar='F', default=0.0,
#                    help='weight of the emitter for computing k.  -1 => weight learned  (default: 0.0)')
parser.add_argument('-rnn', type=int, metavar='N', default=1,
                    help='rnn mode (-1=no rnn, 0=rnn, 1=gru, default=1)')
parser.add_argument('-nbn', type=int, metavar='N', default=100,
                    help='nb max nodes to consider together for non infections (-1=all, default=100)')
parser.add_argument('-noworld', action='store_true', default=False,
                    help='no world node (default False)')
parser.add_argument('-noreinforce', action='store_true', default=False,
                    help='remove reinforce optimization (default False)')
parser.add_argument('--computellLinks', '-cl', action='store_true', default=False,
                    help='compute likelihood of known links or not. only possible if fromWho known in episodes (default False)')
parser.add_argument('--nbsimu', '-ns', type=int, default=0,
                    help='number of simulations to perform at each test (default 0)')
parser.add_argument('--restrainToGraphFromTrain', '-g', type=int, default=2,
                    help='the likelihood is computed on links from a graph built from the train (0=No, 1=only in test, 2=both in test and train, default 2)')
#parser.add_argument('--restrainToGraphFromTrain', '-g', action='store_true', default=False,
#                    help='the likelihood is computed on links from a graph built from the train (default False)')
parser.add_argument('--hwfactor', '-hwf', type=float, metavar='F', default=1.0,
                    help='history weight gradient factor (default: 1.0)')
parser.add_argument('--qfactor', '-qf', type=float, metavar='F', default=1.0,
                    help='gradient factor for variational parameters (default: 1.0)')
parser.add_argument('--condition', '-cm', type=int, default=0,
                    help='conditionning mode (0:aux sampling, 1: importance sampling. default 0)')
parser.add_argument('--condDist', '-cd', type=int, default=0,
                    help='conditionning distribution (0:sampling from aux, 1: random sampling. default 0)')



args = parser.parse_args()
args.commit = subprocess.check_output(['git', 'show-ref']).decode('utf-8')

graphForTrain=False
if args.restrainToGraphFromTrain>1:
    graphForTrain = True

print("device=" + str(args.cuda_device))
print(str(args))

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if (args.cuda_device >= 0):
    # print("cuda")
    cudnn.benchmark = True
    torch.cuda.device(args.cuda_device)
    torch.cuda.manual_seed(args.seed)

args.reinforce=(not args.noreinforce)
args.world=not args.noworld

# floatTensor = torch.cuda.FloatTensor(1,device=args.cuda_device)
# print("a"+str(floatTensor))
torch.set_printoptions(threshold=5000)

episodes, nodes, T = Episode.loadEpisodesFromFile(args.inTrainFile)
#if args.trainT==0:
#    args.trainT=T
trainBins = Episode.makeBins(episodes, nodes, batchSize=args.batchSize, cuda=args.cuda_device, T=args.trainT)

#trainLinks=None
if args.computellLinks:
    trainLinks=Episode.makeBinsOfLinks(trainBins,nodes)
    #print(trainLinks)

graph=None
if args.restrainToGraphFromTrain>0:
    graph=Episode.computeGraph(trainBins,nodes)

testBins = None
if len(args.inTestFile) > 0:
    episodesT, nodesT, TT = Episode.loadEpisodesFromFile(args.inTestFile, nodes)

    testBins = Episode.makeBins(episodesT, nodes, batchSize=args.batchSize, cuda=args.cuda_device, T=args.testT)
    if args.computellLinks:
        testLinks = Episode.makeBinsOfLinks(testBins,nodes)

    #if args.trainT == 0:
    #    args.testT = T

args.nbNodes = len(nodes)
if args.T<0:
    args.T = T
tgraph=None
if args.restrainToGraphFromTrain>0:
    prs("transpose graph")
    tgraph = graph.t()
    print(str(tgraph)+" "+str(tgraph.sum().item()))
    prs("consider graph")

diffMod = None
if len(args.fromFile) > 0:
    if args.cuda_device>=0:
        diffMod = torch.load(args.fromFile, map_location=lambda storage, loc: storage.cuda(args.cuda_device))
        diffMod.setcuda(args.cuda_device)
    else:
        diffMod = torch.load(args.fromFile, map_location=lambda storage, loc: storage)
    for name, param in diffMod.named_parameters():
        # print(str(name) + " : " + str(param.data) + "\n")
        param.requires_grad = True

    #diffMod.temperature1=args.temp1
    #diffMod.temperature2 = args.temp2
    #if args.reinforce:
    #    print("reinforce")
    #    if args.addBaseline:
    #        print("addBaseline")#

    # diffMod.cpu()
    #g = OrderedDict(list(diffMod.qinfer.named_children()))
    #l = [t for t in args.fromTests if str(t) not in g]
    #diffMod.addForTest(testBins, l,tgraph=tgraph)
    #print("auto "+str(diffMod.knet.autoWeightH))
    diffMod.setHW(args.historyWeight,args.rhistoryWeight)
    #diffMod.setUW(args.uWeight)

    if args.cuda_device >= 0:
        diffMod.setcuda(args.cuda_device)

    if args.restrainToGraphFromTrain>0:
        if graphForTrain:
            diffMod.considerGraph(trainBins, tgraph, True)
        if not args.rebuildTest:
            diffMod.considerGraph(testBins, tgraph, False)


else:
    #print(args.temp1)
    #if not args.reinforce:
    #    diffMod = varCTICModel(trainBins, args.nh, args.nbNodes, temperature1=args.temp1, temperature2=args.temp2, dropout=args.dropout, nlayersK=args.nlayersK, nlayersZ=args.nlayersZ,
    #                          testEpisodesBins=testBins,globalR=args.globalR,historyWeight=args.historyWeight,straightThrough=args.straightThrough,fromTests=args.fromTests,rnn=args.rnn)  # ./arti_episodes_train.txt")
    #else:
        #print("reinforce")
        #if args.addBaseline:
        #    print("addBaseline")


    diffMod = EMrecCTICModel(trainBins, args.nh, args.nz, args.nbNodes,
                               dropout=args.dropout, nlayersK=args.nlayersK, nlayersZK=args.nlayersZK, nlayersZ=args.nlayersZ,
                               testEpisodesBins=testBins, globalR=args.globalR, historyWeight=args.historyWeight, rhistoryWeight=args.rhistoryWeight, #uWeight=args.uWeight,
                               rnn=args.rnn,world=args.world,tgraph=tgraph,graphForTrain=graphForTrain)  # ./arti_episodes_train.txt")

    # graph = torch.LongTensor(8,8).fill_(0)
    # graph[0,1]=1
    # graph[0,3] = 1
    # graph[1,3] = 1
    # graph[2,5] = 1
    # diffMod.considerGraph(trainBins, graph, True)



# exit(0)


if (args.cuda_device >= 0):
    diffMod.setcuda(args.cuda_device)
if args.restrainToGraphFromTrain>0:
    graph=graph.to(diffMod.longTensor) #.view(diffMod.nbNodes,diffMod.nbNodes).contiguous()
    tgraph = tgraph.to(diffMod.longTensor) #.view(diffMod.nbNodes,diffMod.nbNodes).contiguous()
    print("graph links "+str(graph.sum()))
    print("tgraph links " + str(tgraph.sum()))
args.nbTrainEpisodes = 0
if (len(trainBins) > 0):
    args.nbTrainEpisodes = (len(trainBins) - 1) * args.batchSize + trainBins[-1][1].size()[0]

args.nbTestEpisodes = 0
if (testBins is not None):
    args.nbTestEpisodes = (len(testBins) - 1) * args.batchSize + testBins[-1][1].size()[0]
#args.T = T

if args.nbTestEpisodes > 0 and args.rebuildTest:
    diffMod.buildForTest(testBins,tgraph)
    if (args.cuda_device >= 0):
        diffMod.setcuda(args.cuda_device)
    #if args.restrainToGraphFromTrain:
    #    diffMod.considerGraph(testBins, tgraph, False)



for param in diffMod.parameters():
    param.requires_grad = True
for param in diffMod.q.parameters():
    param.requires_grad = False


if args.nbTestEpisodes > 0:
    for param in diffMod.qtest.parameters():
        param.requires_grad = False

if args.historyWeight<0:
    for param in diffMod.knet.hW.parameters():
        param.requires_grad=False

if args.rhistoryWeight<0:
    for param in diffMod.rnet.hW.parameters():
        param.requires_grad=False

parsCommon=[{"params":list(filter(lambda p: p.requires_grad, diffMod.parameters()))}]
if args.historyWeight<0:
    #print("pars"+str(parsCommon))
    for param in diffMod.knet.hW.parameters():
        param.requires_grad=True
    parsCommon.append({"params":diffMod.knet.hW.parameters(), 'lr':args.lr*args.hwfactor})
if args.rhistoryWeight<0:
    #print("pars"+str(parsCommon))
    for param in diffMod.rnet.hW.parameters():
        param.requires_grad=True
    parsCommon.append({"params":diffMod.rnet.hW.parameters(), 'lr':args.lr*args.hwfactor})


commonOptimizer = torch.optim.Adam(parsCommon, weight_decay=args.decay, lr=args.lr)

#if  args.historyWeight<0:





print("Logs in: " + args.outDir)
iter_logger = LogCSV(args.outDir, epoch_size=len(trainBins), filename='iter_log.csv')
epoch_logger = LogCSV(args.outDir, filename='epoch_log.csv', wipe=False)
loggerEst = LogCSV(args.outDir, filename='estimations_log.csv', wipe=False)
iter_loggert = None
if args.nbTestEpisodes > 0:
    iter_loggert = LogCSV(args.outDir, epoch_size=len(testBins), filename='iter_logt.csv', wipe=False)

# xp info
cmd=str(vars(args))
cmdline="python "+" ".join(argv)
args.commandLine=cmdline
args.model = str(diffMod).split("(")[0]
args.optim = str(commonOptimizer)
with open(os.path.join(args.outDir, 'info.json'), 'w') as info:
    json.dump(vars(args), info, sort_keys=True, indent=4)

with open(os.path.join(args.outDir, 'info_long.json'), 'w') as info:
    args.model = str(diffMod)
    args.nodes = str(nodes)
    json.dump(vars(args), info, sort_keys=True, indent=4)

torch.save(diffMod, os.path.join(args.outDir, "diffMod_start"))

diffMod.verbose = 0
qm=[[] for i in range(len(trainBins))]
pm=[[] for i in range(len(trainBins))]

if args.nbTestEpisodes > 0 :
    qmt={j:[[] for i in range(len(testBins))] for j in args.fromTests}
    pmt={j:[[] for i in range(len(testBins))] for j in args.fromTests}

best=None
bestChanged=False
for epoch in range(args.epochs):

    diffMod.train()
    print('Epoch ' + str(epoch))
    epoch_logger.new_iteration()

    if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
        loggerEst.new_iteration()

    if not args.noTrain:

        pb = tqdm(trainBins, dynamic_ncols=True)
        bin = 0

        """
        if epoch==171:
            diffMod.verbose=2
            diffMod.zinit.register_forward_hook(printnorm)
            diffMod.zinit.register_backward_hook(printgradnorm)

            diffMod.knet.register_forward_hook(printnorm)
            diffMod.knet.register_backward_hook(printgradnorm)
        """
        estLG = None
        aInf=None
        aLinks=None
        aLinksf = None
        aNext=None
        aFin=None
        anbLinks=None
        anbInfs = None
        crLinks=0
        cnLinks=0
        gra=graph
        if not graphForTrain:
            gra=None
        for batch in pb:


            diffMod.zero_grad()
            iter_logger.new_iteration()
            eps, infections, times = batch
            tt = None
            if args.computellLinks:
                tt = trainLinks[bin]
                #print("trainLinks" + str(tt))
            px, pnotx, qz, _, _ , _,  _, _ = diffMod(infections, times, bin, -1,nbMaxNodes=args.nbn,fromQui=tt,graph=gra)

            r=(px+pnotx-qz).detach()
            loss = -r
            if args.baselineTests>0:
                qm[bin].append(qz.clone()+1)
                pm[bin].append((px + pnotx).clone())
                if len(qm[bin]) > args.baselineTests:
                    qm[bin].pop(0)
                    pm[bin].pop(0)
                if (len(pm[bin])>0):
                    #print(str(pm[bin]))
                    pp=torch.cat(pm[bin],dim=1)

                    qq = torch.cat(qm[bin], dim=1)
                    #print(str(qq.size()))
                    if epoch>(args.baselineTests*0):
                        r -= (torch.sum(pp,dim=1)/len(pm[bin]) - torch.sum(qq,dim=1)/len(qm[bin])).view(-1,1)
                        r=r.detach()
            der=-px-pnotx
            if args.reinforce and epoch > (args.baselineTests * 0):
                der=der-(r-1) * qz
            if not args.reinforce:
                der=der+qz
            der = der.sum()

            if epoch % args.log_interval == 0 or epoch == args.epochs - 1:

                with torch.no_grad():

                    estL = loss.clone().view(-1, 1)
                    #normL = (pz - qz).clone().view(-1, 1)

                    xEst = tqdm(range(args.nbEst-1), dynamic_ncols=True)
                    for xE in xEst:
                        lpx, lpnotx, lqz, _, _, _, rLinks, nLinks = diffMod(infections, times,
                                                    bin, -1,nbMaxNodes=args.nbn,fromQui=tt,graph=gra)
                        l = -(lpx + lpnotx-lqz)
                        estL = torch.cat((estL, l.view(-1, 1)), -1)
                        #normL = torch.cat((normL, (lpz - lqz).view(-1, 1)), -1)
                        if  args.baselineTests > 0:
                            qm[bin].append(lqz.clone()+1)
                            pm[bin].append((lpx + lpnotx).clone())
                            if len(qm[bin]) > args.baselineTests:
                                qm[bin].pop(0)
                                pm[bin].pop(0)
                        # if aLinksf is None:
                        #     aLinksf = rLinks / ((args.nbEst-1)*args.nbTrainEpisodes)
                        #     anbLinks=nbLinks/ (args.nbEst-1)
                        #     #anbInfs = nbLinks[1] / (args.nbEst - 1)
                        # else:
                        #     aLinksf += rLinks / ((args.nbEst-1)*args.nbTrainEpisodes)
                        #     anbLinks += nbLinks/ (args.nbEst-1)
                        crLinks += rLinks
                        cnLinks += nLinks

                    estL = estL * (-1)
                    # prs("estL",estL)
                    val = estL.max(-1, True)[0]
                    val = val.detach()
                    ex = torch.exp(estL - val)
                    ret = val + torch.log(ex.sum(-1, True))

                    ret=-ret+np.log(args.nbEst)
                    ret = -ret.sum()

                    if estLG is None:
                        estLG = ret / args.nbTrainEpisodes
                    else:
                        estLG = estLG + ret / args.nbTrainEpisodes

                    st = 0


            #loss=loss-qz
            loss = loss.sum()

            # print("loss"+str(loss))
            # loss = loss / args.nbTrainEpisodes
            iter_logger.log("epoch", epoch)
            iter_logger.log("it", bin+1)
            iter_logger.log("px", px.sum().detach().cpu().numpy().item(0) / args.nbTrainEpisodes)
            iter_logger.log("pnotx", pnotx.sum().detach().cpu().numpy().item(0) / args.nbTrainEpisodes)
            iter_logger.log("pz", (px+pnotx).sum().detach().cpu().numpy().item(0) / args.nbTrainEpisodes)
            iter_logger.log("qz", qz.sum().detach().cpu().numpy().item(0) / args.nbTrainEpisodes)
            iter_logger.log('loss', loss.detach().cpu().numpy().item(0) / args.nbTrainEpisodes)
            iter_logger.log('der', der.detach().cpu().numpy().item(0) / args.nbTrainEpisodes)
            #iter_logger.log("pm", pm[bin].sum().detach().cpu().numpy().item(0))
            #iter_logger.log("qm", qm[bin].sum().detach().cpu().numpy().item(0))

            with torch.no_grad():
                s = times.new([0])
                pp = 0
                for i in diffMod.parameters():
                    s = s + torch.pow(i, 2).sum()
                    nn = 1
                    for x in list(i.size()):
                        nn = nn * x
                    pp += nn
                iter_logger.log("s", s.detach().cpu().numpy().item(0))

            pb.set_postfix(loss=loss.detach().cpu().numpy().item(0))

            der.backward()
            commonOptimizer.step()


            bin += 1
        if estLG is not None:
            loggerEst.log("NLLTrain", -estLG.item())
            print('\n NLL_Train', -estLG.item())
            if args.computellLinks:
                loggerEst.log("INF_train", crLinks / cnLinks)
                print("INF", crLinks / cnLinks)
                print("nb Links ", cnLinks/((args.nbEst-1)*args.nbTrainEpisodes))
            if args.nbsimu>0 and (aNext is not None):

                loggerEst.log("aNext", aNext.item())
                loggerEst.log("aInf", aInf.item())
                loggerEst.log("aFin", aFin.item())
                loggerEst.log("aLinks", aLinks.item())
            #li=iter_logger.get_epoch('links')/args.nbTrainEpisodes
                print("aNext", aNext.item(),"aInf", aInf.item(),"aFin", aFin.item(),"aLinks", aLinks.item()) #, "links", li)
            #loggerEst.log("linksTrain", str(li))

        epoch_logger.log('epoch', epoch)
        epoch_logger.log('loss', iter_logger.get_epoch('loss'))
        epoch_logger.log('px', iter_logger.get_epoch('px'))
        epoch_logger.log('pnotx', iter_logger.get_epoch('pnotx'))
        epoch_logger.log('pz', iter_logger.get_epoch('pz'))
        epoch_logger.log('qz', iter_logger.get_epoch('qz'))
        epoch_logger.log('s', iter_logger.get_epoch('s'))
        epoch_logger.log('der', iter_logger.get_epoch('der'))
        if args.historyWeight < 0:
            epoch_logger.log("w",diffMod.knet.hW.weight.detach().cpu().numpy().item(0))
        if args.rhistoryWeight < 0:
            epoch_logger.log("wr",diffMod.rnet.hW.weight.detach().cpu().numpy().item(0))


        print('\n train', epoch_logger.get('loss'), epoch_logger.get('px'), epoch_logger.get('pnotx'),
              epoch_logger.get('pz'), epoch_logger.get('qz'), epoch_logger.get('s'),epoch_logger.get('der'))#,epoch_logger.get('pm'),epoch_logger.get('qm'))

    if args.nbTestEpisodes > 0 and epoch % args.log_interval == 0 or epoch == args.epochs - 1:
        if args.historyWeight < 0:
            prs("w",diffMod.knet.hW.weight)
        if args.rhistoryWeight < 0:
            prs("wr",diffMod.rnet.hW.weight)

        estL = None
        diffMod.eval()
        nbE = 1
        testAll=0


        for fromTest in args.fromTests:
            pb = tqdm(testBins, dynamic_ncols=True)
            bin = 0
            estLG = None
            aInf = None
            cnLinks = 0
            crLinks=0
            aNext = None
            aFin = None
            aLinksf = None
            maxT=-1
            if fromTest>0:
                maxT=fromTest
            for batch in pb:


                eps, infections, times = batch

                tt = None
                if args.computellLinks:
                    tt = testLinks[bin]
                    #print("testLinks "+str(tt))

                with torch.no_grad():

                    xEst = tqdm(range(args.nbEst), dynamic_ncols=True)
                    for xE in xEst:
                        lpx, lpnotx, lqz, lpx2, lpnotx2, lqz2, rLinks, nLinks = diffMod(infections, times, -1, bin, tau=fromTest,nbMaxNodes=args.nbn,fromQui=tt,graph=graph,condDist=args.condDist)
                        if fromTest>0:
                            l=(lpx2 + lpnotx2 -lqz2)
                            #if args.condition==1:
                            #    l+=lpx + lpnotx -lqz
                        else:
                            l = (lpx + lpnotx -lqz)
                        #print("l",l.sum().item(),lpx.sum().item(),lpnotx.sum().item(),lqz.sum().item())
                        if xE==0:
                            estL = l.view(-1, 1)
                            norm=(lpx + lpnotx -lqz).view(-1,1)

                        else:
                            estL = torch.cat((estL, l.view(-1, 1)), -1)
                            norm = torch.cat((norm,(lpx + lpnotx - lqz).view(-1, 1)),-1)

                        crLinks+=rLinks
                        cnLinks += nLinks
                        # if aLinksf is None:
                        #     aLinksf = rLinks / (args.nbEst*args.nbTestEpisodes)
                        # else:
                        #     aLinksf += rLinks / (args.nbEst*args.nbTestEpisodes)

                    #print(estL+norm,estL)

                    # prs("estL",estL)
                    val = estL.max(-1, True)[0]
                    val = val.detach()
                    ex = torch.exp(estL - val)
                    ret = val + torch.log(ex.sum(-1, True))

                    ret1=ret-np.log(args.nbEst)

                    ret2=ret1

                    if fromTest>0: # and args.condition==1:
                        estL = estL + norm
                        val = estL.max(-1, True)[0]
                        val = val.detach()
                        ex = torch.exp(estL - val)
                        ret = val + torch.log(ex.sum(-1, True))

                        ret2 = ret - np.log(args.nbEst)

                        val = norm.max(-1, True)[0]
                        val = val.detach()
                        ex = torch.exp(norm - val)
                        norm = val + torch.log(ex.sum(-1, True))
                        norm = norm - np.log(args.nbEst)
                        #print(norm)
                        ret2=ret2-norm
                    #ret2=ret/np.log(2)
                    #pdtau=0.0
                    #if fromTest>0:
                    #    pdtau=ret
                    #    print("log pdtau="+str(ret.sum()))
                    ret1 = ret1.sum()
                    ret2=ret2.sum()

                    if estLG is None:
                        estLG = ret1 / args.nbTestEpisodes
                        estLG2 = ret2 / args.nbTestEpisodes
                    else:
                        estLG = estLG + ret1 / args.nbTestEpisodes
                        estLG2 = estLG2 + ret2 / args.nbTestEpisodes
                    if fromTest==0:
                        ee=estLG.sum().detach().cpu().numpy().item(0)
                    #print("estLG_"+str(fromTest)+"=" + str(ee))

                    st = 0
                    if args.nbsimu>0:
                        rInf, rFin = diffMod.simuFrom(infections, times, bin, False, fromTest, args.nbsimu, args.T,graph=graph,nbMaxNodes=args.nbn) #.sum().detach().cpu().numpy().item(0)
                        if aInf is None:
                            aInf=rInf/ args.nbTestEpisodes
                            aFin=rFin/ args.nbTestEpisodes
                        else:
                            aInf += rInf / args.nbTestEpisodes
                            aFin += rFin / args.nbTestEpisodes


                bin += 1

            if estLG is not None:

                loggerEst.log("epoch_" + str(fromTest), str(epoch))
                loggerEst.log("NLL_"+str(fromTest), -estLG.item())
                #loggerEst.log("estL2_" + str(fromTest), -estLG2.item())
                print('\n NLL_' + str(fromTest), -estLG.item())
                print('\n estimationL2_' + str(fromTest), -estLG2.item())

                if args.computellLinks:
                    rr=0
                    if cnLinks>0:
                        rr=crLinks / cnLinks
                    loggerEst.log("INF_" + str(fromTest), rr)
                    print("INF", rr)

                if args.nbsimu > 0:
                    loggerEst.log("CE_"+str(fromTest), aInf.item())
                    loggerEst.log("nbInfSimu_"+str(fromTest), aFin.item())
                    print("CE", aInf.item(), "nbInfSimu", aFin.item())

                #li = iter_loggert.get_epoch('links_'+str(fromTest))/args.nbTestEpisodes


                #loggerEst.log("links_"+str(fromTest), li)

            if (fromTest == 0) and (best is None or ee < best):
                best = ee
                bestChanged = True




    if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
        print(str(cmd))
        print(str(cmdline))
        iter_logger.flush()
        #if args.nbTestEpisodes > 0:
        #    iter_loggert.flush()

        epoch_logger.flush()

        if args.nbTestEpisodes > 0:
             loggerEst.flush()

        torch.save(diffMod, os.path.join(args.outDir, "diffMod_last"))
        print("Model saved")
        #torch.save(diffMod, os.path.join(args.outDir, "diffMod_last"))
        with open(os.path.join(args.outDir, "params_last.txt"), "w") as f:
            for name, param in diffMod.named_parameters():
                #if param.requires_grad:
                f.write(str(name) + " : " + str(param.data) + "\n")

        if bestChanged:
            shutil.copy(os.path.join(args.outDir, "diffMod_last"),os.path.join(args.outDir, "diffMod_best"))
            shutil.copy(os.path.join(args.outDir, "params_last.txt"), os.path.join(args.outDir, "params_best.txt"))
            bestChanged=False

#python recCTIC.py -o xp/recDiff/arti -itr arti_episodes_train.txt -ite arti_episodes_test.txt -c 0 --fromTests 0 2 5 10 -cl