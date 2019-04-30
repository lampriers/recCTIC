from __future__ import print_function
import subprocess
import random
import argparse
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from operator import itemgetter, attrgetter, methodcaller
import math
from rnnDiffusionModel import rnnDiffusionModel
from Episode import Episode
import scipy.misc

#from torchvision import datasets, transforms
#from torch.autograd import Variable
import torch.autograd.profiler as profiler
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
parser = argparse.ArgumentParser(description='PyTorch variationalCTIC')
parser.add_argument('--batchSize', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
#parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='F',
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
parser.add_argument('--nh', type=int, metavar='N', default=30,
                    help='number of dimensions of the states')
parser.add_argument('--nw', type=int, metavar='N', default=30,
                    help='number of dimensions of the node embeddings')
parser.add_argument('--decay', '-d', type=float, metavar='F', default=0.0,
                    help='weight decay')
parser.add_argument('--dropout', '-dr', type=float, metavar='F', default=0.0,
                    help='dropout rate in train')
parser.add_argument('--nlayers', '-nl', type=int, metavar='N', default=3,
                    help='number of layers of the rnn')
parser.add_argument('--glayers', '-ng', type=int, metavar='N', default=3,
                    help='number of layers of the decoder')
parser.add_argument('--enclayers', '-ne', type=int, metavar='N', default=3,
                    help='number of layers of the encoder')
parser.add_argument('--fromFile', '-fr', type=str, metavar='S', default="",
                    help='model file to load (default: empty)')
parser.add_argument('--freqTest', type=int, metavar='N', default=100,
                    help='frequence for tests (default: 10)')
parser.add_argument('--fromTests', nargs='+', type=int, default=[0],
                    help='fromTests (always prepend with 0)')
parser.add_argument('-cyan', action='store_true', default=False,
                    help='attention or not (default False)')
parser.add_argument('-coverage', action='store_true', default=False,
                    help='coverage or not (default False)')
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
args.T=T

diffMod=None
if len(args.fromFile)>0:
    if args.cuda_device >= 0:
        diffMod = torch.load(args.fromFile,map_location=lambda storage, loc: storage.cuda(args.cuda_device))
        diffMod.setcuda(args.cuda_device)
    else:
        diffMod = torch.load(args.fromFile, map_location=lambda storage, loc: storage)
    diffMod.setDropout(args.dropout)
    #diffMod.cpu()
else:
    diffMod = rnnDiffusionModel(args.nh, args.nw, args.T, args.nbNodes,nlayers=args.nlayers,glayers=args.glayers,enclayers=args.enclayers, dropout=args.dropout,cyanRNN=args.cyan,coverage=args.coverage)  # ./arti_episodes_train.txt")
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
        aLinks = 0
        anbLinks = 0
        for batch in pb:
            diffMod.zero_grad()
            iter_logger.new_iteration()
            eps,infections,times=batch
            #print("ici")
            tt=None
            if args.computellLinks:
                tt = trainLinks[bin]
            #print(str(infections))
            lx, lt,rLinks,nbLinks=diffMod(infections,times,fromWho=tt)
            #print("la")
            aLinks += rLinks
            anbLinks += nbLinks
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
                lx,lt = diffMod(infections, times)
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
        #epoch_logger.log('g', iter_logger.get_epoch('g'))
        if anbLinks>0:
            epoch_logger.log('INF', aLinks/anbLinks)
            #epoch_logger.log('nbLinks', anbLinks)

        print('\n train', epoch_logger.get('NLL'))#,epoch_logger.get('lx'),epoch_logger.get('lt'),epoch_logger.get('s')) #,epoch_logger.get('g'))
        if anbLinks > 0:
            print("goodLinks="+str(aLinks/anbLinks)+" over "+str(anbLinks))
    if args.nbTestEpisodes > 0 and epoch % args.freqTest == 0:

        diffMod.eval()
        for fromTest in args.fromTests:
            pb = tqdm(testBins, dynamic_ncols=True)
            bin = 0
            #print(str(diffMod.rnn.dropout))
            #print(str(diffMod.nextTime.dropout))
            aFin = None
            aInf = None
            aLinks = 0
            anbLinks = 0
            #print(str(diffMod.nextTime.training))
            for batch in pb:
                with torch.no_grad():
                    #diffMod.zero_grad()
                    iter_loggert.new_iteration()
                    eps,infections, times= batch
                    tt=None
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

                    if args.nbsimu > 0 :
                        rInf, rFin = diffMod.simuFrom(infections, times, fromTest, args.nbsimu, T)
                        if aInf is None:
                            aInf=rInf/ args.nbTestEpisodes
                            aFin=rFin/ args.nbTestEpisodes

                        else:
                            aInf += rInf / args.nbTestEpisodes
                            aFin += rFin / args.nbTestEpisodes




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
                print("ainf "+str(aInf.item())+" nbInf "+str(aFin))
            if anbLinks > 0:
                epoch_logger.log('INF', aLinks / anbLinks)
                #epoch_logger.log('nbLinks', anbLinks)
            # epoch_logger.log('g', iter_logger.get_epoch('g'))

            print('\n test', epoch_logger.get('NLL_'+str(fromTest))) #, epoch_logger.get('lxt_'+str(fromTest)), epoch_logger.get('ltt_'+str(fromTest)),
                  #epoch_logger.get('st_'+str(fromTest)))  # ,epoch_logger.get('g'))
            if anbLinks > 0:
                print("goodLinks=" + str(aLinks / anbLinks) + " over " + str(anbLinks))
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

    #torch.save(diffMod.state_dict(), os.path.join(args.outDir, "model_" + str(epoch) + ".pt"))
        #torch.save(optimizer.state_dict(), os.path.join(args.outDir, "optim_" + str(epoch) + ".pt"))




#python rnnDiffusion.py -o xp/rnnDif/arti -itr arti_episodes_train.txt -ite arti_episodes_val.txt
#python rnnDiffusion.py -o xp/rnnDif/arti -itr dataTest.txt -ite dataTest.txt --fromTests 2 -cyan