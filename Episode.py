from tqdm import tqdm
from collections import OrderedDict
from sortedcollections.recipes import OrderedSet
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import torch

class Episode():
    nbEpisodestot=0
    # s : list of <infected nodes:timestamp>
    def __init__(self, s,fromWho={}):
        self.times = {}  # dict of timestamps associated to sets of infected nodes at this time
        self.nodes = {}  # dict of nodes associated to their infection timestamp
        # self.nodesList=[]     # List of nodes in the order of infection
        self.listN=OrderedDict()
        self.length=len(s)
        self.T = None
        self.fromWho=fromWho
        Episode.nbEpisodestot+=1
        self.id=Episode.nbEpisodestot


        #print("frWho"+str(fromWho))

        for x in s:
            ind = x[0]
            t = x[1]
            if (not (ind in self.nodes)) or (t<self.nodes[ind]):
                self.nodes[ind] = t

                g = OrderedSet()
                if (t in self.times):
                    g = self.times[t]
                self.times[t] = g
                g.add(ind)

        k = sorted(self.times.keys())
        #nbPrev = 0
        for t in k:
            #nbPrev += len(self.times[t])
            for x in self.times[t]:
                self.listN[x]=t
        # for x,t in self.listN.items():
        #     if t==13:
        #         print("Here "+str(self.listN))
        #         print(self.times)
        #         print(s)
        #     break


        # self.nodesList=[x[0] for x in sorted(self.nodes.items(), key=lambda t: t[1])]




    def recomputeLength(self,T=-1):
        nb=0
        #print("here",str(T))
        for el, ti in self.listN.items():
            if (T>0) and (ti > T):
                #print("break!",str(nb))
                break
            nb+=1
        self.length=nb
        self.T=T

    @classmethod
    def loadEpisodesFromFile(cls,inputFile,nodes=None): #,world=False):  # , nsteps=100):
        episodes = []
        nodesEp = []
        addNodes=False

        if nodes is None:
            addNodes=True
            nodes = {}  # table node_name => index
            #if world:
            nodes["world"] = 0
        T = 0
        # times=[]
        # steps=[]
        # stlenChanged=False
        # nodesChanged=False
        sumLength=0
        s2Length=0
        i = 0  # len(self.episodes)
        with open(inputFile, "r") as f:
            while (True):
                ligne = str(f.readline())
                ligne=ligne.rstrip()
                if (len(ligne) == 0):
                    break
                # print(ligne)
                ll = ligne.split(";")
                s = ll[0].split("\t")

                infections = []
                #if world:
                infections.append((0, 0))

                for x in s:
                    #print("x "+str(x))
                    x = x.split(":")
                    ep = []
                    name=":".join(x[0:-1])
                    if (name not in nodes):
                    #    ep = nodesEp[nodes[x[0]]]
                    #else:
                        if not addNodes:
                            continue
                        nodes[name] = len(nodes)
                        #nodesEp.append(ep)

                    ind = nodes[name]

                    t = int(float(x[-1])) + 1

                    if (t > T):
                        T = t
                        # stlen = (T / nsteps) + 1

                    #ep.append(i)
                    infections.append((ind, t))
                #print("nodes"+str(nodes))
                fromWho={}
                if len(ll) >= 3:
                    s=ll[2].split("\t")
                    #print(str(s))
                    for x in s:
                        #print(str(x))
                        x = x.split(":")
                        #print("xinnodes "+str(x[0])+":"+str(x[0] in nodes))

                        #print("yinnodes " + str(x[1]) + ":" + str(x[1] in nodes))

                        if x[0] not in nodes or x[1] not in nodes:
                            #print(str(x[0])+" or "+str(x[1])+" not in nodes !")
                            continue
                        fromWho[nodes[x[0]]]=nodes[x[1]]
                #print("fr"+str(fromWho))
                sumLength+=len(infections)
                s2Length += len(infections)**2
                episodes.append(Episode(infections,fromWho=fromWho))
                i += 1

        episodes = np.array(episodes)
        mean=sumLength*1.0/len(episodes)
        var=s2Length*1.0/len(episodes)-mean**2
        print("Mean length = "+str(mean))
        print("StDev length = " + str(np.sqrt(var)))
        # if(stlenChanged):
        #    map(lambda x: x.clearSteps(),self.episodes)

        return (episodes, nodes, T)

    @classmethod
    def computeGraph(cls,bins,nodes):
        print("compute graph")
        graph = set()

        for (eps, infected , _) in bins:

            for ep in eps:
                l=ep.listN
                oldt=0
                l = [(x, y) for x, y in l.items()]
                pr = [(l[k][0],l[j][0])  for j in range(len(l)) for k in range(j) if (l[k][1] < l[j][1])]
                graph.update(pr)

        for i in range(len(nodes)):
            graph.add((i,i))
        ind=torch.LongTensor([[a,b] for a,b in graph])
        v=torch.FloatTensor([1]*len(graph))
        graph=torch.sparse.FloatTensor(ind.t(),v).to_dense().to(bins[0][1])



        print("graph computed "+str(graph.size()))
        return graph

    @classmethod
    def makeEpisodeBins(cls, episodes, nodes, batchSize=128, train=1, val=0, cuda=-1, trainT=-1, testT=1):

        # if not (train+val+test)==1:
        #    raise RuntimeError
        #se = sorted(episodes, key=attrgetter('length'), reverse=True)
        #print(str(se))
        trainE = episodes
        testE = []
        valE = []
        if train < 1:
            ti = np.random.choice([True, False], len(episodes), True, [train, 1.0 - train])
            allI = np.array(range(len(episodes)))
            trainI = allI[ti]
            print("trainI="+str(trainI))
            resteI = allI[np.invert(ti)]
            ti = np.random.choice([True, False], len(resteI), True, [val, 1.0 - val])
            valI = resteI[ti]
            print("valI=" + str(valI))
            testI = resteI[np.invert(ti)]
            print("testI=" + str(testI))
            trainE = [episodes[i] for i in trainI]
            valE = [episodes[i] for i in valI]
            testE = [episodes[i] for i in testI]

        trainBins = cls.makeBins(trainE, nodes, batchSize, cuda,trainT)
        valBins = cls.makeBins(valE, nodes, batchSize, cuda, testT)
        testBins = cls.makeBins(testE, nodes, batchSize, cuda, testT)

        return (trainBins,valBins,testBins)

    @classmethod
    def makeBins(cls, episodes, nodes, batchSize, cuda=-1,T=-1):
        #if T is not None:
        list(map(lambda e:e.recomputeLength(T),episodes))
        episodes= sorted(episodes, key=attrgetter('length'), reverse=True)
        #x=[e.length for e in episodes]
        #print("x:",str(x))
        #print(episodes[0].listN)
        ret = []
        print("make bins for " + str(len(episodes)) + " episodes")
        if (len(episodes) == 0):
            return ret
        b = batchSize
        if (len(episodes) < b):
            b = len(episodes)
        d = episodes[0].length
        eps=[]
        checknb=0
        nbinf=0
        infected = torch.LongTensor(b, d)
        times = torch.FloatTensor(b, d)
        #notinf= torch.LongTensor(b, len(nodes))
        c = 0
        for i in tqdm(range(len(episodes))):
            e = episodes[i]
            l = e.listN
            x = np.ones(d, dtype=int) * (-1)
            t = np.ones(d, dtype=int) * (-1)
            #f = np.ones(len(nodes), dtype=int)

            j = 0
            for el, ti in l.items():

                if (T>0) and  (ti>T):
                    break
                x[j] = el
                t[j] = ti
                #f[el]=0
                j += 1
                nbinf+=1
            infected[c] = torch.LongTensor(x)
            times[c] = torch.FloatTensor(t)
            #notinf[c] = torch.LongTensor(f)
            eps.append(e)
            c += 1

            if c == b:

                if cuda >=0:
                    #infected=infected.pin_memory()
                    #times=times.pin_memory()
                    #notinf = notinf.pin_memory()
                    infected=infected.cuda(cuda)
                    times=times.cuda(cuda)
                    #notinf = notinf.cuda(cuda)

                ret.append((eps,infected, times))
                checknb+=infected.size()[0]
                #print("append for " + str(i))
                #print(str(ret[-1]))
                c = 0

                if len(episodes) > (i + 1):
                    if (len(episodes) < i + 1 + b):
                        b = len(episodes) - i - 1
                    d = episodes[i + 1].length
                    eps = []
                    infected = torch.LongTensor(b, d)
                    times = torch.FloatTensor(b, d)
                    #notinf = torch.LongTensor(b, len(nodes))
        print("Bins done for "+str(checknb)+", "+str(nbinf)+" infections")

        return ret

    @classmethod
    def remakeBins(cls,bins,nodes, batchSize, cuda=-1,T=-1):
        episodes=[]
        for b in bins:
            ep,_,_,_,_=b
            episodes.extend(ep)

        return cls.makeBins(cls, episodes, nodes, batchSize, cuda,T)

    @classmethod
    def makeBinsOfLinks(cls,bins,nodes):
        ret=[]
        nbLinks=0
        #print(str(nodes))
        for b in bins:
            ep, infected, _ = b
            #print("infected"+str(infected))
            x = infected.new(infected.size()).fill_(0)
            lbin=None
            ie=0
            for e in ep:
                links=e.fromWho
                l=e.listN
                #print("episode "+str(e.id))
                #print(str(l))
                #print(str(links))


                order={}
                j=0
                for i in l:
                    order[i]=j
                    f=links.get(i,0)
                    #print(str(f))
                    #print(str(order))
                    if f>=0:
                        f=order[f]
                        if i>0:
                            nbLinks += 1
                    #print("x"+str(x))
                    #print(str(j)+":"+str(f))
                    x[ie][j]=f
                    j+=1


                ie+=1
            ret.append(x)
        print("Nb Links="+str(nbLinks))
        return ret




    #def cutBins(cls,bins,cutTime=1):
