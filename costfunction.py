import numpy as np
import math
#Cost per doubling
def cpd(initial,satnum,lc):
    n = np.arange(1,satnum+1)
    cn = initial*n**(np.log2(lc))
    return np.sum(cn), cn
#Falcon Heavy launch cost, $150 million per launch
def launch_cost(satnum,relaynum):
    launchnum_mars = math.ceil(satnum/4)
    launchnum_relay = math.ceil(relaynum/4)
    return (launchnum_mars + launchnum_relay)*150
#payloads: 4 solar, 10 comms, 10 PNT
def payload(STCarray: np.array,lc):
    c1_s = 0.6*STCarray[0,0]**0.8
    c1_t = 0.6*STCarray[1,0]**0.8
    c1_c = 0.6*STCarray[2,0]**0.8
    solarcost, _ = cpd(c1_s,STCarray[0,1],lc)
    terraincost, _ = cpd(c1_t,STCarray[1,1],lc)
    commscost, _ = cpd(c1_c,STCarray[2,1],lc)
    return solarcost+terraincost+commscost
