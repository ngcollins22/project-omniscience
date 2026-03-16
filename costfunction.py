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
    return (launchnum_mars + launchnum_relay + 1)*150 #Add one for lunar launch
#payloads: 4 solar, 10 comms, 10 PNT
def payload(STCarray: np.array,lc):
    c1_s = 2.5*0.6*(STCarray[0,0]**0.8)*0.8
    c1_t = 2.5*0.6*(STCarray[1,0]**0.8)
    c1_c = 2.5*0.6*(STCarray[2,0]**0.8)
    solarcost, svec= cpd(c1_s,STCarray[0,1],lc)
    terraincost, tvec = cpd(c1_t,STCarray[1,1],lc)
    commscost, cvec = cpd(c1_c,STCarray[2,1],lc)
    return solarcost,svec,terraincost,tvec,commscost,cvec#solarcost,terraincost,commscost#,x,y,z

payloads = np.array(([65,4],[90,10],[113.2,10]))
S1,S2,T1,T2,C1,C2 = payload(payloads,0.85)
print(S1)
print(T1)
print(C1)
print(np.sum([S1,T1,C1]))
print(np.average(S2))
print(np.average(T2))
print(np.average(C2))