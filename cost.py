import math
import numpy as np
from costfunction import cpd, launch_cost, payload

#Primary cost driver for parametric cost modeling in this case is DRY MASS
def omnicost(drymass: int,satnum_mars: int,satnum_relay: int,lifetime: int,payloads: np.array):
    #Total satellite bus cost (cost per doubling, 85% learning curve)
    cost_bus1 = 2.2*0.45*drymass**(0.7)
    lc = 0.85
    cost_bustot, cn = cpd(cost_bus1,satnum_mars,lc)

    #Integration and testing costs
    cost_it = 0.3*cost_bus1 + 0.1*np.sum(cn[1:])
    
    #Payload costs
    cost_payload = payload(payloads,lc)#S,T,C

    #Major cost sectors
    dev_c = cost_bustot + cost_payload + cost_it
    launchcost = launch_cost(satnum_mars,satnum_relay)
    k = np.size(payloads)**0.25 #varies based on number of payloads (Solar EWS, PNT, comms)
    opcost = 1.2*np.sqrt(drymass)*((satnum_mars+satnum_relay)**lc)*k
    return dev_c + launchcost + opcost*lifetime #Total cost in millions of USD

drymass = 1200      #kg
satnum_mars = 24    
satnum_relay = 3
lifetime = 15       #years
payloads = np.array(([65,4],[90,10],[190,10]))
TotalCost = omnicost(drymass,satnum_mars,satnum_relay,lifetime,payloads)
print(TotalCost)



