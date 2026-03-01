import math
import numpy as np
from costfunction import cpd, launch_cost, payload

#Primary cost driver for parametric cost modeling in this case is DRY MASS
def omnicost(drymass: int,satnum_mars: int,satnum_relay: int,lifetime: int,payloads: np.array,gs_num):
    #Total satellite bus cost (cost per doubling, 85% learning curve)
    cost_bus1 = 2.2*0.45*drymass**(0.7)
    lc = 0.9
    cost_bustot, cn = cpd(cost_bus1,satnum_mars+4,lc) #Add four for lunar testing

    #Integration and testing costs
    cost_it = 0.3*cost_bus1 + 0.1*np.sum(cn[1:])
    
    #Total payload costs
    S,_,T,_,C,_ = payload(payloads,lc)#S,T,C
    cost_payload = np.sum([S,T,C])
    #Ground station costs, 20 million per telescope 
    cost_GS = 20*gs_num

    #Major cost sectors
    dev_c = cost_bustot + cost_payload + cost_it + cost_GS
    launchcost = launch_cost(satnum_mars,satnum_relay)
    k = np.size(payloads.shape[0])**0.25 #varies based on number of payloads (Solar EWS, PNT, comms)
    opcost = 1.2*np.sqrt(drymass)*((satnum_mars+satnum_relay)**0.85)*k
    totalcost = dev_c + launchcost + opcost*lifetime
    return totalcost, dev_c, launchcost, opcost, cn #Total cost in millions of USD

drymass = 1200      #kg
satnum_mars = 24    
satnum_relay = 3    
lifetime = 25       #years
payloads = np.array(([65,7],[90,10],[165,10]))
gs_num = 15
totalcost, dev_c, launchcost, opcost, cn = omnicost(drymass,satnum_mars,satnum_relay,lifetime,payloads,gs_num)
print(f'Total lifetime cost ($M): {totalcost:.3f}')
print(f'Mission developmental costs ($M): {dev_c:.3f}')
print(f'Total launch cost aboard Falcon Heavy ($M): {launchcost:.3f}')
print(f'Yearly mission operational cost ($M): {opcost:.3f}')
print(f'Mission operational costs over {lifetime} years ($M): {opcost*lifetime:.3f}')
print("-" * 55)

def subcost(cn):
    ato_ps = []
    pte_ps = []
    comms_ps = []
    struct_ps = []
    prop_ps = []
    for i in range(len(cn)):
        ato_ps.append(0.25*cn[i])
        comms_ps.append(0.3*cn[i])
        prop_ps.append(0.1*cn[i])
        pte_ps.append(0.18*cn[i])
        struct_ps.append(0.21*cn[i])
    return ato_ps, pte_ps, comms_ps, struct_ps, prop_ps

#Calculate average individual satellite costs
_,types,_,typet,_,typec = payload(payloads,0.85)
ato_ps, pte_ps, comms_ps, struct_ps, prop_ps = subcost(cn)
print(f'Estimated satellite cost without payload ($M): {np.average(cn):.3f}')
print(f'Estimated ATO cost per satellite ($M): {np.average(ato_ps):.3f}')
print(f'Estimated PTE cost per satellite ($M): {np.average(pte_ps):.3f}')
print(f'Estimated comms cost per satellite ($M): {np.average(comms_ps):.3f}')
print(f'Estimated structures cost per satellite ($M): {np.average(struct_ps):.3f}')
print(f'Estimated propulsion cost per satellite ($M): {np.average(prop_ps):.3f}')
print(f'Estimated type S satellite cost ($M): {np.average(cn)+np.average(types):.3f}')
print(f'Estimated type T satellite cost ($M): {np.average(cn)+np.average(typet):.3f}')
print(f'Estimated type C satellite cost ($M): {np.average(cn)+np.average(typec):.3f}')
