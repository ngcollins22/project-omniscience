
from mars_constellation import ConstellationConfig, build_constellation, propagate, compute_ground_tracks


#build your constellation
randomVariableYouMightWant= 3
constellationName=f'bruh{randomVariableYouMightWant}'
inc=55
numSats=30
numPlanes=5
f=1
alt=9720


#creates constellation object
cfg = ConstellationConfig(
name = str(constellationName),
inclination_deg=float(inc),
total_sats=int(numSats),
planes=int(numPlanes),
phasing=int(f),
altitude_km=float(alt),
pattern=str('DELTA'))

#build tha hoe like minecraft
constellation = build_constellation(cfg)

#these were the default i saw 
duration_sec = 24 * 3600      # 24 hours
step_sec = 300*6               # 30 min

#farm the propagator
times, inertial_pvs, fixed_pvs = propagate(constellation,duration_sec=duration_sec,step_sec=step_sec)


#its over
tracks=compute_ground_tracks(constellation,times, fixed_pvs)
print(tracks)