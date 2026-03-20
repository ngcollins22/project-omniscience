
from mars_constellation import ConstellationConfig, build_constellation, propagate, compute_ground_tracks
from pyproj import CRS, Transformer
import numpy as np
import matplotlib.pyplot as plt





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
duration_sec = 300 * 3600      # 24 hours
step_sec = 60               # 1 min

#farm the propagator
times, inertial_pvs, fixed_pvs = propagate(constellation,duration_sec=duration_sec,step_sec=step_sec)


#its over
tracks=compute_ground_tracks(constellation,times, fixed_pvs)


satellite = tracks[0]
lats = np.asarray(satellite["lat"], dtype=float)
lons = np.asarray(satellite["lon"], dtype=float)


#want ground track to start all the way to the left 

lat_tol = 1.0   #degress

#fwhere latitude is close to zero
equatorIndex = np.where(np.abs(lats) < lat_tol)[0]


#find the minimum longitude (all the way to the left)
start_idx = equatorIndex[np.argmin(lons[equatorIndex])]

#cap where the longitude is at a max (all the way to the right)
search_lons = lons[start_idx:]
end_rel_idx = np.argmax(search_lons)
end_idx = start_idx + end_rel_idx

lons = lons[start_idx:end_idx+1]
lats = lats[start_idx:end_idx+1]


plt.figure()
plt.plot(lons, lats, label="groundtrack")
plt.show()

data = np.vstack((lats, lons))

np.savetxt(
    "groundtrack_rowsss.csv",
    data,
    delimiter=",",
    fmt="%.6f",   # ← controls decimal format (no scientific notation)
)

print('ay')



##################
#Velocity stuff



# R_MARS = 3396190.0  # meters

# crs_mars_lonlat = CRS.from_proj4(f"+proj=longlat +a={R_MARS} +b={R_MARS} +no_defs")
# crs_mars_eqc    = CRS.from_proj4(f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +a={R_MARS} +b={R_MARS} +units=m +no_defs")
# transformer = Transformer.from_crs(crs_mars_lonlat, crs_mars_eqc, always_xy=True)

# t = np.arange(len(lats)) * step_sec

# x = np.full(len(lats), np.nan)
# y = np.full(len(lats), np.nan)
# vx = np.full(len(lats), np.nan)
# vy = np.full(len(lats), np.nan)

# def segment_velocity(t_seg, x_seg, y_seg):

#     vx_seg = np.gradient(x_seg, t_seg)
#     vy_seg = np.gradient(y_seg, t_seg)
#     return vx_seg, vy_seg

# start = 0
# for i in range(1, len(lons)):
#     if abs(lons[i] - lons[i - 1]) > 180.0:
#         seg_idx = np.arange(start, i)
#         x_seg, y_seg = transformer.transform(lons[seg_idx], lats[seg_idx])
#         vx_seg, vy_seg = segment_velocity(t[seg_idx], x_seg, y_seg)

#         x[seg_idx] = x_seg
#         y[seg_idx] = y_seg
#         vx[seg_idx] = vx_seg
#         vy[seg_idx] = vy_seg

#         start = i  

# seg_idx = np.arange(start, len(lons))
# x_seg, y_seg = transformer.transform(lons[seg_idx], lats[seg_idx])
# vx_seg, vy_seg = segment_velocity(t[seg_idx], x_seg, y_seg)

# x[seg_idx] = x_seg
# y[seg_idx] = y_seg
# vx[seg_idx] = vx_seg
# vy[seg_idx] = vy_seg


# plt.figure()
# plt.plot(t, vx, label="Vx (m/s)")
# plt.plot(t, vy, label="Vy (m/s)")
# plt.grid(True)
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.legend()
# plt.title("Projected Velocity Components")
# plt.show()