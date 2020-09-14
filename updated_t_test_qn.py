#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 18:02:32 2020
This has been done with Ondemand interactive interface (Spyder4).
Script will be uploaded on Github.
@author: Qingc
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:14:19 2020

@author: vicki
"""

# Python Packages
import os
from datetime import datetime, timedelta
import warnings
import pickle
import pandas as pd
# Installed packages
import netCDF4
from netCDF4 import Dataset,num2date,date2num
import numpy as np
#import cmoceanc

from glob import glob
from datetime import datetime
from scipy.interpolate import interp1d
from matplotlib import rc
from matplotlib import dates as mpdates
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from scipy.stats import wilcoxon
from scipy import stats

FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 16
LABELSIZE = 18
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
params = {'legend.fontsize': 10,
          'legend.handlelength': 3}

#%%
# ---------------- #
# Data directories #
# ---------------- #
# home directoryc
home = os.path.expanduser("/scratch/qingyuwang")
# main data directory
BUL = os.path.join(home, "RamanLidar")
radiosonde = os.path.join(home,'RamanLidar','Radiosonde')
# piclke file data (same directory as this file)
#pkl = os.path.join(os.getcwd(), "pkl")
pkl = os.path.join(BUL, "pkl")
# figure save directory
figpath = os.path.join(BUL, "Figures1", "radiosondes")
if not os.path.exists(figpath):
    os.mkdir(figpath)
# close all figures
plt.close("all")
#%%
# -------------------------------------c--------------------- #
# Load 3 pickle files as dictionaries with time/height grids #
# ---------------------------------------------------------- #
# define reading function
def read_pickle(filepath):
    print(f"Reading file: {filepath.split(os.sep)[-1]}")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

# load data
f_AERI = os.path.join(pkl, "aeri.pickle")
data_a = read_pickle(f_AERI)
f_AERIrLID = os.path.join(pkl, "aeri_rlid.pickle")
data_ar = read_pickle(f_AERIrLID)
f_AERIvDIAL = os.path.join(pkl, "aeri_vdial.pickle")
data_av = read_pickle(f_AERIvDIAL)

# ----------------------------------- #
# Loop through all sondes and compare #
# ----------------------------------- #
f_sond = glob(os.path.join(BUL,"Radiosonde", "*sonde*"))
# sort by date and time on filename
dt_sond = [datetime.strptime("".join(ff.split(os.sep)[-1].split(".")[-3:-1]), 
    "%Y%m%d%H%M%S") for ff in f_sond]
i_sond = np.argsort(dt_sond)
# %%
# initialize lists for T, Td, alt, time
T_all = []
w_all = []
alt_all = []
t_all = []
# define some datetime object constants
dt_0 = datetime(2017, 5, 16)
delta = timedelta(seconds=1)

# define function to convert Td and p into wvmr
def dewpoint_to_wvmr(Td, p):
    '''
    input: Td in C, p in hPa
    output: wvmr in g/kg
    use following eqns
    e = 6.112 * exp(17.67*Td / (Td + 243.5))
    w = 0.622 * e / (p - e)
    '''
    e = 6.112 * np.exp(17.67 * Td / (Td + 243.5))
    w = 0.622 * e / (p - e)
    return 1000. * w

# define function to locate nearest aeri point
def find_nearest_timestep(t_comp, t_ref):
    '''
    input: t_comp = single timestep to compare against all reference times
    input: t_ref = array of timesteps to search through
    output: index of reference timestep in t_ref closest to t_comp
    output: how close (in same units of time) to this point
    '''
    return np.argmin(np.abs([(t_comp - it) for it in t_ref]))

# begin loop to load data
for i, f in enumerate(np.array(f_sond)[i_sond]):
    # read
    print(f"Reading file: {f.split(os.sep)[-1]}")
    df = netCDF4.Dataset(f, "r")
    # only grab data from below 4 km to compare
    i_4km = np.where(df.variables["alt"][:].data < 4000.)[0]
    # append to lists
    T_all.append(df.variables["tdry"][i_4km].data)
    alt_all.append(df.variables["alt"][i_4km].data)
    # calculate wvmr from Td and p
    w = dewpoint_to_wvmr(df.variables["dp"][i_4km].data, 
                         df.variables["pres"][i_4km].data)
    w_all.append(w)
    # convert time data to be consistent notation as AERI
    # each file records time as seconds since 0Z on same day
    # create datetime objects for each timestep
    dt_this = np.array(dt_sond)[i_sond][i]
    dt_base = datetime(dt_this.year, dt_this.month, dt_this.day)
    t = df.variables["time"][i_4km].data
    dt_new = np.array([(dt_base + (n * delta)) for n in t])
    # now convert to hours since first day at 0Z to be same as AERI
    t_new = np.array([((it - dt_0).total_seconds()/3600.) for it in dt_new])
    # all done; can append to t_all
    t_all.append(t_new)
    # close file
    df.close()
#%% This block is slow
# define new altitude grid in m
# convert from AGL to MSL for comparison with sondes
z_a = (data_a["height"].data * 1000.) + 237.43
nz = len(z_a)
# initialize arrays to fill with gridded interpolated data
n_sond = len(f_sond)
T_grid = np.full((n_sond, nz), np.nan, dtype=float)
w_grid = np.full((n_sond, nz), np.nan, dtype=float)
t_grid = np.full((n_sond, nz), np.nan, dtype=float)
# initialize arrays to store closest AERI data
# aeri
T_close_a = np.full((n_sond, nz), np.nan, dtype=float)
w_close_a = np.full((n_sond, nz), np.nan, dtype=float)
# aeri + raman
T_close_ar = np.full((n_sond, nz), np.nan, dtype=float)
w_close_ar = np.full((n_sond, nz), np.nan, dtype=float)
# aeri + wv dial
T_close_av = np.full((n_sond, nz), np.nan, dtype=float)
w_close_av = np.full((n_sond, nz), np.nan, dtype=float)
# begin loop to calculate comparisons
for i in range(n_sond):
    # want to get radiosonde data in same gridded format as AERI
    # interpolate to z_a
    # temperature
    fT = interp1d(alt_all[i], T_all[i], fill_value=np.nan, bounds_error=False)
    fnewT = fT(z_a)
    T_grid[i, :] = fnewT
    # wvmr
    fw = interp1d(alt_all[i], w_all[i], fill_value=np.nan, bounds_error=False)
    fneww = fw(z_a)
    w_grid[i, :] = fneww
    # time
    ft = interp1d(alt_all[i], t_all[i], fill_value=np.nan, bounds_error=False)
    fnewt = ft(z_a)
    t_grid[i, :] = fnewt
    # now can loop through each gridded timestep to compare with AERIoe
    for j, jt in enumerate(fnewt):
        i_close = find_nearest_timestep(jt, data_a["hours"])
        T_close_a[i, j] = data_a["temperature"][i_close, j]
        w_close_a[i, j] = data_a["wvmr"][i_close, j]
        T_close_ar[i, j] = data_ar["temperature"][i_close, j]
        w_close_ar[i, j] = data_ar["wvmr"][i_close, j]
        T_close_av[i, j] = data_av["temperature"][i_close, j]
        w_close_av[i, j] = data_av["wvmr"][i_close, j]

    # # better way: find closest timestep of lowest available data point
    # # and find out how close to the next timestep in indices, then can
    # # more efficiently choose timestamps to compare
    # # first point always at iz = 7
    # i_close, t_close = find_nearest_timestep(fnewt[7], data_a["hours"])
    # # find time halfway between i_close and i_close+1
    # t_mid = np.mean(data_a["hours"][i_close:i_close+2])
#%%
# calculate differences and statistics
def RMSD(obs, ref):
    return np.sqrt(np.nanmean((obs - ref)**2.))
# AERIonly
# temperature
T_rmsd_a = RMSD(T_grid, T_close_a)
T_diff_a = T_grid - T_close_a
T_med_a = np.nanmedian(T_diff_a, axis=0)
T_q1_a = np.nanpercentile(T_diff_a, 25., axis=0)
T_q3_a = np.nanpercentile(T_diff_a, 75., axis=0)
# wvmr
w_rmsd_a = RMSD(w_grid, w_close_a)
w_diff_a = w_grid - w_close_a
w_med_a = np.nanmedian(w_diff_a, axis=0)
w_q1_a = np.nanpercentile(w_diff_a, 25., axis=0)
w_q3_a = np.nanpercentile(w_diff_a, 75., axis=0)
# AERI + raman
# temperature
T_rmsd_ar = RMSD(T_grid, T_close_ar)
T_diff_ar = T_grid - T_close_ar
T_med_ar = np.nanmedian(T_diff_ar, axis=0)
T_q1_ar = np.nanpercentile(T_diff_ar, 25., axis=0)
T_q3_ar = np.nanpercentile(T_diff_ar, 75., axis=0)
# wvmr
w_rmsd_ar = RMSD(w_grid, w_close_ar)
w_diff_ar = w_grid - w_close_ar
w_med_ar = np.nanmedian(w_diff_ar, axis=0)
w_q1_ar = np.nanpercentile(w_diff_ar, 25., axis=0)
w_q3_ar = np.nanpercentile(w_diff_ar, 75., axis=0)
# AERI + wv dial
# temperature
T_rmsd_av = RMSD(T_grid, T_close_av)
T_diff_av = T_grid - T_close_av
T_med_av = np.nanmedian(T_diff_av, axis=0)
T_q1_av = np.nanpercentile(T_diff_av, 25., axis=0)
T_q3_av = np.nanpercentile(T_diff_av, 75., axis=0)
# wvmr
w_rmsd_av = RMSD(w_grid, w_close_av)
w_diff_av = w_grid - w_close_av
w_med_av = np.nanmedian(w_diff_av, axis=0)
w_q1_av = np.nanpercentile(w_diff_av, 25., axis=0)
w_q3_av = np.nanpercentile(w_diff_av, 75., axis=0)

# calculate 2d histogram bins and edges
# temperature
T_bins = (np.arange(-10., 35.5, 0.5), np.arange(-10., 35.5, 0.5))
H_a, xe_a, ye_a = np.histogram2d(np.ravel(T_close_a),
                                 np.ravel(T_grid),
                                 bins=T_bins,
                                 density=True)
xaT, yaT = np.meshgrid(xe_a, ye_a)
H_ar, xe_ar, ye_ar = np.histogram2d(np.ravel(T_close_ar),
                                    np.ravel(T_grid),
                                    bins=T_bins,
                                    density=True)
xarT, yarT = np.meshgrid(xe_ar, ye_ar)
H_av, xe_av, ye_av = np.histogram2d(np.ravel(T_close_av),
                                    np.ravel(T_grid),
                                    bins=T_bins,
                                    density=True)
xavT, yavT = np.meshgrid(xe_av, ye_av)
# wvmr
w_bins = (np.arange(0., 18.5, 0.5), np.arange(0., 18.5, 0.5))
W_a, xew_a, yew_a = np.histogram2d(np.ravel(w_close_a),
                                   np.ravel(w_grid),
                                   bins=w_bins,
                                   density=True)
xaw, yaw = np.meshgrid(xew_a, yew_a)

W_ar, xew_ar, yew_ar = np.histogram2d(np.ravel(w_close_ar),
                                      np.ravel(w_grid),
                                      bins=w_bins,
                                      density=True)
xarw, yarw = np.meshgrid(xew_ar, yew_ar)
W_av, xew_av, yew_av = np.histogram2d(np.ravel(w_close_av),
                                      np.ravel(w_grid),
                                      bins=w_bins,
                                      density=True)
xavw, yavw = np.meshgrid(xew_av, yew_av)


#%% When change option, re-run this block at first
## Find continuous cloudy/cloud-free times indexc
#flist_lidar = sorted(glob('sgprlprofmr2news10mC1.c0.*.000500.nc'))c
flist_lidar = sorted(glob('/scratch/qingyuwang/RamanLidar/RamanLidar/*'))

cbh_list = []
time_list = []
seconds_since_epoch = []
for f in flist_lidar:
#    print(i)
    fid = Dataset(f)
    cbh = fid['cbh'][:]
    base_time = fid['base_time'][:]
    time_offset = fid['time_offset'][:]
#    seconds_since_epoch = []
    time_datetime  = []
    for i in time_offset.data.tolist():
        time_datetime.append(datetime(year=1970,month=1,day=1)+timedelta(seconds=base_time.data.tolist())+timedelta(seconds=i))
    fid.close()
    seconds_since_epoch.append(base_time.data+time_offset.data)
    cbh_list.append(cbh)
    time_list.append(time_datetime)
    
#%
x,y = np.shape(cbh_list)   
cbh_list = np.array(cbh_list).reshape(x*y)
time_list = np.reshape(np.array(time_list),(x*y))
seconds_since_epoch = np.reshape(np.array(seconds_since_epoch),(x*y))

cloud_free_inds = np.where(np.array(cbh_list)==-1.)
cloud_free = np.array(cbh_list)[cloud_free_inds]
cloud_free_times_v = np.array(seconds_since_epoch)[cloud_free_inds]
cloudy_inds = np.where(np.array(cbh_list)!=-1.)
cloudy = np.array(cbh_list)[cloudy_inds]
cloudy_times_v = np.array(seconds_since_epoch)[cloudy_inds]
cloudy_times_dt_v = np.array([datetime(year=1970,month=1,day=1)+timedelta(seconds=i) for i in cloudy_times_v])
inds_cf = np.where((cloudy_times_dt_v[1:] - cloudy_times_dt_v[:-1])>=timedelta(hours=2))[0]
cloudf_cont_start = cloudy_times_dt_v[inds_cf]
cloudf_cont_end = cloudy_times_dt_v[inds_cf+1]


cloud_free_times_dt_v = np.array([datetime(year=1970,month=1,day=1)+timedelta(seconds=i) for i in cloud_free_times_v])
inds_c = np.where((cloud_free_times_dt_v[1:] - cloud_free_times_dt_v[:-1])>=timedelta(hours=2))[0]
cloud_cont_start = cloud_free_times_dt_v[inds_c]
cloud_cont_end = cloud_free_times_dt_v[inds_c+1]

# %%
options = ['a','av','ar']
option = 'ar'
# Convert to 1-D arrays
lenx,leny = np.shape(t_grid)
time_both = t_grid.reshape(lenx*leny)
if option == 'a':
    T_diff = (T_close_a-T_grid).reshape(lenx*leny)
    w_diff = (w_close_a-w_grid).reshape(lenx*leny)
    text_opt = "only"
    T_diff_A = T_diff
    w_diff_A = w_diff
elif option == 'av':
    T_diff = (T_close_av-T_grid).reshape(lenx*leny)
    w_diff = (w_close_av-w_grid).reshape(lenx*leny)
    text_opt = "vDial"
    T_diff_AV = T_diff
    w_diff_AV = w_diff
elif option == 'ar':
    T_diff = (T_close_ar-T_grid).reshape(lenx*leny)
    w_diff = (w_close_ar-w_grid).reshape(lenx*leny)
    text_opt = "rLID"
    T_diff_AR = T_diff
    w_diff_AR = w_diff
else:
    print('Error!')
#T_diff = (T_close_ar-T_grid).reshape(lenx*leny)
#T_diff = (T_close_a-T_grid).reshape(lenx*leny)
#w_diff = (w_close_ar-w_grid).reshape(lenx*leny)
#w_diff = (w_close_a-w_grid).reshape(lenx*leny)
z = (np.meshgrid(z_a,range(109))[0]).reshape(lenx*leny)/1000

# convert hours since the first day to datetime
time_datetime = []
for t in time_both:
    if np.isnan(t):
        time_datetime.append(datetime(year=1970,month=1,day=1))
    else:
        time_datetime.append(dt_0+timedelta(hours=t))
time_datetime = np.array(time_datetime)
# Pick cloudy/cloud-free times

inds_cloud_cont = []
for i in range(len(inds_c)):
    inds_cloud_cont_l = np.where((time_datetime>=cloud_cont_start[i])&(time_datetime<=cloud_cont_end[i]))[0]
    inds_cloud_cont.append(inds_cloud_cont_l)
inds_cloudf_cont = []
for i in range(len(inds_cf)):
    inds_cloudf_cont_l = np.where((time_datetime>=cloudf_cont_start[i])&(time_datetime<=cloudf_cont_end[i]))[0]
    inds_cloudf_cont.append(inds_cloudf_cont_l)
inds_c = inds_cloud_cont[0].tolist()
for ii in range(1,len(inds_cloud_cont)):
    inds_c += inds_cloud_cont[ii].tolist() 
inds_cf = inds_cloudf_cont[0].tolist()
for ii in range(1,len(inds_cloudf_cont)):
    inds_cf += inds_cloudf_cont[ii].tolist()
    
    
    
    
# %%
height = np.unique(z)
height_cloudf = [np.nan]*39
height_cloud = [np.nan]*39

wv_height_cloudf = [np.nan]*39
wv_height_cloud = [np.nan]*39
q1_cloud = [np.nan]*39
q2_cloud = [np.nan]*39
q3_cloud = [np.nan]*39
q1_cloudf = [np.nan]*39
q2_cloudf = [np.nan]*39
q3_cloudf = [np.nan]*39

wv_q1_cloud = [np.nan]*39
wv_q2_cloud = [np.nan]*39
wv_q3_cloud = [np.nan]*39
wv_q1_cloudf = [np.nan]*39
wv_q2_cloudf = [np.nan]*39
wv_q3_cloudf = [np.nan]*39

for i in range(39):
    height_cloudf[i] = T_diff[np.where(z[inds_cf] == z[inds_cf][i])[0]]
    height_cloud[i] = T_diff[np.where(z[inds_c] == z[inds_c][i])[0]]
    wv_height_cloudf[i] = w_diff[np.where(z[inds_cf] == z[inds_cf][i])[0]]
    wv_height_cloud[i] = w_diff[np.where(z[inds_c] == z[inds_c][i])[0]]

height_cloud_array  = np.array(height_cloud).reshape(39)
height_cloudf_array  = np.array(height_cloudf).reshape(39)
wv_height_cloud_array  = np.array(wv_height_cloud).reshape(39)
wv_height_cloudf_array  = np.array(wv_height_cloudf).reshape(39)
for i in range(39):
    q1_cloud[i] = np.nanquantile(height_cloud_array[i],0.25)
    q2_cloud[i] = np.nanquantile(height_cloud_array[i],0.5)
    q3_cloud[i] = np.nanquantile(height_cloud_array[i],0.75)

    wv_q1_cloud[i] = np.nanquantile(wv_height_cloud_array[i],0.25)
    wv_q2_cloud[i] = np.nanquantile(wv_height_cloud_array[i],0.5)
    wv_q3_cloud[i] = np.nanquantile(wv_height_cloud_array[i],0.75)
    
for i in range(39):
    q1_cloudf[i] = np.nanquantile(height_cloudf_array[i],0.25)
    q2_cloudf[i] = np.nanquantile(height_cloudf_array[i],0.5)
    q3_cloudf[i] = np.nanquantile(height_cloudf_array[i],0.75)

    wv_q1_cloudf[i] = np.nanquantile(wv_height_cloudf_array[i],0.25)
    wv_q2_cloudf[i] = np.nanquantile(wv_height_cloudf_array[i],0.5)
    wv_q3_cloudf[i] = np.nanquantile(wv_height_cloudf_array[i],0.75)
#%%   
plt.plot(q1_cloud,height,'r-')
plt.plot(q2_cloud,height,'b')
plt.plot(q3_cloud,height,'k-')

#plt.figure()
plt.plot(q1_cloudf,height,'r:')
plt.plot(q2_cloudf,height,'b:')
plt.plot(q3_cloudf,height,'k:')

#%%
print("Begin plotting...")
#rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
#rc('text',usetex='True')
z_a_agl = z_a - 237.43
colors = [(0., 0., 0.), (0./255, 114./255., 178./255), (213./255, 94./255, 0.)]
#%%
#fig,axes = plt.subplots(1,2,figsize=(10,4))
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8))
# temperature
# AERI only
#ax1[0].plot(T_med_a, z_a_agl, color=colors[0], linestyle="-", linewidth=3., 
#label="AERIonly")
# ax1[0].fill_betweenx(z_a_agl, T_q1_a, T_q3_a, alpha=0.3, color=colors[0])
#ax1[0].plot(T_q1_a, z_a_agl, T_q3_a, z_a_agl, color=colors[0], linestyle=":", linewidth=2)
# AERI + Raman
#ax1[0].plot(T_med_ar, z_a_agl, color=colors[1], linestyle="-", linewidth=3., 
#label="AERIrLID")
# ax1[0].fill_betweenx(z_a_agl, T_q1_ar, T_q3_ar, alpha=0.3, color=colors[1])
#ax1[0].plot(T_q1_ar, z_a_agl, T_q3_ar, z_a_agl, color=colors[1], linestyle=":", linewidth=2)
# AERI + wv dial
ax1[0].plot(q2_cloud, z_a_agl, color=colors[2], linestyle="-", linewidth=3.,label="mean cloudy")
# ax1[0].fill_betweenx(z_a_agl, T_q1_av, T_q3_av, alpha=0.3, color=colors[2])
#ax1[0].plot(q1_cloud, z_a_agl, q3_cloud, z_a_agl, color=colors[2], linestyle=":", linewidth=2)
# setup

#ax1[0].plot(q2_cloudf, z_a_agl, color=colors[1], linestyle="-", linewidth=3.,label = 'mean cloud free')
# ax1[1].fill_betweenx(z_a_agl, w_q1_av, w_q3_av, alpha=0.3, color=colors[2])
ax1[0].plot(q1_cloud, z_a_agl, color=colors[2], linestyle=":", linewidth=2)#,label="1st quartile cloud free")
ax1[0].plot(q3_cloud, z_a_agl, color=colors[2], linestyle=":", linewidth=3)#,label="3st quartile cloud free")

ax1[0].axvline(0., linewidth=2., color="k", linestyle="--")
ax1[0].grid()
#ax1[0].legend(loc=(0.01, 0.75), fontsize=16)
ax1[0].set_xlabel("$T_{sonde} - T_{AERI"+text_opt+"}$ [$^\circ$C]")
ax1[0].set_ylabel("Altitude [m AGL]")
ax1[0].set_xlim([-2., 2.])
ax1[0].set_ylim([0., 4000.])
ax1[0].xaxis.set_minor_locator(MultipleLocator(0.25))
ax1[0].yaxis.set_minor_locator(MultipleLocator(250))
props=dict(boxstyle='square',facecolor='white',alpha=0.5)
#ax1[0].text(0.03,0.95,r'Temperature',fontsize=14,bbox=props, transform=ax1[0].transAxes)
# wvmr
# AERI only
#ax1[1].plot(w_med_a, z_a_agl, color=colors[0], linestyle="-", linewidth=3.)
# ax1[1].fill_betweenx(z_a_agl, w_q1_a, w_q3_a, alpha=0.3, color=colors[0])
#ax1[1].plot(w_q1_a, z_a_agl, w_q3_a, z_a_agl, color=colors[0], linestyle=":", linewidth=2)
# AERI + Raman
#ax1[1].plot(w_med_ar, z_a_agl, color=colors[1], linestyle="-", linewidth=3.)
# ax1[1].fill_betweenx(z_a_agl, w_q1_ar, w_q3_ar, alpha=0.3, color=colors[1])
#ax1[1].plot(w_q1_ar, z_a_agl, w_q3_ar, z_a_agl, color=colors[1], linestyle=":", linewidth=2)
#%
# AERI + wv dial
ax1[0].plot(q2_cloudf, z_a_agl, color=colors[1], linestyle="-", linewidth=3.,label = 'mean cloud free')
# ax1[1].fill_betweenx(z_a_agl, w_q1_av, w_q3_av, alpha=0.3, color=colors[2])
ax1[0].plot(q1_cloudf, z_a_agl, color=colors[1], linestyle=":", linewidth=2)#,label="1st quartile cloud free")
ax1[0].plot(q3_cloudf, z_a_agl, color=colors[1], linestyle=":", linewidth=3)#,label="3st quartile cloud free")

ax1[0].legend(loc='upper left')
# setup
ax1[0].axvline(0., linewidth=2., color="k", linestyle="--")
ax1[0].grid()
#ax1[1].set_xlabel("$W_{sonde} - W_{AERIRaman}$ [$^\circ$C]")
#ax1[1].set_xlabel("$WVMR_{sonde} - WVMR_{AERIoe}$ [g kg$^{-1}$]")

ax1[0].xaxis.set_minor_locator(MultipleLocator(0.25))
ax1[0].yaxis.set_minor_locator(MultipleLocator(250))
#ax1[0].text(0.03,0.95,r'cloud_free',fontsize=20,bbox=props, transform=ax1[1].transAxes)
#fig1.tight_layout()
#fsave1 = "diff_vs_alt_T_wvmr"
#fig1.savefig(f"{os.path.join(figpath, fsave1)}.pdf", dpi=300, fmt="pdf")
#fig1.savefig(f"{os.path.join(figpath, fsave1)}.png", dpi=300, fmt="png")
#plt.close(fig1)
#plt.show()

# Humidity
ax1[1].plot(wv_q2_cloud, z_a_agl, color=colors[2], linestyle="-", linewidth=3.,label="mean cloudy")
ax1[1].plot(wv_q1_cloud, z_a_agl, color=colors[2], linestyle=":", linewidth=2)#,label="1st quartile cloud free")
ax1[1].plot(wv_q3_cloud, z_a_agl, color=colors[2], linestyle=":", linewidth=3)#,label="3st quartile cloud free")

ax1[1].axvline(0., linewidth=2., color="k", linestyle="--")
ax1[1].grid()
#ax1[0].legend(loc=(0.01, 0.75), fontsize=16)

#ax1[1].set_ylabel("Altitude [m AGL]")
ax1[1].set_xlim([-2., 2.])
ax1[1].set_ylim([0., 4000.])
ax1[1].xaxis.set_minor_locator(MultipleLocator(0.25))
ax1[1].yaxis.set_minor_locator(MultipleLocator(250))
props=dict(boxstyle='square',facecolor='white',alpha=0.5)
#ax1[1].text(0.03,0.95,r'Humidity',fontsize=14,bbox=props, transform=ax1[1].transAxes)

ax1[1].plot(wv_q2_cloudf, z_a_agl, color=colors[1], linestyle="-", linewidth=3.,label = 'mean cloud free')
# ax1[1].fill_betweenx(z_a_agl, w_q1_av, w_q3_av, alpha=0.3, color=colors[2])
ax1[1].plot(wv_q1_cloudf, z_a_agl, color=colors[1], linestyle=":", linewidth=2)#,label="1st quartile cloud free")
ax1[1].plot(wv_q3_cloudf, z_a_agl, color=colors[1], linestyle=":", linewidth=3)#,label="3st quartile cloud free")

ax1[1].legend(loc='upper left')
# setup
ax1[1].axvline(0., linewidth=2., color="k", linestyle="--")
ax1[1].grid()
ax1[1].set_xlabel("$WVMR_{sonde} - WVMR_{AERI"+text_opt+"}$ [g kg$^{-1}$]")
#ax1[1].set_xlabel("$WVMR_{sonde} - WVMR_{AERIoe}$ [g kg$^{-1}$]")

ax1[1].xaxis.set_minor_locator(MultipleLocator(0.25))
ax1[1].yaxis.set_minor_locator(MultipleLocator(250))
plt.show()
#fig1.savefig(figpath+'/T_W_difference_c_cf_'+option+'.pdf')
    #%%
    
    
    
## -----------------------------Plot-------------------------

fig,axes = plt.subplots(1,2,figsize=(10,4))
H_bins_num = 10
T_bins_array = np.arange(np.min(T_diff[inds_c]),np.max(T_diff[inds_c]),2)
#plt.hist(Temp_aeri_l_cf1d,bins=np.linspace(np.min(Temp_aeri_l_cf1d),np.max(Temp_aeri_l_cf1d),3))
ax0 = axes[0]
p1=ax0.hist2d(T_diff[inds_c],z[inds_c],bins = [T_bins_array,H_bins_num],cmap='Reds',vmin=0,vmax=300)
ax0.set_xlim([-10,5])
ax0.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax0.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax0.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax0.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax0.tick_params(axis='both',which='major',labelsize=12,direction='in')
ax0.tick_params(axis='both',which='minor',direction='in')
ax1 = axes[1]
p2=ax1.hist2d(T_diff[inds_cf],z[inds_cf],bins = [T_bins_array,H_bins_num],cmap='Reds',vmin=0,vmax=300)
ax1.set_xlim([-10,5])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.tick_params(axis='both',which='major',labelsize=12,direction='in')
ax1.tick_params(axis='both',which='minor',direction='in')
plt.colorbar(p1[3],ax=axes,orientation="vertical")
fig.text(0.45,0.03,r'$\Delta$'+'T ($^\circ$C)',horizontalalignment='center',verticalalignment='center',fontweight='bold',fontsize=14)
fig.text(0.08,0.5,'Height (km)',horizontalalignment='center',verticalalignment='center',fontweight='bold',fontsize=14,rotation=90)
fig.text(0.13,0.85, 'Cloudy',horizontalalignment='left',verticalalignment='center',fontweight='bold',fontsize=12)
fig.text(0.47,0.85, 'Cloud-free',horizontalalignment='left',verticalalignment='center',fontweight='bold',fontsize=12)
#plt.colorbar(p2[3],ax=ax1)
#fig.tight_layout()
fig.savefig(figpath+'diff_aeri_'+text_opt+'_radio_T.png')


fig,axes = plt.subplots(1,2,figsize=(10,4))
H_bins_num = 10
T_bins_array = np.arange(np.min(T_diff[inds_c]),np.max(T_diff[inds_c]),2)
#plt.hist(Temp_aeri_l_cf1d,bins=np.linspace(np.min(Temp_aeri_l_cf1d),np.max(Temp_aeri_l_cf1d),3))
ax0 = axes[0]
p1=ax0.hist2d(w_diff[inds_c],z[inds_c],bins = [T_bins_array,H_bins_num],cmap='Greens',vmin=0,vmax=300)
ax0.set_xlim([-10,5])
ax0.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax0.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax0.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax0.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax0.tick_params(axis='both',which='major',labelsize=12,direction='in')
ax0.tick_params(axis='both',which='minor',direction='in')
ax1 = axes[1]
p2=ax1.hist2d(w_diff[inds_cf],z[inds_cf],bins = [T_bins_array,H_bins_num],cmap='Greens',vmin=0,vmax=300)
ax1.set_xlim([-10,5])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.tick_params(axis='both',which='major',labelsize=12,direction='in')
ax1.tick_params(axis='both',which='minor',direction='in')
plt.colorbar(p1[3],ax=axes,orientation="vertical")
fig.text(0.45,0.03,r'$\Delta$'+'q (g/kg)',horizontalalignment='center',verticalalignment='center',fontweight='bold',fontsize=14)
fig.text(0.08,0.5,'Height (km)',horizontalalignment='center',verticalalignment='center',fontweight='bold',fontsize=14,rotation=90)
fig.text(0.13,0.85, 'Cloudy',horizontalalignment='left',verticalalignment='center',fontweight='bold',fontsize=12)
fig.text(0.47,0.85, 'Cloud-free',horizontalalignment='left',verticalalignment='center',fontweight='bold',fontsize=12)
#plt.colorbar(p2[3],ax=ax1)
#fig.tight_layout()
fig.savefig(figpath+'diff_aeri_'+text_opt+'_radio_W.png')


cl = list(T_diff[inds_c])
free = list(T_diff[inds_cf])
total = cl+free
sorted(total)
df = pd.DataFrame()
#df['rank'] = np.arange(2518)
df['values'] = total

#%%

summ=[]
sunn =[]
for ii in range(258):
    if df['values'][ii] in cl:
        summ.append(df['values'][ii])
        print('Cloud:',summ)
        
    else:
        sunn.append(df['rank'][ii])
#        print(sunn)
        
#%%
#import seaborn as sns
#
#green_diamond = dict(markerfacecolor='g', marker='D')
##ax4 = sns.boxplot(x = 'z',y = 't_diff',data = dff_height)
#fig4, ax4 = plt.subplots()
#plt.boxplot(dff_height,vert = False)


#dff_height.boxplot(by = 'z',vert=False,flierprops = green_diamond)
#plt.xlabel('difference')
#plt.ylabel('heihgt(m)')
#plt.title('vertical distribution')
#plt.xticks(height)
#ax4 = plt.gca()
#ax4.set_ylabel('height(m)'
#ax4.yaxis.set_ticks(height)

#ax4.yaxis.set_yticklabels(height)
#plt.yticks(height)
#ax4.xaxis.set_major_locator(ticker.MultipleLocator(5))
#ax4.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax4.tick_params(axis='both',which='major',labelsize=12,direction='in')
#ax4.tick_params(axis='both',which='minor',direction='in')
#ax4.set_title('difference vertical distrubution with Cloud')
#fig4.tight_layout()

#%% STatistical test: T-test for two independent 
#if option == 'a':
#
#elif option == 'av':
#
#elif option == 'ar':
#    T_diff = (T_close_ar-T_grid).reshape(lenx*leny)
#    w_diff = (w_close_ar-w_grid).reshape(lenx*leny)
#    text_opt = "rLID"
#    T_diff_AR = T_diff
#    w_diff_AR = w_diff
#else:
#    print('Error!')
#xx = [np.nan]*39
#yy = [np.nan]*39
#for i in range(39):
#    df1 = pd.DataFrame({'cloudf': height_cloud_array[i]})#
#    df2 = pd.DataFrame({'cloud': height_cloudf_array[i]})
#    data_0  = pd.concat([df1,df2],axis =1)
#    data_0_drop = data_0.dropna()
#    # Considering we have more than 30 samples: cloudy 3435, free 4344, assumued normalized already
#    # First we do the F test(for variance) and then do the t-test to check the miu
#    a_var = data_0['cloudf'].var()
#    b_var = data_0['cloud'].var()
#    # sample scale
#    n1 = data_0.count()[0]
#    n2 = data_0.count()[1]
#    # freedom
#    df_a =n1-1
#    df_b = n2-1
#    if a_var>b_var:
#        F = a_var/b_var
#    else:
#        F = b_var/a_var
##    print('A"s var: %f; B"s var: %f'%(a_var,b_var))
##    print('F value%.2f'%F)
#    p_value = stats.f.sf(F,df_b,df_a)
#    print('c for F test = %f'%p_value)
#    if p_value<.05:
#        print('significant differnt vars')
#    else:
#        print('the same vars')
#    xx[i],yy[i] = stats.ttest_ind(data_0['cloudf'], data_0['cloud'],nan_policy='omit')
#    print(stats.ttest_ind(data_0['cloudf'], data_0['cloud'],nan_policy='omit'))
#    # Ttest_indResult(statistic=0.1856747111626556, pvalue=0.8533476876884907)
#    # The p value is larger than 0.05, we could not deny the null hypothesis, therefore no siginificant difference
#    #stats.fligner(data_0['cloudf'],data_0['cloud'])
#    # 
    
# %%

#%% STatistical test: T-test for two independent 
#if option == 'a':
#    height_cloud_array_A = height_cloud_array
#elif option == 'av':
#    height_cloud_array_AV = height_cloud_array
#elif option == 'ar':
#    height_cloud_array_AR = height_cloud_array
#else:
#    print('Error!')
#%%

#height_cloud_array
xx = [np.nan]*39
yy = [np.nan]*39
wv_xx = [np.nan]*39
wv_yy = [np.nan]*39


for i in range(39):
    
    df1 = pd.DataFrame({'cloudf': height_cloud_array[i]})#
    df2 = pd.DataFrame({'cloud': height_cloudf_array[i]})
    wv_df1 = pd.DataFrame({'cloudf': wv_height_cloud_array[i]})#
    wv_df2 = pd.DataFrame({'cloud': wv_height_cloudf_array[i]})
    data_0  = pd.concat([df1,df2],axis =1)
    data_0_drop = data_0.dropna()
    wv_data_0  = pd.concat([wv_df1,wv_df2],axis =1)
    wv_data_0_drop = wv_data_0.dropna()
    # Considering we have more than 30 samples: cloudy 3435, free 4344, assumued normalized already
    # First we do the F test(for variance) and then do the t-test to check the miu
    a_var = data_0['cloudf'].var()
    wv_a_var = wv_data_0['cloudf'].var()
    b_var = data_0['cloud'].var()
    wv_b_var = wv_data_0['cloud'].var()
    # sample scale
    n1 = data_0.count()[0]
    n2 = data_0.count()[1]
    
    wv_n1 = wv_data_0.count()[0]
    wv_n2 = wv_data_0.count()[1]
    # freedom
    df_a = n1-1
    df_b = n2-1
    wv_df_a =wv_n1-1
    wv_df_b = wv_n2-1
    if a_var>b_var:
        F = a_var/b_var
    else:
        F = b_var/a_var
#    print('A"s var: %f; B"s var: %f'%(a_var,b_var))
#    print('F value%.2f'%F)
    if wv_a_var>wv_b_var:
        wv_F = wv_a_var/wv_b_var
    else:
        wv_F = wv_b_var/wv_a_var
    p_value = stats.f.sf(F,df_b,df_a)
    wv_p_value = stats.f.sf(wv_F,wv_df_b,wv_df_a)
#    print('p value for F test = %f'%p_value)
    if p_value<.05:
        print('significant temp different vars')
#    else:
#        print('the same temp vars')
        
        # Humidity
    if wv_p_value<.05:
        print('significant wvmr different vars')
#    else:
#        print('the same wvmr vars')
    xx[i],yy[i] = stats.ttest_ind(data_0['cloudf'], data_0['cloud'],nan_policy='omit')
    wv_xx[i],wv_yy[i] = stats.ttest_ind(wv_data_0['cloudf'], wv_data_0['cloud'],nan_policy='omit')
#    print(stats.ttest_ind(data_0['cloudf'], data_0['cloud'],nan_policy='omit'))
    # Ttest_indResult(statistic=0.1856747111626556, pvalue=0.8533476876884907)
    # The p value is larger than 0.05, we could not deny the null hypothesis, therefore no siginificant difference
    #stats.fligner(data_0['cloudf'],data_0['cloud'])
    #

#%% Save three results
if option == 'av':
    av_wv_xx = wv_xx
    av_wv_yy = wv_yy
    av_xx = xx
    av_yy = yy
#%%
if option == 'ar':
    ar_wv_xx = wv_xx
    ar_wv_yy = wv_yy
    ar_xx = xx
    ar_yy = yy
#%%
if option == 'a':
    a_xx = xx
    a_yy = yy
    a_wv_xx = wv_xx
    a_wv_yy = wv_yy
#%% Temp
fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 8))    
#fig1, ax1 = 
ax = ax1[0].twiny()
ax1[0].plot(av_xx, z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 't AERIvDial')
ax.plot(av_yy, z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'p AERIvDial')

axx =ax1[1].twiny()
ax1[1].plot(a_xx, z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 't AERIonly')
axx.plot(a_yy, z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'p AERIonly')

axxx = ax1[2].twiny()
ax1[2].plot(ar_xx, z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 't AERIrLID')
axxx.plot(ar_yy, z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'p AERIrLID')

ax.grid()
axx.grid()
axxx.grid()
#ax.grid()
ax1[1].set_xlabel("t value from student t test",color = colors[2])
axx.set_xlabel("p value from student t test",color = colors[1])
#ax1[1].set_xlim([-2., 2.])
#ax1.#plt.xlable('')
# yy, z_a_agl, 
ax1[0].set_ylabel("meter")
ax1[0].set_ylabel("Altitude [m AGL]")
ax1[0].legend(loc=(0.01, 0.75), fontsize=10)
ax.legend(loc=(0.01, 0.15), fontsize=10)
ax1[1].text(0.03,0.97,r'Temperature',fontsize=10,bbox=props, transform=ax1[1].transAxes)
#ax.set_xscale("log", nonposx='clip')
#ax1[].set_yscale("log", nonposy='clip')
axx.legend(loc = 'upper right',fontsize=9)
ax1[1].legend(fontsize=10)

axxx.legend(loc = 0,fontsize=10)
ax1[2].legend(loc = 'lower left',fontsize=10)
#ax1.yaxis.set_minor_locator(MultipleLocator(400))
#ax.yaxis.set_minor_locator(MultipleLocator(500))
c = 'r'
ax.axvline(x=0.05,c=c)
axx.axvline(x=0.05,c=c)
axxx.axvline(x=0.05,c=c)
#fig1.savefig(figpath+'/temp_t_p_difference.pdf')
#%% WVMR
diff_wv_a = np.where(np.asarray(a_wv_yy)<0.1)[0][0]c
diff_wv_ar = np.where(np.asarray(ar_wv_yy)<0.1)[0][0]
#diff_wv_av = np.where(np.array(av_wv_yy)<0.1)[0][0]
#diff_a = np.where(np.asarray(a_yy)<0.1)[0][0]
#diff_av = np.where(np.asarray(av_yy)<0.1)[0][0]
#diff_ar = np.where(np.asarray(ar_yy)<0.1)[0][0]



#npwhere(av_wv_yy<0.05)
#%%
fig = plt.figure(figsize=(5, 8))
ax = plt.gca()
ax.plot(np.abs(av_wv_xx), z_a_agl,color=colors[0], linestyle="-", linewidth=2,label = 'wvmr AERIvDial')
ax.plot(np.abs(a_wv_xx), z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'wvmr AERIonly')
ax.plot(np.abs(ar_wv_xx),z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 'wvmr AERIrLID')

ax.scatter(np.abs(ar_wv_xx[diff_wv_ar]), z_a_agl[diff_wv_ar],color='red',s = 130, marker = '^',label = 'P value < 0.1')
ax.scatter(np.abs(a_wv_xx[diff_wv_a]), z_a_agl[diff_wv_a],color='red',s = 130, marker = '^')
ax.set_ylabel("Altitude [m AGL]")
ax.axvline(x=1, c='blue',linestyle='--')
#ax.axvline(x=-1,  c='blue',linestyle = '--')
#ax.set_yscale('log')
#plt.legend(borderpad = 0.2,framealpha=0.3     )
plt.tight_layout()
fig.savefig(figpath+'/wvmr_t_p_difference_height_no_legend.pdf')
ax.set_xlabel("T value")
#%%
fig = plt.figure(figsize=(5, 8))
ax = plt.gca()
ax.plot(av_wv_xx, z_a_agl,color=colors[0], linestyle="-", linewidth=2,label = 'wvmr AERIvDial')
ax.plot(a_wv_xx, z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'wvmr AERIonly')
ax.plot(ar_wv_xx,z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 'wvmr AERIrLID')

ax.scatter(ar_wv_xx[diff_wv_ar], z_a_agl[diff_wv_ar],color='red',s = 130, marker = '^',label = 'P value < 0.1')
ax.scatter(a_wv_xx[diff_wv_a], z_a_agl[diff_wv_a],color='red',s = 130, marker = '^')

ax.set_ylabel("Altitude [m AGL]")
ax.set_xlabel("T value from student t test")

ax.axvline(x=1, c='blue',linestyle='--')
ax.axvline(x=-1,  c='blue',linestyle = '--')
ax.set_yscale('log')
plt.legend(borderpad = 0.2,framealpha=0.3
           )
plt.tight_layout()
#fig.savefig(figpath+'/wvmr_t_p_difference_.pdf')
#%%
fig = plt.figure(figsize=(5, 8))
ax = plt.gca()
ax.plot(av_xx, z_a_agl,color=colors[0], linestyle="-", linewidth=2,label = 'temp AERIvDial')
ax.plot(a_xx, z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'temp AERIonly')
ax.plot(ar_xx,z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 'temp AERIrLID')
ax.scatter(np.nan, np.nan,color='red',s = 130, marker = '^')
#ax.scatter(ar_xx[diff_ar], z_a_agl[diff_wv_ar],color='red',s = 130, marker = '^',label = 'P value < 0.1')
#ax.scatter(a_wv_xx[diff_wv_a], z_a_agl[diff_wv_a],color='red',s = 130, marker = '^')

ax.set_ylabel("Altitude [m AGL]")
ax.set_xlabel("T value from student t test")

ax.axvline(x=1, c='blue',linestyle='--')
ax.axvline(x=-1,  c='blue',linestyle = '--')
ax.set_yscale('log')
#ax.legend()
plt.legend(borderpad = 0.2,framealpha=0.3)
plt.tight_layout()
#fig.savefig(figpath+'/temp_t_p_difference_log.pdf')
#%%

fig = plt.figure(figsize=(5, 8))
ax = plt.gca()
ax.plot(np.abs(av_xx), z_a_agl,color=colors[0], linestyle="-", linewidth=2,label = 'temp AERIvDial')
ax.plot(np.abs(a_xx), z_a_agl,color=colors[1], linestyle="-", linewidth=2,label = 'temp AERIonly')
ax.plot(np.abs(ar_xx),z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 'temp AERIrLID')
ax.scatter(np.nan, np.nan,color='red',s = 130, marker = '^')
#ax.scatter(ar_wv_xx[diff_wv_ar], z_a_agl[diff_wv_ar],color='red',s = 130, marker = '^',label = 'P value < 0.1')
#ax.scatter(a_wv_xx[diff_wv_a], z_a_agl[diff_wv_a],color='red',s = 130, marker = '^')

ax.set_ylabel("Altitude [m AGL]")
ax.set_xlabel("T value")
#ax.set_xlim(left = 0.001)
ax.axvline(x=1, c='blue',linestyle='--')
#ax.axvline(x=-1,  c='blue',linestyle = '--')
#ax.set_yscale('log')
#ax.legend()
plt.tight_layout()
#ax.legend(loc='upper left',borderpad = 0.2,framealpha=0.3, )
fig.savefig(figpath+'/temp_t_p_difference_height_no_legend.pdf')
#%%
fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 8))    
#fig1, ax1 = 
ax = ax1[0].twiny()
ax1[0].plot(av_wv_xx, z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 't AERIvDial')
#ax.scatter(av_wv_xx[diff_wv_av[0]], z_a_agl[diff_wv_av[0]],color=colors[1], cc= '*', label = 'p AERIvDial')

axx =ax1[1].twiny()
ax1[1].plot(a_wv_xx, z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 't AERIonly')
axx.plot(a_wv_yy[diff_wv_a], z_a_agl[diff_wv_a],color=colors[1], linestyle="-", linewidth=2,label = 'p AERIonly')


axxx = ax1[2].twiny()
ax1[2].plot(ar_wv_xx, z_a_agl,color=colors[2], linestyle="-", linewidth=2,label = 't AERIrLID')
axxx.scatter(ar_wv_xx[diff_wv_ar], z_a_agl[diff_wv_ar],color=colors[1],s = area, marker = '^',label = 'p < 0.1')

ax.grid()
axx.grid()

axxx.grid()
#ax.grid()
ax1[1].set_xlabel("t value from student t test",color = colors[2])
axx.set_xlabel("p value from student t test",color = colors[1])
#ax1[1].set_xlim([-2., 2.])
#ax1.#plt.xlable('')
# yy, z_a_agl, 
ax1[0].set_ylabel("meter")
ax1[0].set_ylabel("Altitude [m AGL]")
ax1[0].legend(loc=(0.01, 0.75), fontsize=10)
ax.legend(loc=(0.01, 0.15), fontsize=10)
ax1[1].text(0.03,0.97,r'WVMR',fontsize=10,bbox=props, transform=ax1[1].transAxes)
#ax.set_xscale("log", nonposx='clip')
#ax1[].set_yscale("log", nonposy='clip')
axx.legend(loc = 'upper right',fontsize=10)
ax1[1].legend(fontsize=10)

axxx.legend(loc = 'lower left',fontsize=10)
ax1[2].legend(fontsize=10)

#for xc,c in zip(xcoords,colors):

ax1[0].axvline(x=1,  c='blue')
ax1[1].axvline(x=1, c='blue')
ax1[2].axvline(x=1, c='blue')
ax1[0].axvline(x=-1,  c='blue')
ax1[1].axvline(x=-1, c='blue')
ax1[2].axvline(x=-1, c='blue')

#fig1.savefig(figpath+'/wvmr_t_p_difference.pdf')
