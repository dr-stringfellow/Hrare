import numpy as np
import h5py
from sys import argv
from re import sub
import uproot as uproot
import awkward as ak

froot = argv[1]
fnpy = froot.split("/")[-1].replace('.root','.h5')
hf = h5py.File(fnpy, 'w')

omega_branches = ['omega_kin_pt','omega_kin_eta']
phi_branches = ['phi_kin_pt','phi_kin_eta']

ifile = uproot.open(froot)
events = ifile['Events']

def pad_array(arr,length):
    arr = ak.Array(arr)
    padarr = ak.fill_none(ak.pad_none(arr,length,clip=True,axis=-1),0)
    return np.array(padarr)

omega_arrs = [np.stack(pad_array(events[k].array(),50), axis=0) for k in omega_branches]
omega_arr = np.stack(omega_arrs, axis=-1).astype(np.float32)             
phi_arrs = [np.stack(pad_array(events[k].array(),20), axis=0) for k in phi_branches]
phi_arr = np.stack(phi_arrs, axis=-1).astype(np.float32)             

hf.create_dataset('omega', data=omega_arr)
hf.create_dataset('omega_shape', data=omega_arr.shape) 
hf.create_dataset('phi', data=phi_arr)
hf.create_dataset('phi_shape', data=phi_arr.shape) 
hf.close()
