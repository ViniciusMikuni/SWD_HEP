import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import uproot_methods
import awkward



split = [int(8.e5),int(2e5),int(8e5)]
#split = [int(100000),int(10000),int(10000)]
apply_sys = False
mode = 'multiplicity' #Modify the pt of quarks in the target dataset

def convert_coordinate(data):
    pt = data[:,:,0]
    eta = data[:,:,1]
    phi = data[:,:,2]
    #energy  = pt*np.sqrt(1 + np.sinh(eta)**2)
    mass  = pt*0 #mass for particles much smaller than their momenta

    mask = pt>0
    n_particles = np.sum(mask, axis=1)

    return awkward.JaggedArray.fromcounts(n_particles,pt[mask]), awkward.JaggedArray.fromcounts(n_particles,eta[mask]), awkward.JaggedArray.fromcounts(n_particles,phi[mask]), awkward.JaggedArray.fromcounts(n_particles,mass[mask]) 


def clustering_simple(data,out_dir,output_name):
    "Create high level features for jets"
    npid = data['y']
    npdata = data['X']
    NFEAT=6
    features = []

    print("Start clustering...")       
    for sample in npdata: #loop every file and combine the info
        pt,eta,phi,m = convert_coordinate(sample)            
        p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt,eta,phi,m)
        jet_p4 = p4.sum()
        z = p4.pt/jet_p4.pt
        z2 = z**2
        dr = np.sqrt((p4.eta - jet_p4.eta)**2 + jet_p4.delta_phi(p4)**2)
        feature = np.zeros((sample.shape[0],NFEAT))
        feature[:,0] += p4.counts #multiplicity
        feature[:,1] += np.sqrt(z2.sum()) #fragmentation
        #angularities
        feature[:,2] +=  (z*dr).sum() 
        feature[:,3] +=  (z*dr**1.5).sum() 
        feature[:,4] +=  (z*dr**2).sum() 
        feature[:,5] +=  np.log(jet_p4.pt)
        
        

        features.append(feature)
    features=np.concatenate(features,0)
    print(features.shape)

    def Apply_change(data,label,mode='pt'):
        #Change evaluation sample to be different compared to nominal
        if mode == 'pt':
            pt_change = 0.1
            data[label==1,-1] += np.log(1-pt_change)
        if mode == 'multiplicity':
            multiplicity_change = 0.1
            data[label==1,0] *= (1+multiplicity_change)
        return data
        
    samples = ['train','test','evaluate']
    event_start = 0
    event_end = split[0]
    
    for isamp, sample in enumerate(samples):
        print(sample,event_start,event_end)
        label = npid[event_start:event_end]         
        if apply_sys and sample == 'evaluate':
            data = Apply_change(features[event_start:event_end],label,mode)
            sample+="_%s"%mode
        else:
            data = features[event_start:event_end]
        with h5py.File('{}/{}_{}.h5'.format(out_dir,sample,output_name), "w") as fh5:
            dset = fh5.create_dataset("data", data=data)
            dset = fh5.create_dataset("pid", data=label)
        if isamp < len(samples)-1:
            event_start = event_end
            event_end += split[isamp+1]




if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default="/clusterfs/ml4hep/vmikuni/SWDAN/PYTHIA", help="Folder containing input files")
    parser.add_option("--dir", type="string", default="/clusterfs/ml4hep/vmikuni/SWDAN/parsed", help="Folder to save output files")
    parser.add_option("--out", type="string", default="PYTHIA", help="Output file name")

    (flags, args) = parser.parse_args()

    samples_path = flags.folder


    files = []
    for r, d, f in os.walk(samples_path):
        for file in f:
            if '.npz' in file:
                files.append(os.path.join(r, file))
    print(files)

    output_name = flags.out

    
    print("Loading data...")
    data = {}
    for ifi, f in enumerate(files):
        dataset = np.load(f)
        for key in dataset.files:
            if ifi==0:
                if key == 'y':
                    data[key] = dataset[key]
                else:
                    data[key] = [dataset[key]]
            else:
                if key == 'y':
                    data[key] = np.concatenate((data[key],dataset[key]),axis=0)
                else:
                    data[key].append(dataset[key])

        
        print(np.array(data[key]).shape)

    clustering_simple(data,flags.dir,output_name)
