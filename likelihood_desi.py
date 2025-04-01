import os

import numpy as np

from pycorr import TwoPointCorrelationFunction, utils

# # function that shadab provided on how to read xi2d
# def read_wp2d(fname,mode='rper-rpar'):
#     #read the projected correlation function from my code output format
#     wp2d=np.loadtxt(fname)
#     flines=open(fname).readlines()
#     linetags={'rper-rpar':['#rper:','#rpar:'],'rmu':['#r:','#mu:']}
#     #read the rper
#     ii=-1;got=0
#     while(got<2):
#         ii=ii+1
#         if(linetags[mode][0] in flines[ii]):
#             rpstr=flines[ii][len(linetags[mode][0]):].split()
#             rperarr=np.array([])
#             for tt in rpstr:
#                 rperarr=np.append(rperarr,np.float(tt))
#             got=got+1
#         elif(linetags[mode][1] in flines[ii]):
#             rpstr=flines[ii][len(linetags[mode][1]):].split()
#             rpararr=np.array([])
#             for tt in rpstr:
#                 rpararr=np.append(rpararr,np.float(tt))
#             got=got+1
#     return rperarr,rpararr, wp2d
# #  the first column is xi which is the mean of jackknife, the second column is error in xi, third column is xi without jackknife and 4-n columns are jn realizations.\


class wp_Data(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, HOD_params, nrpmin = 7, nrpmax = 21, npimax = 8):
        """
        Constructor of the power spectrum data
        """
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std

        rpfac = 2
        pifac = 4
        nbinrp = int(48/rpfac)
        nbinpi = int(40/pifac)
        # load the power spectrum for all tracer combinations
        clustering = {}
        wpstds = {}
        rs = {}
        for key in data_params['tracer_combos'].keys():
            clustering[key] = np.loadtxt(data_params['tracer_combos'][key]['path2wp'])[nrpmin:nrpmax, 2] # wp
            wpstds[key] = np.loadtxt(data_params['tracer_combos'][key]['path2wp'])[nrpmin:nrpmax, 3] # wp
            rs[key] = np.loadtxt(data_params['tracer_combos'][key]['path2wp'])[nrpmin:nrpmax, 0]
        self.clustering = clustering
        self.wpstds = wpstds
        self.rs = rs

        if 'path2cov' in data_params['tracer_combos'][key]:
            # load the covariance matrix for all tracer combinations
            cov = {}
            icov = {}
            for key in data_params['tracer_combos'].keys():

                newcov = np.load(data_params['tracer_combos'][key]['path2cov'])['cov']
                rescale = data_params['tracer_combos'][key].get('rescale', True)
                if rescale:
                    rescaledcov = np.zeros(newcov.shape)

                    for i in range(newcov.shape[0]):
                        for j in range(newcov.shape[1]):
                            rescaledcov[i, j] = newcov[i, j]/np.sqrt(newcov[i, i]*newcov[j, j])*wpstds[key][i]*wpstds[key][j]
                    cov[key] = rescaledcov
                else:
                    cov[key] = newcov
                
                icov[key] = np.linalg.inv(cov[key])
            self.icov = icov
            self.cov = cov


    def compute_likelihood(self, theory_clustering, theory_density, mockcov = False, ic_down = 1, jointcov_inv = None, fullscale = True):
        """
        Computes the likelihood using information from the context
        """
        skiprp = 0
        if not fullscale:
            skiprp = 4
            
        # Calculate a likelihood up to normalization
        print("clustering keys", np.sort(list(self.clustering.keys())))
        lnprob = 0.
        if jointcov_inv is None:
            for key in self.clustering.keys():
                delta = self.clustering[key] - theory_clustering[key][skiprp:]
                if mockcov:
                    lnprob += np.einsum('i,ij,j', delta, self.icov[key][skiprp:][skiprp:], delta)
                else:
                    lnprob += np.sum(delta**2/self.wpstds[key]**2)
        else:
            delta = np.concatenate([self.clustering[key] - theory_clustering[key] for key in np.sort(list(self.clustering.keys()))])
            lnprob += np.einsum('i,ij,j', delta, jointcov_inv, delta)
        lnprob *= -0.5
        print("clustering lnprob", lnprob)

        # likelihood due to number density
        for etracer in self.num_dens_mean.keys():
            lnprob += -0.5*((self.num_dens_mean[etracer] - theory_density[etracer]*ic_down)/self.num_dens_std[etracer])**2
            print("density lnprob", etracer, -0.5*((self.num_dens_mean[etracer] - theory_density[etracer]*ic_down)/self.num_dens_std[etracer])**2)

        # Return the likelihood
        # print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob

    
class xirppi_Data_fuji(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, HOD_params, nrpmin = 7, nrpmax = 21, npimax = 8):
        """
        Constructor of the power spectrum data
        """
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std
        
        # load the power spectrum for all tracer combinations
        clustering = {}
        xistds = {}
        rps_dict = {}
        pis_dict = {}
        cov = {}
        icov = {}
        # mockcov = {}
        # mockicov = {}
        for key in data_params['tracer_combos'].keys():
            # get xi mean
            xiarray = np.loadtxt(data_params['tracer_combos'][key]['path2power'])
            # bin refac 
            rpfac = 2
            pifac = 4
            nbinrp = int(48/rpfac)
            nbinpi = int(40/pifac)

            pibins = np.linspace(0, 40, nbinpi+1)[:npimax+1]
            rpbins = np.geomspace(0.01, 100., nbinrp+1)[nrpmin:nrpmax+1]

            # reshape for extraction
            xiarray = xiarray.reshape((nbinrp, 2*nbinpi, -1))
            xiarray_folded = np.zeros((nbinrp, nbinpi, np.shape(xiarray)[-1]))
            for i in range(nbinpi):
                for j in range(np.shape(xiarray)[-1]):
                    if j in [2,3]:
                        assert np.sum(xiarray[:, nbinpi-1-i, j]+xiarray[:, nbinpi+i, j]) == 0
                        xiarray_folded[:, i, j] = xiarray[:, nbinpi+i, j]
                    else:
                        xiarray_folded[:, i, j] = 0.5*(xiarray[:, nbinpi-1-i, j]+xiarray[:, nbinpi+i, j])

            ximean = xiarray_folded[:, :, 4][nrpmin:nrpmax, :npimax]
            xistd = xiarray_folded[:, :, 5][nrpmin:nrpmax, :npimax]
            rpmids = xiarray_folded[:, :, 0][nrpmin:nrpmax, 0]
            pimids = xiarray_folded[:, :, 2][0, :npimax]

            assert np.all(abs(0.5*(rpbins[:-1]+rpbins[1:])/rpmids-1) < 0.0001)
            assert np.all(abs(0.5*(pibins[:-1]+pibins[1:])/pimids-1) < 0.0001)

            clustering[key] = ximean
            xistds[key] = xistd
            rps_dict[key] = rpmids
            pis_dict[key] = pimids
            # get cov
            
            if 'path2cov' in data_params['tracer_combos'][key]:
                xicov = np.load(data_params['tracer_combos'][key]['path2cov'])['cov']
                rescale = data_params['tracer_combos'][key].get('rescale', True)
                # print(xicov.shape)
                
                # calculate corr
                xistd_flat = xistd.flatten()
                # print(xistd_flat.shape)
                xicov_new = np.zeros(xicov.shape)
                
                if rescale:
                    for i in range(xicov.shape[0]):
                        for j in range(xicov.shape[1]):
                                xicov_new[i, j] = xicov[i, j]/np.sqrt(xicov[i, i]*xicov[j, j])*xistd_flat[i]*xistd_flat[j]
                else:
                    xicov_new = xicov
                    
                # result = TwoPointCorrelationFunction.load(data_params['tracer_combos'][key]['path2cov'])
                # rebinned = result[:(result.shape[0]//rpfac)*rpfac:rpfac,:(result.shape[1]//pifac)*pifac:pifac]
                # covfull = rebinned.cov()
                # corrcoef = utils.cov_to_corrcoef(covfull)
                # # downsample the cov for the rows we selected
                # indices = np.arange(len(covfull)).reshape((nbinrp, nbinpi))[nrpmin:nrpmax, :npimax].flatten()
                # xicov = covfull[indices][:, indices]
                # xicorr = corrcoef[indices][:, indices]
                # assert np.all(abs(np.sqrt(np.diag(xicov))/xistd.flatten() - 1) < 0.0001)
                # assert np.all(abs(np.linalg.inv(np.linalg.inv(xicorr))/xicorr - 1) < 0.0001)
                # print(np.diag(xicov_new)/np.diag(xicov))
                cov[key] = xicov_new
                icov[key] = np.linalg.inv(xicov_new)

        self.clustering = clustering
        self.xistds = xistds
        self.rps = rps_dict
        self.pis = pis_dict
        self.icov = icov
        self.cov = cov
        
    def compute_likelihood(self, theory_clustering, theory_density, mockcov = False, skiprp = 0):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        for key in self.clustering.keys():
            # print(self.clustering[key][30:], theory_clustering[key].flatten()[30:])
            npibin = self.clustering[key].shape[1]
            delta = self.clustering[key][skiprp:].flatten() - theory_clustering[key][skiprp:].flatten()
            if mockcov:
                lnprob += np.einsum('i,ij,j', delta, self.icov[key][skiprp*npibin:,skiprp*npibin:], delta)
            else:
                lnprob += np.sum(delta**2/self.xistds[key][skiprp:].flatten()**2)
        lnprob *= -0.5
        print("clustering lnprob", lnprob)

        # likelihood due to number density
        for etracer in self.num_dens_mean.keys():
            lnprob += -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2
            print("density lnprob", etracer, -0.5*((self.num_dens_mean[etracer] - theory_density[etracer])/self.num_dens_std[etracer])**2)

        # Return the likelihood
        # print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob
    

class ant_Data(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params, HOD_params):
        """
        Constructor of the power spectrum data
        """
        num_dens_mean = {}
        num_dens_std = {}
        for key in HOD_params['tracer_flags'].keys():
            if HOD_params['tracer_flags'][key]:
                num_dens_mean[key] = data_params['tracer_density_mean'][key]
                num_dens_std[key] = data_params['tracer_density_std'][key]
        self.num_dens_mean = num_dens_mean
        self.num_dens_std = num_dens_std

        # antoine's file
        data_file = np.load('./data/data_cov_ELG_EDAv1_z0.8-1.6.npy', allow_pickle=True).item()
        
        self.rp_bin, self.pi_bin = data_file['edges_rppi']
        self.s_bin, self.mu_bin = data_file['edges_smu']
        self.clustering = {'ELG_ELG': data_file['data']}
        self.cov = {'ELG_ELG': data_file['Mcov']}
        self.icov = {'ELG_ELG': np.linalg.inv(self.cov['ELG_ELG'])}


    def compute_likelihood(self, theory_clustering):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        for key in self.clustering.keys():
            delta = self.clustering[key] - theory_clustering[key]
            lnprob += np.einsum('i,ij,j', delta, self.icov[key], delta)
        lnprob *= -0.5
        # Return the likelihood
        # print("theory density and target density", theory_density['LRG'], self.num_dens_mean['LRG'])
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob