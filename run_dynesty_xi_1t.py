#! /usr/bin/env python
import os
import time
import sys

import numpy as np
import argparse
import yaml
import dill
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 13})

# from stochopy import MonteCarlo, Evolutionary

from dynesty import NestedSampler
from dynesty import plotting as dyplot

from likelihood_desi import xirppi_Data_fuji, wp_Data
from abacusnbody.hod.abacus_hod import AbacusHOD
from qso_zerr import sample_redshift_error

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class SampleFileUtil(object):
    """
    Util for handling sample files.
    Copied from Andrina's code.

    :param filePrefix: the prefix to use
    :param reuseBurnin: True if the burn in data from a previous run should be used
    """

    def __init__(self, filePrefix, carry_on=False):
        self.filePrefix = filePrefix
        if carry_on:
            mode = 'a'
        else:
            mode = 'w'
        self.samplesFile = open(self.filePrefix + '.txt', mode)
        self.clFile = open(self.filePrefix + 'cl.txt', mode)
        self.probFile = open(self.filePrefix + 'prob.txt', mode)
        # set permission
        os.chmod(self.filePrefix + '.txt', 0o755)
        os.chmod(self.filePrefix + 'cl.txt', 0o755)
        os.chmod(self.filePrefix + 'prob.txt', 0o755)

    def persistSamplingValues(self, pos, cl, prob):
        self.persistValues(self.samplesFile, self.clFile, self.probFile, pos, cl, prob)

    def persistValues(self, posFile, clFile, probFile, pos, cl, prob):
        """
        Writes the walker positions and the likelihood to the disk
        """
        posFile.write("\t".join([str(q) for q in pos]))
        posFile.write("\n")
        posFile.flush()

        clFile.write("\t".join([str(ecl) for ecl in cl]))
        clFile.write("\n")
        clFile.flush()
        
        probFile.write(str(prob))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.clFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"
    
DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_xi.yaml'

def inrange(p, bounds):
    return np.all((p<=bounds[:, 1]) & (p>=bounds[:, 0]))

def lnprob(p, param_mapping, mytracers, Data, Ball, nthread, bounds, mockcov = True, qso_zerr = False, Nrepeat = 15, skiprp = 0, chain_file = None, writechain = False):
    if inrange(p, bounds):
        print("evaulating ", p)
        startf = time.time()
        for tracer_type in mytracers: 
            for key in param_mapping[tracer_type].keys():
                mapping_idx = param_mapping[tracer_type][key]
                if key == 'logsigma':
                    Ball.tracers[tracer_type]['sigma'] = 10**p[mapping_idx]
                else:
                    Ball.tracers[tracer_type][key] = p[mapping_idx]

        # impose Mmin cut
        for tracer_type in mytracers: 
            if tracer_type == 'LRG' and 10**Ball.tracers[tracer_type]['logM_cut']*Ball.tracers[tracer_type]['kappa'] < 1e12:
                return -np.inf
            elif tracer_type == 'ELG' and 10**Ball.tracers[tracer_type]['logM_cut']*Ball.tracers[tracer_type]['kappa'] < 2e11:
                return -np.inf
            elif tracer_type == 'QSO' and 10**Ball.tracers[tracer_type]['logM_cut']*Ball.tracers[tracer_type]['kappa'] < 2e11:
                return -np.inf
            
        # we need to determine the expected number density 
        for tracer_type in mytracers:
            Ball.tracers[tracer_type]['ic'] = 1

        ngal_dict, fsat_dict = Ball.compute_ngal(Nthread = nthread)
        print(ngal_dict, fsat_dict)
        
        for tracer_type in mytracers:
            if fsat_dict[tracer_type] > 0.6:
                return -np.inf
            if not tracer_type == 'ELG':
                N_tracer = ngal_dict[tracer_type]
                # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
                Ball.tracers[tracer_type]['ic'] = \
                    min(1, Data.num_dens_mean[tracer_type]*Ball.params['Lbox']**3/N_tracer)
            if tracer_type == 'ELG':
                N_tracer = ngal_dict[tracer_type]
                if abs(Data.num_dens_mean[tracer_type]*Ball.params['Lbox']**3/N_tracer - 1) > 1:
                    return -np.inf
                
            print("predicted ic", Ball.tracers[tracer_type]['ic'], "predicted fsat", fsat_dict)

        # pass them to the mock dictionary
        mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread)

        # put a satellite fraction cut
        theory_density = {}
        for tracer_type in mytracers:
            if mock_dict[tracer_type]['Ncent'] < 0.5*len(mock_dict[tracer_type]['x']):
                return -np.inf
            theory_density[tracer_type] = len(mock_dict[tracer_type]['x'])/Ball.params['Lbox']**3

            
        clustering = Ball.compute_xirppi(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = nthread)

        # qso z err following edmund's paper
        if 'QSO' in mytracers and qso_zerr:
            deltaz = sample_redshift_error(mock_dict['QSO']['z'])/Ball.params['velz2kms']
            mock_dict['QSO']['z'] = (mock_dict['QSO']['z'] 
                                                + deltaz) % Ball.params['Lbox']
            clustering = Ball.compute_xirppi(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = nthread)
            for iq in range(Nrepeat):
                mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread, reseed = 150+iq)
                deltaz = sample_redshift_error(mock_dict['QSO']['z'], rseed = 250+iq)/Ball.params['velz2kms']
                mock_dict['QSO']['z'] = (mock_dict['QSO']['z'] 
                                                    + deltaz) % Ball.params['Lbox']
                clustering_new = Ball.compute_xirppi(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = nthread)
                clustering = {k: clustering.get(k, 0) + clustering_new.get(k, 0) for k in set(clustering)}
                
            clustering = {k: clustering.get(k, 0)/(Nrepeat+1) for k in set(clustering)}

        lnP = Data.compute_likelihood(clustering, theory_density, mockcov = mockcov, skiprp = skiprp)
        print("logl took time, ", time.time() - startf)

        # this is gonna be one tracer only
        if writechain:
            for tracer_type in mytracers:
                tracer_combo = tracer_type + '_' + tracer_type
                chain_file.persistSamplingValues(p, clustering[tracer_combo].flatten(), lnP)

        return lnP
    
    else:
        return -np.inf
    
    
# prior transform function
def prior_transform(u, params_hod, params_hod_initial_range):
    return stats.norm.ppf(u, loc = params_hod, scale = params_hod_initial_range)

# uniform ellipsoidal prior
def prior_transform_flat(u, bounds):
    param_range = bounds[:, 1] - bounds[:, 0]
    x = np.array(u)  # scale and shift to [-0.5, 0.5)
    x = x*param_range + bounds[:, 0]    
    return x

def main(path2config):

    # load the yaml parameters
    config = yaml.safe_load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    
    
    nparams = 0
    for key in HOD_params['tracer_flags'].keys():
        if HOD_params['tracer_flags'][key]:
            nparams += len(fit_params[key].keys())
    param_mapping = {}
    bounds = np.zeros((nparams, 2))
    ptrans = np.zeros((nparams, 2))
    mytracers = []
    for tracertype in fit_params.keys():
        mytracers += [tracertype]
        param_mapping[tracertype] = {}
        sub_fit_params = fit_params[tracertype]
        for key in sub_fit_params.keys():
            mapping_idx = sub_fit_params[key][0]
            param_mapping[tracertype][key] = mapping_idx
            bounds[mapping_idx, :] = sub_fit_params[key][3:]
            ptrans[mapping_idx, :] = sub_fit_params[key][1:3]
    print(mytracers)
    
    # Make path to output
    if not os.path.isdir(os.path.expanduser(dynesty_config_params['path2output'])):
        try:
            os.makedirs(os.path.expanduser(dynesty_config_params['path2output']))
        except:
            pass
        
    # dynesty parameters
    nlive = dynesty_config_params['nlive']
    maxcall = dynesty_config_params['maxcall']
    dlogz = dynesty_config_params['dlogz']
    method = dynesty_config_params['method']
    bound = dynesty_config_params['bound']
    nthread = dynesty_config_params['nthread']
    mockcov = dynesty_config_params['mockcov']
    qso_zerr = dynesty_config_params.get('qso_zerr', True)
    qso_repeat = dynesty_config_params.get('qso_repeat', 0)
    fullscale = dynesty_config_params.get('fullscale', False)
    skiprp = dynesty_config_params.get('skiprp', 0)
    writechain = dynesty_config_params.get('writechain', False)
    
    # read data parameters
    if fullscale:
        nrpmin = 3
    else:
        nrpmin = 7
    newData = xirppi_Data_fuji(data_params, HOD_params, nrpmin = nrpmin)

    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    chain_file = SampleFileUtil(prefix_chain, carry_on= not dynesty_config_params['rerun'])

    if mockcov:
        prefix_chain += '_mockcov'
    if fullscale:
        prefix_chain += '_fullscale'
        
    # initiate sampler
    found_file = os.path.isfile(prefix_chain+'.dill')
    if (not found_file) or dynesty_config_params['rerun']:

        # initialize our nested sampler
        sampler = NestedSampler(lnprob, prior_transform, nparams, 
            logl_args = [param_mapping, mytracers, newData, newBall, nthread, bounds, mockcov, qso_zerr, qso_repeat, skiprp, chain_file, writechain], 
            ptform_args = [ptrans[:, 0], ptrans[:, 1]], # [bounds], # 
            nlive=nlive, sample = method, rstate = np.random.RandomState(dynesty_config_params['rseed']))
            # first_update = {'min_eff': 20})

    else:
        # load sampler to continue the run
        with open(prefix_chain+'.dill', "rb") as f:
            sampler = dill.load(f)
        sampler.rstate = np.random.RandomState(dynesty_config_params['rseed']) # np.load(prefix_chain+'_results.npz', allow_pickle = True)['rstate']
    print("run sampler")

    sampler.run_nested(maxcall = maxcall, dlogz = dlogz)

    # save sampler itself
    with open(prefix_chain+'.dill', "wb") as f:
         dill.dump(sampler, f)
    res1 = sampler.results
    np.savez(prefix_chain+'_results.npz', res = res1, rstate = np.random.get_state())

# def bestfit_logl(path2config):

#     # load the yaml parameters
#     config = yaml.safe_load(open(path2config))
#     sim_params = config['sim_params']
#     HOD_params = config['HOD_params']
#     clustering_params = config['clustering_params']
#     data_params = config['data_params']
#     dynesty_config_params = config['dynesty_config_params']
#     fit_params = config['dynesty_fit_params']    
    
#     # create a new abacushod object and load the subsamples
#     newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

#     # read data parameters
#     newData = xirppi_Data_fuji(data_params, HOD_params)

#     # parameters to fit
#     nparams = len(fit_params.keys())
#     param_mapping = {}
#     param_tracer = {}
#     params = np.zeros((nparams, 4))
#     for key in fit_params.keys():
#         mapping_idx = fit_params[key][0]
#         tracer_type = fit_params[key][-1]
#         param_mapping[key] = mapping_idx
#         param_tracer[key] = tracer_type
#         params[mapping_idx, :] = fit_params[key][1:-1]

#     # where to record
#     prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
#                                 dynesty_config_params['chainsPrefix'])

#     datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
#     res1 = datafile['res'].item()

#     res1['samples'][:, 5] = abs(res1['samples'][:, 5])

#     # print the max logl fit
#     logls = res1['logl']
#     indmax = np.argmax(logls)
#     hod_params = res1['samples'][indmax]

#     print("bestfit logl", lnprob(hod_params, params, param_mapping, param_tracer, newData, newBall))

# def traceplots(path2config, tracer = 'ELG'):
#     # load the yaml parameters
#     config = yaml.safe_load(open(path2config))
#     sim_params = config['sim_params']
#     HOD_params = config['HOD_params']
#     clustering_params = config['clustering_params']
#     data_params = config['data_params']
#     dynesty_config_params = config['dynesty_config_params']
#     fit_params = config['dynesty_fit_params']    
#     mytracers = []
#     for tracertype in fit_params.keys():
#         mytracers += [tracertype]
#     tracer = mytracers[0] # assumes single tracer

#     mockcov = dynesty_config_params['mockcov']
#     fullscale = dynesty_config_params.get('fullscale', False)
#     if mockcov:
#         dynesty_config_params['chainsPrefix'] += '_mockcov'
#     if fullscale:
#         dynesty_config_params['chainsPrefix'] += '_fullscale'
        
#     # where to record
#     prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
#                                 dynesty_config_params['chainsPrefix'])
    
#     datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
#     res1 = datafile['res'].item()

#     names = list(fit_params[tracer].keys())
#     for i in range(len(names)):
#         if names[i] == 'sigma' and tracer == 'LRG':
#             res1['samples'][:, i] = 10**res1['samples'][:, i]
#             # names[i] = '$\sigma$'

#     # print the max logl fit
#     logls = res1['logl']
#     indmax = np.argmax(logls)
#     hod_params = res1['samples'][indmax]
#     print("max logl fit ", hod_params, logls[indmax])

#     #     # make plots 
#     #     fig, axes = dyplot.runplot(res1)
#     #     pl.tight_layout()
#     #     fig.savefig('./plots_dynesty/runplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

#     #     fig, axes = dyplot.traceplot(res1,
#     #                              labels = list(fit_params[tracer].keys()),
#     #                              truth_color='black', show_titles=True,
#     #                              trace_cmap='viridis', 
#     #                              trace_kwargs = {'edgecolor' : 'none'})
#     #     pl.tight_layout()
#     #     fig.savefig('./plots_dynesty/traceplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

#     # # plot initial run (res1; left)
#     # rcParams.update({'font.size': 12})
#     # # scatter corner plot
#     # fig, ax = dyplot.cornerpoints(res1, cmap = 'plasma', truths = list(params[:, 0]), labels = list(fit_params.keys()), kde = False)
#     # #pl.tight_layout()
#     # fig.savefig('./plots_dynesty/cornerpoints_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

#     # scatter corner 
#     # res1['samples'] = res1['samples'][:, 5:]
#     fig, ax = dyplot.cornerplot(res1, color='blue', truths=hod_params,
#                                truth_color='black', quantiles_2d = [0.393, 0.865, 0.989], 
#                                labels = list(fit_params[tracer].keys()), # ['$\log M_\mathrm{cut}$', '$\log M_1$', '$\log \sigma$', '$\\alpha$', '$\kappa$', '$\\alpha_c$', '$\\alpha_s$'], 
#                                show_titles=True, smooth = 0.05,
#                                max_n_ticks=5, quantiles=None, 
#                                label_kwargs = {'fontsize' : 18},
#                                hist_kwargs = {'histtype' : 'step'})
#     #pl.tight_layout()
#     fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'.pdf', dpi = 100)

#     # # plot only assembly bias
#     # res1['samples'] = res1['samples'][:, 7:]
#     # fig, ax = dyplot.cornerplot(res1, color='blue', truths= [0, 0], # hod_params,
#     #                            truth_color='red', span = [(-0.8, 0.2), (-0.5, 0.7)], truth_kwargs = {'ls': '--', 'alpha': 0.7},
#     #                            labels = ['$A_\mathrm{cent}$', '$A_\mathrm{sat}$'], # list(fit_params.keys()),
#     #                            show_titles=True, smooth = 0.04, quantiles_2d = [0.393, 0.865], 
#     #                            max_n_ticks=5, quantiles=None, 
#     #                            label_kwargs = {'fontsize' : 18},
#     #                            hist_kwargs = {'histtype' : 'step'})
#     # #pl.tight_layout()
#     # fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'_Aonly.pdf', dpi = 100)

#     # # plot only assembly bias
#     # res1['samples'] = res1['samples'][:, 7:]
#     # fig, ax = dyplot.cornerplot(res1, color='blue', truths= [0, 0], # hod_params,
#     #                            truth_color='red', span = [(-0.15, 0.06), (-0.7, 0.3)], truth_kwargs = {'ls': '--', 'alpha': 0.7},
#     #                            labels = ['$B_\mathrm{cent}$', '$B_\mathrm{sat}$'], # list(fit_params.keys()),
#     #                            show_titles=True, smooth = 0.04, quantiles_2d = [0.393, 0.865, 0.989], 
#     #                            max_n_ticks=5, quantiles=None, 
#     #                            label_kwargs = {'fontsize' : 18},
#     #                            hist_kwargs = {'histtype' : 'step'})
#     # #pl.tight_layout()
#     # fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'_Boldonly.pdf', dpi = 100)


#     # # plot only assembly bias
#     # res1['samples'] = res1['samples'][:, 7:]
#     # fig, ax = dyplot.cornerplot(res1, color='blue', truths= [0, 0], # hod_params,
#     #                            truth_color='red', span = [(-0.8, 0.2), (-0.5, 0.7)], truth_kwargs = {'ls': '--', 'alpha': 0.7},
#     #                            labels = ['$A_\mathrm{cent}$', '$A_\mathrm{sat}$'], # list(fit_params.keys()),
#     #                            show_titles=True, smooth = 0.04, quantiles_2d = [0.393, 0.865], 
#     #                            max_n_ticks=5, quantiles=None, 
#     #                            label_kwargs = {'fontsize' : 18},
#     #                            hist_kwargs = {'histtype' : 'step'})
#     # #pl.tight_layout()
#     # fig.savefig('./plots_dynesty/cornerplot_'+dynesty_config_params['chainsPrefix']+'_Aonly.pdf', dpi = 100)

def plot_bestfit(path2config, key = 'LRG_LRG'):
    # load the yaml parameters
    config = yaml.safe_load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    

    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_tracer = {}
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
        params[mapping_idx, :] = fit_params[key][1:-1]

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    datafile = np.load(prefix_chain+'_results.npz', allow_pickle = True)
    res1 = datafile['res'].item()

    # print the max logl fit
    logls = res1['logl']
    indmax = np.argmax(logls)
    hod_params = res1['samples'][indmax]
    # print("max logl fit ", hod_params, logls[indmax])
    # hod_params = [12.77971427, 14.18003688, -3.68363377,  1.04728458,  0.33468666]
    # hod_params = [12.88692244, 14.38687332, -2.6515272,   0.81305859,  0.33478086,  0.22611396,
    #             1.21681336]
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    newData = xirppi_Data(data_params, HOD_params)

    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        if key == 'sigma':
            newBall.tracers[tracer_type][key] = 10**hod_params[mapping_idx]
        else:
            newBall.tracers[tracer_type][key] = hod_params[mapping_idx] 

    # we need to determine the expected number density 
    newBall.tracers['LRG']['ic'] = 1
    ngal_dict = newBall.compute_ngal(Nthread = 32)[0]
    # we are only dealing with lrgs here
    N_lrg = ngal_dict['LRG']
    print("Nlrg ", N_lrg, "data density ", newData.num_dens_mean['LRG'])
    # print(Ball.tracers['LRG']['ic'], N_lrg, Data.num_dens_mean['LRG']*Ball.params['Lbox']**3)
    newBall.tracers['LRG']['ic'] = \
        min(1, newData.num_dens_mean['LRG']*newBall.params['Lbox']**3/N_lrg)
    print("fic", newBall.tracers['LRG']['ic'])
    print(newBall.tracers)
    mock_dict = newBall.run_hod(newBall.tracers, newBall.want_rsd, Nthread = 64)
    Ncent = mock_dict['LRG']['Ncent']
    Ntot = len(mock_dict['LRG']['x'])
    print('satellite fraction ', (Ntot - Ncent)/Ntot)
    clustering = newBall.compute_xirppi(mock_dict, newBall.rpbins, newBall.pimax, newBall.pi_bin_size, Nthread = 16)
    
    mock_xi = clustering[key]
    data_xi = newData.clustering[key].reshape(np.shape(mock_xi))
    delta_xi_norm = (mock_xi - data_xi) / newData.diag[key].reshape(np.shape(mock_xi))
    np.savez("./xidata_"+dynesty_config_params['chainsPrefix'], mock = mock_xi, data = data_xi, 
        error = newData.diag[key].reshape(np.shape(mock_xi)), rp = newBall.rpbins)

    # make a triple plot, xi, delta xi, chi2
    fig = pl.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(ncols = 3, nrows = 2, width_ratios = [1, 1, 1], height_ratios = [1, 12]) 
    mycmap2 = cm.get_cmap('bwr')

    # plot 1
    ax1 = fig.add_subplot(gs[3])
    ax1.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col1 = ax1.imshow(mock_xi.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(newBall.rpbins[0]), np.log10(newBall.rpbins[-1]), 0, newBall.pimax], 
        cmap = cm.viridis, norm=colors.LogNorm(vmin = 0.01, vmax = 30))
    ax1.set_yticks(np.linspace(0, 30, 7))
    # ax1.set_yticklabels(pibins)

    ax0 = fig.add_subplot(gs[0])
    cbar = pl.colorbar(col1, cax = ax0, orientation="horizontal")
    cbar.set_label('$\\xi(r_p, \pi)$', labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')

    # plot 2
    ax2 = fig.add_subplot(gs[4])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(newBall.rpbins[0]), np.log10(newBall.rpbins[-1]), 0, newBall.pimax], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-10, vmax=10))
    ax2.set_yticks(np.linspace(0, 30, 7))
    # ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs[1])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal", ticks = [-10, -5, 0, 5, 10])
    cbar.set_label("$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    chi2s = (delta_xi_norm.flatten()[6:] * np.dot(newData.icov[key][6:, 6:], delta_xi_norm.flatten()[6:]))
    newshape = (np.shape(delta_xi_norm)[0]-1, np.shape(delta_xi_norm)[1])
    chi2s = chi2s.reshape(newshape)
    print(chi2s, np.sum(chi2s))
    ax2 = fig.add_subplot(gs[5])
    ax2.set_xlabel('$\log r_p$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(chi2s.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(newBall.rpbins[0]), np.log10(newBall.rpbins[-1]), 0, newBall.pimax], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-100, vmax=100))
    ax2.set_yticks(np.linspace(0, 30, 7))
    # ax2.set_yticklabels(pibins)

    ax3 = fig.add_subplot(gs[2])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal")
    cbar.set_label("$\chi^2$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    pl.subplots_adjust(wspace=20)
    pl.tight_layout()
    fig.savefig("./plots_dynesty/plot_bestfit_"+dynesty_config_params['chainsPrefix']+".pdf", dpi=720)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    parser.add_argument('--mode', help='Path to the config file', default='sample')

    args = vars(parser.parse_args())    
    if args['mode'] == 'sample':
        main(args['path2config'])

    elif args['mode'] == 'plot':
        traceplots(args['path2config'])
    # bestfit_logl(**args)

    # plot_bestfit(**args)
