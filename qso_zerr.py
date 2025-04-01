import numpy as np

def red_err_multi_gaussian(amps=np.array([1.0]),sigmas=np.array([500]),plots=False):
    '''build the distribution of redshift error
    you can give as many component as you like the sum of amps should be 1'''
    
    assert(np.abs(amps.sum()-1)<1e-5)
    assert(amps.size==sigmas.size)
    
    max_sigma=sigmas.max()
    
    xmin=-max_sigma*5
    xmax=max_sigma*5
    
    xsamp=np.linspace(xmin,xmax,1000)
    
    plikes=[]
    for ss,sig in enumerate(sigmas):
        pthis=np.exp(-0.5*np.power(xsamp/sig,2))
        pthis=amps[ss]*pthis/pthis.sum()
        if(ss==0):
            plikes=pthis.reshape(pthis.size,1)
        else:
            plikes=np.column_stack([plikes,pthis])
            
    ptotal=np.sum(plikes,axis=1)
    assert(np.abs(ptotal.sum()-1)<1e-5)
    
    if(plots):
        for ss,sig in enumerate(sigmas):
            pl.plot(xsamp,plikes[:,ss],label='a=%4.2f sig=%5.2f'%(amps[ss],sigmas[ss]))
        pl.plot(xsamp,ptotal,'k-')
        
        pl.legend(fontsize=12)
        pl.yscale('log')
            
    return xsamp, plikes, ptotal

def sample_redshift_error(zarr,error_model='sig500',plots=False,tcol=None,rseed=0):
    '''This sample redshift with some given error model in terms of multi-gaussian'''
    #speed of light in km/s
    speed_of_light=3.0e5
    
    if(error_model=='sig500'):
        xsamp,plikes,ptotal=red_err_multi_gaussian(amps=np.array([1.0]),sigmas=np.array([500]),plots=False)
    elif(error_model=='3gauss'):
        xsamp,plikes,ptotal=red_err_multi_gaussian(amps=np.array([0.51,0.45,0.04]),
                            sigmas=np.array([94,400,1500]),plots=False)
    #convert probabolity to cumulative sum and sample
    csm=np.cumsum(ptotal)
    
    #Generate random numbers
    np.random.seed(rseed)
    rand_v=np.random.random(zarr.size)
    
    #convert them to the velocities unit from cumulative distribution
    rand_v=np.interp(rand_v,csm,xsamp)
    
    # zerr=rand_v*(1.0+zarr)/speed_of_light
    
    if(plots):
        pl.figure(1)
        hh=np.histogram(rand_v,bins=200)
        pl.plot(xsamp,ptotal/(xsamp[1]-xsamp[0]),'-',color=tcol)
        #dh=hh[1][1]-hh[1][0]
        #pl.plot(hh[1][:-1],hh[0]/(dh*hh[0].sum()),'b-')
        
        pl.xlabel(r'$\Delta v$',fontsize=22)
        #pl.yscale('log')
        
        pl.figure(2)
        pl.plot(zarr,zerr,'.')
        
    return rand_v
