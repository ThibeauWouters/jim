import copy
import os
# jax
import jax.numpy as jnp
import jax
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
import numpy as np
jax.config.update("jax_enable_x64", True)

import sys
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
# sys.path.append(SCRIPT_DIR)
GWFAST_PATH = "/home/thibeau.wouters/gwfast/gwfast/"
GWFAST_PATH = os.path.dirname(os.path.realpath(os.path.join(GWFAST_PATH)))

import gwfast.gwfastGlobals as glob

alldetectors = copy.deepcopy(glob.detectors)
print('All available detectors are: '+str(list(alldetectors.keys())))
LVdetectors = {det:alldetectors[det] for det in ['L1', 'H1', 'Virgo']}
print('Using detectors '+str(list(LVdetectors.keys())))
# We use the O2 psds
LVdetectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt')
LVdetectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt')
LVdetectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'Hrec_hoft_V1O2Repro2A_16384Hz.txt')
from gwfast.waveforms import IMRPhenomD_NRTidalv2, TaylorF2_RestrictedPN
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams, CheckFisher


##############################
### SCRIPT HYPERPARAMETERS ###
##############################

# waveform = TaylorF2_RestrictedPN(fHigh=None, 
#                                  is_tidal=True, 
#                                  use_3p5PN_SpinHO=False, 
#                                  phiref_vlso=False, 
#                                  is_eccentric=False, 
#                                  fRef_ecc=None, 
#                                  which_ISCO='Schw', 
#                                  use_QuadMonTid=True)

waveform = IMRPhenomD_NRTidalv2()

check_my_params = False
check_tutorial_stuff = True


############
### MAIN ###
############

myLVSignals = {}

for d in LVdetectors.keys():

    myLVSignals[d] = GWSignal(waveform,
                psd_path=LVdetectors[d]['psd_path'],
                detector_shape = LVdetectors[d]['shape'],
                det_lat= LVdetectors[d]['lat'],
                det_long=LVdetectors[d]['long'],
                det_xax=LVdetectors[d]['xax'],
                verbose=True,
                useEarthMotion = False,
                fmin=20.,
                IntTablePath=None)

myLVNet = DetNet(myLVSignals)

###
### MAIN
###

if check_my_params:

    # These are the ref params we found in GW170817 with jim
    ref_params_jim = {
        'M_c': 1.19755196,
        'eta': 0.23889171,
        's1_z': 0.04989052,
        's2_z': -0.02622163,
        'lambda_1': 60.80207005,
        'lambda_2': 316.70276899,
        'd_L': 11.88292643,
        't_c': -0.00154799,
        'phase_c': 3.66348683,
        'iota': 1.81371344,
        'psi': 1.6000364,
        'ra': 3.4179911,
        'dec': -0.40603519
    }

    # Convert the names from jim to gwfast -- see gwfast paper for definitions
    names_dict = {'M_c': 'Mc',
                'd_L': 'dL',
                't_c': 'tcoal',
                's1_z': 'chi1z',
                's2_z': 'chi2z',
                'lambda_1': 'Lambda1',
                'lambda_2': 'Lambda2',
                'phase_c': 'Phicoal'
    }

    # Get the ref params of GWfast
    ref_params_gwfast = copy.deepcopy(ref_params_jim)
    keys = list(ref_params_gwfast.keys())
    for key in keys:
        if key in names_dict.keys():
            ref_params_gwfast[names_dict[key]] = ref_params_gwfast.pop(key)

    # Convert dL
    ref_params_gwfast['dL'] = ref_params_gwfast['dL'] * 1e-3 # convert from Mpc to Gpc

    print("ref_params_gwfast")
    print(ref_params_gwfast)

    # Convert to single arrays
    print("Converting the ref params")
    # Iterate over the key value pairs and put values in an single float array
    for key, value in ref_params_gwfast.items():
        ref_params_gwfast[key] = np.array([value])

    print("ref_params_gwfast")
    print(ref_params_gwfast)


    totF = myLVNet.FisherMatr(ref_params_gwfast)
    totCov, inversion_err = CovMatr(totF)
    _ = check_covariance(totF, totCov)


### Checking the tutorial stuf

if check_tutorial_stuff:
    print("Checking the tutorial stuff")
    from gwfastUtils import GPSt_to_LMST
    from astropy.cosmology import Planck18

    z = np.array([0.00980])
    tGPS = np.array([1187008882.4])

    GW170817 = {'Mc':np.array([1.1859])*(1.+z),
                'dL':Planck18.luminosity_distance(z).value/1000.,
                'theta':np.array([np.pi/2. + 0.4080839999999999]),
                'phi':np.array([3.4461599999999994]),
                'iota':np.array([2.545065595974997]),
                'psi':np.array([0.]),
                'tcoal':GPSt_to_LMST(tGPS, lat=0., long=0.), # GMST is LMST computed at long = 0°
                'eta':np.array([0.24786618323504223]),
                'Phicoal':np.array([0.]),
                'chi1z':np.array([0.005136138323169717]),
                'chi2z':np.array([0.003235146993487445]),
                'Lambda1':np.array([368.17802383555687]),
                'Lambda2':np.array([586.5487031450857])
            }

    print("Params from tutorial are:")
    print(GW170817)

    SNR = myLVNet.SNR(GW170817)
    print('SNR for GW170817 is %.2f to compare with 33'%SNR[0])

    totF = myLVNet.FisherMatr(GW170817)
    print('The computed Fisher matrix has shape %s'%str(totF.shape))

    # Check e.g. that the (dL,dL) element corresponds to (SNR/dL)^2
    ParNums = IMRPhenomD_NRTidalv2().ParNums
    dL_Num = ParNums['dL']
    print('The relative difference is %.2e !'%((1 - totF[ParNums['dL'],ParNums['dL'],:]/(SNR/GW170817['dL'])**2)[0]))
    
    # Check conditioning
    
    print("Checking conditioning")
    totCov, inversion_err = CovMatr(totF)
    print("inversion_err")
    print(inversion_err)
    
    ParNums = IMRPhenomD_NRTidalv2().ParNums

    newFish, newPars = fixParams(totF, ParNums, ['deltaLambda'])

    print('Now the Fisher matrix has shape %s'%str(newFish.shape))

    newCov, new_inversion_err = CovMatr(newFish)

    _ = check_covariance(newFish, newCov)
    
    # Check fisher
    print("Checking fisher")
    evals, evecs, condNumber = CheckFisher(totF, verbose=True)
    print("evals")
    print(evals)
    print("condNumber")
    print(condNumber)
    
    print("Checking new fisher")
    evals, evecs, condNumber = CheckFisher(newFish, verbose=True)
    print("evals")
    print(evals)
    print("condNumber")
    print(condNumber)
    # _ = check_covariance(totF, totCov)    
    
print("DONE")