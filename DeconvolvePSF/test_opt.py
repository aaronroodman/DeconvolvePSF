"I'm trying to figure out what the problem is with my optical model.\
Chris suggested that I generate two optical models with slight modifications and diff them."

from optical_model import get_optical_psf
from optical_model_mod import get_optical_psf_2
import numpy as np

expid = 145329

outputDir = '/u/ki/swmclau2/des/TestOut/'

#get optical PSF
optpsf_stamps, meta_hdulist = get_optical_psf(expid)

np.save(outputDir+'%s_opt_test.npy'%expid, optpsf_stamps)

optpsf_stamps_2, meta_hdulist_2 = get_optical_psf_2(expid)

np.save(outputDir+'%s_opt_test_2.npy'%expid, optpsf_stamps_2)

d = optpsf_stamps - optpsf_stamps_2

print d.mean(), d.std()

d_nonzero = d[d != 0]

print d_nonzero
