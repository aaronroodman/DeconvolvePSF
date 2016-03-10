## Synopsis

DeconvolvePSF is an attempt at high-level PSF modeling for DES and LSST. The key idea for this software is that the PSF can be modeled as a convolution of an optics piece and an atmospheric piece. The optics piece in this work comes from the [WavefrontPSF module](https://github.com/cpadavis/WavefrontPSF) by Chris Davis. This module finds the model of WavefrontPSF, deconvolves it from the observed stars with the Richardson-Lucy deconvolution algorithm, and then models the residuals with the PSFEx PSF modeler. 

## Code Example

The module can be run locally on an exposure `expid` and stored in `output_dir` via

`python afterburner.py <expid> <output_dir>`

Alternatively it can be submitted to the SLAC batch cluster (if you are a SLAC employee logged into ki-ls) via:

`python do_call.py <expid1> <expid2> ... <expidN>` 

or run on 40 random exposures simply via

`python do_call.py`

Please note the output directory in do_call is hardcoded in. 

## License

This software is distibuted open source under the MIT license. 
