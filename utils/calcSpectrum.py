import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift

# TODO: multiprocessing
# import multiprocessing
# pool = multiprocessing.Pool()
# pool._processes
# This gets the number of workers

def calcSpectrum(sigIn, sampRate, specType, unit):
    
    # get definitions of frequency division and range
    dF = sampRate/len(sigIn)
    TSim = 1/dF
    FSim = len(sigIn)/TSim
    
    # Looking at frequency domain
    # fft evaluates the discrete Fourier transform (DFT)
    # at a set of equally spaced points on [0,1].
    # fftshift moves this to [-0.5, 0.5] for better visualization.
    
    # Scale fft to get spectrum of the input signal.
    # NOTE:
    # fft(sig, axis=0, norm="ortho", workers=2) defines the normalization
    # of fft to "ortho", which means 1/sqrt(N) is divided for fft results.
    # However, the default norm is 1 for fft and N for ifft.
    # Here we use defaults
    FreqSpectrum = fftshift(fft(sigIn, axis=0)/(FSim))

    # Computing energy spectrum.
    EnergySpectrum = np.abs(FreqSpectrum)**2
    
    # PSD is an approximation to the continuous time PSD.
    PSD = np.abs(EnergySpectrum)/TSim+1e-16

    # PS is calculated according to PSD with: PS = PSD.*dF = PSD./Tsim.
    PS = PSD/TSim
    
    # Select the output according to the input parameter.
    if (specType=='PS') & (unit=='dBm'):
        sigSpec = 30+10*np.log10(PS);
    elif (specType=='PS') & (unit=='linear'):
        sigSpec = PS;
    elif (specType=='PS') & (unit=='dB'):
        sigSpec = 10*np.log10(PS/np.max(PS));
    elif (specType=='PSD') & (unit=='dBm'):
        sigSpec = 30+10*np.log10(PSD);
    elif (specType=='PSD') & (unit=='linear'):
        sigSpec = PSD;
    elif (specType=='PSD') & (unit=='dB'):
        sigSpec = 10*np.log10(PSD/np.max(PSD));
    elif (specType=='FS') & (unit=='linear'):
        sigSpec = FreqSpectrum;
    elif (specType=='FS') & (unit=='dB'):
        sigSpec = 10*np.log10(FreqSpectrum/np.max(FreqSpectrum));
    elif (specType=='FS') & (unit=='dBm'):
        sigSpec = 30+10*np.log10(FreqSpectrum);
    else:
        print('Invalid spectrum type and/or unit.'+
              'returning default power spectrum in dBm.')
        sigSpec = 30+10*np.log10(PS);
        
    return sigSpec
