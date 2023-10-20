import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


def eyediagrams(
    sigIn, SPS=1, numOfEyes=2, numOfSymbols=5000, 
    sampRate=25e9, upSampSPS=16, sampleOffset=0, 
    color='crimson', alpha=0.1, lw=0.1,
    yLim=[-0.2, 1.2],
    title='Eye diagrams',
    fig_id=999
):
    
    numOfSamplesPerFrame = SPS*numOfEyes*upSampSPS
    
    
    # FIXME: resample use 'spline' method, which results in RC shaped signal if NRZ shape is used
    # use scipy.interpolate.interp1d?
    sigIn_upsampled = resample(sigIn[0:numOfSymbols], len(sigIn[0:numOfSymbols])*upSampSPS)

    sigIn_upsampled = np.roll(sigIn_upsampled, sampleOffset, axis=0)

    sigEye = np.reshape(sigIn_upsampled, [np.uint(len(sigIn_upsampled)/(numOfSamplesPerFrame)), numOfSamplesPerFrame])

    dT = 1/sampRate/upSampSPS
    T = np.arange(0, numOfSamplesPerFrame*dT, dT)

    plt.close(fig_id)
    plt.figure(fig_id)
    plt.plot(T, np.array(sigEye[0:numOfSymbols:,::]).T, color=color, alpha=alpha, lw=lw) #color='crimson'
    plt.title(str(title))
    plt.ylim(yLim[0], yLim[1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (a.u.)')