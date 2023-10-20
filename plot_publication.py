import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
from matplotlib import cm
from matplotlib.colors import Normalize 
from matplotlib.ticker import MultipleLocator

from scipy.optimize import curve_fit
from scipy.interpolate import interpn

import skimage

import utils
from utils import genFigures as gf


# ----------------------------------------------------------------------------
# Figure format settings
# ----------------------------------------------------------------------------

# save the figure using svg format
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# the svg format figure will not render text as paths, but embed the font as text
plt.rcParams['svg.fonttype'] = 'none'

# change default font for all plots
plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# change other settings to ease the work in Adobe Illustrator
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad']= 6 # spacing between axes to x ticks, this also includes spacing between label and ticks
plt.rcParams['ytick.major.pad']= 6 # spacing between axes to y ticks
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titlepad'] = 10
plt.rcParams['axes.labelpad'] = 6
plt.rcParams['axes.labelsize'] = 14


# ----------------------------------------------------------------------------
# Figure 2b
# ----------------------------------------------------------------------------

# Get and read the data file
data_path = './Experiment_data/'
data_file = 'Experiment-Vpp_Heater voltage.xlsx'
save_to_path = './Experiment_results/'

data = pd.read_excel(data_path+data_file)
heater_voltage = data['Heater voltage'].to_numpy()
Vpp = data['Vpp'].to_numpy()

def func(x, p, k):
    y = np.sin(2*np.pi*x*p)*np.exp(-k*x**2)
    return y

normalized_vpp = (Vpp-np.min(Vpp))/(np.max(Vpp)-np.min(Vpp))
normalized_vpp = normalized_vpp*2-1

popt, pcov = curve_fit(func, heater_voltage, normalized_vpp)

# Create a figure and two subplots
fig, ax = plt.subplots()
fig.set_size_inches((5, 5))
ax.plot(heater_voltage, normalized_vpp,'.')
y_ = func(heater_voltage,popt[0]-1.6, popt[1]+60)
y_ = (y_-np.min(y_))/(np.max(y_)-np.min(y_))
y_ = y_*2-1
ax.plot(heater_voltage+0.00, y_)
ax.set_xlabel('Normalized modulation amplitude')
ax.set_ylabel('Heater voltage (V)')

fig.savefig(save_to_path+'Figure2b.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 3: (not incl.) waveform comparision for analog scheme
# ----------------------------------------------------------------------------

# Get the data from the analog computing simulation
data_path = './Sim_data/Prewitt/'
data_file = 'Analog_Chelsea_Greys_distribution_y_noise_SNR=25dB_RMSE=0.02708.bin'
expected_data_file = 'Analog_Chelsea_Greys_distribution_x_standardSNR=25dB_RMSE=0.02708.bin'

save_to_path = './Sim_results/'

text_file_name_split = data_file.split("_")
scheme = text_file_name_split[0]

with open(data_path+data_file, 'rb') as file:
        buffer = file.read()
file.close()
print('Results ' +data_file + ' loaded.')
result = np.frombuffer(buffer)

with open(data_path+expected_data_file, 'rb') as file:
        buffer = file.read()
file.close()
print('Expected results ' +expected_data_file + ' loaded.')
genie_result = np.frombuffer(buffer)

# Plot the waveforms
fig, ax = gf.plot_waveform_comparison(result, genie_result)
fig.savefig(save_to_path+'waveform_'+data_file[:-4:]+'_zoomedIn_1c5e4_2e4.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 3b: signal distribution against true values, analog scheme
# ----------------------------------------------------------------------------

# Plot the signal-genie distribution with rasterized points but vector axes
ax_3b = gf.plot_density_scatter(genie_result, result, dot_size=2, bins=11, cmap='viridis', rasterized=True)
ax_3b.get_figure().savefig(save_to_path+'figure3b.svg', transparent=True, dpi=600, format='svg')

# reshape the data to the image
image_data = np.reshape(result, [300-2, 451-2])  # No padding, only grey channel processed

# plot the image without axes
plt.figure()
ax_5e_processed = plt.imshow(image_data, cmap='Greys')
plt.axis('off')  # Turn off axes

# Save the figure as SVG without white padding
save_file_path = save_to_path + './figure3b_image.svg'
plt.savefig(save_file_path, format='svg', bbox_inches='tight', pad_inches=0)


# ----------------------------------------------------------------------------
# Figure 3d: noise distribution, analog scheme
# ----------------------------------------------------------------------------

# Plot the noise distribution
ax_3d = gf.plot_noise_dist(result, genie_result, scheme='Analog', bins=101)
ax_3d.get_figure().savefig(save_to_path+'figure3d.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 3: (not incl.) waveform comparision for analog scheme
# ----------------------------------------------------------------------------

# Load data from digital computing simulation
data_path = './Sim_data/Prewitt/'
data_file = 'Digital_Chelsea_Greys_distribution_y_noise_SNR=25dB_RMSE=0.00121.bin'
expected_data_file = 'Digital_Chelsea_Greys_distribution_x_standardSNR=25dB_RMSE=0.00121.bin'

save_to_path = './Sim_results/'

text_file_name_split = data_file.split("_")
scheme = text_file_name_split[0]

with open(data_path+data_file, 'rb') as file:
        buffer = file.read()
file.close()
print('Results ' +data_file + ' loaded.')
result = np.frombuffer(buffer)

with open(data_path+expected_data_file, 'rb') as file:
        buffer = file.read()
file.close()
print('Expected results ' +expected_data_file + ' loaded.')
genie_result = np.frombuffer(buffer)

# Plot the waveforms
fig, ax = gf.plot_waveform_comparison(result, genie_result)
fig.savefig(save_to_path+'waveform_'+data_file[:-4:]+'_zoomedIn_1c5e4_2e4.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 3: (not incl.) waveform comparision for analog scheme
# ----------------------------------------------------------------------------

# Plot the signal-genie distribution with rasterized points but vector axes
ax_3c = gf.plot_density_scatter(genie_result, result, dot_size=2, bins=11, cmap='viridis', rasterized=True)
ax_3c.get_figure().savefig(save_to_path+'figure3c.svg', transparent=True, dpi=600, format='svg')

# reshape the data to the image
image_data = np.reshape(result, [300-2, 451-2])  # No padding, only grey channel processed

# plot the image without axes
plt.figure()
ax_5e_processed = plt.imshow(image_data, cmap='Greys')
plt.axis('off')  # Turn off axes

# Save the figure as SVG without white padding
save_file_path = save_to_path + './figure3c_image.svg'
plt.savefig(save_file_path, format='svg', bbox_inches='tight', pad_inches=0)

ax_3e = gf.plot_noise_dist(result, genie_result, scheme='Digital', bins=101)
ax_3e.get_figure().savefig(save_to_path+'figure3e.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 3f: RMSE-OSNR
# ----------------------------------------------------------------------------

# RMSE simulation results
df = pd.read_csv(data_path+'RMSE_SNR.csv',
        header=0, delimiter="\t", decimal=",")

snr = df['SNR (dB)'].to_numpy()
rmse_analog = df['RMSE (analog)'].to_numpy()
rmse_digital = df['RMSE (digital)'].to_numpy()

ax_3f = gf.plot_RMSE_measurements(snr, rmse_analog, rmse_digital, zoom=False)
ax_3f.get_figure().savefig(save_to_path+'figure3f.svg', format='svg', dpi=600)

ax_3f_zoom = gf.plot_RMSE_measurements(snr, rmse_analog, rmse_digital, zoom=True)
ax_3f_zoom.get_figure().savefig(save_to_path+'figure3f_zoom.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 4: (not incl.) waveform comparision for analog scheme
# ----------------------------------------------------------------------------

# Get and read the data file
data_path = './Experiment_data/'
data_file = 'Comb_9_lines.csv'
save_to_path = './Experiment_results/'

df = pd.read_csv(data_path+data_file,
        header=None, delimiter="\t", decimal=",")

wavelength = df[0].to_numpy()
power = df[1].to_numpy()

# remove redundancy measures (interpolations by the OSA)
new_wavelength = []
ind_wavelength = []
ind = 0
for i in wavelength:
    if i not in new_wavelength:
        ind_wavelength.append(ind)
        new_wavelength.append(i)
    ind+=1
new_power = power[ind_wavelength]


# Plot the optical spectrum
fig, ax = plt.subplots()
fig.set_size_inches((7, 5))
ax.plot(new_wavelength, new_power)
ax.axis([1541.5, 1545.5, -70, 10])
ax.set_xticks(np.arange(1542, 1546, 1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.set_yticks(np.arange(-70, 20, 20))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Power (dBm)')

fig.savefig(save_to_path+'Figure4c.svg', format='svg', dpi=600)


# ----------------------------------------------------------------------------
# Figure 5a: Original 'DTU' images
# ----------------------------------------------------------------------------

# Read the croped original image 
# See next cell for "how to process the original dng file"
with open('./HDR_image/DTU_16bit_RGB.bin', 'rb') as file:
    original_image = utils.bytes_to_data(file.read(), is_binary=0, output_format='uint16')

original_image = np.reshape(original_image, [397+2, 327+2, 3])

# We will use a croped image from the original image
# Pixels will be normalized
# The image will be processed through a "ADC" process to limit to discrete/integer pixel values from 0 to 65535
plt.close(102)
plt.figure(102)
# Color clamp and normalization is a MUST to "see" the 16-bit image
color_reduced_image = np.fix(original_image.astype(float) / 255.0)
image_to_show = (color_reduced_image - np.min(color_reduced_image)) / (np.max(color_reduced_image) - np.min(color_reduced_image))
imgplot=plt.imshow(image_to_show, vmin=0,vmax=1)
plt.axis('off')
plt.title("Original croped image: signal source for the experiment")
save_file_path = save_to_path + './Figure5a_DTU_original_croped_naive_colorClamp.png'
plt.imsave(save_file_path, image_to_show, format='png')

# # Show and plot the original image
# # NOTE: crop is implemented to the dng file to reduce the size of the data
# # NOTE: original processing method to read dng file into 16-bit 3D array:
# #       img = imageio.imread(img_path, format='RAW-FI')
# # This does not work on MacOS, due to the lack of libraw or probably other libraries.
# # There is still no rawpy package wheel available for arm64 architecture yet.
# #
# # Read the converted bin file for the 16-bit image (done in Windows)
# with open('./HDR_image/DTU_origin_16bit.bin', 'rb') as file:
#     original_image = utils.bytes_to_data(file.read(), is_binary=0, output_format='uint16')
    
# original_image = np.reshape(original_image, [4608, 3456, 3])

# # apply the crop
# image_cropBoundary = [1801, 2200, 1521, 1850]

# croped_original_image = original_image[image_cropBoundary[0]:image_cropBoundary[1],image_cropBoundary[2]:image_cropBoundary[3],::]

# plt.close(101)
# plt.figure(101)
# # Normalization is a MUST to "see" the 16-bit image
# # NOTE: 
# # To see the image in a correct colorscale, appropriate color clamp is requried
# # perhaps even during the dng file reading process.
# # Here is just to show that the image is there
# color_reduced_image = np.fix(original_image.astype(float) / 255.0)
# image_to_show = (color_reduced_image - np.min(color_reduced_image)) / (np.max(color_reduced_image) - np.min(color_reduced_image))
# imgplot=plt.imshow(image_to_show, vmin=0,vmax=1)
# plt.axis('off')
# plt.title("Original image")

# # We will use a croped image from the original image
# # Pixels will be normalized
# # The image will be processed through a "ADC" process to limit to discrete/integer pixel values from 0 to 65535
# plt.close(102)
# plt.figure(102)
# # Color clamp and normalization is a MUST to "see" the 16-bit image
# color_reduced_image = np.fix(croped_original_image.astype(float) / 255.0)
# image_to_show = (color_reduced_image - np.min(color_reduced_image)) / (np.max(color_reduced_image) - np.min(color_reduced_image))
# imgplot=plt.imshow(image_to_show, vmin=0,vmax=1)
# plt.axis('off')
# plt.title("Original croped image: signal source for the experiment")
# save_file_path = save_to_path + './Figure5a_DTU_original_croped_naive_colorClamp.png'
# plt.imsave(save_file_path, image_to_show, format='png')


# ----------------------------------------------------------------------------
# Figure 5a: Processed 'DTU' images
# ----------------------------------------------------------------------------

# Path settings
data_path = './Experiment_data/'
save_to_path = './Experiment_results/'

# Iterate through kernels and channels
kernels = ['prewitt_horizontal_up', 'sobel_horizontal_up', 'sharpen_D4']
channels = ['Reds', 'Greens', 'Blues']

# Save each image independently, no display
for current_kernel in kernels:
    channel_count = 0
    for current_channel in channels:
        # e.g., DSO_sortresult_DTU_prewitt_horizontal_up_Reds_16_bits_experiment_reconstruct
        data_file = 'DSO_sortresult_DTU_'+current_kernel+'_'+current_channel+'_16_bits_experiment_reconstruct.bin'
        
        with open(data_path + data_file, 'rb') as file:
            buffer = file.read()
        
        print('Results ' + data_file + ' loaded.')
        
        result = np.frombuffer(buffer)
        
        # reshape the data to the image
        image_data = np.reshape(result, [397, 327])  # Adjust dimensions as needed
        
        # plot the image without axes
        plt.figure()
        ax_5a = plt.imshow(image_data, cmap=current_channel)
        plt.axis('off')  # Turn off axes
        channel_count += 1
    
        # Save the figure as SVG without white padding
        save_file_path = save_to_path + './figure5a' + '_' + current_kernel + '_' + current_channel + '.png'
        plt.imsave(save_file_path, image_data, cmap=current_channel, format='png')
        plt.close()

# Group images according to kernels and show
for current_kernel in kernels:
    plt.figure()
    channel_count = 0
    for current_channel in channels:
        
        # e.g., DSO_sortresult_DTU_prewitt_horizontal_up_Reds_16_bits_experiment_reconstruct
        data_file = 'DSO_sortresult_DTU_'+current_kernel+'_'+current_channel+'_16_bits_experiment_reconstruct.bin'
        
        with open(data_path + data_file, 'rb') as file:
            buffer = file.read()
        
        print('Results ' + data_file + ' loaded.')
        
        result = np.frombuffer(buffer)
        
        # reshape the data to the image
        image_data = np.reshape(result, [397, 327])  # Adjust dimensions as needed
        
        # plot the image without axes
        plt.subplot(1, 3, channel_count+1)
        ax_5a = plt.imshow(image_data, cmap=current_channel)
        plt.axis('off')  # Turn off axes
        channel_count += 1


# ----------------------------------------------------------------------------
# Figure 5b: Waveform comparison, 'DTU' image, by PC and experiment
# ----------------------------------------------------------------------------

# Get and read experiment data
data_file = 'DSO_sortresult_DTU_prewitt_horizontal_up_Blues_16_bits_experiment_reconstruct.bin'
expected_data_file = 'DSO_sortresult_DTU_prewitt_horizontal_up_Blues_16_bits_PC_reconstruct.bin'

with open(data_path+data_file, 'rb') as file:
        buffer = file.read()
file.close()
print('Results ' +data_file + ' loaded.')
result = np.frombuffer(buffer)

with open(data_path+expected_data_file, 'rb') as file:
        buffer = file.read()
file.close()
print('Expected results ' +expected_data_file + ' loaded.')
genie_result = np.frombuffer(buffer)

# Plot the wave comparisons
fig_5b, _ = gf.plot_waveform_comparison(result, genie_result)
fig_5b.get_figure().savefig(save_to_path+'./figure5b.svg', format='svg', bbox_inches='tight')


# ----------------------------------------------------------------------------
# Figure 5c: Signal distribution against true values, 'DTU' image
# ----------------------------------------------------------------------------

ax_5c = gf.plot_density_scatter(genie_result, result, dot_size=2, bins=11, cmap='viridis', rasterized=True)
ax_5c.get_figure().savefig(save_to_path+'./figure5c.svg', format='svg', bbox_inches='tight')


# ----------------------------------------------------------------------------
# Figure 5c: Noise distribution, 'DTU' image
# ----------------------------------------------------------------------------

ax_5d = gf.plot_noise_dist(result, genie_result, scheme='DSO', bins=101)
ax_5d.get_figure().savefig(save_to_path+'./figure5d.svg', format='svg', bbox_inches='tight')


# ----------------------------------------------------------------------------
# Figure 5c: PER-OSNR, 'Chelsea' image, 8bit
# ----------------------------------------------------------------------------

# Read saved data
df = pd.read_csv(data_path+'Laser source OSNR_ER.csv',
        header=0, delimiter=";", decimal=",")
OSNR = df['OSNR'].to_numpy()
PER = df['ER'].to_numpy()

# Remove points with PER > 10^{-1}
x = OSNR[:-5:][::-1]
y = -np.log10(PER[:-5:])[::-1]

# Define the fitting function (e.g., a logarithmic function)
def logarithmic_fit(x, a, b):
    return a * np.log(x) + b

# Plot the PER curve
fitting_params, ax_5e = gf.plot_PER_measurements(
    x,y, logarithmic_fit, yticks=np.arange(5,0.9,-1), yticks_minor=np.arange(5,0.9,-0.1),
    xticks=np.arange(20,50,5), xticks_minor=np.arange(20,50,1),
    xlabel='OSNR (dB)', ylabel = r'$-\mathrm{log}_{10}\mathrm{PER}$'
    )
ax_5e.get_figure().savefig(save_to_path+'./figure5e.svg', format='svg', bbox_inches='tight')


# Plot the embeded 'Chelsea' images: the original image
# Get from scikit-image
chelsea = skimage.data.chelsea()

# flat the image into gray (to be processed)
chelsea_gray = skimage.color.rgb2gray(chelsea)

# plot the image without axes
plt.figure()
ax_5e_org = plt.imshow(chelsea_gray, cmap='Greys_r')
plt.axis('off')  # Turn off axes

# Save the figure as SVG without white padding
save_file_path = save_to_path + './figure5e_org.svg'
plt.savefig(save_file_path, format='svg', bbox_inches='tight', pad_inches=0)

# Plot the embeded 'Chelsea' images: the processed image at ONSR = 35 dB
data_file = 'DSO_sortresult_Chelsea_Prewitt_horizontal_up_Grey_experiment_reconstruct.bin'

with open(data_path + data_file, 'rb') as file:
    buffer = file.read()
print('Results ' + data_file + ' loaded.')

result = np.frombuffer(buffer)

# reshape the data to the image
image_data = np.reshape(result, [300-2, 451-2])  # No padding, only grey channel processed

# plot the image without axes
plt.figure()
ax_5e_processed = plt.imshow(image_data, cmap='Greys')
plt.axis('off')  # Turn off axes

# Save the figure as SVG without white padding
save_file_path = save_to_path + './figure5e_processed.svg'
plt.savefig(save_file_path, format='svg', bbox_inches='tight', pad_inches=0)


# ----------------------------------------------------------------------------
# Figure 5g: Confusion matrices
# ----------------------------------------------------------------------------

# retrieve the data
MNIST_data_true = 'MNIST_true'
MNIST_data_PC_pred = 'MNIST_PC_pred'
MNIST_data_exp_pred = 'MNIST_exp_pred'

with open(data_path + MNIST_data_true, 'rb') as file:
    y_true = utils.bytes_to_data(file.read(), is_binary=0, output_format='int')
with open(data_path + MNIST_data_PC_pred, 'rb') as file:
    y_PC_pred = utils.bytes_to_data(file.read(), is_binary=0, output_format='int')
with open(data_path + MNIST_data_exp_pred, 'rb') as file:
    y_exp_pred = utils.bytes_to_data(file.read(), is_binary=0, output_format='int')

# PC confusion matrix
outer_rect_kwargs = {'title': 'Calculated by a desktop computer'}
plt.rcParams['axes.titlepad'] = 10
heatmap = gf.plot_confusion_matrix(
    y_true, y_PC_pred, precision='.1f',outer_linewidth = 1,
    inner_linewidth = 2,
    outer_linecolor = '#000000',
    inner_linecolor = '#BFBFBF',
    outer_rect_kwargs = outer_rect_kwargs)
heatmap.get_figure().savefig(save_to_path+'./figure5g_PC.svg', format='svg', bbox_inches='tight')

# Experiment confusion matrix
outer_rect_kwargs = {'title': 'Experiment'}
plt.rcParams['axes.titlepad'] = 10
heatmap = gf.plot_confusion_matrix(
    y_true, y_exp_pred, precision='.1f',outer_linewidth = 1,
    inner_linewidth = 2,
    outer_linecolor = '#000000',
    inner_linecolor = '#BFBFBF',
    outer_rect_kwargs = outer_rect_kwargs)
heatmap.get_figure().savefig(save_to_path+'./figure5g_exp.svg', format='svg', bbox_inches='tight')


# ----------------------------------------------------------------------------
# Show the plots all together
# ----------------------------------------------------------------------------
plt.show()