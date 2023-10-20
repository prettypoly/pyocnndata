import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, NullFormatter, ScalarFormatter
import matplotlib_inline

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.interpolate import interpn
from sklearn.metrics import confusion_matrix

import seaborn as sns


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







def plot_density_scatter(x, y, figure_size=[5, 5], density_plot=True, bins=11, dot_size=2, cmap='viridis', rasterized=True, **kwargs):
    """
    Scatter plot colored by 2D histogram
    """
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # Normalize the density values from 0 to 1
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))

    plt.figure()
    scatter = plt.scatter(x, y, c=z_normalized, s=dot_size, rasterized=rasterized, **kwargs)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('Expected value')
    plt.ylabel('Result')
    plt.gcf().set_size_inches((figure_size[0], figure_size[1]))
    ax = plt.gca()
    ax.set_aspect('equal')  # set x and y axis to be equal in aspect ratio.

    if density_plot:
        norm = Normalize(vmin=0, vmax=1)  # Normalize density values from 0 to 1
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.05)
        cax.set_label('Density (normalized)')

    return scatter  # Return the scatter plot handle


def plot_waveform_comparison(waveform, waveform_genie, figure_size=[7, 3], pixel_indices=[1.5e4, 2e4], normalized_waveform=True, **kwargs):
    if not normalized_waveform:
        waveform = (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform))
        
    # Create a figure and two subplots
    fig, axs = plt.subplots(2, 1, figsize=(figure_size[0], figure_size[1]))
    
    # Plot the first waveform in the first subplot
    axs[0].plot(waveform, linewidth=0.5, alpha=1, **kwargs)
    axs[0].axis([pixel_indices[0], pixel_indices[1], -0.1, 1.1])
    axs[0].set_xticklabels([])  # Remove x-axis tick labels
    axs[0].set_yticks(np.arange(0, 1.1, 0.2))
    axs[0].set_ylabel('Normalized value')
    
    # Plot the second waveform in the second subplot
    axs[1].plot(waveform_genie, linewidth=0.5, alpha=1, **kwargs)
    axs[1].axis([pixel_indices[0], pixel_indices[1], -0.1, 1.1])
    axs[1].set_yticks(np.arange(0, 1.1, 0.2))
    axs[1].set_xlabel('Pixel index')
    axs[1].set_ylabel('Normalized value')
    
    return fig, axs


def lorentzian(x, x0, gamma, A):
    """
    Lorentzian function equation
    """
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


def plot_noise_dist(waveform, waveform_genie, scheme='Digital', bins=101, figure_size=[5, 5], 
                    bar_plot_kwargs=None, plot_kwargs=None):

    # bar_plot_kwargs and plot_kwargs should be dictionaries if needed. E.g.,
    # bar_plot_kwargs={'alpha': 0.6}, plot_kwargs={'linestyle': '--'}
    if bar_plot_kwargs is None:
        bar_plot_kwargs = {}  # Initialize as an empty dictionary if not provided
    if plot_kwargs is None:
        plot_kwargs = {}  # Initialize as an empty dictionary if not provided
    
    # Plot noise distribution by subtracting y^ with y
    ybar_minus_y = waveform - waveform_genie
    
    # Bin it
    dist, bin_edges = np.histogram(ybar_minus_y, bins)
    
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = dist / float(dist.sum())
    
    # Get the midpoints of every bin
    bin_middles = (bin_edges[1:] + bin_edges[:-1]) / 2.
    
    # Compute the bin-width
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # This line will use the plot_kwargs dictionary if it's provided, 
    # and if not, it will pass an empty dictionary to **kwargs.
    ax.bar(bin_middles, bin_probability, width=bin_width, color='#6667AB', alpha=0.5, **bar_plot_kwargs)
    
    fig.set_size_inches((figure_size[0], figure_size[1]))
    
    ax.set_xlabel('Measured minus expected')
    ax.set_ylabel('Probability')
    print(f"Maximum probability is: {np.max(bin_probability):.8f}.")
    
    # Gaussian fitting
    if scheme == 'Analog':
        from scipy.stats import norm
        
        # Fit a Gaussian distribution to the data
        mu, sigma = norm.fit(ybar_minus_y)
        
        # Generate the x values for the Gaussian curve
        x = np.linspace(bin_edges[0], bin_edges[-1], 100)
        
        # Compute the corresponding y values using the Gaussian distribution parameters
        y = norm.pdf(x, mu, sigma)
    
        # Scale the Gaussian curve to match the probability of the bins
        y_scaled = y / np.sum(y) #* bin_width * len(ybar_minus_y)
    
        # Plot the Gaussian fit
        ax.plot(x, y_scaled, linewidth=2, color='#884C5E', **plot_kwargs)
        ax.axis([-0.15, 0.15, 0, np.ceil(np.max(y_scaled) / 0.01) * 0.01])
        
        print(f"\u03C3 is: {sigma:.3f}.")
        print(f"Calculation precision is: {np.log2(1 / (3 * sigma)):.3f} bits (3\u03C3).")
        print(f"Calculation precision is: {np.log2(1 / sigma):.3f} bits (\u03C3).")
    
    elif scheme == 'Digital':
        y_scaled = bin_probability
        ax.axis([-0.15, 0.15, 0, 1.1])
    
    elif scheme == 'DSO':
        # Digital scheme, do nothing for now [FIXME]
        y_scaled = bin_probability
        ax.axis([-0.15, 0.15, 0, 1.1])
    
    print(f"Sum of all probability is: {np.sum(y_scaled):.3f} (Expected to be 1).")
    
    return ax



def plot_RMSE_measurements(snr, rmse_analog, rmse_digital, zoom=True, zoom_range=[15, 45], figure_size=[5, 5], **kwargs):

    fig, ax = plt.subplots()
    ax.plot(snr, rmse_analog, 'o-', color='#6667AB', label='Analog', **kwargs)
    ax.plot(snr, rmse_digital, '*-', color='#D29381', label='Digital', **kwargs)
    fig.set_size_inches((figure_size[0], figure_size[1]))
    
    if zoom:
        snr_zoomed = snr[np.isin(snr, np.arange(zoom_range[0],zoom_range[1]+1,5))]
        rmse_analog_zoomed = rmse_analog[np.isin(snr, np.arange(zoom_range[0],zoom_range[1]+1,5))]
        rmse_digital_zoomed = rmse_digital[np.isin(snr, np.arange(zoom_range[0],zoom_range[1]+1,5))]
        snr_zoom = snr
        ax.plot(snr_zoomed, rmse_analog_zoomed, 'o-', color='#6667AB')
        ax.plot(snr_zoomed, rmse_digital_zoomed, '*-', color='#D29381')
        fig.set_size_inches((4, 3))
        ax.axis([zoom_range[0], zoom_range[1], -0.01, 0.1])
        ax.set_xticks(np.arange(zoom_range[0], zoom_range[1]+1, 10))
        ax.set_xticks(np.arange(zoom_range[0], zoom_range[1]+1, 5), minor=True)
        ax.set_yticks(np.arange(0, 0.11, 0.02))
        ax.tick_params(which='minor')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('RMSE')
    else:
        ax.axis([-5, 55, -0.05, 0.55])
        ax.set_xticks(np.arange(0, 55, 10))
        ax.set_xticks(np.arange(0, 55, 5), minor=True)
        ax.set_yticks(np.arange(0, 0.6, 0.1))
        ax.tick_params(which='minor')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('RMSE')
    
    ax.legend()
    # plt.show()

    return ax


def plot_PER_measurements(
    x, y, fitting_function, 
    yticks=np.arange(1,6,1), 
    yticks_minor=np.arange(1,5.1,0.1),
    xticks=np.arange(20,51,10),
    xticks_minor=np.arange(20,51,1),
    grid=False,
    xlabel='OSNR',
    ylabel='PER',
    figure_size=[5,5],
    scatter_kwargs=None,  # Keyword arguments for scatter plot
    line_plot_kwargs=None  # Keyword arguments for line plot
    ):
    """
    Fit a curve to the given data and plot the data points and the fitted curve.

    Parameters:
    - x (list or numpy array): x-values of the data.
    - y (list or numpy array): y-values of the data.
    - fitting_function (callable): The fitting function to use for curve fitting.

    Returns:
    - params (tuple): Fitted parameters of the curve.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object of the plot.

    # Example usage:
    x = [23.0379, 25.0102, 27.02135, 29.00665, 31.06725, 33.0742, 35.0448]
    y = [1.30094315, 1.68867005, 2.13786862, 3.14691047, 3.53017798, 4.08990945, 4.56703071]
    
    # Define the fitting function (e.g., a logarithmic function)
    def logarithmic_fit(x, a, b):
        return a * np.log(x) + b
    
    # Call the function with the data and fitting function
    fit_params, ax = plot_PER_measurements(x, y, logarithmic_fit)
    """

    # Perform the curve fit
    params, covariance = curve_fit(fitting_function, x, y)

    # Print the name of the fitting method and the fitted parameters
    print(f"Fitting Method: {fitting_function.__name__}")
    print(f"Fitted Parameters: {params}")

    # Generate x values for the fitted curve
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = fitting_function(x_fit, *params)

    # Create a scatter plot for the data
    fig, ax = plt.subplots()
    scatter_kwargs = scatter_kwargs or {}  # Initialize with an empty dictionary if None
    ax.scatter(x, y, label='Data', **scatter_kwargs)

    # Plot the fitted curve as a line plot
    line_plot_kwargs = line_plot_kwargs or {}  # Initialize with an empty dictionary if None
    ax.plot(x_fit, y_fit, label='Fitted Curve', color='red', **line_plot_kwargs)
    
    # Axis limit
    ax.axis([xticks[0], xticks[1], yticks[0], yticks[1]])

    # Set figure size
    fig.set_size_inches((figure_size[0], figure_size[1]))

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Manually set major and minor ticks on the y-axis
    ax.set_yticks(yticks)
    ax.yaxis.set_minor_locator(MultipleLocator(np.abs(yticks_minor[1]-yticks_minor[0])))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Set major and minor ticks on the x-axis
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(MultipleLocator(np.abs(xticks_minor[1]-xticks_minor[0])))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Add gridlines for minor ticks
    if grid:
        ax.grid(which='minor', linestyle='--', linewidth=0.5)

    # Add labels and a legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # Show the plot
    # plt.show()

    return params, ax


# Two kwargs for contol extra behaviour of the figure plots
# They take dictionary based argument
# # e.g., Define kwargs for the outer rectangular plot
# outer_rect_properties = {
#     'linewidth': 3,
#     'edgecolor': 'red',
#     'facecolor': 'none',
# }
# heatmap_kwargs={
# "annot": False, 
# "linewidths": 0.1
# }
# def plot_confusion_matrix(y_true, y_pred, 
#                           precision='.1f', figure_size=[5, 5], 
#                           color_tone='#6667AB', base_color='#FFFFFF', 
#                           num_shades=10,
#                           outer_linewidth=2,  # Adjust the outer line width (user can customize this)
#                           inner_linewidth=0.5,
#                           outer_linecolor='#000000',  # Outer line color
#                           inner_linecolor='#BFBFBF',  # Inner line color
#                           outer_rect_kwargs=None,  # **kwargs for the outer rectangular plot
#                           ):
    
#     # Calculate the confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Transpose the confusion matrix to reverse the axes, so that x represents true and y represents predicted
#     cm_reversed = cm.T

#     # Automatically determine the number of classes for x-axis tick labels
#     n_classes_x = cm_reversed.shape[1]
    
#     # Normalize the confusion matrix to display percentages (0-100)
#     cm_normalized = cm_reversed.astype('float') / cm_reversed.sum(axis=1)[:, np.newaxis] * 100

#     # Create a custom colormap with a gradient of shades starting from white and transitioning to the base color
#     colors = [base_color]  # Start color
#     for i in range(num_shades):
#         shade = sns.light_palette(color_tone, input='hex', n_colors=num_shades)[i]
#         colors.append(shade)
    
#     cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))

    
#     # Create a heatmap of the normalized confusion matrix using the custom colormap
#     plt.figure(figsize=(figure_size[0], figure_size[1]))
#     heatmap = sns.heatmap(cm_normalized, annot=True, fmt=precision, cmap=cmap, cbar=False, square=True, 
#                 xticklabels=np.arange(n_classes_x), yticklabels=np.arange(n_classes_x),
#                 linewidths=0.5, linecolor=inner_linecolor, cbar_kws={"color": inner_linecolor})
    
#     # Set x and y tick labels to a specified fontsize
#     heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
#     heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

#     # Control the line width and color of the outer rectangular plot
#     for _, spine in heatmap.spines.items():
#         spine.set_visible(True)
#         spine.set_edgecolor(outer_linecolor)
#         spine.set_linewidth(outer_linewidth)  # Use the user-defined value

#     # Apply **kwargs to the outer rectangular plot
#     if outer_rect_kwargs is not None:
#         ax_rect = plt.gca()
#         ax_rect.set(**outer_rect_kwargs)

#     plt.xlabel('True Labels')
#     plt.ylabel('Predicted Labels')
#     plt.show()

#     return heatmap

def plot_confusion_matrix(y_true, y_pred, 
                          precision='.1f', figure_size=[5, 5], 
                          color_tone='#6667AB', base_color='#FFFFFF', 
                          num_shades=10,
                          outer_linewidth=2,  # Adjust the outer line width (user can customize this)
                          inner_linewidth=0.5,
                          outer_linecolor='#000000',  # Outer line color
                          inner_linecolor='#BFBFBF',  # Inner line color
                          outer_rect_kwargs=None,  # **kwargs for the outer rectangular plot
                          heatmap_kwargs=None,  # **kwargs for the heatmap
                          ):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Transpose the confusion matrix to reverse the axes, so that x represents true and y represents predicted
    cm_reversed = cm.T

    # Automatically determine the number of classes for x-axis tick labels
    n_classes_x = cm_reversed.shape[1]
    
    # Normalize the confusion matrix to display percentages (0-100)
    cm_normalized = cm_reversed.astype('float') / cm_reversed.sum(axis=1)[:, np.newaxis] * 100

    # Create a custom colormap with a gradient of shades starting from white and transitioning to the base color
    colors = [base_color]  # Start color
    for i in range(num_shades):
        shade = sns.light_palette(color_tone, input='hex', n_colors=num_shades)[i]
        colors.append(shade)
    
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))

    
    # Create a heatmap of the normalized confusion matrix using the custom colormap
    plt.figure(figsize=(figure_size[0], figure_size[1]))
    
    # Apply **kwargs to the heatmap
    if heatmap_kwargs is not None:
        heatmap = sns.heatmap(cm_normalized, annot=True, fmt=precision, cmap=cmap, cbar=False, square=True, 
            xticklabels=np.arange(n_classes_x), yticklabels=np.arange(n_classes_x),
            linewidths=0.5, linecolor=inner_linecolor, cbar_kws={"color": inner_linecolor}, **heatmap_kwargs)
    else:
        heatmap = sns.heatmap(cm_normalized, annot=True, fmt=precision, cmap=cmap, cbar=False, square=True, 
            xticklabels=np.arange(n_classes_x), yticklabels=np.arange(n_classes_x),
            linewidths=0.5, linecolor=inner_linecolor, cbar_kws={"color": inner_linecolor})
    
    # Set x and y tick labels to a specified fontsize
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

    # Control the line width and color of the outer rectangular plot
    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor(outer_linecolor)
        spine.set_linewidth(outer_linewidth)  # Use the user-defined value

    # Apply **kwargs to the outer rectangular plot
    if outer_rect_kwargs is not None:
        ax_rect = plt.gca()
        ax_rect.set(**outer_rect_kwargs)

    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    # plt.show()

    return heatmap