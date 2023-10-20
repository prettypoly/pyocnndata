import sys
import numpy as np
import skimage, imageio
from skimage.color import lab2rgb, lch2lab, rgb2gray
import matplotlib.pyplot as plt


def select_image(img_label, img_path='./', img_format='RAW-FI'):
    # match - case can be used if python version is > 3.10 (2021)
    # here we use if - else statement
    if img_label == 'user_input':
        img = imageio.imread(img_path, format=img_format)
    elif img_label == 'chelsea':
        img = skimage.data.chelsea()
    elif img_label == 'camera':
        img = skimage.data.camera()
    elif img_label == 'coins':
        img = skimage.data.coins()
    elif img_label == 'astronaut':
        img = skimage.data.astronaut()
    elif img_label == 'motorcycle':
        img = skimage.data.stereo_motorcycle()
    elif img_label == 'rocket':
        img = skimage.data.rocket()
    else:
        raise SystemExit("Image not found.")
    return img


def select_channel(img, ch_label):  
    # select image channel. support channels: gray, red, green, blue
    # if image has only one channel(gray), then quit directly  
    # match - case can be used if python version is > 3.10 (2021)
    # here we use if - else statement
    if img.ndim == 2: # grey image
        # do nothing
        img_out = img
    elif img.ndim == 3: # color image
        if ch_label == 'Greys':
            img_out = rgb2gray(img)
        elif ch_label == 'Reds':
            img_out = img[::,::,0]
        elif ch_label == 'Greens':
            img_out = img[::,::,1]
        elif ch_label == 'Blues':
            img_out = img[::,::,2]
        else:
            raise SystemExit("Image channel should be one of Greys, Reds, Greens, or Blues.")
    else:
        raise SystemExit("An image must contain 1 (grey) or 3 (color) channels.")
    # normalize the image
    img_out = img_out - img_out.min()
    img_out = img_out/img_out.max()
    return img_out


def flatten_image(img):
    # flatten image to gray scale image if it is colored
    if img.shape[-1] > 2:
        img = skimage.color.rgb2gray(img)
    # normalize the image
    img = img/np.max(img)
    return img


def check_dimensions(img, conv_filter):
    # check if image contains multiple channels
    # this requries the filter to have a dimension of 4
    # the last dimension of the filter corresponds to the channels of the image
    # filter: (NOfFilterPerChannel/Size, FltX, FltY, ChannelN)
    # image:  (ImgX, ImgY, ChannelN)
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        # if yes, then the dimensions of image channels and filter must match
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in image and filter must match")
            sys.exit()
    # check if the filter shape is legal, i.e., odd square matrix
    # Check if filter dimensions are equal.
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("Error: Filter must be a square matrix.")
        sys.exit()
    # Check if the filter dimensions are odd
    if conv_filter.shape[1] % 2 == 0:
        print("Error: Filter must have an odd size.")
        sys.exit()


def select_filter(flt_type):
    # choose which filter to use, or construct by user
    flt_switcher = {
        "vertical_right": np.array([[     [-1,  0,  1],
                                          [-1,  0,  1],
                                          [-1,  0,  1]      ]]),
        "vertical_left": np.array([[      [ 1,  0, -1],
                                          [ 1,  0, -1],
                                          [ 1,  0, -1]      ]]),
        "horizontal_up": np.array([[      [ 1,  1,  1],
                                          [ 0,  0,  0],
                                          [-1, -1, -1]      ]]),
        "horizontal_down": np.array([[    [-1, -1, -1],
                                          [ 0,  0,  0],
                                          [ 1,  1,  1]      ]]),
        "left_up": np.array([[            [ 1,  1,  0],
                                          [ 1,  0, -1],
                                          [ 0, -1, -1]      ]]),
        "right_down": np.array([[         [-1, -1,  0],
                                          [-1,  0,  1],
                                          [ 0,  1,  1]      ]]),
        "left_down": np.array([[          [ 0, -1, -1],
                                          [ 1,  0, -1],
                                          [ 1,  1,  0]      ]]),
        "right_up": np.array([[           [ 0,  1,  1],
                                          [-1,  0,  1],
                                          [-1, -1,  0]      ]]),
        "sobel_vertical_right": np.array([[     [-1,  0,  1],
                                                [-2,  0,  2],
                                                [-1,  0,  1]      ]]),
        "sobel_vertical_left": np.array([[      [ 1,  0, -1],
                                                [ 2,  0, -2],
                                                [ 1,  0, -1]      ]]),
        "sobel_horizontal_up": np.array([[      [ 1,  2,  1],
                                                [ 0,  0,  0],
                                                [-1, -2, -1]      ]]),
        "sobel_horizontal_down": np.array([[    [-1, -2, -1],
                                                [ 0,  0,  0],
                                                [ 1,  2,  1]    ]]),
        "sobel_left_up": np.array([[      [ 2,  1,  0],
                                          [ 1,  0, -1],
                                          [ 0, -1, -2]      ]]),
        "sobel_right_down": np.array([[   [-2, -1,  0],
                                          [-1,  0,  1],
                                          [ 0,  1,  2]      ]]),
        "sobel_left_down": np.array([[    [ 0, -1, -2],
                                          [ 1,  0, -1],
                                          [ 2,  1,  0]      ]]),
        "sobel_right_up": np.array([[     [ 0,  1,  2],
                                          [-1,  0,  1],
                                          [-2, -1,  0]      ]]),
        "identical": np.array([[          [ 0,  0,  0],
                                          [ 0,  1,  0],
                                          [ 0,  0,  0]      ]]),
        "sharpen_D4": np.array([[         [ 0, -1,  0],
                                          [-1,  4, -1],
                                          [ 0, -1,  0]      ]]),
        "sharpen_D8": np.array([[         [-1, -1, -1],
                                          [-1,  8, -1],
                                          [-1, -1, -1]      ]]),
        "Laplace_D4": np.array([[         [ 0,  1,  0],
                                          [ 1, -4,  1],
                                          [ 0,  1,  0]      ]]),
        "Laplace_D8": np.array([[         [ 1,  1,  1],
                                          [ 1, -8,  1],
                                          [ 1,  1,  1]      ]])
    }
    return flt_switcher.get(flt_type)


def get_conv_result_size(img, conv_filter):
    output_size = np.zeros(2)
    output_size[0] = img.shape[0] - conv_filter.shape[2] + 1
    output_size[1] = img.shape[1] - conv_filter.shape[2] + 1
    return output_size


def restruct_image(img, conv_filter):
    check_dimensions(img, conv_filter)
    # disassemble the image according to the size of the filter
    # first: row, then: column
    conv_filter_size = conv_filter.shape[2]
    conv_filter_len = conv_filter.shape[2]**2
    output_size_r = img.shape[0] - conv_filter_size + 1
    output_size_c = img.shape[1] - conv_filter_size + 1
    output = np.zeros((output_size_r)*(output_size_c)*conv_filter_len)
    for r in np.uint16(np.arange(0, output_size_r)):
        for c in np.uint16(np.arange(0, output_size_c)):
            # get current image region
            curr_region = img[r:r+conv_filter_size, c:c+conv_filter_size]
            # assemble the output
            # column-wise indexing
            curr_region = np.reshape(curr_region, conv_filter_len) 
            output_ind = r*output_size_c*conv_filter_len + c*conv_filter_len
            output[output_ind:output_ind + conv_filter_len] = curr_region
    #
    # assign disassembled image to signal channels
    # number of signal channels equals to length of filter
    output = np.reshape(output, (-1, conv_filter_len)).T
    return output


# Convolution of the image and the filter(s)
def conv(img, conv_filter):
    check_dimensions(img, conv_filter)
    # An empty feature map to hold the output of convolving the filter(s) with
    # the image: the dimension of the feature map is calculated.
    feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                             img.shape[1] - conv_filter.shape[1] + 1,
                             conv_filter.shape[0]))
    #
    # convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        # get the current filter
        curr_filter = conv_filter[filter_num, :]
        # check if there are multiple channels for the single filter.
        # If so, then each channel will convolve the image.
        # The result of all convolutions are summed to return a single feature
        # map.
        # NOTE: If the image to be convolved has more than one channel,
        # then the filter must has a depth equal to such number of channels.
        # Convolution in this case is done by convolving each image channel
        # with its corresponding channel in the filter.
        # Finally, the sum of the results will be the output feature map.
        # If the image has just a single channel,
        # then convolution will be straight forward.
        if len(curr_filter.shape) > 2:
            # Array holding the sum of all feature maps
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
            # convolving each channel with the image and summing the reulsts
            for ch_num in range(1, curr_filter.shape[-1]):
                conv_map = conv_map + conv_(img[:, :, ch_num],
                                            curr_filter[:, :, ch_num])
        else:  # only one channel in the filter
            conv_map = conv_(img, curr_filter)
        # holding feature map with the current filter
        feature_maps[:, :, filter_num] = conv_map
    # returning all feature maps.
    return feature_maps


# The arthmeric convolution
def conv_(img, conv_filter):
    filter_size = conv_filter.shape[0]
    result = np.zeros((img.shape))
    # looping through the image to apply the conv operation
    # first: row, then: column
    for r in np.uint16(np.arange(0, img.shape[0] - filter_size + 1)):
        for c in np.uint16(np.arange(0, img.shape[1] - filter_size + 1)):
            # getting the current image region
            curr_region = img[r:r+filter_size, c:c+filter_size]
            # element-wise multiplication between current image region and the
            # filter
            curr_result = curr_region * conv_filter
            # summing the result of multiplication
            conv_sum = np.sum(curr_result)
            # saving the summation in the convolution layer feature map
            result[r, c] = conv_sum
    # clipping the outliers of the result matrix
    final_result = result[0:result.shape[0] - filter_size + 1,
                          0:result.shape[1] - filter_size + 1]
    return final_result

'''
#def plot_image(selection):
    if selection == 'original':
    else:

imgplot = plt.imshow(img)
plt.title("The original Chelsea")
imgplot.set_cmap("gray")
imgplot.axes.get_xaxis().set_visible(False)
imgplot.axes.get_yaxis().set_visible(False)
plt.show()
imgplot = plt.imshow(l1_feature_map[:, :, 0])
plt.title("Vertical edge detection")
imgplot.set_cmap("gray")
imgplot.axes.get_xaxis().set_visible(False)
imgplot.axes.get_yaxis().set_visible(False)
plt.show()
imgplot = plt.imshow(l1_feature_map[:, :, 1])
plt.title("Horizontal edge detection")
imgplot.set_cmap("gray")
imgplot.axes.get_xaxis().set_visible(False)
imgplot.axes.get_yaxis().set_visible(False)
plt.show()
'''
