import numpy as np


# > sample codes for 2-level (bits) data:
# > transmitter-side:
# bytesToSave = utils.data_to_bytes(bits_ADC, is_binary=1)
# > receiver side:
# restoredData = utils.bytes_to_data(bytesToSave, is_binary=1, shape=bits_ADC.shape)
#
# > sample codes for multi-level data (int levels)
# > transmitter-side:
# bytesToSave = utils.data_to_bytes(wave, is_binary=0, data_precision='uint8')
# > receiver side:
# restoredData = utils.bytes_to_data(bytesToSave, is_binary=0, output_format='uint8', shape=wave.shape)
#
# > sample codes for floating-point data:
# > transmitter-side:
# bytesToSave = utils.data_to_bytes(wave, is_binary=0, data_precision='float64')
# > receiver side:
# restoredData = utils.bytes_to_data(bytesToSave, is_binary=0, output_format='float64', shape=wave.shape)

        
def data_to_bytes(data_input, is_binary=0, data_precision='uint8'):
    #
    # data_format and bits_format are only useful when multi_level_data=1
    #
    # flatten the input array 
    # Note this is not complied with the column-wise signal tributaries order (i.e., 'F' order)
    # because this 'C' order is much easier for restoring the bytes back to signal.
    data_input = data_input.flatten(order='C')
    # convert data/bits to bytes
    if is_binary:
        # pack bits to numerical uint8 array and convert to bytes
        buffer = bytearray(np.packbits(data_input.astype('bool'), axis=0, bitorder='big'))
    else:
        # convert multi-level data with designated precision to uint8 arrays, and then to bytes
        buffer = bytearray(data_input.astype(data_precision).view(dtype='ubyte'))
    return buffer
        

def bytes_to_data(bytes_input, is_binary=0, output_format='bool', shape=[]):
    # convert bytes to data with designated precision
    if is_binary:
        # byte buffer contains uint8 array
        data_Nbit_blocks = np.frombuffer(bytes_input, dtype='uint8') 
        # need to unpack uint8 array to real bits
        data = np.unpackbits(data_Nbit_blocks, axis=0, bitorder='big').astype('bool') 
    else:
        # unpack the bytes according to precision
        data = np.frombuffer(bytes_input, dtype=output_format) 
    if shape!=[] and len(shape)==2:
        data = data[0:shape[0]*shape[1]].reshape(shape)
    elif shape!=[] and len(shape)==1:
        data = data[0:shape[0]].reshape(shape)
    return data


def find_levels(sigIn, expected_levels=4, max_iteration=10000):
    print(f"Trying to find {expected_levels} signal levels from data.")
    n=0
    clusters = np.array([])
    while 1:
        if sigIn[n] not in clusters:
            clusters = np.append(clusters, sigIn[n])
        if len(clusters)==expected_levels:
            break
        if n>max_iteration:
            print(f'{len(clusters)} levels found. Scanning too long to find required levels, quit...')
            break
        n = n + 1
    clusters.sort()
    print(f'{len(clusters)} signal level clusters found: {clusters}')
    return clusters, n