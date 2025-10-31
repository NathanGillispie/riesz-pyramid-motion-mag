#!/usr/bin/env python
'''
Magnifies motion 
See the following reference for more information:

    Riesz Pyramids for Fast Phase-Based Video Magnification
    Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
    Computational Photography (ICCP), 2014 IEEE International Conference on

DON'T FORGET TO SET PARAMETERS AT TOP

Also you may see comments about color spaces. I prefer RGB, but you can uncomment
the YCbCr stuff because that's what the authors use.
'''

# Magnification factor. Normal range: [1,10]
ALPHA = 10
# Temporal filter parameters (Hz)
f_lo = .5
f_hi = 1.2
# Your video sampling rate (frames/sec)
# Important to get temporal filters right :)
fs = 15.003

from riezs_pyramid import buildNewPyr, reconNewPyr
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.signal as signal


def amplify_spatial_lpyr_temporal_iir(pyr_frames, alpha, lambda_c, r1, r2, chromAttenuation):
    '''
    Spatial Filtering: Laplacian pyramid
    Temporal Filtering: substraction of two IIR lowpass filters

    y1[n] = r1*x[n] + (1-r1)*y1[n-1]
    y2[n] = r2*x[n] + (1-r2)*y2[n-1]
    (r1 > r2)

    y[n] = y1[n] - y2[n]
    '''
    numFrames = len(pyr_frames)
    startIndex = 1
    endIndex = numFrames-10 # CHECK THIS

    numFrames = len(pyr_frames)
    numChannels = len(pyr_frames[0])
    numLevels = len(pyr_frames[0][0])
    vidWidth, vidHeight = pyr_frames[0][0][0].shape

    lowpass1 = pyr_frames[0]
    lowpass2 = pyr_frames[0]

    output = []

    for i in tqdm(range(startIndex,endIndex), ncols=80, desc='Process'):
        pyr = pyr_frames[i]

        # temporal filtering
        lowpass1 = (1-r1)*lowpass1 + r1*pyr
        lowpass2 = (1-r2)*lowpass2 + r2*pyr
        filtered = lowpass1 - lowpass2
        # amplify each spatial frequency bands according to Figure 6 of the paper
        ind = numLevels
        delta = lambda_c/8/(1+alpha)

        # the factor to boost alpha above the bound we have in the
        # paper. (for better visualization)
        exaggeration_factor = 2;

        # compute the representative wavelength lambda for the lowest spatial 
        # freqency band of Laplacian pyramid

        lmbda = (vidWidth**2 + vidHeight**2)**0.5/3 # 3 is experimental constant

        for l in range(numLevels-1,-1,-1):
            level_sm = pyr[l].shape
            indices = (ind-level_sm[0]*level_sm[1], ind-1)
            currAlpha = lmbda/delta/8 - 1
            # compute modified alpha for this level
            currAlpha = currAlpha*exaggeration_factor
            if (l == nLevels-1) or (l==0): # ignore the highest and lowest frequency band
                filtered[*indices,:] = 0
            elif (currAlpha > alpha): # representative lambda exceeds lambda_c
                filtered[*indices,:] = alpha*filtered[*indices,:]
            else:
                filtered[*indices,:] = currAlpha*filtered[*indices,:]
            ind -= level_sm[0]*level_sm[1]
            # go one level down on pyramid,
            # representative lambda will reduce by factor of 2
            lmbda /= 2

        output.append(filtered)
    return output

def image_sequence_to_frames(frames_dir='frames_in'):
    '''
    Hopefully you have enough memory for frames ;)
    '''
    assert type(frames_dir) == str

    import os
    image_filenames = sorted(os.listdir(frames_dir))
    assert len(image_filenames) > 0

    num_images = len(image_filenames)
    num_channels = 3
    with Image.open(f'{frames_dir}/{image_filenames[0]}') as im:
        resolution = im.size

    # Declare array that holds all images
    frames = np.zeros((num_images, num_channels, *resolution), dtype=float)

    for i, image_filename in enumerate(tqdm(image_filenames, ncols=80, desc='Read Frames')):
        with Image.open(f'{frames_dir}/{image_filename}') as im:
            # imdata = np.array(im.convert('YCbCr').getdata())
            imdata = np.array(im.getdata())
            imdata = np.reshape(imdata, (resolution[1], resolution[0], num_channels))
            frames[i] = imdata.transpose((2,1,0))
    return frames


def frames_to_image_sequence(frames, frames_dir='frames_out'):
    '''Same as above'''
    assert type(frames_dir) == str

    numFrames = len(frames)
    assert numFrames < 1000

    for i in tqdm(range(numFrames), ncols=80, desc='Write Frame'):
        im_data = frames[i].transpose((2,1,0))
        # new_image = Image.fromarray(im_data.astype('uint8'),'YCbCr').convert('RGB')
        new_image = Image.fromarray(im_data.astype('uint8'))
        new_image.save(f'{frames_dir}/{i:03g}.png')


def frames_to_riesz_pyr(frames):
    '''
    Convert a list of frames (numFrames,3,width,height) to a list of pyramid frames:
    [numLevels, (numFrames, 3, level_width, level_height)]
    '''
    pyr_frames = []
    numFrames = len(frames)
    numChannels = len(frames[0])

    first_pyr = buildNewPyr(frames[0,0,:,:])
    numLevels = len(first_pyr)

    for l in range(numLevels):
        size_sm = first_pyr[l].shape
        pyr_frames.append(np.zeros((numFrames,numChannels,*size_sm)))

    pbar = tqdm(range(numFrames), ncols=80, desc='Frames->Pyr')
    import concurrent.futures
    def process_frame(frame):
        for channel in range(numChannels):
            pyr = buildNewPyr(frames[frame][channel])
            for l in range(numLevels):
                pyr_frames[l][frame,channel] = pyr[l]
        pbar.update()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        ex.map(process_frame, range(numFrames))

    return pyr_frames


def riesz_pyr_to_frames(pyr_frames):
    '''
    Opposite of above
    '''
    numLevels = len(pyr_frames)
    numFrames = len(pyr_frames[0])
    numChannels = len(pyr_frames[0][0])

    # Object that will hold our temporary pyramid while we reconstruct frame
    pyr = []
    for l in range(numLevels):
        level_shape = pyr_frames[l][0][0].shape
        pyr.append(np.zeros(level_shape))

    resolution = pyr[0].shape # lowest level shape

    frames = np.zeros((numFrames,numChannels,*resolution), dtype=float)

    for f in tqdm(range(numFrames), ncols=80, desc='Pyr->Frames'):
        for channel in range(numChannels):
            for l in range(numLevels):
                pyr[l] = pyr_frames[l][f][channel]
            frames[f][channel] = reconNewPyr(pyr)

    return frames


def fft_amplify_pyr_frames(pyr_frames, fft_level=[-4,-3]):
    '''
    Returns motion amplified pyr_frames to add together before reconstruction.
    '''
    NUM_FRAMES = len(pyr_frames[0])
    bandpass = signal.firwin(numtaps=NUM_FRAMES, cutoff=(f_lo, f_hi), fs=fs, pass_zero=False)
    transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

    return_pyr = []
    for l in range(len(pyr_frames)):
        return_pyr.append(np.zeros_like(pyr_frames[l]))

    for fl in fft_level:
        pyr_fft = np.fft.fft(pyr_frames[fl], axis=0).astype(np.complex64)
        _filtered_pyramid = pyr_fft * transfer_function[:, None, None, None].astype(np.complex64)
        filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real * ALPHA
        return_pyr[fl]=filtered_pyramid

    return return_pyr


def reconstruct_amplified_pyr(pyr_frames, amplified_pyr_frames):
    '''
    Takes pyr_frames and adds filtered amplified_pyr_frames, then returns
    reconstructed frames.

    DON'T FORGET TO INCLUDE chromAttenuation
    DON'T FORGET TO CLIP THE OUTPUT AFTER ADDING MAGNIFICATION
    '''
    for l in range(len(pyr_frames)):
        pyr_frames[l] = amplified_pyr_frames[l]

    ret = riesz_pyr_to_frames(pyr_frames)
    ret = np.clip(ret, a_min=0, a_max=255.1)
    return ret


if __name__=='__main__':
    frames = image_sequence_to_frames()
    pyr_frames = frames_to_riesz_pyr(frames)
    amplified_pyr_frames = fft_amplify_pyr_frames(pyr_frames)
    frames = reconstruct_amplified_pyr(pyr_frames, amplified_pyr_frames)
    frames_to_image_sequence(frames)

