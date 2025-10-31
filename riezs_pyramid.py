#!/usr/bin/env python
'''
Helper functions for generating Riezs Pyramids for Phase-based video magnification.
This is mostly a line-by-line translation of the supplemental information for

    Riesz Pyramids for Fast Phase-Based Video Magnification
    Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
    Computational Photography (ICCP), 2014 IEEE International Conference on
'''

import numpy as np
import scipy


def conv2(A, B, mode):
    '''improves readability / aligns with Matlab code better'''
    return scipy.signal.convolve2d(A, B, mode=mode)


def filterToChebyCoeff(taps):
    '''
    Returns the Chebyshev polynomial coefficients corresponding to a symmetric 1D filter
    '''
    # taps should be an odd symmetric filter
    M = len(taps)
    N = (M+1)//2 # Number of unique entries
    # Compute frequency response
    # g(1) + g(2)*cos(\omega) + g(3) \cos(2\omega) + ...
    g = taps[N-1:]
    g[1:]*=2

    # Only need five polynomials for our filters
    ChebyshevPolynomial = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 2, 0, -1],
        [0, 4, 0, -3, 0],
        [8, 0, -8, 0, 1]
    ], dtype=float)

    # Now, convert frequency response to polynomials form
    # b(1) + b(2)\cos(\omega) + b(3) \cos(\omega)^2 + ...
    b = np.zeros(N)
    for k in range(N):
       p = ChebyshevPolynomial[k]
       b += g[k]*p
    return b[::-1] # chebyshevPolyCoefficients


def filterTo2D(chebyshevPolyCoefficients, mcClellanTransform):
    ctr = len(chebyshevPolyCoefficients)
    N = 2*ctr-1

    # Initial an impulse and then filter it
    X = np.zeros((N,N))
    X[ctr-1, ctr-1] = 1

    Y = np.pad(X[:,:,np.newaxis],((0,0),(0,0),(0,len(chebyshevPolyCoefficients)-1)))
    for k in range(1,len(chebyshevPolyCoefficients)):
        # Filter delta function repeatedly with the McClellan transform
        # Size of X is chosen so boundary conditions don't matter
        Y[:,:,k] = conv2(Y[:,:,k-1], mcClellanTransform, 'same')
    # Take a linear combination of these to get the full 2D response
    return np.sum(Y*chebyshevPolyCoefficients[np.newaxis,np.newaxis,:], axis=2) # impulseResponse

def FilterTaps():
    '''
    Returns the lowpass and highpass filters specified in the supplementary
    materials of "Riesz Pyramid for Fast Phase-Based Video Magnification"


    hL and hH are the one dimenionsal filters designed by our optimization
    bL and bH are the corresponding Chebysheve polynomials
    t is the 3x3 McClellan transform matrix
    directL and directH are the direct forms of the 2d filters
    '''

    hL = np.array([-0.0209,-0.0219,0.0900,0.2723,0.3611,0.2723,0.09,-0.0219,-0.0209])
    hH = np.array([0.0099,0.0492,0.1230,0.2020,-0.7633,0.2020,0.1230,0.0492,0.0099])
    bL = filterToChebyCoeff(hL)
    bH = filterToChebyCoeff(hH)
    t = np.array([[1/8,1/4,1/8],[1/4,-1/2,1/4],[1/8,1/4,1/8]])

    directL = filterTo2D(bL, t)
    directH = filterTo2D(bH, t)
    return hL, hH, bL, bH, t, directL, directH


def buildNewPyr(im):
    '''
    Returns a multi-scale pyramid of im. `pyr` is the pyramid concatenated as a
    column vector. `im` is expected to be a grayscale two-dimenionsal image in
    single floating-point precision.

    This differs from the Matlab code in that I do not return the indices of `pyr`.
    Rather, the shape of the pyramids is found within the numpy objects of the array.
    '''
    pyr = []

    im_sz = np.asarray(im.shape)
    if im_sz[-1] == 3:
        print('WARNING: image must be grayscale... NO PIXEL DATA')

    _, _, bL, bH, t, _, _ = FilterTaps()
    bL = bL[np.newaxis,np.newaxis,:]
    bH = bH[np.newaxis,np.newaxis,:]
    bL *= 2 # To make up for the energy lost during downsampling

    while not np.any(im_sz < 10): # Stop building the pyramid when the image is too small
        Y = np.pad(im[:,:,np.newaxis], ((0,0),(0,0),(0,bL.shape[-1]-1)))

        # We apply the McClellan transform repeated to the image
        for k in range(1,bL.shape[-1]):
            p = Y[:,:,k-1]

            # Reflective boundary conditions
            p_bound = np.pad(p, ((1,1),(1,1)))     # p = np.block([
            p_bound[0,0] = p[1,1]                  #     [p[1 ,1], p[1 ,:], p[1 ,-2]],
            p_bound[0,1:-1] = p[1,:]               #     [p[: ,1], p      , p[: ,-2]],
            p_bound[0,-1] = p[1,-2]                #     [p[-2,1], p[-2,:], p[-2,-2]]
            p_bound[1:-1,0] = p[:,1]               # ])
            p_bound[1:-1,-1] = p[:,-2]             # Above code doesn't work :(
            p_bound[-1,0] = p[-2,1]
            p_bound[-1,1:-1] = p[-2,:]
            p_bound[-1,-1] = p[-2,-2]

            Y[:,:,k] = conv2(p_bound, t, 'valid')

        # Y should have no dimensions as small as bL.
        # Use Y to compute lo and highpass filtered image
        hipassedIm = np.sum(Y*bH,axis=2)
        lopassedIm = np.sum(Y*bL,axis=2)

        # Add highpassed image to the pyramid
        pyr.append(hipassedIm)

        lopassedIm = lopassedIm[::2,::2]
        # Recurse on the lowpassed image
        im_sz = np.asarray(lopassedIm.shape)

        im = lopassedIm;
    # Add a residual level for the remaining low frequencies
    pyr.append(im)

    return pyr


def reconNewPyr(pyr):
    '''
    Collapases a multi-scale pyramid of and returns the reconstructed image.
    pyr is a column vector, in which each level of the pyramid is concatenated.

    Get the filter taps
    Because we won't be simultaneously lowpassing/highpassing anything and
    most of the computational savings comes from the simultaneous application
    of the filters, we use the direct form of the filters rather the
    McClellan transform form.
    '''
    _, _, _, _, _, directL, directH = FilterTaps()
    directL *= 2 # To make up for the energy lost during downsampling
    nLevels = len(pyr)
    lo = pyr[-1]
    for k in range(nLevels-1, 0, -1): # stops at 1
        upsz = pyr[k-1].shape
        # Upsample the lowest level
        lowest = np.zeros(upsz, dtype=float)
        lowest[::2,::2] = lo
        # Lowpass it with reflective boundary conditions
        lowest = np.block([
            [lowest[4:0:-1,4:0:-1], lowest[4:0:-1,:], lowest[4:0:-1, -2:-6:-1]],
            [lowest[:,4:0:-1],      lowest          , lowest[:,-2:-6:-1]      ],
            [lowest[-2:-6:-1,4:0:-1], lowest[-2:-6:-1,:], lowest[-2:-6:-1,-2:-6:-1]],
        ])
        lowest = conv2(lowest, directL, 'valid')
        # get the next level
        nextLevel = pyr[k-1]
        nextLevel = np.block([
            [nextLevel[4:0:-1,4:0:-1], nextLevel[4:0:-1,:], nextLevel[4:0:-1, -2:-6:-1]],
            [nextLevel[:,4:0:-1],      nextLevel          , nextLevel[:,-2:-6:-1]      ],
            [nextLevel[-2:-6:-1,4:0:-1], nextLevel[-2:-6:-1,:], nextLevel[-2:-6:-1,-2:-6:-1]],
        ])

        # Highpass the level and add it to lowest level to form a new lowest level
        lowest += conv2(nextLevel, directH, 'valid')
        lo = lowest
    return lo # reconstructed

