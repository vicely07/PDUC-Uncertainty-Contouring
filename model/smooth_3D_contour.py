# -*- coding: utf-8 -*-
"""
Code written by Vi Ly 04.02.2021
email: vkly@mdanderson.org

Do not distribute without permission
"""
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interpn
import numpy as np
from scipy import signal
import sduc_src
import cv2
print("Export 07072021")
def skip_layer_interp(data, roi, initial):
    skip_layers = np.arange(initial, data.shape[0], 2)

    for z in skip_layers: 
        if z in roi:
            if z != roi[0] or z != roi[-1]:
                data[z, ...] = interp_shape(data[z-1, ...],data[z+1, ...], 0.80)

        else: 
            data[z, ...] = np.zeros(data[z, ...].shape)
    return data

def ndgrid(*args,**kwargs):
    """
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args,**kwargs)

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''    
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im

def interp_shape(top, bottom, precision):
    '''
    Interpolate between two contours

    Input: top 
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate 
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids 
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r**2, 2))
    xi = np.c_[np.full((r**2),precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out

def ndgrid(*args,**kwargs):
    """
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args,**kwargs)

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''    
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im

def interp_shape(top, bottom, precision):
    '''
    Interpolate between two contours

    Input: top 
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate 
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids 
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r**2, 2))
    xi = np.c_[np.full((r**2),precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out

## Prostate
def custom_Gaussian_blur_prostate(img, c):
    '''
    This function use Gaussian blur and threshold method to smooth our the z direction
    *Input
    img: unsmoothed contours after framing preprocess
    *Output
    output: smoothed contour
    '''
    #Convert image to a matrix of floats
    img = np.where(img<1,0.0,1.0) 
    #Using Gaussian blur to find weights for the image
    w = cv2.GaussianBlur(img,(11, 11),0)
    #Normalize the weights
    #norm_w = w/sum(w)
    #Mean thresholding and filling binary values
    threshold = np.mean(w)
    if c < 100:
        output = np.where(w <= threshold+0.35, 0, 1)
    elif c <= 200:
        output = np.where(w <= threshold+0.27, 0, 1)
    else:
        output = np.where(w <= threshold+0.20, 0, 1)
    return output

    
def filter_3D_smoothing_prostate(contour, roi_x, roi_y, roi_z, c):
    '''
    This function preprocess and apply 3D smoothing methods on the slices of interest from the raw unsmoothed contours"
    *Input:
    images: raw images of the subject
    labels: groundtruth labels of all the organ
    organ_i: id of target organ (Ex: rectum, bladder,...)
    *Output:
    smoothed contour: desired smoothed contour inside of framed roi
    '''
    data = contour.copy()
    data =  skip_layer_interp(data, roi_z, 0)
    
    skip_layers_x = np.arange(0, data.shape[1], 4)
    for x in skip_layers_x:
        if x != roi_x[0] or x != roi_x[-1]:
            data[:, x, :] = np.zeros(data[:, x, :].shape)
    for x in roi_x:
        data[:, x, :] = custom_Gaussian_blur_prostate(data[:, x, :], c)
    
    skip_layers_y = np.arange(0, data.shape[2], 4)
    for y in skip_layers_y:
        if y != roi_y[0] or y != roi_y[-1]:
            data[..., y] = np.zeros(data[..., y].shape)
    for y in roi_y:
        data[..., y] = custom_Gaussian_blur_prostate(data[..., y], c)
    for z in roi_z:
        data[z, ...] = custom_Gaussian_blur_prostate(data[z, ...], c)
    
    
    data[..., 0:min(roi_y)-1] = np.zeros(data[..., 0:min(roi_y)-1].shape)
    data[..., max(roi_y)+1:-1] = np.zeros(data[..., max(roi_y)+1:-1].shape)
    return data

## rectum:
def custom_Gaussian_blur_rectum(img, c):
    '''
    This function use Gaussian blur and threshold method to smooth our the z direction
    *Input
    img: unsmoothed contours after framing preprocess
    *Output
    output: smoothed contour
    '''
    #Convert image to a matrix of floats
    img = np.where(img<1,0.0,1.0)
    #Using Gaussian blur to find weights for the image
    w = cv2.GaussianBlur(img,(11, 11),0)
    #Normalize the weights
    #norm_w = w/sum(w)
    #Mean thresholding and filling binary values
    threshold = np.mean(w)
    if c < 100:
        output = np.where(w <= threshold+0.25, 0, 1)
    elif c <= 200:
        output = np.where(w <= threshold+0.20, 0, 1)
    else:
        output = np.where(w <= threshold+0.15, 0, 1)
    return output

def filter_3D_smoothing_rectum(contour, roi_x, roi_y, roi_z, c):
    '''
    This function preprocess and apply 3D smoothing methods on the slices of interest from the raw unsmoothed contours"
    *Input:
    images: raw images of the subject
    labels: groundtruth labels of all the organ
    organ_i: id of target organ (Ex: rectum, bladder,...)
    *Output:
    smoothed contour: desired smoothed contour inside of framed roi
    '''
    data = contour.copy()
    data = skip_layer_interp(data, roi_z, 0)
    skip_layers_x = np.arange(0, data.shape[1], 4)
    for x in skip_layers_x:
        if x != roi_x[0] or x != roi_x[-1]:
            data[:, x, :] = np.zeros(data[:, x, :].shape)
    for x in roi_x:
        data[:, x, :] = custom_Gaussian_blur_rectum(data[:, x, :], c)

    skip_layers_y = np.arange(0, data.shape[2], 4)
    for y in skip_layers_y:
        if y != roi_y[0] or y != roi_y[-1]:
            data[..., y] = np.zeros(data[..., y].shape)
    for y in roi_y:
        data[..., y] = custom_Gaussian_blur_rectum(data[..., y], c)
    for z in roi_z:
        data[z, ...] = custom_Gaussian_blur_rectum(data[z, ...], c)

    data[..., 0:min(roi_y) - 1] = np.zeros(data[..., 0:min(roi_y) - 1].shape)
    data[..., max(roi_y) + 1:-1] = np.zeros(data[..., max(roi_y) + 1:-1].shape)
    return data

## bladder

def custom_Gaussian_blur_bladder(img, c):
    '''
    This function use Gaussian blur and threshold method to smooth our the z direction
    *Input
    img: unsmoothed contours after framing preprocess
    *Output
    output: smoothed contour
    '''
    #Convert image to a matrix of floats
    img = np.where(img<1,0.0,1.0) 
    #Using Gaussian blur to find weights for the image
    w = cv2.GaussianBlur(img,(5,5),0)
    #Normalize the weights
    #norm_w = w/sum(w)
    #Mean thresholding and filling binary values
    threshold = np.mean(w)
    if c < 100:
        output = np.where(w <= threshold+0.35, 0, 1)
    else:
        output = np.where(w <= threshold+0.28, 0, 1)
    return output

    
def filter_3D_smoothing_bladder(contour, roi_x, roi_y, roi_z, c):
    '''
    This function preprocess and apply 3D smoothing methods on the slices of interest from the raw unsmoothed contours"
    *Input:
    images: raw images of the subject
    labels: groundtruth labels of all the organ
    organ_i: id of target organ (Ex: rectum, bladder,...)
    *Output:
    smoothed contour: desired smoothed contour inside of framed roi
    '''
    data = contour.copy()
    data =  skip_layer_interp(data, roi_z, 0)
    
    skip_layers_x = np.arange(0, data.shape[1], 4)
    '''
    for x in skip_layers_x:
        if x != roi_x[0] or x != roi_x[-1]:
            data[:, x, :] = np.zeros(data[:, x, :].shape)
    '''
    for x in roi_x:
        data[:, x, :] = custom_Gaussian_blur_bladder(data[:, x, :], c)
    skip_layers_y = np.arange(0, data.shape[2], 4)
    
    for y in skip_layers_y:
        if y != roi_y[0] or y != roi_y[-1]:
            data[..., y] = np.zeros(data[..., y].shape)
    
    for y in roi_y:
        data[..., y] = custom_Gaussian_blur_bladder(data[..., y], c)
    
    for z in roi_z:
        data[z, ...] = custom_Gaussian_blur_bladder(data[z, ...], c)
    
    
    data[..., 0:min(roi_y)-1] = np.zeros(data[..., 0:min(roi_y)-1].shape)
    data[..., max(roi_y)+1:-1] = np.zeros(data[..., max(roi_y)+1:-1].shape)
    return data


def creating_contour(c, SD_list, organ_i, labels, images, roi_x, roi_y, roi_z, organ_name):
    k = 50
    w = 2
    a = 50
    SD=[c*SD_list[0], c*SD_list[1], c*SD_list[2]]
    voxelsize = np.array([0.1076562, 0.1076562, 2.5])
    circles =3
    assd_contour = labels[..., organ_i].copy()
    seed = 72

    for i in roi_z:
        dx, dy, mask, t, L, roi_z, i0 = sduc_src.sduc_alg(images[i, ...], labels[..., organ_i][i, ...], voxelsize, a, SD, circles, seed, k, w, images, labels, organ_i, ismax=True, smooth=True)
        du = sduc_src.plotting_assd(dx, dy, mask, images[i, ...], quiver=False, plot=False) 
        assd_contour[i, ...] = du
    assd_contour = np.array(assd_contour)

    if organ_name == "bladder":
        smoothed_contour = filter_3D_smoothing_bladder(assd_contour, roi_x, roi_y, roi_z, c)
    elif organ_name == "rectum":
        smoothed_contour = filter_3D_smoothing_rectum(assd_contour, roi_x, roi_y, roi_z, c)
    elif organ_name == "prostate":
        smoothed_contour = filter_3D_smoothing_prostate(assd_contour, roi_x, roi_y, roi_z, c)
    
    return assd_contour