#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

from model.dicom_utils import *
import numpy as np 
import pandas as pd 
import os
import pydicom
import matplotlib.pyplot as plt
#import assd_func as af
from skimage import morphology
import matplotlib.patches as mpatches
import model.sduc_src as af_Sobel
import time
import random
from scipy.stats import uniform,norm
import time
import multiprocessing as mp
from model.smooth_3D_contour import *


# In[2]:


def read_exported_contours(path, images, slices):
    contour = read_contours(path)
    labels = get_labels(contour, images.shape, slices, rois=None)
    labels = np.where(labels == True, 1, 0)
    return labels


# In[3]:


dicom_dir = "Prostate-Cases/11"
output_dir = "Output"
images_11, slices_11, contours_11, labels_11, dummy_mask_11 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/12"
output_dir = "Output"
images_12, slices_12, contours_12, labels_12, dummy_mask_12 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/13"
output_dir = "Output"
images_13, slices_13, contours_13, labels_13, dummy_mask_13 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/14"
output_dir = "Output"
images_14, slices_14, contours_14, labels_14, dummy_mask_14 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/15"
output_dir = "Output"
images_15, slices_15, contours_15, labels_15, dummy_mask_15 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/16"
output_dir = "Output"
images_16, slices_16, contours_16, labels_16, dummy_mask_16 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/17"
output_dir = "Output"
images_17, slices_17, contours_17, labels_17, dummy_mask_17 = main(dicom_dir, output_dir)


dicom_dir = "Prostate-Cases/18"
output_dir = "Output"
images_18, slices_18, contours_18, labels_18, dummy_mask_18 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/19"
output_dir = "Output"
images_19, slices_19, contours_19, labels_19, dummy_mask_19 = main(dicom_dir, output_dir)

dicom_dir = "Prostate-Cases/20"
output_dir = "Output"
images_20, slices_20, contours_20, labels_20, dummy_mask_20 = main(dicom_dir, output_dir)

# # Finding organs ID:

# In[4]:


def find_roi_x(images, labels, i):
    roi_y = []
    for j in range(labels.shape[2]):
        if True in np.unique(labels[..., i][:, :,  j] > 0):
            roi_y.append(j)
    return roi_y

def find_roi_y(images, labels, i):
    roi_x = []
    for j in range(labels.shape[1]):
        if True in np.unique(labels[..., i][:, j,  :] > 0):
            roi_x.append(j)
    return roi_x

def find_roi_z(images, labels, i):
    roi_z = []
    for j in range(images.shape[0]):
        if True in np.unique(labels[..., i][j, ...] > 0):
            roi_z.append(j)
    return roi_z

def find_organ_i(organ, contours):
    contour_table = pd.DataFrame()
    number = []
    name = []
    for i in range(len(contours)):
        number.append(contours[i]["number"])
        name.append(contours[i]["name"])
    contour_table["number"] = number
    contour_table["name"] = name
    organ_i =  contour_table[contour_table["name"].str.contains(organ)].index.tolist()[0]
    return organ_i


# ## Prostate

# In[5]:


def roi_info_extraction(organ, contours, images, labels):
    organ_i_p =  find_organ_i(organ, contours)
    roi_x_p = find_roi_x(images, labels, organ_i_p)
    roi_y_p = find_roi_y(images, labels, organ_i_p)
    roi_z_p = find_roi_z(images, labels, organ_i_p)
    return organ_i_p, roi_x_p, roi_y_p, roi_z_p


## Finding the ROI id for each subject:
organ_p = "prostate" 
organ_r = "rectum" # or Rectum
organ_b = "bladder" # or Bladder

# Subject 11:

organ_i_p_11, roi_x_p_11, roi_y_p_11, roi_z_p_11 = roi_info_extraction(organ_p, contours_11, images_11, labels_11)
organ_i_r_11, roi_x_r_11, roi_y_r_11, roi_z_r_11 = roi_info_extraction(organ_r, contours_11, images_11, labels_11)
organ_i_b_11, roi_x_b_11, roi_y_b_11, roi_z_b_11 = roi_info_extraction(organ_b, contours_11, images_11, labels_11)

organ_p = "Prostate" 
organ_r = "Rectum" # or Rectum
organ_b = "Bladder" # or Bladder

# Subject 12:
organ_i_p_12, roi_x_p_12, roi_y_p_12, roi_z_p_12 = roi_info_extraction(organ_p, contours_12, images_12, labels_12)
organ_i_r_12, roi_x_r_12, roi_y_r_12, roi_z_r_12 = roi_info_extraction(organ_r, contours_12, images_12, labels_12)
organ_i_b_12, roi_x_b_12, roi_y_b_12, roi_z_b_12 = roi_info_extraction(organ_b, contours_12, images_12, labels_12)

organ_p = "Prostate" 
organ_r = "rectum" # or Rectum
organ_b = "bladder" # or Bladder

# Subject 13:
organ_i_p_13, roi_x_p_13, roi_y_p_13, roi_z_p_13 = roi_info_extraction(organ_p, contours_13, images_13, labels_13)
organ_i_r_13, roi_x_r_13, roi_y_r_13, roi_z_r_13 = roi_info_extraction(organ_r, contours_13, images_13, labels_13)
organ_i_b_13, roi_x_b_13, roi_y_b_13, roi_z_b_13 = roi_info_extraction(organ_b, contours_13, images_13, labels_13)

organ_p = "prostate" 
organ_r = "rectum" # or Rectum
organ_b = "bladder" # or Bladder

# Subject 14:
print("subject 14")
organ_i_p_14, roi_x_p_14, roi_y_p_14, roi_z_p_14 = roi_info_extraction(organ_p, contours_14, images_14, labels_14)
organ_i_r_14, roi_x_r_14, roi_y_r_14, roi_z_r_14 = roi_info_extraction(organ_r, contours_14, images_14, labels_14)
organ_i_b_14, roi_x_b_14, roi_y_b_14, roi_z_b_14 = roi_info_extraction(organ_b, contours_14, images_14, labels_14)

# Subject 15:
print("subject 15")
organ_i_p_15, roi_x_p_15, roi_y_p_15, roi_z_p_15 = roi_info_extraction(organ_p, contours_15, images_15, labels_15)
organ_i_r_15, roi_x_r_15, roi_y_r_15, roi_z_r_15 = roi_info_extraction(organ_r, contours_15, images_15, labels_15)
organ_i_b_15, roi_x_b_15, roi_y_b_15, roi_z_b_15 = roi_info_extraction(organ_b, contours_15, images_15, labels_15)

organ_p = "prostate" 
organ_r = "Rectum" # or Rectum
organ_b = "Bladder" # or Bladder

# Subject 16:
print("subject 16")
organ_i_p_16, roi_x_p_16, roi_y_p_16, roi_z_p_16 = roi_info_extraction(organ_p, contours_16, images_16, labels_16)
organ_i_r_16, roi_x_r_16, roi_y_r_16, roi_z_r_16 = roi_info_extraction(organ_r, contours_16, images_16, labels_16)
organ_i_b_16, roi_x_b_16, roi_y_b_16, roi_z_b_16 = roi_info_extraction(organ_b, contours_16, images_16, labels_16)

organ_p = "Prostate" 
organ_r = "Rectum" # or Rectum
organ_b = "Bladder" # or Bladder

# Subject 17:
print("subject 17")
organ_i_p_17, roi_x_p_17, roi_y_p_17, roi_z_p_17 = roi_info_extraction(organ_p, contours_17, images_17, labels_17)
organ_i_r_17, roi_x_r_17, roi_y_r_17, roi_z_r_17 = roi_info_extraction(organ_r, contours_17, images_17, labels_17)
organ_i_b_17, roi_x_b_17, roi_y_b_17, roi_z_b_17 = roi_info_extraction(organ_b, contours_17, images_17, labels_17)

# Subject 18:
#print("subject 18")
#organ_i_p_18, roi_x_p_18, roi_y_p_18, roi_z_p_18 = roi_info_extraction(organ_p, contours_18, images_18, labels_18)
#organ_i_r_18, roi_x_r_18, roi_y_r_18, roi_z_r_18 = roi_info_extraction(organ_r, contours_18, images_18, labels_18)
#organ_i_b_18, roi_x_b_18, roi_y_b_18, roi_z_b_18 = roi_info_extraction(organ_b, contours_18, images_18, labels_18)

# Subject 19:
print("subject 19")
organ_i_p_19, roi_x_p_19, roi_y_p_19, roi_z_p_19 = roi_info_extraction(organ_p, contours_19, images_19, labels_19)
organ_i_r_19, roi_x_r_19, roi_y_r_19, roi_z_r_19 = roi_info_extraction(organ_r, contours_19, images_19, labels_19)
organ_i_b_19, roi_x_b_19, roi_y_b_19, roi_z_b_19 = roi_info_extraction(organ_b, contours_19, images_19, labels_19)

organ_p = "prostate" 
organ_r = "rectum" # or Rectum
organ_b = "bladder" # or Bladder

# Subject 20:
print("subject 20")
organ_i_p_20, roi_x_p_20, roi_y_p_20, roi_z_p_20 = roi_info_extraction(organ_p, contours_20, images_20, labels_20)
organ_i_r_20, roi_x_r_20, roi_y_r_20, roi_z_r_20 = roi_info_extraction(organ_r, contours_20, images_20, labels_20)
organ_i_b_20, roi_x_b_20, roi_y_b_20, roi_z_b_20 = roi_info_extraction(organ_b, contours_20, images_20, labels_20)


# ## Subject 11:
# In[ ]:
print("subject 11")
start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_11 = labels_11[..., organ_i_p_11].copy()

num_cores=1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
#list_of_c = [50, 200, 100, 5, 300]

sub1_results1  = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_11, labels_11, images_11, roi_x_p_11, roi_y_p_11, roi_z_p_11, "prostate")
    sub1_results1.append(contour)

                   
#sub1_results1 = [p.get() for p in sub1_results1]  
 
print("---------------------------Prostate done-------------------------------------------")


# Rectum
#list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_11 = labels_11[..., organ_i_r_11].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2  = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_11, labels_11, images_11, roi_x_r_11, roi_y_r_11, roi_z_r_11, "rectum")
    sub1_results2.append(contour)

#sub1_results2 = [p.get() for p in sub1_results2]    

    
print("---------------------------rectum done---------------------------------------------")

# bladder
#list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_11 = labels_11[..., organ_i_b_11].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3  = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_11, labels_11, images_11, roi_x_b_11, roi_y_b_11, roi_z_b_11, "bladder")
    sub1_results3.append(contour)

#sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")


## Exportation
prostate_list = [gt_p_11, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4", "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_11, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_11, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4", "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

#destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name, color=prostate_color + rectum_color + bladder_color, DICOMImageStruct = slices_11, fname=os.path.join(destination_path, 'subject_11_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end-start))
pool.close()

# ## Subject 12:
# In[ ]:
print("subject 12")
start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_12 = labels_12[..., organ_i_p_12].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_12, labels_12, images_12, roi_x_p_12, roi_y_p_12, roi_z_p_12,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_12 = labels_12[..., organ_i_r_12].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_12, labels_12, images_12, roi_x_r_12, roi_y_r_12, roi_z_r_12,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_12 = labels_12[..., organ_i_b_12].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_12, labels_12, images_12, roi_x_b_12, roi_y_b_12, roi_z_b_12,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_12, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_12, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_12, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_12,
         fname=os.path.join(destination_path, 'subject_12_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()

# ## Subject 13:
# In[ ]:
print("subject 13")
start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_13 = labels_13[..., organ_i_p_13].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_13, labels_13, images_13, roi_x_p_13, roi_y_p_13, roi_z_p_13,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_13 = labels_13[..., organ_i_r_13].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_13, labels_13, images_13, roi_x_r_13, roi_y_r_13, roi_z_r_13,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_13 = labels_13[..., organ_i_b_13].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_13, labels_13, images_13, roi_x_b_13, roi_y_b_13, roi_z_b_13,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_13, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_13, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_13, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_13,
         fname=os.path.join(destination_path, 'subject_13_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()


# ## Subject 14:
# In[ ]:

start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_14 = labels_14[..., organ_i_p_14].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_14, labels_14, images_14, roi_x_p_14, roi_y_p_14, roi_z_p_14,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_14 = labels_14[..., organ_i_r_14].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_14, labels_14, images_14, roi_x_r_14, roi_y_r_14, roi_z_r_14,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_14 = labels_14[..., organ_i_b_14].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_14, labels_14, images_14, roi_x_b_14, roi_y_b_14, roi_z_b_14,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_14, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_14, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_14, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_14,
         fname=os.path.join(destination_path, 'subject_14_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()

# ## Subject 15:
# In[ ]:

start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_15 = labels_15[..., organ_i_p_15].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_15, labels_15, images_15, roi_x_p_15, roi_y_p_15, roi_z_p_15,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_15 = labels_15[..., organ_i_r_15].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_15, labels_15, images_15, roi_x_r_15, roi_y_r_15, roi_z_r_15,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_15 = labels_15[..., organ_i_b_15].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_15, labels_15, images_15, roi_x_b_15, roi_y_b_15, roi_z_b_15,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_15, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_15, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_15, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_15,
         fname=os.path.join(destination_path, 'subject_15_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()

# ## Subject 16:
# In[ ]:

start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_16 = labels_16[..., organ_i_p_16].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_16, labels_16, images_16, roi_x_p_16, roi_y_p_16, roi_z_p_16,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_16 = labels_16[..., organ_i_r_16].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_16, labels_16, images_16, roi_x_r_16, roi_y_r_16, roi_z_r_16,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_16 = labels_16[..., organ_i_b_16].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_16, labels_16, images_16, roi_x_b_16, roi_y_b_16, roi_z_b_16,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_16, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_16, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_16, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_16,
         fname=os.path.join(destination_path, 'subject_16_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()

# ## Subject 17:
# In[ ]:

start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_17 = labels_17[..., organ_i_p_17].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_17, labels_17, images_17, roi_x_p_17, roi_y_p_17, roi_z_p_17,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_17 = labels_17[..., organ_i_r_17].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_17, labels_17, images_17, roi_x_r_17, roi_y_r_17, roi_z_r_17,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_17 = labels_17[..., organ_i_b_17].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_17, labels_17, images_17, roi_x_b_17, roi_y_b_17, roi_z_b_17,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_17, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_17, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_17, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_17,
         fname=os.path.join(destination_path, 'subject_17_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()

# ## Subject 19:
# In[ ]:

start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_19 = labels_19[..., organ_i_p_19].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_19, labels_19, images_19, roi_x_p_19, roi_y_p_19, roi_z_p_19,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_19 = labels_19[..., organ_i_r_19].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_19, labels_19, images_19, roi_x_r_19, roi_y_r_19, roi_z_r_19,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_19 = labels_19[..., organ_i_b_19].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_19, labels_19, images_19, roi_x_b_19, roi_y_b_19, roi_z_b_19,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_19, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_19, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_19, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_19,
         fname=os.path.join(destination_path, 'subject_19_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()

# ## Subject 20:
# In[ ]:

start = time.time()
SD_list1 = [1.7, 1.7, 2]

# Prostate
gt_p_20 = labels_20[..., organ_i_p_20].copy()

num_cores = 1
pool = mp.Pool(num_cores)

list_of_c = [5, 50, 100, 200, 300]
# list_of_c = [50, 200, 100, 5, 300]

sub1_results1 = []
print("prostate:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list1, organ_i_p_20, labels_20, images_20, roi_x_p_20, roi_y_p_20, roi_z_p_20,
                               "prostate")
    sub1_results1.append(contour)

# sub1_results1 = [p.get() for p in sub1_results1]

print("---------------------------Prostate done-------------------------------------------")

# Rectum
# list_of_c = [5, 100, 50, 300, 200]
list_of_c = [5, 50, 100, 200, 300]

gt_r_20 = labels_20[..., organ_i_r_20].copy()

SD_list2 = [1.3, 1.3, 3]
sub1_results2 = []
print("rectum:")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list2, organ_i_r_20, labels_20, images_20, roi_x_r_20, roi_y_r_20, roi_z_r_20,
                               "rectum")
    sub1_results2.append(contour)

# sub1_results2 = [p.get() for p in sub1_results2]


print("---------------------------rectum done---------------------------------------------")

# bladder
# list_of_c = [100, 50, 300, 200, 5]
list_of_c = [5, 50, 100, 200, 300]

gt_b_20 = labels_20[..., organ_i_b_20].copy()

SD_list3 = [0.7, 0.7, 2]

sub1_results3 = []
print("bladder")
for c in list_of_c:
    print(c)
    contour = creating_contour(c, SD_list3, organ_i_b_20, labels_20, images_20, roi_x_b_20, roi_y_b_20, roi_z_b_20,
                               "bladder")
    sub1_results3.append(contour)

# sub1_results3 = [p.get() for p in sub1_results3]

print("---------------------------bladder done-------------------------------------------")

## Exportation
prostate_list = [gt_p_20, sub1_results1[0], sub1_results1[1], sub1_results1[2], sub1_results1[3], sub1_results1[4]]
prostate_name = ["truth_prostate", "auto_prostate_1", "auto_prostate_2", "auto_prostate_3", "auto_prostate_4",
                 "auto_prostate_5"]
prostate_color = ["red", "green", "purple", "yellow", "blue", "orange"]
rectum_list = [gt_r_20, sub1_results2[0], sub1_results2[1], sub1_results2[2], sub1_results2[3], sub1_results2[4]]
rectum_name = ["truth_rectum", "auto_rectum_1", "auto_rectum_2", "auto_rectum_3", "auto_rectum_4", "auto_rectum_5"]
rectum_color = ["red", "purple", "yellow", "blue", "orange", "green"]
bladder_list = [gt_b_20, sub1_results3[0], sub1_results3[1], sub1_results3[2], sub1_results3[3], sub1_results3[4]]
bladder_name = ["truth_bladder", "auto_bladder_1", "auto_bladder_2", "auto_bladder_3", "auto_bladder_4",
                "auto_bladder_5"]
bladder_color = ["red", "green", "purple", "yellow", "blue", "orange"]

# destination_path = "../"
destination_path = "Output"
RTStruct(prostate_list + rectum_list + bladder_list, prostate_name + rectum_name + bladder_name,
         color=prostate_color + rectum_color + bladder_color, DICOMImageStruct=slices_20,
         fname=os.path.join(destination_path, 'subject_20_contours_7_3.dcm'))
end = time.time()
print("Running time:" + str(end - start))
pool.close()
