# SDUC-paremetric-bad-contouring-model

## Instruction:
Please see sample "sduc-exportation-example.ipynb" for instruction of using the SDUC model.
Email me for the sample prostate-case:
vkly@mdanderson.org


## Purpose: 
Contouring is one of the largest sources of variability in radiotherapy.  The purpose of this work is to develop an algorithm to inject known levels of error to clinically accepted contours, giving realistic contours with varying levels of variability.  This is an important first step to investigating the dosimetric impact of contouring variability, and to developing approaches to automatically verify contouring quality.  
 
## Methods: 
We have developed a smoothed-delineation-uncertainties-contouring (SDUC) model to automatically generate alternative, realistic contours.  The SDUC model transforms contours (deformation, contraction and/or expansion) as a function of image contrast, based on the expectation that contouring variability is largest when image contrast is low (i.e. when structure edges are less clear).  The magnitude of the delineation uncertainty is scaled by a voxel-specific displacement vector in 3 dimensions by applying a random base uncertainty, scaled by the local contrast.  Finally, 3D smoothing is applied to ensure the final contours appear realistic. 
 
## Results: 
For our evaluation, we use the algorithm for auto-contouring of 3 organs (the prostate, bladder, and rectum) on 4 different patients, and then visually inspected the generated contours against the ground truths. We find that there is a decreasing trend in DSC as the variation rate C increase. The algorithm also tends to create small variations as the image contrast is high between the ROI and the surrounding region, as for prostate and bladder, and create large variations for low image contrast, as for rectum. Hence, for prostate, rectum, and bladder, the corresponding mean Dice indices were 0.86±0.04, 0.84±0.07, 0.94±0.02, respectively. [CE1] 

## Conclusion:  
We have developed an algorithm that can simulate the effect of variability in contouring, generating variable but realistic contours. 
