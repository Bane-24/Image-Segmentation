#####################      ASSIGNMENT 1: COLOUR IMAGE SEGMENTATION      ######################
#####################           AKSHAY.R - U3228988     #####################

# python script is used instead of .ipynb due to corrupted packages with jupyter and anaconda(MacOS), 
# and I was not able to find a fix for it within the submission time.
# Original code is still unchanged, and comments are added as required.

## Initialisations and inputs
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.filters import gaussian
from sklearn.cluster import KMeans

plt.close('all')
clear = lambda: os.system('clear')
clear()
np.random.seed(110)

# Inititialising variables
colors = [[1,0,0],[0,1,0],[0,0,1],[0,0.5,0.5],[0.5,0,0.5]]
imgNames = ['water_coins','jump','tiger']
segmentCounts = [2,3,4,5]
# Creating an empty plot to use in the later part of the code
fig, axs = plt.subplots(3, 4, figsize=(15,10))


## define functions required for processing
def normalize(img):
    """ min-max normalization """
    h = img.shape[0]
    w = img.shape[1]
    nc = img.shape[2]
    new_img = np.zeros((h,w,nc),dtype='float')
    for cc in range(nc):
        new_img[:,cc] = (img[:,cc] - img[:,cc].min()) / (img[:,cc].max() - img[:,cc].min())
    return(new_img)
def im2double(im):
    try:
        info = np.iinfo(im.dtype) # Get the data type of the input image
        return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype
    except ValueError:
        print('Image is of type Double-- No conversion required')
        return im.astype(np.float)
## End of functions  


for imgName in imgNames:
    for SegCount in segmentCounts:
        # Load the image using the Matplotlib image library
        """ Read Image using mplib library-- 2 points """
        img_path = os.path.join('Input', imgName + '.png')  
        img = plt.imread(img_path)
        print('Using Matplotlib Image Library: Image is of datatype ',img.dtype,'and size ',img.shape) 
        # Imported image will be of type float

        # Load image using the Pillow 
        """ Read Image using PILLOW-- 3 points"""
        img_path = os.path.join('Input', imgName + '.png')         
        pil_img = Image.open(img_path)
        img = np.array(pil_img)        
        print('Using Pillow (Python Image Library): Image is of datatype ', img.dtype, 'and size ', img.shape)  
        # Imported image will be of type uint and this will be used for calulations in the following task
        
        # Checking the dimensions of the image
        h = img.shape[0]
        w = img.shape[1]
        nc = img.shape[2]
        N = h*w
        print('Image dimensions: ',h,'x',w,'x',nc,' Total no of pixels: ',N)

        # Defining parameters
        nSegments = SegCount # of color clusters in image
        maxIterations = 20;
        nColors = 3;
        
        # Determining output path
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/',  imgName , '/']));
        if not(os.path.exists(outputPath)):
            os.makedirs(outputPath)
            outputPath = os.path.join('Output', str(SegCount) + '_segments', imgName)

        # Save input image as in the required folder
        """ save input image as *0.png* under outputPath using imsave command in mpimg library-- 3 points"""
        output_file = os.path.join(outputPath, '0.png')
        mpimg.imsave(output_file, img)

        # Reshaping the pixels for easier operations
        """ Reshape pixels as a nPixels X nColors X 1 matrix-- 5 points"""
        pixels = img
        h, w, nColors = pixels.shape
        nPixels = h * w
        pixels = pixels.reshape(nPixels, nColors, 1)

        # Initialising pi vector, mu matrix and inital guess
        pi = 1/nSegments*(np.ones((nSegments, 1),dtype='float'))
        increment = np.random.normal(0,.0001,1)
        for seg_ctr in range(len(pi)):
            if(seg_ctr%2==1):
                pi[seg_ctr] = pi[seg_ctr] + increment
            else:
                pi[seg_ctr] = pi[seg_ctr] - increment
        
        """Initialize mu to 1/nSegments*['ones' matrix (whose elements are all 1) of size nSegments X nColors]
        5 points"""
        # Extracting RGB values
        mu_R = np.mean(img[:,:,0])
        mu_G = np.mean(img[:,:,1])
        mu_B = np.mean(img[:,:,2])

        # Create a matrix where each row contains the RGB means
        mean_RGB = np.array([mu_R, mu_G, mu_B])
        # Initialize mu matrix based on the provided description.
        # This step sets the initial values of mu, which will represent the centroids of the color clusters.
        mu = (1/nSegments) * np.ones((nSegments, nColors)) * mean_RGB

        #add noise to the initialization (but keep it unit)
        for seg_ctr in range(nSegments):
            if(seg_ctr%2==1):
                increment = np.random.normal(0,.0001,1)
            for col_ctr in range(nColors):
                if(seg_ctr%2==1):
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) + increment
                else:
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) -increment;
        


##########################################   E-Step   ##########################################
        mu_last_iter = mu;
        pi_last_iter = pi;

        for iteration in range(maxIterations):
            print(''.join(['Image: ',imgName,' nSegments:',str(nSegments),' iteration:',str(iteration+1), ' E-step']))
        # Weights that describe the likelihood that pixel denoted by pix_import scipy.miscctr" belongs to a color cluster "seg_ctr"

            Ws = np.ones((nPixels,nSegments),dtype='float') # temporarily reinitialize all weights to 1, before they are recomputed

            for pix_ctr in range(nPixels):
            # Calculate Ajs
                logAjVec = np.zeros((nSegments,1),dtype='float')
                for seg_ctr in range(nSegments):
                    x_minus_mu_T = np.transpose(pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T)
                    x_minus_mu = ((pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T))
                    logAjVec[seg_ctr] = np.log(pi[seg_ctr]) - .5*(np.dot(x_minus_mu_T,x_minus_mu))
                # Note the max
                logAmax = max(logAjVec.tolist())
                # Calculate the third term from the final eqn in the above link
                thirdTerm = 0;
                for seg_ctr in range(nSegments):
                    thirdTerm = thirdTerm + np.exp(logAjVec[seg_ctr]-logAmax)
                # Here Ws are the relative membership weights(p_i/sum(p_i)),but computed in a round-about way
                for seg_ctr in range(nSegments):
                    logY = logAjVec[seg_ctr] - logAmax - np.log(thirdTerm)
                    Ws[pix_ctr][seg_ctr] = np.exp(logY)


##########################################   M-Step   ##########################################
            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),'iteration: ',str(iteration+1), ' M-step: Mixture coefficients']))
            # Temporarily reinitialize mu and pi to 0, before they are␣recomputed
            mu = np.zeros((nSegments,nColors),dtype='float') # mean color for␣each segment
            pi = np.zeros((nSegments,1),dtype='float') #mixture coefficients
            
            """Update RGB color vector of mu[seg_ctr] as current mu[seg_ctr] + pixels[pix_ctr,:] 
            times Ws[pix_ctr,seg_ctr] -- 5 points"""
            # The following code updates the centroids iteratively, denominatorSum keeps track of the weights 
            # The inner loop iterates through pixels and checks the relations to a centroid.RGB values are fetched and flatteed to a 1D array
            # Ws[pix_ctr][seg_ctr] computes how closely a pixel belongs to a segment
            # The weighted contribution of the current pixel to the centroid of the current segment is calculated during the multiplication step.

            for seg_ctr in range(nSegments):
                denominatorSum = 0
                for pix_ctr in range(nPixels):
                    mu[seg_ctr] += pixels[pix_ctr].flatten() * Ws[pix_ctr][seg_ctr]
                    denominatorSum += Ws[pix_ctr][seg_ctr]

                ## Update mu
                mu[seg_ctr,:] = mu[seg_ctr,:]/ denominatorSum;
                ## Update pi
                pi[seg_ctr] = denominatorSum / nPixels; #sum of weights (each weight is a probability) for given segment/total num of pixels
            print(np.transpose(pi))

            muDiffSq = np.sum(np.multiply((mu - mu_last_iter),(mu - mu_last_iter)))
            piDiffSq = np.sum(np.multiply((pi - pi_last_iter),(pi - pi_last_iter)))
            
            if (muDiffSq < .0000001 and piDiffSq < .0000001): #sign of convergence
                print('Convergence Criteria Met at Iteration: ',iteration, '--Exiting code')
                break;
            
            mu_last_iter = mu;
            pi_last_iter = pi;
            ##Draw the segmented image as RGB value for all pixels in that cluster
            segpixels = np.array(pixels)
            cluster = 0
            for pix_ctr in range(nPixels):
                cluster = np.where(Ws[pix_ctr,:] == max(Ws[pix_ctr,:]))
                vec = np.squeeze(np.transpose(mu[cluster,:]))
                segpixels[pix_ctr,:] = vec.reshape(vec.shape[0],1)
            
            # Save segmented image at each iteration    
            segpixels = np.reshape(segpixels,(img.shape[0],img.shape[1],nColors)) ## reshape segpixels to obtain R,G, B image
            # Convert segpixels to grayscale
            """convert segpixels to uint8 gray scale image and convert to grayscale-- 5 points"""
            segpixels_gray = rgb2gray(segpixels) * 255  # Multiply by 255 to be in the range[0,255]
            segpixels_gray = segpixels_gray.astype(np.uint8)  # Convert to uint8 type
            print(segpixels.shape) # Checking the shape 
            # Reshape the grayscale image for kmeans computation(2D array)
            reshaped_graypixels = segpixels_gray.reshape(-1, 1)

            # Use k-means clustering on the grayscale pixels
            """ Use kmeans from sci-kit learn library to cluster pixels in gray scale segpixels 
            image to *nSegments* clusters-- 10 points"""
            kmeans = KMeans(n_clusters=nSegments).fit(reshaped_graypixels)

            # Reshape kmeans.labels_ to the size of the original image for visualisation 
            """ reshape kmeans.labels_ output of kmeans to have the same size as segpixels -- 5 points"""
            seglabels = kmeans.labels_.reshape(segpixels_gray.shape)

            # Converting seglabels with specified colors
            """Use np.clip, Gaussian smoothing with sigma =2 and label2rgb functions to smoothen the seglabels 
            image, and output a float RGB image with pixel values between [0--1]-- 20 points"""
            colored_labels = label2rgb(seglabels, colors=colors)

            # Apply Gaussian smoothing, for a better transition between color segments and reducing noise
            smoothed_labels = gaussian(colored_labels, sigma=2)

            # Clip the values to the range [0,1]
            seglabels = np.clip(smoothed_labels, a_min=0, a_max=1)

            # Saving the images to the output path
            mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabels) 
               
    
        # checking if the convergence happened in the final image or before for visulisation
        """Display the 20th iteration (or final output in case of convergence) segmentation images with 
        nSegments = 2,3,4,5 for the three images--this will be a 3 row X 4 column image matrix-- 15 points"""
        if iteration < 20:
            final_iteration = iteration
        else:
            final_iteration = 20
        output_path_final = ''.join([outputPath, str(final_iteration), '.png'])
        
        # Using the last avaialbe iteration if the image exists to display the output
        if not os.path.exists(output_path_final):
            output_path_final = ''.join([outputPath, str(iteration), '.png'])
        
        segmented_img = plt.imread(output_path_final)

        # Displaying the output grid
        axs[imgNames.index(imgName), segmentCounts.index(SegCount)].imshow(segmented_img)
        axs[imgNames.index(imgName), segmentCounts.index(SegCount)].axis('off')
        axs[imgNames.index(imgName), segmentCounts.index(SegCount)].set_title(f'{imgName} - {SegCount} Segment')

plt.tight_layout()
plt.show()

##############################################################################################################
##############################################################################################################

""" Comment on the results obtained, and discuss your understanding of the 
Image Segmentation problem in general-- 15 points """
# This is answered in the attached final report.

##############################################################################################################
##############################################################################################################
