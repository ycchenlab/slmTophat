import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from functions import costfunction
from functions import converter
import tensorflow as tf
import imageio
import time
import os
import shutil
from cv2 import imread, IMREAD_GRAYSCALE


# ===================================== Parameters config ===============================================

# Loading the target image
# Create grid
N = 500  # Number of grid points
L = 1  # Grid size
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)


# Super-Gaussian profile parameters
m = 10  # Super-Gaussian exponent
A = 100  # Super-Gaussian amplitude
w0 = L/6  # Super-Gaussian width

# Generate super-Gaussian profile -- it is intensity
target = np.square(A * np.exp(-(np.abs(X)/w0)**(2*m)) * np.exp(-(np.abs(Y)/(w0/4))**(2*m)))

# Gaussian beam parameters
w0 = L/6  # Gaussian beam waist radius

# Calculate Gaussian beam profile -- it is electric field amplitude
R = np.sqrt(X**2 + Y**2)  # Circular shape
initial_profile = np.exp(-((R/w0)**2))

target /= np.max(target)  # Normalize matrix

# Defining DOE phase

#DOE = np.random.rand(N,N)*2*np.pi
DOE = np.load('DOE_data.npy')
s = 10

# Create an empty list to store frames
frames = []

# costType: 1 = simple cost function(Ct2), 2 = smoothing neighbor pixels(Cs), 3 = alternating Ct4 / Cs, 4 = alternating Ct2 / Cs, 5 = Ct4 / Ct2
costType = 1
learning_rate=0.0001

# ===================================== Parameters config ===============================================

# conjugate gradient method algorithm
# Iterate to calculate the phase value
for t in range(s):
    # Start the timer
    start_time = time.time()
    
    DOEphase = np.exp(1j * DOE)

####################    # Forward iteration
    iterf = fft2(initial_profile * DOEphase) # field fft
    intf = np.square(np.abs(iterf)) / np.max(np.square(np.abs(iterf))) # normalized training intenstiy
    angf = np.angle(iterf)
    A = target * np.exp(1j * angf)

    # Backward iteration
    '''
    iterb = ifft2(A)
    angb = np.angle(iterb)
    DOE = angb
    '''
    error = target - intf # Calculate error
    E = np.sum(np.abs(error)) / (N * N)
    differences = target - intf
    squaredDifferences = differences**2
    meanSquaredDifferences = np.mean(squaredDifferences)
    
    
    ############################ Optimization funciton
    

    cost, DOE_tf, learning_rate, optimizer_string = costfunction(DOE, target, initial_profile, N,t,learning_rate, costType)
    DOE = DOE_tf.numpy() 

    rmse = np.sqrt(meanSquaredDifferences)
    
    if E < 0.0005:
        iteration = t
        break
    
    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time    

    # Text for Optimazation   
    text = f'Iteration: {t+1}\nLearning Rate: {learning_rate}\nRMSE: {round(rmse, 4)}\nOptimizer: {optimizer_string}\nElapsed time: {round(elapsed_time, 2)} seconds'
    
    # Text for GS
    #text = f'Iteration: {t+1}\nRMSE: {round(rmse, 4)}\nElapsed time: {round(elapsed_time, 2)} seconds'
    
    
    # Plot squared differences and save the figure as an image
    plt.figure(dpi=300)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(intf, cmap='jet')
    plt.axis('image')
    plt.colorbar()
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('Training Tophat')
    plt.annotate(text, xy=(0.05, 0.8), xycoords='axes fraction', color='white', fontsize=7, weight='bold')
    # Save the figure as an image    
    save_path = r'C:\Users\ycche\git repo\slmTophat\tempPNG\\'
    filename = f'plot_{t}.png'
    plt.savefig(save_path + filename, dpi = 300)
    # Convert the plot to an image array
    fig = plt.gcf()
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)
    plt.close()

    # Read the saved image as an array and add it to the frames list

    # Remove the saved image file

# End of iteration


# save DOE data for next time use
np.save('DOE_data.npy', DOE)

'''
# Specify the source file path
source_path = "C:/Users/ycche/git repo/slmTophat/tempPNG/plot_29.png"  # Replace with the actual path of the source file

# Specify the destination directory
destination_directory = "C:/Users/ycche/git repo/slmTophat/IntensityModifiedCostFunction/"  # Replace with the actual path of the destination directory

# Copy the file
shutil.copy(source_path, destination_directory)

'''
# Save the frames as mp4
converter()