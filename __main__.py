import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from functions import costfunction
from functions import converter
import tensorflow as tf
import imageio
import time
import os

# Loading the target image
# Create grid
N = 500  # Number of grid points
L = 1  # Grid size
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# Super-Gaussian profile parameters
m = 16  # Super-Gaussian exponent
A = 100  # Super-Gaussian amplitude
w0 = L/6  # Super-Gaussian width

# Generate super-Gaussian profile
target = np.square(A * np.exp(-(np.abs(X)/w0)**(2*m)) * np.exp(-(np.abs(Y)/(w0/4))**(2*m)))

# Gaussian beam parameters
w0 = L/6  # Gaussian beam waist radius

# Calculate Gaussian beam profile
R = np.sqrt(X**2 + Y**2)  # Circular shape
initial_profile = np.exp(-((R/w0)**2))

target /= np.max(target)  # Normalize matrix

# Defining DOE phase
DOE = np.load('DOE_data.npy')

s = 30

# Create an empty list to store frames
frames = []


# conjugate gradient method algorithm
# Iterate to calculate the phase value
for t in range(s):
    # Start the timer
    start_time = time.time()
    
    DOEphase = np.exp(1j * DOE)

####################    # Forward iteration
    iterf = fft2(initial_profile * DOEphase)
    intf = np.square(np.abs(iterf) / np.max(np.abs(iterf)))
    angf = np.angle(iterf)
    A = target * np.exp(1j * angf)

    # Backward iteration
    iterb = ifft2(A)
    angb = np.angle(iterb)
    DOE = angb
    error = target - intf / np.max(intf)  # Calculate error
    intf /= np.max(intf)
    E = np.sum(np.abs(error)) / (N * N)
    differences = target - intf
    squaredDifferences = differences**2
    meanSquaredDifferences = np.mean(squaredDifferences)
    
    
    ############################ Optimization funciton
    
    learning_rate=1
    costType = 1 # costType: 1 = simple cost function, 2 = smoothing neighbor pixels
    cost, DOE_tf, learning_rate, optimizer_string = costfunction(DOE, target, initial_profile, N,t,learning_rate, costType)
    DOE = DOE_tf.numpy() 

    rmse = np.sqrt(meanSquaredDifferences)
    
    if E < 0.005:
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
    save_path = r'C:\git repo\SLM_program\tempPNG\\'
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
plt.imsave('DOE.png', DOE, cmap='gray')

# Save the frames as mp4
converter()