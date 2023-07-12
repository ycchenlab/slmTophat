import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


# Load the pretrained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze the weights of the pretrained model
base_model.trainable = False

# Define the layer name for intermediate feature extraction
layer_name = 'mixed7'  # Example: using mixed7 layer

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
target = A * np.exp(-(np.abs(X)/w0)**(2*m)) * np.exp(-(np.abs(Y)/(w0/4))**(2*m))

# Gaussian beam parameters
w0 = L/6  # Gaussian beam waist radius

# Calculate Gaussian beam profile
R = np.sqrt(X**2 + Y**2)  # Circular shape
initial_profile = np.exp(-((R/w0)**2))


target /= np.max(target)  # Normalize matrix

# Defining DOE phase
DOE =  np.load('DOE_data.npy')

complex_one = tf.constant(1j, dtype=tf.complex64)

DOE_tf = tf.convert_to_tensor(DOE, dtype=tf.complex64)
target_tf = tf.convert_to_tensor(target, dtype=tf.float32)
initial_profile_tf = tf.convert_to_tensor(initial_profile, dtype=tf.complex64)

DOEphase = tf.exp(complex_one * tf.cast(DOE_tf, tf.complex64))

iterf = tf.signal.fft2d(initial_profile_tf * tf.cast(DOEphase, dtype=tf.complex64))
intf = tf.abs(iterf) / tf.reduce_max(tf.abs(iterf))
differences = target_tf - intf
squaredDifferences = tf.square(differences)
meanSquaredDifferences = tf.reduce_mean(squaredDifferences)
cost = tf.reduce_sum(squaredDifferences)
cost_values = [cost] # save for plot, nothing special

'''
=============================optimization parameters=======================================
'''
DOE_tf = tf.math.real(DOE_tf)
variables = tf.Variable(DOE_tf)
learning_rate=0.05
costType = 2
optimizer = tf.optimizers.Adam(learning_rate)
pp = 2 # power of cost function
'''
===================================================================
'''


'''
============================ Inception base processing ==================

# Preprocess the target and initial profiles
preprocessed_target = preprocess_input(target)
preprocessed_initial = preprocess_input(initial_profile)

preprocessed_target = preprocessed_target.reshape(1, 500, 500, 1)
preprocessed_initial = preprocessed_initial.reshape(1, 500, 500, 1)

# Convert NumPy arrays to TensorFlow tensors
preprocessed_target = tf.convert_to_tensor(preprocessed_target, dtype=tf.float32)
preprocessed_initial = tf.convert_to_tensor(preprocessed_initial, dtype=tf.float32)


# Convert grayscale images to RGB format
preprocessed_target_rgb = tf.image.grayscale_to_rgb(preprocessed_target)
preprocessed_initial_rgb = tf.image.grayscale_to_rgb(preprocessed_initial)

target_features = base_model(preprocessed_target_rgb)
initial_features = base_model(preprocessed_initial_rgb)


# Compute the difference between target and initial features
feature_difference = tf.reduce_mean(tf.square(target_features - initial_features))

======================= no evident contribution ====================
'''

def costFn (variables):
    # Pad the tensor to handle boundaries
    padded_tensor = tf.pad(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))), [[2, 2], [2, 2]])  # Pad with 2 extra rows and columns
    
    input_tensor = tf.constant(padded_tensor)  # Your 500x500 tensor
    
    # Select the center element and adjacent elements
    center_elements = input_tensor[1:-1, 1:-1]
    up_elements = input_tensor[:-2, 1:-1]
    down_elements = input_tensor[2:, 1:-1]
    left_elements = input_tensor[1:-1, :-2]
    right_elements = input_tensor[1:-1, 2:]
      
    # Calculate the squared difference between center element and adjacent elements
    squaredDiff = tf.square(tf.square(center_elements) - tf.square(up_elements)) + tf.square(tf.square(center_elements) - tf.square(down_elements)) + tf.square(tf.square(center_elements) - tf.square(left_elements)) + tf.square(tf.square(center_elements) - tf.square(right_elements))
    
    # Sum up the squared differences for all elements
       
    smoothnessInfo = tf.reduce_sum(squaredDiff)
    # smoothnessInfo += feature_difference
    return smoothnessInfo, squaredDiff

plt.figure()

num_iterations = 100
for i in range(num_iterations):

    # Compute the gradient using tf.GradientTape
    with tf.GradientTape() as tape:
        tape.watch(variables)
        
        if i%2 == 1:
            cost = tf.reduce_sum(tf.pow(target_tf - tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))),pp+2))
            # cost += feature_difference            
                        
        else :
            # costType == 2:
            # cost, squaredDiff = costFn(variables)
            cost = tf.reduce_sum(tf.pow(target_tf - tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))),pp))

                        
    gradients = tape.gradient(cost, variables)
    gradients = tf.reshape(gradients,(N,N))

    # Clear previous plot and update the current plot
    plt.clf()

    # Plot the cost function
    plt.subplot(121)
    plt.plot(range(len(cost_values)), cost_values)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function')
    plt.annotate(f'Learning Rate: {learning_rate}', xy=(0.05, 0.9), xycoords='axes fraction')

    # Plot the image
    plt.subplot(122)
    # show training Tophat
    # plt.imshow(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))))
    
    # inspect smoother 
    plt.imshow(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))))
    plt.title('Training Data')
    plt.axis('image')
    plt.colorbar(shrink=0.5)
    # Display the plot
    display.clear_output(wait=True)
    display.display(plt.gcf())

# Perform optimization
    optimizer.apply_gradients([(gradients, variables)])

    cost_values.append(cost.numpy())


# Close the plot after the optimization is complete
plt.close()
plt.imsave('training data-4.png',10*tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))))
