import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras.applications import InceptionV3


# Load the pretrained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze the weights of the pretrained model
base_model.trainable = False



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
DOE = 2 * np.pi * np.random.rand(N, N)


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


DOE_tf = tf.math.real(DOE_tf)
variables = tf.Variable(DOE_tf)
learning_rate=0.01
optimizer = tf.optimizers.Adam(learning_rate)

cost_values = [cost]

pp = 4 # power of cost function

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
squared_diff = tf.reduce_sum(tf.square(center_elements - up_elements) + tf.square(center_elements - down_elements) + tf.square(center_elements - left_elements) + tf.square(center_elements - right_elements))
# Sum up the squared differences for all elements
smoothness_info = tf.reduce_sum(squared_diff)

# Print the smoothness information
result = smoothness_info.numpy()
print(result)

plt.figure()

num_iterations = 20
for i in range(num_iterations):

    # Compute the gradient using tf.GradientTape
    with tf.GradientTape() as tape:
        tape.watch(variables)

        #cost = tf.reduce_sum(tf.pow(target_tf - tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))),pp))
        cost = result
        
    gradients = tape.gradient(cost, variables)

    gradients = tf.reshape(gradients,(N,N))

# weightnum = np.random.rand(N, N)
# weightnum = weightnum / np.max(weightnum)
# weights = tf.cast(weights,dtype = tf.float32)
# weights = tf.reshape (weights, (500,500))


    # Clear previous plot and update the current plot
    plt.clf()
    plt.plot(range(len(cost_values)), cost_values)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function')
    plt.annotate(f'Learning Rate: {learning_rate}', xy=(0.05, 0.9), xycoords='axes fraction')
    display.clear_output(wait=True)
    display.display(plt.gcf())


# Perform optimization
    optimizer.apply_gradients([(gradients, variables)])
    
    cost_values.append(cost.numpy())
    if (i + 1) % 100 == 0:
        print("Iteration:", i + 1, "Cost:", cost.numpy())

# Close the plot after the optimization is complete
plt.close()
