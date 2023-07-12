import tensorflow as tf
import matplotlib.pyplot as plt
import os
import imageio
from datetime import datetime


def costfunction(DOE, target, initial_profile, N, t, LR, costType):
 # Set device placement for TensorFlow operations


# turn numpy objects into tensor (preparation for deep-learning-based optimization methods)
    complex_one = tf.constant(1j, dtype=tf.complex64)
    DOE_tf = tf.convert_to_tensor(DOE, dtype=tf.complex64)
    target_tf = tf.convert_to_tensor(target, dtype=tf.float32)
    initial_profile_tf = tf.convert_to_tensor(initial_profile, dtype=tf.complex64)
    DOEphase = tf.exp(complex_one * tf.cast(DOE_tf, tf.complex64))

    iterf = tf.signal.fft2d(initial_profile_tf * tf.cast(DOEphase, dtype=tf.complex64))
    intf = tf.abs(iterf) / tf.reduce_max(tf.abs(iterf))
    differences = target_tf - intf
    squaredDifferences = tf.square(differences)
    cost = tf.reduce_sum(squaredDifferences)
    cost_values = []
    plt.figure()
    
# initialize the optimization parameters
    DOE_tf = tf.math.real(DOE_tf)
    variables = tf.Variable(DOE_tf)
    learning_rate=LR
    optimizer = tf.optimizers.Adadelta(learning_rate)
    
    
    def costFnSmoothing (variables):
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
        squaredDiff = tf.reduce_sum(tf.square(center_elements - up_elements) + tf.square(center_elements - down_elements) + tf.square(center_elements - left_elements) + tf.square(center_elements - right_elements))
        
        # Sum up the squared differences for all elements
        
        smoothnessInfo = tf.reduce_sum(squaredDiff)
        return smoothnessInfo

    
    def costFnEfficient (variables):
        
        
        return something
    

    def costFnSimple (variables):
        pp = 2 # power of cost function
        cost = tf.reduce_sum(tf.pow(target_tf - tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))), pp))

        return cost

    num_iterations = 30
    for i in range(num_iterations):
    

        # Compute the gradient using tf.GradientTape
        with tf.GradientTape() as tape:
            tape.watch(variables)
        
            if costType==1:
                cost = costFnSimple(variables)
            
            
            elif costType==2:
            # consider nearest neighbor 
                cost = costFnSmoothing(variables)
                
            elif costType==3:
                # use the most recent paper
                cost = 
        
                        
            gradients = tape.gradient(cost, variables)
            gradients = tf.reshape(gradients,(N,N))

    # Perform optimization
        optimizer.apply_gradients([(gradients, variables)])
        cost_values.append(cost.numpy())
        print (cost.numpy())
        
        
    optimizer_string = str(optimizer)
    
    plt.plot(range(1, num_iterations + 1), cost_values, )
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function')
    plt.annotate(f'Learning Rate: {learning_rate}\nIteration: {t+1}', xy=(0.05, 0.9), xycoords='axes fraction')
    plt.show()

    return cost, variables, learning_rate, optimizer_string


def get_file_creation_time(file_path):
    # Get the creation time of a file
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp)


def converter():
        # Folder path containing the images
    folder_path = 'C:/git repo/python script/tempPNG'
    
    # List to store image file names
    image_files = []
    
    # Iterate through the files in the folder
    for file in os.listdir(folder_path):
        if file.startswith('plot_') and file.endswith('.png'):
            image_files.append(os.path.join(folder_path, file))
    
    # Sort the image files based on creation time
    image_files.sort(key=get_file_creation_time)
    
    # Create an empty list to store frames
    frames = []
    
    # Read the images and add them to the frames list
    for image_file in image_files:
        frame = imageio.imread(image_file)
        frames.append(frame)
    
    # Set the file path and name for the output video
    output_path = 'output.mp4'
    
    writer = imageio.get_writer(output_path, format='mp4', mode='I', fps=10)
    
    # Write the frames to the video
    for frame in frames:
        writer.append_data(frame)
    
    # Close the writer
    writer.close()
