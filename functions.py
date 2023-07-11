import tensorflow as tf
import matplotlib.pyplot as plt


if tf.test.is_gpu_available():
    device = '/GPU:0'
else:
    device = '/CPU:0'

def costfunction(DOE, target, initial_profile, N, t):
 # Set device placement for TensorFlow operations
    with tf.device(device):

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
        learning_rate=1
        optimizer = tf.optimizers.Adadelta(learning_rate)
    
        num_iterations = 30
        for i in range(num_iterations):
        
            # Compute the gradient using tf.GradientTape
            with tf.GradientTape() as tape:
                tape.watch(variables)
                pp = 4 # power of cost function
                cost = tf.reduce_sum(tf.pow(target_tf - tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))), pp))
                
                # consider nearest neighbor 
               # cost = tf.reduce_sum(tf.pow(target_tf - tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))/ tf.reduce_max(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64))))), pp))
        
            gradients = tape.gradient(cost, variables)
            gradients = tf.reshape(gradients,(N,N))
    
        # Perform optimization
            optimizer.apply_gradients([(gradients, variables)])
            cost_values.append(cost.numpy())
    
        optimizer_string = str(optimizer)
        
        # plt.plot(range(1, num_iterations + 1), cost_values, )
        # plt.xlabel('Iteration')
        # plt.ylabel('Cost')
        # plt.title('Cost Function')
        # plt.annotate(f'Learning Rate: {learning_rate}\nIteration: {t+1}', xy=(0.05, 0.9), xycoords='axes fraction')
        # plt.show()

    return cost, variables, learning_rate, optimizer_string
