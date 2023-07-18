import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Load the target image
# ...

# Create grid
N = 500  # Number of grid points
L = 1  # Grid size
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

pp = 2

# Super-Gaussian profile parameters
m = 16  # Super-Gaussian exponent
w0 = L/6  # Super-Gaussian width

# Calculate Gaussian beam profile, and it is electric field format
R = np.sqrt(X**2 + Y**2)  # Circular shape
initial_profile = np.exp(-((R/w0)**2))


# Generate super-Gaussian profile
target = np.square(np.exp(-(np.abs(X)/w0)**(2*m)) * np.exp(-(np.abs(Y)/(w0/4))**(2*m)))

# Normalize target matrix
target /= np.max(target)

# Define your DOE phase tensor
DOE = torch.from_numpy(np.load('DOE_data.npy')).float()

# Create a tensor for target and initial profiles
target = torch.from_numpy(target).float()
initial_profile = torch.from_numpy(initial_profile).float()



target = target.unsqueeze(0).unsqueeze(0)
initial_profile = initial_profile.unsqueeze(0).unsqueeze(0)



# Create a PyTorch variable for the DOE tensor
variables = torch.nn.Parameter(DOE)


# Define the cost function
def costFn(variables):
    # Perform the desired calculations using PyTorch tensor operations
   
    cost = torch.sum(torch.pow(target - torch.abs(torch.fft.fft2(initial_profile * torch.exp(1j * variables)))/torch.max(torch.abs(torch.fft.fft2(initial_profile * torch.exp(1j * variables)))), pp))
    print (cost)
    return cost

# Define the conjugate gradient optimization method
def conjugate_gradient(costFn, variables, max_iter=100, tolerance=1e-6):
    # Initialize variables
    b = -torch.autograd.grad(costFn(variables), variables)[0]
    p = b.clone()
    r = b.clone()
    i=0
    for i in range(max_iter):
        Ap = torch.autograd.grad(costFn(variables), variables, create_graph=True)[0]
        alpha = torch.sum(r * r) / torch.sum(p * Ap)
        variables.data += alpha * p
        new_r = r - alpha * Ap

        if torch.norm(new_r) < tolerance:
            break
        
        print (i)
        beta = torch.sum(new_r * new_r) / torch.sum(r * r)
        p = new_r + beta * p
        r = new_r

    return variables

# Perform optimization using conjugate gradient
max_iterations = 100
tolerance = 1e-6
variables = conjugate_gradient(costFn, variables, max_iter=max_iterations, tolerance=tolerance)

# Plot the cost and the image (similar to your original code)
# ...
# Plot the cost function
# cost = torch.sum(torch.pow(target - torch.square(torch.abs(torch.fft.fft2(initial_profile * torch.exp(1j * variables)))), pp))
# print (cost)

Result = (torch.abs(torch.fft.fft2(initial_profile * torch.exp(1j * variables)))/torch.max(torch.abs(torch.fft.fft2(initial_profile * torch.exp(1j * variables))))).detach().numpy()
tensor_np = Result.squeeze()

plt.imshow(tensor_np)
plt.colorbar()
plt.show()


