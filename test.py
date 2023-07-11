import theano
import theano.tensor as T

x = T.scalar('x')  # Symbolic variable for the input
f = x**2  # Example function, x squared

gradient = T.grad(f, x)

get_gradient = theano.function([x], gradient)

input_value = 2.0  # Example input value
result = get_gradient(input_value)
print(result)
