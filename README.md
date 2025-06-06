# dynamic-vector-fields
Use a Gaussian Process with RBF Kernel to Model Dynamic Vector Fields

This code allows a user to input data from a 2D vector field over discrete time steps
The code will complete a few different tasks:
1) model the flow of a particle dropped into the vector field
2) Fit a Gaussian process to the temporal evolution of the vector field at a chosen spatial point
     2) (a) The Gaussian process uses an RBF kernel where the kernel parameters are derived via
          cross-validation as well as SKLearn's GaussianProcessRegressor functionality
