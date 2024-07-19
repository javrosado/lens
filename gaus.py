import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# Example power array (replace this with your actual data)
power_array = np.random.rand(23, 23)

# Generate x and y coordinates
x = np.linspace(0, power_array.shape[1]-1, power_array.shape[1])
y = np.linspace(0, power_array.shape[0]-1, power_array.shape[0])
x, y = np.meshgrid(x, y)

# Flatten the data for curve fitting
x_data = np.vstack((x.ravel(), y.ravel()))
y_data = power_array.ravel()

# Initial guess
initial_guess = (3, power_array.shape[1]//2, power_array.shape[0]//2, 1, 1, 0, 0)

# Fit the 2D Gaussian
popt, pcov = curve_fit(gaussian_2d, x_data, y_data, p0=initial_guess)
amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt

print(f"Amplitude: {amplitude}")
print(f"Center (xo, yo): ({xo}, {yo})")
print(f"Sigma_x: {sigma_x}")
print(f"Sigma_y: {sigma_y}")
print(f"Theta: {theta}")
print(f"Offset: {offset}")

# Visualize the fit
fit = gaussian_2d((x, y), *popt).reshape(power_array.shape)
plt.figure()
plt.imshow(power_array, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.contour(x, y, fit, colors='w')
plt.title("2D Gaussian Fit")
plt.show()

# Calculate distances to pixels
distances = np.sqrt((x - xo)**2 + (y - yo)**2)