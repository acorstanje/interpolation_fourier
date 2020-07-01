import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import interpolation_Fourier_release2 as interpF

def do_plot_radial(interp_fourier, max_mode=2):
    radial_interpolator = interp_fourier.get_angular_FFT_interpolator()
    fourier = interp_fourier.get_angular_FFT()
    radial_axis = interp_fourier.get_radial_axis()

    fine_radius = np.arange(0.0, 500.0, 0.5)
        
    fourier_interpolated = radial_interpolator(fine_radius)

    (cosines, sines) = interpF.interp2d_fourier.cos_sin_components(fourier)
    (cosines_fine, sines_fine) = interpF.interp2d_fourier.cos_sin_components(fourier_interpolated)
    
    y_0 = cosines_fine[:, 0]
    y_1 = cosines_fine[:, 1]
    
    plt.figure() # Plot cosine modes
    for k in range(max_mode+1):
        plt.plot(radial_axis, cosines[:, k], 'o', label='cos({0} phi) mode'.format(k) if k > 0 else 'Zero mode')
    
    for k in range(max_mode+1):
        plt.plot(fine_radius, cosines_fine[:, k])
    plt.legend(loc='best')
    plt.xlabel('Radial distance [ m ]')
    plt.ylabel('Value')

    plt.figure() # Sine modes
    for k in range(1, max_mode+1):
        plt.plot(radial_axis, sines[:, k], 'o', label='sin({0} phi) mode'.format(k))

    for k in range(1, max_mode+1):
        plt.plot(fine_radius, sines_fine[:, k])
    plt.legend(loc='best')
    plt.xlabel('Radial distance [ m ]')
    plt.ylabel('Value')


def do_plot_angular(interp_fourier, fixed_radius, values_for_radius):
    # Plots angular interpolation at a fixed radius.
    # Inputs: instance of interp2d_fourier, the fixed radius, and 1D-array of values for that radius
    phi_steps = len(values_for_radius)
    phi_step_degrees = 360.0 / phi_steps
    raw_phi_degrees = np.linspace(0.0, 360.0 - phi_step_degrees , phi_steps)

    fine_phi = np.linspace(0.0, 2*np.pi, 1000)
    fine_points_x = fixed_radius * np.cos(fine_phi)
    fine_points_y = fixed_radius * np.sin(fine_phi)
    
    interp_values = interp_fourier(fine_points_x, fine_points_y)
    interp_values_truncated = interp_fourier(fine_points_x, fine_points_y, max_fourier_mode=2)

    plt.figure()
    plt.plot(raw_phi_degrees, values_for_radius, 'o', label='Values at r={0:.1f} m'.format(fixed_radius))

    plt.plot(fine_phi*180/np.pi, interp_values, label='Fourier series')
    plt.plot(fine_phi*180/np.pi, interp_values_truncated, '--', label='Up to 2nd Fourier mode')
    plt.xlabel('Phi [ deg ]')
    plt.ylabel('Value')
    plt.legend(loc='best')


fname = 'sample_data.txt'
data = np.loadtxt(fname)
(x, y, values) = data.T

# Get instance of interpolator, using given values for (x, y)
fourier_interpolator = interpF.interp2d_fourier(x, y, values)

# Plot radial dependence of the lowest Fourier components
do_plot_radial(fourier_interpolator)

# Plot angular interpolation at two fixed radii
all_radius = np.sqrt(x**2 + y**2)
ordering_indices = fourier_interpolator.get_ordering_indices(x, y)
radius_values = all_radius[ordering_indices][:, 0] # unique radius values
for radius_stepnr in [4, 7]:
    fixed_radius = radius_values[radius_stepnr]
    values_for_radius = values[ordering_indices][radius_stepnr, :]
    do_plot_angular(fourier_interpolator, fixed_radius, values_for_radius)

# Make color plot of f(x, y), using a meshgrid
dist_scale = 250.0
ti = np.linspace(-dist_scale, dist_scale, 1000)
XI, YI = np.meshgrid(ti, ti)

# Get interpolated values at each grid point, calling the instance of interp2d_fourier
ZI = fourier_interpolator(XI, YI)

maxp = np.max(ZI)
plt.figure()
plt.gca().pcolor(XI, YI, ZI, vmax=maxp, vmin=0, cmap=cm.jet)
plt.scatter(x, y, marker='+', s=3, color='w')
mm = cm.ScalarMappable(cmap=cm.jet)
mm.set_array([0.0, maxp])
cbar = plt.colorbar(mm)
cbar.set_label('Values of f(x, y)')
plt.xlabel('x [ m ]')
plt.ylabel('y [ m ]')
plt.xlim(-250, 250)
plt.ylim(-250, 250)
plt.gca().set_aspect('equal')

plt.show()

