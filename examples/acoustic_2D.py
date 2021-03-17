from pywave import *
import numpy as np


# Velocity model
vel = np.zeros(shape=(512,512), dtype=np.float32)
vel[:] = 1500.0
vel[100:] = 2000.0

# create the space model
# bounding_box
space_model = SpaceModel(
    bounding_box=(0, 5120, 0, 5120),
    grid_spacing=(10, 10),
    velocity_model=vel,
    space_order=4
)

# config boundary conditions
# (null,  null_dirichlet or null_neumann)
space_model.config_boundary(
    damping_length=0,
    boundary_condition=("null_neumann", "null_dirichlet", "none", "null_dirichlet"),
    damping_polynomial_degree=3,
    damping_alpha=0.001
)

# create the time model
time_model = TimeModel(
    space_model=space_model,
    tf=1.0
)

# create the set of sources
source = Source(
    space_model,
    coordinates=[(2560,2560)],
    window_radius=1
)

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(2560,i) for i in range(0,5112,10)],
    window_radius=1
)

# create a ricker wavelet with 10hz of peak frequency
ricker = RickerWavelet(10.0, time_model)

#plot_wavelet(time_model.time_values, ricker.values, show=True)

# create the solver
solver = Solver(
    space_model=space_model,
    time_model=time_model,
    sources=source,
    receivers=receiver,
    wavelet=ricker,
    saving_stride=0
)

# run the forward
u_full, recv = solver.forward()

print("u_full shape:", u_full.shape)
plot_wavefield(u_full[-1])
plot_shotrecord(recv)
