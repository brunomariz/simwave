# Author: Bruno Mariz
# Run: python3 acoustic_2D_generate_args.py

from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)



# available language options:
# c (sequential)
# cpu_openmp (parallel CPU)
# gpu_openmp (GPU)
# gpu_openacc (GPU)
compiler_options = {
    'c': {
        'cc': 'gcc',
        'language': 'c',
        'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
    },
    'cpu_openmp': {
        'cc': 'gcc',
        'language': 'cpu_openmp',
        'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared -fopenmp'
    },
    'gpu_openmp': {
        'cc': 'clang',
        'language': 'gpu_openmp',
        'cflags': '-O3 -fPIC -ffast-math -fopenmp \
                   -fopenmp-targets=nvptx64-nvidia-cuda \
                   -Xopenmp-target -march=sm_75'
    },
    'gpu_openacc': {
        'cc': 'pgcc',
        'language': 'gpu_openacc',
        'cflags': '-O3 -fPIC -acc:gpu -gpu=pinned -mp'
    },
}

selected_compiler = compiler_options['c']

# set compiler options
compiler = Compiler(
    cc=selected_compiler['cc'],
    language=selected_compiler['language'],
    cflags=selected_compiler['cflags']
)

# Velocity model
vel = np.zeros(shape=(512, 512), dtype=np.float32)
vel[:] = 1500.0
vel[100:] = 2000.0

# create the space model
space_model = SpaceModel(
    bounding_box=(0, 5120, 0, 5120),
    grid_spacing=(10, 10),
    velocity_model=vel,
    space_order=4,
    dtype=np.float32
)

# config boundary conditions
# (none,  null_dirichlet or null_neumann)
space_model.config_boundary(
    damping_length=0,
    boundary_condition=(
        "null_neumann", "null_dirichlet",
        "none", "null_dirichlet"
    ),
    damping_polynomial_degree=3,
    damping_alpha=0.001
)

# create the time model
time_model = TimeModel(
    space_model=space_model,
    tf=1.0,
    saving_stride=0
)

# create the set of sources
source = Source(
    space_model,
    coordinates=[(2560, 2560)],
    window_radius=4
)

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(2560, i) for i in range(0, 5120, 10)],
    window_radius=4
)

# create a ricker wavelet with 10hz of peak frequency
ricker = RickerWavelet(10.0, time_model)

# create the solver
solver = Solver(
    space_model=space_model,
    time_model=time_model,
    sources=source,
    receivers=receiver,
    wavelet=ricker,
    compiler=compiler
)

print("Timesteps:", time_model.timesteps)

# generate args
args = {
    "solver.u_full":{
        "name":"u_full",
        "value":solver.u_full},
    "solver.space_model.extended_velocity_model":{
        "name":"velocity_model",
        "value":solver.space_model.extended_velocity_model},
    "solver.space_model.extended_density_model":{
        "name":"density_model",
        "value":solver.space_model.extended_density_model},
    "solver.space_model.damping_mask":{
        "name":"damping_mask",
        "value":solver.space_model.damping_mask},
    "solver.wavelet.values":{
        "name":"wavelet",
        "value":solver.wavelet.values},
    "solver.wavelet.timesteps":{
        "name":"wavelet_size",
        "value":solver.wavelet.timesteps},
    "solver.wavelet.num_sources":{
        "name":"wavelet_count",
        "value":solver.wavelet.num_sources},
    "solver.space_model.fd_coefficients(2)":{
        "name":"second_order_fd_coefficients",
        "value":solver.space_model.fd_coefficients(2)},
    "solver.space_model.fd_coefficients(1)":{
        "name":"first_order_fd_coefficients",
        "value":solver.space_model.fd_coefficients(1)},
    "solver.space_model.boundary_condition":{
        "name":"boundary_condition",
        "value":solver.space_model.boundary_condition},
    "solver.sources.interpolated_points_and_values[0]":{
        "name":"src_points_interval",
        "value":solver.sources.interpolated_points_and_values[0]},
    "len(solver.sources.interpolated_points_and_values[0])":{
        "name":"src_points_interval_size",
        "value":len(solver.sources.interpolated_points_and_values[0])},
    "solver.sources.interpolated_points_and_values[1]":{
        "name":"src_points_values",
        "value":solver.sources.interpolated_points_and_values[1]},
    "solver.sources.interpolated_points_and_values[2]":{
        "name":"src_points_values_offset",
        "value":solver.sources.interpolated_points_and_values[2]},
    "len(solver.sources.interpolated_points_and_values[1])":{
        "name":"src_points_values_size",
        "value":len(solver.sources.interpolated_points_and_values[1])},
    "solver.receivers.interpolated_points_and_values[0]":{
        "name":"rec_points_interval",
        "value":solver.receivers.interpolated_points_and_values[0]},
    "len(solver.receivers.interpolated_points_and_values[0])":{
        "name":"rec_points_interval_size",
        "value":len(solver.receivers.interpolated_points_and_values[0])},
    "solver.receivers.interpolated_points_and_values[1]":{
        "name":"rec_points_values",
        "value":solver.receivers.interpolated_points_and_values[1]},
    "solver.receivers.interpolated_points_and_values[2]":{
        "name":"rec_points_values_offset",
        "value":solver.receivers.interpolated_points_and_values[2]},
    "len(solver.receivers.interpolated_points_and_values[1])":{
        "name":"rec_points_values_size",
        "value":len(solver.receivers.interpolated_points_and_values[1])},
    "solver.shot_record":{
        "name":"shot_record",
        "value":solver.shot_record},
    "solver.sources.count":{
        "name":"num_sources",
        "value":solver.sources.count},
    "solver.receivers.count":{
        "name":"num_receivers",
        "value":solver.receivers.count},
    "solver.space_model.grid_spacing":{
        "name":"grid_spacing",
        "value":solver.space_model.grid_spacing},
    "solver.time_model.saving_stride":{
        "name":"saving_stride",
        "value":solver.time_model.saving_stride},
    "solver.time_model.dt":{
        "name":"dt",
        "value":solver.time_model.dt},
    "1":{
        "name":"begin_timestep",
        "value":1},
    "solver.time_model.timesteps":{
        "name":"end_timestep",
        "value":solver.time_model.timesteps},
    "solver.space_model.space_order":{
        "name":"space_order",
        "value":solver.space_model.space_order},
    "solver.u_full.shape[0]":{
        "name":"num_snapshots",
        "value":solver.u_full.shape[0]}
}

import json
for arg in args:
    with open(f"args/{args[arg]['name']}", "w+") as f:
        value = args[arg]["value"]
        try:
            value = json.dumps(value)
        except TypeError:
            if type(value)=="ndarray":
                value = json.dumps(value.tolist())
            elif type(value) == "tuple":
                value = json.dumps(list(value))
        f.write(str(value))
        print(args[arg]["name"])

# run the forward
u_full, recv = solver.forward()

print("u_full shape:", u_full.shape)
plot_velocity_model(space_model.velocity_model)
plot_wavefield(u_full[-1])
plot_shotrecord(recv)
