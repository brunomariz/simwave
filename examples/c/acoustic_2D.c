#include <stdio.h>

#include "../../simwave/kernel/backend/c_code/forward/constant_density/2d/wave.h"

int main() {
    // Declare variables
    f_type *u;
    f_type *velocity;
    f_type *damp;
    f_type *wavelet;
    size_t wavelet_size;
    size_t wavelet_count;
    f_type *coeff;
    size_t *boundary_conditions;
    size_t *src_points_interval;
    size_t src_points_interval_size;
    f_type *src_points_values;
    size_t src_points_values_size;
    size_t *src_points_values_offset;
    size_t *rec_points_interval;
    size_t rec_points_interval_size;
    f_type *rec_points_values;
    size_t rec_points_values_size;
    size_t *rec_points_values_offset;
    f_type *receivers;
    size_t num_sources;
    size_t num_receivers;
    size_t nz;
    size_t nx;
    f_type dz;
    f_type dx;
    size_t saving_stride;
    f_type dt;
    size_t begin_timestep;
    size_t end_timestep;
    size_t space_order;
    size_t num_snapshots;

    // TODO: populate variables

    // Call forward simulation
    forward(u, velocity, damp, wavelet, wavelet_size, wavelet_count, coeff,
            boundary_conditions, src_points_interval, src_points_interval_size,
            src_points_values, src_points_values_size, src_points_values_offset,
            rec_points_interval, rec_points_interval_size, rec_points_values,
            rec_points_values_size, rec_points_values_offset, receivers,
            num_sources, num_receivers, nz, nx, dz, dx, saving_stride, dt,
            begin_timestep, end_timestep, space_order, num_snapshots);

    return 0;
}
