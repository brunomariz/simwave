#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#if defined(CPU_OPENMP) || defined(GPU_OPENMP)
#include <omp.h>
#endif

#if defined(GPU_OPENACC)
#include <openacc.h>
#endif

#if defined(FLOAT)
typedef float f_type;
#elif defined(DOUBLE)
typedef double f_type;
#else
typedef float f_type;
#endif

double forward(f_type *u, f_type *velocity, f_type *damp, f_type *wavelet,
               size_t wavelet_size, size_t wavelet_count, f_type *coeff,
               size_t *boundary_conditions, size_t *src_points_interval,
               size_t src_points_interval_size, f_type *src_points_values,
               size_t src_points_values_size, size_t *src_points_values_offset,
               size_t *rec_points_interval, size_t rec_points_interval_size,
               f_type *rec_points_values, size_t rec_points_values_size,
               size_t *rec_points_values_offset, f_type *receivers,
               size_t num_sources, size_t num_receivers, size_t nz, size_t nx,
               f_type dz, f_type dx, size_t saving_stride, f_type dt,
               size_t begin_timestep, size_t end_timestep, size_t space_order,
               size_t num_snapshots);
