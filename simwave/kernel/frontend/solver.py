import numpy as np
from simwave.kernel.backend.middleware import Middleware
from simwave.kernel.frontend.dataset_writer import DatasetWriter


class Solver:
    """
    Acoustic solver for the simulation.

    Parameters
    ----------
    space_model : SpaceModel
        Space model object.
    time_model: TimeModel
        Time model object.
    sources : Source
        Source object.
    receivers : Receiver
        Receiver object.
    wavelet : Wavelet
        Wavelet object.
    compiler : Compiler
        Backend compiler object.
    """

    def __init__(
        self, space_model, time_model, sources, receivers, wavelet, compiler=None
    ):

        self._space_model = space_model
        self._time_model = time_model
        self._sources = sources
        self._receivers = receivers
        self._wavelet = wavelet
        self._compiler = compiler

        # create a middleware to communicate with backend
        self._middleware = Middleware(compiler=self.compiler)

    @property
    def space_model(self):
        """Space model object."""
        return self._space_model

    @property
    def time_model(self):
        """Time model object."""
        return self._time_model

    @property
    def sources(self):
        """Source object."""
        return self._sources

    @property
    def receivers(self):
        """Receiver object."""
        return self._receivers

    @property
    def wavelet(self):
        """Wavelet object."""
        return self._wavelet

    @property
    def compiler(self):
        """Compiler object."""
        return self._compiler

    @property
    def snapshot_indexes(self):
        """List of snapshot indexes (wavefields to be saved)."""

        # if saving_stride is 0, only saves the last timestep
        if self.time_model.saving_stride == 0:
            return [self.time_model.time_indexes[-1]]

        snap_indexes = list(
            range(
                self.time_model.time_indexes[0],
                self.time_model.timesteps,
                self.time_model.saving_stride,
            )
        )

        return snap_indexes

    @property
    def num_snapshots(self):
        """Number of snapshots (wavefields to be saved)."""
        return len(self.snapshot_indexes)

    @property
    def shot_record(self):
        """Return the shot record array."""
        u_recv = np.zeros(
            shape=(self.time_model.timesteps, self.receivers.count),
            dtype=self.space_model.dtype,
        )

        return u_recv

    @property
    def u_full(self):
        """Return the complete grid (snapshots, nz. nz [, ny])."""

        # add 2 halo snapshots (second order in time)
        snapshots = self.num_snapshots + 2

        # define the final shape (snapshots + domain)
        shape = (snapshots,) + self.space_model.extended_shape

        return np.zeros(shape, dtype=self.space_model.dtype)

    def forward(self):
        """
        Run the forward propagator.

        Returns
        ----------
        ndarray
            Full wavefield with snapshots.
        ndarray
            Shot record.
        """

        src_points, src_values, src_offsets = (
            self.sources.interpolated_points_and_values
        )
        rec_points, rec_values, rec_offsets = (
            self.receivers.interpolated_points_and_values
        )

        data = {
            "u_full": {
                "dataset_data": self.u_full,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "velocity_model": {
                "dataset_data": self.space_model.extended_velocity_model,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "density_model": {
                "dataset_data": self.space_model.extended_density_model,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "damping_mask": {
                "dataset_data": self.space_model.damping_mask,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "wavelet": {
                "dataset_data": self.wavelet.values,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "wavelet_size": {
                "dataset_data": self.wavelet.timesteps,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "wavelet_count": {
                "dataset_data": self.wavelet.num_sources,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "second_order_fd_coefficients": {
                "dataset_data": self.space_model.fd_coefficients(2),
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "first_order_fd_coefficients": {
                "dataset_data": self.space_model.fd_coefficients(1),
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "boundary_condition": {
                "dataset_data": self.space_model.boundary_condition,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "src_points_interval": {
                "dataset_data": src_points,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "src_points_interval_size": {
                "dataset_data": len(src_points),
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "src_points_values": {
                "dataset_data": src_values,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "src_points_values_offset": {
                "dataset_data": src_offsets,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "src_points_values_size": {
                "dataset_data": len(src_values),
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "rec_points_interval": {
                "dataset_data": rec_points,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "rec_points_interval_size": {
                "dataset_data": len(rec_points),
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "rec_points_values": {
                "dataset_data": rec_values,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "rec_points_values_offset": {
                "dataset_data": rec_offsets,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "rec_points_values_size": {
                "dataset_data": len(rec_values),
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "shot_record": {
                "dataset_data": self.shot_record,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "num_sources": {
                "dataset_data": self.sources.count,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "num_receivers": {
                "dataset_data": self.receivers.count,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "grid_spacing": {
                "dataset_data": self.space_model.grid_spacing,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "saving_stride": {
                "dataset_data": self.time_model.saving_stride,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "dt": {
                "dataset_data": self.time_model.dt,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "begin_timestep": {
                "dataset_data": 1,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "end_timestep": {
                "dataset_data": self.time_model.timesteps,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "space_order": {
                "dataset_data": self.space_model.space_order,
                "dataset_attributes": {"example_attribute": "hello"},
            },
            "num_snapshots": {
                "dataset_data": self.u_full.shape[0],
                "dataset_attributes": {"example_attribute": "hello"},
            },
        }

        DatasetWriter.write_dataset(data, "tmp/example_data.h5")

        # u_full, recv = self._middleware.exec(
        #     operator='forward',
        #     u_full=self.u_full,
        #     velocity_model=self.space_model.extended_velocity_model,
        #     density_model=self.space_model.extended_density_model,
        #     damping_mask=self.space_model.damping_mask,
        #     wavelet=self.wavelet.values,
        #     wavelet_size=self.wavelet.timesteps,
        #     wavelet_count=self.wavelet.num_sources,
        #     second_order_fd_coefficients=self.space_model.fd_coefficients(2),
        #     first_order_fd_coefficients=self.space_model.fd_coefficients(1),
        #     boundary_condition=self.space_model.boundary_condition,
        #     src_points_interval=src_points,
        #     src_points_interval_size=len(src_points),
        #     src_points_values=src_values,
        #     src_points_values_offset=src_offsets,
        #     src_points_values_size=len(src_values),
        #     rec_points_interval=rec_points,
        #     rec_points_interval_size=len(rec_points),
        #     rec_points_values=rec_values,
        #     rec_points_values_offset=rec_offsets,
        #     rec_points_values_size=len(rec_values),
        #     shot_record=self.shot_record,
        #     num_sources=self.sources.count,
        #     num_receivers=self.receivers.count,
        #     grid_spacing=self.space_model.grid_spacing,
        #     saving_stride=self.time_model.saving_stride,
        #     dt=self.time_model.dt,
        #     begin_timestep=1,
        #     end_timestep=self.time_model.timesteps,
        #     space_order=self.space_model.space_order,
        #     num_snapshots=self.u_full.shape[0]
        # )

        # # remove time halo region
        # u_full = self.time_model.remove_time_halo_region(u_full)

        # # remove spatial halo region
        # u_full = self.space_model.remove_halo_region(u_full)

        # return u_full, recv
