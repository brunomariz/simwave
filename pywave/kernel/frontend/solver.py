import numpy as np
from pywave.kernel.backend.middleware import Middleware


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
    saving_stride : int
        Skipping factor when saving the wavefields.
        If saving_jump is 0, only the last wavefield is saved. Default is 0.
    compiler : Compiler
        Backend compiler object.
    """
    def __init__(self, space_model, time_model, sources,
                 receivers, wavelet, saving_stride=0, compiler=None):

        self._space_model = space_model
        self._time_model = time_model
        self._sources = sources
        self._receivers = receivers
        self._wavelet = wavelet
        self._saving_stride = saving_stride
        self._compiler = compiler

        # create a middleware to communicate with backend
        self._middleware = Middleware(compiler=self.compiler)

        # validate the saving stride
        if not (0 <= self.saving_stride <= self.time_model.timesteps):
            raise Exception(
                "Saving jumps can not be less than zero or "
                "greater than the number of timesteps."
            )

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
    def saving_stride(self):
        """Skipping factor when saving the wavefields."""
        return self._saving_stride

    @property
    def compiler(self):
        """Compiler object."""
        return self._compiler

    @property
    def snapshot_indexes(self):
        """List of snapshot indexes (wavefields to be saved)."""

        # if saving_stride is 0, only saves the last timestep
        if self.saving_stride == 0:
            return [self.time_model.time_indexes[-1]]

        snap_indexes = list(
            range(
                self.time_model.time_indexes[0],
                self.time_model.timesteps,
                self.saving_stride
            )
        )

        # always add the last timestep
        if snap_indexes[-1] != self.time_model.time_indexes[-1]:
            snap_indexes.append(self.time_model.time_indexes[-1])

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
            dtype=np.float32
        )

        return u_recv

    @property
    def u_full(self):
        """Return the complete grid (snapshots, nz. nz [, ny])."""
        shape = (self.num_snapshots,) + self.space_model.extended_shape

        return np.zeros(shape, dtype=np.float32)

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

        src_points, src_values = self.sources.interpolated_points_and_values
        rec_points, rec_values = self.receivers.interpolated_points_and_values

        u_full, recv = self._middleware.exec(
            operator='forward',
            u_full=self.u_full,
            velocity_model=self.space_model.extended_velocity_model,
            density_model=self.space_model.extended_density_model,
            damping_mask=self.space_model.damping_mask,
            wavelet=self.wavelet.values,
            fd_coefficients=self.space_model.fd_coefficients,
            boundary_condition=self.space_model.boundary_condition,
            src_points_interval=src_points,
            src_points_values=src_values,
            rec_points_interval=rec_points,
            rec_points_values=rec_values,
            shot_record=self.shot_record,
            num_sources=self.sources.count,
            num_receivers=self.receivers.count,
            grid_spacing=self.space_model.grid_spacing,
            saving_stride=self.saving_stride,
            dt=self.time_model.dt,
            begin_timestep=int(self.time_model.time_indexes[0]),
            end_timestep=self.time_model.timesteps,
            space_order=self.space_model.space_order
        )

        # remove halo region
        u_full = self.space_model.remove_halo_region(u_full)

        return u_full, recv