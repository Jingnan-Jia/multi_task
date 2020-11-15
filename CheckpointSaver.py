import logging
from typing import TYPE_CHECKING, Dict, Optional
import os
import numpy as np

from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "Checkpoint")
BaseSaveHandler, _ = optional_import("ignite.handlers.checkpoint", "0.4.2", exact_version, "BaseSaveHandler")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.handlers import DiskSaver
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    DiskSaver, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "DiskSaver")


class CheckpointSaver:
    """
    CheckpointSaver acts as an Ignite handler to save checkpoint data into files.
    It supports to save according to metrics result, epoch number, iteration number
    and last model or exception.
    Args:
        save_dir: the target directory to save the checkpoints.
        save_dict: source objects that save to the checkpoint. examples::
            {'network': net, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        file_prefix: prefix for the filenames to which objects will be saved.
        save_final: whether to save checkpoint or session at final iteration or exception.
            If checkpoints are to be saved when an exception is raised, put this handler before
            `StatsHandler` in the handler list, because the logic with Ignite can only trigger
            the first attached handler for `EXCEPTION_RAISED` event.
        final_filename: set a fixed filename to save the final model if `save_final=True`.
            If None, default to `checkpoint_final_iteration=N.pt`.
        save_key_metric: whether to save checkpoint or session when the value of key_metric is
            higher than all the previous values during training.keep 4 decimal places of metric,
            checkpoint name is: {file_prefix}_key_metric=0.XXXX.pth.
        key_metric_name: the name of key_metric in ignite metrics dictionary.
            If None, use `engine.state.key_metric` instead.
        key_metric_n_saved: save top N checkpoints or sessions, sorted by the value of key
            metric in descending order.
        key_metric_filename: set a fixed filename to set the best metric model, if not None,
            `key_metric_n_saved` should be 1 and only keep the best metric model.
        epoch_level: save checkpoint during training for every N epochs or every N iterations.
            `True` is epoch level, `False` is iteration level.
        save_interval: save checkpoint every N epochs, default is 0 to save no checkpoint.
        n_saved: save latest N checkpoints of epoch level or iteration level, 'None' is to save all.
    Note:
        CheckpointHandler can be used during training, validation or evaluation.
        example of saved files:
            - checkpoint_iteration=400.pt
            - checkpoint_iteration=800.pt
            - checkpoint_epoch=1.pt
            - checkpoint_final_iteration=1000.pt
            - checkpoint_key_metric=0.9387.pt
    """

    def __init__(
        self,
        save_dir: str,
        save_dict: Dict,
        name: Optional[str] = None,
        file_prefix: str = "",
        save_final: bool = False,
        final_filename: Optional[str] = None,
        save_key_metric: bool = False,
        key_metric_name: Optional[str] = None,
        key_metric_n_saved: int = 1,
        key_metric_filename: Optional[str] = None,
        epoch_level: bool = True,
        save_interval: int = 0,
        n_saved: Optional[int] = None,
    ) -> None:
        assert save_dir is not None, "must provide directory to save the checkpoints."
        self.save_dir = save_dir
        assert save_dict is not None and len(save_dict) > 0, "must provide source objects to save."
        for k, v in save_dict.items():
            if hasattr(v, "module"):
                save_dict[k] = v.module
        self.save_dict = save_dict
        self.logger = logging.getLogger(name)
        self.epoch_level = epoch_level
        self.save_interval = save_interval
        self._final_checkpoint = self._key_metric_checkpoint = self._interval_checkpoint = None
        self._name = name

        class _DiskSaver(DiskSaver):
            """
            Enhance the DiskSaver to support fixed filename.
            """

            def __init__(self, dirname: str, filename: Optional[str] = None):
                super().__init__(dirname=dirname, require_empty=False)
                self.filename = filename

            def __call__(self, checkpoint: Dict, filename: str, metadata: Optional[Dict] = None) -> None:
                if self.filename is not None:
                    filename = self.filename
                super().__call__(checkpoint=checkpoint, filename=filename, metadata=metadata)

            def remove(self, filename: str) -> None:
                if self.filename is not None:
                    filename = self.filename
                super().remove(filename=filename)

        if save_final:

            def _final_func(engine: Engine):
                return engine.state.iteration

            self._final_checkpoint = Checkpoint(
                to_save=self.save_dict,
                save_handler=_DiskSaver(dirname=self.save_dir, filename=final_filename),
                filename_prefix=file_prefix,
                score_function=_final_func,
                score_name="final_iteration",
            )

        if save_key_metric:

            def _score_func(engine: Engine):
                if isinstance(key_metric_name, str):
                    metric_name = key_metric_name
                elif hasattr(engine.state, "key_metric_name") and isinstance(engine.state.key_metric_name, str):
                    metric_name = engine.state.key_metric_name
                else:
                    raise ValueError(
                        f"Incompatible values: save_key_metric=True and key_metric_name={key_metric_name}."
                    )

                val_log_dir = self.save_dir + '/val_log.csv'
                if os.path.exists(val_log_dir):
                    val_log = np.genfromtxt(val_log_dir, dtype='str', delimiter=',')
                else:
                    val_log = ['epoch', 'val_loss']
                val_log = np.vstack([val_log, [engine.state.epoch, round(engine.state.metrics[metric_name], 4)]])
                np.savetxt(val_log_dir, val_log, fmt='%s', delimiter=',')

                return round(engine.state.metrics[metric_name], 4)

            if key_metric_filename is not None and key_metric_n_saved > 1:
                raise ValueError("if using fixed filename to save the best metric model, we should only save 1 model.")

            self._key_metric_checkpoint = Checkpoint(
                to_save=self.save_dict,
                save_handler=_DiskSaver(dirname=self.save_dir, filename=key_metric_filename),
                filename_prefix=file_prefix,
                score_function=_score_func,
                score_name="key_metric",
                n_saved=key_metric_n_saved,
            )

        if save_interval > 0:

            def _interval_func(engine: Engine):
                return engine.state.epoch if self.epoch_level else engine.state.iteration

            self._interval_checkpoint = Checkpoint(
                to_save=self.save_dict,
                save_handler=_DiskSaver(dirname=self.save_dir),
                filename_prefix=file_prefix,
                score_function=_interval_func,
                score_name="epoch" if self.epoch_level else "iteration",
                n_saved=n_saved,
            )

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if self._final_checkpoint is not None:
            engine.add_event_handler(Events.COMPLETED, self.completed)
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)
        if self._key_metric_checkpoint is not None:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.metrics_completed)
        if self._interval_checkpoint is not None:
            if self.epoch_level:
                engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self.interval_completed)
            else:
                engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.save_interval), self.interval_completed)

    def completed(self, engine: Engine) -> None:
        """Callback for train or validation/evaluation completed Event.
        Save final checkpoint if configure save_final is True.
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        assert callable(self._final_checkpoint), "Error: _final_checkpoint function not specified."
        self._final_checkpoint(engine)
        assert self.logger is not None
        assert hasattr(self.logger, "info"), "Error, provided logger has not info attribute."
        self.logger.info(f"Train completed, saved final checkpoint: {self._final_checkpoint.last_checkpoint}")

    def exception_raised(self, engine: Engine, e: Exception) -> None:
        """Callback for train or validation/evaluation exception raised Event.
        Save current data as final checkpoint if configure save_final is True. This callback may be skipped
        because the logic with Ignite can only trigger the first attached handler for `EXCEPTION_RAISED` event.
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            e: the exception caught in Ignite during engine.run().
        """
        assert callable(self._final_checkpoint), "Error: _final_checkpoint function not specified."
        self._final_checkpoint(engine)
        assert self.logger is not None
        assert hasattr(self.logger, "info"), "Error, provided logger has not info attribute."
        self.logger.info(f"Exception_raised, saved exception checkpoint: {self._final_checkpoint.last_checkpoint}")
        raise e

    def metrics_completed(self, engine: Engine) -> None:
        """Callback to compare metrics and save models in train or validation when epoch completed.
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        assert callable(self._key_metric_checkpoint), "Error: _key_metric_checkpoint function not specified."
        self._key_metric_checkpoint(engine)

    def interval_completed(self, engine: Engine) -> None:
        """Callback for train epoch/iteration completed Event.
        Save checkpoint if configure save_interval = N
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        assert callable(self._interval_checkpoint), "Error: _interval_checkpoint function not specified."
        self._interval_checkpoint(engine)
        assert self.logger is not None
        assert hasattr(self.logger, "info"), "Error, provided logger has not info attribute."
        if self.epoch_level:
            self.logger.info(f"Saved checkpoint at epoch: {engine.state.epoch}")
        else:
            self.logger.info(f"Saved checkpoint at iteration: {engine.state.iteration}")