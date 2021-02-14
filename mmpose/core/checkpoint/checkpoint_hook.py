import os
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.hooks.checkpoint import CheckpointHook

@HOOKS.register_module()
class CheckpointHookV2(CheckpointHook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
        start_epoch (int): Begin saving checkpoint from the given epoch.
        with_indicator: Whether mark checkpoint file with epoch or iteration.
            If set False, only the latest checkpoint will be saved.  
            Default: True
    """
    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.start_epoch = -1
        self.with_indicator = True

    @master_only
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        self.start_epoch = self.args.pop('start_epoch', -1) if self.start_epoch == -1 else self.start_epoch
        if self.start_epoch != -1 and (runner.epoch + 1) < self.start_epoch:
            return

        runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        self.with_indicator = self.args.pop('with_indicator', True) if \
            self.with_indicator else self.with_indicator
        if self.with_indicator:
            runner.save_checkpoint(
                self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        else:
            # if not with_indicator, only the latest_epoch checkpoint will be saved
            runner.save_checkpoint(
                self.out_dir, save_optimizer=self.save_optimizer, with_indicator=self.with_indicator, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pth')
            current_epoch = runner.epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(epoch))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break