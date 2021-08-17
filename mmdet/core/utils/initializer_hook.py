from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class InitializerHook(Hook):

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def before_run(self, runner):
        runner.model.eval()
        for i, data_batch in enumerate(self.dataloader):
            runner.model.val_step(data_batch, runner.optimizer)
            if i % 100 == 0:
                runner.logger.info("Iter of MFS initialization: {} / {}".format(i, len(self.dataloader)))

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
