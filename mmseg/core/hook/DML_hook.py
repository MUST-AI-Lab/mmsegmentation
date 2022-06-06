from mmcv.runner.hooks import HOOKS, Hook

class LossWeightUpdateHook(Hook):
    def __init__(self,
                 max_weight,
                 by_epoch=False):
        self.by_epoch=by_epoch
        self.max_weight = max_weight
        
    def get_weight(self,runner,max_weight):
        raise NotImplementedError
    
    def before_train_iter(self, runner):
        if not self.by_epoch:
            weight = self.get_weight(runner, self.max_weight)
            runner.model.module.unsup_weight=weight

@HOOKS.register_module()
class StepLossWeightUpdateHook(LossWeightUpdateHook):
    def __init__(self,
                 step,
                 gamma,
                 by_epoch=False,
                 reduce=False,
                 max_weight=40):
        '''
        'gamma': update scale
        '''
        super(StepLossWeightUpdateHook, self).__init__(by_epoch=by_epoch,max_weight=max_weight)
        self.step = step
        self.gamma = gamma
        self.reduce=reduce
        
    def get_weight(self,runner,max_weight):
        cur_iter = runner.iter
        new_weight=self.gamma*(cur_iter//self.step)
        if self.reduce==True and new_weight>=max_weight:
            return 0
        return min(new_weight, max_weight)
    
                 
                 
        
        