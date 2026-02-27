import torch.nn.functional as F

class CEPlusGazeReg: 
    def __init__(self, reg_loss_fn, reg_lambda=1.0):
        self.reg_loss_fn = reg_loss_fn
        self.reg_lambda = reg_lambda
        


    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ce = F.cross_entropy(action_preds, action_targs)

        reg = self.reg_lambda * self.reg_loss_fn(gaze_preds, gaze_targs)

        return ce + reg