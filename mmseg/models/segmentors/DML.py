from .. import build_segmentor, SEGMENTORS, build_loss
from .multi_stream_segmentor import MultiStreamSegmentor
from mmseg.utils import dict_split, weighted_loss
import torch.nn.functional as F
import copy

@SEGMENTORS.register_module()
class DML(MultiStreamSegmentor):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(DML, self).__init__(
            dict(branch1=build_segmentor(copy.deepcopy(model)), branch2=build_segmentor(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        if train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            
        assert self.branch1.decode_head.ignore_index == self.branch2.decode_head.ignore_index, "'ignore_index' of two branches must be the same."
        self.ignore_index = self.branch1.decode_head.ignore_index
        # Register a new loss module for culculating CPS loss
        self.criterion = build_loss(dict(
            type='KLDivergence', loss_weight=1.0))

    #def forward_train(self, img, img_metas, **kwargs):

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        
        
        gt_semantic_seg = kwargs['gt_semantic_seg']
        log_vars = dict()
        
        #!!!TODO: Add aux head loss calculation
        
        #optimizer["branch1"].zero_grad()
        #optimizer["branch2"].zero_grad()
        
        """
        print("^^^^^^^^^^^^^^^^^^")
        for key, item in data_groups.items():
            print(key, len(item))
            for subkey, subitem in item.items():
                print(subkey, len(subitem))
        print("vvvvvvvvvvvvvvvvvv")
        """
        pred_1 = self.branch1.encode_decode(img, img_metas)

        #with torch.no_grad():
        pred_2 = self.branch2.encode_decode(img, img_metas)

        # 1. calculate CPS loss for all data
        log_vars_cps = self._forward_DML_train(pred_1, pred_2)
        log_vars.update(log_vars_cps)

        # 2. calculate supervised loss (cross-entropy) for labeled data
        log_vars_sup = dict()
        log_vars_sup['sup_loss1'] = self._loss(self.branch1, pred_1, gt_semantic_seg.squeeze(1))#gt_seg.squeeze(1)
        log_vars_sup['sup_loss2'] = self._loss(self.branch2, pred_2, gt_semantic_seg.squeeze(1))
        log_vars.update(log_vars_sup)
        log_vars['unsup_weight'] =self.unsup_weight
        '''
        if log_vars_sup['sup_loss1']-log_vars_sup['sup_loss2']>0:
            log_vars['unsup_loss2'] = log_vars['unsup_loss2']*0
        else:
            log_vars['unsup_loss1'] = log_vars['unsup_loss1']*0
        '''
        return log_vars

    def _forward_DML_train(self, pred_1, pred_2):
        loss = {}
        pred_1=F.log_softmax(pred_1, dim=1)
        pred_2=F.log_softmax(pred_2, dim=1)
        label1 = pred_1.detach()
        label2 = pred_2.detach()
        loss['unsup_loss1'] = weighted_loss(self._loss(self.branch1, pred_1, label2, True), self.unsup_weight)
        loss['unsup_loss2'] = weighted_loss(self._loss(self.branch2, pred_2, label1, True), self.unsup_weight)
        return loss

    def _loss(self, branch, pred, label, is_DML_loss = False):
        """
        branch: EncoderDecoder
        """
        if is_DML_loss is False:
            loss = branch.decode_head.loss_decode(pred, label, ignore_index=self.ignore_index)
        else:
            loss = self.criterion(pred, label, ignore_index=self.ignore_index)
        return loss

