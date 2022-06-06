from .. import build_segmentor, SEGMENTORS, build_loss
from .multi_stream_segmentor import MultiStreamSegmentor
from mmseg.utils import dict_split, weighted_loss
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import torch
import copy

@SEGMENTORS.register_module()
class DML_triple(MultiStreamSegmentor):
    def __init__(self, model, train_cfg=None, test_cfg=None):
        super(DML_triple, self).__init__(
            dict(branch1=build_segmentor(model), branch2=build_segmentor(model), branch3=build_segmentor(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        if train_cfg is not None:
            self.unsup_weight = self.train_cfg.unsup_weight
            self.t = getattr(self.train_cfg, 't', 1)
            self.score_merge_mode = getattr(self.train_cfg, 'score_merge_mode', None)
        if self.score_merge_mode == 'whole':
            self.score_merge_layer = ConvModule(in_channels=model.decode_head.num_classes * 3,
                                                out_channels=model.decode_head.num_classes,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                conv_cfg=None,
                                                norm_cfg=model.decode_head.norm_cfg,
                                                )
        if self.score_merge_mode == 'class-wise':
            raise NotImplementedError()
            
        assert self.branch1.decode_head.ignore_index == self.branch2.decode_head.ignore_index == self.branch3.decode_head.ignore_index\
            , "'ignore_index' of three branches must be the same."
        self.ignore_index = self.branch1.decode_head.ignore_index
        # Register a new loss module for calculating CPS loss
        self.criterion = build_loss(dict(
            type='KLDivergence', loss_weight=1.0))
        self.merge_criterion = build_loss(dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0))

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
        pred_3 = self.branch3.encode_decode(img, img_metas)

        # 1. calculate CPS loss for all data
        log_vars_cps = self._forward_DML_train(pred_1, pred_2, pred_3, gt_semantic_seg, self.score_merge_mode)
        log_vars.update(log_vars_cps)

        # 2. calculate supervised loss (cross-entropy) for labeled data
        log_vars_sup = dict()
        log_vars_sup['sup_loss1'] = self._loss(self.branch1, pred_1, gt_semantic_seg.squeeze(1))#gt_seg.squeeze(1)
        log_vars_sup['sup_loss2'] = self._loss(self.branch2, pred_2, gt_semantic_seg.squeeze(1))
        log_vars_sup['sup_loss3'] = self._loss(self.branch3, pred_3, gt_semantic_seg.squeeze(1))
        log_vars.update(log_vars_sup)
        log_vars['unsup_weight'] =self.unsup_weight
        '''
        if log_vars_sup['sup_loss1']-log_vars_sup['sup_loss2']>0:
            log_vars['unsup_loss2'] = log_vars['unsup_loss2']*0
        else:
            log_vars['unsup_loss1'] = log_vars['unsup_loss1']*0
        '''
        return log_vars

    def _forward_DML_train(self, pred_1, pred_2, pred_3, gt_semantic_seg, score_merge_mode = None):
        loss = {}
        if score_merge_mode == None:
            pred = []
            pred.append(F.log_softmax(pred_1/self.t, dim=1))
            pred.append(F.log_softmax(pred_2/self.t, dim=1))
            pred.append(F.log_softmax(pred_3/self.t, dim=1))
            label = []
            for p in pred:
                label.append(p.detach())
            #TODO: weighted KL loss
            for i in range(3):
                next_index = (
                    (i+1)%3,
                    (i+2)%3
                )
                _branch_name = self.branches[i]
                #'unsup_loss'+'1', +'2', +'3'
                loss[('unsup_loss'+str(i+1))] = (self._loss(self.model(branch=_branch_name), pred[i], label[next_index[0]], True) + \
                                              self._loss(self.model(branch=_branch_name), pred[i], label[next_index[1]], True))/2.0
                loss[('unsup_loss' + str(i + 1))] = weighted_loss(loss[('unsup_loss'+str(i+1))], self.unsup_weight)
            return loss
        elif score_merge_mode == 'whole':
            merge_layer_input = torch.cat((pred_1, pred_2, pred_3), dim=1)
            merge_layer_output = self.score_merge_layer(merge_layer_input.detach())
            loss['merge_layer_loss'] = self.merge_criterion(merge_layer_output, gt_semantic_seg.squeeze(1), ignore_index=self.ignore_index)
            #TODO: Finish KL loss computation
            label_merged = F.log_softmax(merge_layer_output.detach()/self.t, dim=1)
            pred = []
            pred.append(F.log_softmax(pred_1 / self.t, dim=1))
            pred.append(F.log_softmax(pred_2 / self.t, dim=1))
            pred.append(F.log_softmax(pred_3 / self.t, dim=1))
            for i in range(3):
                _branch_name = self.branches[i]
                loss[('unsup_loss' + str(i + 1))] = self._loss(self.model(branch=_branch_name), pred[i],
                                                                label_merged, True)
                loss[('unsup_loss' + str(i + 1))] = weighted_loss(loss[('unsup_loss' + str(i + 1))], self.unsup_weight)
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

