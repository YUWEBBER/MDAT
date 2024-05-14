# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
import torch

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .DA import *

@DETECTORS.register_module()
class DAYOLOV3(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DAYOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.dac1 = DAClassifer(in_channels=512)
        self.dac2 = DAClassifer(in_channels=256)
        self.dac3 = DAClassifer(in_channels=128)
        self.daclassifer = [self.dac1, self.dac2, self.dac3]

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):


        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        img_ss = []
        assert 'img_source' in img_metas[0].keys()
        for meta in img_metas:
            img_s = meta['img_source']
            if len(img_s.shape) < 3:
                img_s = np.expand_dims(img_s, -1)
            img_s = np.ascontiguousarray(img_s.transpose(2, 0, 1))
            img_ss.append(to_tensor(img_s).cuda())
        img_ss = torch.stack(img_ss, dim=0)   #img from the source domian, i.e., SAR image
        x_ss = self.extract_feat(img_ss)


        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        assert len(x) == len(x_ss) == len(self.daclassifer)
        da_losses = 0
        bs = x[0].shape[0]
        for x_i, x_ss_i, daclassifer_i in zip(x, x_ss, self.daclassifer):
            da_losses += daclassifer_i([x_i], [x_ss_i[:bs]])['da_loss']
        losses.update({'da_loss': da_losses})
        return losses


    def onnx_export(self, img, img_metas):
        """Test function for exporting to ONNX, without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head.forward(x)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels
