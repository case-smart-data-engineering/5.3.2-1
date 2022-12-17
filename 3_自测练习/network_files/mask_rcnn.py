from collections import OrderedDict
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign

from .faster_rcnn_framework import FasterRCNN


class MaskRCNN(FasterRCNN):
    """
        Implements Mask R-CNN.

        The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
        image, and should be in 0-1 range. Different images can have different sizes.

        The behavior of the model changes depending if it is in training or evaluation mode.

        During training, the model expects both the input tensors, as well as a targets (list of dictionary),
        containing:
            - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
              ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (Int64Tensor[N]): the class label for each ground-truth box
            - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

        The model returns a Dict[Tensor] during training, containing the classification and regression
        losses for both the RPN and the R-CNN, and the mask loss.

        During inference, the model requires only the input tensors, and returns the post-processed
        predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
        follows:
            - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
              ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (Int64Tensor[N]): the predicted labels for each image
            - scores (Tensor[N]): the scores or each prediction
            - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
              obtain the final segmentation masks, the soft masks can be thresholded, generally
              with a value of 0.5 (mask >= 0.5)

        Args:
            backbone (nn.Module): 用于计算模型特征的网络。它应该包含out_channels属性，表示输出的数量。
            每个特征图具有的通道(对于所有特征图应该是相同的)。骨干应该返回单个Tensor 或是 OrderedDict[Tensor]。
            num_classes (int): 模型输出类别数(包括背景)。
            如果 box_predictor 是制定的, num_classes 应当为 None。
            min_size (int): 在将图像输入主干之前重新缩放的图像的最小尺寸。
            max_size (int): 在将图像输入主干之前重新缩放的图像的最大尺寸。
            image_mean (Tuple[float, float, float]): 用于输入规范化的平均值。它们通常是骨干训练数据集的平均值
            image_std (Tuple[float, float, float]):  用于输入规范化的std值。
                它们通常是骨干网络训练时使用的数据集的std值。
            rpn_anchor_generator (AnchorGenerator): 为一组特征图生成锚点的模块。
            rpn_head (nn.Module): 从RPN中计算对象性和回归delta的模块。
            rpn_pre_nms_top_n_train (int): 在训练期间应用NMS之前要保留的建议数量。
            rpn_pre_nms_top_n_test (int): 在测试期间应用NMS之前要保留的建议数。
            rpn_post_nms_top_n_train (int): 在训练期间应用NMS后保留的建议数量。
            rpn_post_nms_top_n_test (int): 在测试过程中应用NMS后要保留的建议数。
            rpn_nms_thresh (float): 用于RPN提案的后处理的NMS阈值。
            rpn_fg_iou_thresh (float): 锚点和GT盒之间的最小IoU，以便在RPN训练期间将它们视为正的。
            rpn_bg_iou_thresh (float): 锚点和GT盒之间的最大IoU，以便在RPN训练期间将它们视为负的。
            rpn_batch_size_per_image (int): 在RPN训练期间为计算损失而采样的锚点数量。
            rpn_positive_fraction (float): RPN训练期间小批量中正锚点的比例。
            rpn_score_thresh (float): 在推理过程中，只返回分类分数大于rpn_score_thresh的建议框。
            box_roi_pool (MultiScaleRoIAlign): 在边界框表示的位置上裁剪和调整特征图大小的模块。
            box_head (nn.Module): 将裁剪后的特征图作为输入的模块。
            box_predictor (nn.Module): 接收box_head的输出并返回分类logits和box回归增量的模块。
            box_score_thresh (float): 在推理过程中，只返回分类分数大于box_score_thresh的建议框。
            box_nms_thresh (float): 推理时使用的预测头的NMS阈值。
            box_detections_per_img (int): 对于所有类别，每个图像的最大检测数。
            box_fg_iou_thresh (float): 建议和GT盒之间的最小IoU，以便在分类头训练期间将它们视为正的。
            box_bg_iou_thresh (float): 建议和GT盒之间的最大IoU，以便在分类头训练期间将它们视为负的。
            box_batch_size_per_image (int): 在分类头训练期间采样的建议数。
            box_positive_fraction (float): 在分类头训练期间，小批量积极建议的比例。
            bbox_reg_weights (Tuple[float, float, float, float]): 边界框编码/解码的权重。
            mask_roi_pool (MultiScaleRoIAlign):  在边界框指示的位置上裁剪和调整特征图的大小的模块，并将结果提供给mask_head模块。
            mask_head (nn.Module): 将裁剪后的特征图作为输入的模块。
            mask_predictor (nn.Module): 接收mask_head的输出并返回分段掩码logits的模块。

        """

    def __init__(
            self,
            backbone,
            num_classes=None,
            # transform parameters
            min_size=800,
            max_size=1333,
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # Mask parameters
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,
    ):

        if not isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(mask_roi_pool)}"
            )

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


class MaskRCNNHeads(nn.Sequential):
    #请完成MaskRCNNHeads类
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): 输入通道数
            layers (tuple): 每个FCN层的特征尺寸
            dilation (int): 核的膨胀率
        """

        # 初始参数
        


class MaskRCNNPredictor(nn.Sequential):
    #请完成MaskRCNNPredictor类，
    def __init__(self, in_channels, dim_reduced, num_classes):
        
        # 初始参数
        
