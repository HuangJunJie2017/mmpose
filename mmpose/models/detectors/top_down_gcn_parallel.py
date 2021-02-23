import math

import cv2
import mmcv
import warnings
import numpy as np

import torch
import torch.nn as nn
from mmcv.image import imwrite
from mmcv.visualization.image import imshow

from mmpose.core.evaluation import pose_pck_accuracy, _get_max_preds, transform_preds, keypoint_pck_accuracy
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.core.post_processing import flip_back
from .. import builder
from ..registry import POSENETS
from .base import BasePose


@POSENETS.register_module()
class TopDownGCNParallel(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 gcn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None,
                 extra=None):
        super().__init__()

        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            self.keypoint_head = builder.build_head(keypoint_head)
        self.gcn = builder.build_gcn(gcn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss = builder.build_loss(loss_pose)
        loss_hm = loss_pose.copy()
        loss_hm.pop('crit_type')
        self.loss_hm = builder.build_loss(loss_hm)
        self.init_weights(pretrained=pretrained)
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')
        self.extra = extra

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths
                  and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        if self.extra.get('only_inference', True):
            self.backbone.eval()
            self.keypoint_head.eval()
            with torch.no_grad():
                output = self.backbone(img)
                if self.with_keypoint:
                    '''losses, final_heatmap, [[N, 256, 16, 12] [N, 256, 32, 24] [N, 256, 64, 48]]'''
                    output, cfa_out = self.keypoint_head(output)
                if isinstance(output, list):
                    output_heatmap = output[-1].detach().cpu().numpy()
                else:
                    output_heatmap = output.detach().cpu().numpy()
        else:
            output = self.backbone(img)
            if self.with_keypoint:
                '''losses, final_heatmap, [[N, 256, 16, 12] [N, 256, 32, 24] [N, 256, 64, 48]]'''
                output, cfa_out = self.keypoint_head(output)
                if isinstance(output, list):
                    output_heatmap = output[-1].detach().cpu().numpy()
                else:
                    output_heatmap = output.detach().cpu().numpy()

        # if return loss
        self.losses = dict()

        if self.target_type == 'GaussianHeatMap':
            N, K, H, W = output_heatmap.shape
            _, avg_acc, _, gt, pred = pose_pck_accuracy(
                output_heatmap,
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0, 
                return_gts_preds=True)

        # whether log backbone acc_pose 
        if self.extra.get('backbone_acc', False):
            self.losses['acc_pose'] = float(avg_acc)

        multi_hms, multi_poses = self.gcn(cfa_out)

        # calculate pck acc
        normalize = np.tile(np.array([[H, W]]), (N, 1))
        _, integral_acc, _ = keypoint_pck_accuracy(
            multi_poses[-1].detach().cpu().numpy(), 
            gt, 
            target_weight.detach().cpu().numpy().squeeze(-1) > 0, 
            thr=0.05, normalize=normalize)
        self.losses['acc_gcn'] = float(integral_acc)

        # calculate pose loss
        coord_target = torch.from_numpy(gt).cuda()
        self.calcu_loss(multi_poses, coord_target, target_weight, key='pose_loss')

        # calculate hm loss
        self.calcu_loss(multi_hms, target, target_weight, crit=self.loss_hm, key='hm_loss', use_loss_weight=True)

        return self.losses

    def calcu_loss(self, output, target, target_weight, crit=None, key='mse_loss', use_loss_weight=True):
        if crit is None:
            crit = self.loss
        if isinstance(output, list):
            if target.dim() == 5 and target_weight.dim() == 4:
                # target: [batch_size, num_outputs, num_joints, h, w]
                # target_weight: [batch_size, num_outputs, num_joints, 1]
                assert target.size(1) == len(output)
            if isinstance(crit, nn.Sequential):
                assert len(crit) == len(output)
            if 'loss_weights' in self.train_cfg and self.train_cfg[
                    'loss_weights'] is not None:
                assert len(self.train_cfg['loss_weights']) == len(output)
            for i in range(len(output)):
                if target.dim() == 5 and target_weight.dim() == 4:
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                else:
                    target_i = target
                    target_weight_i = target_weight
                if isinstance(crit, nn.Sequential):
                    loss_func = crit[i]
                else:
                    loss_func = crit

                loss_i = loss_func(output[i], target_i, target_weight_i)
                if 'loss_weights' in self.train_cfg and self.train_cfg[
                        'loss_weights'] and use_loss_weight:
                    loss_i = loss_i * self.train_cfg['loss_weights'][i]
                if key not in self.losses:
                    self.losses[key] = loss_i
                else:
                    self.losses[key] += loss_i
        else:
            assert not isinstance(crit, nn.Sequential)
            assert target.dim() == 4 and target_weight.dim() == 3
            # target: [batch_size, num_joints, h, w]
            # target_weight: [batch_size, num_joints, 1]
            self.losses[key] = crit(output, target, target_weight)

    def train_flip(self, img, output, flip_pairs):
        img_flipped = img.flip(3)
        output_flipped = self.backbone(img_flipped)
        if self.with_keypoint:
            output_flipped, _ = self.keypoint_head(output_flipped)
        
        if isinstance(output_flipped, list):
            output_flipped = output_flipped[-1]
        if isinstance(output, list):
            output = output[-1]
        
        output_heatmap = output.detach().cpu().numpy()
        output_flipped_heatmap = output_flipped.detach().cpu().numpy()
        output_flipped_heatmap = flip_back(
            output_flipped_heatmap, 
            flip_pairs, 
            target_type=self.target_type
        )

        if self.test_cfg['shift_heatmap']:
            output_flipped_heatmap[:, :, :, 1:] = output_flipped_heatmap[:, :, :, :-1]
        output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        return output_heatmap

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        # img_metas list[dict]
        assert img.size(0) == len(img_metas)

        # compute backbone features
        output = self.backbone(img)

        # process head
        result = self.process_head(
            output, img, img_metas, return_heatmap=return_heatmap)

        return result

    # TODO
    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def process_head(self, output, img, img_metas, return_heatmap=False):
        """Process heatmap and keypoints from backbone features."""

        num_images = len(img_metas)
        flip_pairs = img_metas[0]['flip_pairs']

        if self.with_keypoint:
            '''losses, final_heatmap, [[N, 256, 16, 12] [N, 256, 32, 24] [N, 256, 64, 48]]'''
            output, cfa_out = self.keypoint_head(output)

        # TODO flip_test not work for now
        backbone_test = self.extra.get('backbone_test', False)
        if self.test_cfg['flip_test']:
            output_heatmap = self.train_flip(img, output, flip_pairs=flip_pairs)
        else:
            if isinstance(output, list):
                output = output[-1]
            output_heatmap = output.detach().cpu().numpy()
        N, K, H, W = output_heatmap.shape
        pred, maxvals = _get_max_preds(output_heatmap)
        multi_hms, multi_poses = self.gcn(cfa_out)
        preds = multi_poses[-1].detach().cpu().numpy()

        c_list = [item['center'].reshape(1, -1) for item in img_metas]
        s_list = [item['scale'].reshape(1, -1) for item in img_metas]
        c = np.concatenate(c_list)
        s = np.concatenate(s_list)

        if 'bbox_score' in img_metas[0]:
            score = [np.array(item['bbox_score']).reshape(-1) for item in img_metas]
        else:
            score = np.ones(num_images)

        if backbone_test:
            preds, maxvals = keypoints_from_heatmaps(
                output_heatmap,
                c,
                s,
                post_process=self.test_cfg['post_process'],
                unbiased=self.test_cfg.get('unbiased_decoding', False),
                kernel=self.test_cfg['modulate_kernel'],
                use_udp=self.test_cfg.get('use_udp', False),
                valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                    0.0546875),
                target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))
        else:
            # Transform back to the image
            for i in range(N):
                preds[i] = transform_preds(
                    preds[i], c[i], s[i], [W, H], use_udp=False)

        results = []
        for i in range(num_images):
            all_preds = np.zeros((1, preds.shape[1], 3), dtype=np.float32)
            all_boxes = np.zeros((1, 6), dtype=np.float32)
            image_path = []

            all_preds[0, :, 0:2] = preds[i, :, 0:2]
            all_preds[0, :, 2:3] = maxvals[i]
            all_boxes[0, 0:2] = c[i, 0:2]
            all_boxes[0, 2:4] = s[i, 0:2]
            all_boxes[0, 4] = np.prod(s[i][np.newaxis, ...] * 200.0, axis=1)
            all_boxes[0, 5] = score[i]
            image_path.extend(img_metas[i]['image_file'])

            if not return_heatmap:
                output_heatmap = None

            results.append([all_preds, all_boxes, image_path, output_heatmap])

        return results

    def flip_back_coords(self, coords_flipped, flip_pairs, W):
        assert coords_flipped.ndim == 3, \
            'coords_flipped should be [batch_size, num_keypoints, 2]'

        coords_flipped_back = coords_flipped.copy()
        # Swap left-right parts
        for left, right in flip_pairs:
            coords_flipped_back[:, left, ...] = coords_flipped[:, right, ...]
            coords_flipped_back[:, right, ...] = coords_flipped[:, left, ...]
        # Flip horizontally
        coords_flipped_back[:, :, 0] = coords_flipped_back[:, :, 0] * -1 + W - 1

        return coords_flipped_back

    def flip_back_cfa_out(self, cfa_out):
        """cfa_out_flipped (List): [[N, C, H, W]]"""
        cfa_out_flip_back = []
        for item in cfa_out:
            item_flipback = item.detach().cpu().numpy()
            item_flipback = item_flipback[..., ::-1].copy()
            cfa_out_flip_back.append(torch.from_numpy(item_flipback).cuda(device=item.device))

        return cfa_out_flip_back

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for _, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                    (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
