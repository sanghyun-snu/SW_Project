import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_cam_all_class(model, target=None):

    with torch.no_grad():

        feature_map, score = model.get_cam()
        fc_weight = model.fc.weight.squeeze()


        batch, channel, h, w = feature_map.size()
        num_classes = fc_weight.size(0)

        if target is None:
            _, target = score.topk(1, 1, True, True)
        target = target.squeeze()


        # feature_map : batch x c x h x w
        # weight : n x c
        # final : batch x n x c x h x w

        cam_weight = fc_weight.view(1, num_classes, channel, 1, 1).expand(batch, num_classes, channel, h, w)
        feature_map = feature_map.unsqueeze(1).expand_as(cam_weight)
        cam = cam_weight * feature_map
        cam = cam.mean(dim=2)
    return cam


def get_cam_target_class(model, target=None):

    with torch.no_grad():

        feature_map, score = model.get_cam()
        fc_weight = model.fc.weight.squeeze()


        batch, channel, h, w = feature_map.size()
        num_classes = fc_weight.size(0)

        if target is None:
            _, target = score.topk(2, 1, True, True)
        target = target.squeeze()

        # feature_map : batch x c x h x w
        # weight : n x c
        # final : batch x n x c x h x w

        cam_weight = fc_weight.view(1, num_classes, channel, 1, 1).expand(batch, num_classes, channel, h, w)
        feature_map = feature_map.unsqueeze(1).expand_as(cam_weight)
        cam = cam_weight * feature_map
        # NCHW
        cam = cam.mean(dim=2)
        target = target
        final_cam = torch.zeros(batch, num_classes+1, h, w)

        for i, j in enumerate(target):
            for k in j:
                final_cam[i, k.item()+1] = cam[i, k.item()]

    return final_cam


def get_cam_highest_pred(model, target=None):

    with torch.no_grad():
        feature_map, score = model.get_cam()
        fc_weight = model.fc.weight.squeeze()

        batch, channel, h, w = feature_map.size()
        num_classes = fc_weight.size(0)

        if target is None:
            _, target = score.topk(1, 1, True, True)
        target = target.squeeze()

        # feature_map : batch x c x h x w
        # weight : n x c
        # final batch x 1 x h x w
        cam_weight = fc_weight[target]

        cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(feature_map)
        cam = cam_weight * feature_map
        cam = cam.mean(1, keepdim=True)

    return cam

def resize_cam(cam, size=(321, 321)):
    cam = cv2.resize(cam, size, interpolation=cv2.INTER_CUBIC)
    cam -= cam.min()
    cam /= cam.max()
    return cam

def resize_threshold_cam(cam, size=(321, 321), thresh = 0.3):
    cam = cv2.resize(cam, size, interpolation=cv2.INTER_CUBIC)
    cam -= cam.min()
    cam /= cam.max()
    cam[cam < thresh] = 0
    return cam


def blend_cam(image, cam):
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.5 + heatmap * 0.5

    return blend, heatmap

def mark_target(img, text='target', pos=(25, 25), size=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
    return img
