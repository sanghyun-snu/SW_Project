# -*- coding: utf-8 -*-
import os
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import architecture as models
from utils.util import \
    print_progress, AverageMeter, \
    save_checkpoint, load_model, \
    adjust_learning_rate, save_progress, \
    save_images, CAT_LIST, load_sizes, save_result_image
from utils.util_args import get_args
from utils.util_loader import data_loader
from utils.util_acc import APMeter
from utils.util_loc import \
    get_cam_all_class, get_cam_highest_pred, get_cam_target_class, \
    resize_cam, blend_cam, mark_target, resize_threshold_cam

torch.set_num_threads(4)
np.set_printoptions(threshold=sys.maxsize)

def main():
    args = get_args()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # number of classes for each dataset.
    if args.dataset == 'PascalVOC':
        num_classes = 20
    else:
        raise Exception("No dataset named {}.".format(args.dataset))

    # Select Model & Method
    model = models.__dict__[args.arch](pretrained=args.pretrained,
                                       num_classes=num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    # criterion = nn.MultiLabelSoftMarginLoss().cuda(args.gpu)
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    # Take apart parameters to give different Learning Rate
    param_features = []
    param_classifiers = []

    if args.arch.startswith('vgg'):
        for name, parameter in model.named_parameters():
            if 'features.' in name:
                param_features.append(parameter)
            else:
                param_classifiers.append(parameter)
    elif args.arch.startswith('resnet'):
        for name, parameter in model.named_parameters():
            if 'layer4.' in name or 'fc.' in name:
                param_classifiers.append(parameter)
            else:
                param_features.append(parameter)
    else:
        raise Exception("Fail to recognize the architecture")

    # Optimizer
    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifiers, 'lr': args.lr * args.lr_ratio}],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nest)

    # optionally resume from a checkpoint
    if args.resume:
        model, optimizer = load_model(model, optimizer, args)
    train_loader, val_loader, test_loader = data_loader(args)

    saving_dir = os.path.join(args.log_folder, args.name)

    if args.evaluate:
        # test_ap, test_loss = evaluate_cam(val_loader, model, criterion, args)
        # test_ap, test_loss = evaluate_cam2(val_loader, model, criterion, args)
        test_ap, test_loss = evaluate_cam3(val_loader, model, criterion, args)
        print_progress(test_ap, test_loss, 0, 0, prefix='test')
        return

    # Training Phase
    best_m_ap = 0
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train_ap, train_loss = \
            train(train_loader, model, criterion, optimizer, epoch, args)
        print_progress(train_ap, train_loss, epoch+1, args.epochs)

        # Evaluate classification
        val_ap, val_loss = validate(val_loader, model, criterion, epoch, args)
        print_progress(val_ap, val_loss, epoch+1, args.epochs, prefix='validation')

        # # Save checkpoint at best performance:
        is_best = val_ap.mean() > best_m_ap
        if is_best:
            best_m_ap = val_ap.mean()

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_m_ap': best_m_ap,
            'optimizer': optimizer.state_dict(),
        }, is_best, saving_dir)

        save_progress(saving_dir, train_ap, train_loss, val_ap, val_loss, args)

def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    ap = APMeter()

    # switch to train mode
    model.train()
    for i, (images, target) in enumerate(tqdm(train_loader, desc='Train')):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        ap.add(output.detach(), target)
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    return ap.value(), losses.avg


def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    ap = APMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, target, image_id) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ap.add(output.detach(), target)
            losses.update(loss.item(), images.size(0))
    return ap.value(), losses.avg


def evaluate_cam(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    ap = APMeter()

    # switch to evaluate mode
    model.eval()

    # Image de-standardization
    image_mean_value = [0.485, .456, .406]
    image_std_value = [.229, .224, .225]
    image_mean_value = torch.reshape(torch.tensor(image_mean_value), (1, 3, 1, 1))
    image_std_value = torch.reshape(torch.tensor(image_std_value), (1, 3, 1, 1))

    with torch.no_grad():
        for i, (images, target, image_id) in enumerate(tqdm(val_loader, desc='Evaluate')):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ap.add(output.detach(), target)
            losses.update(loss.item(), images.size(0))

            # image de-normalizing
            images = images.clone().detach().cpu() * image_std_value + image_mean_value
            images = images.numpy().transpose(0, 2, 3, 1) * 255.
            images = images[:, :, :, ::-1]

            # extract CAM
            cam = get_cam_all_class(model, target)
            cam = cam.cpu().numpy().transpose(0, 2, 3, 1)

            # for all class
            for j in range(cam.shape[0]):
                blend_tensor = torch.empty((cam.shape[3], 3, 321, 321))
                for k in range(cam.shape[3]):
                    cam_ = resize_cam(cam[j, :, :, k])
                    blend, heatmap = blend_cam(images[j], cam_)
                    if target[j, k]:
                        blend = mark_target(blend, text=CAT_LIST[k])
                    blend = blend[:, :, ::-1] / 255.
                    blend = blend.transpose(2, 0, 1)
                    blend_tensor[k] = torch.tensor(blend)
                save_images('result', i, j, blend_tensor, args)

    return ap.value(), losses.avg


def evaluate_cam2(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    ap = APMeter()

    # switch to evaluate mode
    model.eval()

    # Image de-standardization
    image_mean_value = [0.485, .456, .406]
    image_std_value = [.229, .224, .225]
    image_mean_value = torch.reshape(torch.tensor(image_mean_value), (1, 3, 1, 1))
    image_std_value = torch.reshape(torch.tensor(image_std_value), (1, 3, 1, 1))

    with torch.no_grad():
        for i, (images, target, image_id) in enumerate(tqdm(val_loader, desc='Evaluate')):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ap.add(output.detach(), target)
            losses.update(loss.item(), images.size(0))

            # image de-normalizing
            images = images.clone().detach().cpu() * image_std_value + image_mean_value
            images = images.numpy().transpose(0, 2, 3, 1) * 255.
            images = images[:, :, :, ::-1]

            # extract CAM
            cam = get_cam_highest_pred(model)
            cam = cam.cpu().numpy().transpose(0, 2, 3, 1)

            # for all class
            blend_tensor = torch.empty((images.shape[0], 3, 321, 321))
            for j in range(cam.shape[0]):
                cam_ = resize_cam(cam[j], size=(321,321))
                blend, heatmap = blend_cam(images[j], cam_)
                blend = blend[:, :, ::-1] / 255.
                blend = blend.transpose(2, 0, 1)
                blend_tensor[j] = torch.tensor(blend)
            save_images('result', i, j, blend_tensor, args)

    return ap.value(), losses.avg


def evaluate_cam3(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    ap = APMeter()

    size_dict = load_sizes(args.size_list)

    # switch to evaluate mode
    model.eval()

    # Image de-standardization
    image_mean_value = [0.485, .456, .406]
    image_std_value = [.229, .224, .225]
    image_mean_value = torch.reshape(torch.tensor(image_mean_value), (1, 3, 1, 1))
    image_std_value = torch.reshape(torch.tensor(image_std_value), (1, 3, 1, 1))

    with torch.no_grad():
        for i, (images, target, image_id) in enumerate(tqdm(val_loader, desc='Evaluate')):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ap.add(output.detach(), target)
            losses.update(loss.item(), images.size(0))

            # image de-normalizing
            images = images.clone().detach().cpu() * image_std_value + image_mean_value
            images = images.numpy().transpose(0, 2, 3, 1) * 255.
            images = images[:, :, :, ::-1]

            # extract CAM
            cam = get_cam_target_class(model)
            cam = cam.cpu().numpy().transpose(0, 2, 3, 1)

            for j in range(cam.shape[0]):
                cam_ = resize_threshold_cam(cam[j],
                                            size=(size_dict[image_id[j]][0], size_dict[image_id[j]][1]),
                                            thresh=0.3)
                cam_max = np.argmax(cam_, axis=2)

                save_result_image('final_map', cam_max, image_id[j], args)


    return ap.value(), losses.avg


if __name__ == '__main__':
    main()













