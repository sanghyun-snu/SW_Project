# -*- coding: utf-8 -*-
import shutil
import os
import torch
import cv2
import numpy as np
import torchvision.utils as vutils

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'table', 'dog', 'horse',
            'motorbike', 'person', 'plant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_images(folder_name, epoch, i, blend_tensor, args):
    """ Save Tensor image in the folder. """
    saving_folder = os.path.join(args.log_folder, args.name, folder_name)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    file_name = 'HEAT_TEST_{}_{}.jpg'.format(epoch+1, i)
    saving_path = os.path.join(saving_folder, file_name)
    vutils.save_image(blend_tensor, saving_path, nrow=5)


def print_progress(ap, loss, start_epoch, end_epoch, prefix='Training'):
    print(prefix, '[{0}/{1}]'.format(start_epoch, end_epoch))
    print("Loss: {}".format(loss))
    print("mAP: {}".format(ap.mean()))
    for i in CAT_LIST:
        print("{:>6s} ".format(i[:5]), end='')
    print("")
    for i in ap:
        print("{:>6.2f} ".format(i), end='')
    print("\n")


def save_progress(file_path, train_ap, train_loss, val_ap, val_loss, args):
    with open(os.path.join(file_path, 'result.txt'), 'w') as f:

        print(args, "\n", file=f)

        print("TRAIN_Loss: {}".format(train_loss), file=f)
        print("TRAIN_mAP: {}".format(train_ap.mean()), file=f)
        for i in CAT_LIST:
            print("{:>6s} ".format(i[:5]), end='', file=f)
        print("", file=f)
        for i in train_ap:
            print("{:>6.2f} ".format(i), end='', file=f)
        print("\n", file=f)

        print("VAL_Loss: {}".format(val_loss), file=f)
        print("VAL_mAP: {}".format(val_ap.mean()), file=f)
        for i in CAT_LIST:
            print("{:>6s} ".format(i[:5]), end='', file=f)
        print("", file=f)
        for i in val_ap:
            print("{:>6.2f} ".format(i), end='', file=f)
        print("\n", file=f)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch != 0 and epoch % args.LR_decay == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def load_model(model, optimizer, args):
    """ Loading pretrained / trained model. """
    if os.path.isfile(args.resume):
        if args.gpu == 0:
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        try:
            args.start_epoch = checkpoint['epoch']
        except (TypeError, KeyError) as e:
            print("=> No 'epoch' keyword in checkpoint.")
        try:
            best_acc1 = checkpoint['best_m_ap']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                pass
        except (TypeError, KeyError) as e:
            print("=> No 'best_acc1' keyword in checkpoint.")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except (TypeError, KeyError, ValueError) as e:
            print("=> Fail to load 'optimizer' in checkpoint.")
        try:
            if args.gpu == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        except (TypeError, KeyError) as e:
            if args.gpu == 0:
                print("=> No 'epoch' in checkpoint.")

        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        if args.gpu == 0:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    """ Save checkpoint w.r.t. best cls, best loc and best gt_loc """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best_cls.pth.tar'))


def load_sizes(file_path):
    size_dict = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            size_dict[info[0]] = (int(info[1]), int(info[2]))

    return size_dict


def save_result_image(folder_name, image, image_id, args):
    """ Save Tensor image in the folder. """
    saving_folder = os.path.join(args.log_folder, args.name, folder_name)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    file_name = "{}.png".format(image_id)
    saving_path = os.path.join(saving_folder, file_name)
    cv2.imwrite(saving_path, image)



def draw_bboxes(image, iou, gt_box, pr_box, is_first=True, is_top1=False):

    def draw_bbox(img, box1, box2, color1=(0, 0, 255), color2=(0, 255, 0)):
        cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), color1, 2)
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color2, 2)
        return img
    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()
    if is_first:
        gt_bbox_idx = 0
    else:
        gt_bbox_idx = np.argmax(iou)

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, gt_box[gt_bbox_idx], pr_box)

    # mark the iou
    mark_target(boxed_image, '%.1f' % (iou[gt_bbox_idx] * 100), (150, 30), 2)
    if is_top1:
        mark_target(boxed_image, 'TOP1', pos=(15, 30))

    return boxed_image


def save_each_image(image_list, args, image_id):
    """ Save each image in order of image_list """
    assert len(image_list) > 1

    concat_image = np.concatenate((image_list[0], image_list[1]), axis=1)
    for i in range(2, len(image_list)):
        concat_image = np.concatenate((concat_image, image_list[i]), axis=1)

    base_path = args.log_folder
    if not os.path.isdir(os.path.join(base_path, 'each')):
        os.mkdir(os.path.join(base_path, 'each'))
    save_path = os.path.join(base_path, 'each', str(image_id) + '.jpg')
    cv2.imwrite(save_path, concat_image)
