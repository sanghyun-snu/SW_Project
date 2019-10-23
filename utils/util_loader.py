from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset.voc import VOCDataset

IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


def data_loader(args):

    data_path = args.data

    # transforms for train dataset
    transforms_train = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE)
    ])

    # transforms for validation dataset
    transforms_val = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE),
    ])

    # CUB-200-2011
    if args.dataset == 'PascalVOC':
        img_train = VOCDataset(
            root=data_path,
            datalist=args.train_list,
            transform=transforms_train,
            mode='train'
        )
        img_val = VOCDataset(
            root=data_path,
            datalist=args.val_list,
            transform=transforms_val,
            mode='val'
        )
        img_test = VOCDataset(
            root=data_path,
            datalist=args.test_list,
            transform=transforms_val,
            mode='test'
        )

    else:
        raise Exception("No Dataset for {}".format(args.dataset))

    # DataLoader
    train_loader = DataLoader(
        img_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    val_loader = DataLoader(
        img_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    test_loader = DataLoader(
        img_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    return train_loader, val_loader, test_loader
