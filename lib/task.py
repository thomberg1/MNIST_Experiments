import os

from lib.classifier import *
from lib.utils import *

import argparse

#######################################################################################################################
from torchvision import  transforms

def load_data(H):

    path = os.path.join(H.EXPERIMENT.strip(), H.ROOT_DIR.strip() )

    transform = []
    if 'AUGMENTATION' in H and H.AUGMENTATION is not None:
        transform += [H.AUGMENTATION()]
    transform += [transforms.ToTensor()]

    image_transform_train = transforms.Compose(transform)

    image_transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = MNIST(path, dataset="train", download=True, transform=image_transform_train)
    target_names = ['0','1','2','3','4','5','6','7','8','9']

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS,
                                                shuffle=True, pin_memory=True)

    valid_dataset = MNIST(path, dataset="valid", download=False, transform=image_transform_test)

    valid_loader = torch.utils.data.DataLoader( valid_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS,
                                                shuffle=False, pin_memory=True)

    test_dataset = MNIST(path, dataset="test", download=False, transform=image_transform_test)

    test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS,
                                               shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader, target_names

#######################################################################################################################

def task(experiment_name, timestamp):

    print(experiment_name, timestamp)

    H = HYPERPARAMETERS.load(os.path.join(experiment_name.strip(), "HP_" + timestamp.strip() + ".pkl" ))

    torch.backends.cudnn.deterministic = True

    np.random.seed(H.SEED)
    torch.manual_seed(H.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(H.SEED)

    train_loader, valid_loader, _, _ = load_data(H)

    clf = Classifier(H, train_loader, valid_loader)

    clf()

    return timestamp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-e','--experiment_name', help='experiment_name', required=True)
    parser.add_argument('-t','--timestamp', help='timestamp', required=True)
    args = vars(parser.parse_args())

    res = task(args['experiment_name'], args['timestamp'])

    print(res)
