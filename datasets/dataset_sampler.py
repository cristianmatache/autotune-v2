import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


from datasets.image_dataset_loaders import CIFARLoader, SVHNLoader, MRBILoader


DATA_DIR = "D:/datasets/"
PROBLEM = 'mrbi'


def get_train_data(problem, data_dir):
    if problem == "cifar":
        return CIFARLoader(data_dir, mean_normalize=(), std_normalize=(), augment=False).train_data
    elif problem == "svhn":
        return SVHNLoader(data_dir, mean_normalize=(), std_normalize=()).train_data
    elif problem == "mrbi":
        return MRBILoader(data_dir, mean_normalize=(), std_normalize=()).train_data


def main():
    train_data = get_train_data(PROBLEM, DATA_DIR)
    print('Number of samples: ', len(train_data))

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    # functions to show an image
    def imshow(img):
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

    # get some random training images
    images, labels = iter(train_loader).__next__()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join(f'{labels[j]}' for j in range(4)))
    plt.show(block=True)


if __name__ == "__main__":
    main()
