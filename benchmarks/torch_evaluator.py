from abc import abstractmethod
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch import cuda
from typing import Tuple

from core import Evaluator, OptimisationGoals
from benchmarks.torch_model_builders import ModelBuilder
from datasets.image_dataset_loaders import ImageDatasetLoader


class TorchEvaluator(Evaluator):

    """
    Framework for evaluators based on pytorch models
    """

    def __init__(self, model_builder: ModelBuilder, dataset_loader: ImageDatasetLoader,
                 criterion: Module = torch.nn.CrossEntropyLoss(), output_dir: str = ".", file_name: str = "model.pth"):
        """
        :param model_builder: builder of a machine learning model based on an arm
        :param dataset_loader: dataset loader
        :param criterion: loss function
        :param output_dir: directory where to save the arms and their evaluation progress so far (as checkpoints)
        :param file_name: file (at output_dir/arm<i>/file_name) which stores the progress of the evaluation of an arm
        """
        super().__init__(model_builder, output_dir, file_name)
        self.ml_model, self.optimizer = model_builder.construct_model()
        self.criterion = criterion
        self.dataset_loader = dataset_loader

        # save first checkpoint to file file_path
        self._save_checkpoint(epoch=0, val_error=1, test_error=1)

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        torch.save({
            'epoch': epoch,
            'model': self.ml_model,
            'optimizer': self.optimizer,
            'val_error': val_error,
            'test_error': test_error,
        }, self.file_path)

    def _resume_from_checkpoint(self) -> int:
        """ Load model and optimiser from file to resume training
        :return: start epoch
        """
        checkpoint = torch.load(self.file_path)
        start_epoch = checkpoint['epoch']
        self.ml_model = checkpoint['model']
        self.optimizer = checkpoint['optimizer']
        return start_epoch

    def _train(self, epoch: int, max_batches: int, batch_size: int = 100) -> float:
        """ Train for one epoch
        :param epoch: epoch number
        :param max_batches: maximum number of batches
        :param batch_size: size of batch (in terms of number of examples)
        :return: train loss/error
        """
        loader = self.dataset_loader.train_loader(batch_size=batch_size)

        print('\nEpoch: %d' % epoch)
        self.ml_model.train()

        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            if batch_idx >= max_batches:
                break

            if cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.ml_model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        return train_loss

    def _test(self, is_validation: bool) -> Tuple[float, ...]:
        """
        :param is_validation: whether the function is applied on the validation set or on the test set
        :return: values that will be used in OptimisationGoals (eg. test/validation error, number of true positives)
        """
        loader = self.dataset_loader.val_loader if is_validation else self.dataset_loader.test_loader

        self.ml_model.eval()
        test_loss, correct, total = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            if cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.ml_model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        return 1 - correct / total, correct, total

    @abstractmethod
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        """ Aggregate the steps:
            - train model (available through self._train)
            - evaluate model with respect to the test/validation set(s) (available through self._test)
            - report performance
        :param n_resources: number of resources allocated for training (used by Hyperband methods)
        :return: optimization goals - metrics in terms of which we can perform optimization
                 Eg. validation error, test error
        """
        pass
