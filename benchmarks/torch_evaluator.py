from abc import abstractmethod
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch import cuda
from torch.utils.data import DataLoader
from typing import Tuple

from core.evaluator import Evaluator
from benchmarks.model_builders import ModelBuilder
from util.progress_bar import progress_bar
from datasets.image_dataset_loaders import ImageDatasetLoader


def print_accuracy(train_val_or_test: str, correct: int, total: int) -> None:
    accuracy = 100. * correct / total
    padding = ' ' * (len("validation") - len(train_val_or_test))
    print(f"{train_val_or_test} accuracy:{padding} {accuracy:.3f}% ({correct}/{total})")


class TorchEvaluator(Evaluator):

    def __init__(self, model_builder: ModelBuilder, dataset_loader: ImageDatasetLoader,
                 criterion: Module = torch.nn.CrossEntropyLoss(), output_dir: str = ".", file_name: str = "model.pth"):
        super().__init__(model_builder, criterion, output_dir, file_name)
        self.dataset_loader = dataset_loader

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        torch.save({
            'epoch': epoch,
            'model': self.ml_model,
            'optimizer': self.optimizer,
            'val_error': val_error,
            'test_error': test_error,
        }, self.file_path)

    def resume_from_checkpoint(self) -> int:
        # Load model and optimiser from file to resume training
        checkpoint = torch.load(self.file_path)
        start_epoch = checkpoint['epoch']
        self.ml_model = checkpoint['model']
        self.optimizer = checkpoint['optimizer']
        return start_epoch

    def _train(self, epoch: int, max_batches: int, batch_size: int = 100) -> float:
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

    def _test(self, is_validation: bool) -> float:
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

        print_accuracy("Validation" if is_validation else "Test", correct, total)
        return 1 - correct / total

    @abstractmethod
    def evaluate(self, n_resources: int) -> Tuple[float, float]:
        pass

    @staticmethod
    def _display_progress_bar(batch_idx: int, loader: DataLoader, correct: int, total: int, total_loss: int,
                              disp_interval: int, train_val_or_test: str = None) -> None:
        if train_val_or_test not in ["Train", "Validation", "Test", None]:
            raise ValueError('train_val_or_test must be "Train", "Validation" or "Test"')

        if batch_idx % disp_interval == 0 or batch_idx == len(loader):
            progress_bar(batch_idx, len(loader), f'Loss: %.3f | {train_val_or_test} Acc: %.3f%% (%d/%d)'
                         % (total_loss / batch_idx, 100. * correct / total, correct, total))
