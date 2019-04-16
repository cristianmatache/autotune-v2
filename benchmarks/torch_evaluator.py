from abc import abstractmethod
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch import cuda
from typing import Tuple

from core.evaluator import Evaluator
from core.optimization_goals import OptimizationGoals
from benchmarks.torch_model_builders import ModelBuilder
from datasets.image_dataset_loaders import ImageDatasetLoader


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

    def _test(self, is_validation: bool) -> Tuple[float, ...]:
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
    def evaluate(self, n_resources: int) -> OptimizationGoals:
        pass
