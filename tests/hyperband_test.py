import tempfile
from typing import Tuple, Any, Optional

import numpy as np

from autotune.benchmarks.torch_model_builders import LogisticRegressionBuilder
from autotune.core import Param, Arm, HyperparameterOptimisationProblem, OptimisationGoals, Evaluator, Domain, Optimiser
from autotune.optimisers.sequential.hyperband_optimiser import HyperbandOptimiser
from autotune.util.io import print_evaluation

ETA = 3
MAX_ITER = 243

LEARNING_RATE = Param('learning_rate', np.log(10 ** -6), np.log(10 ** 0), distrib='uniform', scale='log')
WEIGHT_DECAY = Param('weight_decay', np.log(10 ** -6), np.log(10 ** -1), distrib='uniform', scale='log')
MOMENTUM = Param('momentum', 0.3, 0.999, distrib='uniform', scale='linear')
BATCH_SIZE = Param('batch_size', 20, 2000, distrib='uniform', scale='linear', interval=1)
HYPERPARAMS_DOMAIN = Domain(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    momentum=MOMENTUM,
    batch_size=BATCH_SIZE)


class HyperbandTestEvaluator(Evaluator):

    @print_evaluation(verbose=False, goals_to_print=("test_correct",))
    def evaluate(self, n_resources: int) -> OptimisationGoals:
        return OptimisationGoals(validation_error=1, test_error=1)

    def _train(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _test(self, *args: Any, **kwargs: Any) -> Tuple[float, ...]:
        pass

    def _save_checkpoint(self, epoch: int, val_error: float, test_error: float) -> None:
        pass


class HyperbandTestProblem(HyperparameterOptimisationProblem):

    def __init__(self, output_dir: str,
                 hyperparams_domain: Domain = HYPERPARAMS_DOMAIN, hyperparams_to_opt: Tuple[str, ...] = ()):
        super().__init__(hyperparams_domain, hyperparams_to_opt)
        self.output_dir = output_dir

    def get_evaluator(self, arm: Optional[Arm] = None) -> HyperbandTestEvaluator:
        if arm is None:  # if no arm is provided, generate a random arm
            arm = Arm()
            arm.draw_hp_val(domain=self.domain, hyperparams_to_opt=self.hyperparams_to_opt)
        model_builder = LogisticRegressionBuilder(arm)
        return HyperbandTestEvaluator(model_builder, output_dir=self.output_dir)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        problem = HyperbandTestProblem(output_dir=tmp_dir_name)
        optimiser = HyperbandOptimiser(eta=ETA, max_iter=MAX_ITER, min_or_max=min,
                                       optimisation_func=Optimiser.default_optimisation_func)
        res = optimiser.run_optimisation(problem, verbosity=True)
        print("TEST " + "PASSED" if optimiser.eval_history[0].evaluator.arm == res.evaluator.arm else "FAILED")
