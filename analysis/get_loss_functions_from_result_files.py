from pathlib import Path
import pickle
import json

OUTPUTS_DIR = "/vol/biomedic2/jpassera/cristian_param-tuning/outputs/"
TO_SEARCH = "**/results*"

print(f"Searching for {TO_SEARCH} result files in {OUTPUTS_DIR}")
files = [f for f in Path(OUTPUTS_DIR).glob(TO_SEARCH)]
print("Found files")
[print(f) for f in files]

res = {}
for filepath in files:
    try:
        with open(filepath, "rb") as f:
            print(f"----- started {filepath}")
            optimum_evaluation, eval_history, checkpoints = pickle.load(f)
            res[str(filepath)] = [(e.loss_history, e.arm) for e, g in eval_history]
            # res[str(filepath)] = [(optimum_evaluation[0].loss_history, optimum_evaluation[0].arm)]
            print(f"----- finished {filepath}")
    except Exception as e:
        print(f"----------------------file {filepath} failed")
        print(e)

print(json.dumps({k: [len(lf) for lf, _ in v] for k, v in res.items()},indent=4, sort_keys=True))

print("Pickling result")
with open("best_loss_functions_cifar_hb.pkl", "wb") as f:
    pickle.dump(res, f)
