from pathlib import Path
import pickle

OUTPUTS_DIR = "/vol/biomedic2/jpassera/cristian_param-tuning/outputs/hb+tpe+transfer+surv"

print("Searching for result files")
files = [f for f in Path(OUTPUTS_DIR).glob("**/loss_progress.model.pth")]
print("Found files")
print([f for f in files])

res = {}
for filepath in files:
    try:
        with open(filepath, "rb") as f:
            print(f"----- started {filepath}")
            loss_func = pickle.load(f)
            res[str(filepath)] = loss_func
            print(f"----- finished {filepath}")
    except Exception as e:
        print(f"----------------------file {filepath} failed")
        print(e)

print({k:v for k, v in res.items()})
print()
print({len(v) for k, v in res.items()})

print("Pickling result")
with open("loss_functions_from_arm.pkl", "wb") as f:
    pickle.dump(res, f)

