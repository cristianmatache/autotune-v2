from pathlib import Path
import pickle
import json

OUTPUTS_DIR = "/vol/biomedic2/jpassera/cristian_param-tuning/outputs/hb+tpe"
ONLY_OPTIMUMS = True
OPTIMUM_LF_THRESH = 100


print(f"Searching for arm files in {OUTPUTS_DIR}")
files = [f for f in Path(OUTPUTS_DIR).glob("**/loss_progress.model.pth")]
print("Found files")
[print(f) for f in files]

res = {}
for filepath in files:
    try:
        with open(filepath, "rb") as f:
            print(f"----- started {filepath}")
            loss_func = pickle.load(f)
            if ONLY_OPTIMUMS and len(loss_func) > OPTIMUM_LF_THRESH:
                res[str(filepath)] = loss_func
            print(f"----- finished {filepath}")
    except Exception as e:
        print(f"----------------------file {filepath} failed")
        print(e)

res_info = sorted({k: (len(v), v[-1]) for k, v in res.items()}.items(), key=lambda kv: kv[1][0])
print(json.dumps(res_info, indent=4))
print()
print({len(v) for k, v in res.items()})

print("Pickling result")
with open("longest_loss_functions_from_arm_none.pkl", "wb") as f:
    pickle.dump(res, f)

