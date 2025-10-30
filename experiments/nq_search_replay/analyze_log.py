import json
import re

log_file = "E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/training_20251027_211551.log"

# Extract metrics
steps = []
train_success = []
val_success = []
train_rewards = []
offpolicy_ess_ratio = []
offpolicy_kl = []
offpolicy_ratio_mean = []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        if '"system/step"' in line:
            # Extract JSON
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    data = json.loads(match.group())
                    step = data.get('system/step', -1)

                    if step >= 0:
                        steps.append(step)
                        train_success.append(data.get('env/NQSearch/success', None))
                        val_success.append(data.get('val/env/NQSearchVal/success', None))
                        train_rewards.append(data.get('critic/score/mean', None))
                        offpolicy_ess_ratio.append(data.get('offpolicy/ess_ratio', None))
                        offpolicy_kl.append(data.get('offpolicy/kl_divergence', None))
                        offpolicy_ratio_mean.append(data.get('offpolicy/ratio/mean', None))
                except:
                    pass

print(f"Total steps: {len(steps)}")
print(f"Steps range: {steps[0] if steps else 'N/A'} to {steps[-1] if steps else 'N/A'}")

# Calculate statistics
train_success_clean = [x for x in train_success if x is not None]
val_success_clean = [x for x in val_success if x is not None]

if train_success_clean:
    print(f"\n=== Training Success Rate Statistics ===")
    print(f"Mean: {sum(train_success_clean)/len(train_success_clean):.4f}")
    print(f"Min: {min(train_success_clean):.4f}")
    print(f"Max: {max(train_success_clean):.4f}")
    print(f"Last 10 values: {[f'{x:.4f}' for x in train_success_clean[-10:]]}")

if val_success_clean:
    print(f"\n=== Validation Success Rate Statistics ===")
    print(f"Mean: {sum(val_success_clean)/len(val_success_clean):.4f}")
    print(f"Min: {min(val_success_clean):.4f}")
    print(f"Max: {max(val_success_clean):.4f}")
    print(f"Last 10 values: {[f'{x:.4f}' for x in val_success_clean[-10:]]}")

print(f"\n=== First 20 Steps Training Success ===")
for i, (s, ts) in enumerate(zip(steps[:20], train_success[:20])):
    if ts is not None:
        print(f"Step {s}: {ts:.4f}")

print(f"\n=== Last 20 Steps Training Success ===")
for i, (s, ts) in enumerate(zip(steps[-20:], train_success[-20:])):
    if ts is not None:
        print(f"Step {s}: {ts:.4f}")

# Off-policy metrics
offpolicy_ess_clean = [x for x in offpolicy_ess_ratio if x is not None]
if offpolicy_ess_clean:
    print(f"\n=== Off-Policy ESS Ratio ===")
    print(f"Mean: {sum(offpolicy_ess_clean)/len(offpolicy_ess_clean):.4f}")
    print(f"Last 10 values: {[f'{x:.4f}' for x in offpolicy_ess_clean[-10:]]}")
