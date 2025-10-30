import json
import re
from pathlib import Path

def extract_metrics(log_file):
    """Extract key metrics from log file"""
    steps = []
    train_success = []
    val_success = []
    train_rewards = []
    entropy = []
    response_length = []
    valid_tokens = []
    mask_rate = []
    pg_loss = []
    kl_loss = []
    approxkl = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '"system/step"' in line:
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
                            entropy.append(data.get('critic/entropy/mean', None))
                            response_length.append(data.get('tokens/response_length/mean', None))

                            # Off-policy related metrics
                            vt = data.get('offpolicy/valid_tokens', None)
                            tt = data.get('offpolicy/total_tokens', None)
                            valid_tokens.append(vt)
                            mask_rate.append(data.get('offpolicy/mask_rate', None))

                            # Training metrics
                            pg_loss.append(data.get('actor/pg_loss', None))
                            kl_loss.append(data.get('actor/kl_loss', None))
                            approxkl.append(data.get('actor/approxkl', None))
                    except:
                        pass

    return {
        'steps': steps,
        'train_success': train_success,
        'val_success': val_success,
        'train_rewards': train_rewards,
        'entropy': entropy,
        'response_length': response_length,
        'valid_tokens': valid_tokens,
        'mask_rate': mask_rate,
        'pg_loss': pg_loss,
        'kl_loss': kl_loss,
        'approxkl': approxkl
    }

def print_comparison(trajectory_data, turn_data):
    """Print detailed comparison"""
    print("="*80)
    print("TRAJECTORY MODE vs TURN MODE - Detailed Comparison")
    print("="*80)

    # Success rate comparison
    print("\n【1. Success Rate Comparison】")
    print("-"*80)

    traj_train_success = [s for s in trajectory_data['train_success'] if s is not None]
    turn_train_success = [s for s in turn_data['train_success'] if s is not None]

    traj_val_success = [s for s in trajectory_data['val_success'] if s is not None]
    turn_val_success = [s for s in turn_data['val_success'] if s is not None]

    print(f"Training Success Rate:")
    print(f"  Trajectory Mode: {sum(traj_train_success)/len(traj_train_success)*100:.2f}% (avg over {len(traj_train_success)} steps)")
    print(f"  Turn Mode:       {sum(turn_train_success)/len(turn_train_success)*100:.2f}% (avg over {len(turn_train_success)} steps)")

    print(f"\nValidation Success Rate:")
    print(f"  Trajectory Mode: {sum(traj_val_success)/len(traj_val_success)*100:.2f}% (avg over {len(traj_val_success)} evals)")
    print(f"  Turn Mode:       {sum(turn_val_success)/len(turn_val_success)*100:.2f}% (avg over {len(turn_val_success)} evals)")

    # Token efficiency
    print("\n【2. Token Efficiency (Valid Tokens for Training)】")
    print("-"*80)

    traj_valid_tokens = [t for t in trajectory_data['valid_tokens'] if t is not None]
    turn_valid_tokens = [t for t in turn_data['valid_tokens'] if t is not None]

    traj_mask_rate = [m for m in trajectory_data['mask_rate'] if m is not None]
    turn_mask_rate = [m for m in turn_data['mask_rate'] if m is not None]

    print(f"Average Valid Tokens per Batch:")
    print(f"  Trajectory Mode: {sum(traj_valid_tokens)/len(traj_valid_tokens):.0f} tokens")
    print(f"  Turn Mode:       {sum(turn_valid_tokens)/len(turn_valid_tokens):.0f} tokens")
    print(f"  Difference:      {sum(traj_valid_tokens)/len(traj_valid_tokens) - sum(turn_valid_tokens)/len(turn_valid_tokens):.0f} tokens")
    print(f"  Turn saves:      {(1 - sum(turn_valid_tokens)/sum(traj_valid_tokens))*100:.1f}% tokens")

    print(f"\nMask Rate (valid_tokens / total_tokens):")
    print(f"  Trajectory Mode: {sum(traj_mask_rate)/len(traj_mask_rate)*100:.3f}%")
    print(f"  Turn Mode:       {sum(turn_mask_rate)/len(turn_mask_rate)*100:.3f}%")

    # Response length
    print("\n【3. Response Length】")
    print("-"*80)

    traj_resp_len = [r for r in trajectory_data['response_length'] if r is not None]
    turn_resp_len = [r for r in turn_data['response_length'] if r is not None]

    print(f"Average Response Length:")
    print(f"  Trajectory Mode: {sum(traj_resp_len)/len(traj_resp_len):.1f} tokens")
    print(f"  Turn Mode:       {sum(turn_resp_len)/len(turn_resp_len):.1f} tokens")

    # Training stability
    print("\n【4. Training Stability Metrics】")
    print("-"*80)

    traj_entropy = [e for e in trajectory_data['entropy'] if e is not None]
    turn_entropy = [e for e in turn_data['entropy'] if e is not None]

    traj_approxkl = [k for k in trajectory_data['approxkl'] if k is not None]
    turn_approxkl = [k for k in turn_data['approxkl'] if k is not None]

    print(f"Average Entropy:")
    print(f"  Trajectory Mode: {sum(traj_entropy)/len(traj_entropy):.4f}")
    print(f"  Turn Mode:       {sum(turn_entropy)/len(turn_entropy):.4f}")

    print(f"\nAverage ApproxKL:")
    print(f"  Trajectory Mode: {sum(traj_approxkl)/len(traj_approxkl):.4f}")
    print(f"  Turn Mode:       {sum(turn_approxkl)/len(turn_approxkl):.4f}")

    # First 20 steps detailed comparison
    print("\n【5. First 20 Steps - Training Success Rate】")
    print("-"*80)
    print(f"{'Step':<6} {'Trajectory':<12} {'Turn':<12} {'Difference':<12}")
    print("-"*50)
    for i in range(min(20, len(traj_train_success), len(turn_train_success))):
        diff = traj_train_success[i] - turn_train_success[i]
        print(f"{i:<6} {traj_train_success[i]:<12.4f} {turn_train_success[i]:<12.4f} {diff:+.4f}")

    # Summary
    print("\n【6. Summary】")
    print("="*80)

    traj_better = sum([1 for i in range(min(len(traj_train_success), len(turn_train_success)))
                      if traj_train_success[i] > turn_train_success[i]])
    turn_better = sum([1 for i in range(min(len(traj_train_success), len(turn_train_success)))
                      if turn_train_success[i] > traj_train_success[i]])

    print(f"Training Performance:")
    print(f"  Trajectory better in {traj_better} steps")
    print(f"  Turn better in {turn_better} steps")

    token_savings = (1 - sum(turn_valid_tokens)/sum(traj_valid_tokens))*100
    print(f"\nComputational Efficiency:")
    print(f"  Turn mode saves {token_savings:.1f}% tokens for log_probs computation")
    print(f"  Estimated speedup: ~{100/(100-token_savings):.2f}x for that component")

# Main
trajectory_log = "E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/training_20251026_133405.log"
turn_log = "E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/training_20251025_222534.log"

print("Extracting metrics from logs...")
trajectory_data = extract_metrics(trajectory_log)
turn_data = extract_metrics(turn_log)

print(f"Trajectory mode: {len(trajectory_data['steps'])} training steps")
print(f"Turn mode: {len(turn_data['steps'])} training steps")
print()

print_comparison(trajectory_data, turn_data)
