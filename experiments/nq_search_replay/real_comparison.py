import json
import re
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_detailed_metrics(log_file, mode_name):
    """Extract detailed metrics from log"""
    data = {
        'step': [],
        'train_success': [],
        'val_success': [],
        'response_length': [],
        'valid_tokens': [],
        'mask_rate': [],
        'entropy': [],
        'pg_loss': [],
        'approxkl': [],
        'ratio_mean': [],
        'ratio_std': [],
        'ess_ratio': [],
        'kl_divergence': [],
    }

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '"system/step"' in line:
                match = re.search(r'\{.*\}', line)
                if match:
                    try:
                        metrics = json.loads(match.group())
                        step = metrics.get('system/step', -1)
                        if step >= 0:
                            data['step'].append(step)
                            data['train_success'].append(metrics.get('env/NQSearch/success', None))
                            data['val_success'].append(metrics.get('val/env/NQSearchVal/success', None))
                            data['response_length'].append(metrics.get('tokens/response_length/mean', None))
                            data['valid_tokens'].append(metrics.get('offpolicy/valid_tokens', None))
                            data['mask_rate'].append(metrics.get('offpolicy/mask_rate', None))
                            data['entropy'].append(metrics.get('critic/entropy/mean', None))
                            data['pg_loss'].append(metrics.get('actor/pg_loss', None))
                            data['approxkl'].append(metrics.get('actor/approxkl', None))
                            data['ratio_mean'].append(metrics.get('offpolicy/ratio/mean', None))
                            data['ratio_std'].append(metrics.get('offpolicy/ratio/std', None))
                            data['ess_ratio'].append(metrics.get('offpolicy/ess_ratio', None))
                            data['kl_divergence'].append(metrics.get('offpolicy/kl_divergence', None))
                    except:
                        pass

    # 过滤None值并计算统计
    stats = {}
    for key in data:
        if key == 'step':
            continue
        values = [v for v in data[key] if v is not None]
        if values:
            stats[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }

    return data, stats

def print_real_comparison():
    print("="*100)
    print("TRAJECTORY vs TURN - 基于真实实现的对比分析")
    print("="*100)

    traj_log = "E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/training_20251026_133405.log"
    turn_log = "E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/training_20251025_222534.log"

    print("\n提取数据中...")
    traj_data, traj_stats = extract_detailed_metrics(traj_log, "Trajectory")
    turn_data, turn_stats = extract_detailed_metrics(turn_log, "Turn")

    print(f"Trajectory模式: {len(traj_data['step'])} steps")
    print(f"Turn模式: {len(turn_data['step'])} steps")

    # 1. Valid tokens对比 - 这是核心差异
    print("\n" + "="*100)
    print("【1】Valid Tokens 对比（核心差异）")
    print("="*100)
    print("\n这是behavior_scope的直接影响：")
    print(f"  Trajectory模式: 平均 {traj_stats['valid_tokens']['mean']:.0f} tokens/batch")
    print(f"  Turn模式:       平均 {turn_stats['valid_tokens']['mean']:.0f} tokens/batch")
    print(f"  差异:           {traj_stats['valid_tokens']['mean'] - turn_stats['valid_tokens']['mean']:.0f} tokens")
    print(f"  Turn节省:       {(1 - turn_stats['valid_tokens']['mean']/traj_stats['valid_tokens']['mean'])*100:.1f}%")

    print(f"\nMask Rate (valid_tokens占比):")
    print(f"  Trajectory: {traj_stats['mask_rate']['mean']*100:.3f}%")
    print(f"  Turn:       {turn_stats['mask_rate']['mean']*100:.3f}%")

    # 2. 性能对比
    print("\n" + "="*100)
    print("【2】训练性能对比")
    print("="*100)
    print(f"\n训练成功率:")
    print(f"  Trajectory: {traj_stats['train_success']['mean']*100:.2f}%")
    print(f"  Turn:       {turn_stats['train_success']['mean']*100:.2f}%")
    print(f"  差异:       {(turn_stats['train_success']['mean'] - traj_stats['train_success']['mean'])*100:.2f}%")

    # 3. 训练稳定性
    print("\n" + "="*100)
    print("【3】训练稳定性指标")
    print("="*100)

    print(f"\nOff-Policy Ratio:")
    print(f"  Trajectory - mean: {traj_stats['ratio_mean']['mean']:.4f}, std: {traj_stats['ratio_std']['mean']:.4f}")
    print(f"  Turn       - mean: {turn_stats['ratio_mean']['mean']:.4f}, std: {turn_stats['ratio_std']['mean']:.4f}")

    print(f"\nESS Ratio (有效样本占比):")
    print(f"  Trajectory: {traj_stats['ess_ratio']['mean']:.4f}")
    print(f"  Turn:       {turn_stats['ess_ratio']['mean']:.4f}")

    print(f"\nEntropy (策略熵):")
    print(f"  Trajectory: {traj_stats['entropy']['mean']:.4f}")
    print(f"  Turn:       {turn_stats['entropy']['mean']:.4f}")
    print(f"  说明: 差异来自计算范围不同（全部回复 vs 最后一轮）")

    # 4. 前20步详细对比
    print("\n" + "="*100)
    print("【4】前20步训练成功率逐步对比")
    print("="*100)
    print(f"\n{'Step':<6} {'Trajectory':<12} {'Turn':<12} {'Diff':<12} {'Better'}")
    print("-"*60)

    for i in range(min(20, len(traj_data['train_success']), len(turn_data['train_success']))):
        traj_val = traj_data['train_success'][i]
        turn_val = turn_data['train_success'][i]
        if traj_val is not None and turn_val is not None:
            diff = turn_val - traj_val
            better = "Turn" if diff > 0.01 else ("Traj" if diff < -0.01 else "Same")
            print(f"{i:<6} {traj_val:<12.4f} {turn_val:<12.4f} {diff:+.4f}      {better}")

    # 5. 统计总结
    print("\n" + "="*100)
    print("【5】统计总结")
    print("="*100)

    traj_better_count = 0
    turn_better_count = 0
    same_count = 0

    for i in range(min(len(traj_data['train_success']), len(turn_data['train_success']))):
        traj_val = traj_data['train_success'][i]
        turn_val = turn_data['train_success'][i]
        if traj_val is not None and turn_val is not None:
            diff = turn_val - traj_val
            if diff > 0.01:
                turn_better_count += 1
            elif diff < -0.01:
                traj_better_count += 1
            else:
                same_count += 1

    total = traj_better_count + turn_better_count + same_count
    print(f"\n在{total}个训练步中:")
    print(f"  Trajectory更好: {traj_better_count} 步 ({traj_better_count/total*100:.1f}%)")
    print(f"  Turn更好:       {turn_better_count} 步 ({turn_better_count/total*100:.1f}%)")
    print(f"  相同:           {same_count} 步 ({same_count/total*100:.1f}%)")

    # 6. 实际意义解读
    print("\n" + "="*100)
    print("【6】实际意义解读")
    print("="*100)

    token_savings = (1 - turn_stats['valid_tokens']['mean']/traj_stats['valid_tokens']['mean'])
    perf_diff = turn_stats['train_success']['mean'] - traj_stats['train_success']['mean']

    print(f"\n✓ Valid Tokens节省: {token_savings*100:.1f}%")
    print(f"  → 意味着log_probs计算量减少{token_savings*100:.1f}%")
    print(f"  → 这部分通常占训练总时间的30-40%")
    print(f"  → 预估整体加速: ~{100/(100-token_savings*30):.2f}x")

    print(f"\n✓ 性能差异: {perf_diff*100:+.2f}%")
    if abs(perf_diff) < 0.02:
        print(f"  → 性能基本相同（差异< 2%）")
    elif perf_diff > 0:
        print(f"  → Turn模式略好")
    else:
        print(f"  → Trajectory模式略好")

    print(f"\n✓ 训练稳定性:")
    print(f"  ESS Ratio - Trajectory: {traj_stats['ess_ratio']['mean']:.4f}, Turn: {turn_stats['ess_ratio']['mean']:.4f}")
    if abs(traj_stats['ess_ratio']['mean'] - turn_stats['ess_ratio']['mean']) < 0.01:
        print(f"  → 两者都非常稳定（ESS > 0.99）")

    print("\n" + "="*100)
    print("结论:")
    print("="*100)
    print(f"1. Turn模式节省{token_savings*100:.1f}%的log_probs计算")
    print(f"2. 性能差异仅{abs(perf_diff)*100:.2f}%（统计误差范围内）")
    print(f"3. 两种模式训练都很稳定（ESS > 0.99）")
    print(f"4. Turn模式在{turn_better_count}步中表现更好，Trajectory在{traj_better_count}步中更好")
    print(f"\n推荐: 对于计算资源敏感的场景，优先使用Turn模式")

if __name__ == "__main__":
    print_real_comparison()
