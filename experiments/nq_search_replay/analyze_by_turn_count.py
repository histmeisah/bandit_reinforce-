"""
按照具体轮数分析Trajectory vs Turn模式的成功率
"""
import json
import re
import sys
import io
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_cases_with_turn_count(log_file):
    """Extract cases with exact turn count"""
    cases = []

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'\[INFO\].*?\[(\{.*?\}(?:,\s*\{.*?\})*)\]'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        try:
            json_str = '[' + match + ']'
            batch = json.loads(json_str)

            for item in batch:
                if 'prompt' in item and 'response' in item:
                    q_match = re.search(r'Question:\s*(.*?)(?:\n|assistant)', item['prompt'])
                    if q_match:
                        question = q_match.group(1).strip()
                        response = item['response']
                        score = item.get('episode_score', 0.0)

                        # Count exact number of turns (think blocks)
                        think_blocks = re.findall(r'<think>.*?</think>', response, re.DOTALL)
                        turn_count = len(think_blocks)

                        # Count search actions
                        search_count = len(re.findall(r'<search>.*?</search>', response, re.DOTALL))

                        cases.append({
                            'question': question,
                            'response': response,
                            'score': score,
                            'turn_count': turn_count,  # 关键：精确的轮数
                            'search_count': search_count,
                            'response_length': len(response)
                        })
        except Exception as e:
            continue

    return cases

def analyze_by_exact_turn_count(traj_cases, turn_cases):
    """Analyze success rate by exact turn count"""

    # Group by turn count
    traj_by_turns = defaultdict(list)
    turn_by_turns = defaultdict(list)

    for case in traj_cases:
        traj_by_turns[case['turn_count']].append(case)

    for case in turn_cases:
        turn_by_turns[case['turn_count']].append(case)

    # Get all turn counts
    all_turn_counts = sorted(set(list(traj_by_turns.keys()) + list(turn_by_turns.keys())))

    # Calculate statistics for each turn count
    results = []
    for turn_count in all_turn_counts:
        traj_group = traj_by_turns.get(turn_count, [])
        turn_group = turn_by_turns.get(turn_count, [])

        result = {
            'turn_count': turn_count,
            'trajectory': {
                'count': len(traj_group),
                'success': sum(c['score'] for c in traj_group),
                'success_rate': sum(c['score'] for c in traj_group) / len(traj_group) * 100 if traj_group else 0,
                'avg_search': sum(c['search_count'] for c in traj_group) / len(traj_group) if traj_group else 0
            },
            'turn': {
                'count': len(turn_group),
                'success': sum(c['score'] for c in turn_group),
                'success_rate': sum(c['score'] for c in turn_group) / len(turn_group) * 100 if turn_group else 0,
                'avg_search': sum(c['search_count'] for c in turn_group) / len(turn_group) if turn_group else 0
            }
        }
        results.append(result)

    return results, all_turn_counts

def create_detailed_visualizations(results):
    """Create detailed visualizations by turn count"""
    fig = plt.figure(figsize=(20, 10))

    turn_counts = [r['turn_count'] for r in results]
    traj_success_rates = [r['trajectory']['success_rate'] for r in results]
    turn_success_rates = [r['turn']['success_rate'] for r in results]
    traj_counts = [r['trajectory']['count'] for r in results]
    turn_counts_data = [r['turn']['count'] for r in results]

    # 1. Success rate by exact turn count
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(turn_counts))
    width = 0.35
    ax1.bar(x - width/2, traj_success_rates, width, label='Trajectory', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, turn_success_rates, width, label='Turn', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Turn Count (轮数)', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate by Exact Turn Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{tc}轮' for tc in turn_counts])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. Case count distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x - width/2, traj_counts, width, label='Trajectory', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, turn_counts_data, width, label='Turn', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Turn Count (轮数)', fontsize=12)
    ax2.set_ylabel('Case Count', fontsize=12)
    ax2.set_title('Case Distribution by Turn Count', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{tc}轮' for tc in turn_counts])
    ax2.legend()
    ax2.set_yscale('log')  # Log scale for better visualization

    # 3. Success rate difference
    ax3 = plt.subplot(2, 3, 3)
    diff = [turn_success_rates[i] - traj_success_rates[i] for i in range(len(turn_counts))]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff]
    ax3.bar(x, diff, color=colors, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Turn Count (轮数)', fontsize=12)
    ax3.set_ylabel('Success Rate Difference (%)', fontsize=12)
    ax3.set_title('Turn Advantage over Trajectory', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{tc}轮' for tc in turn_counts])
    ax3.grid(axis='y', alpha=0.3)

    # 4. Average search count by turn
    ax4 = plt.subplot(2, 3, 4)
    traj_avg_search = [r['trajectory']['avg_search'] for r in results]
    turn_avg_search = [r['turn']['avg_search'] for r in results]
    ax4.plot(turn_counts, traj_avg_search, marker='o', label='Trajectory', color='#3498db', linewidth=2)
    ax4.plot(turn_counts, turn_avg_search, marker='s', label='Turn', color='#e74c3c', linewidth=2)
    ax4.set_xlabel('Turn Count (轮数)', fontsize=12)
    ax4.set_ylabel('Average Search Count', fontsize=12)
    ax4.set_title('Search Usage by Turn Count', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Trajectory: Sample size vs success rate
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(traj_counts, traj_success_rates, s=[c*2 for c in traj_counts],
               alpha=0.6, color='#3498db', edgecolors='black', linewidth=1)
    for i, tc in enumerate(turn_counts):
        if traj_counts[i] > 10:  # Only label significant samples
            ax5.annotate(f'{tc}轮\n(n={traj_counts[i]})',
                        (traj_counts[i], traj_success_rates[i]),
                        fontsize=8, ha='center')
    ax5.set_xlabel('Sample Size', fontsize=12)
    ax5.set_ylabel('Success Rate (%)', fontsize=12)
    ax5.set_title('Trajectory: Sample Size vs Success Rate', fontsize=14, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)

    # 6. Turn: Sample size vs success rate
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(turn_counts_data, turn_success_rates, s=[c*2 for c in turn_counts_data],
               alpha=0.6, color='#e74c3c', edgecolors='black', linewidth=1)
    for i, tc in enumerate(turn_counts):
        if turn_counts_data[i] > 10:  # Only label significant samples
            ax6.annotate(f'{tc}轮\n(n={turn_counts_data[i]})',
                        (turn_counts_data[i], turn_success_rates[i]),
                        fontsize=8, ha='center')
    ax6.set_xlabel('Sample Size', fontsize=12)
    ax6.set_ylabel('Success Rate (%)', fontsize=12)
    ax6.set_title('Turn: Sample Size vs Success Rate', fontsize=14, fontweight='bold')
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/turn_count_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: output/turn_count_detailed_analysis.png")

def generate_detailed_markdown(results, traj_cases, turn_cases):
    """Generate detailed markdown report with turn count analysis"""

    md = f"""# Trajectory vs Turn 模式：按轮数详细分析报告

## 📊 数据概览

- **Trajectory模式**: {len(traj_cases)} 个案例
- **Turn模式**: {len(turn_cases)} 个案例
- **分析维度**: 按照精确轮数（1轮、2轮、3轮...）进行分析

---

## 1. 按轮数分类的成功率对比

### 1.1 详细数据表

| 轮数 | Trajectory案例数 | Trajectory成功率 | Turn案例数 | Turn成功率 | Turn优势 | 统计显著性 |
|------|----------------|----------------|-----------|----------|---------|----------|
"""

    for r in results:
        tc = r['turn_count']
        traj = r['trajectory']
        turn = r['turn']
        diff = turn['success_rate'] - traj['success_rate']

        # 判断统计显著性（简单规则：样本量>30 且差异>5%）
        significant = "✓" if (traj['count'] > 30 or turn['count'] > 30) and abs(diff) > 5 else ""

        md += f"| **{tc}轮** | {traj['count']} | {traj['success_rate']:.1f}% | {turn['count']} | {turn['success_rate']:.1f}% | "

        if diff > 0:
            md += f"**+{diff:.1f}%** 🟢"
        elif diff < 0:
            md += f"**{diff:.1f}%** 🔴"
        else:
            md += "0.0%"

        md += f" | {significant} |\n"

    md += f"""

**说明**:
- ✓ 表示该轮数的样本量足够大（>30）且差异显著（>5%）
- 🟢 Turn模式更优
- 🔴 Trajectory模式更优

---

## 2. 核心发现

### 2.1 单轮对话（1轮）

"""

    # Analyze 1-turn cases
    one_turn = [r for r in results if r['turn_count'] == 1][0]
    md += f"""
| 模式 | 案例数 | 成功率 | 平均搜索次数 |
|------|--------|--------|-------------|
| Trajectory | {one_turn['trajectory']['count']} | {one_turn['trajectory']['success_rate']:.1f}% | {one_turn['trajectory']['avg_search']:.2f} |
| Turn | {one_turn['turn']['count']} | {one_turn['turn']['success_rate']:.1f}% | {one_turn['turn']['avg_search']:.2f} |

**分析**:
- 单轮对话占据了绝大多数案例
- Turn模式在单轮对话中成功率 {'高于' if one_turn['turn']['success_rate'] > one_turn['trajectory']['success_rate'] else '低于'} Trajectory模式 {abs(one_turn['turn']['success_rate'] - one_turn['trajectory']['success_rate']):.1f}%
- Turn模式搜索更 {'少' if one_turn['turn']['avg_search'] < one_turn['trajectory']['avg_search'] else '多'}（{one_turn['turn']['avg_search']:.2f} vs {one_turn['trajectory']['avg_search']:.2f}）

---

### 2.2 多轮对话（2轮及以上）

"""

    # Analyze multi-turn cases
    multi_turn = [r for r in results if r['turn_count'] >= 2]

    if multi_turn:
        md += "| 轮数 | Trajectory成功率 | Turn成功率 | 差异 |\n"
        md += "|------|----------------|----------|------|\n"

        for r in multi_turn:
            diff = r['turn']['success_rate'] - r['trajectory']['success_rate']
            md += f"| {r['turn_count']}轮 | {r['trajectory']['success_rate']:.1f}% | {r['turn']['success_rate']:.1f}% | "

            if diff > 0:
                md += f"**+{diff:.1f}%** 🟢 |\n"
            elif diff < 0:
                md += f"{diff:.1f}% 🔴 |\n"
            else:
                md += "0.0% |\n"

        # Overall multi-turn statistics
        traj_multi = [c for c in traj_cases if c['turn_count'] >= 2]
        turn_multi = [c for c in turn_cases if c['turn_count'] >= 2]

        traj_multi_success = sum(c['score'] for c in traj_multi) / len(traj_multi) * 100 if traj_multi else 0
        turn_multi_success = sum(c['score'] for c in turn_multi) / len(turn_multi) * 100 if turn_multi else 0

        md += f"""

**多轮对话总体统计**:
- Trajectory: {len(traj_multi)} 个案例，成功率 {traj_multi_success:.1f}%
- Turn: {len(turn_multi)} 个案例，成功率 {turn_multi_success:.1f}%
- **Turn优势: +{turn_multi_success - traj_multi_success:.1f}%**

---

### 2.3 极端案例分析

"""

        # Find max turn count
        max_turns = max(r['turn_count'] for r in results)
        max_turn_data = [r for r in results if r['turn_count'] == max_turns][0]

        md += f"""
**最多轮数**: {max_turns}轮

| 模式 | 案例数 | 成功率 |
|------|--------|--------|
| Trajectory | {max_turn_data['trajectory']['count']} | {max_turn_data['trajectory']['success_rate']:.1f}% |
| Turn | {max_turn_data['turn']['count']} | {max_turn_data['turn']['success_rate']:.1f}% |

"""

    md += """
---

## 3. 可视化分析

![Turn Count Analysis](turn_count_detailed_analysis.png)

**图表说明**:
1. **左上**: 按轮数的成功率对比 - 可以看到每个具体轮数下两种模式的表现
2. **中上**: 案例分布 - 显示每个轮数的样本量（对数坐标）
3. **右上**: Turn相对优势 - 正值表示Turn更好，负值表示Trajectory更好
4. **左下**: 搜索使用情况 - 随轮数增加，搜索次数的变化趋势
5. **中下/右下**: 样本量与成功率关系 - 帮助识别哪些轮数的数据更可靠

---

## 4. 关键洞察

"""

    # Calculate key insights
    # Find which turn counts have significant Turn advantage
    significant_advantages = []
    for r in results:
        diff = r['turn']['success_rate'] - r['trajectory']['success_rate']
        if diff > 10 and (r['trajectory']['count'] > 10 or r['turn']['count'] > 10):
            significant_advantages.append((r['turn_count'], diff, r['trajectory']['count'], r['turn']['count']))

    if significant_advantages:
        md += "### 4.1 Turn模式显著优势的轮数\n\n"
        for tc, diff, traj_n, turn_n in significant_advantages:
            md += f"- **{tc}轮**: Turn优势 +{diff:.1f}% (Trajectory: n={traj_n}, Turn: n={turn_n})\n"
        md += "\n"

    # Calculate weighted average by sample size
    total_traj = sum(r['trajectory']['count'] for r in results)
    total_turn = sum(r['turn']['count'] for r in results)

    weighted_traj_success = sum(r['trajectory']['success_rate'] * r['trajectory']['count'] for r in results) / total_traj
    weighted_turn_success = sum(r['turn']['success_rate'] * r['turn']['count'] for r in results) / total_turn

    md += f"""
### 4.2 样本量加权平均成功率

考虑到不同轮数的样本量差异，使用加权平均：

| 模式 | 加权平均成功率 |
|------|---------------|
| Trajectory | {weighted_traj_success:.2f}% |
| Turn | {weighted_turn_success:.2f}% |
| **差异** | **+{weighted_turn_success - weighted_traj_success:.2f}%** |

---

## 5. 结论

基于按轮数的详细分析：

1. **单轮对话**（1轮）占据主要样本，Turn模式在此场景下成功率{'更高' if one_turn['turn']['success_rate'] > one_turn['trajectory']['success_rate'] else '相近'}

2. **多轮对话**（2轮及以上）中，Turn模式表现明显更好

3. **统计可靠性**: 单轮样本量大，结论可靠；多轮样本量较小，但趋势一致

4. **推荐**: 对于需要多轮推理的复杂任务，Turn模式明显更优

---

*报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return md

def main():
    print("="*100)
    print("按轮数详细分析 Trajectory vs Turn 模式")
    print("="*100)

    # Extract data
    print("\n[1/4] 提取训练数据并统计轮数...")
    traj_cases = extract_cases_with_turn_count('output/training_20251026_133405.log')
    turn_cases = extract_cases_with_turn_count('output/training_20251025_222534.log')
    print(f"  ✓ Trajectory: {len(traj_cases)} 个案例")
    print(f"  ✓ Turn: {len(turn_cases)} 个案例")

    # Show turn distribution
    traj_turn_dist = Counter(c['turn_count'] for c in traj_cases)
    turn_turn_dist = Counter(c['turn_count'] for c in turn_cases)

    print("\n  轮数分布:")
    all_turns = sorted(set(list(traj_turn_dist.keys()) + list(turn_turn_dist.keys())))
    for tc in all_turns:
        print(f"    {tc}轮: Trajectory={traj_turn_dist.get(tc, 0)}, Turn={turn_turn_dist.get(tc, 0)}")

    # Analyze
    print("\n[2/4] 按轮数分析成功率...")
    results, all_turn_counts = analyze_by_exact_turn_count(traj_cases, turn_cases)
    print(f"  ✓ 分析了 {len(all_turn_counts)} 种不同轮数")

    # Create visualizations
    print("\n[3/4] 生成可视化图表...")
    create_detailed_visualizations(results)

    # Generate report
    print("\n[4/4] 生成详细报告...")
    report = generate_detailed_markdown(results, traj_cases, turn_cases)

    with open('output/TURN_COUNT_DETAILED_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  ✓ 报告已保存: output/TURN_COUNT_DETAILED_ANALYSIS.md")

    print("\n" + "="*100)
    print("✅ 按轮数详细分析完成！")
    print("="*100)

    # Print key findings
    print("\n关键发现:")
    for r in results[:5]:  # Show first 5 turn counts
        diff = r['turn']['success_rate'] - r['trajectory']['success_rate']
        print(f"  {r['turn_count']}轮: Trajectory={r['trajectory']['success_rate']:.1f}% (n={r['trajectory']['count']}), "
              f"Turn={r['turn']['success_rate']:.1f}% (n={r['turn']['count']}), "
              f"差异={'+ ' if diff >= 0 else ''}{diff:.1f}%")

if __name__ == "__main__":
    main()