import json
import re
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_all_cases_from_log(log_file):
    """Extract all conversation cases from log file"""
    cases = []

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all JSON array patterns
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

                        cases.append({
                            'question': question,
                            'response': response,
                            'score': score
                        })
        except:
            continue

    return cases

def main():
    print("="*100)
    print("详细案例分析：Trajectory vs Turn 模式")
    print("="*100)

    # Extract all cases
    traj_cases = extract_all_cases_from_log('output/training_20251026_133405.log')
    turn_cases = extract_all_cases_from_log('output/training_20251025_222534.log')

    print(f"\nTrajectory模式: {len(traj_cases)} 个案例")
    print(f"Turn模式: {len(turn_cases)} 个案例")

    # Separate success and failure
    traj_success = [c for c in traj_cases if c['score'] == 1.0]
    traj_fail = [c for c in traj_cases if c['score'] == 0.0]
    turn_success = [c for c in turn_cases if c['score'] == 1.0]
    turn_fail = [c for c in turn_cases if c['score'] == 0.0]

    print(f"\nTrajectory - 成功: {len(traj_success)}, 失败: {len(traj_fail)}")
    print(f"Turn       - 成功: {len(turn_success)}, 失败: {len(turn_fail)}")

    # Detailed analysis of failures
    print("\n\n" + "="*100)
    print("【Trajectory模式 - 失败案例详细分析】")
    print("="*100)

    for idx, case in enumerate(traj_fail[:5], 1):
        print(f"\n{'='*100}")
        print(f"失败案例 {idx}")
        print(f"{'='*100}")
        print(f"\n问题: {case['question']}")
        print(f"\n模型回复:")
        print(case['response'])
        print(f"\n得分: {case['score']}")

    print("\n\n" + "="*100)
    print("【Turn模式 - 失败案例详细分析】")
    print("="*100)

    for idx, case in enumerate(turn_fail[:5], 1):
        print(f"\n{'='*100}")
        print(f"失败案例 {idx}")
        print(f"{'='*100}")
        print(f"\n问题: {case['question']}")
        print(f"\n模型回复:")
        print(case['response'])
        print(f"\n得分: {case['score']}")

    # Success examples
    print("\n\n" + "="*100)
    print("【Trajectory模式 - 成功案例示例】")
    print("="*100)

    for idx, case in enumerate(traj_success[:5], 1):
        print(f"\n{'='*100}")
        print(f"成功案例 {idx}")
        print(f"{'='*100}")
        print(f"\n问题: {case['question']}")
        print(f"\n模型回复:")
        print(case['response'])

    print("\n\n" + "="*100)
    print("【Turn模式 - 成功案例示例】")
    print("="*100)

    for idx, case in enumerate(turn_success[:5], 1):
        print(f"\n{'='*100}")
        print(f"成功案例 {idx}")
        print(f"{'='*100}")
        print(f"\n问题: {case['question']}")
        print(f"\n模型回复:")
        print(case['response'])

    # Analyze response patterns
    print("\n\n" + "="*100)
    print("【回复模式分析】")
    print("="*100)

    def count_patterns(cases):
        """Count different response patterns"""
        multi_turn = 0
        single_turn = 0
        with_search = 0
        no_search = 0

        for c in cases:
            turns = len(re.findall(r'<think>.*?</think>', c['response'], re.DOTALL))
            searches = len(re.findall(r'<search>.*?</search>', c['response'], re.DOTALL))

            if turns > 1:
                multi_turn += 1
            else:
                single_turn += 1

            if searches > 0:
                with_search += 1
            else:
                no_search += 1

        return {
            'multi_turn': multi_turn,
            'single_turn': single_turn,
            'with_search': with_search,
            'no_search': no_search
        }

    traj_patterns = count_patterns(traj_cases)
    turn_patterns = count_patterns(turn_cases)

    print(f"\nTrajectory模式:")
    print(f"  多轮对话: {traj_patterns['multi_turn']}")
    print(f"  单轮回复: {traj_patterns['single_turn']}")
    print(f"  使用搜索: {traj_patterns['with_search']}")
    print(f"  无需搜索: {traj_patterns['no_search']}")

    print(f"\nTurn模式:")
    print(f"  多轮对话: {turn_patterns['multi_turn']}")
    print(f"  单轮回复: {turn_patterns['single_turn']}")
    print(f"  使用搜索: {turn_patterns['with_search']}")
    print(f"  无需搜索: {turn_patterns['no_search']}")

    # Success rate by pattern
    print("\n\n" + "="*100)
    print("【不同模式下的成功率对比】")
    print("="*100)

    def success_by_pattern(cases):
        """Calculate success rate by response pattern"""
        multi_turn = [c for c in cases if len(re.findall(r'<think>', c['response'])) > 1]
        single_turn = [c for c in cases if len(re.findall(r'<think>', c['response'])) <= 1]
        with_search = [c for c in cases if '<search>' in c['response']]
        no_search = [c for c in cases if '<search>' not in c['response']]

        return {
            'multi_turn': sum(c['score'] for c in multi_turn) / max(len(multi_turn), 1) * 100,
            'single_turn': sum(c['score'] for c in single_turn) / max(len(single_turn), 1) * 100,
            'with_search': sum(c['score'] for c in with_search) / max(len(with_search), 1) * 100,
            'no_search': sum(c['score'] for c in no_search) / max(len(no_search), 1) * 100,
        }

    traj_success_rate = success_by_pattern(traj_cases)
    turn_success_rate = success_by_pattern(turn_cases)

    print(f"\n成功率对比 (%):")
    print(f"{'模式':<20} {'Trajectory':>15} {'Turn':>15} {'差异':>15}")
    print(f"{'-'*70}")
    print(f"{'多轮对话':<20} {traj_success_rate['multi_turn']:>15.1f} {turn_success_rate['multi_turn']:>15.1f} {turn_success_rate['multi_turn']-traj_success_rate['multi_turn']:>15.1f}")
    print(f"{'单轮回复':<20} {traj_success_rate['single_turn']:>15.1f} {turn_success_rate['single_turn']:>15.1f} {turn_success_rate['single_turn']-traj_success_rate['single_turn']:>15.1f}")
    print(f"{'使用搜索':<20} {traj_success_rate['with_search']:>15.1f} {turn_success_rate['with_search']:>15.1f} {turn_success_rate['with_search']-traj_success_rate['with_search']:>15.1f}")
    print(f"{'无需搜索':<20} {traj_success_rate['no_search']:>15.1f} {turn_success_rate['no_search']:>15.1f} {turn_success_rate['no_search']-traj_success_rate['no_search']:>15.1f}")

if __name__ == "__main__":
    main()
