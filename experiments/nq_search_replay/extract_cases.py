import json
import re
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_cases_from_log(log_file, max_cases=20):
    """Extract conversation cases from log file"""
    cases = []

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all JSON array patterns with prompt/response/episode_score
    pattern = r'\[INFO\].*?\[(\{.*?\}(?:,\s*\{.*?\})*)\]'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        try:
            # Parse JSON array
            json_str = '[' + match + ']'
            batch = json.loads(json_str)

            for item in batch:
                if 'prompt' in item and 'response' in item:
                    # Extract question from prompt
                    q_match = re.search(r'Question:\s*(.*?)(?:\n|assistant)', item['prompt'])
                    if q_match:
                        question = q_match.group(1).strip()
                        response = item['response']
                        score = item.get('episode_score', 0.0)
                        penalty = item.get('penalty', 0.0)

                        # Parse response to count turns
                        think_count = len(re.findall(r'<think>.*?</think>', response, re.DOTALL))
                        search_count = len(re.findall(r'<search>.*?</search>', response, re.DOTALL))
                        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                        answer = answer_match.group(1).strip() if answer_match else ""

                        cases.append({
                            'question': question,
                            'response': response,
                            'score': score,
                            'penalty': penalty,
                            'think_count': think_count,
                            'search_count': search_count,
                            'answer': answer,
                            'response_length': len(response)
                        })

                        if len(cases) >= max_cases:
                            return cases
        except:
            continue

    return cases

def analyze_case_differences():
    """Compare cases from trajectory and turn modes"""
    print("="*100)
    print("案例分析：Trajectory vs Turn 模式")
    print("="*100)

    # Extract cases
    print("\n提取案例中...")
    traj_cases = extract_cases_from_log(
        'output/training_20251026_133405.log',
        max_cases=30
    )
    turn_cases = extract_cases_from_log(
        'output/training_20251025_222534.log',
        max_cases=30
    )

    print(f"Trajectory模式: {len(traj_cases)} 个案例")
    print(f"Turn模式: {len(turn_cases)} 个案例")

    # Find matching questions
    print("\n" + "="*100)
    print("【匹配相同问题的案例对比】")
    print("="*100)

    matched = []
    for t_case in traj_cases:
        for turn_case in turn_cases:
            if t_case['question'] == turn_case['question']:
                matched.append((t_case, turn_case))
                break

    print(f"\n找到 {len(matched)} 对相同问题的案例\n")

    # Analyze matched cases
    for idx, (traj, turn) in enumerate(matched[:10], 1):  # Show first 10
        print(f"\n{'='*100}")
        print(f"【案例 {idx}】")
        print(f"{'='*100}")
        print(f"\n问题: {traj['question']}")

        print(f"\n--- Trajectory模式 ---")
        print(f"得分: {traj['score']}")
        print(f"思考次数: {traj['think_count']}, 搜索次数: {traj['search_count']}")
        print(f"回复长度: {traj['response_length']} 字符")
        print(f"答案: {traj['answer']}")
        print(f"\n完整回复:\n{traj['response'][:500]}...")

        print(f"\n--- Turn模式 ---")
        print(f"得分: {turn['score']}")
        print(f"思考次数: {turn['think_count']}, 搜索次数: {turn['search_count']}")
        print(f"回复长度: {turn['response_length']} 字符")
        print(f"答案: {turn['answer']}")
        print(f"\n完整回复:\n{turn['response'][:500]}...")

        print(f"\n【差异分析】")
        print(f"得分差异: {turn['score'] - traj['score']:.2f}")
        print(f"搜索次数差异: {turn['search_count'] - traj['search_count']}")
        print(f"回复长度差异: {turn['response_length'] - traj['response_length']} 字符")
        if traj['score'] != turn['score']:
            if traj['score'] > turn['score']:
                print("→ Trajectory模式在此案例表现更好")
            else:
                print("→ Turn模式在此案例表现更好")
        else:
            print("→ 两种模式得分相同")

    # Statistics
    print("\n\n" + "="*100)
    print("【统计分析】")
    print("="*100)

    # Success rate by complexity
    traj_simple = [c for c in traj_cases if c['search_count'] == 0]
    traj_complex = [c for c in traj_cases if c['search_count'] > 0]
    turn_simple = [c for c in turn_cases if c['search_count'] == 0]
    turn_complex = [c for c in turn_cases if c['search_count'] > 0]

    print("\n【按问题复杂度分类】")
    print(f"\n简单问题（无需搜索）:")
    print(f"  Trajectory: {len(traj_simple)} 个, 成功率: {sum(c['score'] for c in traj_simple)/max(len(traj_simple),1)*100:.1f}%")
    print(f"  Turn:       {len(turn_simple)} 个, 成功率: {sum(c['score'] for c in turn_simple)/max(len(turn_simple),1)*100:.1f}%")

    print(f"\n复杂问题（需要搜索）:")
    print(f"  Trajectory: {len(traj_complex)} 个, 成功率: {sum(c['score'] for c in traj_complex)/max(len(traj_complex),1)*100:.1f}%")
    print(f"  Turn:       {len(turn_complex)} 个, 成功率: {sum(c['score'] for c in turn_complex)/max(len(turn_complex),1)*100:.1f}%")

    # Average response length
    print(f"\n【平均回复长度】")
    print(f"  Trajectory: {sum(c['response_length'] for c in traj_cases)/len(traj_cases):.0f} 字符")
    print(f"  Turn:       {sum(c['response_length'] for c in turn_cases)/len(turn_cases):.0f} 字符")

    # Average searches
    print(f"\n【平均搜索次数】")
    print(f"  Trajectory: {sum(c['search_count'] for c in traj_cases)/len(traj_cases):.2f}")
    print(f"  Turn:       {sum(c['search_count'] for c in turn_cases)/len(turn_cases):.2f}")

    # Show some successful and failed cases
    print("\n\n" + "="*100)
    print("【成功案例示例】")
    print("="*100)

    successful_traj = [c for c in traj_cases if c['score'] == 1.0][:3]
    for idx, case in enumerate(successful_traj, 1):
        print(f"\n案例 {idx}:")
        print(f"Q: {case['question']}")
        print(f"A: {case['answer']}")
        print(f"搜索: {case['search_count']} 次")

    print("\n\n" + "="*100)
    print("【失败案例示例】")
    print("="*100)

    failed_traj = [c for c in traj_cases if c['score'] == 0.0][:3]
    for idx, case in enumerate(failed_traj, 1):
        print(f"\n案例 {idx}:")
        print(f"Q: {case['question']}")
        print(f"A: {case['answer']}")
        print(f"搜索: {case['search_count']} 次")

if __name__ == "__main__":
    analyze_case_differences()