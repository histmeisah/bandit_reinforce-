"""详细检查 nq_search 数据集格式

查看数据集的完整结构，特别是 prompt 字段的格式
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 解释器的搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ROLL"))

import logging
import datasets
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_dataset_detailed(dataset_path: str, num_samples: int = 5) -> None:
    """详细检查数据集结构
    
    Args:
        dataset_path: 数据集文件路径
        num_samples: 要检查的样本数量
    """
    logger.info(f"正在加载数据集: {dataset_path}")
    
    df = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"数据集基本信息")
    logger.info(f"{'='*80}")
    logger.info(f"总记录数: {len(df)}")
    logger.info(f"列名: {df.column_names}")
    logger.info(f"特征类型: {df.features}")
    
    # 检查前几个样本
    for i in range(min(num_samples, len(df))):
        sample = df[i]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"样本 {i+1}")
        logger.info(f"{'='*80}")
        
        # 逐字段显示
        logger.info(f"\n【ID】: {sample['id']}")
        logger.info(f"\n【Question】: {sample['question']}")
        logger.info(f"\n【Golden Answers】: {sample['golden_answers']}")
        logger.info(f"\n【Data Source】: {sample['data_source']}")
        logger.info(f"\n【Ability】: {sample['ability']}")
        
        # 详细显示 prompt 字段
        logger.info(f"\n【Prompt 详细信息】:")
        prompt = sample['prompt']
        logger.info(f"  - Prompt 类型: {type(prompt)}")
        logger.info(f"  - Prompt 长度: {len(prompt)}")
        
        for j, msg in enumerate(prompt):
            logger.info(f"\n  消息 {j+1}:")
            logger.info(f"    - Role: {msg.get('role', 'N/A')}")
            content = msg.get('content', '')
            logger.info(f"    - Content 长度: {len(content)}")
            logger.info(f"    - Content 内容:")
            # 显示完整内容，但如果太长就截断
            if len(content) > 1000:
                logger.info(f"      {content[:500]}")
                logger.info(f"      ... [中间省略 {len(content)-1000} 字符] ...")
                logger.info(f"      {content[-500:]}")
            else:
                logger.info(f"      {content}")
        
        # 显示 reward_model
        logger.info(f"\n【Reward Model】:")
        logger.info(f"  {json.dumps(sample['reward_model'], indent=2, ensure_ascii=False)}")
        
        # 显示 extra_info
        logger.info(f"\n【Extra Info】:")
        logger.info(f"  {json.dumps(sample['extra_info'], indent=2, ensure_ascii=False)}")
    
    # 统计 prompt 结构
    logger.info(f"\n{'='*80}")
    logger.info(f"Prompt 结构统计")
    logger.info(f"{'='*80}")
    
    prompt_lengths = [len(item['prompt']) for item in df]
    logger.info(f"Prompt 消息数量统计:")
    logger.info(f"  - 最小: {min(prompt_lengths)}")
    logger.info(f"  - 最大: {max(prompt_lengths)}")
    logger.info(f"  - 平均: {sum(prompt_lengths) / len(prompt_lengths):.2f}")
    
    # 检查是否所有 prompt 都只有一个消息
    single_message_count = sum(1 for length in prompt_lengths if length == 1)
    multi_message_count = sum(1 for length in prompt_lengths if length > 1)
    
    logger.info(f"\n消息数量分布:")
    logger.info(f"  - 单消息 (len=1): {single_message_count} ({single_message_count/len(df)*100:.1f}%)")
    logger.info(f"  - 多消息 (len>1): {multi_message_count} ({multi_message_count/len(df)*100:.1f}%)")
    
    # 如果有多消息的样本，展示一个
    if multi_message_count > 0:
        logger.info(f"\n查找多消息样本...")
        for i, item in enumerate(df):
            if len(item['prompt']) > 1:
                logger.info(f"\n找到多消息样本 (索引 {i}):")
                for j, msg in enumerate(item['prompt']):
                    logger.info(f"  消息 {j+1} - Role: {msg.get('role')}, Content 长度: {len(msg.get('content', ''))}")
                break
    
    # 检查 role 类型分布
    logger.info(f"\n{'='*80}")
    logger.info(f"Role 类型统计")
    logger.info(f"{'='*80}")
    
    all_roles = []
    for item in df:
        for msg in item['prompt']:
            all_roles.append(msg.get('role', 'unknown'))
    
    from collections import Counter
    role_counter = Counter(all_roles)
    
    for role, count in role_counter.most_common():
        logger.info(f"  - {role}: {count} ({count/len(all_roles)*100:.1f}%)")


def main():
    """主函数"""
    dataset_path = "/data1/Agentic_LLM-search/datasets/nq_search/test_sample_128.parquet"
    
    logger.info(f"\n开始详细检查数据集...")
    inspect_dataset_detailed(dataset_path, num_samples=5)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"检查完成!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

