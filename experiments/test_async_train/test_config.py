#!/usr/bin/env python3
"""
配置文件测试脚本
用于验证YAML配置是否正确，以及环境是否满足要求
"""

import os
import sys
import yaml
import torch
import subprocess
from pathlib import Path

def test_gpu_availability():
    """测试GPU可用性"""
    print("=== GPU 可用性测试 ===")
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 张GPU")
    
    if gpu_count < 8:
        print(f"⚠️  警告: 需要8张GPU，但只检测到{gpu_count}张")
        return False
    
    for i in range(min(8, gpu_count)):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")
    
    return True

def test_config_file():
    """测试配置文件"""
    print("\n=== 配置文件测试 ===")
    config_path = "agent_val_frozen_lake_async_8gpus.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        
        # 检查关键配置
        key_configs = [
            ('exp_name', config.get('exp_name')),
            ('num_gpus_per_node', config.get('num_gpus_per_node')),
            ('async_generation_ratio', config.get('async_generation_ratio')),
            ('rollout_batch_size', config.get('rollout_batch_size')),
            ('max_steps', config.get('max_steps')),
        ]

        for key, value in key_configs:
            print(f"  {key}: {value}")

        # 检查异步配置
        train_env = config.get('train_env_manager', {})
        val_env = config.get('val_env_manager', {})

        print(f"\n  异步配置检查:")
        print(f"    LLM异步比例: {config.get('async_generation_ratio', 'N/A')}")
        print(f"    训练环境group_size: {train_env.get('group_size', 'N/A')}")
        print(f"    验证环境group_size: {val_env.get('group_size', 'N/A')}")

        # 异步程度评估
        async_level = "未知"
        if config.get('async_generation_ratio') == 1 and train_env.get('group_size') == 1:
            async_level = "完全异步 (推荐生产环境)"
        elif config.get('async_generation_ratio') == 1:
            async_level = "LLM异步，环境部分同步"
        elif train_env.get('group_size') == 1:
            async_level = "环境异步，LLM部分同步"
        else:
            async_level = "部分异步 (适合调试)"

        print(f"    异步程度: {async_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False

def test_directories():
    """测试目录结构"""
    print("\n=== 目录结构测试 ===")
    
    required_dirs = [
        "../../ROLL",
        "../../ROLL/examples",
        "../../ROLL/examples/config",
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} 不存在")
            return False
    
    # 创建输出目录
    output_dirs = [
        "./output/logs",
        "./output/models", 
        "./output/wandb",
        "./output/render"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建/确认目录: {dir_path}")
    
    return True

def test_python_environment():
    """测试Python环境"""
    print("\n=== Python环境测试 ===")
    
    required_packages = [
        'torch',
        'transformers', 
        'ray',
        'wandb',
        'hydra-core',
        'omegaconf',
        'dacite'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("请使用以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_roll_import():
    """测试ROLL模块导入"""
    print("\n=== ROLL模块测试 ===")
    
    # 添加ROLL路径
    roll_path = os.path.abspath("../../ROLL")
    if roll_path not in sys.path:
        sys.path.insert(0, roll_path)
    
    try:
        from roll.pipeline.agentic.agentic_config import AgenticConfig
        print("✅ AgenticConfig 导入成功")
        
        from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline  
        print("✅ AgenticPipeline 导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ ROLL模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("ROLL 8GPU Async Training 配置测试")
    print("=" * 50)
    
    tests = [
        ("GPU可用性", test_gpu_availability),
        ("配置文件", test_config_file), 
        ("目录结构", test_directories),
        ("Python环境", test_python_environment),
        ("ROLL模块", test_roll_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练。")
        print("运行命令: ./run_agentic_pipeline_frozen_lake_8gpu.sh")
    else:
        print("\n⚠️  部分测试失败，请检查上述问题后再开始训练。")
    
    return all_passed

if __name__ == "__main__":
    main()
