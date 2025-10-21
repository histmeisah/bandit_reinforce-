#!/usr/bin/env python3
"""
Test script to validate Jidi training configuration before running full training
"""

import sys
import os
from pathlib import Path

# Add ROLL project root to path
roll_root = Path(__file__).parent.parent.parent / "ROLL"
sys.path.insert(0, str(roll_root))

# Set environment variables
os.environ['ROLL_PATH'] = str(roll_root)
os.environ['TRAINING_TIMESTAMP'] = '20250118_test'

def test_jidi_environments():
    """Test that all Jidi environments are properly registered and can be created"""
    print("🧪 Testing Jidi Environment Registration")
    print("=" * 50)
    
    try:
        from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
        
        jidi_envs = ['jidi_cliffwalking', 'jidi_gridworld', 'jidi_minigrid', 'jidi_sokoban']
        
        print("Checking environment registration:")
        for env_name in jidi_envs:
            if env_name in REGISTERED_ENVS:
                print(f"  ✅ {env_name} - Environment class registered")
            else:
                print(f"  ❌ {env_name} - Environment class NOT found")
                return False
                
            if env_name in REGISTERED_ENV_CONFIGS:
                print(f"  ✅ {env_name} - Config class registered")
            else:
                print(f"  ❌ {env_name} - Config class NOT found")
                return False
        
        print("\n🔧 Testing environment creation:")
        for env_name in jidi_envs:
            try:
                config_class = REGISTERED_ENV_CONFIGS[env_name]
                env_class = REGISTERED_ENVS[env_name]
                
                config = config_class(max_steps=5)
                env = env_class(config)
                
                # Test reset
                obs, info = env.reset()
                print(f"  ✅ {env_name} - Reset successful")
                
                # Test get_all_actions
                actions = env.get_all_actions()
                print(f"  ✅ {env_name} - Actions: {actions}")
                
                env.close()
                
            except Exception as e:
                print(f"  ❌ {env_name} - Creation failed: {e}")
                return False
        
        print("\n🎉 All Jidi environments are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_config_loading():
    """Test that the training configuration can be loaded properly"""
    print("\n🔧 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        
        config_dir = Path(__file__).parent
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="jidi_environments_training")
            
            print("Configuration loaded successfully!")
            print(f"  - Experiment name: {cfg.exp_name}")
            print(f"  - Max steps: {cfg.max_steps}")
            print(f"  - Rollout batch size: {cfg.rollout_batch_size}")
            print(f"  - Number of environments: {len(cfg.custom_envs)}")
            
            print("\nCustom environments:")
            for env_name, env_config in cfg.custom_envs.items():
                print(f"  - {env_name}: {env_config.env_type}")
            
            print("\n✅ Configuration validation successful!")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    print("🚀 Jidi Training Configuration Test")
    print("=" * 60)
    
    # Test environment availability
    if not test_jidi_environments():
        print("\n❌ Environment test failed!")
        sys.exit(1)
    
    # Test configuration loading
    if not test_config_loading():
        print("\n❌ Configuration test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Ready for training!")
    print("=" * 60)
    print("\nTo start training, run:")
    print("  cd /data1/Weiyu_project/roll_dev/experiments/jidi_environments_training")
    print("  ./run_jidi_training.sh")

if __name__ == "__main__":
    main()
