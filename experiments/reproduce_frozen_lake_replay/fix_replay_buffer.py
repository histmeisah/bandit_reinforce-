#!/usr/bin/env python3
"""
Fix script to switch from distributed buffer to working TensorDict buffer
and restore off-policy monitoring.

Usage:
    python fix_replay_buffer.py
"""

def create_pipeline_patch():
    """Generate a patch for agentic_pipeline.py to fix replay buffer issues"""

    patch_content = """
# Patch for agentic_pipeline.py - Line 151-172
# Replace the distributed buffer initialization with working TensorDict buffer

            # Create replay buffer based on manager type
            from roll.agentic.replay_buffer import (
                create_replay_buffer,
                detect_manager_type_from_config
            )

            manager_type = detect_manager_type_from_config(self.pipeline_config)
            batch_size = self.pipeline_config.rollout_batch_size if rb_cfg.use_rollout_batch_size else rb_cfg.minibatch_size

            # Use TensorDict implementation which is fully working
            self.replay_buffer = create_replay_buffer(
                manager_type=manager_type,
                capacity=rb_cfg.capacity,
                batch_size=batch_size,
                seed=42,
                use_tensordict=True,  # Force TensorDict implementation
                distributed=False     # Disable broken distributed implementation
            )

            logger.info(f"Initialized {self.replay_buffer.__class__.__name__} for {manager_type} env_manager")
            logger.info(f"Replay Buffer Config: capacity={rb_cfg.capacity}, batch_size={batch_size}, sample_method={rb_cfg.sample_method}")
"""

    return patch_content

def create_offpolicy_monitoring_patch():
    """Generate a patch to add off-policy monitoring back"""

    patch_content = """
# Add this after line 488 in agentic_pipeline.py (after buffer_stats calculation)

                        # Off-policy monitoring for replay buffer
                        if self.replay_buffer is not None and mb.batch.get("behavior_log_probs") is not None:
                            try:
                                # Recompute current policy log probs for off-policy ratio
                                current_lp_refs = self.actor_train.compute_log_probs(mb, blocking=False)
                                current_lp = DataProto.materialize_concat(data_refs=current_lp_refs)

                                # Calculate off-policy metrics
                                resp_mask = mb.batch["response_mask"][:, 1:].bool()
                                cur_lp = current_lp.batch["log_probs"][resp_mask]
                                old_lp = mb.batch["behavior_log_probs"][resp_mask]

                                if cur_lp.numel() > 0 and old_lp.numel() > 0:
                                    delta = cur_lp - old_lp
                                    ratio = delta.exp()

                                    metrics.update({
                                        "replay/off_policy_delta": delta.mean().item(),
                                        "replay/off_policy_ratio": ratio.mean().item(),
                                        "replay/off_policy_max_ratio": ratio.max().item(),
                                        "replay/off_policy_min_ratio": ratio.min().item(),
                                    })
                            except Exception as e:
                                logger.debug(f"Off-policy monitoring failed: {e}")
"""

    return patch_content

def main():
    print("=== Replay Buffer Fix Script ===\n")

    print("问题诊断：")
    print("1. 当前使用 DistributedReplayBufferWithFaultTolerance - 采样功能未实现")
    print("2. Off-policy监控日志已被移除")
    print()

    print("解决方案：")
    print("1. 切换到 TensorDictTrajectoryBuffer（完整实现）")
    print("2. 恢复 off-policy ratio 监控")
    print()

    print("修改文件：roll_dev/ROLL/roll/pipeline/agentic/agentic_pipeline.py")
    print()

    print("=" * 60)
    print("PATCH 1: 修复 Replay Buffer 初始化（第151-172行）")
    print("=" * 60)
    print(create_pipeline_patch())

    print("=" * 60)
    print("PATCH 2: 恢复 Off-policy 监控（第488行后添加）")
    print("=" * 60)
    print(create_offpolicy_monitoring_patch())

    print("\n应用方法：")
    print("1. 备份原文件: cp agentic_pipeline.py agentic_pipeline.py.bak")
    print("2. 手动编辑 agentic_pipeline.py 应用上述补丁")
    print("3. 或使用自动修复脚本（需要确认）")

    print("\n验证方法：")
    print("1. 检查日志是否显示 'TensorDictTrajectoryBuffer' 而非 'DistributedReplayBufferWithFaultTolerance'")
    print("2. 检查 WandB 是否有 replay/off_policy_* 指标")
    print("3. 检查 replay_buffer/total_stored 是否正常增长")

if __name__ == "__main__":
    main()