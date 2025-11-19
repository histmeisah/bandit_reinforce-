"""
同步offline wandb到线上 (Windows版本)
直接运行: python sync_wandb_windows.py
"""

import subprocess
import sys
import os
from pathlib import Path

# ========== 配置区 ==========
# 修改这里的路径即可
OFFLINE_PATH = r"E:\code_project\python_code\local_roll_dev\roll_dev\experiments\nq_search_replay\output\offline-run-20251117_055652-obz3vapc"

# WandB配置（从simple_sync.sh提取）
WANDB_API_KEY = "5d830c409e2aa7dff34c333a2f79798a877bfc7b"
WANDB_PROJECT = "roll_async_env"
ENTITY = ""  # 留空使用个人账户
# ============================


def sync():
    path = Path(OFFLINE_PATH)

    if not path.exists():
        print(f"错误：路径不存在 {path}")
        return False

    # 设置环境变量
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    # 构建命令
    cmd = [sys.executable, "-m", "wandb", "sync", str(path)]

    if WANDB_PROJECT:
        cmd.extend(["--project", WANDB_PROJECT])
    if ENTITY:
        cmd.extend(["--entity", ENTITY])

    print(f"同步: {path.name}")
    print(f"项目: {WANDB_PROJECT}")

    # 执行
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ 同步成功!")
        print(result.stdout)
        print(f"\n请访问 https://wandb.ai 查看项目 {WANDB_PROJECT}")
        return True
    else:
        print(f"❌ 失败: {result.stderr}")
        return False


if __name__ == "__main__":
    sync()