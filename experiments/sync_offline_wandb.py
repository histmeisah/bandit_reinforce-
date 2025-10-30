"""
同步offline wandb数据到线上

使用方法：
python sync_offline_wandb.py --path E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/offline-run-20251022_084453-ulph8zyt

或者使用简短路径：
python sync_offline_wandb.py --path nq_search_replay/output/offline-run-20251022_084453-ulph8zyt
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

# 添加wandb
try:
    import wandb
except ImportError:
    print("错误: 未安装wandb")
    print("请运行: pip install wandb")
    sys.exit(1)


class OfflineWandbSync:
    """同步offline wandb数据到线上"""

    def __init__(self, offline_path: str, project: str = None, entity: str = None):
        """
        初始化同步器

        Args:
            offline_path: offline wandb目录路径
            project: wandb项目名（可选，会尝试从offline数据中读取）
            entity: wandb实体名（可选）
        """
        self.offline_path = Path(offline_path)

        # 验证路径
        if not self.offline_path.exists():
            raise ValueError(f"路径不存在: {offline_path}")

        # 查找.wandb文件
        wandb_files = list(self.offline_path.glob("*.wandb"))
        if not wandb_files:
            raise ValueError(f"未找到.wandb文件在: {offline_path}")

        self.wandb_file = wandb_files[0]
        self.run_id = self.wandb_file.stem.replace("run-", "")

        print(f"找到offline run: {self.run_id}")
        print(f"Wandb文件: {self.wandb_file.name}")

        # 读取配置
        self.config = self._read_config()
        self.project = project or self.config.get("project", "offline-sync")
        self.entity = entity or self.config.get("entity")

        print(f"项目: {self.project}")
        if self.entity:
            print(f"实体: {self.entity}")

    def _read_config(self) -> Dict[str, Any]:
        """尝试读取offline配置"""
        config = {}

        # 尝试从files目录读取配置
        config_file = self.offline_path / "files" / "config.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            except:
                pass

        # 尝试从wandb文件名提取信息
        # offline-run-20251022_084453-ulph8zyt 格式
        dir_name = self.offline_path.name
        if dir_name.startswith("offline-run-"):
            parts = dir_name.split("-")
            if len(parts) >= 3:
                # 提取时间戳
                timestamp_str = parts[2]  # 20251022_084453
                config["offline_timestamp"] = timestamp_str

        return config

    def sync(self, force: bool = False) -> bool:
        """
        同步数据到wandb

        Args:
            force: 是否强制重新同步（即使已存在）

        Returns:
            是否成功
        """
        print("\n" + "="*50)
        print("开始同步到WandB")
        print("="*50)

        try:
            # 方法1：使用wandb sync命令（推荐）
            print("\n方法1: 使用wandb sync命令...")
            return self._sync_with_command()

        except Exception as e:
            print(f"\n方法1失败: {e}")
            print("\n尝试方法2: 手动上传...")
            return self._sync_manually()

    def _sync_with_command(self) -> bool:
        """使用wandb sync命令同步"""
        import subprocess

        # 构建命令
        cmd = [
            sys.executable, "-m", "wandb", "sync",
            str(self.offline_path)
        ]

        if self.project:
            cmd.extend(["--project", self.project])

        if self.entity:
            cmd.extend(["--entity", self.entity])

        print(f"执行命令: {' '.join(cmd)}")

        # 执行命令
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.offline_path.parent)
            )

            if result.returncode == 0:
                print("✅ 同步成功!")
                print(result.stdout)
                return True
            else:
                print(f"❌ 同步失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ 执行命令失败: {e}")
            return False

    def _sync_manually(self) -> bool:
        """手动同步（备用方法）"""
        try:
            # 初始化wandb
            print(f"初始化WandB连接...")

            # 设置离线目录
            os.environ["WANDB_DIR"] = str(self.offline_path.parent)
            os.environ["WANDB_MODE"] = "online"  # 强制在线模式

            # 初始化run
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=self.run_id,  # 使用相同的run_id
                resume="allow",  # 允许恢复
                force=True  # 强制同步
            )

            print(f"✅ Run初始化成功: {run.url}")

            # 上传文件
            files_dir = self.offline_path / "files"
            if files_dir.exists():
                print(f"上传文件目录...")
                for file_path in files_dir.glob("*"):
                    if file_path.is_file():
                        print(f"  上传: {file_path.name}")
                        wandb.save(str(file_path), base_path=str(files_dir))

            # 完成
            run.finish()
            print("✅ 手动同步完成!")
            return True

        except Exception as e:
            print(f"❌ 手动同步失败: {e}")
            return False

    def check_status(self) -> Dict[str, Any]:
        """检查同步状态"""
        status = {
            "offline_path": str(self.offline_path),
            "run_id": self.run_id,
            "project": self.project,
            "entity": self.entity,
            "files": []
        }

        # 检查文件
        files_dir = self.offline_path / "files"
        if files_dir.exists():
            for f in files_dir.glob("*"):
                if f.is_file():
                    status["files"].append({
                        "name": f.name,
                        "size": f.stat().st_size
                    })

        # 检查wandb文件大小
        status["wandb_file_size"] = self.wandb_file.stat().st_size

        return status


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='同步offline WandB数据到线上',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用完整路径
  python sync_offline_wandb.py --path E:/code_project/python_code/local_roll_dev/roll_dev/experiments/nq_search_replay/output/offline-run-20251022_084453-ulph8zyt

  # 使用相对路径
  python sync_offline_wandb.py --path nq_search_replay/output/offline-run-20251022_084453-ulph8zyt

  # 指定项目名
  python sync_offline_wandb.py --path <path> --project my-project

  # 指定entity
  python sync_offline_wandb.py --path <path> --entity my-team
        """
    )

    parser.add_argument('--path', '-p', required=True,
                       help='Offline wandb目录路径')
    parser.add_argument('--project', default=None,
                       help='WandB项目名（可选）')
    parser.add_argument('--entity', default=None,
                       help='WandB实体名（可选）')
    parser.add_argument('--force', action='store_true',
                       help='强制重新同步')
    parser.add_argument('--status', action='store_true',
                       help='仅显示状态，不同步')

    args = parser.parse_args()

    # 处理路径
    path = Path(args.path)
    if not path.is_absolute():
        # 相对路径，假设相对于experiments目录
        base_dir = Path(__file__).parent
        path = base_dir / path

    # 规范化Windows路径
    path = Path(str(path).replace('/', '\\'))

    try:
        # 创建同步器
        syncer = OfflineWandbSync(
            str(path),
            project=args.project,
            entity=args.entity
        )

        if args.status:
            # 仅显示状态
            status = syncer.check_status()
            print("\n状态信息:")
            print(json.dumps(status, indent=2, ensure_ascii=False))
        else:
            # 执行同步
            success = syncer.sync(force=args.force)

            if success:
                print("\n✅ 同步成功完成!")
                print(f"请在WandB网站查看: https://wandb.ai/{syncer.entity or 'your-entity'}/{syncer.project}")
                sys.exit(0)
            else:
                print("\n❌ 同步失败，请检查错误信息")
                sys.exit(1)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()