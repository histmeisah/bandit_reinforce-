#!/usr/bin/env python3
"""
WandB实时同步脚本 - VERL项目专用版 - 简化版
用于在跳板机上创建tmux会话，每小时同步训练机的wandb日志到云端
支持代理设置以便在中国网络环境下使用
特点：只同步.wandb核心文件，利用wandb原生增量同步能力
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json

class WandBSyncHandler(FileSystemEventHandler):
    """监控wandb目录变化的处理器"""
    
    def __init__(self, wandb_dir, sync_interval=3600, proxy_host=None, proxy_port=None, 
                 wandb_project=None, wandb_api_key=None):
        self.wandb_dir = Path(wandb_dir)
        self.sync_interval = sync_interval  # 修改默认间隔到1小时
        self.last_sync = 0
        self.logger = logging.getLogger(__name__)
        self.proxy_env = self._setup_proxy_env(proxy_host, proxy_port)
        self.proxy_env = self._setup_wandb_env(self.proxy_env, wandb_project, wandb_api_key)
        
        # 同步状态跟踪
        self.sync_state_file = self.wandb_dir.parent / '.wandb_sync_state.json'
        self.synced_runs = self._load_sync_state()
        
    def _setup_proxy_env(self, proxy_host, proxy_port):
        """设置代理环境变量"""
        env = os.environ.copy()
        
        if proxy_host and proxy_port:
            proxy_url = f"http://{proxy_host}:{proxy_port}"
            env['HTTP_PROXY'] = proxy_url
            env['HTTPS_PROXY'] = proxy_url
            env['http_proxy'] = proxy_url
            env['https_proxy'] = proxy_url
            # 设置不使用代理的地址
            env['NO_PROXY'] = 'localhost,127.0.0.1,::1'
            env['no_proxy'] = 'localhost,127.0.0.1,::1'
            
            self.logger.info(f"已设置代理: {proxy_url}")
        else:
            self.logger.info("未设置代理")
            
        return env
    
    def _setup_wandb_env(self, env, wandb_project=None, wandb_api_key=None):
        """设置wandb环境变量"""
        if wandb_project:
            env['WANDB_PROJECT'] = wandb_project
            self.logger.info(f"设置WANDB_PROJECT: {wandb_project}")
        
        if wandb_api_key:
            env['WANDB_API_KEY'] = wandb_api_key
            self.logger.info("设置WANDB_API_KEY: ****")
        
        return env
    
    def _load_sync_state(self):
        """加载同步状态"""
        try:
            if self.sync_state_file.exists():
                with open(self.sync_state_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"加载同步状态失败: {e}")
            return {}
    
    def _save_sync_state(self):
        """保存同步状态"""
        try:
            with open(self.sync_state_file, 'w') as f:
                json.dump(self.synced_runs, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存同步状态失败: {e}")
    
    def _should_sync_run(self, run_dir):
        """检查运行是否需要同步 - 简化逻辑"""
        run_name = run_dir.name
        current_time = time.time()
        
        # 检查上次同步时间，如果距离上次同步不到1小时，跳过
        if run_name in self.synced_runs:
            last_sync_time = self.synced_runs[run_name].get('synced_at', 0)
            if current_time - last_sync_time < self.sync_interval:
                return False
        
        return True
    
    def _mark_run_synced(self, run_dir, success=True):
        """标记运行已同步 - 简化逻辑"""
        run_name = run_dir.name
        
        self.synced_runs[run_name] = {
            'synced_at': time.time(),
            'success': success
        }
        self._save_sync_state()
    
    def _exclude_large_files(self, run_dir):
        """排除不需要同步的较大文件，只保留.wandb等核心文件"""
        try:
            files_dir = run_dir / "files"
            if not files_dir.exists():
                return
            
            # 需要排除的文件列表（非.wandb的其他大文件）
            exclude_files = [
                "output.log",           # VERL训练输出日志
                "*.log",               # 所有日志文件
                "*.tmp",               # 临时文件
                "*.cache",             # 缓存文件
                "checkpoint_*.pt",     # 大型检查点文件
                "*.bin",               # 二进制文件
                "*.txt",               # 文本日志文件
            ]
            
            excluded_count = 0
            excluded_size = 0
            
            for file_path in files_dir.rglob("*"):
                if file_path.is_file():
                    file_name = file_path.name
                    
                    # 如果是.wandb文件，保留不排除
                    if file_name.endswith('.wandb'):
                        continue
                    
                    # 检查是否需要排除
                    should_exclude = False
                    for pattern in exclude_files:
                        if pattern.startswith("*"):
                            if file_name.endswith(pattern[1:]):
                                should_exclude = True
                                break
                        elif file_name == pattern:
                            should_exclude = True
                            break
                    
                    if should_exclude:
                        try:
                            file_size = file_path.stat().st_size
                            if file_size > 1 * 1024 * 1024:  # 大于1MB的文件
                                backup_path = file_path.with_suffix(file_path.suffix + '.excluded')
                                if not backup_path.exists():
                                    file_path.rename(backup_path)
                                    excluded_count += 1
                                    excluded_size += file_size
                                    self.logger.info(f"排除非核心文件: {file_name} ({file_size/1024/1024:.2f}MB)")
                        except Exception as e:
                            self.logger.warning(f"处理文件 {file_name} 时出错: {e}")
            
            if excluded_count > 0:
                self.logger.info(f"本次排除了 {excluded_count} 个非核心文件，节省 {excluded_size/1024/1024:.2f}MB 流量")
                
        except Exception as e:
            self.logger.error(f"排除大文件时出错: {e}")
        
    def sync_wandb(self):
        """执行wandb同步 - 简化版"""
        current_time = time.time()
        if current_time - self.last_sync < self.sync_interval:
            return
            
        try:
            # 切换到wandb目录
            os.chdir(self.wandb_dir)
            
            # 查找所有的run目录
            if self.wandb_dir.name.startswith('offline-run-'):
                # 如果wandb_dir本身就是一个run目录
                run_dirs = [self.wandb_dir]
            else:
                # 查找目录下的所有run
                run_dirs = [d for d in self.wandb_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('offline-run-')]
            
            if not run_dirs:
                self.logger.debug("未找到需要同步的离线运行目录")
                self.last_sync = current_time
                return
            
            # 过滤出需要同步的运行 - 简化逻辑
            pending_runs = []
            for run_dir in run_dirs:
                if self._should_sync_run(run_dir):
                    pending_runs.append(run_dir)
                else:
                    self.logger.debug(f"跳过距离上次同步不足1小时的运行: {run_dir.name}")
            
            if not pending_runs:
                self.logger.info("所有离线运行都在1小时内同步过，跳过本次同步")
                self.last_sync = current_time
                return
            
            self.logger.info(f"找到 {len(pending_runs)} 个需要同步的离线运行（共{len(run_dirs)}个）")
            
            for run_dir in pending_runs:
                self.logger.info(f"正在同步运行: {run_dir.name}")
                
                # 检查并排除不需要的文件，保留.wandb核心文件
                self._exclude_large_files(run_dir)
                
                # 使用wandb sync同步离线运行，添加超时保护
                cmd = ['wandb', 'sync', str(run_dir)]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          env=self.proxy_env, timeout=300)  # 5分钟超时
                    
                    if result.returncode == 0:
                        self.logger.info(f"成功同步 {run_dir.name}")
                        self._mark_run_synced(run_dir, success=True)
                        if result.stdout:
                            self.logger.debug(f"同步输出: {result.stdout}")
                    else:
                        self.logger.error(f"同步失败 {run_dir.name}: {result.stderr}")
                        self._mark_run_synced(run_dir, success=False)
                        if result.stdout:
                            self.logger.error(f"标准输出: {result.stdout}")
                            
                except subprocess.TimeoutExpired:
                    self.logger.error(f"同步 {run_dir.name} 超时（5分钟）")
                    self._mark_run_synced(run_dir, success=False)
                except Exception as e:
                    self.logger.error(f"同步 {run_dir.name} 时发生异常: {e}")
                    self._mark_run_synced(run_dir, success=False)
                    
            self.last_sync = current_time
            successful_runs = len([r for r in self.synced_runs.values() if r.get('success', False)])
            self.logger.info(f"本轮同步完成，历史成功同步运行总数: {successful_runs}")
            self.logger.info(f"下次检查同步时间: {time.strftime('%H:%M:%S', time.localtime(current_time + self.sync_interval))}")
            
        except Exception as e:
            self.logger.error(f"同步过程中出错: {e}")
            import traceback
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
    
    def on_modified(self, event):
        """文件修改时触发 - 只关注.wandb文件"""
        if not event.is_directory and event.src_path.endswith('.wandb'):
            self.logger.debug(f"检测到.wandb文件变化: {event.src_path}")
            self.sync_wandb()
    
    def on_created(self, event):
        """文件创建时触发 - 只关注.wandb文件"""
        if not event.is_directory and event.src_path.endswith('.wandb'):
            self.logger.debug(f"检测到新的.wandb文件: {event.src_path}")
            self.sync_wandb()


def setup_logging(log_level=logging.INFO, log_dir=None):
    """设置日志"""
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # 确定日志文件路径
    if log_dir:
        log_file_path = Path(log_dir) / 'wandb_sync.log'
    else:
        # 使用脚本所在目录
        script_dir = Path(__file__).parent.absolute()
        log_file_path = script_dir / 'wandb_sync.log'
    
    # 确保日志目录存在
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建文件处理器，确保实时写入
    file_handler = logging.FileHandler(str(log_file_path), encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加新处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 强制刷新日志
    logging.getLogger().info(f"日志系统初始化完成，日志文件: {log_file_path}")


def create_tmux_session(session_name, wandb_dir, sync_interval, proxy_host=None, proxy_port=None,
                       wandb_project=None, wandb_api_key=None, conda_bashrc=None, conda_env=None):
    """创建tmux会话并启动监控"""
    
    # 检查tmux会话是否已存在
    check_cmd = ['tmux', 'has-session', '-t', session_name]
    result = subprocess.run(check_cmd, capture_output=True)
    
    session_exists = (result.returncode == 0)
    
    if not session_exists:
        # 创建新的tmux会话
        create_cmd = [
            'tmux', 'new-session', '-d', '-s', session_name,
            '-c', str(wandb_dir.parent)
        ]
        
        subprocess.run(create_cmd)
        print(f"已创建tmux会话: {session_name}")
    else:
        print(f"Tmux会话 '{session_name}' 已存在，将在其中启动同步程序")
        # 清理会话中可能存在的旧进程
        cleanup_cmd = ['tmux', 'send-keys', '-t', session_name, 'C-c', 'Enter']
        subprocess.run(cleanup_cmd, capture_output=True)
        time.sleep(1)  # 等待清理完成
    
    # 在tmux会话中设置conda环境
    if conda_bashrc and conda_env:
        setup_conda_cmd = [
            'tmux', 'send-keys', '-t', session_name,
            f'source {conda_bashrc}',
            'Enter'
        ]
        subprocess.run(setup_conda_cmd)
        time.sleep(1)  # 等待源码执行完成
        
        activate_conda_cmd = [
            'tmux', 'send-keys', '-t', session_name,
            f'conda activate {conda_env}',
            'Enter'
        ]
        subprocess.run(activate_conda_cmd)
        time.sleep(2)  # 等待conda环境激活
        print(f"已在tmux会话中设置conda环境: {conda_env}")
    
    # 在tmux会话中运行同步脚本
    script_path = Path(__file__).absolute()
    cmd_parts = [f'python3 {script_path}', '--monitor', f'--wandb-dir {wandb_dir}', f'--sync-interval {sync_interval}']
    
    if proxy_host and proxy_port:
        cmd_parts.extend([f'--proxy-host {proxy_host}', f'--proxy-port {proxy_port}'])
    
    if wandb_project:
        cmd_parts.append(f'--wandb-project {wandb_project}')
    
    if wandb_api_key:
        cmd_parts.append(f'--wandb-api-key {wandb_api_key}')
    
    sync_cmd = [
        'tmux', 'send-keys', '-t', session_name,
        ' '.join(cmd_parts),
        'Enter'
    ]
    
    subprocess.run(sync_cmd)
    
    if session_exists:
        print(f"已在现有tmux会话中重新启动监控")
    else:
        print(f"已在新tmux会话中启动监控")
    
    return True


def monitor_wandb_directory(wandb_dir, sync_interval, proxy_host=None, proxy_port=None,
                          wandb_project=None, wandb_api_key=None):
    """监控wandb目录变化"""
    
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        raise ValueError(f"WandB目录不存在: {wandb_path}")
    
    print(f"开始监控目录: {wandb_path}")
    print(f"同步间隔: {sync_interval}秒")
    
    # 设置文件监控
    event_handler = WandBSyncHandler(wandb_path, sync_interval, proxy_host, proxy_port, 
                                   wandb_project, wandb_api_key)
    observer = Observer()
    observer.schedule(event_handler, str(wandb_path), recursive=True)
    
    # 启动监控
    observer.start()
    logger = logging.getLogger(__name__)
    
    try:
        # 初始同步
        logger.info("执行初始同步...")
        event_handler.sync_wandb()
        
        # 持续监控 - 减少强制同步频率
        logger.info("开始持续监控循环...")
        iteration_count = 0
        
        while True:
            try:
                time.sleep(30)  # 增加睡眠时间
                iteration_count += 1
                
                # 每20次循环（约10分钟）输出一次状态
                if iteration_count % 20 == 0:
                    logger.info(f"监控运行中... (第{iteration_count}次检查)")
                    # 检查observer是否还在运行
                    if not observer.is_alive():
                        logger.warning("Observer已停止，尝试重启...")
                        observer.stop()
                        observer.join()
                        observer = Observer()
                        observer.schedule(event_handler, str(wandb_path), recursive=True)
                        observer.start()
                        logger.info("Observer已重启")
                
                # 每240次循环（约120分钟）强制执行一次同步 - 进一步减少频率
                if iteration_count % 240 == 0:
                    logger.info("执行定期强制同步检查...")
                    try:
                        event_handler.sync_wandb()
                    except Exception as sync_error:
                        logger.error(f"定期同步失败: {sync_error}")
                        
            except Exception as loop_error:
                logger.error(f"监控循环中出现错误: {loop_error}")
                # 继续运行，不要因为单个错误而退出
                continue
                
    except KeyboardInterrupt:
        print("\n停止监控...")
        logger.info("收到键盘中断信号，正在停止监控...")
        observer.stop()
    except Exception as e:
        logger.error(f"监控过程中发生严重错误: {e}")
        print(f"\n监控过程中发生错误: {e}")
        observer.stop()
        raise
    finally:
        logger.info("正在清理资源...")
        observer.join()
        logger.info("监控已完全停止")


def main():
    parser = argparse.ArgumentParser(description='VERL WandB实时同步工具 - 简化版')
    parser.add_argument('--wandb-dir', required=True, 
                       help='WandB日志目录路径')
    parser.add_argument('--session-name', default='verl-wandb-sync',
                       help='Tmux会话名称 (默认: verl-wandb-sync)')
    parser.add_argument('--sync-interval', type=int, default=3600,
                       help='同步间隔秒数 (默认: 3600秒/1小时)')
    parser.add_argument('--proxy-host', default='127.0.0.1',
                       help='代理服务器地址 (默认: 127.0.0.1)')
    parser.add_argument('--proxy-port', type=int, default=7890,
                       help='代理服务器端口 (默认: 7890)')
    parser.add_argument('--no-proxy', action='store_true',
                       help='不使用代理')
    parser.add_argument('--wandb-project', default='verl_tool_0603_project',
                       help='WandB项目名称 (默认: verl_tool_0603_project)')
    parser.add_argument('--wandb-api-key', default='5d830c409e2aa7dff34c333a2f79798a877bfc7b',
                       help='WandB API密钥')
    parser.add_argument('--conda-bashrc', 
                       help='Conda bashrc文件路径')
    parser.add_argument('--conda-env', 
                       help='Conda环境名称')
    parser.add_argument('--monitor', action='store_true',
                       help='直接启动监控模式(不创建tmux)')
    parser.add_argument('--verbose', action='store_true',
                       help='详细日志输出')
    parser.add_argument('--reset-sync-state', action='store_true',
                       help='重置同步状态（重新同步所有runs）')
    
    args = parser.parse_args()
    
    # 先解析wandb目录路径
    wandb_dir = Path(args.wandb_dir)
    
    # 重置同步状态
    if args.reset_sync_state:
        sync_state_file = wandb_dir.parent / '.wandb_sync_state.json'
        if sync_state_file.exists():
            sync_state_file.unlink()
            print("✅ 已重置同步状态")
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # 将日志文件保存到wandb目录的父目录，确保路径安全
    log_dir = wandb_dir.parent if wandb_dir.parent.exists() else Path.cwd()
    setup_logging(log_level, log_dir)
    
    # 设置代理参数
    proxy_host = None if args.no_proxy else args.proxy_host
    proxy_port = None if args.no_proxy else args.proxy_port
    
    if args.monitor:
        # 直接监控模式
        logger = logging.getLogger(__name__)
        logger.info(f"启动VERL WandB监控服务 - 简化版")
        logger.info(f"监控目录: {wandb_dir}")
        logger.info(f"同步间隔: {args.sync_interval}秒")
        logger.info(f"代理设置: {'启用' if proxy_host and proxy_port else '禁用'}")
        if proxy_host and proxy_port:
            logger.info(f"代理地址: {proxy_host}:{proxy_port}")
        logger.info(f"WandB项目: {args.wandb_project}")
        logger.info(f"详细日志: {'启用' if args.verbose else '禁用'}")
        logger.info("🔧 优化功能: 只同步.wandb文件、每小时同步一次、利用wandb增量同步")
        
        monitor_wandb_directory(wandb_dir, args.sync_interval, proxy_host, proxy_port,
                              args.wandb_project, args.wandb_api_key)
    else:
        # 创建tmux会话模式
        success = create_tmux_session(args.session_name, wandb_dir, args.sync_interval, 
                                    proxy_host, proxy_port, args.wandb_project, args.wandb_api_key,
                                    args.conda_bashrc, args.conda_env)
        if success:
            proxy_info = f" (使用代理: {proxy_host}:{proxy_port})" if proxy_host and proxy_port else " (无代理)"
            print(f"\n✅ 已启动VERL WandB同步服务 - 简化版{proxy_info}")
            print(f"📊 WandB项目: {args.wandb_project}")
            print(f"⏱️ 同步间隔: {args.sync_interval}秒")
            print(f"🔧 优化功能: 只同步.wandb文件、每小时同步一次、利用wandb增量同步")
            print(f"\n使用以下命令查看同步状态:")
            print(f"tmux attach -t {args.session_name}")
            print(f"\n使用以下命令停止同步:")
            print(f"tmux kill-session -t {args.session_name}")
            print(f"\n重置同步状态（强制重新同步所有runs）:")
            print(f"python3 {Path(__file__).name} --wandb-dir {wandb_dir} --reset-sync-state")
            
            # 确定日志文件路径
            log_dir = wandb_dir.parent if wandb_dir.parent.exists() else Path.cwd()
            log_file_path = log_dir / 'wandb_sync.log'
            sync_state_file = wandb_dir.parent / '.wandb_sync_state.json'
            print(f"\n同步日志文件: {log_file_path}")
            print(f"同步状态文件: {sync_state_file}")


if __name__ == '__main__':
    main()