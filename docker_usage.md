# Docker环境配置指南

本文档详细记录了在高性能服务器上配置Docker环境用于大型语言模型(LLM)训练的完整过程。

## 服务器配置

- **内存**: 1TB (1055GB)
- **CPU**: 128核心 Intel Xeon Platinum 8378A @ 3.00GHz
- **存储**: 
  - 根分区: 60GB (LVM)
  - /data1: 7TB NVMe
  - /data: 7TB NVMe
- **GPU**: 支持CUDA

## 1. Docker安装

```bash
# 更新系统包
sudo apt update

# 安装Docker
sudo apt install docker.io

# 启动并启用Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 验证安装
sudo systemctl status docker
docker --version
```

## 2. 解决存储空间问题

由于根分区空间有限(60GB)，需要将Docker数据目录迁移到大容量磁盘。

### 2.1 停止Docker服务

```bash
sudo systemctl stop docker
sudo systemctl stop docker.socket
sudo systemctl stop containerd
```

### 2.2 迁移Docker数据目录

```bash
# 创建新的Docker数据目录
sudo mkdir -p /data1/docker

# 移动现有Docker数据(如果有)
sudo mv /var/lib/docker/* /data1/docker/ 2>/dev/null || true

# 配置Docker使用新目录
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json << EOF
{
  "data-root": "/data1/docker"
}
EOF
```

### 2.3 重启Docker服务

```bash
sudo systemctl start docker
sudo systemctl start docker.socket
sudo systemctl enable docker

# 验证配置
sudo docker info | grep "Docker Root Dir"
# 应显示: Docker Root Dir: /data1/docker
```

## 3. 拉取预构建镜像

ROLL项目提供了预构建的Docker镜像，包含PyTorch和SGlang环境。

```bash
# 拉取torch2.6.0 + SGlang0.4.6镜像
docker pull roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046

# 验证镜像下载
docker images | grep roll
```

**镜像大小**: 约30GB，下载时间取决于网络速度。

## 4. 创建和运行容器

### 4.1 高性能配置

针对大型LLM训练，分配大部分服务器资源给容器：

```bash
docker run -d --name pytorch-sglang \
  --gpus all \
  --shm-size=256g \
  --memory=950g \
  --cpus=126 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  --network=host \
  --privileged \
  -v /data1:/data1 \
  -v /data:/data \
  --restart unless-stopped \
  roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046 \
  sleep infinity
```

### 4.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--shm-size` | 256g | 共享内存256GB，用于NCCL通信和大模型训练 |
| `--memory` | 950g | 限制容器最大使用950GB内存 |
| `--cpus` | 126 | 分配126个CPU核心 |
| `--ulimit memlock` | -1 | 无限制内存锁定，GPU内存管理需要 |
| `--ulimit stack` | 67108864 | 增加栈大小 |
| `--ipc=host` | - | 使用主机IPC，提升性能 |
| `--network=host` | - | 使用主机网络，减少开销 |
| `--privileged` | - | 完全权限访问 |
| `--restart` | unless-stopped | 自动重启策略 |

### 4.3 挂载目录

- `/data1:/data1` - 项目数据目录
- `/data:/data` - 额外数据目录

## 5. 容器管理

### 5.1 进入容器

```bash
# 进入运行中的容器
docker exec -it roll_vllm /bin/bash
```

### 5.2 常用管理命令

```bash
# 查看容器状态
docker ps -a

# 启动停止的容器
docker start pytorch-sglang

# 停止容器
docker stop pytorch-sglang

# 查看容器日志
docker logs pytorch-sglang

# 查看容器资源使用
docker stats pytorch-sglang
```

## 6. 容器内环境配置

### 6.1 安装必要工具

```bash
# 进入容器后执行
apt update
apt install -y tmux htop nvtop vim git curl wget

# 可选: 安装其他开发工具
apt install -y tree less nano
```

### 6.2 创建tmux会话

```bash
# 创建新的tmux会话
tmux new-session -s llm-training

# tmux常用快捷键
# Ctrl+b, d - 分离会话
# tmux attach -t llm-training - 重新连接会话
# Ctrl+b, c - 创建新窗口
# Ctrl+b, % - 垂直分割窗口
```

## 7. 环境验证

### 7.1 验证系统资源

```bash
# 检查内存分配
free -h

# 检查共享内存
df -h /dev/shm

# 检查CPU核心数
nproc

# 检查GPU
nvidia-smi
```

### 7.2 验证Python环境

```bash
# 检查Python版本
python --version

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 检查SGlang
python -c "import sglang; print('SGlang installed successfully')"
```

### 7.3 预期输出

- **内存**: 接近950GB可用
- **共享内存**: 256GB
- **CPU**: 126个核心
- **GPU**: 显示所有可用GPU
- **PyTorch**: 版本2.6.0，CUDA支持
- **SGlang**: 成功导入

## 8. 故障排除

### 8.1 常见问题

**问题1**: NCCL共享内存错误
```
Error while creating shared memory segment /dev/shm/nccl-xxx
```

**解决**: 增加`--shm-size`参数，建议设置为128g或更大。

**问题2**: 内存不足
```
no space left on device
```

**解决**: 
1. 清理Docker缓存: `docker system prune -a`
2. 迁移Docker数据目录到大容量磁盘

**问题3**: GPU访问问题

**解决**: 
1. 安装nvidia-docker2
2. 确保使用`--gpus all`参数

### 8.2 性能优化

1. **使用主机网络**: `--network=host`
2. **使用主机IPC**: `--ipc=host` 
3. **增加共享内存**: `--shm-size=256g`
4. **优化ulimit设置**: `--ulimit memlock=-1`

## 9. 备用镜像

ROLL项目提供多个预构建镜像:

```bash
# torch2.6.0 + SGlang0.4.6
roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046

# torch2.6.0 + vLLM0.8.4
roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084

# torch2.5.1 + SGlang0.4.3
roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-sglang043

# torch2.5.1 + vLLM0.7.3
roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-vllm073
```

## 10. 注意事项

1. **资源监控**: 定期监控资源使用情况，避免OOM
2. **数据备份**: 重要数据应备份到持久化存储
3. **安全考虑**: 生产环境避免使用`--privileged`
4. **网络安全**: 注意端口映射和防火墙配置
5. **日志管理**: 定期清理容器日志避免占用过多空间

## 11. 联系和支持

- ROLL项目文档: 查看项目README
- Docker官方文档: https://docs.docker.com
- 遇到问题时，检查容器日志: `docker logs pytorch-sglang`

---

**最后更新**: 2025年8月6日
**适用环境**: Ubuntu/Debian系统，高性能服务器