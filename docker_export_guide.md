# Docker容器导出指南

本文档详细说明如何将运行中的Docker容器打包导出，以便在其他环境中复用。

## 当前容器状态

- **容器名称**: `roll_vllm`
- **镜像**: `roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084`
- **状态**: 运行中 (Up 2 weeks)

## 1. 导出方法概览

Docker提供两种主要的导出方式：

| 方法 | 命令 | 优点 | 缺点 | 使用场景 |
|------|------|------|------|----------|
| **容器导出** | `docker export` | 文件小，包含当前状态 | 丢失历史层，无法回滚 | 快速备份当前环境 |
| **镜像保存** | `docker save` | 保留完整镜像信息 | 文件大，包含所有层 | 完整环境迁移 |

## 2. 方法一：导出容器 (推荐用于快速备份)

### 2.1 导出运行中的容器

```bash
# 导出当前运行的容器
docker export roll_vllm > /data1//data1/Weiyu_project/docker_backups/roll_vllm_backup_$(date +%Y%m%d_%H%M%S).tar

# 或使用压缩导出 (推荐，可显著减小文件大小)
docker export roll_vllm | gzip > /data1/Weiyu_project/docker_backups/roll_vllm_backup_$(date +%Y%m%d_%H%M%S).tar.gz
```

### 2.2 查看导出文件

```bash
# 检查导出文件大小
ls -lh /data1/roll_vllm_backup_*.tar*

# 预估文件大小 (容器导出通常比镜像小很多)
docker exec roll_vllm du -sh /
```

### 2.3 导入容器为新镜像

```bash
# 从tar文件导入为新镜像
docker import /data1/roll_vllm_backup_20250106_143000.tar.gz my-custom-pytorch:latest

# 从压缩文件导入
zcat /data1/roll_vllm_backup_20250106_143000.tar.gz | docker import - my-custom-pytorch:latest
```

## 3. 方法二：保存完整镜像

### 3.1 先提交容器为镜像

```bash
# 将当前容器状态提交为新镜像
docker commit roll_vllm my-pytorch-vllm:$(date +%Y%m%d)

# 添加更多元数据
docker commit -m "ROLL vLLM environment with custom configs" \
              -a "Your Name <your.email@example.com>" \
              roll_vllm \
              my-pytorch-vllm:$(date +%Y%m%d)
```

### 3.2 保存镜像

```bash
# 保存镜像到tar文件
docker save my-pytorch-vllm:$(date +%Y%m%d) > /data1/pytorch_vllm_image_$(date +%Y%m%d).tar

# 压缩保存 (强烈推荐)
docker save my-pytorch-vllm:$(date +%Y%m%d) | gzip > /data1/pytorch_vllm_image_$(date +%Y%m%d).tar.gz
```

### 3.3 加载镜像

```bash
# 从tar文件加载镜像
docker load < /data1/pytorch_vllm_image_20250106.tar.gz

# 或使用
docker load -i /data1/pytorch_vllm_image_20250106.tar.gz
```

## 4. 实际操作示例

### 4.1 快速导出当前环境

```bash
# 1. 进入项目目录
cd /data1/Weiyu_project

# 2. 创建导出目录
mkdir -p docker_backups

# 3. 导出容器 (压缩格式)
docker export roll_vllm | gzip > docker_backups/roll_vllm_$(date +%Y%m%d_%H%M%S).tar.gz

# 4. 检查导出结果
ls -lh docker_backups/
```

### 4.2 完整镜像备份

```bash
# 1. 提交容器为镜像
docker commit roll_vllm roll-pytorch-vllm:backup-$(date +%Y%m%d)

# 2. 保存镜像
docker save roll-pytorch-vllm:backup-$(date +%Y%m%d) | gzip > docker_backups/pytorch_vllm_full_$(date +%Y%m%d).tar.gz

# 3. 验证镜像
docker images | grep roll-pytorch-vllm
```

## 5. 文件大小对比

根据经验，不同导出方式的文件大小差异：

| 导出方式 | 预估大小 | 压缩后大小 | 备注 |
|----------|----------|------------|------|
| 容器导出 | 8-15GB | 3-6GB | 仅当前文件系统状态 |
| 镜像保存 | 25-35GB | 8-12GB | 包含完整镜像层 |
| 原始镜像 | ~30GB | - | 基础镜像大小 |

## 6. 传输和部署

### 6.1 传输到其他服务器

```bash
# 使用scp传输
scp docker_backups/roll_vllm_20250106_143000.tar.gz user@target-server:/path/to/destination/

# 使用rsync (支持断点续传)
rsync -avz --progress docker_backups/roll_vllm_20250106_143000.tar.gz user@target-server:/path/to/destination/
```

### 6.2 在目标服务器部署

```bash
# 1. 导入镜像
docker import roll_vllm_20250106_143000.tar.gz my-pytorch:latest

# 2. 运行新容器 (使用相同配置)
docker run -d --name pytorch-vllm-new \
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
  my-pytorch:latest \
  sleep infinity
```

## 7. 最佳实践

### 7.1 导出前的准备工作

```bash
# 1. 清理容器内的临时文件
docker exec roll_vllm bash -c "
  apt clean
  rm -rf /tmp/*
  rm -rf /var/cache/*
  rm -rf ~/.cache/*
"

# 2. 停止不必要的进程
docker exec roll_vllm bash -c "
  # 停止训练进程或其他大内存进程
  pkill -f python || true
"
```

### 7.2 验证导出结果

```bash
# 导入测试
docker import test_backup.tar.gz test-image:latest

# 快速测试容器
docker run --rm test-image:latest python --version
docker run --rm test-image:latest python -c "import torch; print(torch.__version__)"
```

### 7.3 自动化导出脚本

```bash
#!/bin/bash
# 文件名: backup_container.sh

CONTAINER_NAME="roll_vllm"
BACKUP_DIR="/data1/Weiyu_project/docker_backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

echo "开始导出容器 $CONTAINER_NAME..."

# 导出容器
docker export $CONTAINER_NAME | gzip > $BACKUP_DIR/${CONTAINER_NAME}_${DATE}.tar.gz

if [ $? -eq 0 ]; then
    echo "导出成功: $BACKUP_DIR/${CONTAINER_NAME}_${DATE}.tar.gz"
    ls -lh $BACKUP_DIR/${CONTAINER_NAME}_${DATE}.tar.gz
else
    echo "导出失败!"
    exit 1
fi

# 清理旧备份 (保留最近5个)
cd $BACKUP_DIR
ls -t ${CONTAINER_NAME}_*.tar.gz | tail -n +6 | xargs rm -f 2>/dev/null

echo "备份完成!"
```

## 8. 注意事项

### 8.1 数据持久化
- **挂载卷数据**: `/data1` 和 `/data` 的数据不会包含在容器导出中
- **重要数据**: 确保重要数据保存在挂载的持久化卷中
- **配置文件**: 容器内的配置修改会被保存

### 8.2 性能考虑
- **导出时间**: 大容器导出可能需要10-30分钟
- **磁盘空间**: 确保有足够空间存储导出文件
- **I/O影响**: 导出过程会产生大量磁盘I/O

### 8.3 安全考虑
- **敏感信息**: 检查容器内是否包含密钥、密码等敏感信息
- **网络配置**: 导入时注意网络和端口配置
- **权限设置**: 注意容器的权限和用户配置

## 9. 常见问题解决

### 9.1 导出失败
```bash
# 检查磁盘空间
df -h /data1

# 检查容器状态
docker inspect roll_vllm

# 强制导出 (如果容器有问题)
docker export --pause=false roll_vllm | gzip > backup.tar.gz
```

### 9.2 导入失败
```bash
# 检查tar文件完整性
gzip -t backup.tar.gz

# 使用verbose模式导入
docker import -v backup.tar.gz my-image:latest
```

### 9.3 运行问题
```bash
# 检查导入的镜像
docker run --rm -it my-image:latest /bin/bash

# 比较原始和导入的镜像
docker run --rm my-image:latest python -c "import sys; print(sys.path)"
```

---

**推荐方案**: 
- **日常备份**: 使用容器导出 (`docker export`) + 压缩
- **完整迁移**: 使用镜像提交 + 保存 (`docker commit` + `docker save`)
- **定期备份**: 使用自动化脚本每周备份

**最后更新**: 2025年1月6日
