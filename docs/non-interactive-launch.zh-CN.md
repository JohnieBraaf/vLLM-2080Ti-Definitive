# 非交互启动

先执行 `./build.sh`，再传入扁平化的 profile 路径。Profile 只包含路线参数；检查点、GPU 拓扑、端口和 mode 都由 launcher 管理。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` 可省略，默认是 `fast`。没有 `MODE` 的 profile 会保留 launcher 的选择；profile 中显式的 `MODE=normal` 或 `MODE=fast` 可以覆盖它。目录中不再区分 `fast/` 和 `normal/`。

可以使用 `--model-dir`、`--speculative-model`、`--profile`、`--mode`、`--gpu-devices`、`--tp-size`、`--pp-size`、`--port`、`--start-timeout` 和 `--print-config`。高级 launcher/runtime 参数使用 `--set KEY=VALUE`。不要把 Prefix Cache、Mamba cache、GPU、端口或模型路径写入 profile，验证器会拒绝这些字段。

Profile 库是验证矩阵，不代表每个文件在每台机器上都能运行。只有外部审计完成启动、4K/128、32K/512、并发和图文正确性验证后，路线才会被 promote。

## 实验性 SSD 前缀缓存

Launcher 可以启用 vLLM 内置的 `OffloadingConnector`，通过 CPU 暂存层和文件系统层将已完成的前缀 KV 块写入磁盘，不需要常驻的独立缓存进程。目录必须位于持久化存储上；重启后保持模型路径、KV 精度、块布局和并行配置一致。CPU 暂存空间占用主机内存，不会增加 GPU KV 容量。磁盘命名空间不包含模型权重的内容摘要；如果在原路径替换权重，或切换到不兼容的运行时，必须清理旧缓存目录。

文件系统层不会自动淘汰旧文件，也没有磁盘配额；请使用空间充足的专用卷，监控用量，并在服务停止后清理过期缓存。`KV_DISK_CPU_BYTES` 只限制主机内存暂存空间。重启时保持相同的 `PYTHONHASHSEED`（默认 `0`），否则已有前缀无法命中。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-AWQ-INT4 \
  --gpu-devices 0,2,3,4 --tp-size 4 \
  --set KV_DISK_CACHE_DIR=/mnt/nvme/vllm-kv-cache \
  --set KV_DISK_CPU_BYTES=4294967296 \
  --print-config
```

去掉 `--print-config` 即可启动。启用后 launcher 会要求打开前缀缓存，并设置稳定的 `PYTHONHASHSEED`。停止服务前应等待 `vllm:kv_offload_tiering_active_cascade_jobs` 回到零。重启后可查看 `vllm:external_prefix_cache_hits_total`，以及文件系统层的 `vllm:kv_offload_tiering_chunk_hits_total`、`vllm:kv_offload_tiering_read_bytes_total`。

在 4xT10、Qwen3.8-27B AWQ-INT4、FP16 KV 上，整机重启后，相同的 1,929-token 提示词从 SSD 恢复了 1,568 tokens，输出与冷启动一致。文件系统层记录了 5 次命中和约 257 MB 读取。此路线仍属实验性，暂无正式吞吐成绩。
在双 RTX 2080 Ti 上，同模型、FP16 KV 的 1,933-token 提示词在 vLLM 服务退出并重启后命中 1,568 tokens，输出保持一致。双 2080 Ti 测试未单独进行整机重启。

还在双 RTX 2080 Ti（GPU 1、5）上验证了仓库正式 `fast` profile。Qwen3.8-27B
W8A16 FP8 权重 / FP8 KV、MTP4、262K 纯文本路线使用 4 GiB CPU 暂存空间，服务重启后
在 5,356 个提示 token 中命中 3,232 个。W4A16 NVFP4 权重 / FP8 KV、DFlash2、262K
图文路线使用 8 GiB CPU 暂存空间，以纯文本请求测试，服务重启后在 5,355 个提示 token
中命中 3,296 个；文件系统层记录 21 个 chunk 命中、读取 5,244,911,616 字节。两条路线的
输出均与首次请求一致，都是 `READY`。这些测试未包含整机重启或吞吐基准。在这台主机上，
MTP4 FP16 KV 148K 路线启用 SSD offload 时，需要通过 launcher 设置 `GPU_UTIL=0.965`
才能保留其 148,480-token 容量；这是该主机的启动覆盖值，正式 profile 未修改。
