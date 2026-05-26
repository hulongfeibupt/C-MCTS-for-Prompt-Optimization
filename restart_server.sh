#!/bin/bash
# 查找正在使用端口 8082 的进程 PID
PID=$(lsof -ti :8082)

if [ -n "$PID" ]; then
  echo "发现进程 $PID 正在占用端口 8082，正在终止..."
  kill -9 $PID
  echo "进程 $PID 已终止。"
else
  echo "端口 8082 当前空闲。"
fi

# conda activate q3_1 && sh /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/restart_server.sh

#
#