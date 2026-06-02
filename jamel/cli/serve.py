"""
Serve 命令
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# 导入原有的 server 模块
from jamel.models.service.server import run_model_server, app
import uvicorn


def main(args):
    """启动服务"""
    print("🚀 启动模型服务...")
    print(f"   模型路径: {args.model_path}")
    print(f"   模型类型: {args.model_type or '自动检测'}")
    print(f"   服务端口: {args.port}\n")

    # 使用 run_model_server 函数启动服务
    run_model_server(
        model_path=args.model_path,
        model_type=args.model_type,
        port=args.port,
        host="0.0.0.0"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--model-type", help="模型类型")
    parser.add_argument("--port", type=int, default=8888, help="服务端口")
    args = parser.parse_args()
    main(args)