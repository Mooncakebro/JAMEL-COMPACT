"""
CLI 主入口
"""

import argparse
import importlib
import sys


def _lazy_main(module_name: str):
    def _runner(args):
        module = importlib.import_module(module_name, package=__package__)
        module.main(args)

    return _runner


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="JAMEL browser-agent training and evaluation toolkit",
        usage="jamel <command> [options]"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # serve 子命令
    serve_parser = subparsers.add_parser(
        "serve",
        help="启动模型服务"
    )
    serve_parser.add_argument(
        "--model-path",
        required=True,
        help="模型路径"
    )
    serve_parser.add_argument(
        "--model-type",
        help="模型类型"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="服务端口"
    )
    serve_parser.set_defaults(func=_lazy_main(".serve"))

    # demo 子命令
    demo_parser = subparsers.add_parser(
        "demo",
        help="运行演示"
    )
    demo_parser.add_argument(
        "--type",
        choices=["web", "rl"],
        default="web",
        help="演示类型"
    )
    demo_parser.add_argument(
        "--url",
        default="https://news.ycombinator.com/",
        help="起始 URL (web demo)"
    )
    demo_parser.add_argument(
        "--goal",
        default="浏览热门新闻",
        help="用户目标 (web demo)"
    )
    demo_parser.set_defaults(func=_lazy_main(".demo"))

    # train 子命令
    train_parser = subparsers.add_parser(
        "train",
        help="训练 RL 模型"
    )
    train_parser.add_argument(
        "--env",
        default="CartPole-v1",
        help="环境名称"
    )
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="训练回合数"
    )
    train_parser.set_defaults(func=_lazy_main(".train"))

    # preview 子命令
    preview_parser = subparsers.add_parser(
        "preview",
        help="预览 parquet 数据"
    )
    preview_parser.add_argument(
        "parquet_paths",
        nargs="*",
        help="parquet 文件路径（可多个）"
    )
    preview_parser.add_argument(
        "--parquet-dir",
        help="包含 parquet 文件的目录"
    )
    preview_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务 host"
    )
    preview_parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="服务端口"
    )
    preview_parser.add_argument(
        "--columns",
        nargs="*",
        help="只加载指定列"
    )
    preview_parser.add_argument(
        "--max-text-len",
        type=int,
        default=200,
        help="文本截断长度"
    )
    preview_parser.add_argument(
        "--max-collection-len",
        type=int,
        default=2000,
        help="字典/数组展开长度"
    )
    preview_parser.add_argument(
        "--image-thumb-max-side",
        type=int,
        default=256,
        help="缩略图最大边长"
    )
    preview_parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="每页行数"
    )
    preview_parser.add_argument(
        "--image-full-max-side",
        type=int,
        default=1024,
        help="放大图最大边长"
    )
    preview_parser.set_defaults(func=_lazy_main(".preview"))

    analyze_parser = subparsers.add_parser(
        "analyze-curriculum",
        help="分析 stage 内各 iteration 的 reward / error 趋势"
    )
    analyze_parser.add_argument(
        "stage_dir",
        help="stage 目录，例如 exploration_data/main_0330/stage_0"
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式打印结果"
    )
    analyze_parser.add_argument(
        "--output-json",
        help="将结果额外写入 JSON 文件"
    )
    analyze_parser.set_defaults(func=_lazy_main(".analyze_curriculum"))

    audit_coverage_parser = subparsers.add_parser(
        "audit-coverage",
        help="生成轨迹 coverage 审计 HTML"
    )
    audit_coverage_parser.add_argument(
        "parquet_path",
        help="轨迹 parquet 路径"
    )
    audit_coverage_parser.add_argument(
        "--output-html",
        required=True,
        help="输出 HTML 路径"
    )
    audit_coverage_parser.add_argument(
        "--image-max-side",
        type=int,
        default=640,
        help="图片显示最大边长"
    )
    audit_coverage_parser.add_argument(
        "--obs-limit",
        type=int,
        default=5000,
        help="观测文本最大长度"
    )
    audit_coverage_parser.set_defaults(func=_lazy_main(".audit_coverage"))

    weak_label_parser = subparsers.add_parser(
        "weak-model-sft-label",
        help="采集弱模型 ScaleWoB 探索 SFT 标注轨迹"
    )
    weak_label_parser.add_argument("--apps", default="weibo", help="逗号分隔的 app id，例如 weibo,agoda")
    weak_label_parser.add_argument("--output-dir", default=None, help="输出根目录")
    weak_label_parser.add_argument("--scalewob-root", default=None, help="scalewob-env 根目录")
    weak_label_parser.add_argument("--host", default="127.0.0.1")
    weak_label_parser.add_argument("--port", type=int, default=8000)
    weak_label_parser.add_argument("--no-port-search", action="store_true")
    weak_label_parser.add_argument("--use-existing-server", action="store_true")
    weak_label_parser.add_argument("--episodes-per-app", type=int, default=30)
    weak_label_parser.add_argument(
        "--target-episodes-per-app",
        type=int,
        default=None,
        help="补采到每个 app 的总 episode 数；设置后优先于 --episodes-per-app 的追加语义",
    )
    weak_label_parser.add_argument("--steps-per-episode", type=int, default=10)
    weak_label_parser.add_argument("--timeout", type=int, default=60000)
    weak_label_parser.add_argument("--seed", type=int, default=7)
    weak_label_parser.add_argument("--policy", choices=["auto", "model", "heuristic"], default="auto")
    weak_label_parser.add_argument("--model-name", default=None)
    weak_label_parser.add_argument("--model-base-url", default=None)
    weak_label_parser.add_argument("--model-api-key", default=None)
    weak_label_parser.add_argument("--model-temperature", type=float, default=0.7)
    weak_label_parser.add_argument("--model-max-tokens", type=int, default=512)
    weak_label_parser.add_argument("--model-timeout", type=int, default=120)
    weak_label_parser.add_argument("--model-use-screenshot", action="store_true")
    weak_label_parser.add_argument("--model-invalid-action-retries", type=int, default=2)
    weak_label_parser.add_argument("--show-browser", action="store_true")
    weak_label_parser.add_argument("--fresh", action="store_true")
    weak_label_parser.add_argument("--ignore-exhausted", action="store_true")
    weak_label_parser.add_argument("--no-positive-exhaustion", type=int, default=30)
    weak_label_parser.add_argument("--coverage-growth-window", type=int, default=100)
    weak_label_parser.add_argument("--coverage-growth-threshold", type=int, default=1)
    weak_label_parser.add_argument("--valid-action-rate-threshold", type=float, default=0.2)
    weak_label_parser.set_defaults(func=_lazy_main(".weak_model_sft_label"))

    weak_augment_parser = subparsers.add_parser(
        "weak-model-sft-augment",
        help="离线重排 episode 并重新计算 coverage reward，导出增强 SFT 数据",
    )
    weak_augment_parser.add_argument("--source-index", default=None, help="all_episode_manifest.jsonl 路径")
    weak_augment_parser.add_argument("--output-dir", default=None, help="增强数据输出目录")
    weak_augment_parser.add_argument("--backup-dir", default=None, help="源数据备份目录")
    weak_augment_parser.add_argument("--run-id", default=None)
    weak_augment_parser.add_argument("--augmentation-id", dest="run_id", default=None, help=argparse.SUPPRESS)
    weak_augment_parser.add_argument("--apps", default=None, help="逗号分隔 app 过滤")
    weak_augment_parser.add_argument("--max-apps", type=int, default=None)
    weak_augment_parser.add_argument("--max-episodes-per-app", type=int, default=None)
    weak_augment_parser.add_argument("--permutations-per-app", type=int, default=20)
    weak_augment_parser.add_argument("--seed", type=int, default=20260519)
    weak_augment_parser.add_argument("--dedupe-mode", choices=["prefix", "none"], default="prefix")
    weak_augment_parser.add_argument("--audit-canonical-samples", type=int, default=0)
    weak_augment_parser.add_argument("--require-backup", dest="require_backup", action="store_true", default=True)
    weak_augment_parser.add_argument("--no-require-backup", dest="require_backup", action="store_false")
    weak_augment_parser.add_argument("--overwrite", action="store_true")
    weak_augment_parser.set_defaults(func=_lazy_main(".weak_model_sft_augment"))

    baseline_eval_parser = subparsers.add_parser(
        "baseline-eval",
        help="运行 GUI baseline session-level evaluation",
    )
    from jamel.cli.baseline_eval import configure_parser as configure_baseline_eval_parser

    configure_baseline_eval_parser(baseline_eval_parser)
    baseline_eval_parser.set_defaults(func=_lazy_main(".baseline_eval"))

    # 解析参数
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # 执行对应命令
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
