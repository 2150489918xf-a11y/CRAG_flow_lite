"""
项目路径常量 — 所有路径以项目根目录为锚点
迁移部署时只需确保包结构不变，无需修改任何路径。
"""
from pathlib import Path

# 项目根目录：common/ 的父目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 常用子目录
CONF_DIR = PROJECT_ROOT / "conf"
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
GRAPH_DIR = DATA_DIR / "graphs"
