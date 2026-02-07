#!/usr/bin/env python3
"""Streamlit 应用入口。从项目根目录运行: uv run streamlit run app.py 或 pip install rs10env[app] && rs10env-app"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_file = Path(__file__).resolve().parent / "rs10env" / "app.py"
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.headless", "true"] + sys.argv[1:],
        ).returncode
    )
