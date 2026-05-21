"""
Streamlit Community Cloud entry point.
Loads the dashboard from dashboard/streamlit_app.py
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard" / "streamlit_app.py"

spec = importlib.util.spec_from_file_location("dashboard_app", DASHBOARD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
