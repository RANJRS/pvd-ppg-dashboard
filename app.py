import sys
import os

# Add pipeline directory to sys.path
pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "S7 new", "ml_pipeline"))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

from server_dashboard import run_server

if __name__ == "__main__":
    run_server()
