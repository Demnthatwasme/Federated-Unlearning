import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import main

if __name__ == "__main__":
    main()
