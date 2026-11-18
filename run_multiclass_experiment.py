import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import main

if __name__ == "__main__":
    # Configure sys.argv for multi-class Dirichlet federated unlearning
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--partition_mode", "multi_class",
            "--num_classes", "10",
            "--num_clients", "10",
            "--dirichlet_alpha", "0.5",
            "--forget_client", "9",
            "--num_rounds", "6",
            "--local_epochs", "2",
            "--stage", "all",
            "--device", "cuda"
        ])
    main()
