import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

class LoggingFedAvg(FedAvg):
    """
    Flower FedAvg strategy enhanced with metric tracking, history recording,
    and weighted accuracy aggregation across federated rounds.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            "round": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted FedAvg and record client metrics."""
        if not results:
            return None, {}

        # Default parameter aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # Aggregate custom training metrics
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        weighted_loss = sum([fit_res.metrics.get("train_loss", 0.0) * fit_res.num_examples for _, fit_res in results]) / max(1, total_samples)
        weighted_acc = sum([fit_res.metrics.get("train_acc", 0.0) * fit_res.num_examples for _, fit_res in results]) / max(1, total_samples)

        self.history["round"].append(server_round)
        self.history["train_loss"].append(float(weighted_loss))
        self.history["train_acc"].append(float(weighted_acc))

        metrics["train_loss"] = float(weighted_loss)
        metrics["train_acc"] = float(weighted_acc)

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluate results across clients."""
        if not results:
            return None, {}

        loss_aggregated, metrics = super().aggregate_evaluate(server_round, results, failures)

        total_samples = sum([eval_res.num_examples for _, eval_res in results])
        weighted_acc = sum([eval_res.metrics.get("val_acc", 0.0) * eval_res.num_examples for _, eval_res in results]) / max(1, total_samples)

        if loss_aggregated is not None:
            self.history["val_loss"].append(float(loss_aggregated))
        self.history["val_acc"].append(float(weighted_acc))
        metrics["val_acc"] = float(weighted_acc)

        return loss_aggregated, metrics

