import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
import json
import time
import argparse
import torch
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.cifar_loader import SingleClassFLDataManager, MultiClassFLDataManager
from src.models.resnet_cifar import FedVisionModel, get_cifar_model
from src.flwr_fl.runner import run_fl_training, eval_client_split, eval_multiclass_client_split
from src.unlearning.federaser import FedEraserEngine
from src.visualize.gradcam_visualizer import generate_unlearning_gallery

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Federated Unlearning (FedEraser) for Vision Models with Grad-CAM Auditing")
    parser.add_argument("--partition_mode", choices=["single_class", "multi_class"], default="single_class", help="FL dataset partitioning mode: 'single_class' (1 client 1 class) or 'multi_class' (heterogeneous Dirichlet non-IID)")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes (CIFAR-100 classes 0..K-1)")
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of clients in multi_class mode")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Dirichlet concentration parameter for multi-class non-IID partitioning")
    parser.add_argument("--forget_client", type=int, default=9, help="Client ID to unlearn/forget")
    parser.add_argument("--num_rounds", type=int, default=6, help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per client per round")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="SGD learning rate")
    parser.add_argument("--cali_steps", type=int, default=4, help="Max calibration batches per client in FedEraser")
    parser.add_argument("--cali_lr", type=float, default=0.003, help="FedEraser calibration learning rate")
    parser.add_argument("--milestone_interval", type=int, default=2, help="FedEraser milestone calibration round interval")
    parser.add_argument("--stage", choices=["train", "unlearn", "visualize", "all"], default="all", help="Execution pipeline stage")
    parser.add_argument("--batch_size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

    is_multiclass = (args.partition_mode == "multi_class")
    ckpt_dir = "checkpoints/multi_class" if is_multiclass else "checkpoints"
    res_dir = "results/multi_class" if is_multiclass else "results"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    orig_ckpt = os.path.join(ckpt_dir, "M_orig.pt")
    retrain_ckpt = os.path.join(ckpt_dir, "M_retrain.pt")
    unlearn_ckpt = os.path.join(ckpt_dir, "M_unlearn.pt")
    cache_path = os.path.join(ckpt_dir, "fl_training_cache.pt")
    bench_path = os.path.join(res_dir, "unlearning_benchmark.json")
    gallery_path = os.path.join(res_dir, "gradcam_unlearning_comparison.png")

    mode_title = f"MULTI-CLASS NON-IID (DIRICHLET α={args.dirichlet_alpha})" if is_multiclass else "SIMPLIFICATION: 1 CLIENT = 1 CLASS"
    print("=" * 80)
    print("  FEDERATED UNLEARNING FOR VISION MODELS (CIFAR-100)")
    print("  METHOD: FEDERASER (CALIBRATED HISTORICAL RECONSTRUCTION)")
    print(f"  PARTITION MODE: {mode_title}")
    print("=" * 80)
    device_desc = torch.cuda.get_device_name(0) if args.device == "cuda" and torch.cuda.is_available() else "CPU Host (No GPU Acceleration)"
    print(f"Device: {args.device} ({device_desc})")
    print(f"Config: {args.num_classes} Classes | Forget Target Client {args.forget_client} | FL Rounds: {args.num_rounds}")

    # 1. Dataset Partitioning
    selected_classes = list(range(args.num_classes))
    if is_multiclass:
        data_manager = MultiClassFLDataManager(
            root="dataset",
            selected_classes=selected_classes,
            num_clients=args.num_clients,
            alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            num_workers=0,
            seed=args.seed
        )
        all_client_ids = list(range(args.num_clients))
        forget_class_name = f"Client {args.forget_client}"
        print(f"\n[Client Setup] Total Clients: {args.num_clients} (Heterogeneous Multi-Class Non-IID)")
        for cid in all_client_ids:
            dist = data_manager.get_client_class_distribution(cid)
            role = ">>> TARGET FORGET CLIENT <<<" if cid == args.forget_client else "Retain Client"
            top_classes = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{k}: {v}" for k, v in top_classes])
            print(f"  Client {cid:02d} -> {len(data_manager.client_indices[cid])} samples across {len(dist)} classes ({top_str}...) [{role}]")
    else:
        data_manager = SingleClassFLDataManager(
            root="dataset",
            selected_classes=selected_classes,
            batch_size=args.batch_size,
            num_workers=0
        )
        all_client_ids = selected_classes
        forget_class_name = data_manager.class_names[args.forget_client]
        print(f"\n[Client Setup] Total Clients: {args.num_classes} (1 Client = 1 Class)")
        for cid in range(args.num_classes):
            role = ">>> TARGET FORGET CLIENT <<<" if cid == args.forget_client else "Retain Client"
            print(f"  Client {cid:02d} -> Class '{data_manager.class_names[cid]}' ({role})")

    full_test_loader = data_manager.get_full_test_loader()
    orig_model = None
    retrain_model = None
    unlearn_model = None
    orig_history = None
    retrain_history = None
    unlearn_history = None
    history_globals = None
    history_updates = None

    # =========================================================================
    # PHASE 1: Federated Training of Original Baseline Model (M_orig)
    # =========================================================================
    if args.stage in ["train", "all"]:
        print("\n" + "#" * 80)
        print("### PHASE 1: Federated Learning of Original Model (M_orig) across All Clients")
        print("#" * 80)
        orig_model, orig_history, history_globals, history_updates = run_fl_training(
            data_manager=data_manager,
            client_ids=all_client_ids,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            save_checkpoint_path=orig_ckpt,
            forget_client_id=args.forget_client,
            tag="Original FL",
            verbose=True
        )
        # Save training history and update cache
        torch.save({
            "history_globals": history_globals,
            "history_updates": history_updates,
            "orig_history": orig_history
        }, cache_path)

    # =========================================================================
    # PHASE 2: Exact Retraining Gold Standard Baseline (M_retrain)
    # =========================================================================
    if args.stage in ["train", "all"]:
        print("\n" + "#" * 80)
        print("### PHASE 2: Exact Retraining Baseline (M_retrain) on Retain Clients Only")
        print("#" * 80)
        engine = FedEraserEngine(data_manager=data_manager, forget_client_id=args.forget_client, device=args.device)
        retrain_model, retrain_metrics, retrain_history = engine.run_exact_retrain(
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            save_path=retrain_ckpt,
            verbose=True
        )

    # =========================================================================
    # PHASE 3: FedEraser Federated Unlearning (M_unlearn)
    # =========================================================================
    if args.stage in ["unlearn", "all"]:
        print("\n" + "#" * 80)
        print("### PHASE 3: Fast Federated Unlearning via FedEraser (M_unlearn)")
        print("#" * 80)
        if history_globals is None or history_updates is None:
            assert os.path.exists(cache_path), f"FL training cache not found at {cache_path}. Run with --stage train or --stage all."
            cache = torch.load(cache_path, map_location=args.device)
            history_globals = cache["history_globals"]
            history_updates = cache["history_updates"]
            orig_history = cache.get("orig_history", None)

        engine = FedEraserEngine(data_manager=data_manager, forget_client_id=args.forget_client, device=args.device)
        unlearn_model, unlearn_metrics, unlearn_history = engine.run_federaser(
            history_global_models=history_globals,
            history_client_updates=history_updates,
            cali_epochs=1,
            cali_steps=args.cali_steps,
            cali_lr=args.cali_lr,
            milestone_interval=args.milestone_interval,
            save_path=unlearn_ckpt,
            verbose=True
        )

    # Load models if running standalone visualize stage
    if args.stage in ["visualize", "all"]:
        if orig_model is None:
            orig_model = get_cifar_model(num_classes=args.num_classes, device=args.device)
            orig_model.load_state_dict(torch.load(orig_ckpt, map_location=args.device))
        if retrain_model is None and os.path.exists(retrain_ckpt):
            retrain_model = get_cifar_model(num_classes=args.num_classes, device=args.device)
            retrain_model.load_state_dict(torch.load(retrain_ckpt, map_location=args.device))
        if unlearn_model is None:
            unlearn_model = get_cifar_model(num_classes=args.num_classes, device=args.device)
            unlearn_model.load_state_dict(torch.load(unlearn_ckpt, map_location=args.device))

        # =====================================================================
        # PHASE 4: Visual Attribution Evidence via Grad-CAM
        # =====================================================================
        print("\n" + "#" * 80)
        print("### PHASE 4: Generating Grad-CAM Visual Attribution Evidence")
        print("#" * 80)
        
        if is_multiclass:
            # Select dominant class of the forgotten client (e.g., 'bottle' with 177 samples in Client 9)
            dist = data_manager.get_client_class_distribution(args.forget_client)
            dominant_class_name = max(dist.items(), key=lambda x: x[1])[0]
            dominant_class_idx = data_manager.class_names.index(dominant_class_name)
            
            c_forget_indices = data_manager.client_indices[args.forget_client]
            raw_targets = np.array(data_manager.raw_train.targets)[c_forget_indices]
            dominant_indices = [idx for idx, t in zip(c_forget_indices, raw_targets) if t == dominant_class_idx]
            
            # Select samples correctly classified by M_orig
            clean_dominant = []
            with torch.no_grad():
                for idx in dominant_indices:
                    img_t, _ = data_manager.raw_train_eval[idx]
                    p = orig_model(img_t.unsqueeze(0).to(args.device)).argmax(1).item()
                    if p == dominant_class_idx:
                        clean_dominant.append(idx)
            selected_f_indices = clean_dominant[:3] if len(clean_dominant) >= 3 else dominant_indices[:3]
            
            forget_imgs = torch.stack([data_manager.raw_train_eval[i][0] for i in selected_f_indices])
            forget_lbls = [dominant_class_idx] * len(selected_f_indices)
            
            c0_indices = data_manager.client_indices[0][:3]
            retain_imgs = torch.stack([data_manager.raw_train_eval[i][0] for i in c0_indices])
            retain_lbls = [data_manager.raw_train.targets[i] for i in c0_indices]
            
            all_samples = torch.cat([forget_imgs, retain_imgs], dim=0)
            all_labels = forget_lbls + retain_lbls
            is_forget_sample = [True] * len(forget_imgs) + [False] * len(retain_imgs)
        else:
            forget_imgs, forget_lbls = data_manager.get_sample_images_for_class(args.forget_client, count=3)
            retain_samples = []
            retain_lbls = []
            for cid in [c for c in range(args.num_classes) if c != args.forget_client][:3]:
                imgs, lbls = data_manager.get_sample_images_for_class(cid, count=1)
                retain_samples.append(imgs)
                retain_lbls.extend(lbls)
            all_samples = torch.cat([forget_imgs, torch.cat(retain_samples, dim=0)], dim=0)
            all_labels = forget_lbls + retain_lbls
            is_forget_sample = [True] * len(forget_imgs) + [False] * len(retain_lbls)

        models_dict = {
            "Original": orig_model,
            "FedEraser": unlearn_model
        }
        if retrain_model is not None:
            models_dict["Retrained"] = retrain_model

        gallery_info = generate_unlearning_gallery(
            models_dict=models_dict,
            sample_images=all_samples,
            sample_labels=all_labels,
            class_names=data_manager.class_names,
            unnormalizer=data_manager.unnormalizer,
            forget_client_id=args.forget_client,
            save_path=gallery_path,
            device=args.device,
            is_forget_sample=is_forget_sample
        )
        print(f"Saved Grad-CAM Visual Evidence Gallery -> {gallery_info['saved_path']}")
        print(f"Average Grad-CAM Drift on Forgotten Data: {gallery_info['avg_forget_cam_drift']:.3f}")
        print(f"Average Grad-CAM Drift on Retained Data:  {gallery_info['avg_retain_cam_drift']:.3f}")

        # =====================================================================
        # PHASE 5: Compute Final Benchmark Summary Table
        # =====================================================================
        print("\n" + "=" * 80)
        print("  FINAL FEDERATED UNLEARNING BENCHMARK SUMMARY")
        print("=" * 80)

        if is_multiclass:
            forget_loader = data_manager.get_forget_train_loader(args.forget_client)
            eval_orig = eval_multiclass_client_split(orig_model, full_test_loader, forget_loader, device=args.device)
            eval_unl = eval_multiclass_client_split(unlearn_model, full_test_loader, forget_loader, device=args.device)
            eval_ret = eval_multiclass_client_split(retrain_model, full_test_loader, forget_loader, device=args.device) if retrain_model else {}
        else:
            eval_orig = eval_client_split(orig_model, full_test_loader, args.forget_client, device=args.device)
            eval_unl = eval_client_split(unlearn_model, full_test_loader, args.forget_client, device=args.device)
            eval_ret = eval_client_split(retrain_model, full_test_loader, args.forget_client, device=args.device) if retrain_model else {}

        t_orig = orig_history.get("total_time", 0.0) if orig_history else 0.0
        t_ret = retrain_metrics.get("runtime_seconds", 0.0) if 'retrain_metrics' in locals() else 0.0
        t_unl = unlearn_metrics.get("runtime_seconds", 0.0) if 'unlearn_metrics' in locals() else 0.0
        speedup = (t_ret / max(0.01, t_unl)) if t_ret > 0 and t_unl > 0 else 1.0

        headers = ["Model / Metric", "Test Acc", "Forget Acc", "Forget Conf", "Runtime", "Speedup vs Retrain"]
        print(f"{headers[0]:<28} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<11} | {headers[4]:<10} | {headers[5]:<18}")
        print("-" * 105)

        row_orig = f"{'Original FL (M_orig)':<28} | {eval_orig['overall_acc']:>9.2%} | {eval_orig['forget_acc']:>9.2%} | {eval_orig['forget_conf']:>10.3f} | {t_orig:>8.2f}s | {'1.0x':<18}"
        row_unl  = f"{'FedEraser (M_unlearn)':<28} | {eval_unl['overall_acc']:>9.2%} | {eval_unl['forget_acc']:>9.2%} | {eval_unl['forget_conf']:>10.3f} | {t_unl:>8.2f}s | {f'{speedup:.2f}x':<18}"
        print(row_orig)
        print(row_unl)
        if retrain_model:
            row_ret = f"{'Retrained Gold (M_retrain)':<28} | {eval_ret['overall_acc']:>9.2%} | {eval_ret['forget_acc']:>9.2%} | {eval_ret['forget_conf']:>10.3f} | {t_ret:>8.2f}s | {'(Baseline)':<18}"
            print(row_ret)
        print("-" * 105)

        agreement_pct = 0.0
        redistribution_map = {}
        if is_multiclass and retrain_model is not None:
            forget_loader = data_manager.get_forget_train_loader(args.forget_client)
            agrees = 0
            total_f = 0
            with torch.no_grad():
                for x, _ in forget_loader:
                    x = x.to(args.device)
                    p_unl = unlearn_model(x).argmax(1)
                    p_ret = retrain_model(x).argmax(1)
                    agrees += (p_unl == p_ret).sum().item()
                    total_f += len(x)
            agreement_pct = (agrees / max(1, total_f)) * 100
            print(f"\nUnlearning Fidelity on Forgotten Client Data (D_forget):")
            print(f"  Prediction Agreement (M_unlearn == M_retrain): {agrees}/{total_f} ({agreement_pct:.1f}%)")
            print(f"  M_orig Train Loss on D_forget:     {eval_orig['forget_train_loss']:.4f} (Memorized)")
            print(f"  M_unlearn Loss on D_forget:        {eval_unl['forget_train_loss']:.4f} (Unlearned)")
            print(f"  M_retrain Loss on D_forget:        {eval_ret['forget_train_loss']:.4f} (Gold Standard Unseen)")
        elif not is_multiclass:
            forget_loader = data_manager.get_forget_test_loader(args.forget_client)
            preds_unl = []
            with torch.no_grad():
                for x, _ in forget_loader:
                    preds_unl.append(unlearn_model(x.to(args.device)).argmax(1))
            preds_unl = torch.cat(preds_unl)

            print(f"\nEmpirical Redistribution of Forgotten Class '{forget_class_name}' on M_unlearn:")
            for c in range(args.num_classes):
                cnt = (preds_unl == c).sum().item()
                if cnt > 0:
                    pct = (cnt / len(preds_unl)) * 100
                    redistribution_map[data_manager.class_names[c]] = cnt
                    print(f"  -> Predicted as Class {c:02d} ('{data_manager.class_names[c]}'): {cnt}/{len(preds_unl)} ({pct:.1f}%)")

        # Save structured results JSON
        benchmark_results = {
            "config": vars(args),
            "partition_mode": args.partition_mode,
            "forget_client": args.forget_client,
            "forget_class_name": forget_class_name,
            "models": {
                "original": {
                    "overall_acc": eval_orig["overall_acc"],
                    "retain_acc": eval_orig["retain_acc"],
                    "forget_acc": eval_orig["forget_acc"],
                    "forget_conf": eval_orig["forget_conf"],
                    "forget_train_loss": eval_orig.get("forget_train_loss", 0.0),
                    "runtime_seconds": t_orig
                },
                "federaser": {
                    "overall_acc": eval_unl["overall_acc"],
                    "retain_acc": eval_unl["retain_acc"],
                    "forget_acc": eval_unl["forget_acc"],
                    "forget_conf": eval_unl["forget_conf"],
                    "forget_train_loss": eval_unl.get("forget_train_loss", 0.0),
                    "runtime_seconds": t_unl,
                    "speedup_vs_retrain": speedup,
                    "avg_forget_cam_drift": gallery_info["avg_forget_cam_drift"],
                    "avg_retain_cam_drift": gallery_info["avg_retain_cam_drift"],
                    "prediction_agreement_with_retrain": agreement_pct,
                    "redistribution_map": redistribution_map
                },
                "retrained": {
                    "overall_acc": eval_ret.get("overall_acc", 0.0),
                    "retain_acc": eval_ret.get("retain_acc", 0.0),
                    "forget_acc": eval_ret.get("forget_acc", 0.0),
                    "forget_conf": eval_ret.get("forget_conf", 0.0),
                    "forget_train_loss": eval_ret.get("forget_train_loss", 0.0),
                    "runtime_seconds": t_ret
                } if retrain_model else {}
            },
            "gradcam_gallery_path": gallery_info["saved_path"],
            "drift_records": gallery_info["drift_records"]
        }

        with open(bench_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"Saved complete benchmark results -> {bench_path}\n")

if __name__ == "__main__":
    main()
