import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2

from src.data.cifar_loader import SingleClassFLDataManager, MultiClassFLDataManager
from src.models.resnet_cifar import FedVisionModel, get_cifar_model
from src.visualize.gradcam_visualizer import GradCAMExplainer, compute_cam_drift

st.set_page_config(
    page_title="Federated Unlearning (FedEraser) with Grad-CAM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Federated Visual Unlearning with Auditing & Visualization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Implementing the <b>Right to be Forgotten</b> in Federated Learning using <b>FedEraser</b> and <b>Grad-CAM Attention Auditing</b> on CIFAR-100</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Experiment Configuration")

exp_mode = st.sidebar.selectbox(
    "Experiment Partition Mode",
    options=["Multi-Class (Dirichlet Non-IID α=0.5)", "Single-Class (1 Client = 1 Class)"],
    index=0
)
is_mc = "Multi-Class" in exp_mode
ckpt_dir = "checkpoints/multi_class" if is_mc else "checkpoints"
bench_file = "results/multi_class/unlearning_benchmark.json" if is_mc else "results/unlearning_benchmark.json"
gallery_img_path = "results/multi_class/gradcam_unlearning_comparison.png" if is_mc else "results/gradcam_unlearning_comparison.png"

# Cache data manager and models
@st.cache_resource
def load_resources(is_multiclass: bool, num_classes=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    if is_multiclass:
        dm = MultiClassFLDataManager(
            root="dataset",
            selected_classes=list(range(num_classes)),
            num_clients=10,
            alpha=0.5,
            batch_size=32,
            num_workers=0
        )
        c_dir = "checkpoints/multi_class"
    else:
        dm = SingleClassFLDataManager(
            root="dataset",
            selected_classes=list(range(num_classes)),
            batch_size=32,
            num_workers=0
        )
        c_dir = "checkpoints"
    
    models = {}
    m_orig_path = os.path.join(c_dir, "M_orig.pt")
    if os.path.exists(m_orig_path):
        m_orig = get_cifar_model(num_classes=num_classes, device=device)
        m_orig.load_state_dict(torch.load(m_orig_path, map_location=device))
        m_orig.eval()
        models["Original (M_orig)"] = m_orig
        
    m_unl_path = os.path.join(c_dir, "M_unlearn.pt")
    if os.path.exists(m_unl_path):
        m_unlearn = get_cifar_model(num_classes=num_classes, device=device)
        m_unlearn.load_state_dict(torch.load(m_unl_path, map_location=device))
        m_unlearn.eval()
        models["FedEraser (M_unlearn)"] = m_unlearn
        
    m_ret_path = os.path.join(c_dir, "M_retrain.pt")
    if os.path.exists(m_ret_path):
        m_retrain = get_cifar_model(num_classes=num_classes, device=device)
        m_retrain.load_state_dict(torch.load(m_ret_path, map_location=device))
        m_retrain.eval()
        models["Retrained Gold (M_retrain)"] = m_retrain
        
    return dm, models, device

data_manager, models_dict, device = load_resources(is_mc)

if is_mc:
    forget_client_id = st.sidebar.selectbox(
        "Target Forgotten Client ID",
        options=list(range(data_manager.num_clients)),
        index=9,
        format_func=lambda cid: f"Client {cid:02d} ({len(data_manager.client_indices[cid])} samples, {len(data_manager.get_client_class_distribution(cid))} classes)"
    )
    forget_class_name = f"Client {forget_client_id}"
else:
    forget_client_id = st.sidebar.selectbox(
        "Target Forgotten Client ID",
        options=list(range(data_manager.num_classes)),
        index=min(9, data_manager.num_classes - 1),
        format_func=lambda cid: f"Client {cid:02d} - {data_manager.class_names[cid]}"
    )
    forget_class_name = data_manager.class_names[forget_client_id]

cam_mode = st.sidebar.radio(
    "Grad-CAM Target Class",
    options=["True Class (Examine Target Class Attention)", "Predicted Class (Examine Model Focus)"],
    index=0
)
alpha = st.sidebar.slider("Heatmap Overlay Transparency", 0.1, 0.9, 0.5, 0.05)

# Tabs
tab_metrics, tab_gallery, tab_interactive, tab_dist, tab_algo = st.tabs([
    "📊 Benchmark Metrics",
    "🖼️ Visual Evidence Gallery",
    "🔍 Live Sample Inspector",
    "📈 Client Data Distributions",
    "📐 FedEraser Architecture"
])

# -----------------------------------------------------------------------------
# TAB 1: BENCHMARK METRICS
# -----------------------------------------------------------------------------
with tab_metrics:
    mode_str = "Multi-Class Non-IID (Dirichlet α=0.5)" if is_mc else "Single-Class Per Client (1 Client = 1 Class)"
    st.subheader(f"Experimental Results: {mode_str}")
    
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            b_data = json.load(f)
            
        m_data = b_data.get("models", {})
        m_orig = m_data.get("original", {})
        m_unl = m_data.get("federaser", {})
        m_ret = m_data.get("retrained", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Holdout Test Accuracy",
                f"{m_unl.get('overall_acc', 0.0):.2%}",
                delta=f"{(m_unl.get('overall_acc', 0.0) - m_orig.get('overall_acc', 0.0)):+.2%} vs Original"
            )
        with col2:
            if is_mc:
                agreement = m_unl.get("prediction_agreement_with_retrain", 0.0)
                st.metric(
                    "Agreement with Retrained Gold",
                    f"{agreement:.1f}%",
                    delta="Fidelity on D_forget"
                )
            else:
                st.metric(
                    "Forget Class Accuracy",
                    f"{m_unl.get('forget_acc', 0.0):.2%}",
                    delta=f"{(m_unl.get('forget_acc', 0.0) - m_orig.get('forget_acc', 0.0)):-.2%} (Erased)",
                    delta_color="inverse"
                )
        with col3:
            st.metric(
                "Forget Data Softmax Conf",
                f"{m_unl.get('forget_conf', 0.0):.3f}",
                delta=f"{(m_unl.get('forget_conf', 0.0) - m_orig.get('forget_conf', 0.0)):-.3f}",
                delta_color="inverse"
            )
        with col4:
            st.metric(
                "Unlearning Speedup",
                f"{m_unl.get('speedup_vs_retrain', 1.0):.2f}x",
                delta="Faster than Retraining"
            )

        st.markdown("### Detailed Comparison Table")
        table_rows = [
            {
                "Model / Evaluation": "Original FL Model (M_orig)",
                "Test Acc": f"{m_orig.get('overall_acc', 0.0):.2%}",
                "Forget Acc": f"{m_orig.get('forget_acc', 0.0):.2%}",
                "Forget Conf": f"{m_orig.get('forget_conf', 0.0):.3f}",
                "Runtime": f"{m_orig.get('runtime_seconds', 0.0):.2f}s",
                "Speedup": "1.0x (Baseline)"
            },
            {
                "Model / Evaluation": "FedEraser Unlearned (M_unlearn)",
                "Test Acc": f"{m_unl.get('overall_acc', 0.0):.2%}",
                "Forget Acc": f"{m_unl.get('forget_acc', 0.0):.2%}",
                "Forget Conf": f"{m_unl.get('forget_conf', 0.0):.3f}",
                "Runtime": f"{m_unl.get('runtime_seconds', 0.0):.2f}s",
                "Speedup": f"{m_unl.get('speedup_vs_retrain', 1.0):.2f}x faster"
            },
            {
                "Model / Evaluation": "Exact Retraining Gold Standard (M_retrain)",
                "Test Acc": f"{m_ret.get('overall_acc', 0.0):.2%}",
                "Forget Acc": f"{m_ret.get('forget_acc', 0.0):.2%}",
                "Forget Conf": f"{m_ret.get('forget_conf', 0.0):.3f}",
                "Runtime": f"{m_ret.get('runtime_seconds', 0.0):.2f}s",
                "Speedup": "(Full Retraining)"
            }
        ]
        st.table(table_rows)

        if is_mc:
            st.info(
                f"**Multi-Class Unlearning Fidelity**: When Client {b_data.get('forget_client')} requests unlearning, "
                f"**FedEraser** reconstructs the model along the training trajectory in only **{m_unl.get('runtime_seconds', 0.0):.2f}s** "
                f"(a **{m_unl.get('speedup_vs_retrain', 1.0):.2f}x speedup** over retraining), maintaining **{m_unl.get('overall_acc', 0.0):.2%}** holdout test accuracy. "
                f"On the forgotten client's private training data ($D_{{\\text{{forget}}}}$), $M_{{\\text{{unlearn}}}}$ achieves **{m_unl.get('prediction_agreement_with_retrain', 0.0):.1f}% prediction agreement** "
                f"with the gold-standard retrained model, while the memorization loss drops to the unseen baseline ({m_unl.get('forget_train_loss', 0.0):.4f} vs {m_ret.get('forget_train_loss', 0.0):.4f})."
            )
        else:
            st.info(
                f"**Single-Class Unlearning Finding**: When Client {b_data.get('forget_client')} ('{b_data.get('forget_class_name')}') requests unlearning, "
                f"**FedEraser** reconstructs the model along the training trajectory in only **{m_unl.get('runtime_seconds', 0.0):.2f}s** "
                f"(a **{m_unl.get('speedup_vs_retrain', 1.0):.2f}x speedup** over full retraining), achieving **{m_unl.get('retain_acc', 0.0):.2%}** retain accuracy "
                f"while eradicating the target client's concept."
            )
    else:
        st.warning(f"Benchmark results not found at `{bench_file}`. Please run the experiment first.")

# -----------------------------------------------------------------------------
# TAB 2: VISUAL EVIDENCE GALLERY
# -----------------------------------------------------------------------------
with tab_gallery:
    st.subheader("Side-by-Side Visual Attribution Evidence (Grad-CAM)")
    st.markdown(
        "Visual evidence demonstrating how attribution maps evolve between the Original FL model, "
        "the FedEraser unlearned model, and the Retrained gold standard baseline."
    )
    if os.path.exists(gallery_img_path):
        st.image(gallery_img_path, use_container_width=True)
    else:
        st.warning(f"Gallery image not found at `{gallery_img_path}`.")

# -----------------------------------------------------------------------------
# TAB 3: LIVE SAMPLE INSPECTOR
# -----------------------------------------------------------------------------
with tab_interactive:
    st.subheader("Interactive Grad-CAM Attribution Inspector")
    st.markdown("Select any class or client image to inspect real-time Grad-CAM activations across all three models.")

    if is_mc:
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            inspect_cid = st.selectbox(
                "Select Client to Inspect",
                options=list(range(data_manager.num_clients)),
                index=forget_client_id,
                format_func=lambda cid: f"Client {cid:02d} {'[FORGOTTEN CLIENT]' if cid == forget_client_id else '[RETAIN CLIENT]'}"
            )
        with col_select2:
            img_idx = st.slider("Select Private Image Sample Index", 0, min(15, len(data_manager.client_indices[inspect_cid])-1), 0)

        imgs, lbls = data_manager.get_sample_images_for_client(inspect_cid, count=16)
    else:
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            inspect_cid = st.selectbox(
                "Select Client / Class to Inspect",
                options=list(range(data_manager.num_classes)),
                index=forget_client_id,
                format_func=lambda cid: f"Client {cid:02d} ({data_manager.class_names[cid]}) {'[FORGOTTEN CLIENT]' if cid == forget_client_id else '[RETAIN CLIENT]'}"
            )
        with col_select2:
            img_idx = st.slider("Select Image Sample Index", 0, 15, 0)

        imgs, lbls = data_manager.get_sample_images_for_class(inspect_cid, count=16)

    img_t = imgs[img_idx]
    true_label = lbls[img_idx]
    true_name = data_manager.class_names[true_label]
    img_rgb = data_manager.unnormalizer(img_t)

    target_class = true_label if "True Class" in cam_mode else None

    # Compute CAMs across available models
    cams = {}
    preds = {}
    confs = {}

    for name, m in models_dict.items():
        expl = GradCAMExplainer(m, m.gradcam_target_layer, device=device)
        cam, pred_c, conf, _ = expl.generate_cam(img_t, target_class=target_class)
        cams[name] = cam
        preds[name] = pred_c
        confs[name] = conf

    # Render Side-by-Side Comparison Columns
    cols_display = st.columns(len(models_dict) + 2)
    
    with cols_display[0]:
        st.markdown("**Input Image**")
        st.image(img_rgb, caption=f"True Class: {true_name}", use_container_width=True)

    col_idx = 1
    for name in models_dict.keys():
        with cols_display[col_idx]:
            st.markdown(f"**{name}**")
            overlay = GradCAMExplainer.overlay_heatmap(img_rgb, cams[name], alpha=alpha)
            pred_name = data_manager.class_names[preds[name]]
            st.image(overlay, caption=f"Pred: {pred_name} ({confs[name]:.1%})", use_container_width=True)
            col_idx += 1

    # Difference Heatmap
    if "Original (M_orig)" in cams and "FedEraser (M_unlearn)" in cams:
        with cols_display[col_idx]:
            st.markdown("**Attribution Shift (|ΔCAM|)**")
            diff_cam = np.abs(cams["Original (M_orig)"] - cams["FedEraser (M_unlearn)"])
            diff_norm = (diff_cam - diff_cam.min()) / (diff_cam.max() - diff_cam.min() + 1e-8)
            diff_overlay = GradCAMExplainer.overlay_heatmap(img_rgb, diff_norm, alpha=0.6, colormap=cv2.COLORMAP_MAGMA)
            drift = compute_cam_drift(cams["Original (M_orig)"], cams["FedEraser (M_unlearn)"])
            st.image(diff_overlay, caption=f"CAM Drift: {drift:.3f}", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: CLIENT DATA DISTRIBUTIONS
# -----------------------------------------------------------------------------
with tab_dist:
    st.subheader("Federated Client Data Distribution Analysis")
    if is_mc:
        st.markdown("Visualizing class label distributions across all 10 clients under Non-IID Dirichlet partitioning ($\alpha=0.5$).")
        dist_data = {f"Client {cid:02d}": data_manager.get_client_class_distribution(cid) for cid in range(data_manager.num_clients)}
        
        # Display as a dataframe
        all_c_names = data_manager.class_names
        table_matrix = []
        for cid in range(data_manager.num_clients):
            cdist = dist_data[f"Client {cid:02d}"]
            row = {"Client": f"Client {cid:02d} {'[FORGOTTEN]' if cid == forget_client_id else ''}"}
            total_samples = 0
            for cname in all_c_names:
                cnt = cdist.get(cname, 0)
                row[cname] = cnt
                total_samples += cnt
            row["Total Samples"] = total_samples
            table_matrix.append(row)
        st.dataframe(table_matrix, use_container_width=True)
    else:
        st.markdown("In the single-class simplification, each client holds exclusively 1 class (500 training images per client).")
        st.dataframe([
            {"Client": f"Client {cid:02d}", "Class": data_manager.class_names[cid], "Train Samples": 500, "Role": "FORGOTTEN CLIENT" if cid == forget_client_id else "Retain Client"}
            for cid in range(data_manager.num_classes)
        ], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: ARCHITECTURE & FEDERASER METHOD
# -----------------------------------------------------------------------------
with tab_algo:
    st.subheader("FedEraser: Enabling Efficient Client-Level Data Removal in Federated Learning")
    st.markdown("""
    ### 🎯 The Challenge: Right to be Forgotten in Federated Learning
    Under GDPR Article 17, when a participating client requests their data be removed from the global model:
    - **Exact Retraining from Scratch ($M_{\\text{retrain}}$)**: Discards the model and repeats the entire training process without the forgotten client across all communication rounds. While sound, it imposes massive computation and communication costs ($O(T \\times E)$).
    - **Naive Zeroing / Masking**: Setting class output weights or biases to arbitrary negative constants is an artificial heuristic that fails to alter the underlying feature representations.

    ### ⚡ How FedEraser Works
    FedEraser (Wang et al., IEEE/ACM IWQoS 2021) trades server storage for unlearning computation speed:
    1. **Historical Update Logging**: During federated training, the server records global checkpoints $W_0, W_1, \\dots, W_T$ and client updates $\\Delta W_{t, k}$.
    2. **Trajectory Calibration**: When client $u$ requests unlearning, the server reconstructs the model starting from $W_0$.
    3. **Calibrated Updates**: At each milestone round $t$, retained clients ($k \\neq u$) execute fast calibration steps ($E_{\\text{cali}} \\ll E$) to find new update directions $\\Delta W'_{t, k}$.
    4. **Magnitude Scaling**: The server scales the calibrated update to match the historical momentum bounds:
    $$\\widetilde{\\Delta W}_{t, k} = \\frac{\\|\\Delta W_{t, k}\\|}{\\|\\Delta W'_{t, k}\\| + \\epsilon} \\cdot \\Delta W'_{t, k}$$
    5. **Aggregation**: The server updates $W'_t = W'_{t-1} + \\sum_{k \\in \\mathcal{R}} \\frac{n_k}{N_{\\mathcal{R}}} \\widetilde{\\Delta W}_{t, k}$.
    """)
