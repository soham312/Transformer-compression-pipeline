import streamlit as st
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Transformer Compression Pipeline", layout="wide")

st.title("Transformer Compression Pipeline")
st.markdown("Visualizing the compression, latency, and accuracy trade-offs of knowledge distillation and quantization on BERT.")

# Helper to load JSON safely
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# Load Data
benchmark_data = load_json("eval/benchmark_results.json")
teacher_data = load_json("eval/teacher_baseline.json")
multiseed_data = load_json("eval/multiseed_results.json")

# Sidebar
st.sidebar.header("Filters & Metadata")
variants_to_show = st.sidebar.multiseed = []
if benchmark_data:
    meta = benchmark_data.get("metadata", {})
    st.sidebar.markdown("**Benchmark Conditions:**")
    st.sidebar.markdown(f"- **Device:** {meta.get('device', 'cpu')}")
    st.sidebar.markdown(f"- **Threads:** {meta.get('thread_count', 'N/A')}")
    st.sidebar.markdown(f"- **Batch Size:** {meta.get('batch_size', 'N/A')}")
    st.sidebar.markdown(f"- **Repetitions:** {meta.get('timing_repetitions', 'N/A')}")
    
    all_variants = list(benchmark_data.get("results", {}).keys())
    selected_variants = st.sidebar.multiselect(
        "Select Variants to Display (Scatter Plot)", 
        options=all_variants,
        default=all_variants
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Caveat for PyTorch Static Quantization:**\nPyTorch static quantization was applied to the classifier layer only, because nn.TransformerEncoderLayer crashes during eager-mode calibration. Its numbers are not a clean test of static quantization and must not be presented as one.")

# ---------------------------------------------------------
# 1. Trade-off Surface
# ---------------------------------------------------------
st.header("1. The Trade-Off Surface")
if benchmark_data and "results" in benchmark_data:
    results = benchmark_data["results"]
    
    # Filter variants based on selection
    display_results = {k: v for k, v in results.items() if k in selected_variants}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Size vs. Macro F1 (Marker size = Latency)")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = []
        sizes = []
        f1s = []
        latencies = []
        
        for k, v in display_results.items():
            labels.append(k)
            sizes.append(v.get("size_mb", 0))
            f1s.append(v.get("macro_f1", 0))
            latencies.append(v.get("median_latency_ms", 0))
            
        sizes = np.array(sizes)
        f1s = np.array(f1s)
        latencies = np.array(latencies)
        
        # Scale marker size: min size 50, max size 500 based on latency
        if len(latencies) > 0:
            norm_lat = (latencies - latencies.min()) / (latencies.max() - latencies.min() + 1e-9)
            marker_sizes = 50 + norm_lat * 450
            
            scatter = ax.scatter(sizes, f1s, s=marker_sizes, alpha=0.7, edgecolors='w', c=np.arange(len(labels)), cmap='tab10')
            
            for i, label in enumerate(labels):
                ax.annotate(
                    label, 
                    (sizes[i], f1s[i]), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold'
                )
                
            ax.set_xscale("log")
            ax.set_xlabel("Model Size (MB) [Log Scale]")
            ax.set_ylabel("Macro F1")
            ax.set_title("Accuracy vs. Compression")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            st.pyplot(fig)
            st.caption("Teacher sits top-right (accurate, huge). ONNX INT8 sits mid-left (nearly as accurate, 38× smaller). Marker size is proportional to inference latency.")
            
    with col2:
        st.subheader("Compression & Speedup vs. Teacher")
        
        if "teacher" in results:
            teacher_size = results["teacher"].get("size_mb", 1)
            teacher_lat = results["teacher"].get("median_latency_ms", 1)
            
            bar_labels = []
            size_reductions = []
            speedups = []
            
            for k, v in display_results.items():
                if k != "teacher":
                    bar_labels.append(k)
                    size_reductions.append(teacher_size / v.get("size_mb", 1))
                    speedups.append(teacher_lat / v.get("median_latency_ms", 1))
            
            if bar_labels:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                x = np.arange(len(bar_labels))
                width = 0.35
                
                ax2.bar(x - width/2, size_reductions, width, label='Size Reduction (x)', color='skyblue')
                ax2.bar(x + width/2, speedups, width, label='Speedup (x)', color='lightgreen')
                
                ax2.set_ylabel('Factor (Higher is better)')
                ax2.set_title('Improvements Relative to Teacher')
                ax2.set_xticks(x)
                ax2.set_xticklabels(bar_labels, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
                
                st.pyplot(fig2)
                st.caption("Read magnitudes of compression and speedup directly. ONNX INT8 achieves ~38x size reduction.")
else:
    st.error("Artifact not found: eval/benchmark_results.json")

st.markdown("---")

# ---------------------------------------------------------
# 2. Training Curves
# ---------------------------------------------------------
st.header("2. Training Curves")

histories_data = load_json("eval/training_histories.json")

if histories_data:
    col1, col2 = st.columns(2)
    
    def plot_model_history(ax1, ax2, model_type, data):
        # We'll plot individual seed lines with low opacity, and the mean overlaid
        seeds = list(data.keys())
        if not seeds:
            return
            
        # Determine max length
        max_epochs = max([len(data[s]["train_loss"]) for s in seeds])
        epochs = np.arange(1, max_epochs + 1)
        
        # Prepare arrays for means
        # Pad with NaNs if lengths differ
        def pad_array(arr, length):
            if len(arr) == length:
                return np.array(arr)
            return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
            
        all_val_loss = np.array([pad_array(data[s]["val_loss"], max_epochs) for s in seeds])
        all_val_f1 = np.array([pad_array(data[s]["val_macro_f1"], max_epochs) for s in seeds])
        all_train_loss = np.array([pad_array(data[s]["train_loss"], max_epochs) for s in seeds])
        selected_epochs = [data[s]["selected_epoch"] for s in seeds if data[s].get("selected_epoch") is not None]
        
        # Plot individual lines
        for s in seeds:
            e_len = len(data[s]["val_loss"])
            ax1.plot(np.arange(1, e_len + 1), data[s]["val_loss"], color='blue', alpha=0.15)
            ax1.plot(np.arange(1, e_len + 1), data[s]["train_loss"], color='gray', alpha=0.15)
            ax2.plot(np.arange(1, e_len + 1), data[s]["val_macro_f1"], color='red', alpha=0.15)
            
        # Plot means
        mean_val_loss = np.nanmean(all_val_loss, axis=0)
        mean_train_loss = np.nanmean(all_train_loss, axis=0)
        mean_val_f1 = np.nanmean(all_val_f1, axis=0)
        
        ax1.plot(epochs, mean_val_loss, color='blue', linewidth=2.5, label='Val Loss (Mean)')
        ax1.plot(epochs, mean_train_loss, color='gray', linewidth=2.5, label='Train Loss (Mean)')
        ax2.plot(epochs, mean_val_f1, color='red', linewidth=2.5, label='Val Macro F1 (Mean)')
        
        # Mark selected checkpoints
        for se in selected_epochs:
            ax1.axvline(x=se, color='blue', linestyle='--', alpha=0.3)
            
        if selected_epochs:
            mean_se = np.mean(selected_epochs)
            ax1.axvline(x=mean_se, color='blue', linestyle='-', linewidth=2, alpha=0.8, label=f'Avg Checkpoint (Epoch {mean_se:.1f})')
            
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss")
        ax2.set_ylabel("Macro F1")
        
        ax1.set_title(f"{model_type.capitalize()} Student")
        
        # Legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
        
    with col1:
        fig_dist, ax1_dist = plt.subplots(figsize=(8, 5))
        ax2_dist = ax1_dist.twinx()
        if "distilled" in histories_data:
            plot_model_history(ax1_dist, ax2_dist, "distilled", histories_data["distilled"])
        st.pyplot(fig_dist)

    with col2:
        fig_scratch, ax1_scratch = plt.subplots(figsize=(8, 5))
        ax2_scratch = ax1_scratch.twinx()
        if "scratch" in histories_data:
            plot_model_history(ax1_scratch, ax2_scratch, "scratch", histories_data["scratch"])
        st.pyplot(fig_scratch)

    st.caption("**Key Finding (Loss/F1 Divergence):** Notice that the scratch student's validation loss bottoms out at epoch 5 in every seed while its macro F1 keeps climbing to epochs 7–8. Selecting checkpoints strictly on validation loss systematically costs the scratch model relative to its peak F1. Distillation acts as a regularizer—its validation loss safely keeps improving alongside F1 to epochs 7–8.")
else:
    st.warning("Per-epoch training histories (loss and F1 curves) not found at eval/training_histories.json. Run parse_training_logs.py first.")

st.markdown("---")

# ---------------------------------------------------------
# 3. Statistical Significance
# ---------------------------------------------------------
st.header("3. Statistical Significance (Multi-Seed)")

if multiseed_data:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Distilled vs. Scratch Spread")
        
        dist_stats = multiseed_data.get("distilled_stats", {})
        scratch_stats = multiseed_data.get("scratch_stats", {})
        
        metrics = ["final_macro_f1", "final_micro_f1", "final_hamming_acc"]
        metric_names = ["Macro F1", "Micro F1", "Hamming Acc"]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for i, (m, name) in enumerate(zip(metrics, metric_names)):
            if m in dist_stats and m in scratch_stats:
                dist_vals = dist_stats[m].get("values", [])
                scratch_vals = scratch_stats[m].get("values", [])
                
                # Strip plot using scatter
                axes[i].scatter(np.zeros(len(dist_vals)) + 0.1 * np.random.randn(len(dist_vals)), dist_vals, alpha=0.7, label='Distilled')
                axes[i].scatter(np.ones(len(scratch_vals)) + 0.1 * np.random.randn(len(scratch_vals)), scratch_vals, alpha=0.7, label='Scratch')
                
                axes[i].set_xticks([0, 1])
                axes[i].set_xticklabels(['Distilled', 'Scratch'])
                axes[i].set_title(name)
                axes[i].grid(True, axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Strip plot showing the actual spread of 5 seed runs. Notice the micro F1 gap is clear, while macro F1 overlaps heavily.")

    with col2:
        st.subheader("Welch's t-test Results")
        ttest_data = multiseed_data.get("ttest", {})
        
        for metric, name in zip(metrics, metric_names):
            if metric in ttest_data:
                data = ttest_data[metric]
                mean_diff = data.get("mean_diff_distilled_minus_scratch", 0)
                pval = data.get("p_value", 1)
                tstat = data.get("t_statistic", 0)
                
                is_sig = pval < 0.05
                color = "green" if is_sig else "red"
                sig_text = "SIGNIFICANT" if is_sig else "NOT SIGNIFICANT"
                
                st.markdown(f"**{name}:** Diff = {mean_diff:+.4f} | p = {pval:.4f} | t = {tstat:.2f}")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{sig_text}</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
    st.info("""
    **The Mechanism behind the Significance Split:**
    Distillation transfers what the teacher knows. The teacher scores zero F1 on five rare classes (grief, pride, relief, embarrassment, nervousness). Macro F1 weights all 28 classes equally, so it inherits nothing for these rare classes. Micro F1 and Hamming accuracy, however, weight every label decision equally and are dominated by frequent classes (like gratitude, amusement, love) where the teacher is very strong. Thus, distillation provides a highly significant boost to Micro F1 and Hamming Accuracy, but none to Macro F1.
    """)
else:
    st.error("Artifact not found: eval/multiseed_results.json")

st.markdown("---")

# ---------------------------------------------------------
# 4. Per-Class Diagnostics
# ---------------------------------------------------------
st.header("4. Per-Class Diagnostics (Teacher)")

if teacher_data and "per_class" in teacher_data:
    per_class = teacher_data["per_class"]
    
    # Sort by support (tp + fn approximates support if recall is available, or use actual counts if available)
    # The teacher baseline JSON has tp, fp, fn, tn. Support = tp + fn
    classes = []
    f1s = []
    supports = []
    
    for cls, metrics in per_class.items():
        classes.append(cls)
        f1s.append(metrics.get("f1", 0))
        supports.append(metrics.get("tp", 0) + metrics.get("fn", 0))
        
    # Sort by support descending
    sorted_indices = np.argsort(supports)[::-1]
    classes = np.array(classes)[sorted_indices]
    f1s = np.array(f1s)[sorted_indices]
    supports = np.array(supports)[sorted_indices]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax2 = ax1.twinx()
    
    # Bar for support
    ax1.bar(classes, supports, alpha=0.3, color='gray', label='Support (True Count)')
    
    # Line for F1
    ax2.plot(classes, f1s, marker='o', color='red', linewidth=2, label='Teacher F1')
    
    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, rotation=90)
    ax1.set_ylabel("Support Count")
    ax2.set_ylabel("F1 Score")
    ax1.set_title("Teacher Per-Class F1 vs. Support (Rare-class collapse)")
    
    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    st.pyplot(fig)
    st.caption("Notice the rare-class collapse on the right side: classes with low support count have exactly 0.0 F1 score.")
    
    # Calibration Reliability Plot
    if os.path.exists("eval/calibration_reliability.png"):
        st.subheader("Teacher Calibration Reliability")
        st.image("eval/calibration_reliability.png", caption="Calibration reliability plot (Stage 4). Shows how well predicted probabilities match empirical frequencies.")
    else:
        st.warning("Artifact not found: eval/calibration_reliability.png")

else:
    st.error("Artifact not found: eval/teacher_baseline.json")

st.markdown("---")
st.markdown("Dashboard generated by the Antigravity Streamlit builder.")
