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
st.warning("Per-epoch training histories (loss and F1 curves) were not saved to disk in the metric JSON artifacts during the training stages. Therefore, training curves cannot be rendered.")
st.markdown("""
**Note on Stage 11 Findings (Loss/F1 Divergence):**
Even though we cannot visualize the curves, the multi-seed training consistently demonstrated that the scratch student's validation loss bottoms out around **epoch 5**, while its macro F1 continues to climb until **epochs 7-8**. Selecting checkpoints based solely on validation loss systematically costs the scratch model roughly 0.04 macro F1 relative to its peak. Distillation acts as a regularizer, preventing this early loss divergence.
""")

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
