import os
import json
import glob
import numpy as np
from scipy import stats

def aggregate_seeds():
    distilled_files = glob.glob("eval/student_distilled_metrics_seed*.json")
    scratch_files = glob.glob("eval/student_scratch_metrics_seed*.json")
    
    dist_results = []
    scratch_results = []
    
    for f in distilled_files:
        with open(f, 'r') as file:
            dist_results.append(json.load(file))
            
    for f in scratch_files:
        with open(f, 'r') as file:
            scratch_results.append(json.load(file))
            
    n_dist = len(dist_results)
    n_scratch = len(scratch_results)
    
    print(f"Found {n_dist} distilled seed files and {n_scratch} scratch seed files.")
    
    if n_dist == 0 and n_scratch == 0:
        print("No seed files found. Run run_multiseed.py first.")
        return
        
    metrics_to_agg = ["final_macro_f1", "final_micro_f1", "final_hamming_acc"]
    
    def get_stats(results_list):
        if not results_list: return {}
        stats_dict = {}
        for m in metrics_to_agg:
            vals = [r.get(m, 0.0) for r in results_list]
            stats_dict[m] = {
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "values": vals
            }
        return stats_dict
        
    dist_stats = get_stats(dist_results)
    scratch_stats = get_stats(scratch_results)
    
    out_json = {
        "n_distilled": n_dist,
        "n_scratch": n_scratch,
        "distilled_stats": dist_stats,
        "scratch_stats": scratch_stats,
        "ttest": {}
    }
    
    markdown = "# Multi-Seed Aggregation Results\n\n"
    markdown += f"**Distilled runs (n={n_dist})**\n"
    if n_dist > 0:
        markdown += f"- Macro F1: {dist_stats['final_macro_f1']['mean']:.4f} ± {dist_stats['final_macro_f1']['std']:.4f}\n"
        markdown += f"- Micro F1: {dist_stats['final_micro_f1']['mean']:.4f} ± {dist_stats['final_micro_f1']['std']:.4f}\n"
        markdown += f"- Hamming Acc: {dist_stats['final_hamming_acc']['mean']:.4f} ± {dist_stats['final_hamming_acc']['std']:.4f}\n"
        
    markdown += f"\n**Scratch runs (n={n_scratch})**\n"
    if n_scratch > 0:
        markdown += f"- Macro F1: {scratch_stats['final_macro_f1']['mean']:.4f} ± {scratch_stats['final_macro_f1']['std']:.4f}\n"
        markdown += f"- Micro F1: {scratch_stats['final_micro_f1']['mean']:.4f} ± {scratch_stats['final_micro_f1']['std']:.4f}\n"
        markdown += f"- Hamming Acc: {scratch_stats['final_hamming_acc']['mean']:.4f} ± {scratch_stats['final_hamming_acc']['std']:.4f}\n"
        
    if n_dist > 1 and n_scratch > 1:
        markdown += "\n## Statistical Significance (Welch's t-test)\n"
        
        for metric in metrics_to_agg:
            dist_vals = dist_stats[metric]['values']
            scratch_vals = scratch_stats[metric]['values']
            
            diff_mean = np.mean(dist_vals) - np.mean(scratch_vals)
            diff_var = (np.var(dist_vals, ddof=1) / n_dist) + (np.var(scratch_vals, ddof=1) / n_scratch)
            diff_std = np.sqrt(diff_var)
            
            t_stat, p_val = stats.ttest_ind(dist_vals, scratch_vals, equal_var=False)
            
            out_json["ttest"][metric] = {
                "mean_diff_distilled_minus_scratch": float(diff_mean),
                "std_of_diff": float(diff_std),
                "t_statistic": float(t_stat),
                "p_value": float(p_val)
            }
            
            metric_name = metric.replace("final_", "").replace("_", " ").title()
            
            markdown += f"Comparing Distilled vs Scratch on {metric_name} (n={n_dist} vs n={n_scratch}):\n"
            markdown += f"- Mean Difference (Distilled - Scratch): {diff_mean:.4f} ± {diff_std:.4f}\n"
            markdown += f"- t-statistic: {t_stat:.4f}\n"
            markdown += f"- p-value: {p_val:.4f}\n"
            if p_val < 0.05:
                markdown += "=> The difference IS statistically significant (p < 0.05).\n\n"
            else:
                markdown += "=> The difference is NOT statistically significant (p >= 0.05).\n\n"
    else:
        markdown += "\n*Not enough data for Welch's t-test (requires n>1 for both).* \n"
        
    with open("eval/multiseed_results.json", "w") as f:
        json.dump(out_json, f, indent=4)
        
    with open("eval/multiseed_summary.md", "w") as f:
        f.write(markdown)
        
    print(markdown)

if __name__ == "__main__":
    aggregate_seeds()
