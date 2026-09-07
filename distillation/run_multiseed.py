import os
import sys
import subprocess

SEEDS = [42, 123, 456, 789, 1011]

def run_multiseed():
    print("Starting multi-seed training...")
    
    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"Running seed {seed}")
        print(f"{'='*50}")
        
        # 1. Distilled Student
        distilled_out = f"model_checkpoints/student_distilled_seed{seed}"
        distilled_metrics = f"eval/student_distilled_metrics_seed{seed}.json"
        
        if os.path.exists(distilled_metrics):
            print(f"Skipping distilled student for seed {seed} (metrics exist).")
        else:
            print(f"Training distilled student for seed {seed}...")
            cmd_dist = [
                sys.executable, "distillation/train_student.py",
                "--output_dir", distilled_out,
                "--metrics_file", distilled_metrics,
                "--seed", str(seed)
            ]
            subprocess.run(cmd_dist, check=True)
            
        # 2. Scratch Student
        scratch_out = f"model_checkpoints/student_scratch_seed{seed}"
        scratch_metrics = f"eval/student_scratch_metrics_seed{seed}.json"
        
        if os.path.exists(scratch_metrics):
            print(f"Skipping scratch student for seed {seed} (metrics exist).")
        else:
            print(f"Training scratch student for seed {seed}...")
            cmd_scratch = [
                sys.executable, "distillation/train_scratch.py",
                "--output_dir", scratch_out,
                "--metrics_file", scratch_metrics,
                "--seed", str(seed)
            ]
            subprocess.run(cmd_scratch, check=True)

    print("\nMulti-seed training complete!")

if __name__ == "__main__":
    run_multiseed()
