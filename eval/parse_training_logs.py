import re
import json
import os

def parse_logs(log_text):
    histories = {"distilled": {}, "scratch": {}}
    
    current_model = None
    current_seed = None
    
    # State variables for the current run
    run_train_loss = []
    run_val_loss = []
    run_val_macro = []
    run_val_micro = []
    run_val_hamming = []
    selected_epoch = None
    current_epoch = None
    
    lines = log_text.splitlines()
    for line in lines:
        # Match start of a run: "Training distilled student for seed 42..."
        m_start = re.match(r"Training (distilled|scratch) student for seed (\d+)\.\.\.", line)
        if m_start:
            # Save previous run if it exists
            if current_model and current_seed:
                histories[current_model][str(current_seed)] = {
                    "train_loss": run_train_loss,
                    "val_loss": run_val_loss,
                    "val_macro_f1": run_val_macro,
                    "val_micro_f1": run_val_micro,
                    "val_hamming_acc": run_val_hamming,
                    "selected_epoch": selected_epoch
                }
                
            current_model = m_start.group(1)
            current_seed = int(m_start.group(2))
            run_train_loss = []
            run_val_loss = []
            run_val_macro = []
            run_val_micro = []
            run_val_hamming = []
            selected_epoch = None
            current_epoch = None
            continue
            
        if not current_model:
            continue
            
        m_epoch = re.match(r"Epoch (\d+)/\d+", line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            continue
            
        m_loss = re.match(r"Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)", line)
        if m_loss:
            run_train_loss.append(float(m_loss.group(1)))
            run_val_loss.append(float(m_loss.group(2)))
            continue
            
        m_f1 = re.match(r"Val Macro F1: ([\d.]+) \| Val Micro F1: ([\d.]+) \| Val Hamming Acc: ([\d.]+)", line)
        if m_f1:
            run_val_macro.append(float(m_f1.group(1)))
            run_val_micro.append(float(m_f1.group(2)))
            run_val_hamming.append(float(m_f1.group(3)))
            continue
            
        if "Validation loss improved." in line:
            selected_epoch = current_epoch
            
    # Save the last run
    if current_model and current_seed:
        histories[current_model][str(current_seed)] = {
            "train_loss": run_train_loss,
            "val_loss": run_val_loss,
            "val_macro_f1": run_val_macro,
            "val_micro_f1": run_val_micro,
            "val_hamming_acc": run_val_hamming,
            "selected_epoch": selected_epoch
        }
        
    return histories

def main():
    log_file = "multiseed_log.txt"
    out_file = "eval/training_histories.json"
    
    if not os.path.exists(log_file):
        print(f"File {log_file} not found.")
        return
        
    with open(log_file, "r") as f:
        log_text = f.read()
        
    histories = parse_logs(log_text)
    
    with open(out_file, "w") as f:
        json.dump(histories, f, indent=4)
        
    print(f"Saved parsed histories to {out_file}")

if __name__ == "__main__":
    main()
