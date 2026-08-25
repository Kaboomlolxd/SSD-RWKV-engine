import pandas as pd

def simulate_read_disturb():
    print("Simulating NAND Read Disturb Mitigation via Weight Rotation...")
    
    # A single NAND block starts failing if read ~100,000 to 1,000,000 times without erasure
    read_disturb_threshold = 1_000_000 
    
    # 7B Model Weights at Q8 (7GB)
    model_size_gb = 7
    tokens_generated_lifetime = 10_000_000 # Generating 10 million tokens (very heavy inference box)
    
    # Static allocation: The same 7GB block is read for every single token
    reads_per_block_static = tokens_generated_lifetime
    
    # Multi-Bitwidth Rotation (Q2, Q4, Q8):
    # Instead of just dynamically picking precision for quality, we specifically rotate
    # the exact same precision copies across different physical blocks to spread the read heat.
    num_copies = 3 
    reads_per_block_rotated = tokens_generated_lifetime / num_copies
    
    results = [
        {"Strategy": "Static Weight Blocks", "Reads_Per_Block": reads_per_block_static, "Threshold_Breached": reads_per_block_static > read_disturb_threshold, "Silent_WAF_Penalty": 1.0},
        {"Strategy": "3-Copy Rotation", "Reads_Per_Block": reads_per_block_rotated, "Threshold_Breached": reads_per_block_rotated > read_disturb_threshold, "Silent_WAF_Penalty": 0.0}
    ]
    
    # If read disturb triggers, the FTL must silently read the block and write it somewhere else (Write Amplification)
    # 10M reads / 1M threshold = 10 internal rewrite cycles for 7GB data = 70GB wasted writes
    results[0]["Silent_WAF_Penalty"] = (reads_per_block_static // read_disturb_threshold) * model_size_gb
    
    df = pd.DataFrame(results)
    df.to_csv("read_disturb_rotation_metrics.csv", index=False)
    
    print("\nResults:")
    print(df.to_string())

if __name__ == "__main__":
    simulate_read_disturb()
