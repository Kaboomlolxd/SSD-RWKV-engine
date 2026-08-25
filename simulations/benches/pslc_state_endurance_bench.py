import pandas as pd

def simulate_pslc_state_endurance():
    print("Simulating pSLC (pseudo-SLC) State Partitioning for Mamba Inference...")
    
    # Workload parameters
    tokens_generated = 100_000_000 # 100M tokens over SSD lifetime
    mamba_state_write_mb_per_token = 128
    
    total_write_volume_tb = (tokens_generated * mamba_state_write_mb_per_token) / (1024 * 1024)
    
    # Drive specifications (Assume standard 2TB NVMe)
    qlc_pe_cycles = 1000      # QLC endurance
    tlc_pe_cycles = 3000      # TLC endurance
    pslc_pe_cycles = 30000    # SLC endurance (using 1 cell for 1 bit instead of 4)
    
    # Calculate Drive Wear (%) if State was stored in each mode
    # Assuming WAF = 2.0 for constant random state updates (if not delta-logged)
    waf = 2.0
    effective_write_tb = total_write_volume_tb * waf
    
    qlc_wear = (effective_write_tb / (2 * qlc_pe_cycles)) * 100
    tlc_wear = (effective_write_tb / (2 * tlc_pe_cycles)) * 100
    pslc_wear = (effective_write_tb / (2 * pslc_pe_cycles)) * 100
    
    results = [
        {"NAND_Mode": "QLC (Weights only)", "PE_Cycles": qlc_pe_cycles, "Drive_Wear_100M_Tokens_%": qlc_wear, "Viable_for_State": "No"},
        {"NAND_Mode": "TLC (Mainstream)", "PE_Cycles": tlc_pe_cycles, "Drive_Wear_100M_Tokens_%": tlc_wear, "Viable_for_State": "Marginal"},
        {"NAND_Mode": "pSLC (State Partition)", "PE_Cycles": pslc_pe_cycles, "Drive_Wear_100M_Tokens_%": pslc_wear, "Viable_for_State": "Yes"}
    ]
    
    df = pd.DataFrame(results)
    df.to_csv("pslc_state_endurance_metrics.csv", index=False)
    
    print("\nResults:")
    print(df.to_string())

if __name__ == "__main__":
    simulate_pslc_state_endurance()
