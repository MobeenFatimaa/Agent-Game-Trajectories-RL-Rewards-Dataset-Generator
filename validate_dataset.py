import json
import numpy as np
import pandas as pd

def validate_csv_dataset(csv_filepath, expected_rows=None):
    print("=" * 60)
    print(f"  DATASET AUDIT SUITE: {csv_filepath}")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"[FATAL ERROR] Could not read file: {e}")
        return False
        
    tests_passed = 0
    total_tests = 7 if expected_rows else 6
    
    # Test 1: Exact Row Count Check
    if expected_rows:
        if len(df) == expected_rows:
            print(f"[PASS] 1. Exact Row Count: Contains exactly {len(df):,} rows.")
            tests_passed += 1
        else:
            print(f"[FAIL] 1. Exact Row Count: Expected {expected_rows:,}, got {len(df):,}.")

    # Test 2: Missing / Null Values Check
    null_count = df.isnull().sum().sum()
    if null_count == 0:
        print("[PASS] 2. Missing Values: 0 nulls across all columns.")
        tests_passed += 1
    else:
        print(f"[FAIL] 2. Missing Values: Found {null_count} null entries.")

    # Test 3: Schema Column Layout
    expected_cols = [
        "Episode_ID", "Step", "Agent_Type", "State_JSON", "Action", 
        "Action_Name", "Reward", "Cumulative_Reward", "Next_State_JSON", 
        "Policy_Entropy", "Done"
    ]
    if list(df.columns) == expected_cols:
        print("[PASS] 3. Schema Integrity: Column names and order match specification.")
        tests_passed += 1
    else:
        print("[FAIL] 3. Schema Integrity: Mismatch in column structure.")

    # Test 4: Action ID to Name Mapping
    action_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    mismatches = df[df.apply(lambda r: action_map[r["Action"]] != r["Action_Name"], axis=1)]
    if len(mismatches) == 0:
        print("[PASS] 4. Categorical Encoding: Action IDs and names are 100% consistent.")
        tests_passed += 1
    else:
        print(f"[FAIL] 4. Categorical Encoding: Detected {len(mismatches)} invalid action maps.")

    # Test 5: Terminal State Position Verification
    terminals = df[df["Done"] == True]
    terminal_errors = 0
    for _, row in terminals.iterrows():
        next_s = json.loads(row["Next_State_JSON"])
        if next_s["agent_pos"] != next_s["target_pos"]:
            terminal_errors += 1
            
    if terminal_errors == 0:
        print(f"[PASS] 5. Terminal Coordinates: All {len(terminals):,} finished episodes reached target [9, 9].")
        tests_passed += 1
    else:
        print(f"[FAIL] 5. Terminal Coordinates: {terminal_errors} invalid terminal states.")

    # Test 6: Cumulative Reward Sequence Math
    cum_reward_valid = True
    for _, group in df.groupby("Episode_ID"):
        calc_cum = group["Reward"].cumsum().round(2)
        if not np.allclose(group["Cumulative_Reward"].values, calc_cum.values):
            cum_reward_valid = False
            break
            
    if cum_reward_valid:
        print("[PASS] 6. Mathematical Precision: Cumulative rewards correctly accumulate.")
        tests_passed += 1
    else:
        print("[FAIL] 6. Mathematical Precision: Error in reward progression logic.")

    # Test 7: JSON String Parseability
    try:
        df["State_JSON"].apply(json.loads)
        df["Next_State_JSON"].apply(json.loads)
        print("[PASS] 7. JSON Format: State strings are valid, parseable JSON objects.")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] 7. JSON Format: Corrupt JSON detected -> {e}")

    print("-" * 60)
    print(f"AUDIT COMPLETE: {tests_passed}/{total_tests} Tests Passed.")
    print("=" * 60)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    CSV_FILE = "synthetic_rl_trajectories.csv"
    EXPECTED_ROWS = 100000  # Set to None if you don't want to enforce an exact target check
    
    validate_csv_dataset(CSV_FILE, expected_rows=EXPECTED_ROWS)