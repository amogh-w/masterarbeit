from pathlib import Path

import executor as ex
import os
from tqdm import tqdm


# Configuration
INPUT_FILE = Path.cwd() / "prog_data" / "PN_chair.txt"
OUTPUT_DIR = Path.cwd() / "rendered_shapes"
LIMIT = 10  # Number of lines to process

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize the program executor
    program_executor = ex.Program()

    try:
        print(f"Reading from {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        
        # Logic to handle LIMIT vs Total Lines
        if LIMIT < total_lines:
            print(f"Limit is set to {LIMIT}. Processing the first {LIMIT} shapes out of {total_lines} available.")
            lines_to_process = all_lines[:LIMIT]
        else:
            print(f"Limit ({LIMIT}) is higher than total lines ({total_lines}). Processing all {total_lines} shapes.")
            lines_to_process = all_lines

        # Iterate with TQDM progress bar
        for line in tqdm(lines_to_process, desc="Rendering", unit="shape"):
            line = line.strip()
            if not line:
                continue

            # 1. Parse the line (ID : Program)
            try:
                parts = line.split(':', 1)
                shape_id = parts[0].strip()
                prog_text = parts[1].strip()
            except IndexError:
                tqdm.write(f"Skipping malformed line: {line}")
                continue

            # 2. Run the program
            output_path = OUTPUT_DIR / shape_id
            
            try:
                program_executor.run(prog_text, name=output_path)
                # We typically don't print "Success" for every item when using a progress bar
                # to keep the console clean, but if you want to see it, use:
                # tqdm.write(f"Saved {shape_id}")
            except Exception as e:
                tqdm.write(f"Error executing shape {shape_id}: {e}")

    except FileNotFoundError:
        print(f"Error: Could not find file '{INPUT_FILE}'. Please create it with your shape data.")

if __name__ == "__main__":
    main()