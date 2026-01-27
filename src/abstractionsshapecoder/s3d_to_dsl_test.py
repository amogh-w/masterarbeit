from pathlib import Path
from tqdm import tqdm

from abstractionsshapecoder.shape_parser import ShapeParser

# Configuration
INPUT_FILE = Path.cwd() / "prog_data" / "PN_chair.txt"
LIMIT = 5  # Set to None to process all files


def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    parser = ShapeParser()
    converted_shapes = {}

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    process_count = LIMIT if LIMIT and LIMIT < total else total

    print(f"Translating first {process_count} shapes to DSL...")
    
    for i, line in tqdm(enumerate(lines[:process_count]), total=process_count):
        line = line.strip()
        if not line: continue
        
        try:
            # Split ID and Program
            shape_id, prog_text = line.split(':', 1)
            
            # PARSE into DSL Object
            dsl_object = parser.parse(prog_text)
            
            converted_shapes[shape_id] = dsl_object
            
        except Exception as e:
            tqdm.write(f"Failed to parse {shape_id}: {e}")

    # --- DEMONSTRATION: Print the first converted object ---
    if converted_shapes:
        # Just grab the first key available
        first_id = list(converted_shapes.keys())[3]
        print(f"\n[SUCCESS] Dump of Shape ID {first_id} in New DSL:\n")
        print(converted_shapes[first_id])
        
        print(f"\nSuccessfully converted {len(converted_shapes)} programs into DSL objects.")

if __name__ == "__main__":
    main()