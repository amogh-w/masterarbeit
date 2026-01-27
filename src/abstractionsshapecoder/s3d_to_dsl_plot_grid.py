from pathlib import Path

from abstractionsshapecoder.shape_parser import ShapeParser
import plot_utils as vis

# --- CONFIGURATION ---
INPUT_FILE = Path.cwd() / "prog_data" / "PN_chair.txt"
OUTPUT_IMAGE = Path.cwd() / "rendered_shapes" / "chair_grid.png"
LIMIT = 5

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    parser = ShapeParser()
    dsl_objects = []
    names = []

    print(f"Reading first {LIMIT} shapes from {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r') as f:
        # Read lines until we have enough valid shapes
        count = 0
        for line in f:
            if count >= LIMIT: break
            
            line = line.strip()
            if not line: continue
            
            try:
                shape_id, prog_text = line.split(':', 1)
                
                # Parse text -> DSL Object
                obj = parser.parse(prog_text)
                
                dsl_objects.append(obj)
                names.append(f"ID: {shape_id}")
                count += 1
                
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    if dsl_objects:
        print(f"Generating grid plot for {len(dsl_objects)} shapes...")
        
        # 
        
        vis.plot_dsl_grid(
            dsl_objects, 
            names, 
            save_path=OUTPUT_IMAGE, 
            grid_cols=3, 
            grid_title=f"First {LIMIT} Shapes from PartNet"
        )
    else:
        print("No shapes found.")

if __name__ == "__main__":
    main()