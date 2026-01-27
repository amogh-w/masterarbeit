import re
from scipy.spatial.transform import Rotation as R
import abstractionsshapecoder.dsl_nodes as dsl

class ShapeParser:
    def parse(self, expr_str):
        """Main entry point to parse a program string."""
        expr_str = expr_str.strip()
        if not expr_str: return None
        return self._parse_recursive(expr_str)

    def _parse_recursive(self, expr):
        # 1. Match "FunctionName(Arguments)"
        match = re.match(r'([a-zA-Z0-9_]+)\((.*)\)', expr, re.DOTALL)
        if not match:
            # It's a raw value
            return self._parse_value(expr)

        func_name = match.group(1)
        args_str = match.group(2)
        
        # 2. Split arguments while respecting nested parentheses
        args = []
        cur = ''
        lp = 0
        for c in args_str:
            if c == '(': lp += 1
            if c == ')': lp -= 1
            if c == ',' and lp == 0:
                args.append(cur.strip())
                cur = ''
            else:
                cur += c
        if cur: args.append(cur.strip())

        # 3. Map Function Names to New DSL Classes
        
        if func_name == 'Cuboid':
            # Source: Cuboid(w, h, d)
            # Target: dsl.Cuboid(size=[w, h, d])
            w, h, d = map(float, args)
            return dsl.Cuboid(size=[w, h, d])

        elif func_name == 'Move':
            # Source: Move(Shape, x, y, z)
            # Target: dsl.Translate(child, vector=[x, y, z])
            child = self._parse_recursive(args[0])
            x, y, z = map(float, args[1:])
            return dsl.Translate(child, vector=[x, y, z])
            
        elif func_name == 'Union':
            # Source: Union(Shape, Shape)
            # Target: dsl.Union(left, right)
            child1 = self._parse_recursive(args[0])
            child2 = self._parse_recursive(args[1])
            return dsl.Union(child1, child2)

        elif func_name == 'Rotate':
            # Source: Rotate(Shape, AxisString, Radians)
            # Target: dsl.Rotate(child, quaternion=[x, y, z, w])
            child = self._parse_recursive(args[0])
            
            axis_str = args[1].lower().replace("'", "")
            radians = float(args[2])
            
            # Convert Axis-Angle -> Quaternion
            # The source format usually implies rotation around a principal axis
            if 'x' in axis_str:
                rot = R.from_euler('x', radians, degrees=False)
            elif 'y' in axis_str:
                rot = R.from_euler('y', radians, degrees=False)
            elif 'z' in axis_str:
                rot = R.from_euler('z', radians, degrees=False)
            else:
                # Default fallback (should not happen in PartNet)
                rot = R.from_euler('y', radians, degrees=False)
                
            quat = rot.as_quat() # returns [x, y, z, w]
            return dsl.Rotate(child, quaternion=quat)

        else:
            raise ValueError(f"Unknown function: {func_name}")

    def _parse_value(self, val):
        val = val.strip()
        try:
            return float(val)
        except ValueError:
            return val.replace("'", "")