import re

def extract_size_tolerance(filename):
    """
    Extract size and tolerance from filename following the pattern like:
    '2000_1e-3_1.txt'
    
    Returns:
        size (str): extracted size
        tolerance (str): extracted tolerance
    """
    # Remove file extension
    base_name = filename.split('.')[0]
    # Split by underscore
    parts = base_name.split('_')
    if len(parts) < 2:
        raise ValueError("Filename pattern not recognized")
    
    size = parts[0]
    tolerance = parts[1]
    
    return size, tolerance

# Example usage
if __name__ == "__main__":
    filename = "2000_1e-3_1.txt"
    size, tolerance = extract_size_tolerance(filename)
    size, tolerance = float(size), float(tolerance)
    print(f"Size: {size}",type(size))
    print(f"Tolerance: {tolerance}",type(tolerance),)

