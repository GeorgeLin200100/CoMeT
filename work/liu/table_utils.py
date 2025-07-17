#!/usr/bin/env python3
"""
table_utils.py

Utility functions for table formatting and CSV conversion.
"""

import csv
import os


def csv_to_aligned_table(csv_file, output_file=None, delimiter=None, min_width=8, max_width=None, skip_header=False):
    """
    Convert a CSV file to a plain text table with aligned columns.
    
    Args:
        csv_file (str): Path to input CSV file
        output_file (str): Path to output text file (if None, returns string)
        delimiter (str): CSV delimiter character (None for auto-detect)
        min_width (int): Minimum column width
        max_width (int): Maximum column width (None for no limit)
        skip_header (bool): Whether to skip the first line as header
    
    Returns:
        str: Formatted table string (if output_file is None)
    """
    
    # Read CSV data
    rows = []
    detected_delimiter = delimiter
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if not lines:
            print("Error: No data found in CSV file")
            return None
        
        # Auto-detect delimiter from first line if not specified
        if delimiter is None:
            first_line = lines[0].strip()
            if ',' in first_line:
                detected_delimiter = ','
            elif '\t' in first_line:
                detected_delimiter = '\t'
            else:
                # Assume space-separated
                detected_delimiter = ' '
        
        # Process each line
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Skip header if requested
            if skip_header and line_num == 0:
                continue
            
            # Split by detected delimiter and strip whitespace
            if detected_delimiter == ',':
                # Handle CSV properly (respect quotes, etc.)
                reader = csv.reader([line])
                parts = next(reader)
            else:
                parts = [part.strip() for part in line.split(detected_delimiter)]
            
            rows.append(parts)
    
    if not rows:
        print("Error: No data found in CSV file")
        return None
    
    # Calculate column widths
    num_cols = max(len(row) for row in rows)
    col_widths = [min_width] * num_cols
    
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                cell_width = len(str(cell))
                if max_width:
                    cell_width = min(cell_width, max_width)
                col_widths[i] = max(col_widths[i], cell_width)
    
    # Generate table
    table_lines = []
    
    for row in rows:
        # Pad row to match number of columns
        padded_row = row + [''] * (num_cols - len(row))
        
        # Format each cell
        formatted_cells = []
        for i, cell in enumerate(padded_row):
            cell_str = str(cell)
            if max_width and len(cell_str) > max_width:
                cell_str = cell_str[:max_width-3] + '...'
            
            # Right-align numbers, left-align text
            if cell_str.replace('.', '').replace('-', '').isdigit():
                formatted_cell = cell_str.rjust(col_widths[i])
            else:
                formatted_cell = cell_str.ljust(col_widths[i])
            
            formatted_cells.append(formatted_cell)
        
        # Join cells with spaces
        table_line = '  '.join(formatted_cells)
        table_lines.append(table_line)
    
    # Create output content
    output_content = '\n'.join(table_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_content)
        print(f"Table written to {output_file}")
        return None
    else:
        return output_content


def format_data_as_table(data, headers=None, output_file=None, min_width=8, max_width=None):
    """
    Format data as an aligned table.
    
    Args:
        data (list): List of rows (each row is a list of values)
        headers (list): Optional list of column headers
        output_file (str): Path to output text file (if None, returns string)
        min_width (int): Minimum column width
        max_width (int): Maximum column width (None for no limit)
    
    Returns:
        str: Formatted table string (if output_file is None)
    """
    
    if not data:
        print("Error: No data provided")
        return None
    
    # Add headers if provided
    rows = []
    if headers:
        rows.append(headers)
    rows.extend(data)
    
    # Calculate column widths
    num_cols = max(len(row) for row in rows)
    col_widths = [min_width] * num_cols
    
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                cell_width = len(str(cell))
                if max_width:
                    cell_width = min(cell_width, max_width)
                col_widths[i] = max(col_widths[i], cell_width)
    
    # Generate table
    table_lines = []
    
    for row_idx, row in enumerate(rows):
        # Pad row to match number of columns
        padded_row = row + [''] * (num_cols - len(row))
        
        # Format each cell
        formatted_cells = []
        for i, cell in enumerate(padded_row):
            cell_str = str(cell)
            if max_width and len(cell_str) > max_width:
                cell_str = cell_str[:max_width-3] + '...'
            
            # Right-align numbers, left-align text (except headers)
            if headers and row_idx == 0:
                # Headers are left-aligned
                formatted_cell = cell_str.ljust(col_widths[i])
            elif cell_str.replace('.', '').replace('-', '').isdigit():
                formatted_cell = cell_str.rjust(col_widths[i])
            else:
                formatted_cell = cell_str.ljust(col_widths[i])
            
            formatted_cells.append(formatted_cell)
        
        # Join cells with spaces
        table_line = '  '.join(formatted_cells)
        table_lines.append(table_line)
    
    # Create output content
    output_content = '\n'.join(table_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_content)
        print(f"Table written to {output_file}")
        return None
    else:
        return output_content


# Example usage functions
def convert_csv_example():
    """Example: Convert output_baseline1.csv to aligned table"""
    csv_file = "output_baseline1.csv"
    output_file = "output_baseline1_table.txt"
    
    if os.path.exists(csv_file):
        result = csv_to_aligned_table(csv_file, output_file, min_width=10)
        print(f"Converted {csv_file} to {output_file}")
    else:
        print(f"File {csv_file} not found")


if __name__ == "__main__":
    # Example usage
    convert_csv_example() 