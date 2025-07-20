#!/usr/bin/env python3
"""
test_liu.py

Example script demonstrating how to use the standalone thermal simulation
with custom bank access patterns.
"""

import os
import sys
import shutil
import math
import argparse
print(sys.executable)
import numpy as np
from standalone_thermal import StandaloneMemTherm, BankAccessProvider
from dsc2access import Dsc2AccessConverter
from table_utils import csv_to_aligned_table
from config_manager import ConfigManager
import random

class FileAccessProvider(BankAccessProvider):
    """Access provider that reads ms-accurate access data from a file"""
    
    def __init__(self, num_banks, access_file, repeat_times=1):
        super().__init__(num_banks)
        self.access_file = access_file
        self.repeat_times = repeat_times
        self.current_step = 0
        self.access_data = self.load_access_data()
        
    def load_access_data(self):
        """Load access data from file"""
        access_data = []
        
        try:
            with open(self.access_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle both comma-separated and space-aligned formats
                    if ',' in line:
                        # Comma-separated format
                        parts = line.split(',')
                    else:
                        # Space-aligned format - split on whitespace and filter out empty strings
                        # Use regex to split on multiple whitespace characters
                        import re
                        parts = re.split(r'\s+', line.strip())
                        parts = [part.strip() for part in parts if part.strip()]
                    
                    # Skip header line (first non-comment line that doesn't start with a number)
                    if not parts[0].isdigit():
                        continue
                    
                    # Check if we have enough data: step + 64 reads + 64 writes + 64 read_lw + 64 write_lw = 257 columns
                    expected_columns = 1 + 4 * self.num_banks
                    if len(parts) < expected_columns:
                        print(f"Warning: Line {line_num} has insufficient data: {line}")
                        print(f"  Expected {expected_columns} columns, got {len(parts)}")
                        continue
                    
                    step = int(parts[0])
                    read_accesses = [int(x) for x in parts[1:1+self.num_banks]]
                    write_accesses = [int(x) for x in parts[1+self.num_banks:1+2*self.num_banks]]
                    read_lanewidths = [int(x) for x in parts[1+2*self.num_banks:1+3*self.num_banks]]
                    write_lanewidths = [int(x) for x in parts[1+3*self.num_banks:1+4*self.num_banks]]
                    
                    # Optional low power data (not present in current format)
                    low_read_accesses = None
                    low_write_accesses = None
                    bank_modes = None
                    
                    if len(parts) >= 1 + 4 * self.num_banks:  # Has low power data
                        low_read_accesses = [int(x) for x in parts[1+2*self.num_banks:1+3*self.num_banks]]
                        low_write_accesses = [int(x) for x in parts[1+3*self.num_banks:1+4*self.num_banks]]
                    
                    if len(parts) >= 1 + 5 * self.num_banks:  # Has bank modes
                        bank_modes = [int(x) for x in parts[1+4*self.num_banks:1+5*self.num_banks]]
                    
                    access_data.append({
                        'step': step,
                        'read_accesses': read_accesses,
                        'write_accesses': write_accesses,
                        'read_lanewidths': read_lanewidths,
                        'write_lanewidths': write_lanewidths,
                        'low_read_accesses': low_read_accesses,
                        'low_write_accesses': low_write_accesses,
                        'bank_modes': bank_modes
                    })
            
            print(f"Loaded {len(access_data)} access steps from {self.access_file}")
            if self.repeat_times > 1:
                print(f"Will repeat this pattern {self.repeat_times} times (total {len(access_data) * self.repeat_times} steps)")
            return access_data
            
        except FileNotFoundError:
            print(f"Error: Access file '{self.access_file}' not found")
            return []
        except Exception as e:
            print(f"Error loading access file: {e}")
            return []
    
    def update_pattern(self):
        """Update access pattern for next time step"""
        if self.access_data:
            # Calculate the actual step within the original pattern
            original_pattern_length = len(self.access_data)
            if original_pattern_length == 0:
                return
                
            actual_step = self.current_step % original_pattern_length
            data = self.access_data[actual_step]
            self.set_access_data(
                data['read_accesses'],
                data['write_accesses'],
                data['low_read_accesses'],
                data['low_write_accesses'],
                data['bank_modes'],
                data['read_lanewidths'],
                data['write_lanewidths']
            )
            self.current_step += 1
            
            # Check if we've completed all repeats
            total_steps = original_pattern_length * self.repeat_times
            if self.current_step >= total_steps:
                print(f"Completed {self.repeat_times} cycles of access pattern ({total_steps} total steps)")
        else:
            # No access data available
            print(f"Warning: No access data available at step {self.current_step}")

def run_file_based_simulation(duration_ms=20, output_file='output_baseline1.csv', output_dir='output_baseline1', config_manager=None, repeat_times=1, no_feedback=False):
    """Run simulation with file-based access data
    
    Args:
        duration_ms (int): Simulation duration in milliseconds
        config_manager (ConfigManager): Configuration manager instance
        repeat_times (int): Number of times to repeat the access pattern
    """
    print("\n" + "="*50)
    print(f"Running simulation with file-based access pattern for {duration_ms}ms")
    print(f"Repeating access pattern {repeat_times} times")
    print("="*50)
    
    # Print configuration summary
    if config_manager:
        config_manager.print_summary()
    
    #access file
    access_file = output_file
    
    # Create access provider with file input using config values
    num_banks = config_manager.num_banks
    access_provider = FileAccessProvider(num_banks, access_file, repeat_times)
    
    # Create thermal simulation
    thermal_sim = StandaloneMemTherm(config_manager.config_file, access_provider)
    
    # Calculate simulation duration based on access pattern length and repeat times
    # Each step in the access pattern represents 1ms
    original_pattern_length = len(access_provider.access_data)
    total_pattern_length = original_pattern_length * repeat_times
    
    if duration_ms > 0:
        # Use the specified duration, but ensure it's not longer than the pattern
        simulation_duration_ms = min(duration_ms, total_pattern_length)
        print(f"Using simulation duration: {simulation_duration_ms}ms (pattern has {total_pattern_length}ms total)")
    else:
        # If duration is 0 or negative, use the full pattern length
        simulation_duration_ms = total_pattern_length
        print(f"Using full pattern duration: {simulation_duration_ms}ms")
    
    # Convert milliseconds to nanoseconds (1ms = 1,000,000ns)
    duration_ns = simulation_duration_ms * 1000000
    print(f"Running simulation for {simulation_duration_ms}ms ({duration_ns}ns)")
    
    # Run simulation for specified duration
    thermal_sim.run(duration_ns, no_feedback)
    
    print(f"Completed simulation with file-based pattern ({duration_ms}ms)")
    
    # Move output to dedicated directory
    dst_dir = output_dir
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    if os.path.exists("output"):
        shutil.move("output", dst_dir)
    else:
        print("Warning: 'output' directory does not exist, nothing to move.")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CoMeT Standalone Thermal Simulation Example')
    parser.add_argument('--duration', '-d', type=int, default=20, 
                       help='Simulation duration in milliseconds (default: 20ms)')
    parser.add_argument('--input_file', '-i', type=str, default='input_baseline1.csv', 
                       help='Input file (default: input_baseline1.csv)')
    parser.add_argument('--output_file', '-o', type=str, default='output_baseline1.csv', 
                       help='Output file (default: output_baseline1.csv)')
    parser.add_argument('--output_dir', '-od', type=str, default='output_baseline1', 
                       help='Output directory (default: output_baseline1)')
    parser.add_argument('--config_file', '-cf', type=str, default='example_config.cfg', 
                       help='Configuration file (default: example_config.cfg)')
    parser.add_argument('--repeat_times', '-r', type=int, default=1,
                       help='Number of times to repeat the access pattern (default: 1)')
    parser.add_argument('--no_feedback', '-nf', default=False,
                       help='Disable feedback loop, hotspot will only run once (default: False)')
    
    # Preprocessing options
    parser.add_argument('--preprocess', action='store_true', 
                       help='Preprocess description file to split large operations')
    parser.add_argument('--preprocessed_output', type=str, 
                       help='Output file for preprocessed description (default: input_file.preprocessed)')
    parser.add_argument('--split_on_total_capacity', action='store_true', default=True,
                       help='Split operations that exceed total bank group capacity')
    parser.add_argument('--split_on_stack_capacity', action='store_true', default=True,
                       help='Split operations that exceed single stack capacity')
    parser.add_argument('--no_split_on_total_capacity', action='store_true',
                       help='Disable splitting on total capacity')
    parser.add_argument('--no_split_on_stack_capacity', action='store_true',
                       help='Disable splitting on stack capacity')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Capture the full command line
    import sys
    command_line = ' '.join(sys.argv)
    
    # print("CoMeT Standalone Thermal Simulation Example")
    print("="*50)
    print(f"Simulation duration: {args.duration}ms")
    
    # Check if numpy is available
    try:
        import numpy as np
        print("NumPy available for advanced access patterns")
    except ImportError:
        print("NumPy not available, using basic access patterns")
        # Fallback to basic random without numpy
        np = None
    
    # Load configuration
    config_manager = ConfigManager(args.config_file)
    
    os.system("echo creating floorplan files for first run")
    os.system("mkdir -p ./config")
    # Build floorplan creation command from config
    floorplan_cmd = (
        f"python3 ../../floorplanlib/create.py "
        f"--mode {config_manager.type_of_stack} "
        f"--cores {config_manager.cores_in_x}x{config_manager.cores_in_y} "
        f"--corex {6.828/config_manager.cores_in_x}mm --corey {6.828/config_manager.cores_in_y}mm "
        f"--banks {config_manager.banks_in_x}x{config_manager.banks_in_y}x{config_manager.banks_in_z} "
        f"--bankx {6.828/config_manager.banks_in_x}mm --banky {6.828/config_manager.banks_in_y}mm "
        f"--out ./config"
    )
    os.system(floorplan_cmd)
    
    converter = Dsc2AccessConverter(
        total_banks=config_manager.num_banks,
        bank_size=config_manager.bank_size,
        stack_height=config_manager.banks_in_z,
        clock_freq=config_manager.clock_freq,
        bank_per_row=config_manager.banks_in_x,
        bank_per_col=config_manager.banks_in_y,
        group_rows=config_manager.group_rows, # each group has "group_rows" rows
        group_cols=config_manager.group_cols, # each group has "group_cols" columns
        num_groups=config_manager.num_groups,
        debug=False
    )
    
    # Handle preprocessing if requested
    input_file = args.input_file
    if args.preprocess:
        # Determine preprocessing output file
        if args.preprocessed_output:
            preprocessed_file = args.preprocessed_output
        else:
            preprocessed_file = args.input_file + ".preprocessed"
        
        # Determine splitting options
        split_on_total = args.split_on_total_capacity and not args.no_split_on_total_capacity
        split_on_stack = args.split_on_stack_capacity and not args.no_split_on_stack_capacity
        
        print(f"Preprocessing description file...")
        print(f"  Input file: {args.input_file}")
        print(f"  Preprocessed file: {preprocessed_file}")
        print(f"  Split on total capacity: {split_on_total}")
        print(f"  Split on stack capacity: {split_on_stack}")
        
        # Preprocess the file
        try:
            converter.preprocess_description_file(
                args.input_file, 
                preprocessed_file,
                split_on_total_capacity=split_on_total,
                split_on_stack_capacity=split_on_stack
            )
            print(f"Successfully preprocessed {args.input_file} to {preprocessed_file}")
            input_file = preprocessed_file
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return
    
    # Convert the description file to access pattern
    converter.convert_file(input_file, args.output_file)
    
    # Run example simulation with built-in patterns
    run_file_based_simulation(args.duration, args.output_file, args.output_dir, config_manager, args.repeat_times, args.no_feedback)

    # Write command line to output directory
    command_line_file = os.path.join(args.output_dir, "commandline.txt")
    try:
        with open(command_line_file, 'w') as f:
            f.write(f"Command executed: {command_line}\n")
            f.write(f"Execution time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write(f"Python executable: {sys.executable}\n")
        print(f"Command line saved to: {command_line_file}")
    except Exception as e:
        print(f"Warning: Could not write command line to file: {e}")

    # Convert CSV to aligned table
    csv_to_aligned_table(args.output_dir + "/" + "temperature.trace", args.output_dir + "/" + "temperature.trace" + '.table')
    csv_to_aligned_table(args.output_dir + "/" + "power.trace", args.output_dir + "/" + "power.trace" + '.table')
    
    # print("\nExample completed successfully!")
    # print("Check the output files for thermal simulation results.")

if __name__ == "__main__":
    main() 