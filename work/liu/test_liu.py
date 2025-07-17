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
import random

class FileAccessProvider(BankAccessProvider):
    """Access provider that reads ms-accurate access data from a file"""
    
    def __init__(self, num_banks, access_file):
        super().__init__(num_banks)
        self.access_file = access_file
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
                        parts = [part.strip() for part in line.split() if part.strip()]
                    
                    # Skip header line (first non-comment line that doesn't start with a number)
                    if not parts[0].isdigit():
                        continue
                    
                    # Check if we have enough data: step + 64 reads + 64 writes = 129 columns
                    expected_columns = 1 + 2 * self.num_banks
                    if len(parts) < expected_columns:
                        print(f"Warning: Line {line_num} has insufficient data: {line}")
                        print(f"  Expected {expected_columns} columns, got {len(parts)}")
                        continue
                    
                    step = int(parts[0])
                    read_accesses = [int(x) for x in parts[1:1+self.num_banks]]
                    write_accesses = [int(x) for x in parts[1+self.num_banks:1+2*self.num_banks]]
                    
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
                        'low_read_accesses': low_read_accesses,
                        'low_write_accesses': low_write_accesses,
                        'bank_modes': bank_modes
                    })
            
            print(f"Loaded {len(access_data)} access steps from {self.access_file}")
            return access_data
            
        except FileNotFoundError:
            print(f"Error: Access file '{self.access_file}' not found")
            return []
        except Exception as e:
            print(f"Error loading access file: {e}")
            return []
    
    def update_pattern(self):
        """Update access pattern for next time step"""
        if self.current_step < len(self.access_data):
            data = self.access_data[self.current_step]
            self.set_access_data(
                data['read_accesses'],
                data['write_accesses'],
                data['low_read_accesses'],
                data['low_write_accesses'],
                data['bank_modes']
            )
            self.current_step += 1
        else:
            # End of file - keep last pattern or set to zero
            print(f"Warning: Reached end of access file at step {self.current_step}")
            # Optionally set to zero accesses
            # self.set_access_data([0] * self.num_banks, [0] * self.num_banks)

def create_sample_config():
    """Create a sample configuration file for testing"""
    config_content = """[general]
total_cores = 4

[memory]
bank_size = 28
energy_per_read_access = 20.55
energy_per_write_access = 20.55
logic_core_power = 0.272
energy_per_refresh_access = 3.55
t_refi = 7.8
no_refesh_commands_in_t_refw = 8
banks_in_x = 4
banks_in_y = 4
banks_in_z = 4
num_banks = 64
cores_in_x = 2
cores_in_y = 2
cores_in_z = 1
type_of_stack = 3Dmem

[hotspot]
sampling_interval = 1000000
tool_path = ../../hotspot_tool
config_path = ./  
hotspot_config_file_mem = config/liu/mem_hotspot.config
floorplan_folder = config/liu
layer_file_mem = config/liu/mem.lcf

[hotspot/log_files_mem]
power_trace_file = power.trace
temperature_trace_file = temperature.trace
init_file = init.temp
init_file_external_mem = 
steady_temp_file = steady.temp
all_transient_file = all_transient.temp
grid_steady_file = grid_steady.temp

[hotspot/log_files]
combined_temperature_trace_file = combined_temperature.trace
combined_power_trace_file = combined_power.trace

[scheduler/open/dram/dtm]
dtm = off

[perf_model/dram/lowpower]
lpm_dynamic_power = 0.5
lpm_leakage_power = 0.1

[perf_model/core]
min_frequency = 1.0
max_frequency = 4.0
frequency_step_size = 0.1

[power]
technology_node = 22
vdd = 1.2
vth = 0.3

[core_thermal]
enabled = true

[reliability]
enabled = false
"""
    
    with open('example_config.cfg', 'w') as f:
        f.write(config_content)
    
    print("Created example configuration file: example_config.cfg")


def run_file_based_simulation(duration_ms=20, output_file='output_baseline1.csv', output_dir='output_baseline1'):
    """Run simulation with file-based access data
    
    Args:
        duration_ms (int): Simulation duration in milliseconds
    """
    print("\n" + "="*50)
    print(f"Running simulation with file-based access pattern for {duration_ms}ms")
    print("="*50)
    
    #access file
    access_file = output_file
    
    # Create access provider with file input
    access_provider = FileAccessProvider(64, access_file)

    create_sample_config()
    
    # Create thermal simulation
    thermal_sim = StandaloneMemTherm('example_config.cfg', access_provider)
    
    # Convert milliseconds to nanoseconds (1ms = 1,000,000ns)
    duration_ns = duration_ms * 1000000
    print(f"Running simulation for {duration_ms}ms ({duration_ns}ns)")
    
    # Run simulation for specified duration
    thermal_sim.run(duration_ns)
    
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
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("CoMeT Standalone Thermal Simulation Example")
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
    
    os.system("echo creating floorplan files for first run")
    os.system("mkdir -p ./config/liu")
    os.system("python3 ../../floorplanlib/create.py --mode 3Dmem --cores 2x2 --corex 3.414mm --corey 3.414mm --banks 4x4x4 --bankx 3.414mm --banky 3.414mm --out ./config/liu")
    
    converter = Dsc2AccessConverter(
        total_banks=64,
        bank_size=28,
        stack_height=4,
        clock_freq=500,
        bank_per_row=4,
        bank_per_col=4,
        group_rows=2,
        group_cols=2,
        num_groups=4,
    )
    converter.convert_file(args.input_file, args.output_file)
    
    # Run example simulation with built-in patterns
    run_file_based_simulation(args.duration, args.output_file, args.output_dir)

    # Convert CSV to aligned table
    csv_to_aligned_table(args.output_dir + "/" + "temperature.trace", args.output_dir + "/" + "temperature.trace" + '.table')
    csv_to_aligned_table(args.output_dir + "/" + "power.trace", args.output_dir + "/" + "power.trace" + '.table')
    
    print("\nExample completed successfully!")
    print("Check the output files for thermal simulation results.")

if __name__ == "__main__":
    main() 