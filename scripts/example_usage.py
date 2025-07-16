#!/usr/bin/env python3
"""
example_usage.py

Example script demonstrating how to use the standalone thermal simulation
with custom bank access patterns.
"""

import os
import sys
import shutil
import math
print(sys.executable)
import numpy as np
from standalone_thermal import StandaloneMemTherm, BankAccessProvider
import random

class CustomAccessProvider(BankAccessProvider):
    """Custom access provider that generates realistic access patterns"""
    
    def __init__(self, num_banks, access_pattern='random'):
        super().__init__(num_banks)
        self.access_pattern = access_pattern
        self.time_step = 0
        self.generate_access_pattern()
    
    def generate_access_pattern(self):
        """Generate access patterns based on the specified type"""
        if self.access_pattern == 'random':
            self.generate_random_pattern()
        elif self.access_pattern == 'hotspot':
            self.generate_hotspot_pattern()
        elif self.access_pattern == 'uniform':
            self.generate_uniform_pattern()
        else:
            raise ValueError("Unknown access pattern: {}".format(self.access_pattern))
    
    def generate_random_pattern(self):
        """Generate random access patterns"""
        # Random read accesses between 5-25 per bank
        self.read_accesses = np.random.randint(5, 25, self.num_banks).tolist()
        # Random write accesses between 2-12 per bank
        self.write_accesses = np.random.randint(2, 12, self.num_banks).tolist()
        # Some banks in low power mode (10% of banks)
        self.bank_modes = [0 if i % 10 == 0 else 1 for i in range(self.num_banks)]
        
        # Low power accesses are 20% of normal accesses
        self.read_accesses_lowpower = [int(r * 0.2) for r in self.read_accesses]
        self.write_accesses_lowpower = [int(w * 0.2) for w in self.write_accesses]
    
    def generate_hotspot_pattern(self):
        """Generate hotspot access pattern (some banks accessed more frequently)"""
        # Create hotspot in the center banks
        center_banks = [60, 61, 62, 63, 68, 69, 70, 71, 76, 77, 78, 79, 84, 85, 86, 87]
        
        self.read_accesses = [5] * self.num_banks  # Base access rate
        self.write_accesses = [2] * self.num_banks
        
        # Increase access rate for hotspot banks
        for bank in center_banks:
            if bank < self.num_banks:
                self.read_accesses[bank] = 40000
                self.write_accesses[bank] = 20000
        
        # All banks in normal power mode
        self.bank_modes = [1] * self.num_banks
        
        # Low power accesses
        self.read_accesses_lowpower = [int(r * 0.1) for r in self.read_accesses]
        self.write_accesses_lowpower = [int(w * 0.1) for w in self.write_accesses]
    
    def generate_uniform_pattern(self):
        """Generate uniform access pattern"""
        # Uniform access across all banks
        self.read_accesses = [15] * self.num_banks
        self.write_accesses = [8] * self.num_banks
        self.bank_modes = [1] * self.num_banks
        
        # Low power accesses
        self.read_accesses_lowpower = [int(r * 0.15) for r in self.read_accesses]
        self.write_accesses_lowpower = [int(w * 0.15) for w in self.write_accesses]
    
    def update_pattern(self):
        """Update access pattern for next time step"""
        self.time_step += 1
        
        if self.access_pattern == 'random':
            # Regenerate random pattern every few steps
            if self.time_step % 5 == 0:
                self.generate_random_pattern()
        elif self.access_pattern == 'hotspot':
            # Move hotspot slightly
            if self.time_step % 3 == 0:
                self.generate_hotspot_pattern()
        # Uniform pattern stays the same

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
                    
                    # Expected format: step,read_0,read_1,...,read_127,write_0,write_1,...,write_127
                    # Or: step,read_0,read_1,...,read_127,write_0,write_1,...,write_127,low_read_0,...,low_write_127
                    parts = line.split(',')
                    
                    if len(parts) < 2 + 2 * self.num_banks:  # step + reads + writes
                        print(f"Warning: Line {line_num} has insufficient data: {line}")
                        continue
                    
                    step = int(parts[0])
                    read_accesses = [int(x) for x in parts[1:1+self.num_banks]]
                    write_accesses = [int(x) for x in parts[1+self.num_banks:1+2*self.num_banks]]
                    
                    # Optional low power data
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
total_cores = 16

[memory]
bank_size = 64
energy_per_read_access = 20.55
energy_per_write_access = 20.55
logic_core_power = 0.272
energy_per_refresh_access = 3.55
t_refi = 7.8
no_refesh_commands_in_t_refw = 8
banks_in_x = 4
banks_in_y = 4
banks_in_z = 8
num_banks = 128
cores_in_x = 4
cores_in_y = 4
cores_in_z = 1
type_of_stack = 3Dmem

[hotspot]
sampling_interval = 1000000
tool_path = ../hotspot_tool
config_path = ..
hotspot_config_file_mem = config/hotspot/test_standalone1/mem_hotspot.config
floorplan_folder = config/hotspot/test_standalone1
layer_file_mem = config/hotspot/test_standalone1/mem.lcf

[hotspot/log_files_mem]
power_trace_file = power.trace
temperature_trace_file = temperature.trace
init_file = init.temp
init_file_external_mem = config/hotspot/3Dmem/mem.init
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

def create_sample_access_file():
    """Create a sample access file for testing"""
    
    filename = 'sample_access.csv'
    num_banks = 128
    num_steps = 10  # 10ms simulation
    
    with open(filename, 'w') as f:
        f.write("# Sample access file format: step,read_0,read_1,...,read_127,write_0,write_1,...,write_127\n")
        f.write("# Optional: add low_read_0,...,low_write_127,bank_mode_0,...,bank_mode_127\n")
        
        for step in range(num_steps):
            # Generate realistic access patterns
            read_accesses = []
            write_accesses = []
            low_read_accesses = []
            low_write_accesses = []
            bank_modes = []
            
            for bank in range(num_banks):
                # Base access rate
                base_read = random.randint(5, 25)
                base_write = random.randint(2, 12)
                
                # Add some hotspots
                if step < 3 and bank in [60, 61, 62, 63, 68, 69, 70, 71]:
                    base_read = random.randint(3000, 5000)
                    base_write = random.randint(1500, 2500)
                
                # Add some variation over time
                time_factor = 1.0 + 0.2 * math.sin(step * 0.5)
                read_count = int(base_read * time_factor)
                write_count = int(base_write * time_factor)
                
                read_accesses.append(read_count)
                write_accesses.append(write_count)
                
                # Low power accesses (20% of normal)
                low_read_accesses.append(int(read_count * 0.2))
                low_write_accesses.append(int(write_count * 0.2))
                
                # Bank modes (1 = normal, 0 = low power)
                bank_modes.append(1 if random.random() > 0.1 else 0)
            
            # Write line: step,reads,writes,low_reads,low_writes,bank_modes
            line_parts = [str(step)] + read_accesses + write_accesses + low_read_accesses + low_write_accesses + bank_modes
            f.write(','.join(map(str, line_parts)) + '\n')
    
    print(f"Created sample access file: {filename}")
    return filename

def run_example_simulation():
    """Run example thermal simulation with different access patterns"""
    
    # Create configuration file
    create_sample_config()
    
    # Test different access patterns
    patterns = ['random', 'hotspot', 'uniform']
    
    for pattern in patterns:
        print("\n" + "="*50)
        print("Running simulation with {} access pattern".format(pattern))
        print("="*50)
        
        # Create access provider with specific pattern
        access_provider = CustomAccessProvider(128, pattern)
        
        # Create thermal simulation
        thermal_sim = StandaloneMemTherm('example_config.cfg', access_provider)
        
        # Run simulation for 5ms
        thermal_sim.run(5000000)

        # Define source and destination directories
        dst_dir = f'output_{pattern}'

        # Remove destination if it already exists to avoid errors
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        # Move the output directory to the new destination
        if os.path.exists("output"):
            shutil.move("output", dst_dir)
        else:
            print(f"Warning: 'output' directory does not exist, nothing to move.")
        
        print("Completed simulation with {} pattern".format(pattern))
    
    print("\nAll example simulations completed!")

def run_file_based_simulation():
    """Run simulation with file-based access data"""
    print("\n" + "="*50)
    print("Running simulation with file-based access pattern")
    print("="*50)
    
    # Create sample access file
    access_file = create_sample_access_file()
    
    # Create access provider with file input
    access_provider = FileAccessProvider(128, access_file)
    
    # Create thermal simulation
    thermal_sim = StandaloneMemTherm('example_config.cfg', access_provider)
    
    # Run simulation for 10ms (10 steps)
    thermal_sim.run(10000000)
    
    print("Completed simulation with file-based pattern")
    
    # Move output to dedicated directory
    dst_dir = 'output_file_based'
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    if os.path.exists("output"):
        shutil.move("output", dst_dir)
    else:
        print("Warning: 'output' directory does not exist, nothing to move.")

def analyze_results():
    """Analyze the results of the thermal simulation"""
    print("\nAnalyzing simulation results...")
    
    # Check if output files exist in output directory
    output_dir = 'output'
    output_files = [
        'power.trace',
        'temperature.trace',
        'combined_temperature.trace',
        'combined_power.trace',
        'bank_mode.trace'
    ]
    
    for filename in output_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print("Found output file: {}".format(filepath))
            
            # Read and display first few lines
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    print("  First 3 lines:")
                    for i, line in enumerate(lines[:3]):
                        print("    Line {}: {}".format(i+1, line.strip()))
            except Exception as e:
                print("  Error reading file: {}".format(e))
        else:
            print("Missing output file: {}".format(filepath))
    
    print("\nAnalysis complete!")

def main():
    """Main function"""
    print("CoMeT Standalone Thermal Simulation Example")
    print("="*50)
    
    # Check if numpy is available
    try:
        import numpy as np
        print("NumPy available for advanced access patterns")
    except ImportError:
        print("NumPy not available, using basic access patterns")
        # Fallback to basic random without numpy
        np = None
    
    os.system("echo creating floorplan files for first run")
    os.system("mkdir -p ../config/hotspot/test_standalone1")
    os.system("python3 ../floorplanlib/create.py --mode 3Dmem --cores 4x4 --corex 3.414mm --corey 3.414mm --banks 4x4x8 --bankx 3.414mm --banky 3.414mm --out ../config/hotspot/test_standalone1")
    
    # Run example simulation with built-in patterns
    run_example_simulation()
    
    # Run file-based simulation
    run_file_based_simulation()
    
    # Analyze results
    analyze_results()
    
    print("\nExample completed successfully!")
    print("Check the output files for thermal simulation results.")

if __name__ == "__main__":
    main() 