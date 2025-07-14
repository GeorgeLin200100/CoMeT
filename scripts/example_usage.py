#!/usr/bin/env python2
"""
example_usage.py

Example script demonstrating how to use the standalone thermal simulation
with custom bank access patterns.
"""

import os
import sys
import numpy as np
from standalone_thermal import StandaloneMemTherm, BankAccessProvider

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
                self.read_accesses[bank] = 40
                self.write_accesses[bank] = 20
        
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

def create_sample_config():
    """Create a sample configuration file for testing"""
    config_content = """[general]
total_cores = 16

[memory]
bank_size = 67108864
energy_per_read_access = 1.0
energy_per_write_access = 1.0
logic_core_power = 0.1
energy_per_refresh_access = 100.0
t_refi = 7.8
no_refesh_commands_in_t_refw = 8
banks_in_x = 4
banks_in_y = 4
banks_in_z = 8
num_banks = 128
cores_in_x = 4
cores_in_y = 4
cores_in_z = 1
type_of_stack = 3D

[hotspot]
sampling_interval = 1000000
tool_path = hotspot_tool
config_path = .
hotspot_config_file_mem = hotspot.config
floorplan_folder = hotspot/floorplans
layer_file_mem = hotspot/layers.config

[hotspot/log_files_mem]
power_trace_file = power.trace
temperature_trace_file = temperature.trace
init_file = init.temp
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
        
        print("Completed simulation with {} pattern".format(pattern))
    
    print("\nAll example simulations completed!")

def analyze_results():
    """Analyze the results of the thermal simulation"""
    print("\nAnalyzing simulation results...")
    
    # Check if output files exist
    output_files = [
        'power.trace',
        'temperature.trace',
        'combined_temperature.trace',
        'combined_power.trace',
        'bank_mode.trace'
    ]
    
    for filename in output_files:
        if os.path.exists(filename):
            print("✓ Found output file: {}".format(filename))
            
            # Read and display first few lines
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    print("  First 3 lines:")
                    for i, line in enumerate(lines[:3]):
                        print("    Line {}: {}".format(i+1, line.strip()))
            except Exception as e:
                print("  Error reading file: {}".format(e))
        else:
            print("✗ Missing output file: {}".format(filename))
    
    print("\nAnalysis complete!")

def main():
    """Main function"""
    print("CoMeT Standalone Thermal Simulation Example")
    print("="*50)
    
    # Check if numpy is available
    try:
        import numpy as np
        print("✓ NumPy available for advanced access patterns")
    except ImportError:
        print("⚠ NumPy not available, using basic access patterns")
        # Fallback to basic random without numpy
        import random
        np = None
    
    # Run example simulation
    run_example_simulation()
    
    # Analyze results
    analyze_results()
    
    print("\nExample completed successfully!")
    print("Check the output files for thermal simulation results.")

if __name__ == "__main__":
    main() 