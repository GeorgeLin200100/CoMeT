#!/usr/bin/env python3
"""
create_sample_access.py

Create sample access files for the thermal simulation.
"""

import random
import math

def create_simple_access_file():
    """Create a simple access file with basic format"""
    filename = 'simple_access.csv'
    num_banks = 128
    num_steps = 5  # 5ms simulation
    
    with open(filename, 'w') as f:
        f.write("# Simple access file: step,read_0,read_1,...,read_127,write_0,write_1,...,write_127\n")
        
        for step in range(num_steps):
            read_accesses = [random.randint(5, 25) for _ in range(num_banks)]
            write_accesses = [random.randint(2, 12) for _ in range(num_banks)]
            
            line_parts = [str(step)] + read_accesses + write_accesses
            f.write(','.join(map(str, line_parts)) + '\n')
    
    print(f"Created simple access file: {filename}")

def create_advanced_access_file():
    """Create an advanced access file with low power and bank modes"""
    filename = 'advanced_access.csv'
    num_banks = 128
    num_steps = 10  # 10ms simulation
    
    with open(filename, 'w') as f:
        f.write("# Advanced access file: step,reads,writes,low_reads,low_writes,bank_modes\n")
        
        for step in range(num_steps):
            read_accesses = []
            write_accesses = []
            low_read_accesses = []
            low_write_accesses = []
            bank_modes = []
            
            for bank in range(num_banks):
                # Base access rate
                base_read = random.randint(5, 25)
                base_write = random.randint(2, 12)
                
                # Add hotspots in center banks
                if step < 3 and bank in [60, 61, 62, 63, 68, 69, 70, 71]:
                    base_read = random.randint(30, 50)
                    base_write = random.randint(15, 25)
                
                # Add time variation
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
    
    print(f"Created advanced access file: {filename}")

def create_burst_access_file():
    """Create an access file with burst patterns"""
    filename = 'burst_access.csv'
    num_banks = 128
    num_steps = 15  # 15ms simulation
    
    with open(filename, 'w') as f:
        f.write("# Burst access file with periodic high activity\n")
        
        for step in range(num_steps):
            read_accesses = []
            write_accesses = []
            
            # Create burst pattern every 5 steps
            burst_active = (step % 5 == 0)  # Burst every 5ms
            
            for bank in range(num_banks):
                if burst_active:
                    # High activity during bursts
                    read_count = random.randint(40, 80)
                    write_count = random.randint(20, 40)
                else:
                    # Low activity between bursts
                    read_count = random.randint(2, 8)
                    write_count = random.randint(1, 4)
                
                read_accesses.append(read_count)
                write_accesses.append(write_count)
            
            line_parts = [str(step)] + read_accesses + write_accesses
            f.write(','.join(map(str, line_parts)) + '\n')
    
    print(f"Created burst access file: {filename}")

def main():
    """Create sample access files"""
    print("Creating sample access files...")
    
    create_simple_access_file()
    create_advanced_access_file()
    create_burst_access_file()
    
    print("\nAccess file formats:")
    print("1. Simple: step,read_0,...,read_127,write_0,...,write_127")
    print("2. Advanced: step,reads,writes,low_reads,low_writes,bank_modes")
    print("3. Burst: step,reads,writes (with periodic high activity)")
    
    print("\nUsage:")
    print("  access_provider = FileAccessProvider(128, 'simple_access.csv')")
    print("  thermal_sim = StandaloneMemTherm('config.cfg', access_provider)")

if __name__ == "__main__":
    main() 