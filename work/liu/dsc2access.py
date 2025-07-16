#! /usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


'''
a python script to convert a description-level access pattern to a bank-level access pattern

input:
    - a description-level access pattern file
    - total number of banks
    - bank size (in MB)
    - total number of bank-groups
    - stack height
    - clock frequency (in MHz)

output:
    - a bank-level access pattern file (each line represents one ms)

input file example:
#Step    (read/write@Bank Group@Access Amount(in MB)@lanewidth(in Byte/s)) * N
0       read@0@128@256
        read@1@128@256
        read@2@128@256
        read@3@128@256
1       read@0@128@4
        read@1@128@4
        read@2@128@4
        read@3@128@4

output file example: (suppose total number of banks is 16, stack height is 4)
#step,read_0,read_1,...,read_{total_number_of_banks-1},write_0,write_1,...,write_{total_number_of_banks-1}
0,2,3,5,1,3,5,3,2,1,4,5,6,4,2,3,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
1,8,15,3,12,7,14,2,10,6,13,1,9,11,5,4,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
2,16,13,1,14,11,2,15,12,9,0,13,10,7,4,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
3,12,9,5,13,10,6,14,11,7,3,15,12,8,4,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0



operation:
each bank group has {total_number_of_banks/total_number_of_bank_groups} banks, spreaded in a stack of {stack_height} height
therefore, suppose total number of banks is 16, total number of bank groups is 4, stack height is 2,
then bank 0,1,8,9 are in bank group 0, bank 2,3,10,11 are in bank group 1, bank 4,5,12,13 are in bank group 2, bank 6,7,14,15 are in bank group 3
for each read/write access to a bank group, start from bank with smallest index, when the access amount exceeds the bank size, move to the next bank in the same bank group, until the access amount is less than the bank size
each access takes {lanewidth} bytes, so for each ms, the access amount is {lanewidth}x10^6/(1000/{ClockFrequency}) bytes
for input file, within the step, all operations executed in parallel, only when all operations in the step are finished, then move to the next step
for output file, each step represents a ms, each column represents a bank

'''


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert description-level access pattern to bank-level access pattern')
    parser.add_argument('input_file', help='Input description-level access pattern file')
    parser.add_argument('output_file', help='Output bank-level access pattern file')
    parser.add_argument('--total-banks', type=int, required=True, help='Total number of banks')
    parser.add_argument('--bank-size', type=float, required=True, help='Bank size in MB')
    parser.add_argument('--bank-groups', type=int, required=True, help='Total number of bank groups')
    parser.add_argument('--stack-height', type=int, required=True, help='Stack height')
    parser.add_argument('--clock-freq', type=float, required=True, help='Clock frequency in MHz')
    
    return parser.parse_args()


def get_bank_group_mapping(total_banks, bank_groups, stack_height):
    """Create mapping from bank group to bank indices."""
    banks_per_group = total_banks // bank_groups
    bank_mapping = {}
    
    for group in range(bank_groups):
        bank_mapping[group] = []
        for i in range(banks_per_group):
            # Calculate bank index based on group and position within group
            # Banks are distributed across the stack height
            bank_index = group * banks_per_group + i
            bank_mapping[group].append(bank_index)
    
    return bank_mapping


def calculate_access_amount(lanewidth, clock_freq_mhz):
    """Calculate access amount per millisecond based on lanewidth and clock frequency."""
    access_per_ms = lanewidth * 10**6 / (1000 / clock_freq_mhz)  # Convert bytes per second to bytes per millisecond
    print(f"DEBUG: calculate_access_amount - lanewidth={lanewidth} B, clock_freq={clock_freq_mhz} MHz")
    print(f"DEBUG:   access_per_ms = {lanewidth} * 10^6 / (1000 / {clock_freq_mhz}) = {access_per_ms} bytes/ms")
    return access_per_ms


def calculate_operation_duration(access_amount_mb, lanewidth, clock_freq_mhz):
    """Calculate how many milliseconds an operation will take."""
    access_amount_bytes = access_amount_mb * 1024 * 1024  # Convert MB to bytes
    access_per_ms = calculate_access_amount(lanewidth, clock_freq_mhz)
    
    # Calculate duration in milliseconds
    duration_ms = access_amount_bytes / access_per_ms
    duration = max(1, int(duration_ms))  # At least 1 ms
    
    print(f"DEBUG: calculate_operation_duration - {access_amount_mb}MB, {lanewidth} B lanewidth")
    print(f"DEBUG:   access_amount_bytes = {access_amount_mb} * 1024 * 1024 = {access_amount_bytes} bytes")
    print(f"DEBUG:   duration_ms = {access_amount_bytes} / {access_per_ms} = {duration_ms}")
    print(f"DEBUG:   final_duration = max(1, int({duration_ms})) = {duration} ms")
    
    return duration


def distribute_access_to_banks(access_type, bank_group, access_amount_mb, lanewidth, clock_freq_mhz, bank_size_mb, bank_mapping):
    """Distribute access amount across banks in the bank group sequentially."""
    banks_in_group = bank_mapping[bank_group]
    access_amount_bytes = access_amount_mb * 1024 * 1024  # Convert MB to bytes
    
    print(f"DEBUG: distribute_access_to_banks - {access_type} access to bank group {bank_group}")
    print(f"DEBUG:   access_amount = {access_amount_mb} MB = {access_amount_bytes} bytes")
    print(f"DEBUG:   banks_in_group = {banks_in_group}")
    print(f"DEBUG:   bank_size = {bank_size_mb} MB = {bank_size_mb * 1024 * 1024} bytes")
    
    # Initialize bank accesses
    bank_accesses = {bank: 0 for bank in banks_in_group}
    
    remaining_amount = access_amount_bytes
    current_bank_index = 0
    
    # Sequential access: fill one bank completely before moving to the next
    while remaining_amount > 0 and current_bank_index < len(banks_in_group):
        current_bank = banks_in_group[current_bank_index]
        bank_capacity = bank_size_mb * 1024 * 1024  # Convert MB to bytes
        
        print(f"DEBUG:   processing bank {current_bank} (index {current_bank_index})")
        print(f"DEBUG:     remaining_amount = {remaining_amount} bytes")
        print(f"DEBUG:     bank_capacity = {bank_capacity} bytes")
        
        if remaining_amount <= bank_capacity:
            # This bank can handle the remaining amount
            # Calculate number of accesses based on lanewidth
            accesses = int(remaining_amount / lanewidth)
            bank_accesses[current_bank] = accesses
            remaining_amount = 0
            print(f"DEBUG:     bank {current_bank} can handle remaining amount")
            print(f"DEBUG:     accesses = int({remaining_amount} / {lanewidth}) = {accesses}")
        else:
            # This bank is fully utilized
            accesses = int(bank_capacity / lanewidth)
            bank_accesses[current_bank] = accesses
            remaining_amount -= bank_capacity
            current_bank_index += 1
            print(f"DEBUG:     bank {current_bank} fully utilized")
            print(f"DEBUG:     accesses = int({bank_capacity} / {lanewidth}) = {accesses}")
            print(f"DEBUG:     remaining_amount = {remaining_amount} bytes")
    
    print(f"DEBUG:   final bank_accesses = {bank_accesses}")
    return bank_accesses


def process_operations_to_timeline(operations, bank_mapping, bank_size_mb, clock_freq_mhz):
    """Process operations and return millisecond-by-millisecond access pattern."""
    total_banks = max(max(banks) for banks in bank_mapping.values()) + 1
    
    print(f"DEBUG: process_operations_to_timeline - {len(operations)} operations")
    print(f"DEBUG:   total_banks = {total_banks}")
    
    # Process each operation to get its duration and bank accesses
    operation_timeline = []
    
    for operation in operations:
        # Parse operation: read@bank_group@access_amount@lanewidth
        parts = operation.split('@')
        if len(parts) != 4:
            print(f"Warning: Invalid operation format: {operation}")
            continue
            
        access_type, bank_group_str, access_amount_str, lanewidth_str = parts
        
        try:
            bank_group = int(bank_group_str)
            access_amount = float(access_amount_str)
            op_lanewidth = float(lanewidth_str)
        except ValueError:
            print(f"Warning: Invalid values in operation: {operation}")
            continue
        
        if bank_group not in bank_mapping:
            print(f"Warning: Invalid bank group {bank_group}")
            continue
        
        print(f"DEBUG: processing operation: {operation}")
        print(f"DEBUG:   access_type = {access_type}, bank_group = {bank_group}")
        print(f"DEBUG:   access_amount = {access_amount} MB, lanewidth = {op_lanewidth} B")
        
        # Calculate duration of this operation
        duration_ms = calculate_operation_duration(access_amount, op_lanewidth, clock_freq_mhz)
        
        # Distribute access to banks
        bank_accesses = distribute_access_to_banks(
            access_type, bank_group, access_amount, op_lanewidth, 
            clock_freq_mhz, bank_size_mb, bank_mapping
        )
        
        # Add to timeline
        operation_timeline.append({
            'type': access_type.lower(),
            'duration': duration_ms,
            'bank_accesses': bank_accesses,
            'bank_group': bank_group
        })
        
        print(f"DEBUG:   operation added to timeline: duration={duration_ms}ms, bank_accesses={bank_accesses}")
    
    # Find the maximum duration among all operations in this step
    max_duration = max(op['duration'] for op in operation_timeline) if operation_timeline else 1
    
    print(f"DEBUG: max_duration = {max_duration}ms")
    
    # Generate millisecond-by-millisecond access pattern
    timeline = []
    for ms in range(max_duration):
        read_accesses = [0] * total_banks
        write_accesses = [0] * total_banks
        
        print(f"DEBUG: processing millisecond {ms}")
        
        for op in operation_timeline:
            # For sequential access, determine which bank should be active at this millisecond
            banks_in_group = bank_mapping[op['bank_group']]
            total_accesses = sum(op['bank_accesses'].values())
            
            print(f"DEBUG:   operation {op['type']}@{op['bank_group']}: total_accesses={total_accesses}")
            
            if total_accesses > 0:
                # Calculate which bank should be active at this millisecond
                # Banks are activated sequentially, so we need to find which one is active now
                current_access = 0
                active_bank = None
                active_accesses = 0
                
                for bank in banks_in_group:
                    bank_access_count = op['bank_accesses'].get(bank, 0)
                    if bank_access_count > 0:
                        # Calculate the time period this bank should be active
                        bank_duration = (bank_access_count / total_accesses) * op['duration']
                        bank_start_ms = (current_access / total_accesses) * op['duration']
                        bank_end_ms = bank_start_ms + bank_duration
                        
                        print(f"DEBUG:     bank {bank}: access_count={bank_access_count}, duration={bank_duration}ms")
                        print(f"DEBUG:       active period: {bank_start_ms}ms to {bank_end_ms}ms")
                        
                        # Check if this bank should be active at current millisecond
                        if bank_start_ms <= ms < bank_end_ms:
                            active_bank = bank
                            active_accesses = bank_access_count / bank_duration
                            print(f"DEBUG:       bank {bank} is ACTIVE at ms {ms}")
                            break
                        else:
                            print(f"DEBUG:       bank {bank} is INACTIVE at ms {ms}")
                        
                        current_access += bank_access_count
                
                # Add to appropriate access type
                if active_bank is not None:
                    if op['type'] == 'read':
                        read_accesses[active_bank] += int(active_accesses)
                        print(f"DEBUG:     adding {int(active_accesses)} read accesses to bank {active_bank}")
                    elif op['type'] == 'write':
                        write_accesses[active_bank] += int(active_accesses)
                        print(f"DEBUG:     adding {int(active_accesses)} write accesses to bank {active_bank}")
        
        print(f"DEBUG:   ms {ms} result: read={read_accesses}, write={write_accesses}")
        timeline.append((read_accesses, write_accesses))
    
    print(f"DEBUG: generated {len(timeline)} millisecond entries")
    return timeline


def parse_input_file(filename):
    """Parse the input description-level access pattern file."""
    steps = []
    
    with open(filename, 'r') as f:
        current_step = None
        current_operations = []
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check if this is a new step (starts with a number)
            if line[0].isdigit():
                # Save previous step if exists
                if current_step is not None:
                    steps.append((current_step, current_operations))
                
                # Start new step
                parts = line.split()
                current_step = int(parts[0])
                current_operations = []
                
                # Parse the first operation on this line
                if len(parts) > 1:
                    operation = parts[1]
                    current_operations.append(operation)
            else:
                # This is a continuation line with more operations
                parts = line.split()
                if parts:
                    current_operations.append(parts[0])
        
        # Don't forget the last step
        if current_step is not None:
            steps.append((current_step, current_operations))
    
    return steps

def write_output_file(filename, steps_data, total_banks):
    """Write the bank-level access pattern to output file."""
    with open(filename, 'w') as f:
        # Write header
        header_parts = ['step']
        for i in range(total_banks):
            header_parts.append(f'read_{i}')
        for i in range(total_banks):
            header_parts.append(f'write_{i}')
        
        f.write(','.join(header_parts) + '\n')
        
        # Write data
        for step_num, (read_accesses, write_accesses) in steps_data:
            row_parts = [str(step_num)]
            row_parts.extend([str(int(access)) for access in read_accesses])
            row_parts.extend([str(int(access)) for access in write_accesses])
            f.write(','.join(row_parts) + '\n')


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"DEBUG: Starting dsc2access.py")
    print(f"DEBUG:   input_file = {args.input_file}")
    print(f"DEBUG:   output_file = {args.output_file}")
    print(f"DEBUG:   total_banks = {args.total_banks}")
    print(f"DEBUG:   bank_size = {args.bank_size} MB")
    print(f"DEBUG:   bank_groups = {args.bank_groups}")
    print(f"DEBUG:   stack_height = {args.stack_height}")
    print(f"DEBUG:   clock_freq = {args.clock_freq} MHz")
    
    # Validate inputs
    if args.total_banks % args.bank_groups != 0:
        print("Error: Total banks must be divisible by number of bank groups")
        sys.exit(1)
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    # Create bank group mapping
    bank_mapping = get_bank_group_mapping(args.total_banks, args.bank_groups, args.stack_height)
    print(f"DEBUG: bank_mapping = {bank_mapping}")
    
    # Parse input file
    steps = parse_input_file(args.input_file)
    print(f"DEBUG: parsed {len(steps)} steps from input file")
    
    # Process each step
    steps_data = []
    current_ms = 0
    
    for step_num, operations in steps:
        print(f"DEBUG: processing step {step_num} with {len(operations)} operations")
        # Use the new function to process operations and generate timeline
        timeline = process_operations_to_timeline(
            operations, bank_mapping, args.bank_size, args.clock_freq
        )
        
        # Convert timeline to steps_data format
        for ms, (read_accesses, write_accesses) in enumerate(timeline):
            steps_data.append((current_ms + ms, (read_accesses, write_accesses)))
        
        # Update current_ms for next step
        current_ms += len(timeline)
        print(f"DEBUG: step {step_num} generated {len(timeline)} millisecond entries")
    
    print(f"DEBUG: total steps_data entries = {len(steps_data)}")
    
    # Write output file
    write_output_file(args.output_file, steps_data, args.total_banks)
    
    print(f"Successfully converted {args.input_file} to {args.output_file}")


if __name__ == "__main__":
    main()


