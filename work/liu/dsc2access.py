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

CONCURRENT ACCESS BEHAVIOR:
When the total access time for an operation within a bank group is ≤1ms, all banks in that bank group can be accessed concurrently.
When the total access time exceeds 1ms, banks are accessed sequentially as before.

'''


class Dsc2AccessConverter:
    """A class to convert description-level access patterns to bank-level access patterns."""
    
    def __init__(self, total_banks, bank_size, stack_height, clock_freq, 
                 bank_per_row=None, bank_per_col=None, group_rows=None, 
                 group_cols=None, num_groups=None, debug=False):
        """
        Initialize the converter with memory configuration.
        
        Args:
            total_banks (int): Total number of banks
            bank_size (float): Bank size in MB
            stack_height (int): Stack height
            clock_freq (float): Clock frequency in MHz
            bank_per_row (int, optional): Number of banks per row
            bank_per_col (int, optional): Number of banks per column
            group_rows (int, optional): Number of rows per group
            group_cols (int, optional): Number of columns per group
            num_groups (int, optional): Number of bank groups
            debug (bool): Enable debug output
        """
        self.total_banks = total_banks
        self.bank_size = bank_size
        self.stack_height = stack_height
        self.clock_freq = clock_freq
        self.bank_per_row = bank_per_row
        self.bank_per_col = bank_per_col
        self.group_rows = group_rows
        self.group_cols = group_cols
        self.num_groups = num_groups
        self.debug = debug
        
        # Create bank group mapping
        self.bank_mapping = self._get_bank_group_mapping()
        
        if self.debug:
            print(f"DEBUG: Dsc2AccessConverter initialized")
            print(f"DEBUG:   total_banks = {self.total_banks}")
            print(f"DEBUG:   bank_size = {self.bank_size} MB")
            print(f"DEBUG:   stack_height = {self.stack_height}")
            print(f"DEBUG:   clock_freq = {self.clock_freq} MHz")
            print(f"DEBUG:   bank_mapping = {self.bank_mapping}")
    
    def _get_bank_group_mapping(self):
        """
        Flexible bank group mapping.
        Returns:
            dict: {group_id: [bank indices]}
        """
        # Infer stack layout if not provided
        if self.bank_per_row is None or self.bank_per_col is None:
            banks_per_stack = self.total_banks // self.stack_height
            # Try to infer a square/rectangular layout
            for possible_row in range(1, int(banks_per_stack ** 0.5) + 1):
                if banks_per_stack % possible_row == 0:
                    possible_col = banks_per_stack // possible_row
                    # Prefer square-ish
                    if abs(possible_row - possible_col) <= 2:
                        self.bank_per_row = possible_col
                        self.bank_per_col = possible_row
            if self.bank_per_row is None or self.bank_per_col is None:
                raise ValueError("Cannot infer stack layout. Please specify banks_per_row and banks_per_col.")

        # Infer group shape if not provided
        if self.group_rows is None or self.group_cols is None:
            # Default: divide stack into 2x2 blocks if possible
            self.group_rows = 2 if self.bank_per_col % 2 == 0 else 1
            self.group_cols = 2 if self.bank_per_row % 2 == 0 else 1

        # Compute number of groups if not provided
        groups_per_row = self.bank_per_row // self.group_cols
        groups_per_col = self.bank_per_col // self.group_rows
        inferred_num_groups = groups_per_row * groups_per_col
        if self.num_groups is None:
            self.num_groups = inferred_num_groups
        else:
            if self.num_groups != inferred_num_groups:
                raise ValueError(f"num_groups ({self.num_groups}) does not match inferred ({inferred_num_groups}) from shape.")

        bank_mapping = {g: [] for g in range(self.num_groups)}
        banks_per_stack = self.bank_per_row * self.bank_per_col

        # For each stack, divide it into group blocks and assign to groups
        for stack in range(self.stack_height):
            stack_start = stack * banks_per_stack
            
            # For each group block position within the stack
            for group_block_row in range(groups_per_col):
                for group_block_col in range(groups_per_row):
                    # Calculate which group this block belongs to
                    group_id = group_block_row * groups_per_row + group_block_col
                    if group_id >= self.num_groups:
                        continue
                    
                    # Add all banks in this block to the group
                    for r in range(self.group_rows):
                        for c in range(self.group_cols):
                            # Calculate the actual row and column in the stack
                            actual_row = group_block_row * self.group_rows + r
                            actual_col = group_block_col * self.group_cols + c
                            
                            # Calculate the bank index
                            bank_index = stack_start + actual_row * self.bank_per_row + actual_col
                            if bank_index < self.total_banks:
                                bank_mapping[group_id].append(bank_index)
        
        return bank_mapping
    
    def _calculate_access_amount(self, lanewidth):
        """Calculate access amount per millisecond based on lanewidth and clock frequency."""
        access_per_ms = lanewidth * 10**6 / (1000 / self.clock_freq)  # Convert bytes per second to bytes per millisecond
        if self.debug:
            print(f"DEBUG: calculate_access_amount - lanewidth={lanewidth} B, clock_freq={self.clock_freq} MHz")
            print(f"DEBUG:   access_per_ms = {lanewidth} * 10^6 / (1000 / {self.clock_freq}) = {access_per_ms} bytes/ms")
        return access_per_ms

    def _calculate_operation_duration(self, access_amount_mb, lanewidth):
        """Calculate how many milliseconds an operation will take."""
        access_amount_bytes = access_amount_mb * 1024 * 1024  # Convert MB to bytes
        access_per_ms = self._calculate_access_amount(lanewidth)
        
        # Calculate duration in milliseconds
        duration_ms = access_amount_bytes / access_per_ms
        duration = max(1, int(duration_ms))  # At least 1 ms
        
        if self.debug:
            print(f"DEBUG: calculate_operation_duration - {access_amount_mb}MB, {lanewidth} B lanewidth")
            print(f"DEBUG:   access_amount_bytes = {access_amount_mb} * 1024 * 1024 = {access_amount_bytes} bytes")
            print(f"DEBUG:   duration_ms = {access_amount_bytes} / {access_per_ms} = {duration_ms}")
            print(f"DEBUG:   final_duration = max(1, int({duration_ms})) = {duration} ms")
        
        return duration

    def _distribute_access_to_banks(self, access_type, bank_group, access_amount_mb, lanewidth):
        """Distribute access amount across banks in the bank group sequentially."""
        banks_in_group = self.bank_mapping[bank_group]
        access_amount_bytes = access_amount_mb * 1024 * 1024  # Convert MB to bytes
        
        if self.debug:
            print(f"DEBUG: distribute_access_to_banks - {access_type} access to bank group {bank_group}")
            print(f"DEBUG:   access_amount = {access_amount_mb} MB = {access_amount_bytes} bytes")
            print(f"DEBUG:   banks_in_group = {banks_in_group}")
            print(f"DEBUG:   bank_size = {self.bank_size} MB = {self.bank_size * 1024 * 1024} bytes")
        
        # Initialize bank accesses
        bank_accesses = {bank: 0 for bank in banks_in_group}
        
        remaining_amount = access_amount_bytes
        current_bank_index = 0
        
        # Sequential access: fill one bank completely before moving to the next
        while remaining_amount > 0 and current_bank_index < len(banks_in_group):
            current_bank = banks_in_group[current_bank_index]
            bank_capacity = self.bank_size * 1024 * 1024  # Convert MB to bytes
            
            if self.debug:
                print(f"DEBUG:   processing bank {current_bank} (index {current_bank_index})")
                print(f"DEBUG:     remaining_amount = {remaining_amount} bytes")
                print(f"DEBUG:     bank_capacity = {bank_capacity} bytes")
            
            if remaining_amount <= bank_capacity:
                # This bank can handle the remaining amount
                # Calculate number of accesses based on lanewidth
                accesses = int(remaining_amount / lanewidth)
                bank_accesses[current_bank] = accesses
                remaining_amount = 0
                if self.debug:
                    print(f"DEBUG:     bank {current_bank} can handle remaining amount")
                    print(f"DEBUG:     accesses = int({remaining_amount} / {lanewidth}) = {accesses}")
            else:
                # This bank is fully utilized
                accesses = int(bank_capacity / lanewidth)
                bank_accesses[current_bank] = accesses
                remaining_amount -= bank_capacity
                current_bank_index += 1
                if self.debug:
                    print(f"DEBUG:     bank {current_bank} fully utilized")
                    print(f"DEBUG:     accesses = int({bank_capacity} / {lanewidth}) = {accesses}")
                    print(f"DEBUG:     remaining_amount = {remaining_amount} bytes")
        
        if self.debug:
            print(f"DEBUG:   final bank_accesses = {bank_accesses}")
        return bank_accesses

    def _process_operations_to_timeline(self, operations):
        """Process operations and return millisecond-by-millisecond access pattern."""
        total_banks = max(max(banks) for banks in self.bank_mapping.values()) + 1
        
        if self.debug:
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
            
            if bank_group not in self.bank_mapping:
                print(f"Warning: Invalid bank group {bank_group}")
                continue
            
            if self.debug:
                print(f"DEBUG: processing operation: {operation}")
                print(f"DEBUG:   access_type = {access_type}, bank_group = {bank_group}")
                print(f"DEBUG:   access_amount = {access_amount} MB, lanewidth = {op_lanewidth} B")
            
            # Calculate duration of this operation
            duration_ms = self._calculate_operation_duration(access_amount, op_lanewidth)
            
            # Distribute access to banks
            bank_accesses = self._distribute_access_to_banks(
                access_type, bank_group, access_amount, op_lanewidth
            )
            
            # Add to timeline
            operation_timeline.append({
                'type': access_type.lower(),
                'duration': duration_ms,
                'bank_accesses': bank_accesses,
                'bank_group': bank_group,
                'lanewidth': op_lanewidth
            })
            
            if self.debug:
                print(f"DEBUG:   operation added to timeline: duration={duration_ms}ms, bank_accesses={bank_accesses}")
        
        # Find the maximum duration among all operations in this step
        max_duration = max(op['duration'] for op in operation_timeline) if operation_timeline else 1
        
        if self.debug:
            print(f"DEBUG: max_duration = {max_duration}ms")
        
        # Generate millisecond-by-millisecond access pattern
        timeline = []
        for ms in range(max_duration):
            read_accesses = [0] * total_banks
            write_accesses = [0] * total_banks
            
            if self.debug:
                print(f"DEBUG: processing millisecond {ms}")
            
            for op in operation_timeline:
                banks_in_group = self.bank_mapping[op['bank_group']]
                total_accesses = sum(op['bank_accesses'].values())
                
                if self.debug:
                    print(f"DEBUG:   operation {op['type']}@{op['bank_group']}: total_accesses={total_accesses}")
                
                if total_accesses > 0:
                    # Check if this operation can be completed within 1ms (concurrent access)
                    # Calculate total access time for this operation
                    # total_accesses is the number of accesses, each access is op['lanewidth'] bytes
                    total_bytes = total_accesses * op['lanewidth']
                    access_per_ms = self._calculate_access_amount(op['lanewidth'])
                    total_access_time_ms = total_bytes / access_per_ms if access_per_ms > 0 else 1
                    
                    if self.debug:
                        print(f"DEBUG:     total_accesses = {total_accesses}")
                        print(f"DEBUG:     total_bytes = {total_accesses} * {op['lanewidth']} = {total_bytes}")
                        print(f"DEBUG:     access_per_ms = {access_per_ms} bytes/ms")
                        print(f"DEBUG:     total_access_time_ms = {total_bytes} / {access_per_ms} = {total_access_time_ms}ms")
                    
                    if total_access_time_ms <= 1.0:
                        # Concurrent access: all banks in the group can be accessed simultaneously
                        if self.debug:
                            print(f"DEBUG:     using CONCURRENT access (≤1ms)")
                        for bank in banks_in_group:
                            bank_access_count = op['bank_accesses'].get(bank, 0)
                            if bank_access_count > 0:
                                if op['type'] == 'read':
                                    read_accesses[bank] += int(bank_access_count)
                                    if self.debug:
                                        print(f"DEBUG:       adding {int(bank_access_count)} read accesses to bank {bank}")
                                elif op['type'] == 'write':
                                    write_accesses[bank] += int(bank_access_count)
                                    if self.debug:
                                        print(f"DEBUG:       adding {int(bank_access_count)} write accesses to bank {bank}")
                    else:
                        # Sequential access: determine which banks should be active at this millisecond
                        if self.debug:
                            print(f"DEBUG:     using SEQUENTIAL access (>1ms) for ms {ms}")
                        current_access = 0
                        active_banks = []  # Collect all active banks for this millisecond
                        
                        for bank in banks_in_group:
                            bank_access_count = op['bank_accesses'].get(bank, 0)
                            if bank_access_count > 0:
                                # Calculate the time period this bank should be active
                                bank_duration = (bank_access_count / total_accesses) * op['duration']
                                bank_start_ms = (current_access / total_accesses) * op['duration']
                                bank_end_ms = bank_start_ms + bank_duration
                                
                                if self.debug:
                                    print(f"DEBUG:       bank {bank}: access_count={bank_access_count}, duration={bank_duration}ms")
                                    print(f"DEBUG:         active period: {bank_start_ms}ms to {bank_end_ms}ms")
                                
                                # Check if this bank should be active at current millisecond
                                # A bank is active if its active period overlaps with the current millisecond
                                # Current millisecond period: [ms, ms+1)
                                # Bank active period: [bank_start_ms, bank_end_ms)
                                # Overlap occurs if: bank_start_ms < ms+1 AND bank_end_ms > ms
                                if bank_start_ms < (ms + 1) and bank_end_ms > ms:
                                    # Calculate how much of this bank's access should occur in this millisecond
                                    # The proportion of the bank's duration that overlaps with this millisecond
                                    overlap_start = max(bank_start_ms, ms)
                                    overlap_end = min(bank_end_ms, ms + 1)
                                    overlap_duration = overlap_end - overlap_start
                                    bank_total_duration = bank_end_ms - bank_start_ms
                                    
                                    # Proportion of bank's accesses that should occur in this millisecond
                                    proportion = overlap_duration / bank_total_duration
                                    active_accesses = bank_access_count * proportion
                                    
                                    active_banks.append((bank, active_accesses))
                                    if self.debug:
                                        print(f"DEBUG:         bank {bank} is ACTIVE at ms {ms} (overlap: {overlap_start:.3f}ms to {overlap_end:.3f}ms, proportion: {proportion:.3f}, accesses: {active_accesses:.0f})")
                                else:
                                    if self.debug:
                                        print(f"DEBUG:         bank {bank} is INACTIVE at ms {ms}")
                                
                                current_access += bank_access_count
                        
                        if self.debug:
                            print(f"DEBUG:       active_banks for ms {ms}: {active_banks}")
                        
                        # Add all active banks to appropriate access type
                        for active_bank, active_accesses in active_banks:
                            if op['type'] == 'read':
                                read_accesses[active_bank] += int(active_accesses)
                                if self.debug:
                                    print(f"DEBUG:       adding {int(active_accesses)} read accesses to bank {active_bank}")
                            elif op['type'] == 'write':
                                write_accesses[active_bank] += int(active_accesses)
                                if self.debug:
                                    print(f"DEBUG:       adding {int(active_accesses)} write accesses to bank {active_bank}")
            
            if self.debug:
                print(f"DEBUG:   ms {ms} result: read={read_accesses}, write={write_accesses}")
            timeline.append((read_accesses, write_accesses))
        
        if self.debug:
            print(f"DEBUG: generated {len(timeline)} millisecond entries")
        return timeline

    def parse_input_file(self, filename):
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

    def write_output_file(self, filename, steps_data):
        """Write the bank-level access pattern to output file."""
        with open(filename, 'w') as f:
            # Write header
            header_parts = ['#step']
            for i in range(self.total_banks):
                header_parts.append(f'read_{i:02d}')
            for i in range(self.total_banks):
                header_parts.append(f'write_{i:02d}')
            
            f.write(','.join(header_parts) + '\n')
            
            # Write data with consistent formatting
            for step_num, (read_accesses, write_accesses) in steps_data:
                row_parts = [f'{step_num:02d}']
                row_parts.extend([f'{int(access):08d}' for access in read_accesses])
                row_parts.extend([f'{int(access):08d}' for access in write_accesses])
                f.write(','.join(row_parts) + '\n')

    def write_output_table(self, filename, steps_data):
        # Prepare headers
        headers = ['step'] + [f'read_{i}' for i in range(self.total_banks)] + [f'write_{i}' for i in range(self.total_banks)]
        # Gather all rows as strings
        rows = []
        for step_num, (read_accesses, write_accesses) in steps_data:
            row = [str(step_num)] + [str(int(x)) for x in read_accesses] + [str(int(x)) for x in write_accesses]
            rows.append(row)
        # Calculate max width for each column
        col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
        # Write to file
        with open(filename, 'w') as f:
            # Write header
            f.write('  '.join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + '\n')
            # Write rows
            for row in rows:
                f.write('  '.join(row[i].rjust(col_widths[i]) for i in range(len(row))) + '\n')

    def convert_file(self, input_file, output_file):
        """
        Convert a description-level access pattern file to bank-level access pattern.
        
        Args:
            input_file (str): Path to input file
            output_file (str): Path to output file
        """
        if self.debug:
            print(f"DEBUG: Starting conversion")
            print(f"DEBUG:   input_file = {input_file}")
            print(f"DEBUG:   output_file = {output_file}")
        
        # Validate inputs
        if self.num_groups and self.total_banks % self.num_groups != 0:
            raise ValueError("Total banks must be divisible by number of bank groups")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} does not exist")
        
        # Parse input file
        steps = self.parse_input_file(input_file)
        if self.debug:
            print(f"DEBUG: parsed {len(steps)} steps from input file")
        
        # Process each step
        steps_data = []
        current_ms = 0
        
        for step_num, operations in steps:
            if self.debug:
                print(f"DEBUG: processing step {step_num} with {len(operations)} operations")
            # Use the new function to process operations and generate timeline
            timeline = self._process_operations_to_timeline(operations)
            
            # Convert timeline to steps_data format
            for ms, (read_accesses, write_accesses) in enumerate(timeline):
                steps_data.append((current_ms + ms, (read_accesses, write_accesses)))
            
            # Update current_ms for next step
            current_ms += len(timeline)
            if self.debug:
                print(f"DEBUG: step {step_num} generated {len(timeline)} millisecond entries")
        
        if self.debug:
            print(f"DEBUG: total steps_data entries = {len(steps_data)}")
        
        # Write output file
        #self.write_output_file(output_file, steps_data)
        self.write_output_table(output_file, steps_data)
        
        if self.debug:
            print(f"Successfully converted {input_file} to {output_file}")
        
        return steps_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert description-level access pattern to bank-level access pattern')
    parser.add_argument('input_file', help='Input description-level access pattern file')
    parser.add_argument('output_file', help='Output bank-level access pattern file')
    parser.add_argument('--total-banks', type=int, required=True, help='Total number of banks')
    parser.add_argument('--bank-size', type=float, required=True, help='Bank size in MB')
    parser.add_argument('--stack-height', type=int, required=True, help='Stack height')
    parser.add_argument('--clock-freq', type=float, required=True, help='Clock frequency in MHz')
    parser.add_argument('--bank-per-row', type=int, help='Number of banks per row (auto-inferred if not specified)')
    parser.add_argument('--bank-per-col', type=int, help='Number of banks per column (auto-inferred if not specified)')
    parser.add_argument('--group-rows', type=int, help='Number of rows per group (auto-inferred if not specified)')
    parser.add_argument('--group-cols', type=int, help='Number of columns per group (auto-inferred if not specified)')
    parser.add_argument('--num-groups', type=int, help='Number of bank groups (auto-inferred if not specified)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.debug:
        print(f"DEBUG: Starting dsc2access.py")
        print(f"DEBUG:   input_file = {args.input_file}")
        print(f"DEBUG:   output_file = {args.output_file}")
        print(f"DEBUG:   total_banks = {args.total_banks}")
        print(f"DEBUG:   bank_size = {args.bank_size} MB")
        print(f"DEBUG:   bank_per_row = {args.bank_per_row}")
        print(f"DEBUG:   bank_per_col = {args.bank_per_col}")
        print(f"DEBUG:   group_rows = {args.group_rows}")
        print(f"DEBUG:   group_cols = {args.group_cols}")
        print(f"DEBUG:   num_groups = {args.num_groups}")
        print(f"DEBUG:   stack_height = {args.stack_height}")
        print(f"DEBUG:   clock_freq = {args.clock_freq} MHz")
    
    # Create converter instance
    converter = Dsc2AccessConverter(
        total_banks=args.total_banks,
        bank_size=args.bank_size,
        stack_height=args.stack_height,
        clock_freq=args.clock_freq,
        bank_per_row=args.bank_per_row,
        bank_per_col=args.bank_per_col,
        group_rows=args.group_rows,
        group_cols=args.group_cols,
        num_groups=args.num_groups,
        debug=args.debug
    )
    
    # Convert the file
    try:
        converter.convert_file(args.input_file, args.output_file)
        print(f"Successfully converted {args.input_file} to {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


