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
0,2,3,5,1,3,5,3,2,1,4,5,6,4,2,3,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
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
                 group_cols=None, num_groups=None, debug=False, distribute_across_group=False, distribution_noise=0.2):
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
            distribute_across_group (bool): Distribute access across all banks in group with noise
            distribution_noise (float): Noise level for distribution (0.0 to 1.0, default 0.2)
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
        self.distribute_across_group = distribute_across_group
        self.distribution_noise = distribution_noise
        
        # Create bank group mapping
        self.bank_mapping = self._get_bank_group_mapping()
        
        if self.debug:
            print(f"DEBUG: Dsc2AccessConverter initialized")
            print(f"DEBUG:   total_banks = {self.total_banks}")
            print(f"DEBUG:   bank_size = {self.bank_size} MB")
            print(f"DEBUG:   stack_height = {self.stack_height}")
            print(f"DEBUG:   clock_freq = {self.clock_freq} MHz")
            print(f"DEBUG:   distribute_across_group = {self.distribute_across_group}")
            print(f"DEBUG:   distribution_noise = {self.distribution_noise}")
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

    def _distribute_access_to_banks(self, access_type, bank_group, access_amount_mb, lanewidth, starting_stack=None, starting_bank=0):
        """Distribute access amount across banks in the bank group sequentially, starting from specified stack."""
        banks_in_group = self.bank_mapping[bank_group]
        access_amount_bytes = access_amount_mb * 1024 * 1024  # Convert MB to bytes
        
        if self.debug:
            print(f"DEBUG: distribute_access_to_banks - {access_type} access to bank group {bank_group}")
            print(f"DEBUG:   access_amount = {access_amount_mb} MB = {access_amount_bytes} bytes")
            print(f"DEBUG:   banks_in_group = {banks_in_group}")
            print(f"DEBUG:   starting_stack = {starting_stack}")
            print(f"DEBUG:   bank_size = {self.bank_size} MB = {self.bank_size * 1024 * 1024} bytes")
            print(f"DEBUG:   distribute_across_group = {self.distribute_across_group}")
        
        # If starting_stack is specified and NOT distributing across group, filter banks to only those from that stack
        if starting_stack is not None and not self.distribute_across_group:
            banks_per_stack = self.total_banks // self.stack_height
            stack_start_bank = starting_stack * banks_per_stack
            stack_end_bank = (starting_stack + 1) * banks_per_stack
            
            # Filter banks to only those from the specified stack
            filtered_banks = [bank for bank in banks_in_group if stack_start_bank <= bank < stack_end_bank]
            
            if self.debug:
                print(f"DEBUG:   stack {starting_stack} banks: {stack_start_bank} to {stack_end_bank-1}")
                print(f"DEBUG:   filtered_banks = {filtered_banks}")
            
            # If no banks in this stack for this group, return empty
            if not filtered_banks:
                if self.debug:
                    print(f"DEBUG:   no banks in stack {starting_stack} for group {bank_group}")
                return {}
            
            banks_in_group = filtered_banks
        
        # Initialize bank accesses
        bank_accesses = {bank: 0 for bank in banks_in_group}
        
        if self.distribute_across_group:
            # Distribute access across ALL banks in the group with noise
            if self.debug:
                print(f"DEBUG:   distributing across ALL {len(banks_in_group)} banks with noise")
                print(f"DEBUG:   all banks in group {bank_group}: {banks_in_group}")
            
            # Calculate total number of accesses needed
            total_accesses = int(access_amount_bytes / lanewidth)
            
            if total_accesses > 0:
                # Generate distribution weights with noise for ALL banks
                weights = np.random.normal(1.0, self.distribution_noise, len(banks_in_group))
                weights = np.maximum(weights, 0.1)  # Ensure minimum weight
                weights = weights / np.sum(weights)  # Normalize to sum to 1
                
                # Distribute accesses according to weights to ALL banks
                for i, bank in enumerate(banks_in_group):
                    bank_accesses[bank] = int(total_accesses * weights[i])
                
                # Ensure at least one access per bank if there are enough total accesses
                if total_accesses >= len(banks_in_group):
                    for bank in banks_in_group:
                        if bank_accesses[bank] == 0:
                            bank_accesses[bank] = 1
                
                # If we have fewer total accesses than banks, distribute evenly with noise
                if total_accesses < len(banks_in_group):
                    # Distribute the accesses evenly across all banks with some randomness
                    base_accesses_per_bank = total_accesses // len(banks_in_group)
                    remaining_accesses = total_accesses % len(banks_in_group)
                    
                    # Add base accesses to all banks
                    for bank in banks_in_group:
                        bank_accesses[bank] = base_accesses_per_bank
                    
                    # Distribute remaining accesses randomly
                    if remaining_accesses > 0:
                        import random
                        selected_banks = random.sample(banks_in_group, remaining_accesses)
                        for bank in selected_banks:
                            bank_accesses[bank] += 1
                
                if self.debug:
                    print(f"DEBUG:   total_accesses = {total_accesses}")
                    print(f"DEBUG:   weights = {weights}")
                    print(f"DEBUG:   distributed_accesses = {bank_accesses}")
        else:
            # Original sequential access behavior
            remaining_amount = access_amount_bytes
            current_bank_index = starting_bank  # Start from the specified bank within the group
            
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
            
            # if still remaining amount, map to the next stack until all remaining amount is mapped
            # If there is still remaining_amount after filling all banks in the current stack,
            # continue mapping to the next stack(s) in a round-robin fashion until all remaining_amount is mapped.
            # This only applies if stack_height > 1 (i.e., multiple stacks exist).
            next_stack = (starting_stack + 1 + self.stack_height) % self.stack_height if self.stack_height > 1 else starting_stack
            while remaining_amount > 0 and self.stack_height > 1:
                # Get banks in the same group for the next stack
                stack_start_bank = next_stack * self.total_banks // self.stack_height # This line was incorrect, should be self.bank_per_row * self.bank_per_col
                stack_end_bank = stack_start_bank + self.total_banks // self.stack_height # This line was incorrect, should be self.bank_per_row * self.bank_per_col
                banks_in_next_stack = [
                    bank for bank in self.bank_mapping[bank_group]
                    if stack_start_bank <= bank < stack_end_bank
                ]
                if not banks_in_next_stack:
                    # No banks in this stack for this group, skip to next stack
                    next_stack = (next_stack + 1) % self.stack_height
                    # If we've looped all stacks and found nothing, break to avoid infinite loop
                    if next_stack == starting_stack:
                        break
                    continue
                # Fill banks in the next stack sequentially
                for bank in banks_in_next_stack:
                    bank_capacity = self.bank_size * 1024 * 1024  # MB to bytes
                    already_accessed = bank_accesses.get(bank, 0)
                    # Only fill if not already filled in this round
                    if already_accessed == 0:
                        if remaining_amount <= bank_capacity:
                            accesses = int(remaining_amount / lanewidth)
                            bank_accesses[bank] = accesses
                            remaining_amount = 0
                            if self.debug:
                                print(f"DEBUG:     (extra stack) bank {bank} can handle remaining amount")
                                print(f"DEBUG:     accesses = int({remaining_amount} / {lanewidth}) = {accesses}")
                            break  # Done mapping
                        else:
                            accesses = int(bank_capacity / lanewidth)
                            bank_accesses[bank] = accesses
                            remaining_amount -= bank_capacity
                            if self.debug:
                                print(f"DEBUG:     (extra stack) bank {bank} fully utilized")
                                print(f"DEBUG:     accesses = int({bank_capacity} / {lanewidth}) = {accesses}")
                                print(f"DEBUG:     remaining_amount = {remaining_amount} bytes")
                # Move to next stack
                next_stack = (next_stack + 1) % self.stack_height
                # # If we've looped all stacks, break to avoid infinite loop
                if next_stack == starting_stack:
                    break

        if self.debug:
            print(f"DEBUG:   final bank_accesses = {bank_accesses}")
        return bank_accesses

    def _process_operations_to_timeline(self, operations, global_max_lanewidth):
        """Process operations and return millisecond-by-millisecond access pattern."""
        total_banks = self.total_banks
        
        if self.debug:
            print(f"DEBUG: process_operations_to_timeline - {len(operations)} operations")
            print(f"DEBUG:   total_banks = {total_banks}")
            print(f"DEBUG:   global_max_lanewidth = {global_max_lanewidth} B")
        
        # Process each operation to get its duration and bank accesses
        operation_timeline = []
        has_prs_1ms_op = []
        for operation in operations:
            # Parse operation: read@bank_group@starting_stack@starting_bank@access_amount@lanewidth
            parts = operation.split('@')
            if len(parts) < 5:
                print(f"Warning: Invalid operation format: {operation}")
                continue
            
            # Handle both old format (5 parts) and new format (6 parts)
            if len(parts) == 5:
                # Old format: read@bank_group@starting_stack@access_amount@lanewidth
                access_type, bank_group_str, starting_stack_str, access_amount_str, lanewidth_str = parts
                starting_bank_str = "0"  # Default to first bank in group
            elif len(parts) == 6:
                # New format: read@bank_group@starting_stack@starting_bank@access_amount@lanewidth
                access_type, bank_group_str, starting_stack_str, starting_bank_str, access_amount_str, lanewidth_str = parts
            else:
                print(f"Warning: Invalid operation format: {operation}")
                continue
            
            try:
                bank_group = int(bank_group_str)
                starting_stack = int(starting_stack_str)
                starting_bank = int(starting_bank_str)
                access_amount = float(access_amount_str)
                op_lanewidth = float(lanewidth_str)
            except ValueError:
                print(f"Warning: Invalid values in operation: {operation}")
                continue
            
            if bank_group not in self.bank_mapping:
                print(f"Warning: Invalid bank group {bank_group}")
                continue
            
            if starting_stack < 0 or starting_stack >= self.stack_height:
                print(f"Warning: Invalid starting stack {starting_stack}")
                continue
            
            # Validate starting_bank within the bank group
            banks_in_group = self.bank_mapping[bank_group]
            if starting_bank < 0 or starting_bank >= len(banks_in_group):
                print(f"Warning: Invalid starting bank {starting_bank} for bank group {bank_group} (group has {len(banks_in_group)} banks)")
                continue
            
            if self.debug:
                print(f"DEBUG: processing operation: {operation}")
                print(f"DEBUG:   access_type = {access_type}, bank_group = {bank_group}")
                print(f"DEBUG:   starting_stack = {starting_stack}, starting_bank = {starting_bank}")
                print(f"DEBUG:   access_amount = {access_amount} MB, lanewidth = {op_lanewidth} B")
            
            # Calculate duration of this operation
            duration_ms = self._calculate_operation_duration(access_amount, op_lanewidth)
            
            # Distribute access to banks
            bank_accesses = self._distribute_access_to_banks(
                access_type, bank_group, access_amount, op_lanewidth, starting_stack, starting_bank
            )
            
            # Add to timeline
            operation_timeline.append({
                'type': access_type.lower(),
                'duration': duration_ms,
                'bank_accesses': bank_accesses,
                'bank_group': bank_group,
                'starting_stack': starting_stack,
                'starting_bank': starting_bank,
                'lanewidth': op_lanewidth
            })
            
            if self.debug:
                print(f"DEBUG:   operation added to timeline: duration={duration_ms}ms, bank_accesses={bank_accesses}")
        
        # Find the maximum duration among all operations in this step
        max_duration = max(op['duration'] for op in operation_timeline) if operation_timeline else 1
        
        if self.debug:
            print(f"DEBUG: max_duration = {max_duration}ms")
            print(f"DEBUG: using global_max_lanewidth = {global_max_lanewidth} B")
        
        # Generate millisecond-by-millisecond access pattern
        timeline = []
        for ms in range(max_duration):
            read_accesses = [0] * total_banks
            write_accesses = [0] * total_banks
            read_lanewidths = [0] * total_banks  # Track lanewidth for each bank's read accesses
            write_lanewidths = [0] * total_banks  # Track lanewidth for each bank's write accesses
            
            if self.debug:
                print(f"DEBUG: processing millisecond {ms}")
            
            # If distribute_across_group is enabled, ensure ALL banks in the group get some access
            if self.distribute_across_group and operation_timeline:
                # Get all banks in the group (assuming all operations are in the same group)
                all_banks_in_group = set()
                for op in operation_timeline:
                    all_banks_in_group.update(self.bank_mapping[op['bank_group']])
                
                if self.debug:
                    print(f"DEBUG:   distribute_across_group enabled, ensuring ALL {len(all_banks_in_group)} banks get access")
                
                # Calculate original total access amounts and preserve them
                original_total_read = 0
                original_total_write = 0
                original_total_accesses = 0
                
                for op in operation_timeline:
                    op_total_accesses = sum(op['bank_accesses'].values())
                    original_total_accesses += op_total_accesses
                    if op['type'] == 'read':
                        original_total_read += op_total_accesses
                    elif op['type'] == 'write':
                        original_total_write += op_total_accesses
                
                if original_total_accesses > 0:
                    # Calculate read/write ratios to preserve original proportions
                    read_ratio = original_total_read / original_total_accesses if original_total_accesses > 0 else 0.6
                    write_ratio = original_total_write / original_total_accesses if original_total_accesses > 0 else 0.4
                    
                    if self.debug:
                        print(f"DEBUG:   original_total_accesses = {original_total_accesses}")
                        print(f"DEBUG:   read_ratio = {read_ratio:.3f}, write_ratio = {write_ratio:.3f}")
                    
                    # Distribute access across ALL banks with noise while preserving total
                    import numpy as np
                    weights = np.random.normal(1.0, self.distribution_noise, len(all_banks_in_group))
                    weights = np.maximum(weights, 0.1)  # Ensure minimum weight
                    weights = weights / np.sum(weights)  # Normalize to sum to 1
                    
                    # Distribute original total across all banks while preserving total power
                    # Calculate adaptive scaling factor based on power characteristics
                    # We need to estimate how much power increases when distributing across all banks
                    # and scale down accordingly to preserve total power
                    
                    # Estimate power scaling: when distributing across N banks, power typically increases
                    # by a factor related to the number of active banks and their power characteristics
                    num_banks_in_group = len(all_banks_in_group)
                    
                    
                    # Calculate adaptive scaling factor using power law
                    scale_factor = 1.

                    if self.debug:
                        print(f"DEBUG:   adaptive scaling: {num_banks_in_group} banks -> scale_factor = {scale_factor}")
                    
                    for i, bank in enumerate(sorted(all_banks_in_group)):
                        # Calculate proportional access for this bank, scaled down to preserve total power
                        bank_total_accesses = int(original_total_accesses * weights[i] / scale_factor)
                        
                        if bank_total_accesses > 0:
                            # Preserve original read/write ratios
                            bank_read_accesses = int(bank_total_accesses * read_ratio)
                            bank_write_accesses = int(bank_total_accesses * write_ratio)
                            
                            if bank_read_accesses > 0:
                                read_accesses[bank] += bank_read_accesses/max_duration
                                read_lanewidths[bank] = global_max_lanewidth
                            if bank_write_accesses > 0:
                                write_accesses[bank] += bank_write_accesses/max_duration
                                write_lanewidths[bank] = global_max_lanewidth
                    
                    if self.debug:
                        actual_total = sum(read_accesses) + sum(write_accesses)
                        print(f"DEBUG:   distributed {original_total_accesses} accesses across {len(all_banks_in_group)} banks")
                        print(f"DEBUG:   actual total = {actual_total} (should be ~{original_total_accesses})")
                    
                    if self.debug:
                        print(f"DEBUG:   distributed {original_total_accesses} accesses across {len(all_banks_in_group)} banks")
                    
                    # Add the distributed access to the timeline
                    timeline.append((read_accesses, write_accesses, read_lanewidths, write_lanewidths))
                    continue  # Skip the normal operation processing
            
            # Normal operation processing (when distribute_across_group is disabled)
            for op in operation_timeline:
                banks_in_group = self.bank_mapping[op['bank_group']]
                total_accesses = sum(op['bank_accesses'].values())
                
                if self.debug:
                    print(f"DEBUG:   operation {op['type']}@{op['bank_group']}@{op['starting_stack']}: total_accesses={total_accesses}")
                
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
                        if op not in has_prs_1ms_op:
                            if self.debug:
                                print(f"DEBUG:     using CONCURRENT access (≤1ms)")
                            for bank in banks_in_group:
                                bank_access_count = op['bank_accesses'].get(bank, 0)
                                if bank_access_count > 0:
                                    if op['type'] == 'read':
                                        read_accesses[bank] += int(bank_access_count * op['lanewidth'] / global_max_lanewidth) #scale the access count by the lanewidth
                                        read_lanewidths[bank] = op['lanewidth']  # Set lanewidth for this bank
                                        if self.debug:
                                            print(f"DEBUG:       adding {int(bank_access_count)} read access to bank {bank}, current read_accesses = {read_accesses[bank]}")
                                            print(f"DEBUG:       op['lanewidth'] = {op['lanewidth']}")
                                            print(f"DEBUG:       global_max_lanewidth = {global_max_lanewidth}")
                                    elif op['type'] == 'write':
                                        write_accesses[bank] += int(bank_access_count * op['lanewidth'] / global_max_lanewidth) #scale the access count by the lanewidth
                                        write_lanewidths[bank] = op['lanewidth']  # Set lanewidth for this bank
                                        if self.debug:
                                            print(f"DEBUG:       adding {int(bank_access_count)} write access to bank {bank}, current write_accesses = {write_accesses[bank]}")
                            has_prs_1ms_op.append(op)
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
                                read_accesses[active_bank] += int(active_accesses * op['lanewidth'] / global_max_lanewidth) #scale the access count by the lanewidth
                                read_lanewidths[active_bank] = op['lanewidth']  # Set lanewidth for this bank
                                if self.debug:
                                    print(f"DEBUG:       op['lanewidth'] = {op['lanewidth']}")
                                    print(f"DEBUG:       global_max_lanewidth = {global_max_lanewidth}")
                                    print(f"DEBUG:       adding {int(active_accesses * op['lanewidth'] / global_max_lanewidth)} read accesses to bank {active_bank}, current read_accesses = {read_accesses[active_bank]}")
                            elif op['type'] == 'write':
                                write_accesses[active_bank] += int(active_accesses * op['lanewidth'] / global_max_lanewidth) #scale the access count by the lanewidth
                                write_lanewidths[active_bank] = op['lanewidth']  # Set lanewidth for this bank
                                if self.debug:
                                    print(f"DEBUG:       adding {int(active_accesses * op['lanewidth'] / global_max_lanewidth)} write accesses to bank {active_bank}, current write_accesses = {write_accesses[active_bank]}")
            
            if self.debug:
                print(f"DEBUG:   ms {ms} result: read={read_accesses}, write={write_accesses}")
                print(f"DEBUG:   ms {ms} lanewidths: read={read_lanewidths}, write={write_lanewidths}")
            timeline.append((read_accesses, write_accesses, read_lanewidths, write_lanewidths))
        
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
                    if self.debug:
                        print(f"DEBUG: parsing line: '{line}'")
                        print(f"DEBUG: parts = {parts}")
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
        headers = ['step'] + [f'read_{i}' for i in range(self.total_banks)] + [f'write_{i}' for i in range(self.total_banks)] + [f'read_lw_{i}' for i in range(self.total_banks)] + [f'write_lw_{i}' for i in range(self.total_banks)]
        # Gather all rows as strings
        rows = []
        for step_num, (read_accesses, write_accesses, read_lanewidths, write_lanewidths) in steps_data:
            row = [str(step_num)] + [str(int(x)) for x in read_accesses] + [str(int(x)) for x in write_accesses] + [str(int(x)) for x in read_lanewidths] + [str(int(x)) for x in write_lanewidths]
            rows.append(row)
        
        # Ensure all rows have the same number of columns as headers
        expected_cols = len(headers)
        for i, row in enumerate(rows):
            if len(row) != expected_cols:
                print(f"Warning: Row {i} has {len(row)} columns, expected {expected_cols}")
                # Pad or truncate row to match header length
                if len(row) < expected_cols:
                    row.extend(['0'] * (expected_cols - len(row)))
                else:
                    row = row[:expected_cols]
                rows[i] = row
        
        # Calculate max width for each column
        col_widths = []
        for i in range(len(headers)):
            max_width = len(headers[i])
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)
        
        # Write to file
        with open(filename, 'w') as f:
            # Write header
            f.write('  '.join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + '\n')
            # Write rows
            for row in rows:
                f.write('  '.join(row[i].rjust(col_widths[i]) for i in range(len(row))) + '\n')

    def preprocess_description_file(self, input_file, output_file, split_on_total_capacity=True, split_on_stack_capacity=True):
        """
        Preprocess description file by splitting operations that exceed capacity.
        
        Args:
            input_file (str): Path to input description file
            output_file (str): Path to output preprocessed description file
            split_on_total_capacity (bool): Split when access exceeds total bank group capacity
            split_on_stack_capacity (bool): Split when access exceeds single stack capacity
        """
        if self.debug:
            print(f"DEBUG: Preprocessing description file")
            print(f"DEBUG:   input_file = {input_file}")
            print(f"DEBUG:   output_file = {output_file}")
            print(f"DEBUG:   split_on_total_capacity = {split_on_total_capacity}")
            print(f"DEBUG:   split_on_stack_capacity = {split_on_stack_capacity}")
        
        # Calculate capacities
        total_bank_group_capacity = self.bank_size * len(self.bank_mapping[0])  # Assuming all groups have same capacity
        banks_per_stack = self.total_banks // self.stack_height
        single_stack_capacity = self.bank_size * (len(self.bank_mapping[0]) // self.stack_height)
        
        if self.debug:
            print(f"DEBUG:   total_bank_group_capacity = {total_bank_group_capacity} MB")
            print(f"DEBUG:   single_stack_capacity = {single_stack_capacity} MB")
        
        # Parse input file
        steps = self.parse_input_file(input_file)
        
        # Process each step
        processed_steps = []
        current_step_num = 0
        
        for step_num, operations in steps:
            if self.debug:
                print(f"DEBUG: Processing step {step_num} with {len(operations)} operations")
            
            # Process operations in this step
            processed_operations = []
            extra_steps = []  # Operations that need to be moved to next step
            
            for operation in operations:
                parts = operation.split('@')
                if len(parts) < 5:
                    processed_operations.append(operation)
                    continue
                
                # Handle both old format (5 parts) and new format (6 parts)
                if len(parts) == 5:
                    # Old format: read@bank_group@starting_stack@access_amount@lanewidth
                    access_type, bank_group_str, starting_stack_str, access_amount_str, lanewidth_str = parts
                    starting_bank_str = "0"  # Default to first bank in group
                elif len(parts) == 6:
                    # New format: read@bank_group@starting_stack@starting_bank@access_amount@lanewidth
                    access_type, bank_group_str, starting_stack_str, starting_bank_str, access_amount_str, lanewidth_str = parts
                else:
                    processed_operations.append(operation)
                    continue
                
                try:
                    bank_group = int(bank_group_str)
                    starting_stack = int(starting_stack_str)
                    starting_bank = int(starting_bank_str)
                    access_amount = float(access_amount_str)
                    lanewidth = float(lanewidth_str)
                except ValueError:
                    processed_operations.append(operation)
                    continue
                
                # Check if splitting is needed
                should_split = False
                split_reason = ""
                
                if split_on_total_capacity and access_amount > total_bank_group_capacity:
                    should_split = True
                    split_reason = f"exceeds total capacity ({access_amount} > {total_bank_group_capacity})"
                
                if split_on_stack_capacity and access_amount > single_stack_capacity:
                    should_split = True
                    split_reason = f"exceeds stack capacity ({access_amount} > {single_stack_capacity})"
                
                if should_split and self.debug:
                    print(f"DEBUG:   Operation {operation} needs splitting: {split_reason}")
                    print(f"DEBUG:     Original: {access_type}@{bank_group}@{starting_stack}@{starting_bank}@{access_amount}@{lanewidth}")
                
                if should_split:
                    # Calculate how many steps we need
                    if split_on_total_capacity and access_amount > total_bank_group_capacity:
                        num_steps_needed = int(access_amount / total_bank_group_capacity) + 1
                        access_per_step = access_amount / num_steps_needed
                    else:
                        num_steps_needed = int(access_amount / single_stack_capacity) + 1
                        access_per_step = access_amount / num_steps_needed
                    
                    if self.debug:
                        print(f"DEBUG:     Splitting into {num_steps_needed} steps, {access_per_step:.2f} MB per step")
                    
                    # First operation goes to current step
                    first_operation = f"{access_type}@{bank_group}@{starting_stack}@{starting_bank}@{access_per_step:.2f}@{lanewidth}"
                    processed_operations.append(first_operation)
                    
                    # Remaining operations go to extra steps
                    for i in range(1, num_steps_needed):
                        extra_operation = f"{access_type}@{bank_group}@{starting_stack}@{starting_bank}@{access_per_step:.2f}@{lanewidth}"
                        extra_steps.append(extra_operation)
                else:
                    processed_operations.append(operation)
            
            # Add current step
            if processed_operations:
                processed_steps.append((current_step_num, processed_operations))
                current_step_num += 1
            
            # Add extra steps before the next original step
            for extra_operation in extra_steps:
                processed_steps.append((current_step_num, [extra_operation]))
                current_step_num += 1
        
        # Write preprocessed file
        with open(output_file, 'w') as f:
            for step_num, operations in processed_steps:
                if operations:
                    # Write first operation with step number
                    f.write(f"{step_num}\t{operations[0]}\n")
                    # Write remaining operations without step number
                    for operation in operations[1:]:
                        f.write(f"\t{operation}\n")
        
        if self.debug:
            print(f"DEBUG: Preprocessing completed")
            print(f"DEBUG:   Original steps: {len(steps)}")
            print(f"DEBUG:   Processed steps: {len(processed_steps)}")
        
        return processed_steps

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
        
        # First pass: collect all lanewidths to find global maximum
        all_lanewidths = []
        for step_num, operations in steps:
            for operation in operations:
                parts = operation.split('@')
                if len(parts) >= 5:
                    try:
                        # Handle both old format (5 parts) and new format (6 parts)
                        if len(parts) == 5:
                            # Old format: read@bank_group@starting_stack@access_amount@lanewidth
                            lanewidth = float(parts[4])
                        elif len(parts) == 6:
                            # New format: read@bank_group@starting_stack@starting_bank@access_amount@lanewidth
                            lanewidth = float(parts[5])
                        else:
                            continue
                        all_lanewidths.append(lanewidth)
                    except ValueError:
                        continue
        
        global_max_lanewidth = max(all_lanewidths) if all_lanewidths else 1
        if self.debug:
            print(f"DEBUG: global_max_lanewidth = {global_max_lanewidth} B")
        
        # Process each step
        steps_data = []
        current_ms = 0
        
        for step_num, operations in steps:
            if self.debug:
                print(f"DEBUG: processing step {step_num} with {len(operations)} operations")
            # Use the new function to process operations and generate timeline
            timeline = self._process_operations_to_timeline(operations, global_max_lanewidth)
            
            # Convert timeline to steps_data format
            for ms, (read_accesses, write_accesses, read_lanewidths, write_lanewidths) in enumerate(timeline):
                steps_data.append((current_ms + ms, (read_accesses, write_accesses, read_lanewidths, write_lanewidths)))
            
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
    
    # Preprocessing options
    parser.add_argument('--preprocess', action='store_true', help='Preprocess description file to split large operations')
    parser.add_argument('--preprocessed-output', type=str, help='Output file for preprocessed description (default: input_file.preprocessed)')
    parser.add_argument('--split-on-total-capacity', action='store_true', default=True, help='Split operations that exceed total bank group capacity')
    parser.add_argument('--split-on-stack-capacity', action='store_true', default=True, help='Split operations that exceed single stack capacity')
    parser.add_argument('--no-split-on-total-capacity', action='store_true', help='Disable splitting on total capacity')
    parser.add_argument('--no-split-on-stack-capacity', action='store_true', help='Disable splitting on stack capacity')
    
    # New arguments for distribution
    parser.add_argument('--distribute-across-group', action='store_true', help='Distribute access across all banks in group with noise')
    parser.add_argument('--distribution-noise', type=float, default=0.2, help='Noise level for distribution (0.0 to 1.0, default 0.2)')
    
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
        print(f"DEBUG:   distribute_across_group = {args.distribute_across_group}")
        print(f"DEBUG:   distribution_noise = {args.distribution_noise}")
    
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
        debug=args.debug,
        distribute_across_group=args.distribute_across_group,
        distribution_noise=args.distribution_noise
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
        
        if args.debug:
            print(f"DEBUG: Preprocessing enabled")
            print(f"DEBUG:   preprocessed_file = {preprocessed_file}")
            print(f"DEBUG:   split_on_total = {split_on_total}")
            print(f"DEBUG:   split_on_stack = {split_on_stack}")
        
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
            sys.exit(1)
    
    # Convert the file
    try:
        converter.convert_file(input_file, args.output_file)
        print(f"Successfully converted {input_file} to {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


