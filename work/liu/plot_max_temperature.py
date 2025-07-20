#!/usr/bin/env python3
"""
Temperature Analysis Script
Analyzes temperature.trace files from multiple folders and plots maximum temperature trends.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def read_temperature_file(file_path):
    """
    Read temperature.trace file and return max temperature for each time step.
    
    Args:
        file_path (str): Path to temperature.trace file
        
    Returns:
        tuple: (time_points, max_temperatures, folder_name)
    """
    try:
        # Read the file with tab separator
        df = pd.read_csv(file_path, sep='\t')
        
        # Get folder name for legend
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        # Exclude columns with names starting with "LC_"
        columns_to_exclude = [col for col in df.columns if col.startswith('LC_')]
        df_filtered = df.drop(columns=columns_to_exclude)
        
        print(f"  Excluded {len(columns_to_exclude)} LC_* columns from {folder_name}")
        print(f"  Remaining columns: {len(df_filtered.columns)}")
        
        # Calculate max temperature for each row (excluding LC_* columns)
        max_temps = df_filtered.max(axis=1)
        
        # Time points in milliseconds (1ms per row)
        time_points = np.arange(len(max_temps))  # 0, 1, 2, ... ms
        
        return time_points, max_temps.values, folder_name
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

def plot_max_temperature_comparison(folders, output_file="max_temperature_comparison.png", show_peaks=True):
    """
    Plot maximum temperature trends from multiple folders.
    
    Args:
        folders (list): List of folder paths containing temperature.trace files
        output_file (str): Output filename for the plot
        show_peaks (bool): Whether to show peak temperature annotations
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    color_idx = 0
    
    # Store all data for peak detection
    all_data = []
    
    for folder in folders:
        temp_file = os.path.join(folder, "temperature.trace")
        
        if not os.path.exists(temp_file):
            print(f"Warning: {temp_file} not found, skipping...")
            continue
            
        time_points, max_temps, folder_name = read_temperature_file(temp_file)
        
        if time_points is not None:
            # Store data for peak detection
            all_data.append({
                'time_points': time_points,
                'max_temps': max_temps,
                'folder_name': folder_name,
                'color': colors[color_idx % len(colors)]
            })
            
            # Plot with different color for each folder
            color = colors[color_idx % len(colors)]
            plt.plot(time_points, max_temps, 
                    label=folder_name, 
                    color=color, 
                    linewidth=2,
                    alpha=0.9)
            
            # Print statistics
            print(f"\n{folder_name}:")
            print(f"  Max temperature: {max_temps.max():.2f}°C")
            print(f"  Min temperature: {max_temps.min():.2f}°C")
            print(f"  Average max temperature: {max_temps.mean():.2f}°C")
            print(f"  Duration: {len(time_points)} ms")
            
            color_idx += 1
    
    # Find global maximum and minimum temperatures
    if all_data:
        global_max = max(data['max_temps'].max() for data in all_data)
        global_min = min(data['max_temps'].min() for data in all_data)
        
        print(f"\nGlobal Temperature Statistics:")
        print(f"  Global max temperature: {global_max:.2f}°C")
        print(f"  Global min temperature: {global_min:.2f}°C")
        
        # Handle peak detection or global max/min marking
        if show_peaks:
            # Define significant temperature threshold (e.g., 90% of global max)
            temp_threshold = global_min + 0.3 * (global_max - global_min)
            
            print(f"  Temperature threshold for peaks: {temp_threshold:.2f}°C")
            
            # Find peaks for each dataset
            for data in all_data:
                time_points = data['time_points']
                max_temps = data['max_temps']
                folder_name = data['folder_name']
                color = data['color']
                
                # Find peaks using scipy if available, otherwise use simple method
                try:
                    from scipy.signal import find_peaks
                    # Find peaks above threshold
                    peaks, _ = find_peaks(max_temps, height=temp_threshold, distance=10)
                except ImportError:
                    # Simple peak detection
                    peaks = []
                    for i in range(1, len(max_temps) - 1):
                        if (max_temps[i] > temp_threshold and 
                            max_temps[i] > max_temps[i-1] and 
                            max_temps[i] > max_temps[i+1]):
                            peaks.append(i)
                
                # Store peak data for annotation after plot setup
                for peak_idx in peaks:
                    peak_time = time_points[peak_idx]
                    peak_temp = max_temps[peak_idx]
                    print(f"  {folder_name}: Peak at {peak_time}ms, Temperature: {peak_temp:.2f}°C")
        
        else:
            # Mark global max and min when peaks are disabled
            print(f"\nMarking Global Max and Min:")
            # Note: Actual annotation will be done after plot setup for proper bounds checking
    
    plt.xlabel('Time (ms)', fontsize=14, weight='bold')
    plt.ylabel('Maximum Temperature (°C)', fontsize=14, weight='bold')
    if show_peaks:
        plt.title(f'Maximum Temperature Comparison\n(Peaks marked)\n{output_file}', fontsize=16, weight='bold', pad=20)
    else:
        plt.title(f'Maximum Temperature Comparison\n{output_file}', fontsize=16, weight='bold', pad=20)
    plt.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Set reasonable y-axis limits and improve tick formatting
    #plt.ylim(bottom=45, top=65)  # Start from 40°C for better visibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add global max/min annotations after plot setup (for proper bounds checking)
    if all_data and not show_peaks:
        # Find which dataset contains the global max and min
        global_max_dataset = None
        global_min_dataset = None
        global_max_time = 0
        global_min_time = 0
        
        for data in all_data:
            max_idx = np.argmax(data['max_temps'])
            min_idx = np.argmin(data['max_temps'])
            
            if data['max_temps'][max_idx] == global_max:
                global_max_dataset = data
                global_max_time = data['time_points'][max_idx]
            
            if data['max_temps'][min_idx] == global_min:
                global_min_dataset = data
                global_min_time = data['time_points'][min_idx]
        
        # Mark global maximum
        if global_max_dataset:
            # Calculate annotation position with bounds checking
            x_pos = global_max_time
            y_pos = global_max
            
            # Get current axis limits after plot setup
            xlim = plt.xlim()
            ylim = plt.ylim()
            
            # Adjust text position to stay within bounds
            text_x = x_pos + (xlim[1] - xlim[0]) * 0.05  # 5% of x range
            text_y = y_pos + (ylim[1] - ylim[0]) * 0.05  # 5% of y range
            
            # Adjust if text would go outside bounds
            if text_x > xlim[1]:
                text_x = x_pos - (xlim[1] - xlim[0]) * 0.05
            if text_y > ylim[1]:
                text_y = y_pos - (ylim[1] - ylim[0]) * 0.05
            
            plt.annotate(f'Global Max: {global_max:.2f}°C', 
                       xy=(x_pos, y_pos),
                       xytext=(text_x, text_y),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=2),
                       fontsize=12, color='red', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='red'))
            plt.plot(x_pos, y_pos, 'o', color='red', markersize=10, markeredgecolor='black', markeredgewidth=2)
            print(f"  Global Max: {global_max:.2f}°C at {global_max_time}ms ({global_max_dataset['folder_name']})")
        
        # Mark global minimum
        if global_min_dataset:
            # Calculate annotation position with bounds checking
            x_pos = global_min_time
            y_pos = global_min
            
            # Get current axis limits after plot setup
            xlim = plt.xlim()
            ylim = plt.ylim()
            
            # Adjust text position to stay within bounds
            text_x = x_pos + (xlim[1] - xlim[0]) * 0.05  # 5% of x range
            text_y = y_pos - (ylim[1] - ylim[0]) * 0.05  # 5% of y range
            
            # Adjust if text would go outside bounds
            if text_x > xlim[1]:
                text_x = x_pos - (xlim[1] - xlim[0]) * 0.05
            if text_y < ylim[0]:
                text_y = y_pos + (ylim[1] - ylim[0]) * 0.05
            
            plt.annotate(f'Global Min: {global_min:.2f}°C', 
                       xy=(x_pos, y_pos),
                       xytext=(text_x, text_y),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.8, lw=2),
                       fontsize=12, color='blue', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='blue'))
            plt.plot(x_pos, y_pos, 'o', color='blue', markersize=10, markeredgecolor='black', markeredgewidth=2)
            print(f"  Global Min: {global_min:.2f}°C at {global_min_time}ms ({global_min_dataset['folder_name']})")
    
    # Add peak annotations after plot setup (for proper bounds checking)
    if all_data and show_peaks:
        # Define significant temperature threshold (e.g., 90% of global max)
        temp_threshold = global_min + 0.3 * (global_max - global_min)
        
        # Find and annotate peaks for each dataset
        for data in all_data:
            time_points = data['time_points']
            max_temps = data['max_temps']
            folder_name = data['folder_name']
            color = data['color']
            
            # Find peaks using scipy if available, otherwise use simple method
            try:
                from scipy.signal import find_peaks
                # Find peaks above threshold
                peaks, _ = find_peaks(max_temps, height=temp_threshold, distance=10)
            except ImportError:
                # Simple peak detection
                peaks = []
                for i in range(1, len(max_temps) - 1):
                    if (max_temps[i] > temp_threshold and 
                        max_temps[i] > max_temps[i-1] and 
                        max_temps[i] > max_temps[i+1]):
                        peaks.append(i)
            
            # Annotate significant peaks
            for peak_idx in peaks:
                peak_time = time_points[peak_idx]
                peak_temp = max_temps[peak_idx]
                
                # Calculate annotation position with bounds checking
                x_pos = peak_time
                y_pos = peak_temp
                
                # Get current axis limits after plot setup
                xlim = plt.xlim()
                ylim = plt.ylim()
                
                # Adjust text position to stay within bounds (use relative positioning)
                text_x = x_pos + (xlim[1] - xlim[0]) * 0.03  # 3% of x range
                text_y = y_pos + (ylim[1] - ylim[0]) * 0.03  # 3% of y range
                
                # Adjust if text would go outside bounds
                if text_x > xlim[1]:
                    text_x = x_pos - (xlim[1] - xlim[0]) * 0.03
                if text_y > ylim[1]:
                    text_y = y_pos - (ylim[1] - ylim[0]) * 0.03
                
                # Add annotation with temperature value
                plt.annotate(f'{peak_temp:.2f}°C', 
                           xy=(x_pos, y_pos),
                           xytext=(text_x, text_y),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=1.5),
                           fontsize=10, color=color, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=color))
                
                # Add a marker at the peak
                plt.plot(x_pos, y_pos, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    
    # Tight layout with minimal padding
    plt.tight_layout(pad=1.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot maximum temperature trends from multiple folders')
    parser.add_argument('folders', nargs='+', help='Folders containing temperature.trace files')
    parser.add_argument('--output', '-o', default='max_temperature_comparison.png', 
                       help='Output filename for the plot (default: max_temperature_comparison.png)')
    parser.add_argument('--no-peaks', action='store_true', 
                       help='Disable peak temperature annotations on the plot')
    
    args = parser.parse_args()
    
    # Validate folders
    valid_folders = []
    for folder in args.folders:
        if os.path.exists(folder):
            temp_file = os.path.join(folder, "temperature.trace")
            if os.path.exists(temp_file):
                valid_folders.append(folder)
            else:
                print(f"Warning: {temp_file} not found in {folder}")
        else:
            print(f"Warning: Folder {folder} does not exist")
    
    if not valid_folders:
        print("Error: No valid folders with temperature.trace files found!")
        return
    
    print(f"Analyzing temperature files from {len(valid_folders)} folders:")
    for folder in valid_folders:
        print(f"  - {folder}")
    
    # Determine whether to show peaks based on command line argument
    show_peaks = not args.no_peaks
    print(f"Peak annotations: {'Enabled' if show_peaks else 'Disabled'}")
    
    plot_max_temperature_comparison(valid_folders, args.output, show_peaks)

if __name__ == "__main__":
    # If no command line arguments, use the folders mentioned in the user query
    import sys
    if len(sys.argv) == 1:
        # Default folders from the user's query
        folders = [
            "output_rotate_speedup3",
            "output_baseline1", 
            "output_baseline2"
        ]
        
        print("Using default folders:")
        for folder in folders:
            print(f"  - {folder}")
        
        plot_max_temperature_comparison(folders, show_peaks=True)
    else:
        main() 