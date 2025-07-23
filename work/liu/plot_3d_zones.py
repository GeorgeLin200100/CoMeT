#!/usr/bin/env python3
"""
3D Plot with Temperature Zones

This script creates a 3D visualization with:
- X-axis: Memory footprint (GB)
- Y-axis: Throughput (GBps) 
- Z-axis: Temperature (C)

Color-coded zones:
- Green: Temperature < 85°C
- Yellow: Temperature 85-95°C
- Red: Temperature > 95°C

New feature: --mountain-3d creates a 3D mountain plot where:
- X-axis: Memory footprint (GB)
- Y-axis: Throughput (GBps)
- Z-axis: Temperature (C) - height of the mountain
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import argparse
import os
import sys
import datetime
import subprocess

def get_figure_info():
    """
    Get command and timestamp information for figure annotation.
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get command line arguments
    cmd_args = sys.argv
    if len(cmd_args) > 0:
        # Get just the script name and arguments, not the full path
        script_name = os.path.basename(cmd_args[0])
        args = cmd_args[1:]
        command = f"{script_name} {' '.join(args)}"
    else:
        command = "plot_3d_zones.py"
    
    return command, timestamp

def create_3d_zones_plot(memory_range=(0, 32), throughput_range=(0, 100),
                         temp_resolution=50, output_file=None, show_plot=True):
    """
    Create a 3D plot with temperature zones.
    
    Args:
        memory_range: Tuple of (min, max) memory footprint in GB
        throughput_range: Tuple of (min, max) throughput in GBps
        temp_resolution: Number of temperature points to sample
        output_file: Optional file to save the plot
        show_plot: Whether to display the plot
    """
    
    # Create coordinate grids
    memory_min, memory_max = memory_range
    throughput_min, throughput_max = throughput_range
    
    # Create 2D grids for memory and throughput
    memory = np.linspace(memory_min, memory_max, 50)
    throughput = np.linspace(throughput_min, throughput_max, 50)
    memory_grid, throughput_grid = np.meshgrid(memory, throughput)
    
    # Create flat temperature zones based on memory and throughput
    # This creates clear boundaries without mountain-like elevation
    temp_grid = np.zeros_like(memory_grid)
    
    # Calculate temperature based on memory and throughput
    # Memory has stronger effect on temperature
    memory_factor = (memory_grid / memory_max) * 30  # 0-30°C range
    throughput_factor = (throughput_grid / throughput_max) * 20  # 0-20°C range
    
    # Base temperature of 70°C
    temp_grid = 70 + memory_factor + throughput_factor
    
    # Add some realistic variations but keep it mostly flat
    temp_grid += np.random.normal(0, 0.5, memory_grid.shape)
    
    # Ensure temperature is within reasonable bounds
    temp_grid = np.clip(temp_grid, 65, 105)
    
    # Create clearer zone separation with distinct colors
    colors = np.zeros_like(temp_grid)
    
    # Green zone: temp < 85°C
    green_mask = temp_grid < 85
    colors[green_mask] = 0  # Green
    
    # Yellow zone: 85°C <= temp < 95°C
    yellow_mask = (temp_grid >= 85) & (temp_grid < 95)
    colors[yellow_mask] = 1  # Yellow
    
    # Red zone: temp >= 95°C
    red_mask = temp_grid >= 95
    colors[red_mask] = 2  # Red
    
    # Create custom colormap with more distinct colors
    zone_colors = ['#2E8B57', '#FFD700', '#DC143C']  # Sea Green, Gold, Crimson
    cmap = ListedColormap(zone_colors)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with temperature-based coloring (mostly flat)
    surf = ax.plot_surface(memory_grid, throughput_grid, temp_grid, 
                          facecolors=cmap(colors), alpha=0.9, 
                          edgecolor='black', linewidth=0.2)
    
    # Add zone boundary planes with better visibility
    # Green-Yellow boundary at 85°C
    boundary_85 = ax.plot_surface(memory_grid, throughput_grid, 
                                 np.full_like(temp_grid, 85), 
                                 alpha=0.4, color='#FFD700', linewidth=0)
    
    # Yellow-Red boundary at 95°C
    boundary_95 = ax.plot_surface(memory_grid, throughput_grid, 
                                 np.full_like(temp_grid, 95), 
                                 alpha=0.4, color='#DC143C', linewidth=0)
    
    # Add zone labels on the boundary planes
    # Green zone label
    ax.text(memory_max * 0.3, throughput_max * 0.3, 85, 
            'SAFE ZONE\n(< 85°C)', fontsize=12, fontweight='bold', 
            color='#2E8B57', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Yellow zone label
    ax.text(memory_max * 0.7, throughput_max * 0.7, 90, 
            'WARNING ZONE\n(85-95°C)', fontsize=12, fontweight='bold', 
            color='#FFD700', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Red zone label
    ax.text(memory_max * 0.8, throughput_max * 0.8, 100, 
            'CRITICAL ZONE\n(> 95°C)', fontsize=12, fontweight='bold', 
            color='#DC143C', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Memory Footprint (GB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (GBps)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax.set_title('3D Memory-Throughput-Temperature Zones\n(Flat Visualization)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(memory_min, memory_max)
    ax.set_ylim(throughput_min, throughput_max)
    ax.set_zlim(65, 105)
    
    # Add enhanced legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', alpha=0.9, label='Safe Zone (< 85°C)'),
        Patch(facecolor='#FFD700', alpha=0.9, label='Warning Zone (85-95°C)'),
        Patch(facecolor='#DC143C', alpha=0.9, label='Critical Zone (> 95°C)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.0),
             fontsize=12, framealpha=0.9)
    
    # Add grid with better visibility
    ax.grid(True, alpha=0.4, linewidth=0.5)
    
    # Rotate the plot for better viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add enhanced statistics
    green_count = np.sum(green_mask)
    yellow_count = np.sum(yellow_mask)
    red_count = np.sum(red_mask)
    total_points = green_count + yellow_count + red_count
    
    stats_text = f'Zone Distribution:\n\nGreen (Safe): {green_count/total_points*100:.1f}%\nYellow (Warning): {yellow_count/total_points*100:.1f}%\nRed (Critical): {red_count/total_points*100:.1f}%'
    ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes, 
              fontsize=11, verticalalignment='top', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Add temperature range info
    temp_range_text = f'Temperature Range:\n{temp_grid.min():.1f}°C - {temp_grid.max():.1f}°C'
    ax.text2D(0.02, 0.75, temp_range_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    # Add figure information (command and timestamp)
    command, timestamp = get_figure_info()
    info_text = f'Command: {command}\nTime: {timestamp}'
    ax.text2D(0.98, 0.02, info_text, transform=ax.transAxes, 
              fontsize=8, verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig, ax

def create_2d_zones_plot(memory_range=(0, 32), throughput_range=(0, 100), 
                         output_file=None, show_plot=True, data_file=None):
    """
    Create a 2D flat plot with temperature zones.
    
    Args:
        memory_range: Tuple of (min, max) memory footprint in GB
        throughput_range: Tuple of (min, max) throughput in GBps
        output_file: Optional file to save the plot
        show_plot: Whether to display the plot
        data_file: Optional CSV file to calibrate temperature model
    """
    
    # Create coordinate grids
    memory_min, memory_max = memory_range
    throughput_min, throughput_max = throughput_range
    
    # Create 2D grids for memory and throughput
    memory = np.linspace(memory_min, memory_max, 200)  # Higher resolution for smoother boundaries
    throughput = np.linspace(throughput_min, throughput_max, 200)
    memory_grid, throughput_grid = np.meshgrid(memory, throughput)
    
    # Calculate temperature based on actual data if available
    if data_file and os.path.exists(data_file):
        try:
            import pandas as pd
            data = pd.read_csv(data_file)
            
            # Simple manual fitting to the actual data
            # Calculate coefficients based on the data range
            memory_coeff = (data['temperature_c'].max() - data['temperature_c'].min()) / (data['memory_gb'].max() - data['memory_gb'].min())
            throughput_coeff = (data['temperature_c'].max() - data['temperature_c'].min()) / (data['throughput_gbps'].max() - data['throughput_gbps'].min()) * 0.5
            
            # Calculate intercept to match the data
            base_temp = data['temperature_c'].min() - (memory_coeff * data['memory_gb'].min() + throughput_coeff * data['throughput_gbps'].min())
            
            # Use the fitted model to calculate temperatures
            temp_grid = base_temp + memory_coeff * memory_grid + throughput_coeff * throughput_grid
            
            print(f"Temperature model fitted from data:")
            print(f"  Memory coefficient: {memory_coeff:.3f}")
            print(f"  Throughput coefficient: {throughput_coeff:.3f}")
            print(f"  Base temperature: {base_temp:.3f}")
            print(f"  Data temperature range: {data['temperature_c'].min():.1f}°C - {data['temperature_c'].max():.1f}°C")
            
        except Exception as e:
            print(f"Warning: Could not fit model from data: {e}")
            print("Using default temperature model...")
            # Fallback to default model
            memory_factor = (memory_grid / memory_max) * 30
            throughput_factor = (throughput_grid / throughput_max) * 20
            temp_grid = 70 + memory_factor + throughput_factor
    else:
        # Use default temperature model
        memory_factor = (memory_grid / memory_max) * 30
        throughput_factor = (throughput_grid / throughput_max) * 20
        temp_grid = 70 + memory_factor + throughput_factor
    
    # Create zone mask based on actual temperature values
    zone_mask = np.zeros_like(temp_grid, dtype=int)
    zone_mask[temp_grid < 85] = 0  # Green
    zone_mask[(temp_grid >= 85) & (temp_grid < 95)] = 1  # Yellow
    zone_mask[temp_grid >= 95] = 2  # Red
    
    # Create the 2D plot with elegant styling
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Set elegant background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Create custom colormap with more elegant colors
    zone_colors = ['#28a745', '#ffc107', '#dc3545']  # Bootstrap-style colors
    cmap = ListedColormap(zone_colors)
    
    # Plot the zones using imshow with higher resolution and transparency
    im = ax.imshow(zone_mask, extent=[memory_min, memory_max, throughput_min, throughput_max], 
                   origin='lower', cmap=cmap, alpha=0.6, aspect='auto', interpolation='gaussian')
    
    # Add smooth contour lines for zone boundaries with elegant styling
    try:
        # Smooth the temperature grid for better contour lines
        from scipy.ndimage import gaussian_filter
        smooth_temp = gaussian_filter(temp_grid, sigma=1.0)
        
        # Add contour lines with elegant styling
        contour1 = ax.contour(memory_grid, throughput_grid, smooth_temp, 
                             levels=[85], colors=['#ffc107'], 
                             linewidths=2.5, linestyles='-', alpha=0.8)
        contour2 = ax.contour(memory_grid, throughput_grid, smooth_temp, 
                             levels=[95], colors=['#dc3545'], 
                             linewidths=2.5, linestyles='-', alpha=0.8)
        
        # Add contour labels
        ax.clabel(contour1, inline=True, fontsize=10, fmt='85°C')
        ax.clabel(contour2, inline=True, fontsize=10, fmt='95°C')
    except:
        # Fallback if contour fails
        pass
    
    # Add elegant zone labels with better positioning
    ax.text(memory_max * 0.25, throughput_max * 0.25, 'SAFE ZONE\n(< 85°C)', 
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, 
                     edgecolor='#28a745', linewidth=2),
            color='#28a745')
    
    ax.text(memory_max * 0.6, throughput_max * 0.6, 'WARNING ZONE\n(85-95°C)', 
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, 
                     edgecolor='#ffc107', linewidth=2),
            color='#ffc107')
    
    ax.text(memory_max * 0.8, throughput_max * 0.8, 'CRITICAL ZONE\n(> 95°C)', 
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, 
                     edgecolor='#dc3545', linewidth=2),
            color='#dc3545')
    
    # Set elegant labels and title
    ax.set_xlabel('Memory Footprint (GB)', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Throughput (GBps)', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_title('Memory-Throughput Temperature Zones\n(Elegant 2D Visualization)', 
                fontsize=18, fontweight='bold', pad=25, color='#2c3e50')
    
    # Add elegant legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#28a745', alpha=0.7, label='Safe Zone (< 85°C)'),
        Patch(facecolor='#ffc107', alpha=0.7, label='Warning Zone (85-95°C)'),
        Patch(facecolor='#dc3545', alpha=0.7, label='Critical Zone (> 95°C)')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                      framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#dee2e6')
    
    # Add elegant grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#6c757d')
    ax.set_axisbelow(True)  # Put grid behind data
    
    # Add elegant statistics panel
    green_count = np.sum(zone_mask == 0)
    yellow_count = np.sum(zone_mask == 1)
    red_count = np.sum(zone_mask == 2)
    total_points = green_count + yellow_count + red_count
    
    stats_text = f'Zone Distribution:\n\n● Safe: {green_count/total_points*100:.1f}%\n● Warning: {yellow_count/total_points*100:.1f}%\n● Critical: {red_count/total_points*100:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, 
                     edgecolor='#dee2e6', linewidth=1.5))
    
    # Add temperature range info with elegant styling
    temp_range_text = f'Temperature Range:\n{temp_grid.min():.1f}°C - {temp_grid.max():.1f}°C'
    ax.text(0.02, 0.85, temp_range_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e3f2fd', alpha=0.9, 
                     edgecolor='#2196f3', linewidth=1.5))
    
    # Add figure information (command and timestamp)
    command, timestamp = get_figure_info()
    info_text = f'Command: {command}\nTime: {timestamp}'
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_color('#dee2e6')
        spine.set_linewidth(1.5)
    
    # Tight layout with more padding
    plt.tight_layout(pad=2.0)
    
    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Elegant plot saved to: {output_file}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig, ax

def plot_data_points(data_file, fig=None, ax=None, output_file=None, show_plot=True):
    """
    Plot actual data points on the 3D zones plot.
    
    Args:
        data_file: CSV file with columns: memory_gb, throughput_gbps, temperature_c
        fig: Existing figure (optional)
        ax: Existing axis (optional)
        output_file: Optional file to save the plot
        show_plot: Whether to display the plot
    """
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return None, None
    
    try:
        import pandas as pd
        data = pd.read_csv(data_file)
        
        # Check required columns
        required_cols = ['memory_gb', 'throughput_gbps', 'temperature_c']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(data.columns)}")
            return None, None
        
        # Create plot if not provided
        if fig is None or ax is None:
            fig, ax = create_3d_zones_plot(show_plot=False)
        
        # Plot data points with color coding
        for _, row in data.iterrows():
            memory = row['memory_gb']
            throughput = row['throughput_gbps']
            temp = row['temperature_c']
            
            # Determine color based on temperature with enhanced colors
            if temp < 85:
                color = '#2E8B57'  # Sea Green
                marker = 'o'
                size = 120
            elif temp < 95:
                color = '#FFD700'  # Gold
                marker = 's'
                size = 120
            else:
                color = '#DC143C'  # Crimson
                marker = '^'
                size = 120
            
            # Plot the point
            ax.scatter(memory, throughput, temp, 
                      c=color, marker=marker, s=size, 
                      edgecolor='black', linewidth=2, alpha=0.9)
        
        # Update title
        ax.set_title('3D Memory-Throughput-Temperature Zones with Data Points\n(Enhanced Visualization)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Save plot if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot with data points saved to: {output_file}")
        
        # Show plot
        if show_plot:
            plt.show()
        
        return fig, ax
        
    except ImportError:
        print("Error: pandas is required for reading CSV files. Install with: pip install pandas")
        return None, None
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None, None

def plot_2d_data_points(data_file, fig=None, ax=None, output_file=None, show_plot=True):
    """
    Plot actual data points on the 2D zones plot.
    
    Args:
        data_file: CSV file with columns: memory_gb, throughput_gbps, temperature_c
        fig: Existing figure (optional)
        ax: Existing axis (optional)
        output_file: Optional file to save the plot
        show_plot: Whether to display the plot
    """
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return None, None
    
    try:
        import pandas as pd
        data = pd.read_csv(data_file)
        
        # Check required columns
        required_cols = ['memory_gb', 'throughput_gbps', 'temperature_c']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(data.columns)}")
            return None, None
        
        # Create plot if not provided
        if fig is None or ax is None:
            fig, ax = create_2d_zones_plot(show_plot=False)
        
        # Plot data points with elegant color coding
        for _, row in data.iterrows():
            memory = row['memory_gb']
            throughput = row['throughput_gbps']
            temp = row['temperature_c']
            
            # Determine color based on temperature with elegant colors
            if temp < 85:
                color = '#28a745'  # Bootstrap Green
                marker = 'o'
                size = 180
                edge_color = '#1e7e34'
            elif temp < 95:
                color = '#ffc107'  # Bootstrap Yellow
                marker = 's'
                size = 180
                edge_color = '#e0a800'
            else:
                color = '#dc3545'  # Bootstrap Red
                marker = '^'
                size = 180
                edge_color = '#c82333'
            
            # Plot the point with elegant styling
            ax.scatter(memory, throughput, 
                      c=color, marker=marker, s=size, 
                      edgecolor=edge_color, linewidth=2.5, alpha=0.9, zorder=10)
            
            # Add elegant temperature label
            ax.annotate(f'{temp:.1f}°C', (memory, throughput), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, 
                                edgecolor=color, linewidth=1),
                       color=color)
        
        # Update title with elegant styling
        ax.set_title('Memory-Throughput Temperature Zones with Data Points\n(Elegant 2D Visualization)', 
                    fontsize=18, fontweight='bold', pad=25, color='#2c3e50')
        
        # Save plot if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Elegant plot with data points saved to: {output_file}")
        
        # Show plot
        if show_plot:
            plt.show()
        
        return fig, ax
        
    except ImportError:
        print("Error: pandas is required for reading CSV files. Install with: pip install pandas")
        return None, None
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None, None

def create_3d_mountain_plot(data_file, memory_range=None, throughput_range=None, 
                           output_file=None, show_plot=True):
    """
    Create a 3D contour filled plot (三维等高线填充图) where temperature is shown as contours.
    
    Args:
        data_file: CSV file with memory_gb, throughput_gbps, temperature_c columns
        memory_range: Tuple of (min, max) memory footprint in GB (auto-determined if None)
        throughput_range: Tuple of (min, max) throughput in GBps (auto-determined if None)
        output_file: Optional file to save the plot
        show_plot: Whether to display the plot
    """
    try:
        import pandas as pd
        from scipy.interpolate import griddata, RBFInterpolator
        from scipy.ndimage import gaussian_filter
        
        # Read the data
        print(f"DEBUG: Reading data from {data_file}")
        data = pd.read_csv(data_file)
        print(f"DEBUG: Data shape: {data.shape}")
        print(f"DEBUG: Data columns: {data.columns.tolist()}")
        print(f"DEBUG: First few rows:")
        print(data.head())
        
        # Auto-determine ranges if not provided
        if memory_range is None:
            print(f"DEBUG: Auto-determining memory range")
            memory_padding = (data['memory_gb'].max() - data['memory_gb'].min()) * 0.1
            memory_range = (
                max(0, data['memory_gb'].min() - memory_padding),
                data['memory_gb'].max() + memory_padding
            )
            print(f"DEBUG: Memory range: {memory_range}")
        
        if throughput_range is None:
            print(f"DEBUG: Auto-determining throughput range")
            throughput_padding = (data['throughput_gbps'].max() - data['throughput_gbps'].min()) * 0.1
            throughput_range = (
                max(0, data['throughput_gbps'].min() - throughput_padding),
                data['throughput_gbps'].max() + throughput_padding
            )
            print(f"DEBUG: Throughput range: {throughput_range}")
        
        # Create coordinate grids for smooth surface
        memory_min, memory_max = memory_range
        throughput_min, throughput_max = throughput_range
        
        print(f"DEBUG: Creating grids with memory range {memory_min}-{memory_max}, throughput range {throughput_min}-{throughput_max}")
        
        # Create grids for contour plot
        memory = np.linspace(memory_min, memory_max, 100)
        throughput = np.linspace(throughput_min, throughput_max, 100)
        memory_grid, throughput_grid = np.meshgrid(memory, throughput)
        
        print(f"DEBUG: Grid shapes - memory_grid: {memory_grid.shape}, throughput_grid: {throughput_grid.shape}")
        
        # Prepare data points for interpolation
        points = np.column_stack((data['memory_gb'], data['throughput_gbps']))
        values = data['temperature_c']
        
        print(f"DEBUG: Points shape: {points.shape}, values shape: {values.shape}")
        print(f"DEBUG: Points range: {points.min(axis=0)} to {points.max(axis=0)}")
        print(f"DEBUG: Values range: {values.min()} to {values.max()}")
        
        # Interpolate temperature values
        try:
            print(f"DEBUG: Attempting RBF interpolation")
            # Use RBF interpolation for smoother results
            rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
            grid_points = np.column_stack((memory_grid.ravel(), throughput_grid.ravel()))
            temp_grid = rbf(grid_points).reshape(memory_grid.shape)
            print(f"DEBUG: RBF interpolation successful, temp_grid shape: {temp_grid.shape}")
        except Exception as e:
            print(f"DEBUG: RBF interpolation failed: {e}")
            print(f"DEBUG: Falling back to griddata")
            # Fallback to griddata with cubic interpolation
            temp_grid = griddata(points, values, (memory_grid, throughput_grid), 
                               method='cubic', fill_value=np.nan)
            
            # Fill NaN values with linear interpolation
            temp_grid_linear = griddata(points, values, (memory_grid, throughput_grid), 
                                      method='linear', fill_value=np.nan)
            
            # Fill remaining NaN values with nearest neighbor
            temp_grid_nearest = griddata(points, values, (memory_grid, throughput_grid), 
                                       method='nearest')
            
            # Use the best available interpolation for each point
            temp_grid = np.where(np.isnan(temp_grid), temp_grid_linear, temp_grid)
            temp_grid = np.where(np.isnan(temp_grid), temp_grid_nearest, temp_grid)
            print(f"DEBUG: Griddata interpolation completed, temp_grid shape: {temp_grid.shape}")
        
        # Apply Gaussian smoothing
        print(f"DEBUG: Applying Gaussian smoothing")
        temp_grid = gaussian_filter(temp_grid, sigma=0.5)
        print(f"DEBUG: Smoothing completed")
        
        # Create the 3D plot
        print(f"DEBUG: Creating 3D plot")
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create temperature contours
        temp_min, temp_max = data['temperature_c'].min(), data['temperature_c'].max()
        print(f"DEBUG: Temperature range: {temp_min} to {temp_max}")
        
        # Generate contour levels
        num_levels = 10
        levels = np.linspace(temp_min, temp_max, num_levels)
        print(f"DEBUG: Generated {num_levels} levels: {levels}")
        
        # Create filled contours in 3D using a simpler approach
        print(f"DEBUG: Creating filled contours")
        for i, level in enumerate(levels):
            print(f"DEBUG: Processing level {i+1}/{num_levels}: {level:.2f}")
            # Create a plane at this temperature level
            level_plane = np.full_like(temp_grid, level)
            
            # Create mask for areas where temperature is above this level
            mask = temp_grid >= level
            
            # Only plot if there are valid points at this level
            if np.any(mask):
                # Color based on temperature level
                color_ratio = float((level - temp_min) / (temp_max - temp_min))
                print(f"DEBUG: Color ratio for level {level:.2f}: {color_ratio:.3f}")
                
                # Get color using a simpler approach
                color = plt.cm.viridis(color_ratio)
                print(f"DEBUG: Color for level {level:.2f}: {color}")
                
                # Create masked grid for this level
                masked_grid = np.where(mask, level_plane, np.nan)
                
                # Plot filled contour at this level
                alpha = 0.5 + 0.4 * (i / num_levels)  # Increase alpha for higher levels
                ax.plot_surface(memory_grid, throughput_grid, masked_grid,
                              color=color, alpha=alpha,
                              edgecolor='none', linewidth=0, antialiased=True)
        
        print(f"DEBUG: Adding main surface")
        # Add the main surface for better visualization
        main_surf = ax.plot_surface(memory_grid, throughput_grid, temp_grid,
                                   facecolors=plt.cm.viridis((temp_grid - temp_min) / (temp_max - temp_min)),
                                   alpha=0.3, edgecolor='none', linewidth=0, antialiased=True)
        
        print(f"DEBUG: Adding data points")
        # Add data points with better visibility
        scatter = ax.scatter(data['memory_gb'], data['throughput_gbps'], data['temperature_c'],
                           c=data['temperature_c'], cmap=plt.cm.viridis, s=100, alpha=1.0, 
                           edgecolors='black', linewidth=2, zorder=10)
        
        print(f"DEBUG: Adding text labels")
        # Add text labels for some data points to make them more visible
        for idx in range(0, len(data), max(1, len(data)//10)):  # Label every 10th point
            ax.text(data.iloc[idx]['memory_gb'], data.iloc[idx]['throughput_gbps'], 
                   data.iloc[idx]['temperature_c'] + 2, 
                   f'({data.iloc[idx]["memory_gb"]:.0f}, {data.iloc[idx]["throughput_gbps"]:.0f})',
                   fontsize=8, ha='center', va='bottom')
        
        print(f"DEBUG: Adding colorbar")
        # Add colorbar
        norm = plt.Normalize(temp_min, temp_max)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')
        
        print(f"DEBUG: Setting labels and title")
        # Set labels and title
        ax.set_xlabel('Memory Footprint (GB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (GBps)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('3D Temperature Contour Filled Plot\n(3D Contour Filled Visualization)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set axis limits
        ax.set_xlim(memory_range)
        ax.set_ylim(throughput_range)
        ax.set_zlim(temp_min - 5, temp_max + 5)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Rotate for better view
        ax.view_init(elev=25, azim=45)
        
        # Add figure information (command and timestamp)
        command, timestamp = get_figure_info()
        info_text = f'Command: {command}\nTime: {timestamp}'
        ax.text2D(0.98, 0.02, info_text, transform=ax.transAxes, 
                  fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        print(f"DEBUG: Saving plot")
        # Save plot if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"3D contour filled plot saved to: {output_file}")
        
        # Show plot
        if show_plot:
            plt.show()
        
        return fig, ax
        
    except ImportError as e:
        print(f"Error: Required libraries not available. {e}")
        print("Install with: pip install pandas scipy")
        return None, None
    except Exception as e:
        print(f"Error creating 3D contour filled plot: {e}")
        return None, None

def plot_3d_scatter(data_file, memory_range=None, throughput_range=None, output_file=None, show_plot=True, log_scale=False):
    """
    Plot a 3D scatter plot where:
    - X: memory_gb
    - Y: throughput_gbps (can use log scale for powers of 2 data)
    - Z: temperature_c
    
    Args:
        log_scale: If True, use logarithmic scale for throughput axis (useful for powers of 2 data)
    """
    import pandas as pd
    
    data = pd.read_csv(data_file)
    x = data['memory_gb'].values
    y = data['throughput_gbps'].values
    z = data['temperature_c'].values

    # Auto-adjust ranges if not provided
    if memory_range is None:
        memory_padding = (x.max() - x.min()) * 0.1 if len(x) > 1 else 1
        memory_range = [max(0, x.min() - memory_padding), x.max() + memory_padding]
    
    if throughput_range is None:
        if log_scale:
            # For log scale, use actual min/max of data
            throughput_range = [y.min(), y.max()]
        else:
            throughput_padding = (y.max() - y.min()) * 0.1 if len(y) > 1 else 1
            throughput_range = [max(0, y.min() - throughput_padding), y.max() + throughput_padding]

    # Create figure with optimized size and DPI
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Color code points by temperature
    norm = plt.Normalize(z.min(), z.max())
    colors = plt.cm.plasma(norm(z))  # Warm colors: red/orange/yellow

    # Plot scatter points with enhanced styling
    scatter = ax.scatter(x, y, z, 
                        c=colors, s=150, alpha=0.8, 
                        edgecolors='black', linewidth=1.5, zorder=10)

    # Add text labels for data points
    for i, (mem, thr, temp) in enumerate(zip(x, y, z)):
        ax.text(mem, thr, temp + 2, 
               f'({mem:.0f}, {thr:.0f})', 
               fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                        edgecolor='black', linewidth=0.5))

    # Optimized labels with better typography
    ax.set_xlabel('Memory Footprint (GB)', fontsize=13, fontweight='bold', color='#2c3e50')
    if log_scale:
        ax.set_ylabel('Throughput (GBps) - Log Scale', fontsize=13, fontweight='bold', color='#2c3e50')
    else:
        ax.set_ylabel('Throughput (GBps)', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_zlabel('Temperature (°C)', fontsize=13, fontweight='bold', color='#2c3e50')

    # Set axis limits
    ax.set_xlim(memory_range[0], memory_range[1])
    ax.set_ylim(throughput_range[0], throughput_range[1])
    ax.set_zlim(z.min() - 5, z.max() + 5)

    # Add optimized colorbar with actual temperature range
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold', color='#2c3e50')
    cbar.ax.tick_params(labelsize=10)
    
    # Update title to reflect the scale type
    scale_suffix = " (Log Scale)" if log_scale else ""
    temp_range_suffix = f" (Temp: {z.min():.1f}°C - {z.max():.1f}°C)"
    ax.set_title(f'3D Scatter: Memory vs Throughput vs Temperature', 
                fontsize=15, fontweight='bold', pad=25, color='#2c3e50')

    # Adjust view angle for better visualization
    ax.view_init(elev=16, azim=-106)
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.3, color='#95a5a6')
    
    # Optimize tick parameters
    ax.tick_params(axis='x', labelsize=10, colors='#2c3e50')
    ax.tick_params(axis='y', labelsize=10, colors='#2c3e50')
    ax.tick_params(axis='z', labelsize=10, colors='#2c3e50')

    # Add figure information (command and timestamp)
    command, timestamp = get_figure_info()
    info_text = f'Command: {command}\nTime: {timestamp}'
    ax.text2D(0.98, 0.02, info_text, transform=ax.transAxes, 
              fontsize=8, verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Optimize layout
    plt.tight_layout(pad=2.0)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"3D scatter plot saved to: {output_file}")
    if show_plot:
        plt.show()
    return fig, ax

def plot_3d_histogram(data_file, memory_range=None, throughput_range=None, output_file=None, show_plot=True, bins=10, interpolate=True, log_scale=False):
    """
    Plot a 3D histogram (bar plot) where:
    - X: memory_gb
    - Y: throughput_gbps (can use log scale for powers of 2 data)
    - Z: temperature_c (height)
    
    Args:
        log_scale: If True, use logarithmic scale for throughput axis (useful for powers of 2 data)
    """
    import pandas as pd
    from scipy.interpolate import griddata
    
    data = pd.read_csv(data_file)
    x = data['memory_gb'].values
    y = data['throughput_gbps'].values
    z = data['temperature_c'].values

    # Auto-adjust ranges if not provided
    if memory_range is None:
        memory_padding = (x.max() - x.min()) * 0.1 if len(x) > 1 else 1
        memory_range = [max(0, x.min() - memory_padding), x.max() + memory_padding]
    
    if throughput_range is None:
        if log_scale:
            # For log scale, use actual min/max of data
            throughput_range = [y.min(), y.max()]
        else:
            throughput_padding = (y.max() - y.min()) * 0.1 if len(y) > 1 else 1
            throughput_range = [max(0, y.min() - throughput_padding), y.max() + throughput_padding]

    if interpolate:
        # Create a regular grid for interpolation
        grid_size = 20  # Number of points in each dimension
        xi = np.linspace(memory_range[0], memory_range[1], grid_size)
        
        if log_scale:
            # Use log-spaced points for throughput axis
            yi = np.logspace(np.log10(throughput_range[0]), np.log10(throughput_range[1]), grid_size)
        else:
            yi = np.linspace(throughput_range[0], throughput_range[1], grid_size)
            
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate temperature values for the grid
        points = np.column_stack((x, y))
        zi_grid = griddata(points, z, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
        
        # For points outside convex hull, use nearest neighbor interpolation
        mask = np.isnan(zi_grid)
        if np.any(mask):
            zi_grid_fill = griddata(points, z, (xi_grid[mask], yi_grid[mask]), method='nearest')
            zi_grid[mask] = zi_grid_fill

        # Flatten the grid for plotting
        x_plot = xi_grid.flatten()
        y_plot = yi_grid.flatten()
        temp_min = zi_grid.min()
        temp_max = zi_grid.max()
        z_padding = (temp_max - temp_min) * 0.1
        
        # Only show bars above the x-y plane (z >= 0)
        valid_mask = zi_grid.flatten() >= 0
        x_plot = x_plot[valid_mask]
        y_plot = y_plot[valid_mask]
        # Start bars from minimum temperature instead of 0 for better visualization
        temp_min = zi_grid.min()
        z_plot = np.full_like(x_plot, temp_min)  # Bars start from minimum temperature
        dz_plot = zi_grid.flatten()[valid_mask] - temp_min  # Bar heights are relative to minimum temperature
    else:
        # Use only original data points
        temp_min = z.min()
        temp_max = z.max()
        z_padding = (temp_max - temp_min) * 0.1
        
        print(f"Temperature range: {temp_min:.2f} to {temp_max:.2f}")
        print(f"Z padding: {z_padding:.2f}")
        print(f"Bar start height: {temp_min - z_padding:.2f}")
        
        # Only show bars with positive temperature values
        valid_mask = z >= 0
        x_plot = x[valid_mask]
        y_plot = y[valid_mask]
        # Start bars from minimum temperature instead of 0 for better visualization
        z_plot = np.full_like(x_plot, z.min())  # Bars start from minimum temperature
        dz_plot = z[valid_mask] - z.min()  # Bar heights are relative to minimum temperature

    # Normalize color by temperature using warm colors (plasma colormap)
    # Use original temperature values for coloring, not the relative heights
    if interpolate:
        temp_values = zi_grid.flatten()[valid_mask]
    else:
        temp_values = z[valid_mask]
    
    norm = plt.Normalize(temp_values.min(), temp_values.max())
    colors = plt.cm.plasma(norm(temp_values))  # Warm colors: red/orange/yellow

    # Debug information
    print(f"Number of bars to plot: {len(x_plot)}")
    print(f"X range: {x_plot.min():.2f} to {x_plot.max():.2f}")
    print(f"Y range: {y_plot.min():.2f} to {y_plot.max():.2f}")
    print(f"Z range: {z_plot.min():.2f} to {z_plot.max():.2f}")
    print(f"Bar heights range: {dz_plot.min():.2f} to {dz_plot.max():.2f}")

    # Create figure with optimized size and DPI
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Bar width/length - make them smaller for better visualization
    if interpolate:
        dx = (memory_range[1] - memory_range[0]) / 20
        if log_scale:
            # For log scale, use proportional bar width based on log spacing
            dy = (np.log10(throughput_range[1]) - np.log10(throughput_range[0])) / 20
        else:
            dy = (throughput_range[1] - throughput_range[0]) / 20
    else:
        # For original data points, make bars much smaller to prevent overlap
        dx = (memory_range[1] - memory_range[0]) / 50  # Much smaller bars
        if log_scale:
            # For log scale with original data, use smaller proportional width
            dy = (np.log10(throughput_range[1]) - np.log10(throughput_range[0])) / 50
        else:
            dy = (throughput_range[1] - throughput_range[0]) / 50  # Much smaller bars

    print(f"Bar width (dx): {dx:.2f}, Bar height (dy): {dy:.2f}")
    print(f"First 5 bar positions: X={x_plot[:5]}, Y={y_plot[:5]}, Z={z_plot[:5]}")
    print(f"First 5 bar heights: {dz_plot[:5]}")

    # Plot bars with optimized styling
    ax.bar3d(x_plot, y_plot, z_plot, dx, dy, dz_plot, 
              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Optimized labels with better typography
    ax.set_xlabel('Memory Footprint (GB)', fontsize=13, fontweight='bold', color='#2c3e50')
    if log_scale:
        ax.set_ylabel('Throughput (GBps) - Log Scale', fontsize=13, fontweight='bold', color='#2c3e50')
    else:
        ax.set_ylabel('Throughput (GBps)', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_zlabel('Temperature (°C)', fontsize=13, fontweight='bold', color='#2c3e50')

    # Set axis limits to position smaller values closer to viewer
    ax.set_xlim(memory_range[0], memory_range[1])
    ax.set_ylim(throughput_range[0], throughput_range[1])
    
    # Set z-axis to start from minimum temperature for better visualization
    if interpolate:
        z_min = zi_grid.min()
        z_max = zi_grid.max() + z_padding if 'z_padding' in locals() else zi_grid.max() * 1.1
    else:
        z_min = z.min()
        z_max = z.max() + z_padding if 'z_padding' in locals() else z.max() * 1.1
    
    ax.set_zlim(z_min, z_max)
    print(f"Z-axis range: {z_min:.2f} to {z_max:.2f}")

    # Add optimized colorbar with actual temperature range
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold', color='#2c3e50')
    cbar.ax.tick_params(labelsize=10)
    
    # Update title to reflect the temperature range adjustment and scale type
    title_suffix = " (Interpolated)" if interpolate else " (Original Data)"
    scale_suffix = " (Log Scale)" if log_scale else ""
    temp_range_suffix = f" (Temp: {temp_values.min():.1f}°C - {temp_values.max():.1f}°C)"
    ax.set_title(f'3D Histogram: Memory vs Throughput vs Temperature', 
                fontsize=15, fontweight='bold', pad=25, color='#2c3e50')

    # Adjust view angle to position smaller values in the foreground
    # This makes smaller memory/throughput values appear closer to the viewer
    ax.view_init(elev=16, azim=-106)
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.3, color='#95a5a6')
    
    # Optimize tick parameters
    ax.tick_params(axis='x', labelsize=10, colors='#2c3e50')
    ax.tick_params(axis='y', labelsize=10, colors='#2c3e50')
    ax.tick_params(axis='z', labelsize=10, colors='#2c3e50')

    # Add figure information (command and timestamp)
    command, timestamp = get_figure_info()
    info_text = f'Command: {command}\nTime: {timestamp}'
    ax.text2D(0.98, 0.02, info_text, transform=ax.transAxes, 
              fontsize=8, verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Optimize layout
    plt.tight_layout(pad=2.0)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"3D histogram plot saved to: {output_file}")
    if show_plot:
        plt.show()
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='Create plot with temperature zones')
    parser.add_argument('--memory-range', nargs=2, type=float, default=[0, 32],
                       help='Memory range in GB (min max)')
    parser.add_argument('--throughput-range', nargs=2, type=float, default=[0, 100],
                       help='Throughput range in GBps (min max)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for the plot')
    parser.add_argument('--data-file', type=str, default=None,
                       help='CSV file with actual data points to plot')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot (only save if output specified)')
    parser.add_argument('--flat-2d', action='store_true',
                       help='Create 2D flat visualization instead of 3D')
    parser.add_argument('--auto-range', action='store_true',
                       help='Automatically adjust ranges based on data file')
    parser.add_argument('--mountain-3d', action='store_true',
                       help='Create a 3D contour filled plot (3D Contour Filled Visualization) where temperature is shown as contours')
    parser.add_argument('--hist3d', action='store_true',
                       help='Create a 3D histogram (bar plot) of memory, throughput, and temperature')
    parser.add_argument('--no-interpolate', action='store_true',
                       help='Do not interpolate data points for 3D histogram, show only original data points')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use logarithmic scale for throughput axis in 3D histogram (useful for powers of 2 data)')
    parser.add_argument('--scatter3d', action='store_true',
                       help='Create a 3D scatter plot of memory, throughput, and temperature')

    args = parser.parse_args()

    # Auto-adjust ranges based on data if requested
    if args.auto_range and args.data_file and os.path.exists(args.data_file):
        try:
            import pandas as pd
            data = pd.read_csv(args.data_file)
            memory_padding = (data['memory_gb'].max() - data['memory_gb'].min()) * 0.1
            throughput_padding = (data['throughput_gbps'].max() - data['throughput_gbps'].min()) * 0.1
            args.memory_range = [
                max(0, data['memory_gb'].min() - memory_padding),
                data['memory_gb'].max() + memory_padding
            ]
            args.throughput_range = [
                max(0, data['throughput_gbps'].min() - throughput_padding),
                data['throughput_gbps'].max() + throughput_padding
            ]
            print(f"Auto-adjusted ranges based on data:")
            print(f"  Memory: {args.memory_range[0]:.1f} - {args.memory_range[1]:.1f} GB")
            print(f"  Throughput: {args.throughput_range[0]:.1f} - {args.throughput_range[1]:.1f} GBps")
        except Exception as e:
            print(f"Warning: Could not auto-adjust ranges: {e}")

    # Choose between 2D, 3D zones, 3D mountain, and 3D histogram visualization
    if args.flat_2d:
        fig, ax = create_2d_zones_plot(
            memory_range=tuple(args.memory_range),
            throughput_range=tuple(args.throughput_range),
            output_file=args.output if not args.data_file else None,
            show_plot=not args.no_show and not args.data_file,
            data_file=args.data_file
        )
        if args.data_file:
            plot_2d_data_points(args.data_file, fig, ax, args.output, not args.no_show)
    elif args.mountain_3d:
        if not args.data_file:
            print("Error: --mountain-3d requires a data file. Use --data-file <filename>")
            sys.exit(1)
        memory_range = None
        throughput_range = None
        if not args.auto_range:
            memory_range = tuple(args.memory_range)
            throughput_range = tuple(args.throughput_range)
        fig, ax = create_3d_mountain_plot(
            data_file=args.data_file,
            memory_range=memory_range,
            throughput_range=throughput_range,
            output_file=args.output,
            show_plot=not args.no_show
        )
    elif args.scatter3d:
        if not args.data_file:
            print("Error: --scatter3d requires a data file. Use --data-file <filename>")
            sys.exit(1)
        plot_3d_scatter(
            data_file=args.data_file,
            memory_range=tuple(args.memory_range),
            throughput_range=tuple(args.throughput_range),
            output_file=args.output,
            show_plot=not args.no_show,
            log_scale=args.log_scale
        )
    elif args.hist3d:
        if not args.data_file:
            print("Error: --hist3d requires a data file. Use --data-file <filename>")
            sys.exit(1)
        plot_3d_histogram(
            data_file=args.data_file,
            memory_range=tuple(args.memory_range),
            throughput_range=tuple(args.throughput_range),
            output_file=args.output,
            show_plot=not args.no_show,
            interpolate=not args.no_interpolate,
            log_scale=args.log_scale
        )
    else:
        fig, ax = create_3d_zones_plot(
            memory_range=tuple(args.memory_range),
            throughput_range=tuple(args.throughput_range),
            output_file=args.output if not args.data_file else None,
            show_plot=not args.no_show and not args.data_file
        )
        if args.data_file:
            plot_data_points(args.data_file, fig, ax, args.output, not args.no_show)

if __name__ == "__main__":
    main() 