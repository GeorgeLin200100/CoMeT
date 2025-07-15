#!/usr/bin/env python3
"""
standalone_thermal.py

Standalone thermal simulation script that decouples from Sniper's performance simulation frontend.
This script provides a custom interface for bank access data and manages simulation timing independently.

Usage:
    python standalone_thermal.py --config <config_file> --duration <simulation_duration_ns>
"""

import sys
import os
import time
import argparse
import configparser
from collections import defaultdict

# Mock sim module to replace Sniper's simulation framework
class MockSim:
    def __init__(self):
        self.config = MockConfig()
        self.stats = MockStats()
        self.dvfs = MockDvfs()
        self.util = MockUtil()
        
class MockConfig:
    def __init__(self):
        self.config_data = {}
        self.output_dir = "./output"
        
    def get(self, key, default=None):
        # """Get configuration value with dot notation (e.g., 'memory/bank_size')"""
        # keys = key.split('/')
        # current = self.config_data
        # for k in keys:
        #     if isinstance(current, dict) and k in current:
        #         current = current[k]
        #     else:
        #         return default
        # return current
        return self.config_data.get(key, default)
    
    def get_bool(self, key):
        """Get boolean configuration value"""
        val = self.get(key)
        if isinstance(val, bool):
            return val
        #return val.lower() in ('true', '1', 'yes', 'on')
        return str(val).lower() in ('true', '1', 'yes', 'on')
    
    def get_int(self, key):
        """Get integer configuration value"""
        val = self.get(key)
        return int(val) if val is not None else 0
    
    def get_float(self, key):
        """Get float configuration value"""
        val = self.get(key)
        return float(val) if val is not None else 0.0
    
    def get_string(self, key):
        """Get string configuration value"""
        val = self.get(key)
        return str(val) if val is not None else ""
    
    def ncores(self):
        """Get number of cores"""
        return self.get('general/total_cores', 1)
    
    def output_dir(self):
        """Get output directory"""
        return self.output_dir

class MockStats:
    def __init__(self):
        self.current_time = 0
        self.stats_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    def time(self):
        """Get current simulation time"""
        return self.current_time
    
    def write(self, name):
        """Write statistics snapshot"""
        pass  # Mock implementation
    
    def getter(self, component, index, metric):
        """Get statistics getter"""
        return MockStatsGetter(component, index, metric, self.stats_data)

class MockStatsGetter:
    def __init__(self, component, index, metric, stats_data):
        self.component = component
        self.index = index
        self.metric = metric
        self.stats_data = stats_data
        self.last_value = self.stats_data[self.component][self.index][self.metric]
    
    def last(self):
        """Get last value"""
        return self.stats_data[self.component][self.index][self.metric]
    
    def delta(self):
        """Get delta value"""
        current = self.stats_data[self.component][self.index][self.metric]
        delta = current - self.last_value
        self.last_value = current
        return delta

class MockDvfs:
    def __init__(self):
        self.frequencies = {}
    
    def get_frequency(self, core):
        """Get frequency for a core"""
        return self.frequencies.get(core, 2000)  # Default 2GHz

class MockUtil:
    class Time:
        NS = 1
        US = 1000
        MS = 1000000
    
    class Every:
        def __init__(self, interval, callback, statsdelta=None, roi_only=True):
            self.interval = interval
            self.callback = callback
            self.statsdelta = statsdelta
            self.roi_only = roi_only
            self.time_next = 0
            self.time_last = 0
    
    class StatsDelta:
        def __init__(self):
            self.members = []
        
        def getter(self, component, index, metric):
            """Get statistics getter"""
            return MockStatsGetter(component, index, metric, {})
        
        def update(self):
            """Update statistics delta"""
            return True

# Global sim object
sim = MockSim()

# Import the original memTherm_core module and modify it
import sys
import os

# Add the scripts directory to the path so we can import the original module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the original reliability module
try:
    import reliability as rlb
except ImportError:
    # Create a mock reliability module if not available
    class MockReliability:
        def __init__(self):
            self.enabled = False
        
        def clean_reliability_files(self):
            pass
        
        def init_reliability_files(self, combined_header, ptrace_header):
            pass
        
        def update_reliability_values(self, delta_t, timestamp):
            pass
    
    rlb = MockReliability()

# Bank access data provider interface
class BankAccessProvider:
    """Interface for providing bank access data to the thermal simulation"""
    
    def __init__(self, num_banks):
        self.num_banks = num_banks
        self.read_accesses = [0] * num_banks
        self.write_accesses = [0] * num_banks
        self.read_accesses_lowpower = [0] * num_banks
        self.write_accesses_lowpower = [0] * num_banks
        self.bank_modes = [1] * num_banks  # 1 = normal power, 0 = low power
    
    def get_read_accesses(self):
        """Get read access counts for all banks"""
        return self.read_accesses.copy()
    
    def get_write_accesses(self):
        """Get write access counts for all banks"""
        return self.write_accesses.copy()
    
    def get_read_accesses_lowpower(self):
        """Get low power read access counts for all banks"""
        return self.read_accesses_lowpower.copy()
    
    def get_write_accesses_lowpower(self):
        """Get low power write access counts for all banks"""
        return self.write_accesses_lowpower.copy()
    
    def get_bank_modes(self):
        """Get bank power modes for all banks"""
        return self.bank_modes.copy()
    
    def set_access_data(self, read_accesses, write_accesses, 
                       read_accesses_lowpower=None, write_accesses_lowpower=None,
                       bank_modes=None):
        """Set access data for all banks"""
        self.read_accesses = read_accesses[:self.num_banks]
        self.write_accesses = write_accesses[:self.num_banks]
        
        if read_accesses_lowpower is not None:
            self.read_accesses_lowpower = read_accesses_lowpower[:self.num_banks]
        if write_accesses_lowpower is not None:
            self.write_accesses_lowpower = write_accesses_lowpower[:self.num_banks]
        if bank_modes is not None:
            self.bank_modes = bank_modes[:self.num_banks]

# Modified memTherm class for standalone operation
class StandaloneMemTherm:
    """Standalone version of memTherm that doesn't depend on Sniper's framework"""
    
    def __init__(self, config_file, access_provider=None):
        """Initialize the standalone thermal simulation"""
        self.load_config(config_file)
        self.access_provider = access_provider or BankAccessProvider(self.NUM_BANKS)
        self.current_time = 0
        self.sampling_interval = self.sampling_interval_ns
        self.setup_files()
        self.setup_hotspot_command()
        
        # Initialize statistics
        self.stats_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Create EnergyStats instance
        self.ES = StandaloneEnergyStats(self)
        
        print("Standalone thermal simulation initialized")
        print("Number of banks: {}".format(self.NUM_BANKS))
        print("Number of cores: {}".format(self.NUM_CORES))
        print("Sampling interval: {} ns".format(self.sampling_interval))
        self.header_written = False  # Track if header is written
    
    def load_config(self, config_file):
        """Load configuration from file"""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Load configuration into sim.config
        for section in config.sections():
            for key, value in config.items(section):
                config_key = "{}/{}".format(section, key)
                sim.config.config_data[config_key] = value
        
        # Set default values for required parameters
        defaults = {
            'memory/bank_size': 67108864,  # 64MB
            'memory/energy_per_read_access': 1.0,
            'memory/energy_per_write_access': 1.0,
            'memory/logic_core_power': 0.1,
            'memory/energy_per_refresh_access': 100.0,
            'memory/t_refi': 7.8,
            'memory/no_refesh_commands_in_t_refw': 8,
            'memory/banks_in_x': 4,
            'memory/banks_in_y': 4,
            'memory/banks_in_z': 8,
            'memory/num_banks': 128,
            'memory/cores_in_x': 4,
            'memory/cores_in_y': 4,
            'memory/cores_in_z': 1,
            'memory/type_of_stack': '3D',
            'hotspot/sampling_interval': 1000000,  # 1ms
            'hotspot/tool_path': 'hotspot_tool',
            'hotspot/config_path': '.',
            'hotspot/hotspot_config_file_mem': 'hotspot.config',
            'hotspot/floorplan_folder': 'hotspot/floorplans',
            'hotspot/layer_file_mem': 'hotspot/layers.config',
            'hotspot/log_files_mem/power_trace_file': 'power.trace',
            'hotspot/log_files_mem/temperature_trace_file': 'temperature.trace',
            'hotspot/log_files_mem/init_file': 'init.temp',
            'hotspot/log_files_mem/steady_temp_file': 'steady.temp',
            'hotspot/log_files_mem/all_transient_file': 'all_transient.temp',
            'hotspot/log_files_mem/grid_steady_file': 'grid_steady.temp',
            'hotspot/log_files/combined_temperature_trace_file': 'combined_temperature.trace',
            'hotspot/log_files/combined_power_trace_file': 'combined_power.trace',
            'scheduler/open/dram/dtm': 'off',
            'perf_model/dram/lowpower/lpm_dynamic_power': 0.5,
            'perf_model/dram/lowpower/lpm_leakage_power': 0.1,
            'perf_model/core/min_frequency': 1.0,
            'perf_model/core/max_frequency': 4.0,
            'perf_model/core/frequency_step_size': 0.1,
            'power/technology_node': 22,
            'power/vdd': 1.2,
            'power/vth': 0.3,
            'general/total_cores': 16,
            'core_thermal/enabled': 'true',
            'reliability/enabled': 'false'
        }
        
        for key, value in defaults.items():
            if sim.config.get(key) is None:
                sim.config.config_data[key] = str(value)
        
        # Extract commonly used values
        self.bank_size = int(sim.config.get('memory/bank_size'))
        self.energy_per_read_access = float(sim.config.get('memory/energy_per_read_access'))
        self.energy_per_write_access = float(sim.config.get('memory/energy_per_write_access'))
        self.logic_core_power = float(sim.config.get('memory/logic_core_power'))
        self.energy_per_refresh_access = float(sim.config.get('memory/energy_per_refresh_access'))
        self.sampling_interval_ns = int(sim.config.get('hotspot/sampling_interval'))
        self.t_refi = float(sim.config.get('memory/t_refi'))
        self.no_refesh_commands_in_t_refw = int(sim.config.get('memory/no_refesh_commands_in_t_refw'))
        self.type_of_stack = sim.config.get('memory/type_of_stack')
        self.mem_dtm = sim.config.get('scheduler/open/dram/dtm')
        self.lpm_dynamic_power = float(sim.config.get('perf_model/dram/lowpower/lpm_dynamic_power'))
        self.lpm_leakage_power = float(sim.config.get('perf_model/dram/lowpower/lpm_leakage_power'))
        self.core_frequency_min = float(sim.config.get('perf_model/core/min_frequency')) * 1000
        self.core_frequency_max = float(sim.config.get('perf_model/core/max_frequency')) * 1000
        self.core_frequency_step = float(sim.config.get('perf_model/core/frequency_step_size')) * 1000
        
        # Architecture parameters
        self.cores_in_x = int(sim.config.get('memory/cores_in_x'))
        self.cores_in_y = int(sim.config.get('memory/cores_in_y'))
        self.cores_in_z = int(sim.config.get('memory/cores_in_z'))
        self.NUM_CORES = int(sim.config.get('general/total_cores'))
        
        self.banks_in_x = int(sim.config.get('memory/banks_in_x'))
        self.banks_in_y = int(sim.config.get('memory/banks_in_y'))
        self.banks_in_z = int(sim.config.get('memory/banks_in_z'))
        self.NUM_BANKS = int(sim.config.get('memory/num_banks'))
        
        self.logic_cores_in_x = self.banks_in_x
        self.logic_cores_in_y = self.banks_in_y
        self.NUM_LC = self.logic_cores_in_x * self.logic_cores_in_y
        
        # Calculate derived parameters
        self.no_columns = 1
        self.no_bits_per_column = 8
        self.no_rows = self.bank_size / self.no_columns / self.no_bits_per_column
        self.interval_sec = self.sampling_interval_ns * 1e-9
        self.timestep = self.sampling_interval_ns / 1000
        self.rows_refreshed_in_refresh_interval = self.no_rows / self.no_refesh_commands_in_t_refw
        self.bank_static_power = 0
        self.core_thermal_enabled = sim.config.get('core_thermal/enabled')
    
    def setup_files(self):
        """Setup output files and directories"""
        os.system('mkdir -p hotspot')
        os.system('mkdir -p output')
        
        # Setup file paths first
        self.power_trace_file = sim.config.get('hotspot/log_files_mem/power_trace_file')
        self.temperature_trace_file = sim.config.get('hotspot/log_files_mem/temperature_trace_file')
        self.init_file = sim.config.get('hotspot/log_files_mem/init_file')
        self.steady_temp_file = sim.config.get('hotspot/log_files_mem/steady_temp_file')
        self.all_transient_file = sim.config.get('hotspot/log_files_mem/all_transient_file')
        self.grid_steady_file = sim.config.get('hotspot/log_files_mem/grid_steady_file')
        self.combined_temperature_trace_file = sim.config.get('hotspot/log_files/combined_temperature_trace_file')
        self.combined_power_trace_file = sim.config.get('hotspot/log_files/combined_power_trace_file')
        
        # Clean up power trace file at the start of each simulation
        if os.path.exists(self.power_trace_file):
            os.remove(self.power_trace_file)
        self.header_written = False
        
        # Initialize files
        # Do not write header here, do it in write_power_trace
        self.gen_combined_trace_header()
    
    def setup_hotspot_command(self):
        """Setup hotspot command"""
        hotspot_path = os.path.join(os.getcwd(), sim.config.get('hotspot/tool_path'))
        executable = os.path.join(hotspot_path, 'hotspot')
        hotspot_config_file = os.path.join(sim.config.get('hotspot/config_path'), sim.config.get('hotspot/hotspot_config_file_mem'))
        hotspot_layer_file = os.path.join(sim.config.get('hotspot/config_path'), sim.config.get('hotspot/layer_file_mem'))
        hotspot_floorplan_folder   = os.path.join(sim.config.get('hotspot/config_path'), sim.config.get('hotspot/floorplan_folder'))
        #initialization and setting up files
        os.system("echo copying files for first run")
        os.system("cp -r " + hotspot_floorplan_folder + " " + './hotspot')
        self.hotspot_command = "{} -c {} -p {} -o {} -model_secondary 1 -model_type grid -steady_file {} -all_transient_file {} -grid_steady_file {} -steady_state_print_disable 1 -l 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, -type {} -sampling_intvl {} -grid_layer_file {} -detailed_3D on".format(
            executable, hotspot_config_file, self.power_trace_file, self.temperature_trace_file,
            self.steady_temp_file, self.all_transient_file, self.grid_steady_file,
            self.type_of_stack, self.interval_sec, hotspot_layer_file
        )
    
    def gen_ptrace_header(self):
        """Generate power trace header"""
        ptrace_header = ''
        
        if self.type_of_stack == "2.5D":
            for x in range(self.cores_in_x):
                for y in range(self.cores_in_y):
                    ptrace_header += "C_{}\t".format(x * self.cores_in_y + y)
        
        if self.type_of_stack in ["3Dmem", "2.5D"]:
            for x in range(self.logic_cores_in_x):
                for y in range(self.logic_cores_in_y):
                    ptrace_header += "LC_{}\t".format(x * self.logic_cores_in_y + y)
        
        if self.type_of_stack == "2.5D":
            for x in range(1, 4):
                ptrace_header += "X{}\t".format(x)
        
        for z in range(self.banks_in_z):
            for x in range(self.banks_in_x):
                for y in range(self.banks_in_y):
                    bank_number = z * self.banks_in_x * self.banks_in_y + x * self.banks_in_y + y
                    ptrace_header += "B_{}\t".format(bank_number)
            if self.type_of_stack == "2.5D":
                for x in range(1, 4):
                    ptrace_header += "X{}\t".format(x)
        
        if self.type_of_stack == "3D":
            for z in range(self.cores_in_z):
                for x in range(self.cores_in_x):
                    for y in range(self.cores_in_y):
                        core_number = z * self.cores_in_x * self.cores_in_y + x * self.cores_in_y + y
                        ptrace_header += "C_{}\t".format(core_number)
        
        ptrace_header = ptrace_header.rstrip('\t')
        return ptrace_header
    
    def gen_combined_trace_header(self):
        """Generate combined trace header"""
        trace_header = ""
        for x in range(self.NUM_CORES):
            trace_header += "C_{}\t".format(x)
        for x in range(self.NUM_BANKS):
            trace_header += "B_{}\t".format(x)
        
        #trace_header = trace_header.rstrip('\t')
        with open(self.combined_temperature_trace_file, "w") as f:
            f.write("{}\n".format(trace_header))
        f.close()
        with open(self.combined_power_trace_file, "w") as f:
            f.write("{}\n".format(trace_header))
        f.close()
    
    def get_access_rates(self):
        """Get access rates from the access provider"""
        if self.access_provider:
            return (self.access_provider.get_read_accesses(),
                   self.access_provider.get_write_accesses(),
                   self.access_provider.get_read_accesses_lowpower(),
                   self.access_provider.get_write_accesses_lowpower())
        else:
            # Return zero access rates if no provider
            return ([0] * self.NUM_BANKS, [0] * self.NUM_BANKS, 
                   [0] * self.NUM_BANKS, [0] * self.NUM_BANKS)
    
    def get_bank_modes(self):
        """Get bank modes from the access provider"""
        if self.access_provider:
            return self.access_provider.get_bank_modes()
        else:
            return [1] * self.NUM_BANKS  # Default to normal power mode
    
    def calc_power_trace(self):
        """Calculate power trace"""
        accesses_read, accesses_write, accesses_read_lowpower, accesses_write_lowpower = self.get_access_rates()
        
        # Calculate refresh power
        avg_no_refresh_intervals_in_timestep = self.timestep / self.t_refi
        avg_no_refresh_rows_in_timestep = avg_no_refresh_intervals_in_timestep * self.rows_refreshed_in_refresh_interval
        refresh_energy_in_timestep = avg_no_refresh_rows_in_timestep * self.energy_per_refresh_access
        avg_refresh_power = refresh_energy_in_timestep / (self.timestep * 1000)
        
        # Calculate bank power
        bank_power_trace = [0] * self.NUM_BANKS
        for bank in range(self.NUM_BANKS):
            if self.mem_dtm != 'off':
                normal_power_access = (accesses_read[bank] * self.energy_per_read_access + 
                                     accesses_write[bank] * self.energy_per_write_access)
                low_power_access = ((accesses_read_lowpower[bank] * self.energy_per_read_access + 
                                   accesses_write_lowpower[bank] * self.energy_per_write_access) * 
                                  self.lpm_dynamic_power)
                bank_power_trace[bank] = ((normal_power_access + low_power_access) / 
                                        (self.timestep * 1000) + self.bank_static_power + avg_refresh_power)
            else:
                bank_power_trace[bank] = ((accesses_read[bank] * self.energy_per_read_access + 
                                         accesses_write[bank] * self.energy_per_write_access) / 
                                        (self.timestep * 1000) + self.bank_static_power + avg_refresh_power)
            bank_power_trace[bank] = round(bank_power_trace[bank], 3)
        
        # Build the power trace line as a list of values (no trailing tab)
        power_values = []
        # Add core power for 2.5D
        if self.type_of_stack == "2.5D":
            # Mock core power data
            power_values.extend(['0.1'] * self.NUM_CORES)
        # Add logic core power
        if self.type_of_stack in ["2.5D", "3Dmem"]:
            power_values.extend([str(self.logic_core_power)] * self.NUM_LC)
        # Add X1, X2, X3 for 2.5D
        if self.type_of_stack == "2.5D":
            power_values.extend(["0.00"] * 3)
        # Add bank power
        for bank in range(len(bank_power_trace)):
            if self.type_of_stack == "2.5D" and bank % (self.banks_in_x * self.banks_in_y) == 0 and bank > 0:
                power_values.extend(["0.00"] * 3)
            power_values.append(str(bank_power_trace[bank]))
        # Add X1, X2, X3 for 2.5D at the end
        if self.type_of_stack == "2.5D":
            power_values.extend(["0.00"] * 3)
        # Add core power for 3D
        if self.type_of_stack == "3D":
            power_values.extend(['0.1'] * self.NUM_CORES)
        power_trace = '\t'.join(power_values) + '\n'
        return power_trace
    
    def write_power_trace(self, power_trace):
        """Write power trace following the golden reference approach"""
        if not self.header_written:
            # Write header twice due to Hotspot double-read bug
            header = self.gen_ptrace_header()
            with open(self.power_trace_file, "w") as f:
                f.write("{}\n".format(header))
                f.write("{}\n".format(header))  # duplicate header
                f.write(power_trace)
            self.header_written = True
        else:
            # Append only data for subsequent steps
            with open(self.power_trace_file, "a") as f:
                f.write(power_trace)
    
    def write_bank_leakage_trace(self):
        """Write bank leakage trace"""
        bank_modes = self.get_bank_modes()
        bank_mode_trace_string = ""
        
        for bank in range(self.NUM_BANKS):
            leakage = 1.0
            if bank_modes[bank] == 0:  # LOW_POWER
                leakage = self.lpm_leakage_power
            bank_mode_trace_string += "{:.2f}\t".format(leakage)
        bank_mode_trace_string += "\r\n"
        
        # Write bank mode information
        bank_mode_header = '\t'.join(["B_{}".format(i) for i in range(self.NUM_BANKS)])
        with open("bank_mode.trace", "w") as f:
            f.write("{}\n".format(bank_mode_header))
            f.write(bank_mode_trace_string)
        f.close()
    
    def execute_hotspot(self):
        """Execute hotspot thermal simulation"""
        cmd = self.hotspot_command
        if os.path.exists(self.init_file):
            cmd += ' -init_file ' + self.init_file
        
        print("Executing hotspot command: {}".format(cmd))
        result = os.system(cmd)
        
        if result == 0:
            print("Hotspot simulation completed successfully")
            # Copy transient file to init file for next iteration
            if os.path.exists(self.all_transient_file):
                os.system("cp {} {}".format(self.all_transient_file, self.init_file))
        else:
            print("Hotspot simulation failed with exit code {}".format(result))
    
    def step(self, time_delta_ns):
        """Execute one simulation step"""
        print("Simulation step at time {} ns (delta: {} ns)".format(self.current_time, time_delta_ns))
        
        # Update simulation time
        sim.stats.current_time = self.current_time
        
        # Write bank leakage trace
        self.write_bank_leakage_trace()
        
        # Calculate power trace and write it robustly
        power_trace = self.calc_power_trace()
        self.write_power_trace(power_trace)

        if (os.path.exists('integration_power.trace')):
            with open('integration_power.trace', 'r') as f:
                for i, line in enumerate(f):
                    #print(f"line {i+1}: {line.strip()}")
                    print(f"line {i+1}: {line}")
            f.close()
        # Execute hotspot
        self.execute_hotspot()
        
        # Update time
        self.current_time += time_delta_ns
        
        print("Step completed")
    
    def run(self, duration_ns):
        """Run the thermal simulation for the specified duration"""
        print("Starting thermal simulation for {} ns".format(duration_ns))
        print("Sampling interval: {} ns".format(self.sampling_interval))
        
        steps = duration_ns // self.sampling_interval
        print("Will execute {} steps".format(steps))
        
        for step in range(steps):
            print("\n--- Step {} of {} ---".format(step + 1, steps))
            self.step(self.sampling_interval)
            
            # Small delay to make output readable
            time.sleep(0.1)
        
        print("\nThermal simulation completed")

class StandaloneEnergyStats:
    """Standalone version of EnergyStats"""
    
    def __init__(self, thermal_sim):
        self.thermal_sim = thermal_sim
        self.dvfs_table = self.build_dvfs_table(int(sim.config.get('power/technology_node')))
    
    def build_dvfs_table(self, tech):
        """Build DVFS table"""
        if tech <= 22:
            def v(f):
                return 0.6 + f / self.thermal_sim.core_frequency_max * 0.8
            return [(f, v(f)) for f in reversed(range(int(self.thermal_sim.core_frequency_min), 
                                                     int(self.thermal_sim.core_frequency_max) + 1, 
                                                     int(self.thermal_sim.core_frequency_step)))]
        elif tech == 45:
            return [(2000, 1.2), (1800, 1.1), (1500, 1.0), (1000, 0.9), (0, 0.8)]
        else:
            raise ValueError('No DVFS table available for {} nm technology node'.format(tech))
    
    def get_vdd_from_freq(self, f):
        """Get VDD from frequency"""
        if f > self.thermal_sim.core_frequency_max:
            raise ValueError('Could not find a Vdd for invalid frequency {} exceeding the core\'s maximum frequency'.format(f))
        for _f, _v in self.dvfs_table:
            if f >= _f:
                return _v
        raise ValueError('Could not find a Vdd for invalid frequency {}'.format(f))
    
    def periodic(self, time, time_delta):
        """Periodic update"""
        pass  # Mock implementation

def create_sample_config():
    """Create a sample configuration file"""
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
    
    with open('sample_config.cfg', 'w') as f:
        f.write(config_content)
    f.close()
    
    print("Sample configuration file created: sample_config.cfg")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Standalone thermal simulation')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--duration', type=int, default=10000000, help='Simulation duration in nanoseconds')
    parser.add_argument('--create-sample-config', action='store_true', help='Create a sample configuration file')
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    if not os.path.exists(args.config):
        print("Error: Configuration file '{}' not found".format(args.config))
        return
    
    # Create access provider with sample data
    access_provider = BankAccessProvider(128)  # 128 banks
    
    # Set some sample access data
    read_accesses = [10 + i % 20 for i in range(128)]  # Varying read accesses
    write_accesses = [5 + i % 10 for i in range(128)]  # Varying write accesses
    bank_modes = [1 if i % 10 != 0 else 0 for i in range(128)]  # Some banks in low power
    
    access_provider.set_access_data(read_accesses, write_accesses, 
                                   read_accesses, write_accesses, bank_modes)
    
    # Create and run thermal simulation
    thermal_sim = StandaloneMemTherm(args.config, access_provider)
    thermal_sim.run(args.duration)

if __name__ == "__main__":
    main() 