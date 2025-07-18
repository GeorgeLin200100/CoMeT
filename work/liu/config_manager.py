#!/usr/bin/env python3
"""
config_manager.py

Shared configuration manager for CoMeT thermal simulation.
Provides a unified interface for accessing configuration values across modules.
"""

import configparser
import os

class ConfigManager:
    """Centralized configuration manager for CoMeT thermal simulation"""
    
    def __init__(self, config_file):
        """Initialize configuration manager with config file path"""
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # Load all configuration values
        self._load_config()
    
    def _load_config(self):
        """Load all configuration values into instance variables"""
        # General settings
        self.total_cores = self._get_int('general', 'total_cores', 16)
        
        # Memory settings
        self.bank_size = self._get_int('memory', 'bank_size', 64)
        self.energy_per_read_access = self._get_float('memory', 'energy_per_read_access', 20.55)
        self.energy_per_write_access = self._get_float('memory', 'energy_per_write_access', 20.55)
        self.logic_core_power = self._get_float('memory', 'logic_core_power', 0.272)
        self.energy_per_refresh_access = self._get_float('memory', 'energy_per_refresh_access', 3.55)
        self.t_refi = self._get_float('memory', 't_refi', 7.8)
        self.no_refesh_commands_in_t_refw = self._get_int('memory', 'no_refesh_commands_in_t_refw', 8)
        
        # Architecture settings
        self.banks_in_x = self._get_int('memory', 'banks_in_x', 6)
        self.banks_in_y = self._get_int('memory', 'banks_in_y', 8)
        self.banks_in_z = self._get_int('memory', 'banks_in_z', 4)
        self.num_banks = self._get_int('memory', 'num_banks', 192)
        self.cores_in_x = self._get_int('memory', 'cores_in_x', 2)
        self.cores_in_y = self._get_int('memory', 'cores_in_y', 2)
        self.cores_in_z = self._get_int('memory', 'cores_in_z', 1)
        self.type_of_stack = self._get_string('memory', 'type_of_stack', '3Dmem')
        
        # Additional memory controller/architecture parameters
        self.clock_freq = self._get_int('memory', 'clock_freq', 500)  # MHz
        self.group_rows = self._get_int('memory', 'group_rows', 2)
        self.group_cols = self._get_int('memory', 'group_cols', 2)
        self.num_groups = self._get_int('memory', 'num_groups', 4)
        
        # Hotspot settings
        self.sampling_interval = self._get_int('hotspot', 'sampling_interval', 1000000)
        self.tool_path = self._get_string('hotspot', 'tool_path', '../../hotspot_tool')
        self.config_path = self._get_string('hotspot', 'config_path', './')
        self.hotspot_config_file_mem = self._get_string('hotspot', 'hotspot_config_file_mem', 'config/mem_hotspot.config')
        self.floorplan_folder = self._get_string('hotspot', 'floorplan_folder', 'config/')
        self.layer_file_mem = self._get_string('hotspot', 'layer_file_mem', 'config/mem.lcf')
        
        # Hotspot log files
        self.power_trace_file = self._get_string('hotspot/log_files_mem', 'power_trace_file', 'power.trace')
        self.temperature_trace_file = self._get_string('hotspot/log_files_mem', 'temperature_trace_file', 'temperature.trace')
        self.init_file = self._get_string('hotspot/log_files_mem', 'init_file', 'init.temp')
        self.init_file_external_mem = self._get_string('hotspot/log_files_mem', 'init_file_external_mem', 'config/init.temp')
        self.steady_temp_file = self._get_string('hotspot/log_files_mem', 'steady_temp_file', 'steady.temp')
        self.all_transient_file = self._get_string('hotspot/log_files_mem', 'all_transient_file', 'all_transient.temp')
        self.grid_steady_file = self._get_string('hotspot/log_files_mem', 'grid_steady_file', 'grid_steady.temp')
        
        # Combined log files
        self.combined_temperature_trace_file = self._get_string('hotspot/log_files', 'combined_temperature_trace_file', 'combined_temperature.trace')
        self.combined_power_trace_file = self._get_string('hotspot/log_files', 'combined_power_trace_file', 'combined_power.trace')
        
        # Power settings
        self.technology_node = self._get_int('power', 'technology_node', 22)
        self.vdd = self._get_float('power', 'vdd', 1.2)
        self.vth = self._get_float('power', 'vth', 0.3)
        
        # Performance model settings
        self.lpm_dynamic_power = self._get_float('perf_model/dram/lowpower', 'lpm_dynamic_power', 0.5)
        self.lpm_leakage_power = self._get_float('perf_model/dram/lowpower', 'lpm_leakage_power', 0.1)
        self.min_frequency = self._get_float('perf_model/core', 'min_frequency', 1.0)
        self.max_frequency = self._get_float('perf_model/core', 'max_frequency', 4.0)
        self.frequency_step_size = self._get_float('perf_model/core', 'frequency_step_size', 0.1)
        
        # Scheduler settings
        self.dtm = self._get_string('scheduler/open/dram/dtm', 'dtm', 'off')
        
        # Feature flags
        self.core_thermal_enabled = self._get_bool('core_thermal', 'enabled', True)
        self.reliability_enabled = self._get_bool('reliability', 'enabled', False)
        
        # Calculate derived values
        self._calculate_derived_values()
    
    def _get_int(self, section, key, default):
        """Get integer configuration value"""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def _get_float(self, section, key, default):
        """Get float configuration value"""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def _get_string(self, section, key, default):
        """Get string configuration value"""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def _get_bool(self, section, key, default):
        """Get boolean configuration value"""
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def _calculate_derived_values(self):
        """Calculate derived configuration values"""
        # Calculate total banks if not explicitly set
        if self.num_banks == 192 and self.banks_in_x * self.banks_in_y * self.banks_in_z != 192:
            self.num_banks = self.banks_in_x * self.banks_in_y * self.banks_in_z
        
        # Calculate total cores if not explicitly set
        if self.total_cores == 16 and self.cores_in_x * self.cores_in_y * self.cores_in_z != 16:
            self.total_cores = self.cores_in_x * self.cores_in_y * self.cores_in_z
    
    def get_sim_config_dict(self):
        """Get configuration as dictionary for sim.config compatibility"""
        """This method returns a dictionary that can be used to populate sim.config.config_data
        for compatibility with the existing standalone_thermal.py code"""
        config_dict = {}
        
        # Add all configuration values in the format expected by sim.config
        config_dict.update({
            'general/total_cores': str(self.total_cores),
            'memory/bank_size': str(self.bank_size),
            'memory/energy_per_read_access': str(self.energy_per_read_access),
            'memory/energy_per_write_access': str(self.energy_per_write_access),
            'memory/logic_core_power': str(self.logic_core_power),
            'memory/energy_per_refresh_access': str(self.energy_per_refresh_access),
            'memory/t_refi': str(self.t_refi),
            'memory/no_refesh_commands_in_t_refw': str(self.no_refesh_commands_in_t_refw),
            'memory/banks_in_x': str(self.banks_in_x),
            'memory/banks_in_y': str(self.banks_in_y),
            'memory/banks_in_z': str(self.banks_in_z),
            'memory/num_banks': str(self.num_banks),
            'memory/cores_in_x': str(self.cores_in_x),
            'memory/cores_in_y': str(self.cores_in_y),
            'memory/cores_in_z': str(self.cores_in_z),
            'memory/type_of_stack': str(self.type_of_stack),
            'memory/clock_freq': str(self.clock_freq),
            'memory/group_rows': str(self.group_rows),
            'memory/group_cols': str(self.group_cols),
            'memory/num_groups': str(self.num_groups),
            'hotspot/sampling_interval': str(self.sampling_interval),
            'hotspot/tool_path': str(self.tool_path),
            'hotspot/config_path': str(self.config_path),
            'hotspot/hotspot_config_file_mem': str(self.hotspot_config_file_mem),
            'hotspot/floorplan_folder': str(self.floorplan_folder),
            'hotspot/layer_file_mem': str(self.layer_file_mem),
            'hotspot/log_files_mem/power_trace_file': str(self.power_trace_file),
            'hotspot/log_files_mem/temperature_trace_file': str(self.temperature_trace_file),
            'hotspot/log_files_mem/init_file': str(self.init_file),
            'hotspot/log_files_mem/init_file_external_mem': str(self.init_file_external_mem),
            'hotspot/log_files_mem/steady_temp_file': str(self.steady_temp_file),
            'hotspot/log_files_mem/all_transient_file': str(self.all_transient_file),
            'hotspot/log_files_mem/grid_steady_file': str(self.grid_steady_file),
            'hotspot/log_files/combined_temperature_trace_file': str(self.combined_temperature_trace_file),
            'hotspot/log_files/combined_power_trace_file': str(self.combined_power_trace_file),
            'power/technology_node': str(self.technology_node),
            'power/vdd': str(self.vdd),
            'power/vth': str(self.vth),
            'perf_model/dram/lowpower/lpm_dynamic_power': str(self.lpm_dynamic_power),
            'perf_model/dram/lowpower/lpm_leakage_power': str(self.lpm_leakage_power),
            'perf_model/core/min_frequency': str(self.min_frequency),
            'perf_model/core/max_frequency': str(self.max_frequency),
            'perf_model/core/frequency_step_size': str(self.frequency_step_size),
            'scheduler/open/dram/dtm': str(self.dtm),
            'core_thermal/enabled': str(self.core_thermal_enabled).lower(),
            'reliability/enabled': str(self.reliability_enabled).lower(),
        })
        
        return config_dict
    
    def print_summary(self):
        """Print a summary of the configuration"""
        print("Configuration Summary:")
        print(f"  Total cores: {self.total_cores}")
        print(f"  Memory banks: {self.num_banks} ({self.banks_in_x}x{self.banks_in_y}x{self.banks_in_z})")
        print(f"  Bank size: {self.bank_size} MB")
        print(f"  Stack type: {self.type_of_stack}")
        print(f"  Sampling interval: {self.sampling_interval} ns")
        print(f"  Technology node: {self.technology_node} nm") 