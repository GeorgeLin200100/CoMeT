#!/usr/bin/env python2
"""
test_standalone.py

Test script to verify the standalone thermal simulation functionality.
"""

import os
import sys
import unittest
from standalone_thermal import MockSim, MockConfig, BankAccessProvider, StandaloneMemTherm

class TestMockFramework(unittest.TestCase):
    """Test the mock framework components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sim = MockSim()
        self.config = MockConfig()
    
    def test_mock_config(self):
        """Test MockConfig functionality"""
        # Test setting and getting config values
        self.config.config_data['memory/bank_size'] = '67108864'
        self.config.config_data['memory/energy_per_read_access'] = '1.0'
        
        self.assertEqual(self.config.get('memory/bank_size'), '67108864')
        self.assertEqual(self.config.get_int('memory/bank_size'), 67108864)
        self.assertEqual(self.config.get_float('memory/energy_per_read_access'), 1.0)
        
        # Test default values
        self.assertEqual(self.config.get('nonexistent/key', 'default'), 'default')
    
    def test_mock_stats(self):
        """Test MockStats functionality"""
        # Test statistics tracking
        self.sim.stats.stats_data['dram'][0]['bank_read_access_counter'] = 100
        self.sim.stats.stats_data['dram'][1]['bank_write_access_counter'] = 50
        
        getter1 = self.sim.stats.getter('dram', 0, 'bank_read_access_counter')
        getter2 = self.sim.stats.getter('dram', 1, 'bank_write_access_counter')
        
        self.assertEqual(getter1.last(), 100)
        self.assertEqual(getter2.last(), 50)
        
        # Test delta calculation
        self.sim.stats.stats_data['dram'][0]['bank_read_access_counter'] = 150
        self.assertEqual(getter1.delta(), 50)

class TestBankAccessProvider(unittest.TestCase):
    """Test BankAccessProvider functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.provider = BankAccessProvider(128)
    
    def test_initial_state(self):
        """Test initial state of access provider"""
        self.assertEqual(len(self.provider.get_read_accesses()), 128)
        self.assertEqual(len(self.provider.get_write_accesses()), 128)
        self.assertEqual(len(self.provider.get_bank_modes()), 128)
        
        # All should be zero initially
        self.assertEqual(sum(self.provider.get_read_accesses()), 0)
        self.assertEqual(sum(self.provider.get_write_accesses()), 0)
        
        # All banks should be in normal power mode initially
        self.assertEqual(sum(self.provider.get_bank_modes()), 128)
    
    def test_set_access_data(self):
        """Test setting access data"""
        read_accesses = [i for i in range(128)]
        write_accesses = [i * 2 for i in range(128)]
        bank_modes = [1 if i % 2 == 0 else 0 for i in range(128)]
        
        self.provider.set_access_data(read_accesses, write_accesses, 
                                    read_accesses, write_accesses, bank_modes)
        
        # Verify data was set correctly
        self.assertEqual(self.provider.get_read_accesses(), read_accesses)
        self.assertEqual(self.provider.get_write_accesses(), write_accesses)
        self.assertEqual(self.provider.get_bank_modes(), bank_modes)
    
    def test_partial_data_setting(self):
        """Test setting partial access data"""
        read_accesses = [10] * 128
        write_accesses = [5] * 128
        
        self.provider.set_access_data(read_accesses, write_accesses)
        
        # Should use defaults for unspecified data
        self.assertEqual(self.provider.get_read_accesses(), read_accesses)
        self.assertEqual(self.provider.get_write_accesses(), write_accesses)
        self.assertEqual(sum(self.provider.get_bank_modes()), 128)  # All normal power

class TestStandaloneMemTherm(unittest.TestCase):
    """Test StandaloneMemTherm functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal config file for testing
        self.config_content = """[general]
total_cores = 4

[memory]
bank_size = 67108864
energy_per_read_access = 1.0
energy_per_write_access = 1.0
logic_core_power = 0.1
energy_per_refresh_access = 100.0
t_refi = 7.8
no_refesh_commands_in_t_refw = 8
banks_in_x = 2
banks_in_y = 2
banks_in_z = 2
num_banks = 8
cores_in_x = 2
cores_in_y = 2
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
power_trace_file = test_power.trace
temperature_trace_file = test_temperature.trace
init_file = test_init.temp
steady_temp_file = test_steady.temp
all_transient_file = test_all_transient.temp
grid_steady_file = test_grid_steady.temp

[hotspot/log_files]
combined_temperature_trace_file = test_combined_temperature.trace
combined_power_trace_file = test_combined_power.trace

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
        
        with open('test_config.cfg', 'w') as f:
            f.write(self.config_content)
    
    def tearDown(self):
        """Clean up test files"""
        test_files = [
            'test_config.cfg',
            'test_power.trace',
            'test_temperature.trace',
            'test_combined_temperature.trace',
            'test_combined_power.trace',
            'bank_mode.trace'
        ]
        
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_config_loading(self):
        """Test configuration loading"""
        access_provider = BankAccessProvider(8)
        thermal_sim = StandaloneMemTherm('test_config.cfg', access_provider)
        
        # Test that configuration was loaded correctly
        self.assertEqual(thermal_sim.NUM_BANKS, 8)
        self.assertEqual(thermal_sim.NUM_CORES, 4)
        self.assertEqual(thermal_sim.type_of_stack, '3D')
        self.assertEqual(thermal_sim.energy_per_read_access, 1.0)
        self.assertEqual(thermal_sim.energy_per_write_access, 1.0)
    
    def test_power_trace_generation(self):
        """Test power trace generation"""
        access_provider = BankAccessProvider(8)
        
        # Set some test access data
        read_accesses = [10, 15, 20, 25, 30, 35, 40, 45]
        write_accesses = [5, 8, 12, 15, 18, 22, 25, 28]
        bank_modes = [1, 1, 0, 1, 0, 1, 1, 0]  # Some banks in low power
        
        access_provider.set_access_data(read_accesses, write_accesses, 
                                      read_accesses, write_accesses, bank_modes)
        
        thermal_sim = StandaloneMemTherm('test_config.cfg', access_provider)
        
        # Test power trace calculation
        power_trace = thermal_sim.calc_power_trace()
        
        # Verify power trace was generated
        self.assertIsNotNone(power_trace)
        self.assertIn('B_0', power_trace)
        self.assertIn('B_1', power_trace)
    
    def test_file_generation(self):
        """Test file generation"""
        access_provider = BankAccessProvider(8)
        thermal_sim = StandaloneMemTherm('test_config.cfg', access_provider)
        
        # Test header generation
        thermal_sim.gen_ptrace_header()
        thermal_sim.gen_combined_trace_header()
        
        # Check that files were created
        self.assertTrue(os.path.exists('test_power.trace'))
        self.assertTrue(os.path.exists('test_combined_temperature.trace'))
        self.assertTrue(os.path.exists('test_combined_power.trace'))
    
    def test_bank_leakage_trace(self):
        """Test bank leakage trace generation"""
        access_provider = BankAccessProvider(8)
        
        # Set some banks in low power mode
        bank_modes = [1, 0, 1, 0, 1, 0, 1, 0]
        access_provider.set_access_data([10]*8, [5]*8, [10]*8, [5]*8, bank_modes)
        
        thermal_sim = StandaloneMemTherm('test_config.cfg', access_provider)
        thermal_sim.write_bank_leakage_trace()
        
        # Check that bank mode trace was created
        self.assertTrue(os.path.exists('bank_mode.trace'))

class TestIntegration(unittest.TestCase):
    """Test integration of components"""
    
    def test_full_simulation_flow(self):
        """Test complete simulation flow"""
        # Create minimal config
        config_content = """[general]
total_cores = 2

[memory]
bank_size = 67108864
energy_per_read_access = 1.0
energy_per_write_access = 1.0
logic_core_power = 0.1
energy_per_refresh_access = 100.0
t_refi = 7.8
no_refesh_commands_in_t_refw = 8
banks_in_x = 2
banks_in_y = 2
banks_in_z = 1
num_banks = 4
cores_in_x = 2
cores_in_y = 1
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
power_trace_file = integration_power.trace
temperature_trace_file = integration_temperature.trace
init_file = integration_init.temp
steady_temp_file = integration_steady.temp
all_transient_file = integration_all_transient.temp
grid_steady_file = integration_grid_steady.temp

[hotspot/log_files]
combined_temperature_trace_file = integration_combined_temperature.trace
combined_power_trace_file = integration_combined_power.trace

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
        
        with open('integration_config.cfg', 'w') as f:
            f.write(config_content)
        
        try:
            # Create access provider
            access_provider = BankAccessProvider(4)
            access_provider.set_access_data([10, 15, 20, 25], [5, 8, 12, 15])
            
            # Create thermal simulation
            thermal_sim = StandaloneMemTherm('integration_config.cfg', access_provider)
            
            # Run a single step
            thermal_sim.step(1000000)  # 1ms step
            
            # Verify files were created
            self.assertTrue(os.path.exists('integration_power.trace'))
            self.assertTrue(os.path.exists('integration_combined_temperature.trace'))
            self.assertTrue(os.path.exists('integration_combined_power.trace'))
            
        finally:
            # Clean up
            test_files = [
                'integration_config.cfg',
                'integration_power.trace',
                'integration_temperature.trace',
                'integration_combined_temperature.trace',
                'integration_combined_power.trace',
                'bank_mode.trace'
            ]
            
            for filename in test_files:
                if os.path.exists(filename):
                    os.remove(filename)

def run_tests():
    """Run all tests"""
    print("Running standalone thermal simulation tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestMockFramework))
    test_suite.addTest(unittest.makeSuite(TestBankAccessProvider))
    test_suite.addTest(unittest.makeSuite(TestStandaloneMemTherm))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("Tests run: {}".format(result.testsRun))
    print("Failures: {}".format(len(result.failures)))
    print("Errors: {}".format(len(result.errors)))
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print("  {}: {}".format(test, traceback))
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print("  {}: {}".format(test, traceback))
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 