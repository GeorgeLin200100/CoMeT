# Standalone Thermal Simulation

This directory contains a standalone thermal simulation script that decouples CoMeT's thermal simulation from Sniper's performance simulation frontend.

## Overview

The `standalone_thermal.py` script provides a complete thermal simulation environment that:

1. **Imports necessary CoMeT modules and configurations** - Uses a custom configuration system that mimics Sniper's `sim.config`
2. **Instantiates the memTherm class directly** - Creates a `StandaloneMemTherm` class that provides the same functionality as the original `memTherm` class
3. **Provides custom bank access data** - Uses a `BankAccessProvider` interface to supply memory access patterns
4. **Manages simulation timing and periodic callbacks** - Implements its own timing system independent of Sniper

## Key Components

### MockSim Framework
The script includes mock implementations of Sniper's core components:
- `MockSim` - Main simulation object
- `MockConfig` - Configuration management
- `MockStats` - Statistics tracking
- `MockDvfs` - Dynamic voltage/frequency scaling
- `MockUtil` - Utility functions

### BankAccessProvider Interface
This interface allows you to provide custom memory access data:
```python
class BankAccessProvider:
    def get_read_accesses(self):
        """Return read access counts for all banks"""
        return self.read_accesses.copy()
    
    def get_write_accesses(self):
        """Return write access counts for all banks"""
        return self.write_accesses.copy()
    
    def get_bank_modes(self):
        """Return bank power modes (1=normal, 0=low power)"""
        return self.bank_modes.copy()
```

### StandaloneMemTherm Class
This is the main thermal simulation class that:
- Loads configuration from files
- Calculates power traces based on access patterns
- Executes HotSpot thermal simulation
- Manages output files and traces

## Usage

### Basic Usage
```bash
python standalone_thermal.py --config sample_config.cfg --duration 10000000
```

### Command Line Options
- `--config <file>` - Configuration file (required)
- `--duration <ns>` - Simulation duration in nanoseconds (default: 10000000)
- `--create-sample-config` - Create a sample configuration file

### Creating a Sample Configuration
```bash
python standalone_thermal.py --create-sample-config
```

## Configuration File Format

The configuration file uses INI format with sections and key-value pairs:

```ini
[general]
total_cores = 16

[memory]
bank_size = 67108864
energy_per_read_access = 1.0
energy_per_write_access = 1.0
type_of_stack = 3D
num_banks = 128

[hotspot]
sampling_interval = 1000000
tool_path = hotspot_tool
```

### Key Configuration Parameters

#### Memory Parameters
- `bank_size` - Size of each memory bank in bytes
- `energy_per_read_access` - Energy per read access (nJ)
- `energy_per_write_access` - Energy per write access (nJ)
- `type_of_stack` - Memory architecture (3D, 3Dmem, 2.5D, DDR)
- `num_banks` - Total number of memory banks

#### HotSpot Parameters
- `sampling_interval` - Thermal simulation interval in nanoseconds
- `tool_path` - Path to HotSpot tool
- `hotspot_config_file_mem` - HotSpot configuration file

#### Architecture Parameters
- `banks_in_x/y/z` - Bank layout dimensions
- `cores_in_x/y/z` - Core layout dimensions
- `total_cores` - Total number of cores

## Customizing Bank Access Data

You can provide custom access patterns by creating a custom `BankAccessProvider`:

```python
class CustomAccessProvider(BankAccessProvider):
    def __init__(self, num_banks):
        super().__init__(num_banks)
        # Initialize with your custom access patterns
        self.read_accesses = [10 + i % 20 for i in range(num_banks)]
        self.write_accesses = [5 + i % 10 for i in range(num_banks)]
        self.bank_modes = [1 if i % 10 != 0 else 0 for i in range(num_banks)]

# Use in simulation
access_provider = CustomAccessProvider(128)
thermal_sim = StandaloneMemTherm(config_file, access_provider)
```

## Output Files

The simulation generates several output files:
- `power.trace` - Power consumption traces
- `temperature.trace` - Temperature traces
- `combined_temperature.trace` - Combined temperature data
- `combined_power.trace` - Combined power data
- `bank_mode.trace` - Bank power mode information

## Integration with Existing Code

To integrate this with your existing CoMeT setup:

1. **Replace Sniper calls** - Use the mock framework instead of `sim.config`
2. **Provide access data** - Implement a `BankAccessProvider` for your access patterns
3. **Configure HotSpot** - Ensure HotSpot tool and configuration files are available
4. **Run simulation** - Execute the standalone script with your configuration

## Example Integration

```python
from standalone_thermal import StandaloneMemTherm, BankAccessProvider

# Create access provider with your data
access_provider = BankAccessProvider(128)
access_provider.set_access_data(
    read_accesses=[10, 15, 20, ...],  # Your read access data
    write_accesses=[5, 8, 12, ...],   # Your write access data
    bank_modes=[1, 1, 0, ...]         # Your bank power modes
)

# Run thermal simulation
thermal_sim = StandaloneMemTherm('my_config.cfg', access_provider)
thermal_sim.run(10000000)  # 10ms simulation
```

## Dependencies

- Python 2.7
- HotSpot thermal simulation tool
- Configuration files for HotSpot
- Floorplan files for your architecture

## Troubleshooting

1. **HotSpot not found** - Ensure HotSpot tool is installed and path is correct in config
2. **Configuration errors** - Check that all required parameters are set in config file
3. **File permission errors** - Ensure write permissions for output directory
4. **Import errors** - Make sure all required modules are available

## Differences from Original

The standalone version differs from the original Sniper-integrated version in several ways:

1. **Independent timing** - Uses its own timing system instead of Sniper's
2. **Custom access data** - Provides interface for custom access patterns
3. **Mock framework** - Replaces Sniper dependencies with mock implementations
4. **Simplified interface** - Focuses on thermal simulation without performance modeling
5. **Standalone execution** - Can run independently of Sniper simulation

This makes it suitable for thermal-only studies or integration with other simulation frameworks. 