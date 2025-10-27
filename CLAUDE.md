# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python framework for simulating On-Off Keying (OOK) optical communication systems. The simulation follows the complete signal chain: binary data → NRZ signal → pulse shaping → electric field modulation → optical power conversion → channel impairments → photodiode detection → data recovery.

## Architecture

The codebase follows a **dual-layer architecture**:

1. **High-Level Layer** (`ook_simulation.py`):
   - `OOKSimulator` class provides a monolithic approach
   - `SimulationConfig` dataclass for parameter management
   - Complete pipeline in a single class for quick simulations

2. **Modular Layer** (`ook_components.py`):
   - Separate components: `OOKTransmitter`, `OpticalChannel`, `OOKReceiver`
   - Abstract `PulseShaper` base class with three implementations
   - `OOKSystem` class that integrates all components

### Signal Flow
`data_bits` → `nrz_signal` → `e_field` → `optical_power` → `received_signal` → `recovered_data`

## Common Development Commands

### Environment Setup
```bash
uv install  # Install dependencies (numpy, matplotlib, scipy, pyyaml)
```

### Running Simulations
```bash
# Basic simulation with default parameters
uv run ook_simulation.py

# Use custom configuration
uv run ook_simulation.py --config ook_config.yaml

# Generate default configuration file
uv run ook_simulation.py --create-config

# Run BER vs SNR analysis
uv run ook_simulation.py --ber-sweep

# Save plots without displaying
uv run ook_simulation.py --save --no-plots
```

### Examples and Testing
```bash
# Run comprehensive examples (6 different scenarios)
uv run example_usage.py

# Quick demo with basic parameters
uv run demo.py
```

## Configuration System

The framework uses a **hierarchical configuration**:
- Python `SimulationConfig` dataclass as primary
- YAML file support for external configuration
- Command-line argument overrides
- Factory function `create_system_from_config()` for component assembly

### Critical Implementation Details

- **Signal flow**: Binary → NRZ → Pulse shaping → E-field → |E|² → Channel → Photodiode → Sampling → Decision
- **Three pulse shapers**: Gaussian, Rectangular, Raised Cosine
- **Channel impairments**: AWGN noise, Rayleigh fading, chromatic dispersion
- **BER calculation**: Simple bit-by-bit comparison
- **Visualization**: 4-panel plots showing all signal stages

## Component-Based Development

### Adding New Pulse Shapers
```python
class CustomPulseShaper(PulseShaper):
    def __init__(self, samples_per_bit, custom_param):
        self.samples_per_bit = samples_per_bit
        self.custom_param = custom_param
        self._create_filter()

    def _create_filter(self):
        # Implement custom filter kernel
        pass

    def shape(self, signal):
        return np.convolve(signal, self.filter_kernel, mode='same')
```

### Component Integration
```python
# Create individual components
pulse_shaper = GaussianPulseShaper(samples_per_bit=16, std_ratio=0.25)
transmitter = OOKTransmitter(16, pulse_shaper)
channel = OpticalChannel(noise_power=0.03, use_fading=True)
receiver = OOKReceiver(16)

# Assemble complete system
system = OOKSystem(transmitter, channel, receiver)
original_data, transmitted_field, received_signal, sampled_values, detected_data, ber = system.transmit_and_receive(
    num_bits=1000, seed=42, amplitude=1.0
)
```

## Performance Analysis Patterns

### BER vs SNR Sweeps
```python
noise_powers = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
for noise_power in noise_powers:
    config = SimulationConfig(noise_power=noise_power, show_plots=False)
    simulator = OOKSimulator(config)
    _, ber = simulator.run_simulation()
    # Store and plot results
```

## Key Implementation Details

### Signal Representation
- **Electric Field**: Complex array (real for OOK, imaginary for future phase modulation)
- **Optical Power**: `|E|²` (intensity detected by photodiode)
- **NRZ Signal**: Binary upsampled signal before pulse shaping
- **Sampling Points**: Middle of each bit period (`oversampling_rate/2` offset)

### Noise and Impairments
- **AWGN**: `np.random.normal(0, sqrt(noise_power), length)`
- **Rayleigh Fading**: `np.random.rayleigh(1.0, length)` (if enabled)
- **Dispersion**: Butterworth filter `filtfilt(b, a, signal)` (if enabled)

### BER Calculation
```python
ber = np.sum(original_data != recovered_data) / num_bits
```

## Dependencies and Environment

- **Python 3.8+** required
- **NumPy**: Numerical computations and signal processing
- **Matplotlib**: Visualization and plotting
- **SciPy**: Signal processing functions (`windows`, `butter`, `filtfilt`)
- **PyYAML**: Configuration file support

The project uses `uv` for dependency management, creating an isolated virtual environment in `.venv/`.