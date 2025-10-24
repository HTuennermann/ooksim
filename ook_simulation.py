#!/usr/bin/env python3
"""
On-Off Keying (OOK) Optical Communication Simulation Framework

This script simulates the complete chain of optical communication using OOK modulation:
Data Generation → Time Domain E-field → Photodiode Detection → Data Recovery

Author: Claude Code Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, butter, filtfilt
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import argparse


@dataclass
class SimulationConfig:
    """Configuration class for OOK simulation parameters"""
    # Data parameters
    bit_rate: float = 10e9  # Data rate in bits per second (e.g., 10 Gbit/s)
    num_bits: int = 100     # Number of bits to simulate
    oversampling_rate: int = 16  # Samples per bit period

    # Modulation parameters
    pulse_shape: str = 'gaussian'  # 'gaussian', 'rectangular', 'raised_cosine'
    pulse_width: float = 0.8  # Pulse width as fraction of bit period

    # Channel parameters
    noise_power: float = 0.05  # Noise level (AWGN)
    use_fading: bool = False   # Enable/disable fading effects

    # Detection parameters
    decision_threshold: Optional[float] = None  # Auto-calculated if None

    # Visualization
    show_plots: bool = True
    save_plots: bool = False

    # Random seed for reproducibility
    seed: int = 42


class OOKSimulator:
    """Main OOK simulation class"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        np.random.seed(config.seed)

        # Derived parameters
        self.bit_duration = 1 / config.bit_rate
        self.sample_rate = config.bit_rate * config.oversampling_rate
        self.total_time = config.num_bits * self.bit_duration
        self.time = np.linspace(
            0, self.total_time,
            config.num_bits * config.oversampling_rate,
            endpoint=False
        )

        # Storage for simulation results
        self.data_bits = None
        self.nr_z_signal = None
        self.e_field = None
        self.optical_power = None
        self.received_signal = None
        self.recovered_data = None
        self.ber = None

    def generate_data(self) -> np.ndarray:
        """Generate random binary data"""
        self.data_bits = np.random.randint(0, 2, self.config.num_bits)
        print(f"Generated {self.config.num_bits} bits: {self.data_bits[:20]}{'...' if self.config.num_bits > 20 else ''}")
        return self.data_bits

    def create_pulse_shape(self) -> np.ndarray:
        """Create pulse shaping filter"""
        samples_per_bit = self.config.oversampling_rate

        if self.config.pulse_shape == 'rectangular':
            return np.ones(samples_per_bit)

        elif self.config.pulse_shape == 'gaussian':
            std = samples_per_bit / 4
            pulse = windows.gaussian(samples_per_bit, std=std)
            return pulse / np.sum(pulse)

        elif self.config.pulse_shape == 'raised_cosine':
            # Simplified raised cosine pulse
            beta = 0.3  # Roll-off factor
            t = np.linspace(-2, 2, samples_per_bit)
            pulse = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
            pulse = np.nan_to_num(pulse)
            return pulse / np.sum(pulse)

        else:
            raise ValueError(f"Unknown pulse shape: {self.config.pulse_shape}")

    def modulate_signal(self, data_bits: np.ndarray) -> np.ndarray:
        """Modulate data bits into electric field amplitude"""
        # Create NRZ signal
        self.nr_z_signal = np.repeat(data_bits, self.config.oversampling_rate)

        # Apply pulse shaping
        pulse_shape = self.create_pulse_shape()
        e_field_amplitude = np.convolve(self.nr_z_signal, pulse_shape, mode='same')

        # Apply pulse width scaling
        e_field_amplitude *= self.config.pulse_width

        # Create complex E-field (real for OOK)
        self.e_field = e_field_amplitude * np.exp(1j * 0)

        return self.e_field

    def add_channel_effects(self, e_field: np.ndarray) -> np.ndarray:
        """Add channel impairments to the signal"""
        # Convert to optical power
        self.optical_power = np.abs(e_field)**2

        # Add fading if enabled
        if self.config.use_fading:
            # Simple Rayleigh fading
            fading = np.random.rayleigh(1.0, len(self.optical_power))
            self.optical_power *= fading

        # Add AWGN noise
        noise = np.random.normal(0, np.sqrt(self.config.noise_power), len(self.optical_power))
        self.received_signal = self.optical_power + noise

        return self.received_signal

    def detect_and_decode(self, received_signal: np.ndarray) -> np.ndarray:
        """Detect and decode the received signal"""
        # Sample at optimal decision points (middle of each bit period)
        sample_points = np.arange(
            int(self.config.oversampling_rate/2),
            len(received_signal),
            self.config.oversampling_rate
        )

        sampled_values = received_signal[sample_points]

        # Set decision threshold
        if self.config.decision_threshold is None:
            # Use automatic threshold based on signal statistics
            threshold = 0.5 * np.max(self.optical_power)
        else:
            threshold = self.config.decision_threshold

        # Make decisions
        self.recovered_data = (sampled_values > threshold).astype(int)

        return self.recovered_data

    def calculate_ber(self) -> float:
        """Calculate Bit Error Rate"""
        if self.data_bits is None or self.recovered_data is None:
            raise ValueError("Must run simulation first")

        num_errors = np.sum(self.data_bits != self.recovered_data)
        self.ber = num_errors / self.config.num_bits

        print(f"\n--- Simulation Results ---")
        print(f"Bits simulated: {self.config.num_bits}")
        print(f"Bit errors: {num_errors}")
        print(f"Bit Error Rate (BER): {self.ber:.6f}")

        return self.ber

    def plot_results(self, save_path: Optional[str] = None):
        """Plot simulation results"""
        if self.received_signal is None:
            raise ValueError("Must run simulation first")

        fig, axs = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle(
            f'OOK Optical Communication Simulation\n'
            f'Bit Rate: {self.config.bit_rate/1e9:.1f} Gbit/s, '
            f'BER: {self.ber:.2e}, '
            f'Pulse Shape: {self.config.pulse_shape}',
            fontsize=14
        )

        # Plot 1: Original Data and NRZ Signal
        axs[0].plot(self.time, self.nr_z_signal, 'b-', linewidth=1, alpha=0.7, label='NRZ Signal')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Transmitter: Original Data and NRZ Modulation')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Plot 2: E-field Amplitude
        axs[1].plot(self.time, np.real(self.e_field), 'g-', linewidth=1.5, label='E-field Amplitude')
        axs[1].set_ylabel('Field Amplitude')
        axs[1].set_title('Transmitter: Modulated Electric Field')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # Plot 3: Received Signal with Noise
        sample_points = np.arange(
            int(self.config.oversampling_rate/2),
            len(self.received_signal),
            self.config.oversampling_rate
        )
        threshold = (self.config.decision_threshold if self.config.decision_threshold is not None
                    else 0.5 * np.max(self.optical_power))

        axs[2].plot(self.time, self.received_signal, 'r-', linewidth=1, alpha=0.8, label='Received + Noise')
        axs[2].axhline(threshold, color='k', linestyle='--', linewidth=2, label='Decision Threshold')
        axs[2].plot(self.time[sample_points], self.received_signal[sample_points],
                   'mo', markersize=4, label='Sample Points')
        axs[2].set_ylabel('Power (a.u.)')
        axs[2].set_title('Receiver: Photodiode Signal with Noise')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        # Plot 4: Original vs Recovered Data
        bit_indices = np.arange(self.config.num_bits)
        axs[3].step(bit_indices, self.data_bits, 'bo-', where='post',
                   linewidth=2, markersize=6, label='Original Data')
        axs[3].step(bit_indices, self.recovered_data, 'r^--', where='post',
                   linewidth=2, markersize=6, label='Recovered Data')
        axs[3].set_xlabel('Bit Index')
        axs[3].set_ylabel('Bit Value')
        axs[3].set_title(f'Data Recovery (BER = {self.ber:.2e})')
        axs[3].set_yticks([0, 1])
        axs[3].set_ylim(-0.1, 1.1)
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        if self.config.show_plots:
            plt.show()
        else:
            plt.close()

    def run_simulation(self) -> Tuple[np.ndarray, float]:
        """Run complete simulation pipeline"""
        print("Starting OOK simulation...")
        print(f"Configuration: {self.config}")

        # Step 1: Generate data
        data = self.generate_data()

        # Step 2: Modulate signal
        e_field = self.modulate_signal(data)

        # Step 3: Add channel effects
        received = self.add_channel_effects(e_field)

        # Step 4: Detect and decode
        recovered = self.detect_and_decode(received)

        # Step 5: Calculate performance metrics
        ber = self.calculate_ber()

        # Step 6: Visualize results
        if self.config.show_plots or self.config.save_plots:
            save_path = 'ook_simulation_results.png' if self.config.save_plots else None
            self.plot_results(save_path)

        return recovered, ber


def load_config_from_file(config_path: str) -> SimulationConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return SimulationConfig(**config_dict)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return SimulationConfig()


def create_default_config():
    """Create a default configuration file"""
    config = SimulationConfig()
    config_dict = {
        'bit_rate': config.bit_rate,
        'num_bits': config.num_bits,
        'oversampling_rate': config.oversampling_rate,
        'pulse_shape': config.pulse_shape,
        'pulse_width': config.pulse_width,
        'noise_power': config.noise_power,
        'use_fading': config.use_fading,
        'show_plots': config.show_plots,
        'save_plots': config.save_plots,
        'seed': config.seed
    }

    with open('ook_config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print("Default configuration saved to 'ook_config.yaml'")


def run_ber_sweep():
    """Run BER vs SNR sweep"""
    print("Running BER sweep...")

    snr_db_range = np.arange(-5, 21, 2)
    ber_results = []

    for snr_db in snr_db_range:
        snr_linear = 10**(snr_db/10)
        noise_power = 1.0 / snr_linear

        config = SimulationConfig(
            num_bits=1000,
            noise_power=noise_power,
            show_plots=False,
            seed=42
        )

        simulator = OOKSimulator(config)
        _, ber = simulator.run_simulation()
        ber_results.append(ber)

        print(f"SNR: {snr_db:2d} dB, BER: {ber:.2e}")

    # Plot BER curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ber_results, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('OOK System Performance: BER vs SNR')
    plt.grid(True, which='both', alpha=0.3)
    plt.ylim([1e-6, 1])

    # Save BER curve
    plt.savefig('ook_ber_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    return snr_db_range, ber_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='OOK Optical Communication Simulation')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    parser.add_argument('--ber-sweep', action='store_true', help='Run BER vs SNR sweep')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot display')
    parser.add_argument('--save', action='store_true', help='Save plots to files')

    args = parser.parse_args()

    if args.create_config:
        create_default_config()
        return

    if args.ber_sweep:
        run_ber_sweep()
        return

    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = SimulationConfig()

    # Override config with command line arguments
    if args.no_plots:
        config.show_plots = False
    if args.save:
        config.save_plots = True

    # Run simulation
    simulator = OOKSimulator(config)
    recovered_data, ber = simulator.run_simulation()

    print(f"\nSimulation completed successfully!")
    print(f"Final BER: {ber:.2e}")


if __name__ == "__main__":
    main()