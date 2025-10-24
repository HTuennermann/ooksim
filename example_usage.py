"""
Example usage of the OOK simulation framework

This script demonstrates various ways to use the OOK simulation components:
1. Basic simulation using the main script
2. Component-based simulation
3. BER performance analysis
4. Custom configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from ook_simulation import OOKSimulator, SimulationConfig
from ook_components import (
    OOKSystem, OOKTransmitter, OpticalChannel, OOKReceiver,
    GaussianPulseShaper, RectangularPulseShaper, RaisedCosinePulseShaper,
    create_system_from_config
)


def example_1_basic_simulation():
    """Example 1: Basic OOK simulation with default parameters"""
    print("=" * 60)
    print("Example 1: Basic OOK Simulation")
    print("=" * 60)

    config = SimulationConfig(
        bit_rate=10e9,      # 10 Gbit/s
        num_bits=50,        # Simulate 50 bits
        oversampling_rate=16,
        pulse_shape='gaussian',
        noise_power=0.05,
        seed=42
    )

    simulator = OOKSimulator(config)
    recovered_data, ber = simulator.run_simulation()

    print(f"Results: BER = {ber:.4f}")


def example_2_component_based():
    """Example 2: Using individual components"""
    print("\n" + "=" * 60)
    print("Example 2: Component-Based Simulation")
    print("=" * 60)

    # Create pulse shaper
    pulse_shaper = GaussianPulseShaper(samples_per_bit=16)

    # Create components
    transmitter = OOKTransmitter(samples_per_bit=16, pulse_shaper=pulse_shaper)
    channel = OpticalChannel(noise_power=0.03, use_fading=False)
    receiver = OOKReceiver(samples_per_bit=16)

    # Create system
    system = OOKSystem(transmitter, channel, receiver)

    # Run simulation
    (original_data, transmitted_field, received_signal,
     sampled_values, detected_data, ber) = system.transmit_and_receive(
        num_bits=100, seed=42, amplitude=1.0
    )

    print(f"Original data (first 20 bits): {original_data[:20]}")
    print(f"Detected data (first 20 bits): {detected_data[:20]}")
    print(f"BER: {ber:.6f}")


def example_3_pulse_shape_comparison():
    """Example 3: Compare different pulse shaping techniques"""
    print("\n" + "=" * 60)
    print("Example 3: Pulse Shape Comparison")
    print("=" * 60)

    pulse_shapes = ['rectangular', 'gaussian', 'raised_cosine']
    results = {}

    for shape in pulse_shapes:
        config = SimulationConfig(
            bit_rate=5e9,
            num_bits=200,
            oversampling_rate=20,
            pulse_shape=shape,
            noise_power=0.02,
            show_plots=False,
            seed=42
        )

        simulator = OOKSimulator(config)
        _, ber = simulator.run_simulation()
        results[shape] = ber

        print(f"{shape:15s}: BER = {ber:.6f}")

    # Plot comparison
    plt.figure(figsize=(10, 6))
    shapes = list(results.keys())
    bers = list(results.values())

    plt.bar(shapes, bers)
    plt.yscale('log')
    plt.ylabel('Bit Error Rate (BER)')
    plt.xlabel('Pulse Shape')
    plt.title('BER Comparison: Different Pulse Shaping Techniques')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('pulse_shape_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def example_4_snr_analysis():
    """Example 4: SNR vs BER analysis"""
    print("\n" + "=" * 60)
    print("Example 4: SNR vs BER Analysis")
    print("=" * 60)

    # Test different noise levels (SNR values)
    noise_powers = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    snr_db = [10 * np.log10(1.0 / noise) for noise in noise_powers]
    ber_results = []

    print("SNR (dB)   Noise Power    BER")
    print("-" * 35)

    for noise_power in noise_powers:
        config = SimulationConfig(
            num_bits=500,
            oversampling_rate=16,
            pulse_shape='gaussian',
            noise_power=noise_power,
            show_plots=False,
            seed=42
        )

        simulator = OOKSimulator(config)
        _, ber = simulator.run_simulation()
        ber_results.append(ber)

        current_snr = 10 * np.log10(1.0 / noise_power)
        print(f"{current_snr:8.1f}   {noise_power:11.4f}   {ber:.2e}")

    # Plot BER curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db, ber_results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('OOK System Performance: BER vs SNR')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim([min(snr_db) - 1, max(snr_db) + 1])
    plt.ylim([1e-5, 1])

    # Add theoretical BER curve for comparison (OOK with AWGN)
    snr_linear = 10**(np.array(snr_db)/10)
    theoretical_ber = 0.5 * np.exp(-snr_linear/2)
    plt.semilogy(snr_db, theoretical_ber, 'r--', linewidth=2,
                label='Theoretical (OOK + AWGN)', alpha=0.7)
    plt.legend()

    plt.savefig('ook_snr_ber_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def example_5_high_speed_simulation():
    """Example 5: High-speed simulation (40 Gbit/s)"""
    print("\n" + "=" * 60)
    print("Example 5: High-Speed Simulation (40 Gbit/s)")
    print("=" * 60)

    config = SimulationConfig(
        bit_rate=40e9,      # 40 Gbit/s
        num_bits=100,
        oversampling_rate=32,  # Higher oversampling for high speed
        pulse_shape='gaussian',
        noise_power=0.08,
        pulse_width=0.7,     # Narrower pulses for high speed
        seed=42
    )

    simulator = OOKSimulator(config)
    recovered_data, ber = simulator.run_simulation()

    print(f"40 Gbit/s simulation completed")
    print(f"BER: {ber:.6f}")


def example_6_custom_pulse_shape():
    """Example 6: Custom pulse shaping configuration"""
    print("\n" + "=" * 60)
    print("Example 6: Custom Pulse Shape Configuration")
    print("=" * 60)

    # Create custom Gaussian pulse shaper with specific parameters
    custom_shaper = GaussianPulseShaper(samples_per_bit=20, std_ratio=0.15)

    # Create system with custom components
    transmitter = OOKTransmitter(samples_per_bit=20, pulse_shaper=custom_shaper)
    channel = OpticalChannel(noise_power=0.025, use_fading=True)  # Enable fading
    receiver = OOKReceiver(samples_per_bit=20)

    system = OOKSystem(transmitter, channel, receiver)

    # Run simulation
    (original_data, transmitted_field, received_signal,
     sampled_values, detected_data, ber) = system.transmit_and_receive(
        num_bits=150, seed=42, amplitude=1.2
    )

    print(f"Custom pulse shape simulation completed")
    print(f"Original data (first 15): {original_data[:15]}")
    print(f"Detected data (first 15): {detected_data[:15]}")
    print(f"BER: {ber:.6f}")


def main():
    """Run all examples"""
    print("OOK Simulation Framework - Example Usage")
    print("=========================================")

    try:
        example_1_basic_simulation()
        example_2_component_based()
        example_3_pulse_shape_comparison()
        example_4_snr_analysis()
        example_5_high_speed_simulation()
        example_6_custom_pulse_shape()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the generated plots for visualization results.")
        print("=" * 60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()