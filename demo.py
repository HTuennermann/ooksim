#!/usr/bin/env python3
"""
Quick demo of the OOK simulation framework
Based on the original code you provided, but enhanced with modern Python features
"""

import numpy as np
import matplotlib.pyplot as plt
from ook_simulation import OOKSimulator, SimulationConfig

def main():
    print("🚀 OOK Optical Communication Simulation Demo")
    print("=" * 50)

    # Configure simulation similar to your original example
    config = SimulationConfig(
        bit_rate=10e9,      # 10 Gbit/s data rate
        num_bits=20,         # 20 bits (like your original)
        oversampling_rate=16,
        pulse_shape='gaussian',
        noise_power=0.05,
        show_plots=True,
        save_plots=False,
        seed=42
    )

    # Create and run simulation
    print("📡 Running simulation...")
    simulator = OOKSimulator(config)
    recovered_data, ber = simulator.run_simulation()

    # Display results
    print(f"\n📊 Results:")
    print(f"   Original data (first 20): {simulator.data_bits}")
    print(f"   Recovered data (first 20): {recovered_data}")
    print(f"   Bit Error Rate: {ber:.4f}")
    print(f"   Errors: {np.sum(simulator.data_bits != recovered_data)} out of {config.num_bits}")

    if ber == 0:
        print("   🎉 Perfect recovery!")
    elif ber < 0.1:
        print("   ✅ Good performance")
    else:
        print("   ⚠️  High error rate - consider reducing noise")

    print(f"\n💡 Try these variations:")
    print(f"   • Change noise_power: 0.01 (better) to 0.1 (worse)")
    print(f"   • Try different pulse_shape: 'rectangular', 'raised_cosine'")
    print(f"   • Increase bit_rate to 40e9 for high-speed simulation")
    print(f"   • Run 'uv run ook_simulation.py --ber-sweep' for SNR analysis")

if __name__ == "__main__":
    main()