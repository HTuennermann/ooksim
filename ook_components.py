"""
Modular OOK Communication System Components

This module provides individual components for the OOK communication chain:
- Transmitter: Data generation and modulation
- Channel: Signal propagation and noise
- Receiver: Detection and demodulation
"""

import numpy as np
from scipy.signal import windows, butter, filtfilt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


class PulseShaper(ABC):
    """Abstract base class for pulse shaping filters"""

    @abstractmethod
    def shape(self, signal: np.ndarray) -> np.ndarray:
        pass


class GaussianPulseShaper(PulseShaper):
    """Gaussian pulse shaping filter"""

    def __init__(self, samples_per_bit: int, std_ratio: float = 0.25):
        self.samples_per_bit = samples_per_bit
        self.std = samples_per_bit * std_ratio
        self._create_filter()

    def _create_filter(self):
        """Create Gaussian filter kernel"""
        self.filter_kernel = windows.gaussian(self.samples_per_bit, std=self.std)
        self.filter_kernel /= np.sum(self.filter_kernel)

    def shape(self, signal: np.ndarray) -> np.ndarray:
        """Apply Gaussian pulse shaping"""
        return np.convolve(signal, self.filter_kernel, mode='same')


class RectangularPulseShaper(PulseShaper):
    """Rectangular (NRZ) pulse shaping"""

    def __init__(self, samples_per_bit: int):
        self.samples_per_bit = samples_per_bit
        self.filter_kernel = np.ones(samples_per_bit)

    def shape(self, signal: np.ndarray) -> np.ndarray:
        """Apply rectangular pulse shaping"""
        return np.convolve(signal, self.filter_kernel, mode='same')


class RaisedCosinePulseShaper(PulseShaper):
    """Raised cosine pulse shaping filter"""

    def __init__(self, samples_per_bit: int, rolloff: float = 0.3):
        self.samples_per_bit = samples_per_bit
        self.rolloff = rolloff
        self._create_filter()

    def _create_filter(self):
        """Create raised cosine filter kernel"""
        t = np.linspace(-2, 2, self.samples_per_bit)
        self.filter_kernel = np.sinc(t) * np.cos(np.pi * self.rolloff * t)
        self.filter_kernel /= (1 - (2 * self.rolloff * t)**2)
        self.filter_kernel = np.nan_to_num(self.filter_kernel)
        self.filter_kernel /= np.sum(self.filter_kernel)

    def shape(self, signal: np.ndarray) -> np.ndarray:
        """Apply raised cosine pulse shaping"""
        return np.convolve(signal, self.filter_kernel, mode='same')


class OOKTransmitter:
    """OOK Transmitter component"""

    def __init__(self, samples_per_bit: int, pulse_shaper: PulseShaper):
        self.samples_per_bit = samples_per_bit
        self.pulse_shaper = pulse_shaper

    def generate_data(self, num_bits: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate random binary data"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, 2, num_bits)

    def create_nrz_signal(self, data_bits: np.ndarray) -> np.ndarray:
        """Create NRZ (Non-Return-to-Zero) signal from binary data"""
        return np.repeat(data_bits, self.samples_per_bit)

    def modulate(self, data_bits: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        Modulate binary data into electric field amplitude

        Args:
            data_bits: Binary data sequence
            amplitude: Peak amplitude of the electric field

        Returns:
            Complex electric field signal
        """
        # Create NRZ signal
        nrz_signal = self.create_nrz_signal(data_bits)

        # Apply pulse shaping
        e_field_amplitude = self.pulse_shaper.shape(nrz_signal) * amplitude

        # Create complex field (real for OOK)
        e_field = e_field_amplitude * np.exp(1j * 0)

        return e_field


class OpticalChannel:
    """Optical channel model with impairments"""

    def __init__(self, noise_power: float = 0.05, use_fading: bool = False,
                 use_dispersion: bool = False):
        self.noise_power = noise_power
        self.use_fading = use_fading
        self.use_dispersion = use_dispersion

    def propagate(self, e_field: np.ndarray) -> np.ndarray:
        """
        Propagate signal through optical channel with impairments

        Args:
            e_field: Input electric field

        Returns:
            Received optical power with impairments
        """
        # Convert to optical power (|E|^2)
        optical_power = np.abs(e_field)**2

        # Apply fading if enabled
        if self.use_fading:
            fading = np.random.rayleigh(1.0, len(optical_power))
            optical_power *= fading

        # Apply dispersion if enabled (simplified)
        if self.use_dispersion:
            # Add inter-symbol interference with simple filtering
            b, a = butter(2, 0.3)
            optical_power = filtfilt(b, a, optical_power)

        # Add AWGN noise
        noise = np.random.normal(0, np.sqrt(self.noise_power), len(optical_power))
        received_signal = optical_power + noise

        return received_signal


class OOKReceiver:
    """OOK Receiver with detection and decision logic"""

    def __init__(self, samples_per_bit: int, decision_threshold: Optional[float] = None):
        self.samples_per_bit = samples_per_bit
        self.decision_threshold = decision_threshold

    def sample_signal(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Sample received signal at optimal decision points

        Args:
            received_signal: Received signal from photodiode

        Returns:
            Sampled values at middle of each bit period
        """
        sample_points = np.arange(
            int(self.samples_per_bit/2),
            len(received_signal),
            self.samples_per_bit
        )
        return received_signal[sample_points], sample_points

    def detect_data(self, sampled_values: np.ndarray,
                   signal_max: Optional[float] = None) -> np.ndarray:
        """
        Detect binary data from sampled values

        Args:
            sampled_values: Values sampled at decision points
            signal_max: Maximum signal level for threshold calculation

        Returns:
            Detected binary data
        """
        # Calculate decision threshold
        if self.decision_threshold is None:
            if signal_max is None:
                signal_max = np.max(sampled_values)
            threshold = 0.5 * signal_max
        else:
            threshold = self.decision_threshold

        # Make hard decisions
        detected_data = (sampled_values > threshold).astype(int)

        return detected_data, threshold


class OOKSystem:
    """Complete OOK communication system"""

    def __init__(self, transmitter: OOKTransmitter, channel: OpticalChannel,
                 receiver: OOKReceiver):
        self.transmitter = transmitter
        self.channel = channel
        self.receiver = receiver

    def transmit_and_receive(self, num_bits: int, seed: Optional[int] = None,
                            amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray,
                                                            np.ndarray, float]:
        """
        Run complete transmission chain

        Returns:
            Tuple of (original_data, transmitted_field, received_signal,
                     sampled_values, detected_data, ber)
        """
        # Generate data
        original_data = self.transmitter.generate_data(num_bits, seed)

        # Modulate
        transmitted_field = self.transmitter.modulate(original_data, amplitude)

        # Propagate through channel
        received_signal = self.channel.propagate(transmitted_field)

        # Sample and detect
        sampled_values, sample_points = self.receiver.sample_signal(received_signal)
        detected_data, threshold = self.receiver.detect_data(
            sampled_values, np.max(np.abs(transmitted_field)**2)
        )

        # Calculate BER
        ber = np.sum(original_data != detected_data) / num_bits

        return (original_data, transmitted_field, received_signal,
                sampled_values, detected_data, ber)


def create_system_from_config(config) -> OOKSystem:
    """Factory function to create OOK system from configuration"""
    # Create pulse shaper based on configuration
    if config.pulse_shape == 'gaussian':
        pulse_shaper = GaussianPulseShaper(config.oversampling_rate)
    elif config.pulse_shape == 'rectangular':
        pulse_shaper = RectangularPulseShaper(config.oversampling_rate)
    elif config.pulse_shape == 'raised_cosine':
        pulse_shaper = RaisedCosinePulseShaper(config.oversampling_rate)
    else:
        raise ValueError(f"Unknown pulse shape: {config.pulse_shape}")

    # Create components
    transmitter = OOKTransmitter(config.oversampling_rate, pulse_shaper)
    channel = OpticalChannel(
        noise_power=config.noise_power,
        use_fading=config.use_fading
    )
    receiver = OOKReceiver(config.oversampling_rate, config.decision_threshold)

    # Create system
    system = OOKSystem(transmitter, channel, receiver)
    return system