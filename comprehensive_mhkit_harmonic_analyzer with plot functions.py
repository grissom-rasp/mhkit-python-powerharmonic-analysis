"""
Comprehensive MHKiT-Python Analysis with plot functions for Oak Ridge Harmonic Signals Dataset

This script provides a complete analysis framework for HDF5 harmonic datasets using MHKiT-Python,
including all the visualization types mentioned in the documentation and comprehensive metrics.

REQUIREMENTS:
- This script must be run with the mhkit-venv conda environment activated
- Use one of the following methods to run:

1. Using conda_runner.py (recommended):
   python conda_runner.py "comprehensive_mhkit_harmonic_analyzer with plot functions.py"

2. Using run_with_conda.sh:
   ./run_with_conda.sh "comprehensive_mhkit_harmonic_analyzer with plot functions.py"

3. Manual activation:
   conda activate mhkit-venv
   python "comprehensive_mhkit_harmonic_analyzer with plot functions.py"
   conda deactivate

SETUP:
- Run setup_conda_environment.py to create the required conda environment
- This will install MHKiT-Python and all required dependencies

Author: Generated for Power Harmonic Analysis Project
Date: 2025
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime, timedelta
from typing import Union
import warnings
warnings.filterwarnings('ignore')

# MHKiT modules
from mhkit import power, qc, utils

# MHKiT dolfyn FFT tools
try:
    from mhkit.dolfyn.tools.fft import psd_1D, fft_frequency
    MHKIT_DOLFYN_AVAILABLE = True
    print("MHKiT dolfyn FFT tools available")
except ImportError:
    MHKIT_DOLFYN_AVAILABLE = False
    print("Warning: MHKiT dolfyn FFT tools not available. Using scipy/numpy FFT.")

# Scientific computing - using numpy for FFT and basic signal processing
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

# GPU acceleration with CuPy
try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    CUPY_AVAILABLE = True
    print("CuPy GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Using CPU-only processing.")

# Advanced signal processing
try:
    from pywt import cwt as pywt_cwt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets not available. CWT analysis will use scipy implementation.")

# Plotting enhancements
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MHKiTHarmonicAnalyzer:
    """
    Comprehensive harmonic analysis class using MHKiT-Python for HDF5 datasets
    with GPU acceleration support
    """

    def __init__(self, h5_file_path, sampling_rate=800000, use_gpu=True):
        """
        Initialize the analyzer with H5 file path and sampling parameters

        Parameters:
        h5_file_path (str): Path to the HDF5 file
        sampling_rate (float): Sampling rate in Hz (default: 800kHz for Oak Ridge dataset)
        use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.h5_file_path = h5_file_path
        self.sampling_rate = sampling_rate
        self.grid_freq = 60  # Grid frequency for harmonic analysis (should be 50 or 60)
        self.datasets = {}
        self.analysis_results = {}
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        # Performance optimization settings
        self.max_samples_per_load = 50000000 if self.use_gpu else 10000000  # Increase for GPU
        self.chunk_size = 1000000  # Process data in chunks for memory efficiency
        self.downsample_factor = 1  # For visualization efficiency

        if self.use_gpu:
            print(f"GPU acceleration enabled. Using CuPy on device: {cp.cuda.Device()}")
            # Pre-allocate GPU memory pool for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else:
            print("Using CPU-only processing")

        # Load the H5 file structure
        self._load_h5_structure()
        
    def _load_h5_structure(self):
        """Load and inspect the H5 file structure"""
        with h5py.File(self.h5_file_path, 'r') as f:
            print("H5 File Structure:")
            self._print_h5_structure(f, indent=0)
            self.h5_file = f
            
    def _print_h5_structure(self, item, indent=0):
        """Recursively print H5 file structure"""
        for key in item.keys():
            if isinstance(item[key], h5py.Group):
                print("  " * indent + f"Group: {key}")
                self._print_h5_structure(item[key], indent + 1)
            elif isinstance(item[key], h5py.Dataset):
                print("  " * indent + f"Dataset: {key} - Shape: {item[key].shape}, Dtype: {item[key].dtype}")
                
    def load_dataset(self, dataset_name, start_time="2023-01-02 08:05:00", max_samples=None):
        """
        Load a specific dataset from the H5 file and convert to pandas DataFrame
        with chunked loading for memory efficiency
        """
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                if dataset_name not in f:
                    available_datasets = list(f.keys())
                    raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")
                dataset = f[dataset_name]
                total_samples = dataset.shape[0]

                # Determine samples to load
                samples_to_load = min(total_samples, max_samples or self.max_samples_per_load)

                if samples_to_load < total_samples:
                    print(f"Warning: Loading only first {samples_to_load:,} samples out of {total_samples:,} total samples")

                # Chunked loading to reduce memory pressure
                chunk_size = min(self.chunk_size, samples_to_load)
                data_chunks = []

                for start_idx in range(0, samples_to_load, chunk_size):
                    end_idx = min(start_idx + chunk_size, samples_to_load)
                    chunk = dataset[start_idx:end_idx]
                    data_chunks.append(chunk)

                # Concatenate chunks
                data = np.concatenate(data_chunks) if len(data_chunks) > 1 else data_chunks[0]

                num_samples = len(data)
                time_delta = pd.Timedelta(seconds=1/self.sampling_rate)
                time_index = pd.date_range(start=start_time, periods=num_samples, freq=time_delta)
                df = pd.DataFrame({dataset_name: data}, index=time_index)

                # Store data on GPU if available
                if self.use_gpu:
                    gpu_data = cp.asarray(data)
                    df.attrs['gpu_data'] = gpu_data
                    print(f"  GPU memory usage: {gpu_data.nbytes / 1024**3:.2f} GB")

                self.datasets[dataset_name] = df
                print(f"Loaded dataset: {dataset_name}")
                print(f"  Shape: {df.shape}")
                print(f"  Time range: {df.index[0]} to {df.index[-1]}")
                print(f"  Duration: {df.index[-1] - df.index[0]}")
                print(f"  Sampling rate: {self.sampling_rate} Hz")
                print(f"  Data type: {df.iloc[:, 0].dtype}")
                print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                return df
        except FileNotFoundError:
            raise FileNotFoundError(f"H5 file not found: {self.h5_file_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset '{dataset_name}': {str(e)}")
            
    def analyze_harmonics_mhkit(self, dataset_name, rated_current=None, grid_freq=None):
        """
        Perform comprehensive harmonic analysis using MHKiT-Python
        
        Parameters:
        dataset_name (str): Name of the dataset to analyze
        rated_current (float): Rated current for THCD calculation
        grid_freq (float): Grid frequency in Hz (ignored, always 60 Hz)
        
        Returns:
        dict: Dictionary containing all MHKiT analysis results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        df = self.datasets[dataset_name]
        # Use grid_freq as freq, defaulting to 60 Hz if not provided
        freq: Union[float, int] = grid_freq if grid_freq is not None else 60
        print(f"Performing MHKiT harmonic analysis on {dataset_name} (freq={freq})...")
        # 1. Harmonics calculation (IEC 61000-4-7 compliant)
        harmonics = power.quality.harmonics(df, self.sampling_rate, freq)  # signal_data, freq, grid_freq
        # 2. Harmonic subgroups
        harmonic_subgroups = power.quality.harmonic_subgroups(harmonics, freq)  # harmonics, grid_freq
        # 3. Total Harmonic Current Distortion (THCD)
        if rated_current is None:
            rated_current = np.sqrt(np.mean(df.iloc[:, 0]**2))
        thcd = power.quality.total_harmonic_current_distortion(harmonic_subgroups)
        # 4. Interharmonics
        interharmonics = power.quality.interharmonics(df, self.sampling_rate, freq)  # signal_data, freq, grid_freq
        # Store results
        results = {
            'harmonics': harmonics,
            'harmonic_subgroups': harmonic_subgroups,
            'thcd': thcd,
            'interharmonics': interharmonics,
            'rated_current': rated_current,
            'dataset_name': dataset_name
        }
        self.analysis_results[dataset_name] = results
        print(f"Analysis completed for {dataset_name}")
        print(f"  THCD: {thcd:.2f}%")
        print(f"  Number of harmonics detected: {len(harmonics.columns)}")
        return results
        
    def convert_voltage_to_current(self, voltage_data, dataset_name):
        """
        Convert voltage data to current data based on dataset characteristics
        
        Parameters:
        voltage_data (pd.DataFrame): Voltage data in volts
        dataset_name (str): Name of the dataset to determine conversion factor
        
        Returns:
        pd.DataFrame: Current data in amperes
        """
        # Define voltage-to-current conversion factors based on dataset characteristics
        conversion_factors = {
            'main_power_frp2_rogowski_480': 5.0,  # 1 V = 5 Amps (480V power line)
            'transformer_frp2_ct_480': 5.0,       # 1 V = 5 Amps (480V transformer side)
            'transformer_frp2_ct_ground': 5.0,    # 1 V = 5 Amps (ground line)
            'transformer_frp2_rogowski_240': 5.0, # 1 V = 5 Amps (240V transformer side)
            'signal_injection_frp2_ct_120': 5.0,  # 1 V = 5 Amps (120V signal injection)
            'buried_conduit_outside_b_field_unknown': 1.0,  # B-field antenna (no conversion)
            'outlet_strip_frp1_ct_120': 5.0,      # 1 V = 5 Amps (120V outlet)
            'buried_powerline_outside_b_field_13800': 1.0   # B-field antenna (no conversion)
        }
        
        # Get conversion factor for this dataset
        conversion_factor = conversion_factors.get(dataset_name, 5.0)  # Default to 5.0
        
        # Convert voltage to current
        current_data = voltage_data * conversion_factor
        
        # Rename columns to indicate current
        current_columns = {}
        for col in current_data.columns:
            if 'voltage' in col.lower() or 'v' in col.lower():
                current_columns[col] = col.replace('voltage', 'current').replace('V', 'I')
            else:
                current_columns[col] = f"{col}_I"
        
        current_data = current_data.rename(columns=current_columns)
        
        print(f"  Converted voltage to current using factor: 1 V = {conversion_factor} Amps")
        return current_data
    
    def analyze_instantaneous_frequency(self, dataset_name, voltage_data=None, current_data=None):
        """
        Analyze instantaneous frequency using MHKiT power.characteristics.instantaneous_frequency
        Following the MHKiT documentation pattern: https://mhkit-software.github.io/MHKiT/power_example.html
        
        Parameters:
        dataset_name (str): Name of the dataset to analyze
        voltage_data (pd.DataFrame, optional): Voltage data. If None, uses the main dataset
        current_data (pd.DataFrame, optional): Current data. If None, converts from voltage
        
        Returns:
        dict: Dictionary containing instantaneous frequency analysis results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        df = self.datasets[dataset_name]
        
        # Use provided voltage data or default to main dataset
        if voltage_data is None:
            voltage_data = df
        
        # Convert voltage to current if current_data not provided
        if current_data is None:
            current_data = self.convert_voltage_to_current(voltage_data, dataset_name)
        
        print(f"Analyzing instantaneous frequency for {dataset_name}...")
        print(f"  Voltage data shape: {voltage_data.shape}")
        print(f"  Current data shape: {current_data.shape}")
        
        try:
            # Calculate instantaneous frequency using MHKiT (following documentation pattern)
            # Reference: https://mhkit-software.github.io/MHKiT/power_example.html
            instantaneous_freq = power.characteristics.instantaneous_frequency(voltage_data)
            
            # Calculate basic statistics
            freq_values = instantaneous_freq.iloc[:, 0].values
            freq_stats = {
                'mean_frequency': np.mean(freq_values),
                'std_frequency': np.std(freq_values),
                'min_frequency': np.min(freq_values),
                'max_frequency': np.max(freq_values),
                'median_frequency': np.median(freq_values),
                'frequency_range': np.max(freq_values) - np.min(freq_values)
            }
            
            # Calculate frequency deviation from nominal (60 Hz)
            nominal_freq = 60.0
            freq_deviation = freq_values - nominal_freq
            freq_stats.update({
                'mean_deviation': np.mean(freq_deviation),
                'max_deviation': np.max(np.abs(freq_deviation)),
                'rms_deviation': np.sqrt(np.mean(freq_deviation**2))
            })
            
            results = {
                'instantaneous_frequency': instantaneous_freq,
                'frequency_statistics': freq_stats,
                'nominal_frequency': nominal_freq,
                'voltage_data': voltage_data,
                'current_data': current_data,
                'dataset_name': dataset_name,
                'analysis_timestamp': pd.Timestamp.now()
            }
            
            print(f"  ✓ Instantaneous frequency analysis completed")
            print(f"  Mean frequency: {freq_stats['mean_frequency']:.2f} Hz")
            print(f"  Frequency range: {freq_stats['min_frequency']:.2f} - {freq_stats['max_frequency']:.2f} Hz")
            print(f"  Frequency deviation: {freq_stats['mean_deviation']:.2f} ± {freq_stats['std_frequency']:.2f} Hz")
            print(f"  Max deviation from nominal: {freq_stats['max_deviation']:.2f} Hz")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Error in instantaneous frequency analysis: {e}")
            raise Exception(f"Failed to analyze instantaneous frequency for {dataset_name}: {str(e)}")
    
    def calculate_additional_metrics(self, dataset_name):
        """
        Calculate additional power quality metrics with GPU acceleration
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")

        df = self.datasets[dataset_name]
        signal_data = df.iloc[:, 0].values

        if self.use_gpu and 'gpu_data' in df.attrs:
            # Use GPU data if available
            gpu_signal = df.attrs['gpu_data']
            # Basic statistics on GPU
            rms = cp.sqrt(cp.mean(gpu_signal**2)).get()
            peak = cp.max(cp.abs(gpu_signal)).get()
            crest_factor = peak / rms if rms > 0 else 0

            # GPU FFT for frequency domain analysis
            gpu_fft = cufft.fft(gpu_signal)
            gpu_psd = cp.abs(gpu_fft)**2 / len(gpu_signal)
            gpu_freqs = cufft.fftfreq(len(gpu_signal), 1/self.sampling_rate)

            # Only use positive frequencies
            positive_idx = gpu_freqs >= 0
            freqs = gpu_freqs[positive_idx].get()
            psd = gpu_psd[positive_idx].get()

            dominant_freq_idx = cp.argmax(gpu_psd[positive_idx])
            dominant_freq = gpu_freqs[positive_idx][dominant_freq_idx].get()

            # THD calculation on GPU
            gpu_fft_magnitude = cp.abs(gpu_fft)
            gpu_freqs_fft = cufft.fftfreq(len(gpu_signal), 1/self.sampling_rate)

            # Find fundamental frequency
            fundamental_idx = cp.argmax(gpu_fft_magnitude[1:len(gpu_fft_magnitude)//2]) + 1
            fundamental_freq = gpu_freqs_fft[fundamental_idx].get()

            # Calculate THD (simplified)
            harmonic_power = 0
            fundamental_power = gpu_fft_magnitude[fundamental_idx]**2

            for n in range(2, 51):  # Up to 50th harmonic
                harmonic_idx = int(n * fundamental_idx.get())
                if harmonic_idx < len(gpu_fft_magnitude):
                    harmonic_power += gpu_fft_magnitude[harmonic_idx]**2

            thd = cp.sqrt(harmonic_power / fundamental_power).get() * 100 if fundamental_power.get() > 0 else 0

            # Get data back to CPU for compatibility
            fft_frequencies = gpu_freqs_fft.get()
            fft_magnitude = gpu_fft_magnitude.get()

        else:
            # CPU fallback
            # Basic statistics
            rms = np.sqrt(np.mean(signal_data**2))
            peak = np.max(np.abs(signal_data))
            crest_factor = peak / rms if rms > 0 else 0

            # Frequency domain analysis using numpy
            freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)
            fft_result = np.fft.fft(signal_data)
            psd = np.abs(fft_result)**2 / len(signal_data)

            # Only use positive frequencies
            positive_idx = freqs >= 0
            freqs = freqs[positive_idx]
            psd = psd[positive_idx]

            dominant_freq_idx = np.argmax(psd)
            dominant_freq = freqs[dominant_freq_idx]

            # Total harmonic distortion (simplified)
            fft_result = np.fft.fft(signal_data)
            fft_magnitude = np.abs(fft_result)
            freqs_fft = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)

            # Find fundamental frequency
            fundamental_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
            fundamental_freq = freqs_fft[fundamental_idx]

            # Calculate THD (simplified)
            harmonic_power = 0
            fundamental_power = fft_magnitude[fundamental_idx]**2

            for n in range(2, 51):  # Up to 50th harmonic
                harmonic_idx = int(n * fundamental_idx)
                if harmonic_idx < len(fft_magnitude):
                    harmonic_power += fft_magnitude[harmonic_idx]**2

            thd = np.sqrt(harmonic_power / fundamental_power) * 100 if fundamental_power > 0 else 0

            fft_frequencies = freqs_fft

        metrics = {
            'rms': rms,
            'peak': peak,
            'crest_factor': crest_factor,
            'dominant_frequency': dominant_freq,
            'fundamental_frequency': fundamental_freq,
            'thd_simplified': thd,
            'psd_frequencies': freqs,
            'psd_values': psd,
            'fft_frequencies': fft_frequencies,
            'fft_magnitude': fft_magnitude
        }

        return metrics
        
    def calculate_fft_mhkit_dolfyn(self, dataset_name, nfft=1024):
        """
        Calculate FFT using MHKiT dolfyn tools for improved spectral analysis
        
        Parameters:
        dataset_name (str): Name of the dataset to analyze
        nfft (int): Number of points in the FFT (default: 1024)
        
        Returns:
        dict: Dictionary containing FFT results from MHKiT dolfyn
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        if not MHKIT_DOLFYN_AVAILABLE:
            print("MHKiT dolfyn FFT tools not available. Using fallback method.")
            return self._calculate_fft_fallback(dataset_name, nfft)
        
        df = self.datasets[dataset_name]
        signal_data = df.iloc[:, 0].values
        
        print(f"Calculating FFT using MHKiT dolfyn tools for {dataset_name}...")
        print(f"  Signal length: {len(signal_data):,} samples")
        print(f"  FFT size: {nfft}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        
        try:
            # Use MHKiT dolfyn psd_1D function for power spectral density
            psd = psd_1D(signal_data, nfft, self.sampling_rate)
            
            # Generate frequency vector using MHKiT dolfyn
            freqs = fft_frequency(nfft, self.sampling_rate)
            
            # Calculate FFT magnitude for additional analysis
            fft_result = np.fft.fft(signal_data, n=nfft)
            fft_magnitude = np.abs(fft_result)
            
            # Find dominant frequency
            positive_freqs = freqs[freqs >= 0]
            positive_psd = psd[freqs >= 0]
            dominant_freq_idx = np.argmax(positive_psd)
            dominant_freq = positive_freqs[dominant_freq_idx]
            
            # Calculate fundamental frequency (closest to grid frequency)
            grid_freq = self.grid_freq
            fundamental_idx = np.argmin(np.abs(positive_freqs - grid_freq))
            fundamental_freq = positive_freqs[fundamental_idx]
            
            results = {
                'psd': psd,
                'frequencies': freqs,
                'fft_magnitude': fft_magnitude,
                'dominant_frequency': dominant_freq,
                'fundamental_frequency': fundamental_freq,
                'nfft': nfft,
                'sampling_rate': self.sampling_rate,
                'method': 'MHKiT dolfyn'
            }
            
            print(f"  Dominant frequency: {dominant_freq:.2f} Hz")
            print(f"  Fundamental frequency: {fundamental_freq:.2f} Hz")
            print(f"  Max PSD value: {np.max(psd):.2e}")
            
            return results
            
        except Exception as e:
            print(f"Error in MHKiT dolfyn FFT calculation: {e}")
            print("Falling back to standard FFT method...")
            return self._calculate_fft_fallback(dataset_name, nfft)
    
    def _calculate_fft_fallback(self, dataset_name, nfft=1024):
        """
        Fallback FFT calculation using numpy/scipy when MHKiT dolfyn is not available
        """
        df = self.datasets[dataset_name]
        signal_data = df.iloc[:, 0].values
        
        print(f"Using fallback FFT calculation for {dataset_name}...")
        
        # Standard FFT calculation
        fft_result = np.fft.fft(signal_data, n=nfft)
        fft_magnitude = np.abs(fft_result)
        freqs = np.fft.fftfreq(nfft, 1/self.sampling_rate)
        
        # Calculate PSD
        psd = np.abs(fft_result)**2 / nfft
        
        # Find dominant frequency
        positive_freqs = freqs[freqs >= 0]
        positive_psd = psd[freqs >= 0]
        dominant_freq_idx = np.argmax(positive_psd)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        # Calculate fundamental frequency
        grid_freq = self.grid_freq
        fundamental_idx = np.argmin(np.abs(positive_freqs - grid_freq))
        fundamental_freq = positive_freqs[fundamental_idx]
        
        results = {
            'psd': psd,
            'frequencies': freqs,
            'fft_magnitude': fft_magnitude,
            'dominant_frequency': dominant_freq,
            'fundamental_frequency': fundamental_freq,
            'nfft': nfft,
            'sampling_rate': self.sampling_rate,
            'method': 'numpy/scipy fallback'
        }
        
        return results
        
    def plot_time_domain_signal(self, dataset_name, max_duration=10):
        """
        Plot time domain signal with phase markers
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]
        
        plt.figure(figsize=(15, 6))
        plt.plot(plot_df.index, plot_df.iloc[:, 0], 'b-', linewidth=0.8)
        plt.title(f'Time Domain Signal - {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_fft_spectrum_mhkit_dolfyn(self, dataset_name, nfft=1024, max_freq=None):
        """
        Plot FFT spectrum using MHKiT dolfyn tools for enhanced spectral analysis
        
        Parameters:
        dataset_name (str): Name of the dataset to analyze
        nfft (int): Number of points in the FFT (default: 1024)
        max_freq (float): Maximum frequency to display (default: sampling_rate/2)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        # Calculate FFT using MHKiT dolfyn
        fft_results = self.calculate_fft_mhkit_dolfyn(dataset_name, nfft)
        
        psd = fft_results['psd']
        freqs = fft_results['frequencies']
        fft_magnitude = fft_results['fft_magnitude']
        method = fft_results['method']
        
        # Set maximum frequency for display
        if max_freq is None:
            max_freq = self.sampling_rate / 2
        
        # Filter for positive frequencies and within max_freq
        positive_mask = (freqs >= 0) & (freqs <= max_freq)
        positive_freqs = freqs[positive_mask]
        positive_psd = psd[positive_mask]
        positive_magnitude = fft_magnitude[positive_mask]
        
        # Create subplots for comprehensive FFT visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'FFT Spectrum Analysis using {method} - {dataset_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Power Spectral Density (PSD)
        axes[0, 0].plot(positive_freqs, positive_psd, 'b-', linewidth=1)
        axes[0, 0].set_title('Power Spectral Density (PSD)', fontweight='bold')
        axes[0, 0].set_xlabel('Frequency [Hz]')
        axes[0, 0].set_ylabel('PSD [V²/Hz]')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, max_freq)
        
        # Plot 2: FFT Magnitude Spectrum (Linear)
        axes[0, 1].plot(positive_freqs, positive_magnitude, 'r-', linewidth=1)
        axes[0, 1].set_title('FFT Magnitude Spectrum (Linear)', fontweight='bold')
        axes[0, 1].set_xlabel('Frequency [Hz]')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, max_freq)
        
        # Plot 3: FFT Magnitude Spectrum (Log scale)
        axes[1, 0].semilogy(positive_freqs, positive_magnitude + 1e-10, 'g-', linewidth=1)
        axes[1, 0].set_title('FFT Magnitude Spectrum (Log Scale)', fontweight='bold')
        axes[1, 0].set_xlabel('Frequency [Hz]')
        axes[1, 0].set_ylabel('Magnitude (log scale)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, max_freq)
        
        # Plot 4: PSD (Log scale)
        axes[1, 1].semilogy(positive_freqs, positive_psd + 1e-10, 'm-', linewidth=1)
        axes[1, 1].set_title('Power Spectral Density (Log Scale)', fontweight='bold')
        axes[1, 1].set_xlabel('Frequency [Hz]')
        axes[1, 1].set_ylabel('PSD (log scale)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, max_freq)
        
        # Add frequency markers for grid frequency and harmonics
        grid_freq = self.grid_freq
        for ax in axes.flat:
            # Mark fundamental frequency (grid frequency)
            ax.axvline(x=grid_freq, color='red', linestyle='--', alpha=0.7, 
                      label=f'Fundamental ({grid_freq} Hz)')
            
            # Mark first few harmonics
            for n in range(2, 6):  # 2nd to 5th harmonic
                harmonic_freq = n * grid_freq
                if harmonic_freq <= max_freq:
                    ax.axvline(x=harmonic_freq, color='orange', linestyle=':', alpha=0.5,
                              label=f'{n}th Harmonic' if n <= 3 else '')
            
            if ax == axes[0, 0]:  # Only show legend on first subplot
                ax.legend(loc='upper right', fontsize=8)
        
        # Add text box with analysis results
        dominant_freq = fft_results['dominant_frequency']
        fundamental_freq = fft_results['fundamental_frequency']
        max_psd = np.max(positive_psd)
        
        info_text = f"""Analysis Results:
Method: {method}
FFT Size: {nfft}
Sampling Rate: {self.sampling_rate:,} Hz
Dominant Frequency: {dominant_freq:.2f} Hz
Fundamental Frequency: {fundamental_freq:.2f} Hz
Max PSD: {max_psd:.2e}"""
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for info text
        plt.show()
        
        return fft_results
        
    def plot_fft_magnitude_spectrum(self, dataset_name, use_mhkit_dolfyn=True):
        """
        Plot FFT Magnitude Spectrum - Chart Type #3 from documentation
        with GPU acceleration and downsampling for efficiency
        Now supports MHKiT dolfyn FFT tools when available
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")

        # Use MHKiT dolfyn if available and requested
        if use_mhkit_dolfyn and MHKIT_DOLFYN_AVAILABLE:
            print("Using MHKiT dolfyn for FFT magnitude spectrum...")
            fft_results = self.calculate_fft_mhkit_dolfyn(dataset_name, nfft=2048)
            
            freqs = fft_results['frequencies']
            fft_magnitude = fft_results['fft_magnitude']
            method = fft_results['method']
            
            # Filter for positive frequencies
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_magnitude = fft_magnitude[positive_mask]
            
            plt.figure(figsize=(15, 8))
            plt.plot(positive_freqs, 20 * np.log10(positive_magnitude + 1e-10), 'b-', linewidth=1)
            plt.title(f'FFT Magnitude Spectrum ({method}) - {dataset_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude [dB]')
            plt.xlim(0, self.sampling_rate/2)
            plt.grid(True, alpha=0.3)
            
            # Add grid frequency marker
            plt.axvline(x=self.grid_freq, color='red', linestyle='--', alpha=0.7, 
                       label=f'Grid Frequency ({self.grid_freq} Hz)')
            plt.legend()
            plt.tight_layout()
            plt.show()
            return

        # Fallback to original method
        df = self.datasets[dataset_name]

        if self.use_gpu and 'gpu_data' in df.attrs:
            # Use GPU data
            gpu_signal = df.attrs['gpu_data']
            # Downsample for visualization if needed
            downsample_factor = max(1, len(gpu_signal) // 1000000)  # Limit to ~1M points for plotting
            if downsample_factor > 1:
                gpu_signal = gpu_signal[::downsample_factor]

            gpu_fft = cufft.fft(gpu_signal)
            gpu_magnitude = cp.abs(gpu_fft)
            gpu_freqs = cufft.fftfreq(len(gpu_signal), downsample_factor/self.sampling_rate)

            positive_freqs = gpu_freqs[:len(gpu_freqs)//2].get()
            positive_magnitude = gpu_magnitude[:len(gpu_magnitude)//2].get()
        else:
            signal_data = df.iloc[:, 0].values
            # Downsample for visualization if needed
            downsample_factor = max(1, len(signal_data) // 1000000)  # Limit to ~1M points for plotting
            if downsample_factor > 1:
                signal_data = signal_data[::downsample_factor]

            fft_result = np.fft.fft(signal_data)
            fft_magnitude = np.abs(fft_result)
            freqs = np.fft.fftfreq(len(signal_data), downsample_factor/self.sampling_rate)
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]

        plt.figure(figsize=(15, 8))
        plt.plot(positive_freqs, 20 * np.log10(positive_magnitude + 1e-10), 'b-', linewidth=1)
        plt.title(f'FFT Magnitude Spectrum (Fallback) - {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')
        plt.xlim(0, self.sampling_rate/(2*downsample_factor))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_power_spectral_density(self, dataset_name, use_mhkit_dolfyn=True):
        """
        Plot Power Spectral Density (PSD) - Chart Type #2 from documentation
        with GPU acceleration and downsampling
        Now supports MHKiT dolfyn FFT tools when available
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")

        # Use MHKiT dolfyn if available and requested
        if use_mhkit_dolfyn and MHKIT_DOLFYN_AVAILABLE:
            print("Using MHKiT dolfyn for Power Spectral Density...")
            fft_results = self.calculate_fft_mhkit_dolfyn(dataset_name, nfft=2048)
            
            freqs = fft_results['frequencies']
            psd = fft_results['psd']
            method = fft_results['method']
            
            # Filter for positive frequencies
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_psd = psd[positive_mask]
            
            plt.figure(figsize=(15, 8))
            plt.semilogy(positive_freqs, positive_psd, 'r-', linewidth=1)
            plt.title(f'Power Spectral Density ({method}) - {dataset_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power Spectral Density [V²/Hz]')
            plt.xlim(0, self.sampling_rate/2)
            plt.grid(True, alpha=0.3)
            
            # Add grid frequency marker
            plt.axvline(x=self.grid_freq, color='red', linestyle='--', alpha=0.7, 
                       label=f'Grid Frequency ({self.grid_freq} Hz)')
            plt.legend()
            plt.tight_layout()
            plt.show()
            return positive_freqs, positive_psd

        # Fallback to original method
        df = self.datasets[dataset_name]

        if self.use_gpu and 'gpu_data' in df.attrs:
            # Use GPU data
            gpu_signal = df.attrs['gpu_data']
            # Downsample for visualization if needed
            downsample_factor = max(1, len(gpu_signal) // 1000000)  # Limit to ~1M points
            if downsample_factor > 1:
                gpu_signal = gpu_signal[::downsample_factor]

            gpu_fft = cufft.fft(gpu_signal)
            gpu_psd = cp.abs(gpu_fft)**2 / len(gpu_signal)
            gpu_freqs = cufft.fftfreq(len(gpu_signal), downsample_factor/self.sampling_rate)

            positive_idx = gpu_freqs >= 0
            freqs = gpu_freqs[positive_idx].get()
            psd = gpu_psd[positive_idx].get()
        else:
            signal_data = df.iloc[:, 0].values
            # Downsample for visualization if needed
            downsample_factor = max(1, len(signal_data) // 1000000)  # Limit to ~1M points
            if downsample_factor > 1:
                signal_data = signal_data[::downsample_factor]

            freqs = np.fft.fftfreq(len(signal_data), downsample_factor/self.sampling_rate)
            fft_result = np.fft.fft(signal_data)
            psd = np.abs(fft_result)**2 / len(signal_data)
            positive_idx = freqs >= 0
            freqs = freqs[positive_idx]
            psd = psd[positive_idx]

        plt.figure(figsize=(15, 8))
        plt.semilogy(freqs, psd, 'r-', linewidth=1)
        plt.title(f'Power Spectral Density (Fallback) - {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density [V²/Hz]')
        plt.xlim(0, self.sampling_rate/(2*downsample_factor))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return freqs, psd
        
    def plot_spectrogram(self, dataset_name, max_duration=60):
        """
        Plot Spectrogram (Time-Frequency Heatmap) - Chart Type #1 from documentation
        with GPU acceleration and optimized processing
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")

        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]

        if self.use_gpu and 'gpu_data' in df.attrs:
            # Use GPU data
            gpu_signal_full = df.attrs['gpu_data']
            # Extract the time range
            start_idx = 0
            end_idx = int(duration_seconds * self.sampling_rate)
            end_idx = min(end_idx, len(gpu_signal_full))
            gpu_signal = gpu_signal_full[start_idx:end_idx]

            # Downsample for spectrogram efficiency
            downsample_factor = max(1, len(gpu_signal) // 500000)  # Limit to ~500k points
            if downsample_factor > 1:
                gpu_signal = gpu_signal[::downsample_factor]

            window_size = min(1024, len(gpu_signal)//8)
            overlap = window_size // 2
            hop_size = window_size - overlap
            n_windows = (len(gpu_signal) - window_size) // hop_size + 1

            # GPU-based spectrogram computation
            gpu_window = cp.hanning(window_size)
            gpu_Sxx = cp.zeros((window_size//2, n_windows), dtype=cp.complex64)

            for i in range(n_windows):
                start_idx_win = i * hop_size
                end_idx_win = start_idx_win + window_size
                if end_idx_win <= len(gpu_signal):
                    gpu_windowed = gpu_signal[start_idx_win:end_idx_win] * gpu_window
                    gpu_fft = cufft.fft(gpu_windowed)
                    gpu_Sxx[:, i] = gpu_fft[:window_size//2]

            Sxx = cp.abs(gpu_Sxx).get()
            t = np.arange(n_windows) * hop_size / (self.sampling_rate / downsample_factor)
            f = cufft.fftfreq(window_size, downsample_factor/self.sampling_rate)[:window_size//2].get()
        else:
            signal_data = plot_df.iloc[:, 0].values
            # Downsample for efficiency
            downsample_factor = max(1, len(signal_data) // 500000)  # Limit to ~500k points
            if downsample_factor > 1:
                signal_data = signal_data[::downsample_factor]

            window_size = min(1024, len(signal_data)//8)
            overlap = window_size // 2
            hop_size = window_size - overlap
            n_windows = (len(signal_data) - window_size) // hop_size + 1
            t = np.arange(n_windows) * hop_size / (self.sampling_rate / downsample_factor)
            f = np.fft.fftfreq(window_size, downsample_factor/self.sampling_rate)[:window_size//2]
            Sxx = np.zeros((len(f), n_windows))
            window = np.hanning(window_size)
            for i in range(n_windows):
                start_idx = i * hop_size
                end_idx = start_idx + window_size
                if end_idx <= len(signal_data):
                    windowed_data = signal_data[start_idx:end_idx] * window
                    fft_result = np.fft.fft(windowed_data)
                    Sxx[:, i] = np.abs(fft_result[:window_size//2])

        plt.figure(figsize=(15, 8))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.colorbar(label='Power [dB]')
        plt.title(f'Spectrogram (Time-Frequency Heatmap) - {dataset_name}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim(0, min(10000, self.sampling_rate/(2*downsample_factor)))
        plt.tight_layout()
        plt.show()
        return f, t, Sxx
        
    def plot_harmonic_line_plot(self, dataset_name):
        """
        Plot Harmonic Line Plot (Amplitude vs Harmonic Order) - Chart Type #4 from documentation
        """
        # --- FIX: If harmonic analysis results are missing, run analysis automatically ---
        if dataset_name not in self.analysis_results:
            print(f"No harmonic analysis results for '{dataset_name}'. Running analyze_harmonics_mhkit first...")
            try:
                # Always pass grid_freq=60 to avoid passing sampling_rate by accident
                self.analyze_harmonics_mhkit(dataset_name, grid_freq=60)
            except Exception as e:
                print(f"Failed to analyze harmonics for '{dataset_name}': {e}")
                return
        
        harmonics_df = self.analysis_results[dataset_name]['harmonics']
        harmonic_orders = []
        harmonic_amplitudes = []
        for col in harmonics_df.columns:
            if 'harmonic' in col.lower():
                try:
                    order = int(col.split('_')[-1])
                    amplitude = np.mean(harmonics_df[col])
                    harmonic_orders.append(order)
                    harmonic_amplitudes.append(amplitude)
                except:
                    continue
        if not harmonic_orders:
            print("No harmonic data found in the analysis results")
            return
        plt.figure(figsize=(15, 8))
        plt.stem(harmonic_orders, harmonic_amplitudes, basefmt=' ')
        plt.title(f'Harmonic Line Plot (Amplitude vs Harmonic Order) - {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Harmonic Order')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(harmonic_orders) + 1)
        plt.tight_layout()
        plt.show()
        
    def plot_phase_amplitude_polar(self, dataset_name):
        """
        Plot Phase-Amplitude Polar Plot - Chart Type #5 from documentation
        with GPU acceleration and downsampling
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")

        df = self.datasets[dataset_name]

        if self.use_gpu and 'gpu_data' in df.attrs:
            # Use GPU data
            gpu_signal = df.attrs['gpu_data']
            # Downsample for efficiency
            downsample_factor = max(1, len(gpu_signal) // 1000000)  # Limit to ~1M points
            if downsample_factor > 1:
                gpu_signal = gpu_signal[::downsample_factor]

            gpu_fft = cufft.fft(gpu_signal)
            gpu_magnitude = cp.abs(gpu_fft)
            gpu_phase = cp.angle(gpu_fft)
            gpu_freqs = cufft.fftfreq(len(gpu_signal), downsample_factor/self.sampling_rate)

            positive_idx = (gpu_freqs > 0) & (gpu_freqs < self.sampling_rate/(2*downsample_factor))
            positive_freqs = gpu_freqs[positive_idx].get()
            positive_magnitude = gpu_magnitude[positive_idx].get()
            positive_phase = gpu_phase[positive_idx].get()
        else:
            signal_data = df.iloc[:, 0].values
            # Downsample for efficiency
            downsample_factor = max(1, len(signal_data) // 1000000)  # Limit to ~1M points
            if downsample_factor > 1:
                signal_data = signal_data[::downsample_factor]

            fft_result = np.fft.fft(signal_data)
            fft_magnitude = np.abs(fft_result)
            fft_phase = np.angle(fft_result)
            freqs = np.fft.fftfreq(len(signal_data), downsample_factor/self.sampling_rate)
            positive_idx = (freqs > 0) & (freqs < self.sampling_rate/(2*downsample_factor))
            positive_freqs = freqs[positive_idx]
            positive_magnitude = fft_magnitude[positive_idx]
            positive_phase = fft_phase[positive_idx]

        top_indices = np.argsort(positive_magnitude)[-50:]
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        normalized_magnitude = positive_magnitude[top_indices] / np.max(positive_magnitude[top_indices])
        scatter = ax.scatter(positive_phase[top_indices], normalized_magnitude[top_indices],
                            c=positive_freqs[top_indices], cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Frequency [Hz]')
        plt.title(f'Phase-Amplitude Polar Plot - {dataset_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_thcd_over_time(self, dataset_name):
        """
        Plot THCD evolution over time
        """
        if dataset_name not in self.analysis_results:
            print(f"No harmonic analysis results for '{dataset_name}'. Running analyze_harmonics_mhkit first...")
            try:
                self.analyze_harmonics_mhkit(dataset_name, grid_freq=60)
            except Exception as e:
                print(f"Failed to analyze harmonics for '{dataset_name}': {e}")
                return
        
        thcd = self.analysis_results[dataset_name]['thcd']
        plt.figure(figsize=(15, 6))
        plt.plot(thcd.index, thcd.iloc[:, 0], 'g-', linewidth=2)
        plt.title(f'Total Harmonic Current Distortion (THCD) Over Time - {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('THCD [%]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_cwt_scalogram(self, dataset_name, max_duration=10):
        """
        Plot Continuous Wavelet Transform (CWT) Scalogram - PhD-level Chart Type #1
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]
        signal_data = plot_df.iloc[:, 0].values
        if PYWT_AVAILABLE:
            widths = np.arange(1, 128)
            cwt_result, freqs = pywt_cwt(signal_data, widths, 'morl')
            time_axis = np.linspace(0, duration_seconds, len(signal_data))
            plt.figure(figsize=(15, 8))
            plt.pcolormesh(time_axis, freqs, np.abs(cwt_result), shading='gouraud')
            plt.colorbar(label='Magnitude')
            plt.title(f'CWT Scalogram (PyWavelets) - {dataset_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.yscale('log')
            plt.tight_layout()
            plt.show()
        else:
            print("PyWavelets not available. Using alternative time-frequency analysis...")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Time-Frequency Analysis (Alternative to CWT) - {dataset_name}', 
                        fontsize=14, fontweight='bold')
            window_sizes = [64, 128, 256, 512]
            for i, window_size in enumerate(window_sizes):
                if window_size < len(signal_data) // 4:
                    overlap = window_size // 2
                    hop_size = window_size - overlap
                    n_windows = (len(signal_data) - window_size) // hop_size + 1
                    t = np.arange(n_windows) * hop_size / self.sampling_rate
                    f = np.fft.fftfreq(window_size, 1/self.sampling_rate)[:window_size//2]
                    Sxx = np.zeros((len(f), n_windows))
                    window = np.hanning(window_size)
                    for j in range(n_windows):
                        start_idx = j * hop_size
                        end_idx = start_idx + window_size
                        if end_idx <= len(signal_data):
                            windowed_data = signal_data[start_idx:end_idx] * window
                            fft_result = np.fft.fft(windowed_data)
                            Sxx[:, j] = np.abs(fft_result[:window_size//2])
                    row, col = i // 2, i % 2
                    axes[row, col].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
                    axes[row, col].set_title(f'Window Size: {window_size}')
                    axes[row, col].set_xlabel('Time [s]')
                    axes[row, col].set_ylabel('Frequency [Hz]')
                    axes[row, col].set_ylim(0, min(10000, self.sampling_rate/2))
            plt.tight_layout()
            plt.show()
        
    def plot_cepstrum(self, dataset_name):
        """
        Plot Cepstrum Plot (Power Cepstrum) - PhD-level Chart Type #3
        with GPU acceleration and downsampling
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")

        df = self.datasets[dataset_name]

        if self.use_gpu and 'gpu_data' in df.attrs:
            # Use GPU data
            gpu_signal = df.attrs['gpu_data']
            # Downsample for efficiency
            downsample_factor = max(1, len(gpu_signal) // 500000)  # Limit to ~500k points
            if downsample_factor > 1:
                gpu_signal = gpu_signal[::downsample_factor]

            gpu_fft = cufft.fft(gpu_signal)
            gpu_psd = cp.abs(gpu_fft)**2 / len(gpu_signal)
            gpu_freqs = cufft.fftfreq(len(gpu_signal), downsample_factor/self.sampling_rate)

            positive_idx = gpu_freqs >= 0
            gpu_log_psd = cp.log(gpu_psd[positive_idx] + 1e-10)
            gpu_cepstrum = cp.abs(cufft.fft(gpu_log_psd))

            cepstrum = gpu_cepstrum.get()
            quefrency = np.arange(len(cepstrum)) / (self.sampling_rate / downsample_factor)
        else:
            signal_data = df.iloc[:, 0].values
            # Downsample for efficiency
            downsample_factor = max(1, len(signal_data) // 500000)  # Limit to ~500k points
            if downsample_factor > 1:
                signal_data = signal_data[::downsample_factor]

            freqs = np.fft.fftfreq(len(signal_data), downsample_factor/self.sampling_rate)
            fft_result = np.fft.fft(signal_data)
            psd = np.abs(fft_result)**2 / len(signal_data)
            positive_idx = freqs >= 0
            freqs = freqs[positive_idx]
            psd = psd[positive_idx]
            log_psd = np.log(psd + 1e-10)
            cepstrum = np.abs(np.fft.fft(log_psd))
            quefrency = np.arange(len(cepstrum)) / (self.sampling_rate / downsample_factor)

        plt.figure(figsize=(15, 8))
        plt.plot(quefrency[:len(quefrency)//2], cepstrum[:len(cepstrum)//2], 'm-', linewidth=1)
        plt.title(f'Cepstrum Plot (Power Cepstrum) - {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Quefrency [s]')
        plt.ylabel('Cepstrum Magnitude')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_instantaneous_frequency(self, dataset_name, max_duration=60, voltage_data=None, current_data=None):
        """
        Plot instantaneous frequency analysis using MHKiT power.characteristics.instantaneous_frequency
        Following the MHKiT documentation pattern: https://mhkit-software.github.io/MHKiT/power_example.html
        
        Parameters:
        dataset_name (str): Name of the dataset to analyze
        max_duration (float): Maximum duration to plot in seconds
        voltage_data (pd.DataFrame, optional): Voltage data. If None, uses the main dataset
        current_data (pd.DataFrame, optional): Current data. If None, converts from voltage
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        # Perform instantaneous frequency analysis
        try:
            freq_results = self.analyze_instantaneous_frequency(dataset_name, voltage_data, current_data)
        except Exception as e:
            print(f"Failed to analyze instantaneous frequency: {e}")
            return
        
        instantaneous_freq = freq_results['instantaneous_frequency']
        freq_stats = freq_results['frequency_statistics']
        nominal_freq = freq_results['nominal_frequency']
        voltage_data = freq_results['voltage_data']
        current_data = freq_results['current_data']
        
        # Limit duration for plotting
        duration_seconds = min(max_duration, (instantaneous_freq.index[-1] - instantaneous_freq.index[0]).total_seconds())
        end_time = instantaneous_freq.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_freq = instantaneous_freq[instantaneous_freq.index <= end_time]
        
        # Create visualization following MHKiT documentation pattern
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Instantaneous Frequency Analysis (MHKiT) - {dataset_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Instantaneous frequency over time (following MHKiT documentation style)
        axes[0, 0].plot(plot_freq.index, plot_freq.iloc[:, 0], 'b-', linewidth=1, alpha=0.8)
        axes[0, 0].axhline(y=nominal_freq, color='r', linestyle='--', alpha=0.7, 
                          label=f'Nominal ({nominal_freq} Hz)')
        axes[0, 0].set_title('Instantaneous Frequency', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency [Hz]')
        axes[0, 0].set_ylim(0, 100)  # Following MHKiT documentation ylim
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Voltage signal (original data)
        plot_voltage = voltage_data[voltage_data.index <= end_time]
        axes[0, 1].plot(plot_voltage.index, plot_voltage.iloc[:, 0], 'g-', linewidth=1, alpha=0.8)
        axes[0, 1].set_title('Voltage Signal', fontweight='bold')
        axes[0, 1].set_ylabel('Voltage [V]')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Current signal (converted from voltage)
        plot_current = current_data[current_data.index <= end_time]
        axes[1, 0].plot(plot_current.index, plot_current.iloc[:, 0], 'orange', linewidth=1, alpha=0.8)
        axes[1, 0].set_title('Current Signal (Converted)', fontweight='bold')
        axes[1, 0].set_ylabel('Current [A]')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Frequency deviation from nominal
        freq_deviation = plot_freq.iloc[:, 0] - nominal_freq
        axes[1, 1].plot(plot_freq.index, freq_deviation, 'purple', linewidth=1, alpha=0.8)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Frequency Deviation from Nominal', fontweight='bold')
        axes[1, 1].set_ylabel('Deviation [Hz]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add frequency bands for power quality assessment
        axes[1, 1].axhspan(-0.1, 0.1, alpha=0.1, color='green', label='Normal Range')
        axes[1, 1].axhspan(-0.5, 0.5, alpha=0.1, color='yellow', label='Acceptable Range')
        
        # Add statistics text box
        stats_text = f"""Dataset: {dataset_name}
Conversion: 1 V = 5 Amps

Frequency Statistics:
Mean: {freq_stats['mean_frequency']:.3f} Hz
Std Dev: {freq_stats['std_frequency']:.3f} Hz
Min: {freq_stats['min_frequency']:.3f} Hz
Max: {freq_stats['max_frequency']:.3f} Hz

Deviation from Nominal:
Mean: {freq_stats['mean_deviation']:.3f} Hz
Max: {freq_stats['max_deviation']:.3f} Hz
RMS: {freq_stats['rms_deviation']:.3f} Hz"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for stats text
        plt.show()
        
        return freq_results
        
    def plot_frequency_spectrogram_instantaneous(self, dataset_name, max_duration=30, voltage_data=None):
        """
        Plot frequency spectrogram with instantaneous frequency overlay
        
        Parameters:
        dataset_name (str): Name of the dataset to analyze
        max_duration (float): Maximum duration to plot in seconds
        voltage_data (pd.DataFrame, optional): Voltage data. If None, uses the main dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        df = self.datasets[dataset_name]
        if voltage_data is None:
            voltage_data = df
        
        # Limit duration
        duration_seconds = min(max_duration, (voltage_data.index[-1] - voltage_data.index[0]).total_seconds())
        end_time = voltage_data.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_data = voltage_data[voltage_data.index <= end_time]
        
        # Get instantaneous frequency
        try:
            freq_results = self.analyze_instantaneous_frequency(dataset_name, voltage_data)
            instantaneous_freq = freq_results['instantaneous_frequency']
            plot_freq = instantaneous_freq[instantaneous_freq.index <= end_time]
        except Exception as e:
            print(f"Failed to get instantaneous frequency: {e}")
            return
        
        # Create spectrogram
        signal_data = plot_data.iloc[:, 0].values
        window_size = min(1024, len(signal_data)//8)
        overlap = window_size // 2
        hop_size = window_size - overlap
        n_windows = (len(signal_data) - window_size) // hop_size + 1
        
        # Calculate spectrogram
        t = np.arange(n_windows) * hop_size / self.sampling_rate
        f = np.fft.fftfreq(window_size, 1/self.sampling_rate)[:window_size//2]
        Sxx = np.zeros((len(f), n_windows))
        window = np.hanning(window_size)
        
        for i in range(n_windows):
            start_idx = i * hop_size
            end_idx = start_idx + window_size
            if end_idx <= len(signal_data):
                windowed_data = signal_data[start_idx:end_idx] * window
                fft_result = np.fft.fft(windowed_data)
                Sxx[:, i] = np.abs(fft_result[:window_size//2])
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Frequency Spectrogram with Instantaneous Frequency Overlay - {dataset_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Spectrogram
        im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        axes[0].set_title('Power Spectral Density', fontweight='bold')
        axes[0].set_ylabel('Frequency [Hz]')
        axes[0].set_ylim(0, min(1000, self.sampling_rate/2))
        plt.colorbar(im1, ax=axes[0], label='Power [dB]')
        
        # Plot 2: Instantaneous frequency overlaid on spectrogram
        im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        
        # Overlay instantaneous frequency
        freq_time = (plot_freq.index - plot_freq.index[0]).total_seconds()
        axes[1].plot(freq_time, plot_freq.iloc[:, 0], 'r-', linewidth=3, alpha=0.8, 
                    label='Instantaneous Frequency')
        axes[1].axhline(y=60, color='yellow', linestyle='--', linewidth=2, alpha=0.8, 
                       label='Nominal (60 Hz)')
        
        axes[1].set_title('Spectrogram with Instantaneous Frequency Overlay', fontweight='bold')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Frequency [Hz]')
        axes[1].set_ylim(0, min(1000, self.sampling_rate/2))
        axes[1].legend()
        plt.colorbar(im2, ax=axes[1], label='Power [dB]')
        
        plt.tight_layout()
        plt.show()
        
    def generate_comprehensive_report(self, dataset_name):
        """
        Generate a comprehensive statistical summary report
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        
        if dataset_name not in self.analysis_results:
            print(f"No harmonic analysis results for '{dataset_name}'. Running analyze_harmonics_mhkit first...")
            try:
                self.analyze_harmonics_mhkit(dataset_name, grid_freq=60)
            except Exception as e:
                print(f"Failed to analyze harmonics for '{dataset_name}': {e}")
                return
        
        df = self.datasets[dataset_name]
        results = self.analysis_results[dataset_name]
        metrics = self.calculate_additional_metrics(dataset_name)
        
        print("="*80)
        print(f"COMPREHENSIVE HARMONIC ANALYSIS REPORT - {dataset_name}")
        print("="*80)
        print("\n1. DATASET INFORMATION:")
        print(f"   File: {os.path.basename(self.h5_file_path)}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Sampling Rate: {self.sampling_rate:,} Hz")
        print(f"   Duration: {df.index[-1] - df.index[0]}")
        print(f"   Total Samples: {len(df):,}")
        print("\n2. BASIC SIGNAL STATISTICS:")
        signal_data = df.iloc[:, 0].values
        print(f"   RMS Value: {metrics['rms']:.4f}")
        print(f"   Peak Value: {metrics['peak']:.4f}")
        print(f"   Crest Factor: {metrics['crest_factor']:.4f}")
        print(f"   Mean: {np.mean(signal_data):.4f}")
        print(f"   Std Dev: {np.std(signal_data):.4f}")
        print("\n3. FREQUENCY ANALYSIS:")
        print(f"   Dominant Frequency: {metrics['dominant_frequency']:.2f} Hz")
        print(f"   Fundamental Frequency: {metrics['fundamental_frequency']:.2f} Hz")
        print(f"   Grid Frequency: {self.grid_freq} Hz")
        print("\n4. HARMONIC ANALYSIS (MHKiT):")
        print(f"   THCD: {results['thcd']:.2f}%")
        print(f"   Rated Current: {results['rated_current']:.4f}")
        print(f"   THD (Simplified): {metrics['thd_simplified']:.2f}%")
        harmonics_df = results['harmonics']
        if len(harmonics_df.columns) > 0:
            print(f"   Number of Harmonics Detected: {len(harmonics_df.columns)}")
            print("   Top 5 Harmonics by Amplitude:")
            harmonic_amplitudes = {}
            for col in harmonics_df.columns:
                if 'harmonic' in col.lower():
                    try:
                        order = int(col.split('_')[-1])
                        amplitude = np.mean(harmonics_df[col])
                        harmonic_amplitudes[order] = amplitude
                    except:
                        continue
            sorted_harmonics = sorted(harmonic_amplitudes.items(), key=lambda x: x[1], reverse=True)
            for i, (order, amplitude) in enumerate(sorted_harmonics[:5]):
                print(f"     {i+1}. {order}th Harmonic: {amplitude:.4f}")
        interharmonics_df = results['interharmonics']
        if len(interharmonics_df.columns) > 0:
            print(f"   Number of Interharmonics Detected: {len(interharmonics_df.columns)}")
        print("\n5. POWER QUALITY ASSESSMENT:")
        thcd_value = results['thcd']
        if thcd_value < 3:
            quality = "Excellent"
        elif thcd_value < 5:
            quality = "Good"
        elif thcd_value < 8:
            quality = "Acceptable (IEEE 519 limit)"
        elif thcd_value < 12:
            quality = "Poor"
        else:
            quality = "Unacceptable"
        print(f"   Power Quality Rating: {quality}")
        print(f"   THCD Level: {thcd_value:.2f}%")
        print("\n6. STANDARDS COMPLIANCE:")
        print(f"   IEEE 519-2014 Residential Limit: 5% THCD")
        print(f"   IEEE 519-2014 Commercial Limit: 8% THCD")
        print(f"   Measured THCD: {thcd_value:.2f}%")
        if thcd_value <= 8:
            print("   ✓ Complies with IEEE 519-2014 standards")
        else:
            print("   ✗ Exceeds IEEE 519-2014 limits")
        print("\n" + "="*80)
        
    def plot_wigner_ville_distribution(self, dataset_name, max_duration=5):
        """
        Plot Wigner-Ville Distribution (WVD) - PhD-level Chart Type #2
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]
        signal_data = plot_df.iloc[:, 0].values
        try:
            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(signal_data, fs=self.sampling_rate, 
                                  nperseg=min(1024, len(signal_data)//8),
                                  noverlap=min(512, len(signal_data)//16),
                                  window='hann')
            plt.figure(figsize=(15, 8))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            plt.colorbar(label='Power [dB]')
            plt.title(f'Wigner-Ville Distribution (Approximation) - {dataset_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.ylim(0, min(10000, self.sampling_rate/2))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"WVD calculation failed: {e}")
            print("Using alternative time-frequency representation...")
            self.plot_spectrogram(dataset_name, max_duration)
    
    def plot_hilbert_huang_transform(self, dataset_name, max_duration=5):
        """
        Plot Hilbert-Huang Transform (HHT) - PhD-level Chart Type #4
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]
        signal_data = plot_df.iloc[:, 0].values
        try:
            analytic_signal = hilbert(signal_data)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) / 
                                     (2.0 * np.pi) * self.sampling_rate)
            time_axis = np.linspace(0, duration_seconds, len(signal_data))
            freq_time_axis = time_axis[1:]
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 1, 1)
            plt.plot(time_axis, signal_data, 'b-', alpha=0.7, label='Original Signal')
            plt.plot(time_axis, amplitude_envelope, 'r-', linewidth=2, label='Amplitude Envelope')
            plt.plot(time_axis, -amplitude_envelope, 'r-', linewidth=2)
            plt.title(f'Hilbert-Huang Transform - {dataset_name}', fontsize=14, fontweight='bold')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.subplot(3, 1, 2)
            plt.plot(freq_time_axis, instantaneous_frequency, 'g-', linewidth=1)
            plt.ylabel('Instantaneous Frequency [Hz]')
            plt.grid(True, alpha=0.3)
            plt.subplot(3, 1, 3)
            freq_bins = np.linspace(0, min(1000, self.sampling_rate/2), 100)
            time_bins = np.linspace(0, duration_seconds, 50)
            freq_time_matrix = np.zeros((len(freq_bins)-1, len(time_bins)-1))
            for i in range(len(time_bins)-1):
                start_idx = int(i * len(instantaneous_frequency) / (len(time_bins)-1))
                end_idx = int((i+1) * len(instantaneous_frequency) / (len(time_bins)-1))
                if end_idx > start_idx and start_idx < len(instantaneous_frequency):
                    avg_freq = np.mean(instantaneous_frequency[start_idx:end_idx])
                    freq_idx = np.digitize(avg_freq, freq_bins) - 1
                    if 0 <= freq_idx < len(freq_time_matrix):
                        freq_time_matrix[freq_idx, i] = np.mean(amplitude_envelope[start_idx:end_idx])
            plt.pcolormesh(time_bins, freq_bins, freq_time_matrix, shading='gouraud')
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.ylim(0, min(1000, self.sampling_rate/2))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"HHT calculation failed: {e}")
            print("Using alternative analysis...")
            self.plot_spectrogram(dataset_name, max_duration)
    
    def plot_bispectrum(self, dataset_name, max_duration=5):
        """
        Plot Bispectrum / Bicoherence - PhD-level Chart Type #5
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]
        signal_data = plot_df.iloc[:, 0].values
        try:
            downsample_factor = max(1, len(signal_data) // 10000)
            signal_downsampled = signal_data[::downsample_factor]
            fft_result = fft(signal_downsampled)
            freqs = fftfreq(len(signal_downsampled), downsample_factor/self.sampling_rate)
            n = len(fft_result)
            bispectrum = np.zeros((n//2, n//2), dtype=complex)
            for i in range(n//2):
                for j in range(n//2):
                    k = i + j
                    if k < n//2:
                        bispectrum[i, j] = fft_result[i] * fft_result[j] * np.conj(fft_result[k])
            bispectrum_mag = np.abs(bispectrum)
            bispectrum_log = np.log10(bispectrum_mag + 1e-10)
            freq_axis = freqs[:n//2]
            plt.figure(figsize=(15, 8))
            plt.pcolormesh(freq_axis, freq_axis, bispectrum_log, shading='gouraud')
            plt.colorbar(label='Log Magnitude')
            plt.title(f'Bispectrum - {dataset_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Frequency f1 [Hz]')
            plt.ylabel('Frequency f2 [Hz]')
            plt.xlim(0, min(1000, self.sampling_rate/2))
            plt.ylim(0, min(1000, self.sampling_rate/2))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Bispectrum calculation failed: {e}")
            print("Using alternative analysis...")
            self.plot_power_spectral_density(dataset_name)
    
    def plot_reassigned_spectrogram(self, dataset_name, max_duration=10):
        """
        Plot Reassigned Spectrogram - PhD-level Chart Type #6
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded. Please load it first.")
        df = self.datasets[dataset_name]
        duration_seconds = min(max_duration, (df.index[-1] - df.index[0]).total_seconds())
        end_time = df.index[0] + pd.Timedelta(seconds=duration_seconds)
        plot_df = df[df.index <= end_time]
        signal_data = plot_df.iloc[:, 0].values
        try:
            from scipy.signal import spectrogram
            nperseg = min(2048, len(signal_data)//4)
            noverlap = nperseg // 2
            f, t, Sxx = spectrogram(signal_data, fs=self.sampling_rate, 
                                  nperseg=nperseg, noverlap=noverlap,
                                  window='hann')
            plt.figure(figsize=(15, 8))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            plt.colorbar(label='Power [dB]')
            plt.title(f'Reassigned Spectrogram (High Resolution) - {dataset_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.ylim(0, min(10000, self.sampling_rate/2))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Reassigned spectrogram calculation failed: {e}")
            print("Using standard spectrogram...")
            self.plot_spectrogram(dataset_name, max_duration)
        
    def analyze_all_datasets(self, dataset_names):
        """
        Analyze multiple datasets and generate comparative report
        """
        print("Starting comprehensive analysis of all datasets...")
        for dataset_name in dataset_names:
            print(f"\nProcessing {dataset_name}...")
            try:
                self.load_dataset(dataset_name)
                # Always use grid_freq=60 (fixed)
                self.analyze_harmonics_mhkit(dataset_name)
                self.generate_comprehensive_report(dataset_name)
            except Exception as e:
                print(f"Error analyzing {dataset_name}: {str(e)}")
                continue
        self._generate_comparative_summary(dataset_names)
        
    def _generate_comparative_summary(self, dataset_names):
        """Generate comparative summary across all analyzed datasets"""
        print("\n" + "="*80)
        print("COMPARATIVE SUMMARY - ALL DATASETS")
        print("="*80)
        comparison_data = []
        for dataset_name in dataset_names:
            if dataset_name in self.analysis_results:
                results = self.analysis_results[dataset_name]
                metrics = self.calculate_additional_metrics(dataset_name)
                comparison_data.append({
                    'Dataset': dataset_name,
                    'THCD (%)': f"{results['thcd']:.2f}",
                    'RMS': f"{metrics['rms']:.4f}",
                    'Peak': f"{metrics['peak']:.4f}",
                    'Crest Factor': f"{metrics['crest_factor']:.4f}",
                    'Dominant Freq (Hz)': f"{metrics['dominant_frequency']:.2f}",
                    'Quality': "Excellent" if results['thcd'] < 3 else 
                              "Good" if results['thcd'] < 5 else
                              "Acceptable" if results['thcd'] < 8 else
                              "Poor" if results['thcd'] < 12 else "Unacceptable"
                })
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\nCOMPARATIVE METRICS TABLE:")
            print(comparison_df.to_string(index=False))
            thcd_values = [float(data['THCD (%)']) for data in comparison_data]
            best_idx = np.argmin(thcd_values)
            worst_idx = np.argmax(thcd_values)
            print(f"\nBEST PERFORMING DATASET: {comparison_data[best_idx]['Dataset']} (THCD: {comparison_data[best_idx]['THCD (%)']}%)")
            print(f"WORST PERFORMING DATASET: {comparison_data[worst_idx]['Dataset']} (THCD: {comparison_data[worst_idx]['THCD (%)']}%)")
        print("\n" + "="*80)

def check_conda_environment():
    """Check if running in the correct conda environment"""
    import os
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'mhkit-venv':
        print("⚠️  WARNING: Not running in mhkit-venv conda environment!")
        print(f"   Current environment: {conda_env if conda_env else 'None'}")
        print("   Please use one of the following methods:")
        print("   1. python conda_runner.py 'comprehensive_mhkit_harmonic_analyzer with plot functions.py'")
        print("   2. ./run_with_conda.sh 'comprehensive_mhkit_harmonic_analyzer with plot functions.py'")
        print("   3. conda activate mhkit-venv && python 'comprehensive_mhkit_harmonic_analyzer with plot functions.py'")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Exiting...")
            return False
        print("Continuing with current environment...")
    else:
        print(f"✓ Running in conda environment: {conda_env}")
    return True

def main():
    """
    Main function to demonstrate the comprehensive MHKiT harmonic analysis
    """
    # Check conda environment
    if not check_conda_environment():
        return
    
    # Import path utilities for consistent path resolution
    try:
        from path_utils import get_h5_files, verify_h5_files, print_path_info
    except ImportError:
        # Fallback to direct path resolution if path_utils not available
        current_dir = Path(__file__).parent
        h5_folder = current_dir / "01 Oak Ridge dataset" / "Collected Dataset H5 format"
        h5_files = sorted([str(f) for f in h5_folder.glob("*.h5")])
        if not h5_files:
            print(f"No .h5 files found in {h5_folder}")
            print(f"Current directory: {current_dir}")
            return
    else:
        # Use path utilities for better error handling
        success, message, h5_files = verify_h5_files()
        if not success:
            print(f"Error: {message}")
            print_path_info()
            return
        print(f"✓ {message}")

    datasets_to_analyze = [
        "transformer_frp2_rogowski_240",
        "transformer_frp2_ct_ground", 
        "transformer_frp2_ct_480",
        "signal_injection_frp2_ct_120",
        "main_power_frp2_rogowski_480"
    ]
    
    for h5_file_path in h5_files:
        print(f"\n=== Analyzing file: {h5_file_path} ===")
        analyzer = MHKiTHarmonicAnalyzer(h5_file_path, sampling_rate=800000)
        analyzer.analyze_all_datasets(datasets_to_analyze)
        example_dataset = datasets_to_analyze[0]
        print(f"\nGenerating comprehensive visualizations for {example_dataset} in file {os.path.basename(h5_file_path)}...")
        # Ensure harmonic analysis is performed before plotting functions that require it
        if example_dataset not in analyzer.analysis_results:
            try:
                analyzer.analyze_harmonics_mhkit(example_dataset)
            except Exception as e:
                print(f"Failed to analyze harmonics for '{example_dataset}': {e}")
        analyzer.plot_time_domain_signal(example_dataset, max_duration=5)
        
        # Demonstrate new MHKiT dolfyn FFT functionality
        print(f"\n=== Demonstrating MHKiT dolfyn FFT Analysis for {example_dataset} ===")
        analyzer.plot_fft_spectrum_mhkit_dolfyn(example_dataset, nfft=2048, max_freq=10000)
        
        # Use updated methods with MHKiT dolfyn support
        analyzer.plot_fft_magnitude_spectrum(example_dataset, use_mhkit_dolfyn=True)
        analyzer.plot_power_spectral_density(example_dataset, use_mhkit_dolfyn=True)
        
        # Continue with other visualizations
        analyzer.plot_spectrogram(example_dataset, max_duration=10)
        analyzer.plot_harmonic_line_plot(example_dataset)
        analyzer.plot_phase_amplitude_polar(example_dataset)
        analyzer.plot_thcd_over_time(example_dataset)
        
        # Demonstrate new instantaneous frequency analysis (following MHKiT documentation)
        print(f"\n=== Demonstrating MHKiT Instantaneous Frequency Analysis for {example_dataset} ===")
        analyzer.plot_instantaneous_frequency(example_dataset, max_duration=30)
        
        analyzer.plot_cwt_scalogram(example_dataset, max_duration=5)
        analyzer.plot_cepstrum(example_dataset)

if __name__ == "__main__":
    main()
