"""
Data Loader for Zense BCI Recordings

Handles loading and parsing of CSV recordings from both Arduino UNO and R4 boards.
"""

import os
import re
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

# Optional pandas support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class Recording:
    """Container for a single EEG recording session."""
    
    # Metadata
    experiment: str
    subject: str
    note: str
    start_time: Optional[datetime]
    sample_rate: int
    num_channels: int
    adc_bits: int
    board: str
    filepath: str
    
    # Data arrays
    timestamps: np.ndarray      # Arduino microseconds
    elapsed_ms: np.ndarray      # Elapsed time in ms
    raw_ch0: np.ndarray         # Raw ADC values (centered)
    raw_ch1: np.ndarray
    filtered_ch0: np.ndarray    # Filtered values
    filtered_ch1: np.ndarray
    
    @property
    def duration_seconds(self) -> float:
        """Total recording duration in seconds."""
        return len(self.timestamps) / self.sample_rate
    
    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self.timestamps)
    
    def get_channel(self, channel: int, filtered: bool = True) -> np.ndarray:
        """Get data for a specific channel."""
        if channel == 0:
            return self.filtered_ch0 if filtered else self.raw_ch0
        elif channel == 1:
            return self.filtered_ch1 if filtered else self.raw_ch1
        else:
            raise ValueError(f"Invalid channel: {channel}. Must be 0 or 1.")
    
    def get_time_vector(self, unit: str = 'seconds') -> np.ndarray:
        """Get time vector in specified units."""
        samples = np.arange(len(self.timestamps))
        if unit == 'seconds':
            return samples / self.sample_rate
        elif unit == 'ms':
            return (samples / self.sample_rate) * 1000
        elif unit == 'samples':
            return samples
        else:
            raise ValueError(f"Invalid unit: {unit}")
    
    def get_segment(self, start_sec: float, end_sec: float) -> 'Recording':
        """Extract a time segment from the recording."""
        start_idx = int(start_sec * self.sample_rate)
        end_idx = int(end_sec * self.sample_rate)
        
        return Recording(
            experiment=self.experiment,
            subject=self.subject,
            note=self.note,
            start_time=self.start_time,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            adc_bits=self.adc_bits,
            board=self.board,
            filepath=self.filepath,
            timestamps=self.timestamps[start_idx:end_idx],
            elapsed_ms=self.elapsed_ms[start_idx:end_idx],
            raw_ch0=self.raw_ch0[start_idx:end_idx],
            raw_ch1=self.raw_ch1[start_idx:end_idx],
            filtered_ch0=self.filtered_ch0[start_idx:end_idx],
            filtered_ch1=self.filtered_ch1[start_idx:end_idx],
        )
    
    def __repr__(self) -> str:
        return (f"Recording(experiment='{self.experiment}', subject='{self.subject}', "
                f"duration={self.duration_seconds:.1f}s, samples={self.num_samples:,}, "
                f"sample_rate={self.sample_rate}Hz)")


class ZenseDataLoader:
    """
    Load and parse Zense BCI CSV recordings.
    
    Handles metadata extraction from comment headers and data parsing.
    Supports both Arduino UNO (256Hz) and R4 (512Hz) formats.
    """
    
    # Regex patterns for metadata extraction
    METADATA_PATTERNS = {
        'experiment': r'# Experiment:\s*(.+)',
        'subject': r'# Subject:\s*(.+)',
        'note': r'# Note:\s*(.*)',
        'start_time': r'# Start Time:\s*(.+)',
        'sample_rate': r'# Sample Rate:\s*(\d+)',
        'num_channels': r'# Channels:\s*(\d+)',
        'adc_bits': r'# ADC Resolution:\s*(\d+)',
        'board': r'# ZENSE BCI Recording \((.+)\)',
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base directory for recordings (default: auto-detect)
        """
        if base_path is None:
            # Auto-detect Recordings folder
            self.base_path = self._find_recordings_folder()
        else:
            self.base_path = Path(base_path)
    
    def _find_recordings_folder(self) -> Path:
        """Find the Recordings folder relative to the Analysis module."""
        current = Path(__file__).parent
        # Go up to Zense root and look for Recordings
        zense_root = current.parent
        recordings = zense_root / "Recordings"
        if recordings.exists():
            return recordings
        return Path(".")
    
    def _parse_metadata(self, lines: List[str]) -> Dict[str, any]:
        """Extract metadata from comment header lines."""
        metadata = {
            'experiment': 'unknown',
            'subject': 'unknown',
            'note': '',
            'start_time': None,
            'sample_rate': 256,  # Default for UNO
            'num_channels': 2,
            'adc_bits': 10,
            'board': 'Arduino UNO Classic',
        }
        
        for line in lines:
            if not line.startswith('#'):
                break
            
            for key, pattern in self.METADATA_PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    value = match.group(1).strip()
                    if key in ('sample_rate', 'num_channels', 'adc_bits'):
                        metadata[key] = int(value)
                    elif key == 'start_time':
                        try:
                            metadata[key] = datetime.fromisoformat(value)
                        except ValueError:
                            metadata[key] = None
                    else:
                        metadata[key] = value
        
        return metadata
    
    def _load_csv_numpy(self, filepath: Path, data_start: int) -> Dict[str, np.ndarray]:
        """Load CSV data using numpy (fallback when pandas unavailable)."""
        data = {
            'timestamps': [],
            'elapsed_ms': [],
            'raw_ch0': [],
            'raw_ch1': [],
            'filtered_ch0': [],
            'filtered_ch1': [],
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # Skip header lines
                if i <= data_start:
                    continue
                
                # Skip comment lines and empty rows
                if not row or (row[0] and row[0].startswith('#')):
                    continue
                
                try:
                    if len(row) >= 7:
                        data['timestamps'].append(float(row[1]))
                        data['elapsed_ms'].append(float(row[2]))
                        data['raw_ch0'].append(float(row[3]))
                        data['raw_ch1'].append(float(row[4]))
                        data['filtered_ch0'].append(float(row[5]))
                        data['filtered_ch1'].append(float(row[6]))
                except (ValueError, IndexError):
                    continue  # Skip malformed rows
        
        return {k: np.array(v) for k, v in data.items()}
    
    def load(self, filepath: Union[str, Path]) -> Recording:
        """
        Load a single recording from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Recording object with metadata and data arrays
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Recording not found: {filepath}")
        
        # Read file and separate headers from data
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract metadata from comments
        metadata = self._parse_metadata(lines)
        
        # Find data start (skip comments and header row)
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                data_start = i  # This is the column header row
                break
        
        # Load data using numpy (no pandas dependency)
        data = self._load_csv_numpy(filepath, data_start)
        
        return Recording(
            experiment=metadata['experiment'],
            subject=metadata['subject'],
            note=metadata['note'],
            start_time=metadata['start_time'],
            sample_rate=metadata['sample_rate'],
            num_channels=metadata['num_channels'],
            adc_bits=metadata['adc_bits'],
            board=metadata['board'],
            filepath=str(filepath),
            timestamps=data['timestamps'],
            elapsed_ms=data['elapsed_ms'],
            raw_ch0=data['raw_ch0'],
            raw_ch1=data['raw_ch1'],
            filtered_ch0=data['filtered_ch0'],
            filtered_ch1=data['filtered_ch1'],
        )
    
    def load_directory(self, directory: Optional[Union[str, Path]] = None, 
                       pattern: str = "*.csv") -> List[Recording]:
        """
        Load all recordings from a directory.
        
        Args:
            directory: Path to directory (default: base_path)
            pattern: Glob pattern for file matching
            
        Returns:
            List of Recording objects
        """
        if directory is None:
            directory = self.base_path
        else:
            directory = Path(directory)
        
        recordings = []
        for filepath in sorted(directory.glob(pattern)):
            try:
                recording = self.load(filepath)
                recordings.append(recording)
                print(f"Loaded: {filepath.name} ({recording.num_samples:,} samples)")
            except Exception as e:
                print(f"Warning: Failed to load {filepath.name}: {e}")
        
        return recordings
    
    def list_recordings(self, directory: Optional[Union[str, Path]] = None) -> List[Path]:
        """List all CSV files in directory without loading them."""
        if directory is None:
            directory = self.base_path
        else:
            directory = Path(directory)
        
        return sorted(directory.glob("**/*.csv"))


# Convenience functions
def load_recording(filepath: Union[str, Path]) -> Recording:
    """Load a single recording file."""
    loader = ZenseDataLoader()
    return loader.load(filepath)


def load_directory(directory: Union[str, Path], pattern: str = "*.csv") -> List[Recording]:
    """Load all recordings from a directory."""
    loader = ZenseDataLoader()
    return loader.load_directory(directory, pattern)
