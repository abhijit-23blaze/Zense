#!/usr/bin/env python
"""
Zense BCI Recording Analyzer

Command-line tool for offline analysis of EEG recordings.

Usage:
    python analyze_recording.py --input recording.csv
    python analyze_recording.py --input Recordings/Attentiontest/ --output reports/
    python analyze_recording.py --input recording.csv --ssvep --frequencies 10,12,15
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from Analysis.data_loader import ZenseDataLoader, load_recording
from Analysis.preprocessing import Preprocessor
from Analysis.features import FeatureExtractor
from Analysis.visualization import plot_recording, plot_bands, plot_spectrogram
from Analysis.utils import generate_report


def analyze_single_recording(filepath: str, output_dir: str = None,
                              show_plots: bool = True,
                              ssvep_mode: bool = False,
                              ssvep_frequencies: list = None) -> dict:
    """
    Analyze a single EEG recording.
    
    Args:
        filepath: Path to CSV recording
        output_dir: Directory for output files
        show_plots: Whether to display plots
        ssvep_mode: Enable SSVEP analysis
        ssvep_frequencies: Target frequencies for SSVEP
        
    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(filepath).name}")
    print('='*60)
    
    # Load recording
    recording = load_recording(filepath)
    print(f"Duration: {recording.duration_seconds:.1f}s | Samples: {recording.num_samples:,}")
    print(f"Subject: {recording.subject} | Experiment: {recording.experiment}")
    
    # Initialize processors
    preprocessor = Preprocessor(recording.sample_rate)
    feature_extractor = FeatureExtractor(recording.sample_rate)
    
    results = {
        'filepath': filepath,
        'metadata': {
            'experiment': recording.experiment,
            'subject': recording.subject,
            'duration': recording.duration_seconds,
            'samples': recording.num_samples,
            'sample_rate': recording.sample_rate,
        }
    }
    
    # Stack channels for analysis
    data = np.vstack([recording.filtered_ch0, recording.filtered_ch1])
    
    # Feature extraction
    print("\nExtracting features...")
    features = feature_extractor.extract_all(data)
    results['features'] = features.to_dict()
    
    # Band power summary
    print("\n📊 Band Power Analysis:")
    relative = features.band_powers.get('relative', {})
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        if band in relative:
            power = np.mean(relative[band])
            bar = '█' * int(power / 2)
            print(f"  {band.capitalize():8}: {power:5.1f}% {bar}")
    
    # Band ratios
    if features.band_ratios:
        print("\n📈 Key Ratios:")
        for ratio, value in features.band_ratios.items():
            print(f"  {ratio:15}: {value:.3f}")
    
    # Artifact detection
    print("\n🔍 Artifact Detection...")
    from Analysis.models.artifact_detector import ArtifactDetector
    detector = ArtifactDetector(recording.sample_rate)
    artifacts = detector.detect_all(data)
    artifact_ratio = detector.compute_artifact_ratio(data)
    
    results['artifacts'] = {
        'count': len(artifacts),
        'ratio': artifact_ratio,
        'types': detector.generate_report(artifacts)
    }
    
    print(f"  Found {len(artifacts)} artifacts ({artifact_ratio*100:.1f}% of data)")
    
    # SSVEP Analysis
    if ssvep_mode:
        print("\n🎯 SSVEP Analysis...")
        from Analysis.models.ssvep_detector import SSVEPDetector
        
        freqs = ssvep_frequencies or [10, 12, 15]
        ssvep_detector = SSVEPDetector(recording.sample_rate, freqs)
        
        ssvep_results = ssvep_detector.analyze_recording(data, epoch_length=4.0)
        summary = ssvep_detector.get_detection_summary(ssvep_results)
        
        results['ssvep'] = summary
        
        print(f"  Detection rate: {summary['detection_rate']*100:.1f}%")
        if summary['dominant_frequency']:
            print(f"  Dominant frequency: {summary['dominant_frequency']} Hz")
        print(f"  Mean SNR: {summary['mean_snr']:.1f} dB")
    
    # Mental state assessment
    print("\n🧠 Mental State Assessment:")
    beta = np.mean(relative.get('beta', [0]))
    alpha = np.mean(relative.get('alpha', [0]))
    theta = np.mean(relative.get('theta', [0]))
    
    if beta > 15:
        state = "FOCUSED - High beta activity indicates concentration"
        color = '\033[92m'  # Green
    elif alpha > 25:
        state = "RELAXED - High alpha activity indicates calm awareness"
        color = '\033[96m'  # Cyan
    elif theta > 20:
        state = "DROWSY - High theta activity indicates decreased alertness"
        color = '\033[93m'  # Yellow
    else:
        state = "NEUTRAL - Balanced brainwave activity"
        color = '\033[0m'   # Default
    
    print(f"  {color}{state}\033[0m")
    results['mental_state'] = state.split(' - ')[0]
    
    # Generate plots
    if show_plots or output_dir:
        print("\n📊 Generating visualizations...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Main recording plot
        fig1 = plot_recording(recording, channel=0)
        if output_dir:
            save_path = os.path.join(output_dir, f"{Path(filepath).stem}_overview.png")
            fig1.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        # Band power over time
        fig2 = plot_bands(recording.filtered_ch0, recording.sample_rate,
                          title=f"Band Powers - {recording.experiment}")
        if output_dir:
            save_path = os.path.join(output_dir, f"{Path(filepath).stem}_bands.png")
            fig2.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close('all')
    
    # Generate text report
    if output_dir:
        report_path = os.path.join(output_dir, f"{Path(filepath).stem}_report.txt")
        generate_report(recording, features, report_path)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Zense BCI EEG recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Analyze a single recording:
    python analyze_recording.py --input Recordings/test.csv
    
  Analyze all recordings in a directory:
    python analyze_recording.py --input Recordings/Attentiontest/
    
  SSVEP analysis with custom frequencies:
    python analyze_recording.py --input recording.csv --ssvep --frequencies 10,12,15
    
  Save outputs without displaying:
    python analyze_recording.py --input recording.csv --output reports/ --no-show
'''
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file or directory')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory for reports and plots')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (save only)')
    parser.add_argument('--ssvep', action='store_true',
                        help='Enable SSVEP analysis mode')
    parser.add_argument('--frequencies', '-f', default=None,
                        help='SSVEP target frequencies (comma-separated, e.g., "10,12,15")')
    
    args = parser.parse_args()
    
    # Parse SSVEP frequencies
    ssvep_freqs = None
    if args.frequencies:
        ssvep_freqs = [float(f) for f in args.frequencies.split(',')]
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        analyze_single_recording(
            str(input_path),
            output_dir=args.output,
            show_plots=not args.no_show,
            ssvep_mode=args.ssvep,
            ssvep_frequencies=ssvep_freqs
        )
    
    elif input_path.is_dir():
        # Directory - analyze all CSVs
        csv_files = list(input_path.glob('*.csv'))
        print(f"Found {len(csv_files)} recordings in {input_path}")
        
        for csv_file in csv_files:
            try:
                analyze_single_recording(
                    str(csv_file),
                    output_dir=args.output,
                    show_plots=False,  # Don't show for batch
                    ssvep_mode=args.ssvep,
                    ssvep_frequencies=ssvep_freqs
                )
            except Exception as e:
                print(f"Error analyzing {csv_file.name}: {e}")
        
        print(f"\n✅ Batch analysis complete!")
        if args.output:
            print(f"Reports saved to: {args.output}")
    
    else:
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
