"""
Real-Time EEG Visualization from Arduino - 2 Channel Version
Reads 2-channel raw data from Arduino and displays:
1. Raw signal waveforms (both channels)
2. All brainwave bands over time (line graph)
Filtering is done in Python for flexibility
Uses threaded serial reading for robust data capture
Includes data storage to CSV for later analysis
"""

import serial
import serial.tools.list_ports
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
import threading
import queue
import csv
import os
from datetime import datetime

# ===== CONFIGURATION =====
SAMPLE_RATE = 512  # Must match Arduino
FFT_SIZE = 256     # FFT window size
BAUD_RATE = 230400
NUM_CHANNELS = 2
ADC_MIDPOINT = 16383 / 2  # 14-bit ADC midpoint

# ===== SESSION METADATA =====
print("=" * 60)
print("       ZENSE BCI - EEG Recording Session Setup")
print("=" * 60)

experiment_name = input("\nExperiment name: ").strip() or "experiment"
subject_name = input("Subject name/ID: ").strip() or "subject"
session_note = input("Session notes (optional): ").strip()

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")
os.makedirs(DATA_DIR, exist_ok=True)

# Generate filename with timestamp
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_experiment = "".join(c if c.isalnum() or c in "._-" else "_" for c in experiment_name)
safe_subject = "".join(c if c.isalnum() or c in "._-" else "_" for c in subject_name)
csv_filename = f"{safe_experiment}_{safe_subject}_{session_timestamp}.csv"
csv_filepath = os.path.join(DATA_DIR, csv_filename)

print(f"\nData will be saved to: {csv_filepath}")

# ===== DATA STORAGE CLASS =====
class DataRecorder:
    """Handles writing EEG data to CSV file"""
    
    def __init__(self, filepath, experiment, subject, note):
        self.filepath = filepath
        self.file = None
        self.writer = None
        self.sample_count = 0
        self.start_time = None
        
        # Store metadata
        self.metadata = {
            'experiment': experiment,
            'subject': subject,
            'note': note,
            'start_time': datetime.now().isoformat(),
            'sample_rate': SAMPLE_RATE,
            'num_channels': NUM_CHANNELS
        }
    
    def start(self):
        """Open file and write header"""
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        
        # Write metadata as comments
        self.file.write(f"# ZENSE BCI Recording\n")
        self.file.write(f"# Experiment: {self.metadata['experiment']}\n")
        self.file.write(f"# Subject: {self.metadata['subject']}\n")
        self.file.write(f"# Note: {self.metadata['note']}\n")
        self.file.write(f"# Start Time: {self.metadata['start_time']}\n")
        self.file.write(f"# Sample Rate: {self.metadata['sample_rate']} Hz\n")
        self.file.write(f"# Channels: {self.metadata['num_channels']}\n")
        self.file.write(f"#\n")
        
        # Write CSV header
        self.writer.writerow([
            'sample_index',
            'timestamp_us',      # Arduino microseconds
            'elapsed_ms',        # Elapsed time in milliseconds
            'raw_ch0',           # Raw ADC value (centered)
            'raw_ch1',
            'filtered_ch0',      # After filtering
            'filtered_ch1'
        ])
        
        self.start_time = time.time()
        print(f"Recording started: {self.filepath}")
    
    def write_sample(self, timestamp_us, raw_ch0, raw_ch1, filtered_ch0, filtered_ch1):
        """Write a single sample to the CSV"""
        if self.writer is None:
            return
        
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        self.writer.writerow([
            self.sample_count,
            timestamp_us,
            f"{elapsed_ms:.2f}",
            f"{raw_ch0:.2f}",
            f"{raw_ch1:.2f}",
            f"{filtered_ch0:.4f}",
            f"{filtered_ch1:.4f}"
        ])
        
        self.sample_count += 1
        
        # Flush periodically to prevent data loss
        if self.sample_count % 512 == 0:
            self.file.flush()
    
    def stop(self):
        """Close file and finalize recording"""
        if self.file:
            # Write footer with summary
            duration = time.time() - self.start_time if self.start_time else 0
            self.file.write(f"\n# Recording ended: {datetime.now().isoformat()}\n")
            self.file.write(f"# Total samples: {self.sample_count}\n")
            self.file.write(f"# Duration: {duration:.2f} seconds\n")
            self.file.write(f"# Effective sample rate: {self.sample_count / duration:.2f} Hz\n" if duration > 0 else "")
            
            self.file.close()
            print(f"\nRecording saved: {self.filepath}")
            print(f"Total samples: {self.sample_count:,}")
            print(f"Duration: {duration:.2f} seconds")

# Create recorder instance
recorder = DataRecorder(csv_filepath, experiment_name, subject_name, session_note)

# ===== THREAD-SAFE DATA QUEUE =====
# Serial thread pushes data here, main thread reads from here
data_queue = queue.Queue(maxsize=10000)
stop_event = threading.Event()

# ===== FILTER DESIGN =====
# Design filters once at startup for efficiency

# High-pass filter (0.5 Hz cutoff) - removes DC drift
hp_sos = signal.butter(2, 0.5, btype='highpass', fs=SAMPLE_RATE, output='sos')

# Notch filter (50 Hz) - removes power line noise
notch_b, notch_a = signal.iirnotch(50, 30, SAMPLE_RATE)

# Low-pass filter (45 Hz cutoff) - anti-aliasing for EEG bands
lp_sos = signal.butter(2, 45, btype='lowpass', fs=SAMPLE_RATE, output='sos')

# Filter states for each channel (for real-time filtering)
hp_zi = [signal.sosfilt_zi(hp_sos) for _ in range(NUM_CHANNELS)]
notch_zi = [signal.lfilter_zi(notch_b, notch_a) for _ in range(NUM_CHANNELS)]
lp_zi = [signal.sosfilt_zi(lp_sos) for _ in range(NUM_CHANNELS)]

def apply_filters(sample, channel):
    """Apply all filters to a single sample for a specific channel"""
    global hp_zi, notch_zi, lp_zi
    
    # High-pass filter
    filtered, hp_zi[channel] = signal.sosfilt(hp_sos, [sample], zi=hp_zi[channel])
    sample = filtered[0]
    
    # Notch filter
    filtered, notch_zi[channel] = signal.lfilter(notch_b, notch_a, [sample], zi=notch_zi[channel])
    sample = filtered[0]
    
    # Low-pass filter
    filtered, lp_zi[channel] = signal.sosfilt(lp_sos, [sample], zi=lp_zi[channel])
    sample = filtered[0]
    
    return sample

# ===== AUTO-DETECT ARDUINO PORT =====
def find_arduino_port():
    """Automatically find Arduino COM port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
            return port.device
    return None

# Try to find Arduino automatically
SERIAL_PORT = find_arduino_port()

if SERIAL_PORT is None:
    print("\nCould not auto-detect Arduino port.")
    print("Available ports:")
    ports = serial.tools.list_ports.comports()
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
    
    if ports:
        choice = input("\nEnter port number (or type port name like 'COM3'): ")
        if choice.isdigit():
            SERIAL_PORT = ports[int(choice)].device
        else:
            SERIAL_PORT = choice
    else:
        print("No serial ports found! Is Arduino connected?")
        exit(1)

print(f"\nUsing port: {SERIAL_PORT}")

# ===== CONNECT TO ARDUINO =====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to Arduino!")
    time.sleep(2)  # Wait for Arduino to reset
    
    # Read and print header info
    for _ in range(10):
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            print(f"Arduino: {line}")
            
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit(1)

# ===== SERIAL READER THREAD =====
def serial_reader_thread():
    """
    Dedicated thread for reading serial data.
    Runs continuously in background, pushing parsed data to queue.
    This ensures no samples are missed even if main thread is busy.
    """
    last_ts = 0
    dropped = 0
    
    print("Serial reader thread started")
    
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                # Skip header lines
                if not line or line == 'READY' or ':' in line:
                    continue
                
                # Parse: timestamp,ch0,ch1
                parts = line.split(',')
                if len(parts) == 3:
                    timestamp = int(parts[0])
                    raw_ch0 = float(parts[1])
                    raw_ch1 = float(parts[2])
                    
                    # Check for dropped samples
                    if last_ts > 0:
                        expected_interval = 1000000 / SAMPLE_RATE
                        actual_gap = timestamp - last_ts
                        if actual_gap > expected_interval * 2:
                            dropped += int(actual_gap / expected_interval) - 1
                    last_ts = timestamp
                    
                    # Center around zero
                    centered_ch0 = raw_ch0 - ADC_MIDPOINT
                    centered_ch1 = raw_ch1 - ADC_MIDPOINT
                    
                    # Push to queue (non-blocking)
                    try:
                        data_queue.put_nowait({
                            'timestamp': timestamp,
                            'raw_ch0': centered_ch0,
                            'raw_ch1': centered_ch1,
                            'dropped': dropped
                        })
                    except queue.Full:
                        # Queue full - main thread not keeping up
                        # Drop oldest data to prevent memory issues
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait({
                                'timestamp': timestamp,
                                'raw_ch0': centered_ch0,
                                'raw_ch1': centered_ch1,
                                'dropped': dropped
                            })
                        except:
                            pass
                            
        except Exception as e:
            if not stop_event.is_set():
                print(f"Serial read error: {e}")
            time.sleep(0.001)
    
    print("Serial reader thread stopped")

# Start the serial reader thread
reader_thread = threading.Thread(target=serial_reader_thread, daemon=True)
reader_thread.start()

# Start recording
recorder.start()

# ===== DATA STORAGE =====
raw_data_ch0 = deque(maxlen=500)    # Last 500 samples for channel 0
raw_data_ch1 = deque(maxlen=500)    # Last 500 samples for channel 1
buffer_ch0 = []
buffer_ch1 = []

# History for all bands (for line graph) - using channel 0 for now
BAND_HISTORY_LEN = 100
delta_history = deque(maxlen=BAND_HISTORY_LEN)
theta_history = deque(maxlen=BAND_HISTORY_LEN)
alpha_history = deque(maxlen=BAND_HISTORY_LEN)
beta_history = deque(maxlen=BAND_HISTORY_LEN)
gamma_history = deque(maxlen=BAND_HISTORY_LEN)

# Stats
dropped_samples = 0
queue_size_avg = 0

# ===== SETUP MATPLOTLIB FIGURE =====
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1, 1, 1])

# Subplot 1: Raw signal waveform - Channel 0
ax1 = fig.add_subplot(gs[0])
line_ch0, = ax1.plot([], [], 'cyan', linewidth=1.5, label='Channel 0 (A0)')
line_ch1, = ax1.plot([], [], 'lime', linewidth=1.5, alpha=0.8, label='Channel 1 (A1)')
ax1.set_xlim(0, 500)
ax1.set_ylim(-500, 500)
ax1.set_xlabel('Samples', fontsize=10)
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.set_title('Real-Time EEG Signal (2 Channels)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: All bands over time (LINE GRAPH)
ax2 = fig.add_subplot(gs[1])
line_delta, = ax2.plot([], [], '#1f77b4', linewidth=2, label='Delta (0.5-4Hz)')
line_theta, = ax2.plot([], [], '#9467bd', linewidth=2, label='Theta (4-8Hz)')
line_alpha, = ax2.plot([], [], '#2ca02c', linewidth=2, label='Alpha (8-13Hz)')
line_beta, = ax2.plot([], [], '#ff7f0e', linewidth=2, label='Beta (13-30Hz)')
line_gamma, = ax2.plot([], [], '#d62728', linewidth=2, label='Gamma (30-45Hz)')
ax2.set_xlim(0, BAND_HISTORY_LEN)
ax2.set_ylim(0, 60)
ax2.set_xlabel('Time (updates)', fontsize=10)
ax2.set_ylabel('Power (%)', fontsize=10)
ax2.set_title('Brainwave Bands Over Time (Channel 0)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: Status text
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
status_text = ax3.text(0.5, 0.5, '', fontsize=11, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                       family='monospace')

# Add recording indicator
recording_text = ax3.text(0.5, 0.85, f'● RECORDING: {csv_filename}', fontsize=10, 
                          ha='center', va='center', color='red', fontweight='bold')

fig.suptitle(f'ZENSE BCI - {experiment_name} | Subject: {subject_name}', 
             fontsize=16, fontweight='bold', y=0.98)

# ===== SIGNAL PROCESSING FUNCTIONS =====
def calculate_bandpower(fft_power, freqs, band_range):
    """Calculate power in a frequency band"""
    idx = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
    return np.sum(fft_power[idx])

# ===== UPDATE FUNCTION =====
sample_count = 0
last_update_time = time.time()

def update(frame):
    global buffer_ch0, buffer_ch1, sample_count, last_update_time
    global dropped_samples, queue_size_avg
    
    # Process all available data from queue
    samples_processed = 0
    current_queue_size = data_queue.qsize()
    queue_size_avg = 0.9 * queue_size_avg + 0.1 * current_queue_size
    
    while not data_queue.empty() and samples_processed < 200:
        try:
            data = data_queue.get_nowait()
            
            raw_ch0 = data['raw_ch0']
            raw_ch1 = data['raw_ch1']
            timestamp = data['timestamp']
            dropped_samples = data['dropped']
            
            # Apply filters
            filtered_ch0 = apply_filters(raw_ch0, 0)
            filtered_ch1 = apply_filters(raw_ch1, 1)
            
            # Record to CSV (both raw and filtered)
            recorder.write_sample(timestamp, raw_ch0, raw_ch1, filtered_ch0, filtered_ch1)
            
            # Store filtered data for visualization
            raw_data_ch0.append(filtered_ch0)
            raw_data_ch1.append(filtered_ch1)
            buffer_ch0.append(filtered_ch0)
            buffer_ch1.append(filtered_ch1)
            
            sample_count += 1
            samples_processed += 1
            
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error processing: {e}")
    
    # Update raw signal plot
    if len(raw_data_ch0) > 10:
        line_ch0.set_data(range(len(raw_data_ch0)), list(raw_data_ch0))
        line_ch1.set_data(range(len(raw_data_ch1)), list(raw_data_ch1))
        
        # Auto-scale Y axis based on data
        all_data = list(raw_data_ch0) + list(raw_data_ch1)
        if len(all_data) > 0:
            y_min = min(all_data) - 50
            y_max = max(all_data) + 50
            ax1.set_ylim(y_min, y_max)
    
    # Process FFT when buffer is full (using channel 0)
    if len(buffer_ch0) >= FFT_SIZE:
        # Apply Hamming window
        windowed = np.array(buffer_ch0[:FFT_SIZE]) * signal.windows.hamming(FFT_SIZE)
        
        # Compute FFT
        fft_vals = rfft(windowed)
        fft_power = np.abs(fft_vals) ** 2
        freqs = rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
        
        # Calculate bandpowers
        total_power = np.sum(fft_power)
        
        if total_power > 0:
            delta = calculate_bandpower(fft_power, freqs, (0.5, 4)) / total_power * 100
            theta = calculate_bandpower(fft_power, freqs, (4, 8)) / total_power * 100
            alpha = calculate_bandpower(fft_power, freqs, (8, 13)) / total_power * 100
            beta = calculate_bandpower(fft_power, freqs, (13, 30)) / total_power * 100
            gamma = calculate_bandpower(fft_power, freqs, (30, 45)) / total_power * 100
            
            # Update band histories
            delta_history.append(delta)
            theta_history.append(theta)
            alpha_history.append(alpha)
            beta_history.append(beta)
            gamma_history.append(gamma)
            
            # Update all bands line graph
            line_delta.set_data(range(len(delta_history)), list(delta_history))
            line_theta.set_data(range(len(theta_history)), list(theta_history))
            line_alpha.set_data(range(len(alpha_history)), list(alpha_history))
            line_beta.set_data(range(len(beta_history)), list(beta_history))
            line_gamma.set_data(range(len(gamma_history)), list(gamma_history))
            
            # Determine mental state
            if beta > 15:
                state = "HIGHLY FOCUSED "
                color = 'lime'
            elif beta > 10:
                state = "FOCUSED "
                color = 'yellow'
            elif alpha > 25:
                state = "RELAXED "
                color = 'cyan'
            elif theta > 20:
                state = "DROWSY "
                color = 'orange'
            else:
                state = "NORMAL "
                color = 'white'
            
            # Update status text
            current_time = time.time()
            elapsed = current_time - last_update_time
            sample_rate_actual = sample_count / elapsed if elapsed > 0 else 0
            
            status_str = (f"Mental State: {state}\n"
                         f"Samples: {sample_count:,} | Rate: {sample_rate_actual:.1f} Hz | "
                         f"Queue: {current_queue_size} | Dropped: {dropped_samples}\n"
                         f"Delta: {delta:.1f}% | Theta: {theta:.1f}% | Alpha: {alpha:.1f}% | "
                         f"Beta: {beta:.1f}% | Gamma: {gamma:.1f}%")
            status_text.set_text(status_str)
            status_text.set_color(color)
        
        # Shift buffer (50% overlap for smoother updates)
        buffer_ch0 = buffer_ch0[FFT_SIZE//2:]
        buffer_ch1 = buffer_ch1[FFT_SIZE//2:]
    
    return line_ch0, line_ch1, line_delta, line_theta, line_alpha, line_beta, line_gamma, status_text

# ===== START ANIMATION =====
print("\n" + "="*60)
print("Starting Real-Time 2-Channel EEG Visualization...")
print("Mode: THREADED serial reading for robust capture")
print("Filters: High-pass 0.5Hz | Notch 50Hz | Low-pass 45Hz")
print(f"Recording to: {csv_filename}")
print("Close the plot window to stop and save.")
print("="*60 + "\n")

try:
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    # Signal thread to stop and clean up
    stop_event.set()
    reader_thread.join(timeout=1)
    ser.close()
    print("Serial connection closed")
    
    # Stop recording and save file
    recorder.stop()
