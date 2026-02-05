# 🧠 Zense

**Open-source 2-channel EEG headset for accessible brain-computer interface research.**

---

## The Story

It all started when my Emotiv EPOC X license expired in the middle of an experiment. Frustrated by the limitations of proprietary BCI hardware and software, I decided to build my own.

**Zense is the result** — an open-source initiative to democratize neuroscience and make BCI accessible to researchers, hobbyists, and students worldwide.

---

## What's Inside (v0.1)

| Component | Description |
|-----------|-------------|
| **Hardware** | 2-channel EEG using [BioAmp EXG Pill](https://github.com/upsidedownlabs/BioAmp-EXG-Pill) |
| **Firmware** | Arduino sketches for UNO Classic (10-bit, 256Hz) and R4 (14-bit, 512Hz) |
| **Software** | Python real-time visualization with FFT brainwave analysis |

### Features
- 🎯 Real-time brainwave band analysis (Delta, Theta, Alpha, Beta, Gamma)
- 📊 Live signal visualization with mental state detection
- 💾 Automatic CSV recording with full metadata
- 🔧 Configurable digital filters (high-pass, 50Hz notch, low-pass)

---

## Quick Start

```bash
# 1. Flash firmware to Arduino
#    Upload Data_streaming/Data_streaming_UNO/Data_streaming_UNO.ino

# 2. Install Python dependencies
pip install numpy scipy matplotlib pyserial

# 3. Run the interface
python Interface/interface_uno.py
```

---

## Project Structure

```
Zense/
├── Data_streaming/          # Arduino firmware
│   ├── Data_streaming.ino       # For Arduino R4 (14-bit ADC)
│   └── Data_streaming_UNO/      # For Arduino UNO Classic
├── Interface/               # Python visualization
│   ├── interface_uno.py         # Main interface (UNO)
│   └── interface_R4.py          # R4 variant
├── electrode_test/          # Connection testing utility
└── Recordings/              # Stored EEG sessions (CSV)
```

---

## Vision

This is just the beginning. From here, we're building the future of open BCI neuroscience:
- 🔬 More channels, higher resolution
- 🤖 ML-based signal classification  
- 🌐 Community-driven protocols and experiments

**Join us in making brain research accessible to everyone.**

---

## License

Open source — because science should be free.

---

*Built with frustration and determination. 🚀*
