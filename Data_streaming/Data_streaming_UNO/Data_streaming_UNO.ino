// BCI EEG - Raw Data Streaming
// BioAmp EXG Pill - Arduino UNO Classic
// Streams 2 channels of RAW samples to computer
// Filtering is done in Python for flexibility
// Uses 10-bit ADC (0-1023 range)

#define SAMPLE_RATE 256       // Samples per second (256 is more stable for UNO)
#define BAUD_RATE 115200      // Serial communication speed
#define NUM_CHANNELS 2        // Number of EEG channels
#define ADC_BITS 10           // Arduino UNO Classic has 10-bit ADC
#define ADC_MAX_VALUE 1023    // 2^10 - 1

// Input pins for each channel
const int INPUT_PINS[NUM_CHANNELS] = {A0, A1};

void setup() {
  Serial.begin(BAUD_RATE);
  
  // Note: Arduino UNO Classic uses 10-bit ADC by default (no analogReadResolution)
  
  // Set up input pins
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(INPUT_PINS[i], INPUT);
  }
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Wait for serial connection
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Send header information (computer can parse this)
  Serial.println("READY");
  Serial.print("SAMPLE_RATE:");
  Serial.println(SAMPLE_RATE);
  Serial.print("NUM_CHANNELS:");
  Serial.println(NUM_CHANNELS);
  Serial.print("ADC_BITS:");
  Serial.println(ADC_BITS);
  
  delay(100);
}

void loop() {
  static unsigned long lastMicros = 0;
  unsigned long currentMicros = micros();
  
  // Precise timing - sample at exactly SAMPLE_RATE Hz
  if (currentMicros - lastMicros >= (1000000 / SAMPLE_RATE)) {
    lastMicros = currentMicros;
    
    // Read raw analog values (0-1023 for 10-bit ADC)
    int raw0 = analogRead(INPUT_PINS[0]);
    int raw1 = analogRead(INPUT_PINS[1]);
    
    // Send raw samples to computer
    // Format: timestamp,ch0,ch1 (comma-separated values per line)
    Serial.print(currentMicros);
    Serial.print(",");
    Serial.print(raw0);
    Serial.print(",");
    Serial.println(raw1);
    
    // Optional: Blink LED to show activity (every SAMPLE_RATE samples = 1 second)
    static int sampleCount = 0;
    sampleCount++;
    if (sampleCount >= SAMPLE_RATE) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      sampleCount = 0;
    }
  }
}
