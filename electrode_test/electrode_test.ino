// Simple Electrode Test
// This will show you if your sensor is connected and working

#define INPUT_PIN A0
#define BAUD_RATE 115200

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(INPUT_PIN, INPUT);
  
  Serial.println("=== Electrode Connection Test ===");
  Serial.println("Open Serial Plotter (Tools -> Serial Plotter)");
  Serial.println("You should see values between 0-1023");
  Serial.println("If flat at 0 or 1023, check your connections!");
  delay(2000);
}

void loop() {
  // Read raw analog value (0-1023)
  int rawValue = analogRead(INPUT_PIN);
  
  // Print it (Serial Plotter will graph this)
  Serial.println(rawValue);
  
  // Small delay for readability
  delay(10);
}
