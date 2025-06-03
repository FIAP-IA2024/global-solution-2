/**
 * Disaster Monitoring System - ESP32 Firmware
 * Global Solution 2025.1 - FIAP
 * 
 * This firmware implements an environmental monitoring system
 * for natural disaster prediction using ESP32 and various sensors.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <HCSR04.h>

// WiFi Configuration
const char* ssid = "NETWORK_NAME";      // Replace with your network name
const char* password = "NETWORK_PASSWORD";  // Replace with your network password

// API Configuration
const char* serverUrl = "http://your-server.com/api/data";  // API URL that will receive the data

// Pin Definitions
#define DHT_PIN 4          // DHT22 sensor pin (temperature and humidity)
#define DHT_TYPE DHT22     // DHT sensor type
#define TRIGGER_PIN 5      // HC-SR04 ultrasonic sensor trigger pin
#define ECHO_PIN 18        // HC-SR04 ultrasonic sensor echo pin
#define SOIL_MOISTURE_PIN 34 // Soil moisture sensor pin
#define RAIN_SENSOR_PIN 35   // Rain sensor pin
#define VIBRATION_PIN 32   // Vibration sensor pin
#define LED_PIN 2          // ESP32 integrated LED pin
#define BUZZER_PIN 15      // Alert buzzer pin

// Alert Thresholds
#define TEMP_THRESHOLD_HIGH 35.0  // High temperature (°C)
#define TEMP_THRESHOLD_LOW 5.0    // Low temperature (°C)
#define HUMIDITY_THRESHOLD_HIGH 90.0  // High humidity (%)
#define HUMIDITY_THRESHOLD_LOW 20.0   // Low humidity (%)
#define PRESSURE_THRESHOLD_LOW 1000.0  // Low atmospheric pressure (hPa) - indicates storm
#define WATER_LEVEL_THRESHOLD 10.0  // Critical water level (cm)
#define VIBRATION_THRESHOLD 500     // Vibration threshold for alert

// Sensor Instances
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;
UltraSonicDistanceSensor distanceSensor(TRIGGER_PIN, ECHO_PIN);

// Variables to store sensor data
float temperature = 0;
float humidity = 0;
float pressure = 0;
float waterLevel = 0;
int soilMoisture = 0;
int rainLevel = 0;
int vibrationLevel = 0;

// Time control variables
unsigned long lastDataSendTime = 0;
const unsigned long dataSendInterval = 60000;  // Data sending interval (60 seconds)

// Alert state
bool alertState = false;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Initializing Disaster Monitoring System...");
  
  // Pin configuration
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(SOIL_MOISTURE_PIN, INPUT);
  pinMode(RAIN_SENSOR_PIN, INPUT);
  pinMode(VIBRATION_PIN, INPUT);
  
  // Initialize sensors
  dht.begin();
  
  // Initialize BMP280 sensor
  if (!bmp.begin()) {
    Serial.println("Could not find BMP280 sensor, check wiring!");
  } else {
    Serial.println("BMP280 sensor initialized successfully.");
    // BMP280 sensor settings
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     // Operating mode
                   Adafruit_BMP280::SAMPLING_X2,     // Temperature oversampling
                   Adafruit_BMP280::SAMPLING_X16,    // Pressure oversampling
                   Adafruit_BMP280::FILTER_X16,      // Filtering
                   Adafruit_BMP280::STANDBY_MS_500); // Standby time
  }
  
  // Connect to WiFi network
  connectToWiFi();
  
  // Indicate that the system is ready
  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("Disaster Monitoring System ready!");
}

void loop() {
  // Read sensor data
  readSensorData();
  
  // Check alert conditions
  checkAlertConditions();
  
  // Send data to server at defined interval
  unsigned long currentTime = millis();
  if (currentTime - lastDataSendTime >= dataSendInterval) {
    lastDataSendTime = currentTime;
    sendDataToServer();
  }
  
  // Display data in Serial Monitor (for debugging)
  printSensorData();
  
  // Wait before next reading
  delay(2000);
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi network: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  // Wait for connection (with timeout)
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 20) {
    delay(500);
    Serial.print(".");
    timeout++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("");
    Serial.println("WiFi connection failed. Operating in offline mode.");
  }
}

void readSensorData() {
  // Read temperature and humidity from DHT22
  temperature = dht.readTemperature();
  humidity = dht.readHumidity();
  
  // Check if DHT22 reading failed
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor!");
    temperature = 0;
    humidity = 0;
  }
  
  // Read atmospheric pressure from BMP280
  pressure = bmp.readPressure() / 100.0F; // Convert to hPa
  
  // Read water level (distance) from ultrasonic sensor
  waterLevel = distanceSensor.measureDistanceCm();
  
  // Read soil moisture
  soilMoisture = analogRead(SOIL_MOISTURE_PIN);
  // Normalize to percentage (0-100%)
  soilMoisture = map(soilMoisture, 4095, 0, 0, 100);
  
  // Read rain sensor
  rainLevel = analogRead(RAIN_SENSOR_PIN);
  // Normalize to percentage (0-100%)
  rainLevel = map(rainLevel, 4095, 0, 0, 100);
  
  // Read vibration sensor
  vibrationLevel = analogRead(VIBRATION_PIN);
}

void checkAlertConditions() {
  bool newAlertState = false;
  
  // Check each alert condition
  if (temperature > TEMP_THRESHOLD_HIGH) {
    Serial.println("ALERT: Temperature above threshold!");
    newAlertState = true;
  }
  
  if (temperature < TEMP_THRESHOLD_LOW) {
    Serial.println("ALERT: Temperature below threshold!");
    newAlertState = true;
  }
  
  if (humidity > HUMIDITY_THRESHOLD_HIGH) {
    Serial.println("ALERT: Humidity above threshold!");
    newAlertState = true;
  }
  
  if (humidity < HUMIDITY_THRESHOLD_LOW) {
    Serial.println("ALERT: Humidity below threshold!");
    newAlertState = true;
  }
  
  if (pressure < PRESSURE_THRESHOLD_LOW) {
    Serial.println("ALERT: Low atmospheric pressure, possible storm!");
    newAlertState = true;
  }
  
  if (waterLevel < WATER_LEVEL_THRESHOLD) {
    Serial.println("ALERT: Critical water level, possible flooding!");
    newAlertState = true;
  }
  
  if (vibrationLevel > VIBRATION_THRESHOLD) {
    Serial.println("ALERT: Abnormal vibration detected!");
    newAlertState = true;
  }
  
  // If entered alert state
  if (newAlertState && !alertState) {
    alertState = true;
    triggerAlert();
  }
  
  // If exited alert state
  if (!newAlertState && alertState) {
    alertState = false;
    stopAlert();
  }
}

void triggerAlert() {
  // Activate LED
  digitalWrite(LED_PIN, HIGH);
  
  // Activate buzzer (alert pattern)
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
  }
  
  // Send data immediately in case of alert
  sendDataToServer();
}

void stopAlert() {
  // Deactivate LED
  digitalWrite(LED_PIN, LOW);
}

void sendDataToServer() {
  // Check if connected to WiFi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Trying to reconnect...");
    connectToWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("Reconnection failed. Data not sent.");
      return;
    }
  }
  
  // Create JSON object to send data
  DynamicJsonDocument doc(1024);
  
  doc["device_id"] = "ESP32_DISASTER_01"; // Unique device ID
  doc["temperature"] = temperature;
  doc["humidity"] = humidity;
  doc["pressure"] = pressure;
  doc["water_level"] = waterLevel;
  doc["soil_moisture"] = soilMoisture;
  doc["rain_level"] = rainLevel;
  doc["vibration"] = vibrationLevel;
  doc["alert"] = alertState;
  doc["timestamp"] = millis();
  
  // Serialize JSON
  String jsonData;
  serializeJson(doc, jsonData);
  
  // Configure HTTP request
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");
  
  // Send POST request
  int httpResponseCode = http.POST(jsonData);
  
  // Check response
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("Server response: " + response);
    Serial.println("Data sent successfully. HTTP code: " + String(httpResponseCode));
    
    // Blink LED once to indicate successful sending
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
  } else {
    Serial.println("Error sending data. HTTP code: " + String(httpResponseCode));
  }
  
  http.end();
}

void printSensorData() {
  Serial.println("\n--- SENSOR READINGS ---");
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
  
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");
  
  Serial.print("Atmospheric Pressure: ");
  Serial.print(pressure);
  Serial.println(" hPa");
  
  Serial.print("Water Level: ");
  Serial.print(waterLevel);
  Serial.println(" cm");
  
  Serial.print("Soil Moisture: ");
  Serial.print(soilMoisture);
  Serial.println(" %");
  
  Serial.print("Rain Level: ");
  Serial.print(rainLevel);
  Serial.println(" %");
  
  Serial.print("Vibration Level: ");
  Serial.println(vibrationLevel);
  
  Serial.print("Alert State: ");
  Serial.println(alertState ? "ACTIVE" : "Inactive");
  Serial.println("-----------------------------");
}
