# edge_device_proj

This project uses a Raspberry Pi, an ultrasonic distance sensor, and a servo motor to detect objects (specifically pills) using a Machine Learning classifier. When a "pill" is detected via the distance sensor, the system updates an I2C LCD and triggers a physical response via the servo.

## Features
Machine Learning Integration: Uses a Scikit-Learn DecisionTreeClassifier to distinguish between "empty" space and a "pill" based on distance data.

Live Hardware Control:
- Servo Motor: Rotates incrementally to scan or position components.
- Ultrasonic Sensor: Provides real-time distance measurements.
- LCD (I2C): Displays real-time status, distances, and predictions.
- Physical Button: Acts as a trigger to start the live inference process.

Data Logging: Includes utility functions to collect and label sensor data to a CSV file for model training.


## Hardware Requirements
- Raspberry Pi (with pigpio daemon recommended)
- Ultrasonic Sensor (HC-SR04 or similar)
- Servo Motor
- I2C LCD Display (16x2)
- Momentary Push Button
- Jumper Wires & Breadboard


## Software Setup
### 1. Prerequisites
Ensure you have the pigpio daemon running for jitter-free servo control: sudo pigpiod
### 2. Dependencies
Install the required Python libraries: pip install gpiozero pigpio pandas numpy scikit-learn lcd-i2c


## How to Use
### Phase 1: Data Collection & Training
Before running the main script, you need a trained model.
- Collect Data: Use log_ultrasonic_data(label) to record distances for both 'empty' and 'pill' states. This generates ultrasonic_data.csv.
- Train Model: Uncomment setup_model() in the if __name__ == "__main__": block.
-   This will:Load the CSV data. Encode labels and split data into training/testing sets. Train a Decision Tree. Save the model as model_package.pkl.
### Phase 2: Live Inference
Once model_package.pkl exists, you can run the main() function:
- The system initializes hardware and waits for a button press.
- Upon pressing the Button, the Servo begins to rotate.
- The Ultrasonic Sensor continuously feeds data to the ML model.
- If the model predicts a "pill" (label 1), the servo stops.
- The LCD displays the detection and the specific distance.
- The system performs a cleanup and exits after 5 seconds.


## Future Improvements
Implement a computer vision to detect number of pills based on user input and better accuracy for any type of pills.
