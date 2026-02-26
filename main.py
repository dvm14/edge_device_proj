from time import sleep
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time
from datetime import datetime
import csv
import os
from gpiozero import DistanceSensor
from gpiozero import Button
from lcd_i2c import LCD_I2C

#imports for model training
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

SERVO_PIN = 21  # BCM pin number

MIN_PULSE_MS = 1.0
MAX_PULSE_MS = 2.0

servo = None
sensor = None

# Use pigpio for better PWM (reduces jitter)
factory = PiGPIOFactory()

CSV_FILE = "ultrasonic_data.csv"
CSV_HEADERS = ["timestamp", "distance", "label"]
MODEL_FILE = "model_package.pkl"

TRIG_PIN = 16   # BCM pin for TRIG
ECHO_PIN = 20   # BCM pin for ECHO

BUTTON_PIN = 12 # BCM pin for button input

df = None

X = None
y = None

X_train = None
X_test = None
y_train = None
y_test = None

model = None
model_package = None

button = None
lcd = None

#setup Button and LCD hardware
def setup_input_hardware():
    global button, lcd

    button = Button(BUTTON_PIN, pin_factory=factory)
    lcd = LCD_I2C(39, 16, 2)
    lcd.backlight.on()
    lcd.blink.on()

#write text on the LCD
def set_text_on_lcd(text):
    global lcd
    if lcd is None:
        print("LCD not initialized.")
        return
    lcd.clear()
    lcd.cursor.setPos(0, 0)
    lcd.write_text(text[:16])  # Truncate to 16 chars
    if len(text) > 16:
        lcd.cursor.setPos(1, 0)
        lcd.write_text(text[16:32])  # Next 16 chars on second line

#### MODEL FUNCTIONS
#load data from csv
def load_data():
    global df, X, y, X_train, X_test, y_train, y_test, model

    df = pd.read_csv(CSV_FILE)
    X = None
    y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    model = None
    print(f"\nLoaded {len(df)} rows.")
    print(f"Columns: {', '.join(df.columns)}\n")

#turn empty and pill labels to 0 and 1
def encode_labels():
    global df
    df['label_encoded'] = df['label'].astype('category').cat.codes

#create dataframe with data and labels
def get_data():
    global df, X, y
    X = df[['distance']].values  # Features (distance)
    y = df['label_encoded'].values  # Target (encoded labels)

#split the data into train and test dfs
def split_data(test_size=0.2, random_state=42):
    global X, y, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}\n")
    #show distribution of labels in train vs test
    unique, counts = np.unique(y_train, return_counts=True)
    print("Training label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")

#train a binary classification model 
def train_classifier():
    global X_train, y_train, model

    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    model.fit(X_train, y_train)
    print("Model trained successfully.\n")

#test the model with the test data
def test_classifier():
    global X_test, y_test, model

    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    print(f"Accuracy: {accuracy:.3f}\n")
    print("First 10 predictions (true label, predicted label):")
    for true, pred in zip(y_test[:10], preds[:10]):
        print(f"  ({true}, {pred})")

#save the model
def save_model_package():
    global model, model_package

    model_package = {
        "model": model,
        "label_mapping": dict(enumerate(df['label'].astype('category').cat.categories))
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_package, f)
    print(f"Model package saved to {MODEL_FILE}\n")

#load the model
def load_model_package():
    global model_package, model
    with open(MODEL_FILE, "rb") as f:
        model_package = pickle.load(f)
    model = model_package["model"]
    print(f"Model package loaded from {MODEL_FILE}\n")

#give predictions with live sensor data
def run_live_inference():
    global model, sensor

    if model is None:
        print("Model not loaded. Please call load_model_package() first.")
        return
    if sensor is None:
        print("Sensor not created. Please call create_sensor() first.")
        return

    print("Running live inference. Press Ctrl+C to stop.")
    try:
        while True:
            distance_m = sensor.distance
            pred_encoded = model.predict([[distance_m]])[0]
            pred_label = model_package["label_mapping"].get(pred_encoded, "unknown")
            print(f"Distance: {distance_m:.6f} m | Predicted label: {pred_label}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exiting live inference.")



##### SENSOR FUNCTIONS

#create the ultrasonic distance sensor
def create_sensor():
    global sensor
    sensor = DistanceSensor(trigger=TRIG_PIN, echo=ECHO_PIN, max_distance=4.0, pin_factory=factory)

#log distance data with a specified label to a csv file
def log_ultrasonic_data(label):
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

    global sensor
    while True:
        timestamp = datetime.now().isoformat(timespec="seconds")
        distance_m = sensor.distance   # this is meters

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{distance_m:.6f}", label])

        print(f"{timestamp} | {distance_m:.6f} m | label={label}")
        sleep(0.25)

#### SERVO FUNCTIONS

#create servo
def create_servo():
    global servo

    if servo is not None:
        servo.value = None
        servo.close()
    servo = Servo(SERVO_PIN, pin_factory=factory)

#set the servo to a specified angle
def set_angle(angle):
    """
    Move the servo to a specific angle (0–180 degrees).
    """
    # Convert angle to gpiozero's -1 to 1 range
    value = (angle / 90.0) - 1  # 0° = -1, 90° = 0, 180° = 1
    servo.value = value
    #sleep(0.5)

#increase the servo angle incrementally
def increase_angle(final_angle=180, increment=15):
    """
    Increase the servo angle by a certain increment (in degrees).
    """
    current_angle = 0
    while current_angle <= final_angle:
        set_angle(current_angle)
        print(f"Set angle to {current_angle}°")
        current_angle += increment
        sleep(1)

####### UTILITY FUNCTIONS

#cleanup after finishing pill prediction
def cleanup():
    global servo, sensor, lcd
    print("Exiting...")
    set_text_on_lcd("Exiting...")
    set_angle(0)
    servo.close()
    sensor.close()
    sleep(5)
    lcd.clear()
    lcd.backlight.off()

#setup the model and save it
def setup_model():
    load_data()
    encode_labels()
    get_data()
    split_data()
    train_classifier()
    test_classifier()
    save_model_package()

#run the live inference with the saved model, control the servo, and get sensor live data 
def main():
    global servo, sensor, model, button
    create_sensor()
    create_servo()
    setup_input_hardware()

    load_model_package()

    final_angle = 180
    increment = 1
    current_angle = 0
    set_angle(current_angle)

    distance_m = sensor.distance
    print(f"Initial distance: {distance_m:.6f} m")
    print("Press the button to start live inference.")
    set_text_on_lcd(f"Init:{distance_m:.6f}m, button to start.")
    button.wait_for_press()
    set_text_on_lcd("Running live inference")
    while True:
        try:
            current_angle += increment
            set_angle(current_angle)

            distance_m = sensor.distance
            pred_encoded = model.predict([[distance_m]])[0]
            if pred_encoded == 1:
                print(f"Distance: {distance_m:.6f} m | Predicted label: pill")
                set_text_on_lcd(f"Pill predicted! Dist:{distance_m:.6f}m")
                break
            sleep(0.25)
        except KeyboardInterrupt:
            break
    sleep(5)
    cleanup()


if __name__ == "__main__":
    #setup_model()
    main()
