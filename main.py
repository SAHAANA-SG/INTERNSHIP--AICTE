import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# Main window setup
window = tk.Tk()
window.title("Attendance Management System using Face Recognition")
window.geometry('800x500')
window.configure(background='grey80')

# Directory paths
training_image_dir = "TrainingImage"
student_details_file = "StudentDetails/StudentDetails.csv"
trained_model_file = "TrainingImageLabel/trainner.yml"

# Ensure required directories exist
os.makedirs(training_image_dir, exist_ok=True)
os.makedirs("StudentDetails", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)

# Function to take images
def take_images():
    enrollment = txt_enrollment.get()
    name = txt_name.get()

    if not enrollment or not name:
        messagebox.showerror("Error", "Enrollment and Name fields cannot be empty.")
        return

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            return

        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sample_num = 0

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sample_num += 1
                cv2.imwrite(f"{training_image_dir}/{name}.{enrollment}.{sample_num}.jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Taking Images", img)

            if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 70:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save student details to CSV
        with open(student_details_file, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow([enrollment, name])

        messagebox.showinfo("Success", f"Images saved for {name} and details added to the CSV file.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to train images
def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            id_ = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(image_np)
            for (x, y, w, h) in faces:
                face_samples.append(image_np[y:y + h, x:x + w])
                ids.append(id_)
        return face_samples, ids

    try:
        faces, ids = get_images_and_labels(training_image_dir)
        recognizer.train(faces, np.array(ids))
        recognizer.save(trained_model_file)
        messagebox.showinfo("Success", "Images trained successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to automatically fill attendance
def automatic_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trained_model_file)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            return

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]  # Extract the face
                face = cv2.resize(face, (200, 200))  # Resize for consistency
                id_, confidence = recognizer.predict(face)

                # Debugging confidence values
                print(f"ID: {id_}, Confidence: {confidence}")

                if confidence < 60:  # Adjusted confidence threshold
                    with open(student_details_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if int(row[0]) == id_:
                                name = row[1]
                                ts = time.time()
                                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                                time_stamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                                with open('Attendance.csv', 'a+', newline='') as file:
                                    writer = csv.writer(file, delimiter=',')
                                    writer.writerow([id_, name, date, time_stamp])
                                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to view registered students
def view_registered_students():
    try:
        if os.path.exists(student_details_file):
            df = pd.read_csv(student_details_file, header=None)
            messagebox.showinfo("Registered Students", df.to_string(index=False, header=False))
        else:
            messagebox.showinfo("Info", "No students registered yet.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Layout
tk.Label(window, text="Enrollment", bg="grey80", font=('times', 15, 'bold')).place(x=50, y=50)
txt_enrollment = tk.Entry(window, width=20, font=('times', 15, 'bold'))
txt_enrollment.place(x=200, y=50)

tk.Label(window, text="Name", bg="grey80", font=('times', 15, 'bold')).place(x=50, y=100)
txt_name = tk.Entry(window, width=20, font=('times', 15, 'bold'))
txt_name.place(x=200, y=100)

# Buttons
tk.Button(window, text="Take Images", command=take_images, width=15, font=('times', 15, 'bold')).place(x=50, y=200)
tk.Button(window, text="Train Images", command=train_images, width=15, font=('times', 15, 'bold')).place(x=250, y=200)
tk.Button(window, text="Automatic Attendance", command=automatic_attendance, width=20, font=('times', 15, 'bold')).place(x=450, y=200)
tk.Button(window, text="View Registered Students", command=view_registered_students, width=25, font=('times', 15, 'bold')).place(x=650, y=200)

# Run the application
window.mainloop()
