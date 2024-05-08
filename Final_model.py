import tkinter as tk
from tkinter import messagebox
from tkinter.simpledialog import askstring
from PIL import Image, ImageTk
import cv2
import cv2
import numpy as np
from keras.models import load_model
import schedule 
import time 
import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
from tkinter import *
from PIL import Image, ImageDraw, ImageFont, ImageTk
import mysql.connector
from mysql.connector import Error

cascade_filename = "/home/surjit/Documents/Project/Main/haarcascade_frontalface_default.xml"

host = 'localhost'  # e.g., 'localhost' or 'yourhostname.com'
database = 'project one'
user = 'surjit'
password = 'StrongP@ssw0rd123'




class MainPage:
    def __init__(self, root, window_title):
        self.root = root
        self.root.title(window_title)  # Replace this with the actual path to your image file
        self.root.geometry("1200x650")
        image_path = "/home/surjit/Documents/img2.png"  # Replace this with the actual path to your image file
        self.root.config(bg="white")
        # Open the image using PIL (Pillow)
        self.img = tk.PhotoImage(file=image_path)
        self.label = tk.Label(root, text="Peformance Detector", borderwidth=0,font=("Arial", 35, "bold"), bg="white",fg="#6A5ACD")
        self.label.place(x=5, y=10)
        self.label = tk.Label(root, image=self.img, borderwidth=0)
        self.label.place(x=20, y=70)
        
        

        
# Create a label to display the image
        self.button1 = tk.Button(root, text="Take Face Photo", command=self.open_photo_taker)
        self.button1.place(x=850, y=300, width=300, height=50)  # Adjust the x, y, width, and height as needed

        self.button2 = tk.Button(root, text="Meeting Page", command=self.open_third_page)
        self.button2.place(x=850, y=400, width=300, height=50)  # Adjust the x, y, width, and height as needed



    def open_photo_taker(self):
        self.root.withdraw()  # Hide the main page window
        PhotoTakerPage(self.root)  # Open the photo taker page


    def open_third_page(self):
        self.root.withdraw()  # Hide the main page window
        PhotoTaker(self.root) # Open the third page
    
    

class PhotoTakerPage:
    def __init__(self, root):
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.title("Face Photo Taker")
        self.face_taker = FacePhotoTaker(self.window, self.root)  # Pass the root window reference to FacePhotoTaker

    def on_close(self):
        if self.video.isOpened():
            self.video.release()
        self.root.destroy()  # Show the main page window when the photo taker window is closed

class FacePhotoTaker:
    def __init__(self, root, main_page_root):
        self.root = root
        self.main_page_root = main_page_root
        self.video_source = 0
        self.video = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(root, width=self.video.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        self.btn_snapshot = tk.Button(root, text="Take Photo", width=10, command=self.take_photo)
        self.btn_snapshot.pack(padx=20, pady=10)
        self.btn_back = tk.Button(root, text="Back to Main Page", width=15, command=self.back_to_main_page)
        self.btn_back.pack(padx=20, pady=10)
        self.person_name = None
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def take_photo(self):
        self.person_name = askstring("Input", "Enter your name:")
        if self.person_name:
            ret, frame = self.video.read()
            if ret:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_roi = frame[y:y + h, x:x + w]
                    photo_name = f"{self.person_name}.jpg"
                    save_path = os.path.join("/home/surjit/Documents/face/", photo_name)

                    cv2.imwrite(save_path, face_roi)  # Use save_path, not photo_name
                    messagebox.showinfo("Success", f"Face cropped and saved as: {save_path}")
                else:
                    messagebox.showwarning("Warning", "No face detected!")


    def update(self):
        ret, frame = self.video.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update)

    def on_close(self):
        if self.video.isOpened():
            self.video.release()
        self.root.destroy() # Destroy the window

    def back_to_main_page(self):
        self.on_close()  # Release the video capture before going back to the main page
        self.main_page_root.deiconify()  


class PhotoTaker:
    def __init__(self, root):
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.title("Face Photo Taker")
        self.third_page = ThirdPage(self.window, self.root)  # Pass the correct root references

    def on_close(self):
        if self.video.isOpened():
            self.video.release()
        self.root.destroy() 
class ThirdPage:

    def __init__(self, root,main_page_root):
        self.root = root
        self.load_model = load_model("/home/surjit/Documents/Project/models/model_file_30epochs.h5")

        self.main_page_root = main_page_root
        self.video_source = 0
        self.video = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(root, width=self.video.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack(padx=0, pady=10)
        
        self.btn_snapshot = tk.Button(root, text="Start Meeting", width=10, command=self.run_model)
        self.btn_snapshot.pack(padx=20, pady=10)

        self.btn_back = tk.Button(root, text="Stop Meeting", width=15, command=self.back_to_main_page)
        self.btn_back.pack(padx=20, pady=10)
        
        self.btn_back = tk.Button(root, text="Back to Main Page", width=15, command=self.back_to_main_page)
        self.btn_back.pack(padx=20, pady=10)

        

        self.root.title("Emotion Detection App")
        self.label = Label(root, text="", font=("Helvetica", 16))
        self.label.pack(side=TOP)
        self.data = []
        self.label_dict = {
            0: "Angry Focused",
            1: "Disgust Unfocused",
            2: "Drowsiness",
            3: "laugh focused",
            4: "Happy Full Attention",
            5: "Focused Neutral",
            6: "sad Unfocused",
            7: "surprise"
        }
        self.point_dict={
                        0: 0.9,
                        1: 0.2,
                        2:  0.1,
                        3: 0.8,
                        4: 0.7,
                        5: 0.6,
                        6: 0.3,
                        7: 0.4

                    }

        self.faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.path = "/home/surjit/Documents/face"
        self.images = []
        self.classnames = []

        self.mylist = os.listdir(self.path)
        for cl in self.mylist:
            curimg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curimg)
            self.classnames.append(os.path.splitext(cl)[0])
        self.update()
        self.video_label = tk.Label(root)
        self.video_label.place(x=100,y=500)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.encodelistknown = self.findendcoding(self.images)


    def findendcoding(self,images):  # Add self as the first parameter
        encodelist = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encode = face_encodings[0]
                encodelist.append(encode)
            else:
                print("No face found in the image. Skipping...")
        return encodelist

    def run_model(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=img)
            self.video_label.image = img

            if len(faces) == 0:
                label = -1  # Set label to -1 when no faces are detected
            else:
                facescurframe = face_recognition.face_locations(frame)
                encodecurframe = face_recognition.face_encodings(frame, facescurframe)

                for encodeface, faceloc in zip(encodecurframe, faces):
                    matches = face_recognition.compare_faces(self.encodelistknown, encodeface)
                    facedis = face_recognition.face_distance(self.encodelistknown, encodeface)
                    matchIndex = np.argmin(facedis)

                    x, y, w, h = faceloc
                    sub_face_img = gray[y:y + h, x:x + w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = self.load_model.predict(reshaped)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), -1)
                    label = np.argmax(result, axis=1)[0]

                    if matches[matchIndex]:
                        name = self.classnames[matchIndex].upper()
                        point = self.point_dict[label]

                        y1, x2, y2, x1 = faceloc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scaling back to the original size
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {self.label_dict[label]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    self.data.append({'x': x, 'y': y, 'w': w, 'h': h, 'label': label, 'name': matchIndex, 'point': point})

                    print(x, y, h, w, label, name)
                    
                    
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if self.handle_key_event(key):
                break

        # Create a DataFrame from the 'data' list outside the loop
        df = pd.DataFrame(self.data)
        print(df)


        try:
            # Establish a connection to MySQL server
            connection = mysql.connector.connect(host=host, database=database, user=user, password=password)

            if connection.is_connected():
                # Create a MySQL cursor object using the cursor() method
                cursor = connection.cursor()

                # Insert data into the MySQL table using executemany
                insert_query = f'''
                INSERT INTO Data (x, y, w, h, label, name, point)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                '''
                records_to_insert = [(row['x'], row['y'], row['w'], row['h'], row['label'], row['name'], row['point']) for _, row in df.iterrows()]
                cursor.executemany(insert_query, records_to_insert)

                # Commit changes
                connection.commit()
                print("Data transferred to MySQL table successfully.")

        except Error as e:
            print("Error while connecting to MySQL", e)

        finally:
            # Close the cursor and connection
            if connection is not None and connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed.")



    

    def update(self):
        ret, frame = self.video.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update)


    def on_close(self):
        if self.video.isOpened():
            self.video.release()
        self.root.destroy()

    def back_to_main_page(self):
            self.on_close()  # Release the video capture before going back to the main page
            self.main_page_root.deiconify()


    def handle_key_event(self, key):
        if key == ord('q'):
            self.on_close()  # Release the video capture before going back to the main page
            self.main_page_root.deiconify()
            return True  # Return True to indicate the event was handled
        return False  # Return False if the event was not handled


    


root = tk.Tk()
app = MainPage(root, "Main Page")
root.mainloop()
