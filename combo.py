import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import dlib

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the speech recognition engine to use the default system microphone
microphone = sr.Microphone()

# Set the properties for the text-to-speech engine
engine.setProperty('rate', 150)  # Adjust the speaking rate

# Function to convert speech to text
def speech_to_text():
    with microphone as source:
        print("Speak now...")
        audio = recognizer.listen(source)

    try:
        # Use the speech recognition engine to convert speech to text
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Error: {str(e)}")

# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Function to save the NumPy array with a given name
def save_numpy_array(array, name):
    filename = name + '.npy'
    np.save(filename, array)
    print(f"NumPy array saved as {filename}")

# Load face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Example usage for face detection and landmarks
video_capture = cv2.VideoCapture(0)
features = np.empty((0, 68, 2), dtype=int)
name = ''

while len(name) == 0:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray_frame)

    for face in faces:
        # Get the facial landmarks for each face
        landmarks = predictor(gray_frame, face)

        # Append landmarks to features array
        landmarks_array = np.empty((68, 2), dtype=int)
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            landmarks_array[i] = [x, y]
        features = np.append(features, [landmarks_array], axis=0)

        # Draw bounding box around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Iterate over the facial landmarks and draw them on the frame
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the frame with bounding boxes and landmarks
    cv2.imshow('Face Detection', frame)

    # Prompt the user for a name to save the NumPy array
    if len(name) == 0:
        text_to_speech("Hello, can I please know your name?")
        name = speech_to_text()

    # Save the processed frame as a NumPy array
    save_numpy_array(gray_frame, name)

    # Break the loop if 'q' is pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Save the features array to a .npy file
np.save('face_features.npy', features)

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
