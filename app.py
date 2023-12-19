import cv2
import time
from flask import Flask, render_template, Response, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import easyocr
import os 

app = Flask(__name__)

model = tf.keras.models.load_model('human_vs_nonhuman_model_dropout_pt5.h5')
reader = easyocr.Reader(['en']) #- #for optical charecter recognition

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_color = (0, 0, 255) 
thickness = 2
position = (50, 50)  


def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        _, framec = video_capture.read()
        
        # framec = cv2.resize(frame,(720,720))
        framec = framec[:720,:720,:]
        framebgr = framec.copy()
        framec = cv2.cvtColor(framec,cv2.COLOR_BGR2RGB)
        img = np.expand_dims(framec, axis=0)
        img = img / 255.0  # scaling to fit to the ML model requirements
    
        predictions = model.predict(img) #predict if human or not
    
        result = reader.readtext(framec) #read the text from the frame
        
        labels = [item[1].lower() for item in result] #get the list of all the text found.
                                                      #change it to lower case so that there is no mismatch
        
        deliveryname = 'zomato ' if 'zomato' in labels else False  #identify if zomato is there in the text
        
        if predictions[0][0] > 0.5:
            text = "Prediction: Non-Human" 
        else:
            text = "Prediction: Human" 
            if deliveryname:
                text = f"Prediction: Human: {deliveryname} - delivery" 
        
        cv2.putText(framebgr, text, position, font, font_scale, font_color, thickness)
        
        
        ret, buffer = cv2.imencode('.jpg', framebgr)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index_button.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Helper function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle manual photo uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return "File uploaded successfully"



if __name__ == '__main__':
    app.run(debug=True)
