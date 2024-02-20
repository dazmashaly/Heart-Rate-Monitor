import cv2
import numpy as np
import time
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os



video = False


fu = Face_utilities()
sp = Signal_processing()

i=0
last_rects = None
last_shape = None
last_age = None
last_gender = None

face_detect_on = False
age_gender_on = False

t = time.time()

#for signal_processing
BUFFER_SIZE = 100

fps=0 #for real time capture
times = []
data_buffer = []

# data for plotting


bpm = 0

app = Flask(__name__)
@app.route('/hr', methods=['POST'])
def get_detections():
    image = request.files['image']  # Access the single image file
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))

    frame = cv2.imread(os.path.join(os.getcwd(), image_name))
    global data_buffer
    global times
    global BUFFER_SIZE
    global fps

    
    ret_process = fu.no_age_gender_face_process(frame)

    rects, face, shape, aligned_face, aligned_shape = ret_process

    (x, y, w, h) = rects[0][0][0], rects[0][0][1], rects[0][1][0], rects[0][1][1]
    #for signal_processing
    ROIs = fu.ROI_extraction(aligned_face, aligned_shape)
    green_val = sp.extract_color(ROIs)
    
    data_buffer.append(green_val)
    
    times.append(time.time() - t)
    
    L = len(data_buffer)
    
    if L > BUFFER_SIZE:
        data_buffer = data_buffer[-BUFFER_SIZE:]
        times = times[-BUFFER_SIZE:]
        L = BUFFER_SIZE
    #print(times)
    if L==100:
        
        fps = float(L) / (times[-1] - times[0])
        detrended_data = sp.signal_detrending(data_buffer)
        
        interpolated_data = sp.interpolation(detrended_data, times)
        
        normalized_data = sp.normalization(interpolated_data)
        fft_of_interest, freqs_of_interest = sp.fft(normalized_data, 20)
        max_arg = np.argmax(fft_of_interest)
        bpm = freqs_of_interest[max_arg]
       
        return jsonify({"HR":bpm, "L":L})
    else:
        return jsonify({"HR":0, "L":L})
    
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)
