![Alt text](https://github.com/dazmashaly/Heart-Rate-Monitor/blob/main/test_frame.jpg)


# Methods 
- Detect face, align and get ROI using facial landmarks
- Apply band pass filter with fl = 0.8 Hz and fh = 3 Hz, which are 48 and 180 bpm respectively
- Average color value of ROI in each frame is calculate pushed to a data buffer which is 150 in length
- FFT the data buffer. The highest peak is Heart rate
- Amplify color to make the color variation visible 

# Requirements
```
pip install -r requirements.txt
```

# Implementation
```
The app.py is deployed on a free server (render) and can ba accessed throw this api:


files = {'image': open(image_path, 'rb')}

# Send POST request to API endpoint
response = requests.post('https://heart-monitor-1vor.onrender.com/hr', files=files)
response.json() -> {HR : hr, L: frames in the buffer}

```

# Results
- Data from a specialized device, Compact 5 medical Econet, is used for the ground truth. In certain circumstances, the Heart rate values measured using the application and the device are the same

# Reference
- Real Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam by H. Rahman, M.U. Ahmed, S. Begum, P. Funk
- Remote Monitoring of Heart Rate using Multispectral Imaging in Group 2, 18-551, Spring 2015 by Michael Kellman Carnegie (Mellon University), Sophia Zikanova (Carnegie Mellon University) and Bryan Phipps (Carnegie Mellon University)
- Non-contact, automated cardiac pulse measurements using video imaging and blind source separation by Ming-Zher Poh, Daniel J. McDuff, and Rosalind W. Picard
- Camera-based Heart Rate Monitoring by Janus Nørtoft Jensen and Morten Hannemose
- Signal processing is based on https://github.com/habom2310/Heart-rate-measurement-using-camera

# Note
- Application can only detect HR for 1 person at a time
- Application will return 0 HR until buffer is filled with atleast 100 frames
- Sudden change can cause incorrect HR calculation. In the most case, HR can be correctly detected after 10 seconds being stable infront of the camera
- This github project is for study purpose only.
