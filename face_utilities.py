import cv2
import numpy as np
from collections import OrderedDict
import mediapipe as mp


class Face_utilities():
    '''
    This class contains all needed functions to work with faces in a frame
    '''
    
    def __init__(self, face_width = 200):
        self.landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                        296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                        380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.predictor = None 
       
        
        
        
        self.desiredLeftEye=(0.35, 0.35)
        self.desiredFaceWidth = face_width
        self.desiredFaceHeight = None
        
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
        #For mediapipe to dlib’s 68-point facial landmark detector:
        self.FACIAL_LANDMARKS_68_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])

        
        
       
        
        #FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS
    
    def face_alignment(self, frame, shape):
        '''
        Align the face by vertical axis
        
        Args:
            frame (cv2 image): the original frame. In RGB format.
            shape (array): 68 facial landmarks' co-ords in format of of tuples (x,y)
        
        Outputs:
            aligned_face (cv2 image): face after alignment
        '''
        
       
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = self.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    
        
        leftEyePts = np.array(shape[lStart:lEnd])
        rightEyePts = np.array(shape[rStart:rEnd])
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(frame, M, (w, h),
            flags=cv2.INTER_CUBIC)
            
        #print("1: aligned_shape_1 = {}".format(aligned_shape))
        #print(aligned_shape.shape)
        
        shape = np.reshape(shape,(68,1,2))
            
        # cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                # (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
        # cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                # (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
                
      
                  
        aligned_shape = cv2.transform(shape, M)
        aligned_shape = np.squeeze(aligned_shape)        
            
        # print("---")
        # return aligned_face, aligned_shape
        return aligned_face,aligned_shape
    
    def face_detection(self,frame):
        '''
        Detect faces in a frame
        
        Args:
            frame (cv2 image): a normal frame grab from camera or video
            
        Outputs:
            rects (array): detected faces as rectangles
        '''
        
        if frame is None:
            return
            
        #get all faces in the frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection_:
            results = face_detection_.process(image_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    

                    # Draw a rectangle around the face
                    rects = [[(x,y),(w,h)]]
                    break
        
        return rects
    
    
        
    def get_landmarks(self,frame):
        '''
        Get all facial landmarks in a face 
        
        Args:
            frame (cv2 image): the original frame. In RGB format.
        
        Outputs:
            shape (array): facial landmarks' co-ords in format of of tuples (x,y)
        '''
       
        
        
        if frame is None:
            return None, None
        # all face will be resized to a fix size, e.g width = 200
        #face = imutils.resize(face, width=200)
        # face must be gray
        rects = self.face_detection(frame)
        
        if len(rects)<0 or len(rects)==0:
            return None, None
        try:
            with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection_:
                with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
                    results = face_detection_.process(frame)
                    if results.detections:
                        for detection in results.detections:
                            # Get face landmarks
                            face_landmarks = face_mesh.process(frame).multi_face_landmarks[0]
                            shape = []
                            for index in self.landmark_points_68:
                                x = int(face_landmarks.landmark[index].x * frame.shape[1])
                                y = int(face_landmarks.landmark[index].y * frame.shape[0])
                                shape.append((x, y))
                            break
        except:
            return None, None
        # in shape, there are 68 pairs of (x, y) carrying coords of 68 points.
        # to draw landmarks, use: for (x, y) in shape: cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        
        return shape, rects
    
    def ROI_extraction(self, face, shape):
        '''
        Extract 2 cheeks as the ROIs
        
        Args:
            face (cv2 image): face cropped from the original frame. In RGB format.
            shape (array): facial landmarks' co-ords in format of of tuples (x,y)
            
        Outputs:
            ROI1 (cv2 image): right-cheek pixels
            ROI2 (cv2 image): left-cheek pixels
        '''
        ROI1 = face[shape[29][1]:shape[33][1], #right cheek
                shape[54][0]:shape[12][0]]
                
        ROI2 =  face[shape[29][1]:shape[33][1], #left cheek
                shape[4][0]:shape[48][0]]
    
                
        return ROI1, ROI2        
   
    
    
    def no_age_gender_face_process(self, frame):
        '''
        full process to extract face, ROI but no age and gender detection
        
        Args:
            frame (cv2 image): input frame 
            
        Outputs:
            rects (array): detected faces as rectangles
            face (cv2 image): face
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): facial landmarks' co-ords of the aligned face in format of tuples (x,y)
        
        '''
       
        shape, rects = self.get_landmarks(frame)
        if shape is None:
            return None
        (x, y, w, h) = rects[0][0][0], rects[0][0][1], rects[0][1][0], rects[0][1][1]
        
        face = frame[y:y+h,x:x+w]
        aligned_face,aligned_shape = self.face_alignment(frame, shape)
        
       
                
        return rects, face, shape, aligned_face, aligned_shape
        
    
