from scipy.spatial import distance as dist
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import dlib
#import playsound
import time
import cv2
import pyttsx3 as pyttsx
#import engineio as k
import speech_recognition as sr
import face_recognition

my_image = face_recognition.load_image_file("IMG_20170728_122617.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

known_face_encodings=[my_face_encoding]
known_face_names=['Myself']

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

out=0
count_me=0
count_frame=0
count_time=0
count_frame_thresh=60

font = cv2.FONT_HERSHEY_SIMPLEX

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):

    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2*C)
    return ear
def mouth_aspect_ratio(mouth):
        
        A=dist.euclidean(mouth[1],mouth[5])
        B=dist.euclidean(mouth[2],mouth[4])
        C=dist.euclidean(mouth[0],mouth[3])
        mar=(A+B)/(2*C)
        return mar

def dist_for_eff(leftEye,rightEye,mouth):
        
        A=dist.euclidean(rightEye[3],mouth[0])
        B=dist.euclidean(leftEye[0],mouth[3])
        D=dist.euclidean(mouth[0],mouth[3])
        C=(A+B)/(2*D)
        return C

PRED_PATH="shape_predictor_68_face_landmarks.dat"

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 15

COUNTER = 0
ALARM_ON = False

print("... loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("...starting video stream thread...")
vs = VideoStream(0).start()
#time.sleep(0.5)

process_this_frame = True

while True:
    flag=0
    frame = vs.read()
    frame1=vs.read()
    frame = imutils.resize(frame, width=450)
    small_frame = cv2.resize(frame1,None, fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    if process_this_frame:
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names = []
    
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
            name = "Unknown"

            if True in matches:
                flag=1
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)
    
    process_this_frame = not process_this_frame 

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthEAR = mouth_aspect_ratio(mouth)

        ear = (leftEAR + rightEAR) / 2.0
        if ear:
            #flag=1
            count_me+=1
            dist_eff=2.2
            dist_eff1=2.3
            if leftEAR>0.3 and rightEAR>0.3 and ear>0.35:
                    dist_eff2=dist_for_eff(leftEye,rightEye,mouth)
                    dist_eff=dist_eff2
            else:
                dist_eff1=dist_for_eff(leftEye,rightEye,mouth)        
            if dist_eff:
               diff=dist_eff-dist_eff1       
            else:
                diff=2.2-dist_eff1
                    
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [mouthHull], -1 , (0,255,0), 1)

        if leftEAR< EYE_AR_THRESH and rightEAR< EYE_AR_THRESH and ear<0.23 and diff<0.4:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    
                    '''if sound_alarm:
                        sound_alarm(path='alarm.wav')'''
                    with sr.Microphone() as source:
                        k=pyttsx.init()
                        k.say('Hey! I guess... you feeling drowsy??')
                        k.runAndWait()
                        

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame,'C:{:.2f}'.format(diff), (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        '''for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)'''

    if flag==1 and count_me==1:
        cv2.imwrite('driver_me.png',frame)
        k=pyttsx.init()
        r=sr.Recognizer()
        with sr.Microphone() as source:
            k=pyttsx.init()
            k.say('Heyyy Praanshoo! Welcome... Drive safe.')
            k.runAndWait()
        count_me+=1
        
    elif flag==0 and count_me==1:
        cv2.imwrite('driver.png',frame)
        k=pyttsx.init()
        r=sr.Recognizer()
        with sr.Microphone() as source:
            k=pyttsx.init()
            k.say('Heyyy Didnot recognize you. Anyway Welcome... Drive safe.')
            k.runAndWait()
        count_me+=1
        
    if count_me>1:
        if flag==0:
            out=0
            count_frame+=1
            if count_frame>=count_frame_thresh or count_time>1:
                if count_frame==count_frame_thresh:
                    with sr.Microphone() as source:
                        k=pyttsx.init()
                        k.say('Heyy! What are you doing? Focus on driving.')
                        k.runAndWait()
                    count_frame=0
                    count_time+=1
                out=1
                                   
        elif flag==1 and out==1:
            count_frame=0
            print(count_time)
            if 1< count_time <= 3:
                with sr.Microphone() as source:
                    k=pyttsx.init()
                    k.say('Hmmm good.')
                    k.runAndWait()
                count_time=0
            if 4<= count_time:
                with sr.Microphone() as source:
                    k=pyttsx.init()
                    k.say('Where were you? Always switch the car off before you leave.')
                    k.runAndWait()
                count_time=0
                cv2.imwrite('driver_update.png',frame)
                #break
        cv2.putText(frame,'count_time={:.2f}'.format(count_time), (20,40), font, 0.6, (0, 255, 255), 1)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame1, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #cv2.imshow('Video', frame1)           
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
