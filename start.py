import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from playsound import playsound

mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=1)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


model = keras.models.load_model('model/keypoint_classifier_5_yoga_pose.hdf5')

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (100, 185)
org1 = (100, 285)
  
# fontScale
fontScale = 1
   
# Red color in BGR
color = (0, 255, 0)
  
# Line thickness of 2 px
thickness = 2

def detectPose(image, pose):

    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = np.array([])
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks = np.append(landmarks, landmark.x)
            landmarks = np.append(landmarks, landmark.y)
            landmarks = np.append(landmarks, landmark.z)

    return landmarks, results

def downdog():
    camera = cv2.VideoCapture(0)
    while True:
        sucess, frame = camera.read()
        if not sucess:
            break
        else :
            
            frame = cv2.flip(frame, 1)
                
            coord, results = detectPose(frame, pose)
            
            if len(coord) == 99:
                reshape_coord = np.array([coord])
                predicted_pose = np.argmax(model.predict(reshape_coord))
                    
                if predicted_pose == 0:
                    # Define the reference landmarks for the yoga pose
                    REF_LANDMARKS = np.array([ 0.40649787,  0.64131659, -0.0738324 ,  0.39140034,  0.64962232, -0.11281894,  0.39036965,  0.64704812, -0.11282347,  0.38911515, 0.64461964, -0.11289372,  0.39101368,  0.65024519, -0.05913403,0.38989538,  0.64800686, -0.05905756,  0.38852233,  0.64587933,-0.05899595,  0.37956071,  0.61289918, -0.21922262,  0.37807664,0.61600602,  0.02902312,  0.41136178,  0.61520803, -0.1048842 ,0.41260371,  0.61533576, -0.03301336,  0.39537975,  0.51326185,-0.29495302,  0.3985447 ,  0.49927324,  0.13853624,  0.31691265,0.68898022, -0.50483251,  0.31332666,  0.65613759,  0.20553873,0.22333047,  0.85001862, -0.49337167,  0.23655665,  0.78864294,0.00683151,  0.20259583,  0.86170286, -0.55497301,  0.20498687,0.80929148, -0.00377477,  0.19411033,  0.85497028, -0.51594168,0.20165375,  0.81442511, -0.04806514,  0.20195964,  0.85135734,-0.48007891,  0.21343514,  0.80899394, -0.01722329,  0.58472174,0.17383119, -0.18048529,  0.58111298,  0.1728348 ,  0.18098997,0.69695377,  0.4719843 , -0.1141892 ,  0.68486571,  0.46729305,0.20942803,  0.80145192,  0.74790329, -0.05576112,  0.78363812,0.72172821,  0.31902122,  0.83246815,  0.80093664, -0.0574379 ,0.80505633,  0.76623869,  0.32469895,  0.72996128,  0.85325259,-0.19753887,  0.71656936,  0.83155984,  0.23756176])

                    # Calculate the Euclidean distance between each detected landmark and the corresponding reference landmark
                    def calculate_distance(detected_landmarks, ref_landmarks):
                        distances = np.sqrt(np.sum((detected_landmarks - ref_landmarks) ** 2, axis=0))
                        return distances

                    # Compute the accuracy score based on the average distance normalized by the maximum possible distance
                    def calculate_accuracy(detected_landmarks, ref_landmarks):
                        distances = calculate_distance(detected_landmarks, ref_landmarks)
                        accuracy = 1 - np.mean(distances) / np.sqrt(len(ref_landmarks))
                        return accuracy*100

                    # Define the detected landmarks for the pose
                    DETECTED_LANDMARKS = coord

                    # Calculate the accuracy score for the pose
                    accuracy = calculate_accuracy(DETECTED_LANDMARKS, REF_LANDMARKS)

                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,250,0), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, "Accuracy: {:.0f} %".format(accuracy+5), org1, font, fontScale, color, thickness, cv2.LINE_AA)

                else:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,250), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, 'Wrong pose', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    playsound('static\phone-beep-fx.wav')



            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def tree():
    camera = cv2.VideoCapture(0)
    while True:
        sucess, frame = camera.read()
        if not sucess:
            break
        else :
            frame = cv2.flip(frame, 1)
                
            coord, results = detectPose(frame, pose)
            
            if len(coord) == 99:
                reshape_coord = np.array([coord])
                predicted_pose = np.argmax(model.predict(reshape_coord))
                    
                if predicted_pose == 3:
                    # Define the reference landmarks for the yoga pose
                    REF_LANDMARKS = np.array([ 0.34954551,  0.26988155, -0.53814518,  0.35935581,  0.25811541,-0.49295911,  0.36695766,  0.25785768, -0.49296272,  0.37371388,0.25766721, -0.49318302,  0.33805355,  0.25884345, -0.4998394 ,0.33090723,  0.25886202, -0.49992263,  0.32513377,  0.25921375,-0.50003129,  0.3829565 ,  0.26376069, -0.26882762,  0.3188493 ,0.26609749, -0.29498062,  0.3641201 ,  0.28244105, -0.45505026,0.33672562,  0.28242463, -0.46334282,  0.41973439,  0.32303873,-0.18242796,  0.29010248,  0.32424578, -0.33224902,  0.42980516,0.21602419, -0.29265618,  0.28917739,  0.22305036, -0.53966397,0.38039431,  0.11881831, -0.39594647,  0.33369192,  0.12180063,-0.57804668,  0.37226272,  0.09380782, -0.43682587,  0.34539858,0.09491271, -0.63758475,  0.36802909,  0.09294137, -0.43205222,0.34478512,  0.0925115 , -0.60615039,  0.3686175 ,  0.10232943,-0.40337798,  0.34375906,  0.10216552, -0.57557178,  0.40914237,0.51556253,  0.06612955,  0.31658587,  0.53446484, -0.06618252,0.59215713,  0.58403814, -0.24052267,  0.35256341,  0.6826793 ,-0.22037894,  0.40830255,  0.59253401,  0.27277866,  0.37916705,0.83930826,  0.09947404,  0.37360612,  0.57131869,  0.32424793,0.39085481,  0.85919166,  0.12288759,  0.36553112,  0.62971252, 0.24902873,  0.36709443,  0.87506258, -0.08671939])                    
                    
                    
                    # Calculate the Euclidean distance between each detected landmark and the corresponding reference landmark
                    def calculate_distance(detected_landmarks, ref_landmarks):
                        distances = np.sqrt(np.sum((detected_landmarks - ref_landmarks) ** 2, axis=0))
                        return distances

                    # Compute the accuracy score based on the average distance normalized by the maximum possible distance
                    def calculate_accuracy(detected_landmarks, ref_landmarks):
                        distances = calculate_distance(detected_landmarks, ref_landmarks)
                        accuracy = 1 - np.mean(distances) / np.sqrt(len(ref_landmarks))
                        return accuracy*100

                    # Define the detected landmarks for the pose
                    DETECTED_LANDMARKS = coord

                    # Calculate the accuracy score for the pose
                    accuracy = calculate_accuracy(DETECTED_LANDMARKS, REF_LANDMARKS)

                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,250,0), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, "Accuracy: {:.0f} %".format(accuracy+5), org1, font, fontScale, color, thickness, cv2.LINE_AA)
    
                else:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,250), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, 'Wrong pose', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    playsound('static\phone-beep-fx.wav')
                # cv2.imshow('Yoga', frame)
                # cv2.waitKey(1)



            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def warrior2():
    camera = cv2.VideoCapture(0)
    while True:
        sucess, frame = camera.read()
        if not sucess:
            break
        else :
            frame = cv2.flip(frame, 1)
                
            coord, results = detectPose(frame, pose)
            
            if len(coord) == 99:
                reshape_coord = np.array([coord])
                predicted_pose = np.argmax(model.predict(reshape_coord))
                    
                if predicted_pose == 4:
                    # Define the reference landmarks for the yoga pose
                    REF_LANDMARKS = np.array([ 0.47520137,  0.24952373, -0.14433497,  0.48577982,  0.22990593,-0.13984787,  0.48980924,  0.22965203, -0.13982569,  0.49377397,0.22965017, -0.13980833,  0.47979438,  0.22913116, -0.11674455,0.47935688,  0.22834544, -0.1168258 ,  0.47912729,  0.22745222,-0.11680772,  0.51226878,  0.2395031 , -0.11269749,  0.49446213,0.23646183, -0.00198019,  0.48383501,  0.26965755, -0.13686918,0.47642794,  0.26837271, -0.10480082,  0.57032782,  0.35372004,-0.11988246,  0.453944  ,  0.34564614,  0.02776575,  0.67040062,0.34388638, -0.17331095,  0.3621121 ,  0.33959091,  0.0507685 ,0.75088978,  0.34071881, -0.27766258,  0.28433567,  0.33121079,-0.0756111 ,  0.77628565,  0.34432349, -0.30472714,  0.2575722 ,0.33686388, -0.10284465,  0.77787924,  0.34086713, -0.3416813 ,0.25538832,  0.33175462, -0.1396777 ,  0.76778281,  0.34231511,-0.29991347,  0.26371971,  0.33262032, -0.10054167,  0.56747735,0.6240294 , -0.06747307,  0.49930257,  0.62545711,  0.0672813 ,0.68366796,  0.73325229, -0.14978184,  0.37945122,  0.64455354,-0.0562252 ,  0.78776443,  0.84013373, -0.05997738,  0.36327404,0.84844762,  0.03515018,  0.79812014,  0.85583568, -0.05608364,0.37387633,  0.8735745 ,  0.03903206,  0.79622877,  0.88151002,-0.19250694,  0.29990858,  0.87609088, -0.04093421])
                    
                    # Calculate the Euclidean distance between each detected landmark and the corresponding reference landmark
                    def calculate_distance(detected_landmarks, ref_landmarks):
                        distances = np.sqrt(np.sum((detected_landmarks - ref_landmarks) ** 2, axis=0))
                        return distances

                    # Compute the accuracy score based on the average distance normalized by the maximum possible distance
                    def calculate_accuracy(detected_landmarks, ref_landmarks):
                        distances = calculate_distance(detected_landmarks, ref_landmarks)
                        accuracy = 1 - np.mean(distances) / np.sqrt(len(ref_landmarks))
                        return accuracy*100

                    # Define the detected landmarks for the pose
                    DETECTED_LANDMARKS = coord

                    # Calculate the accuracy score for the pose
                    accuracy = calculate_accuracy(DETECTED_LANDMARKS, REF_LANDMARKS)

                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,250,0), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, "Accuracy: {:.0f} %".format(accuracy+5), org1, font, fontScale, color, thickness, cv2.LINE_AA)
    
                else:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,250), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, 'Wrong pose', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    playsound('static\phone-beep-fx.wav')
                # cv2.imshow('Yoga', frame)
                # cv2.waitKey(1)



            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def goddess():
    camera = cv2.VideoCapture(0)
    while True:
        sucess, frame = camera.read()
        if not sucess:
            break
        else :
            frame = cv2.flip(frame, 1)
                
            coord, results = detectPose(frame, pose)
            
            if len(coord) == 99:
                reshape_coord = np.array([coord])
                predicted_pose = np.argmax(model.predict(reshape_coord))
                    
                if predicted_pose == 1:
                    # Define the reference landmarks for the yoga pose
                    REF_LANDMARKS = np.array([ 0.50665361,  0.11738768, -0.08061411,  0.5152424 ,  0.09527943, -0.04192471,  0.52016795,  0.09442672, -0.04188844,  0.52583289,0.09392014, -0.04202097,  0.49596509,  0.09735033, -0.04215175,0.48921239,  0.09837058, -0.04218635,  0.48308259,  0.0995529 ,-0.04211694,  0.53373533,  0.10352847,  0.12881747,  0.47640023,0.11118937,  0.13052569,  0.51761156,  0.14148831, -0.01959713,0.49560493,  0.1438992 , -0.01961447,  0.58875757,  0.26908597,0.18633007,  0.42952502,  0.27933994,  0.15108036,  0.64581603,0.40572405,  0.0702468 ,  0.37326667,  0.40874749,  0.01915436,0.52580422,  0.38481003, -0.02726313,  0.49769551,  0.38589525,-0.059114  ,  0.51255167,  0.35198691, -0.05660341,  0.51716429,0.35367829, -0.08619601,  0.51284504,  0.34575963, -0.02154803,0.51471782,  0.3488465 , -0.06404831,  0.51481515,  0.35456908,-0.013232  ,  0.51049095,  0.35831004, -0.04885354,  0.56256109,0.57768959,  0.00597104,  0.4604333 ,  0.57662976, -0.00607474,0.7164886 ,  0.6791057 , -0.28523907,  0.30624896,  0.6912055 ,-0.24843559,  0.69848204,  0.90131563,  0.08580762,  0.32071817,0.91555911,  0.0919673 ,  0.67873591,  0.93321532,  0.11872505,0.33880651,  0.94951051,  0.12150063,  0.7687602 ,  0.9585973 ,-0.00334202,  0.25046822,  0.96907806, -0.00618722])

                    # Calculate the Euclidean distance between each detected landmark and the corresponding reference landmark
                    def calculate_distance(detected_landmarks, ref_landmarks):
                        distances = np.sqrt(np.sum((detected_landmarks - ref_landmarks) ** 2, axis=0))
                        return distances

                    # Compute the accuracy score based on the average distance normalized by the maximum possible distance
                    def calculate_accuracy(detected_landmarks, ref_landmarks):
                        distances = calculate_distance(detected_landmarks, ref_landmarks)
                        accuracy = 1 - np.mean(distances) / np.sqrt(len(ref_landmarks))
                        return accuracy*100

                    # Define the detected landmarks for the pose
                    DETECTED_LANDMARKS = coord

                    # Calculate the accuracy score for the pose
                    accuracy = calculate_accuracy(DETECTED_LANDMARKS, REF_LANDMARKS)

                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,250,0), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, "Accuracy: {:.0f} %".format(accuracy+5), org1, font, fontScale, color, thickness, cv2.LINE_AA)
    
                else:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,250), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, 'Wrong pose', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    playsound('static\phone-beep-fx.wav')
                # cv2.imshow('Yoga', frame)
                # cv2.waitKey(1)



            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def plank():
    camera = cv2.VideoCapture(0)
    while True:
        sucess, frame = camera.read()
        if not sucess:
            break
        else :
            frame = cv2.flip(frame, 1)
                
            coord, results = detectPose(frame, pose)
            
            if len(coord) == 99:
                reshape_coord = np.array([coord])
                predicted_pose = np.argmax(model.predict(reshape_coord))
                    
                if predicted_pose == 2:
                    # Define the reference landmarks for the yoga pose
                    REF_LANDMARKS = np.array([ 0.81792843,  0.40310267, -0.06687889,  0.82744181,  0.38259691,-0.05526715,  0.8271755 ,  0.38084149, -0.05536072,  0.82688868,0.37843305, -0.05544443,  0.82608473,  0.37854415, -0.09741729,0.82471609,  0.37384048, -0.09743343,  0.82308847,  0.36872941,-0.09745654,  0.81054765,  0.35201925,  0.02774313,  0.80628139,0.3451103 , -0.15984529,  0.79868245,  0.40916979, -0.02912244,0.79685187,  0.40264398, -0.08365995,  0.70064265,  0.37499261,0.16502902,  0.7057575 ,  0.39089632, -0.24107817,  0.6889528 ,0.56714702,  0.17963441,  0.69105279,  0.58353221, -0.27760562,0.67553365,  0.7330091 ,  0.08314355,  0.67687619,  0.76343328,-0.24083635,  0.69934386,  0.73999524,  0.07896398,  0.70219117,0.78015363, -0.28879267,  0.70344907,  0.74302119,  0.03859566,0.70702326,  0.76967573, -0.25790772,  0.69374835,  0.74240977,0.06151484,  0.69876915,  0.76423335, -0.23018055,  0.49206978,0.47128963,  0.11574665,  0.48991147,  0.4870984 , -0.11566976,0.32380205,  0.58671415,  0.14791231,  0.31710422,  0.58675826,-0.07158129,  0.15524229,  0.67390811,  0.2304717 ,  0.15027356,0.67762804,  0.00869495,  0.12009275,  0.66332126,  0.23392184,0.11267438,  0.66779226,  0.00996997,  0.14523217,  0.76767141,0.1530181 ,  0.13929546,  0.77664256, -0.09361399])
                    
                    # Calculate the Euclidean distance between each detected landmark and the corresponding reference landmark
                    def calculate_distance(detected_landmarks, ref_landmarks):
                        distances = np.sqrt(np.sum((detected_landmarks - ref_landmarks) ** 2, axis=0))
                        return distances

                    # Compute the accuracy score based on the average distance normalized by the maximum possible distance
                    def calculate_accuracy(detected_landmarks, ref_landmarks):
                        distances = calculate_distance(detected_landmarks, ref_landmarks)
                        accuracy = 1 - np.mean(distances) / np.sqrt(len(ref_landmarks))
                        return accuracy*100

                    # Define the detected landmarks for the pose
                    DETECTED_LANDMARKS = coord

                    # Calculate the accuracy score for the pose
                    accuracy = calculate_accuracy(DETECTED_LANDMARKS, REF_LANDMARKS)

                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,250,0), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, "Accuracy: {:.0f} %".format(accuracy+5), org1, font, fontScale, color, thickness, cv2.LINE_AA)
    
                else:
                    mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing.DrawingSpec(color=(250,0,0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,250), thickness=2, circle_radius=2))
                    frame = cv2.putText(frame, 'Wrong pose', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    playsound('static\phone-beep-fx.wav')
                # cv2.imshow('Yoga', frame)
                # cv2.waitKey(1)



            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')