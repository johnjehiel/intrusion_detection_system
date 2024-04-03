import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import time
import atexit
import base64

load_dotenv()

app=Flask(__name__)

app.config['MONGO_URI'] = os.getenv('MONGO_URI')

client = MongoClient(app.config['MONGO_URI'])

db = client.get_database('intrusion-detection')

IntrusionLogs = db["IntrusionLogs"]
AuthorizedMembers = db["AuthorizedMembers"]

is_generating_dataset = False
previous_timestamp = -1

def train_classifier(data_dir = "data"):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
     
    faces = []
    ids = []
     
    for image in path:
        # img = Image.open(image).convert('L')
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
         
        faces.append(imageNp)
        ids.append(id)
         
    ids = np.array(ids)
     
    # Train and save classifier
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,ids)
    recognizer.write("classifier.xml")

def release_camera(video_capture):
    video_capture.release()
    cv2.destroyAllWindows()

def detect_face():
    if is_generating_dataset:
        return ""
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        global previous_timestamp
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        
        coords = []

        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
            
            member_id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))
            result = AuthorizedMembers.find_one({"member_id": member_id}, {"name": 1})
            name = result["name"]
            if confidence>75:
                cv2.putText(img, name, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
                timestamp = datetime.now()

                if previous_timestamp == -1 or timestamp - previous_timestamp > timedelta(seconds=3):
                    unknown_face_data = {
                        "timestamp": timestamp,
                        "image": cv2.imencode('.jpg', img[y:y+h, x:x+w])[1].tobytes()  # Convert image to bytes
                    }
                    IntrusionLogs.insert_one(unknown_face_data)
                    previous_timestamp = timestamp
            coords=[x,y,w,h]
        return coords
    
    def recognize(img,clf,faceCascade):
        coords = draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
        return img
        # loading classifier

    cascade_dir = cv2.data.haarcascades
    face_cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(face_cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")
    
    if is_generating_dataset == False:
        video_capture = cv2.VideoCapture(0)
        atexit.register(release_camera, video_capture)

        while True:
            success,frame=video_capture.read()
            if not success:
                break
            else:
                if is_generating_dataset:  # Check if dataset generation is in progress
                    release_camera(video_capture)  # Release camera
                    break
                frame = recognize(frame,recognizer,faceCascade)
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_dataset(name, role, dob):
    global is_generating_dataset

    existingFaceEntryCount = AuthorizedMembers.count_documents({})
    member_id = existingFaceEntryCount + 1

    cascade_dir = cv2.data.haarcascades
    face_cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    face_classifier = cv2.CascadeClassifier(face_cascade_path)

    def face_cropped(img):

        if img is None or len(img) == 0:
            print("image is None", end = ' ')
            return None
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor = 1.3
        # minimum neighbor = 5

        if len(faces) == 0:
            return None
        
        cropped_face = None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]

        return cropped_face
     
    cap = cv2.VideoCapture(0)
    atexit.register(release_camera, cap)
    img_id = 0

    while True:
        ret, frame = cap.read()

        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "data/user."+str(member_id)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)
            print(img_id, end = " ")
            # cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            # ret,buffer=cv2.imencode('.jpg',face)
            # newframe=buffer.tobytes()
            # cv2.imshow("Cropped face", face)
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + newframe + b'\r\n')
        
             
        if int(img_id)==200:
            break

    
    def encode_image(img):
        _, encoded_img = cv2.imencode('.jpg', img)
        return encoded_img.tobytes()

    image_path = os.path.join("data", "user." + str(member_id) + ".10.jpg")  # Assuming the image file is located in the "data" folder

    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            encoded_image = encode_image(img)

            member_details = {
                "member_id": member_id,
                "name": name,
                "role": role,
                "dob": dob,
                "timestamp": datetime.now(),
                "image": encoded_image
            }

            AuthorizedMembers.insert_one(member_details)
            print("Member details inserted successfully.")
        else:
            print("Failed to read the image file.")
    else:
        print("Image file not found.")

    train_classifier()

    print(7)
    release_camera(cap)
    is_generating_dataset = False
    # cap.release()
    # cv2.destroyAllWindows()
    # print("Collecting samples is completed....")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return "lol"
    if (is_generating_dataset): return ""
    return Response(detect_face(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show-intrusion-logs')
def show_intrusion_logs():
    return render_template('intrusion_logs.html')

@app.route('/get-intrusion-logs')
def get_intrusion_logs():
    intrusion_logs = IntrusionLogs.find()  
    intrusion_logs_list = []
    for log in intrusion_logs:
        intrusion_logs_list.append({
            "_id": str(log["_id"]),
            "timestamp": log["timestamp"],
            "image": base64.b64encode(log["image"]).decode('utf-8')
        })
    return jsonify(intrusion_logs_list)

@app.route('/authorized-members')
def authorized_members():
    return render_template('authorized_members.html')

@app.route('/get-authorized-members')
def get_authorized_members():
    authorized_members = AuthorizedMembers.find()  # Assuming `IntrusionLogs` is your MongoDB collection
    authorized_members_list = []
    for member in authorized_members:
        authorized_members_list.append({
            "member_id": member["member_id"],
            "name": member["name"],
            "role": member["role"],
            "dob": member["dob"],
            "timestamp": member["timestamp"],
            "image": base64.b64encode(member["image"]).decode('utf-8')
        })
    return jsonify(authorized_members_list)

@app.route('/add-member')
def add_member():
    return render_template("add_member.html")

@app.route('/scan-member', methods=['GET', 'POST'])
def scan_member():
    global is_generating_dataset
    is_generating_dataset = True
    name = request.json.get('name')
    role = request.json.get('role')
    dob = request.json.get('dob')
    if name is None or role is None or dob is None:
        return jsonify({'message': 'Invalid request. Please provide name, role, and date of birth.'}), 400
    date_obj = datetime.strptime(dob, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%d/%m/%Y")

    # print(name, role, formatted_date)
    # return ""
    generate_dataset(name, role, formatted_date)
    return jsonify({'message': 'Registered successfully.'})



if __name__=="__main__":
    app.run(debug=True)