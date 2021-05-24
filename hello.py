# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import pickle
import cv2
from flask import Flask,request, render_template
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import sys
import os.path
import glob
app = Flask(__name__, static_url_path='')
@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("[INFO] loading encodings...")
        data = pickle.loads(open('encodings.pickle', "rb").read())
        # load the input image, convert it from BGR to RGB channel ordering,
        # and use Tesseract to localize each area of text in the input image
        image = cv2.imread(file_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load the known faces and embeddings
        
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,
        	model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        # initialize the list of names for each face detected
        names = []
        
        # loop over the facial embeddings
        for encoding in encodings:
        	# attempt to match each face in the input image to our known
        	# encodings
        	matches = face_recognition.compare_faces(data["encodings"],
        		encoding)
        	name = "Unknown"
        
        	# check to see if we have found a match
        	if True in matches:
        		# find the indexes of all matched faces then initialize a
        		# dictionary to count the total number of times each face
        		# was matched
        		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        		counts = {}
        
        		# loop over the matched indexes and maintain a count for
        		# each recognized face face
        		for i in matchedIdxs:
        			name = data["names"][i]
        			counts[name] = counts.get(name, 0) + 1
        
        		# determine the recognized face with the largest number of
        		# votes (note: in the event of an unlikely tie Python will
        		# select first entry in the dictionary)
        		name = max(counts, key=counts.get)
        	
        	# update the list of names
        	names.append(name)
        
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
        	# draw the predicted face name on the image
        	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        	y = top - 15 if top - 15 > 15 else top + 15
        	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        		0.75, (0, 255, 0), 2)
        
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    return ','.join(map(str, names))
if __name__ == '__main__':
      port = int(os.getenv('PORT', 8000))
     #app.run(host='0.0.0.0', port=port, debug=True)
      http_server = WSGIServer(('0.0.0.0', port), app)
      http_server.serve_forever()