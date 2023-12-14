from flask import Flask, render_template, Response, request, jsonify
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import time

app = Flask(__name__)

def loadAugImages(path):
    """
    :param path : folder in which all the marker images with ids are stored
    : return : dictionary with key as the id and values

    """

    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of Markers Detected: ",noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def findArucoMarkers(img,markerSize=6, totalMarkers=250, draw=True):
    """
    :param image in which to find the aruco makers
    :param markersize : the size of the markers
    :param totalMarkers : total number of markers that compose the dictionary
    :param draw : flag to draw bbox around markers detected
    : return : bounding boxexs and id numbers of markers detected

    """
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParam)
    bboxs,ids, rejected = detector.detectMarkers(imgGray)

    print(ids)

    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs, ids]



def augmentAruco(bbox,id, img,imgAug, drawId=True):
    """
    :param bbox : four corner points of box
    :param id : marker id of the corresponding box used only for display
    :param img : final image on which to draw
    :param the image that will be overlapped on the marker
    :param drawId : flag t display id of the marker
    : return : image with the augment image

    """
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]) )
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]


    h,w,c = imgAug.shape

    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])

    matrix, _ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))
    imgOut = img + imgOut

    if drawId:
        cv2.putText(imgOut,str(id), tl, cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)

    return imgOut
    
def generate_frames():
    # cap = cv2.VideoCapture(0)
    # while not cap.isOpened():
    #     cap = cv2.VideoCapture(0)
    #     cv2.waitKey(1000)
    #     print("Wait for the camera to start")
    augDics = loadAugImages("Markers")

    while True:
        # Receive image data from the client
        content = request.get_json(silent=True)
        if content and 'image_data' in content:
            image_data = content['image_data']
            img_data = np.array(image_data, dtype=np.uint8)
            img = img_data.reshape((480, 640, 4))  # Assuming 4 channels (RGBA)

            arucoFound = findArucoMarkers(img)

            if len(arucoFound[0]) != 0:
                for bbox, id in zip(arucoFound[0], arucoFound[1]):
                    if int(id) in augDics.keys():
                        img = augmentAruco(bbox, id, img, augDics[int(id)])

            ret, frame = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            data = frame.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

        else:
            time.sleep(0.1)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
