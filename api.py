from __future__ import print_function # In python 2.7
import os
from flask import Flask, flash, request, redirect, url_for, session, send_file, send_from_directory
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename
from matplotlib import cm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import data, io, filters, measure
from skimage.color import rgb2lab, lab2rgb
from PIL import Image, ImageFilter, ImageDraw
import PIL
from descartes import PolygonPatch
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import json

import sys

np.set_printoptions(threshold=sys.maxsize)

from flask_cors import CORS

import png

app = Flask(__name__)
CORS(app)
api = Api(app)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


UPLOAD_FOLDER = '../uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), UPLOAD_FOLDER)

users = [
    {
        "id": "1",
        "name": "Markus"
    },
    {
        "id": "2",
        "name": "Lennard"
    }
]


class User(Resource):
    def get(self, id):
        for user in users:
            if(user["id"] == id):
                return user, 200
        return "User not found", 404

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Bucket:
    def __init__(self):
        self.elements = []
    def add(self, point):
        self.elements.append(point)
    def get(self):
        return self.elements
    def size(self):
        return len(self.elements)

class Mask:
    def __init__(self, w, h):
        self.map = np.zeros((w,h))
    def set(self, x, y):
        self.map[x][y] = 255
    def getMap(self):
        return self.map

def getMasks(region_labels):
    labels = np.unique(region_labels.flatten() - 1)
    print(labels, file=sys.stderr)
    n_labels = labels.shape[0]
    masks = []
    for i in range(n_labels):
        masks.append(
            Mask(region_labels.shape[0], region_labels.shape[1])
        )
    for x in range(region_labels.shape[0]):
        for y in range(region_labels.shape[1]):
            index = region_labels[x][y] - 1
            masks[index].set(x,y)

    return masks, n_labels

class MyPolygon:
    def __init__(self):
        self.points = []
        self.color = ""
        self.orientation = None
        self.color = "#000000"
    def add(self, point):
        self.points.append(point)
    def add(self, x, y):
        self.points.append((x, y))
    def getPoints(self):
        return self.points
    def setOrientation(self, orientation):
        self.orientation = orientation
    def getOrientation(self):
        return self.orientation
    def setColor(self, color):
        self.color = color
    def getColor(self):
        return self.color

def exportPolygon(polygon):
    text = ""
    color = "#000000"
    outline = []
    points = polygon.getPoints()
    for i in range(len(points)):
        outline.append({"x": int(points[i][0]), "y": int(points[i][1])})
    return {"outline": outline, "text": text, "color": polygon.getColor(), "orientation": polygon.getOrientation().tolist()}

# exports = []
# for i in range(len(polygonList)):
#     exports.append(exportPolygonToJson(polygonList[i]))

def exportPolygonListToJson(myList):
    export = []
    for i in range(len(myList)):
        export.append(exportPolygon(myList[i]))
    return export

def getOrientation(points):

    p_arr = np.asarray(points).reshape((len(points[0]), 2))


    rect = cv2.minAreaRect(p_arr)
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int8(box)


    edge1 = box[0] - box[1]
    edge2 = box[0] - box[3]
    edge1norm = np.linalg.norm(edge1)
    edge2norm = np.linalg.norm(edge2)
    edge = edge1
    if(edge2norm > edge1norm): edge = edge2
    if(edge[0] < 0): edge *= -1
    return edge

def processImage(path, filename):
    n_colors = 12
    img = Image.open(path).convert('RGB')

    scale_1 = 1.0 / img.size[0] * 300

    img = img.resize((int(img.size[0]*scale_1), int(img.size[1] * scale_1)), PIL.Image.ANTIALIAS)

    if(img.size[1] > 300):
        scale_2 = 1.0 / img.size[1] * 300
        img = img.resize((int(img.size[0] * scale_2), int(img.size[1] * scale_2)), PIL.Image.ANTIALIAS)

    # quantize a image
    img = img.quantize(n_colors, method=0)

    # Apply Mode Filter
    n_iterations = 10
    img_mode = img.copy()
    for k in range(n_iterations):
        img_mode = img_mode.filter(ImageFilter.ModeFilter(15))

    img_lowres = img_mode.convert("RGB")
    img_lowres.save(path + "_lowres.jpg", "JPEG")

    class_labels = np.array(img_mode)
    img_arr =     img_mode.convert()

    _img = np.array(img_mode)
    region_labels = measure.label(_img, connectivity=1, background=-1)

    region_labels_img = Image.fromarray(np.uint8(cm.gist_earth(region_labels)*255))
    region_labels_img.save(path + "_labels.jpg", "PNG")

    masks, n_masks = getMasks(region_labels)

    contours = []
    hierarchies = []
    orientations = []
    c = []
    for num in range(n_masks):
        _map = masks[num].getMap().astype('uint8')
        _map_img = Image.fromarray(np.uint8(_map))
        _map_img = _map_img.filter(ImageFilter.MinFilter(5))
        _map = np.array(_map_img)

        mask_img_img = Image.fromarray(np.uint8(cm.gist_earth(_map)*255))
        mask_img_img.save(path + "_mask.jpg", "PNG")


        ret, thresh = cv2.threshold(_map, 127, 255, cv2.THRESH_BINARY)
        print(region_labels, file=sys.stderr)
        print("---------------------------------------------------------------------------------------------------------------", file=sys.stderr)
        contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(contour)
        orientation = getOrientation(contour)
        orientations.append(orientation)
        hierarchies.append(hierarchy)
        c.append(cv2.drawContours(_map, contour, 0, (0,255,0), 3))
    polygonList = []
    for i in range(len(contours)):
        mypoly = MyPolygon()
        mypoly.setOrientation(orientations[i])
        polygonList.append(mypoly)
        myc = contours[i][0]
        for pi in range(len(myc)):
            mypoly.add(myc[pi][0][0], myc[pi][0][1])

    return exportPolygonListToJson(polygonList)

@app.route('/image', methods=['GET', 'POST'])
def upload_file():
    # session.init_app(app)

    if request.method == 'POST':
        print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        print(file)
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return json.dumps({"polygons" : processImage(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename), "link" : "http://localhost:3333/image?filename=" + filename})
    filename = request.args.get("filename")
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                                           filename + "_lowres.jpg" , as_attachment=True)




api.add_resource(User, "/users/<string:id>")
app.run(debug=True, port=3333, host="0.0.0.0")
