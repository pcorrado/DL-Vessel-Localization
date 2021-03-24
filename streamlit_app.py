import sys
import os
import numpy as np
import scipy
import cv2
from common.utils import readPlaneLocations, readCase
import matplotlib.pyplot as plt
sys.path.append('/export/home/pcorrado/.local/bin')
import streamlit as st
import OneShotCNN.Model as oneShotModel
import OneShotCNN.DataGenerator as generator
import tensorflow as tf
import requests
import tarfile

def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("demo_instructions.md"))

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        showFigures()
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():

    img = readCaseCached()
    # Manually annotated plane location, size, and normal vector for the aorta, mpa, svc, and ivc
    label = np.array([[60.43185042,	52.30437038,	54.49545298,	-15.18513197,	0.002,	      -13.0158274],
                      [71.9789919,	    47.29195794,	53.70544135,	0.760368264,	10.9279566,	  -11.66197259],
                      [50.22421655,	60.22166319,	50.91803201,	0.497330459,	-0.129592946, 9.986784672],
                      [52.68816532,	67.17255745,	81.37310384,	2.825293567,	-2.881176231, -11.30117427]])

    x, y, z = location_selector_ui()

    labels2 = None
    if st.checkbox('Run CNN to locate vessels.'):
        labels2 = runCNN(img)
        st.write('Red: Manually placed planes.')
        st.write('Yellow: CNN Predictions.')
    else:
        labels2 = None
        st.write('Red: Manually placed planes.')

    # Display image slices and vessel cut planes
    st.image(np.concatenate((formatImage(img[x,:,:,1], label, labels2, 0, x, y, z), \
                             formatImage(img[:,y,:,1], label, labels2, 1, y, x, z), \
                             formatImage(img[:,:,z,1], label, labels2, 2, z, x, y)), axis=1), width=768)

# Choose image slices for each view
def location_selector_ui():
    st.sidebar.markdown("Pick image location")

    # Choose a frame out of the selected frames.
    x = st.sidebar.slider("X", 0, 128 - 1, 53)
    y = st.sidebar.slider("Y", 0, 128 - 1, 55)
    z = st.sidebar.slider("Z", 0, 128 - 1, 58)

    return x,y,z

# Read in demo data set, this will happen only when the app is loaded.
@st.cache(suppress_st_warning=True)
def readCaseCached():
    return np.load('./demo_case.npy').astype(np.float32)

# Calculates plane endpoints and arrow location for displaying on the given image slice.
@st.cache(suppress_st_warning=True)
def getPlaneIntersection(label,dim,slice):
    planes = []
    for v in range(label.shape[0]):
        z = label[v,dim]
        length = np.sqrt(label[v,3]**2 + label[v,4]**2 + label[v,5]**2)
        labelV = np.delete(np.delete(label[v,:],dim+3),dim)
        if np.abs(z-slice)<(length/2):
            x = labelV[0]
            y = labelV[1]
            nx = labelV[2]
            ny = labelV[3]
            s = np.sqrt(length**2-(2*(z-slice))**2)
            dx = ny/np.sqrt(nx**2+ny**2)
            dy = -nx/np.sqrt(nx**2+ny**2)
            planes.append([y-s*0.5*dy, y+s*0.5*dy, x-s*0.5*dx, x+s*0.5*dx, y, y+0.5*ny, x, x+0.5*nx])
    return planes

# Display image slices and plane annotations
@st.cache(suppress_st_warning=True)
def formatImage(img, labels, labels2, dim, slice,x,y):
    img = np.tile(np.expand_dims(np.uint8(scipy.ndimage.zoom(np.clip(np.squeeze(img)/np.percentile(img, 99.5),0.0,1.0), 2, order=0)*255),axis=2),(1,1,3))
    img = cv2.line(img, (2*x, 0), (2*x, img.shape[1]), (255,255,255), 1)
    img = cv2.line(img, (0, 2*y), (img.shape[0], 2*y), (255,255,255), 1)

    for pl in getPlaneIntersection(labels,dim,slice):
        img = cv2.line(img, (int(2*pl[0]), int(2*pl[2])), (int(2*pl[1]), int(2*pl[3])), (255,0,0), 2)
        img = cv2.arrowedLine(img, (int(2*pl[4]), int(2*pl[6])), (int(2*pl[5]), int(2*pl[7])), (255,0,0), 3)
    if labels2 is not None:
        for pl in getPlaneIntersection(labels2,dim,slice):
            img = cv2.line(img, (int(2*pl[0]), int(2*pl[2])), (int(2*pl[1]), int(2*pl[3])), (255,255,0), 2)
            img = cv2.arrowedLine(img, (int(2*pl[4]), int(2*pl[6])), (int(2*pl[5]), int(2*pl[7])), (255,255,0), 3)
    return img

# Load the CNN model and then run it with the demo image to predict plane locations.
@st.cache(suppress_st_warning=True)
def downloadModelWeights(id):
    st.write('Downloading model weights.')
    download_file_from_google_drive(id, './model_weights.tar.gz')
    my_tar = tarfile.open('./model_weights.tar.gz')
    my_tar.extractall('.')
    my_tar.close()
    myModel = tf.keras.models.load_model('./OneShot_batch_64/', custom_objects={"myLossFunction": oneShotModel.myLossFunction})
    os.remove('./model_weights.tar.gz')
    os.system('rm -rf {}'.format('./OneShot_batch_64/'))
    return myModel

# Load the CNN model and then run it with the demo image to predict plane locations.
@st.cache(suppress_st_warning=True)
def runCNN(img):
    st.write('Running CNN to predict vessel locations (will execute only once). May take up to 1 minute.')
    myModel = downloadModelWeights('1vMWAOYf5q5_M4374K5qsQhlFn7yF4RYX')
    gen = generator.DataGenerator(["_"], {"_": np.zeros((8,7))}, images=np.expand_dims(img,axis=0), shuffle=False)
    pred, _ = oneShotModel.MyModel.predictFullImages(myModel, gen)
    pred = np.array(pred["_"])
    pred[:,4] = pred[:,4]*pred[:,3]
    pred[:,5] = pred[:,5]*pred[:,3]
    pred[:,6] = pred[:,6]*pred[:,3]
    return np.delete(pred[0:4,:],3, axis=1)

# Display instructions on screen.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    f = open(path, "r")
    text = f.read()
    f.close()
    return text

def showFigures():
    st.write("\n\n\n")
    st.image("./figures/Fig1.JPG")
    st.write("\n\n\n")
    st.image("./figures/Fig2.JPG")
    st.write("\n\n\n")
    st.image("./figures/Fig3.JPG")
    st.write("\n\n\n")
    st.image("./figures/Fig4.JPG", width=512)
    st.write("\n\n\n")
    st.image("./figures/Fig5.JPG", width=512)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    main()
