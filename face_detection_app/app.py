import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np
import os


@st.cache
def load_image(img):
    im = Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('openface/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('openface/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('openface/haarcascade_smile.xml')

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
    faces = face_cascade.detectMultiScale(gray,1.1,4)
	# Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0),2)
        return img,faces 

def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detect_smiles(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img


def cartonize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Edges
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	#Color
	color = cv2.bilateralFilter(img, 9, 300, 300)
	#Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon


def cannize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny



def app():
    st.title('Face detection and image editor app')
    activities = ['Edit','Detection','About']
    choice = st.sidebar.selectbox('Select activities',activities)

    #image editor
    if choice == 'Edit':
        st.subheader('Image Editor')
        image_file = st.file_uploader("Upload Image",type = ['jpg','png','jpeg'],encoding =None)

        if image_file is not None:
            our_image = Image.open(image_file)
            st.header('Original Image')
            st.image(our_image,width=400)

        enhance_type = st.sidebar.radio('Enhance Type',['Original','Gray-scale','Contrast','Brightness','Bluring'])
        if enhance_type == 'Gray-scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.header('Your Edited Image is Here!')
            st.image(img,width=400)

        elif enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Constrast',0.5,4.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.header('Your Edited Image is Here!')
            st.image(img_output,width=400)

        elif enhance_type == 'Brightness':
            br_rate = st.sidebar.slider('Brightness',0.5,4.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_out = enhancer.enhance(br_rate)
            st.header('Your Edited Image is Here!')
            st.image(img_out,width=400)
        
        elif enhance_type == 'Bluring':
            new_img = np.array(our_image.convert('RGB'))
            b_rate = st.sidebar.slider('Bluring',0.5,4.5)
            img = cv2.cvtColor(new_img,1)
            blur_img = cv2.GaussianBlur(img,(11,11),b_rate)
            st.header('Your Blured image is here!')
            st.image(blur_img,width=400)
        else:
            st.header('Your Original Image is Here!')
            st.image(our_image,width=400)

 #=================================================================================================

    elif choice == 'Detection':
        st.subheader('Face Detection')
        tasks = ["Faces","Smiles","Eyes","Cannize","Cartonize"]
        feature_choice = st.sidebar.selectbox("Find Features",tasks)
        image_file = st.file_uploader("Upload Image",type = ['jpg','png','jpeg'],encoding =None)
        if image_file is not None:
            our_image = Image.open(image_file)
            st.header('Original Image')
            st.image(our_image,width=400)

        if st.button("Process"):
            if feature_choice == 'Faces':
                result_img,result_faces = detect_faces(our_image)
                st.image(result_img,width=400)

                st.success("Found {} faces".format(len(result_faces)))
                
            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img,width=400)
            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img,width=400)
            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_image)
                st.image(result_img,width=400)
            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_image)
                st.image(result_canny,width=400)

    elif choice == 'About':
        st.header('This is the about page.')


if __name__ == "__main__":
    app()