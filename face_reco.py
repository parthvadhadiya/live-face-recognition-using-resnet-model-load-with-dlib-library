import dlib
import scipy.misc
import numpy as np
import os
import cv2

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
TOLERANCE = 0.5

def get_face_encodings(path_to_image):
    image = scipy.misc.imread(path_to_image)
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)


def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    return 'Not Found'

video = cv2.VideoCapture(0)

image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
image_filenames = sorted(image_filenames)
paths_to_images = ['images/' + x for x in image_filenames]
face_encodings = []
for path_to_image in paths_to_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    face_encodings.append(get_face_encodings(path_to_image)[0])

names = [x[:-4] for x in image_filenames]
while True:
    _ ,path_to_image = video.read()
    cv2.imwrite('frame.jpg',path_to_image)
    face_encodings_in_image = get_face_encodings('frame.jpg')
    if os.path.exists('frame.jpg'):
    	os.remove('frame.jpg')
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    match = find_match(face_encodings, names, face_encodings_in_image[0])
    print(path_to_image, match)
    cv2.putText(path_to_image, match.capitalize(),
                    (5, path_to_image.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (206, 0, 209), 2, cv2.LINE_AA)
    cv2.imshow('Video', path_to_image)	
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows() 
video.release()