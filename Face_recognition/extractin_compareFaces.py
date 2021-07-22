import face_recognition
from PIL import Image, ImageDraw
import os
from face_recognition.api import face_locations 

def face_rec():
    arnold_face_img = face_recognition.load_image_file('dataset/Arnold_1.jpeg')
    arnold_face_location = face_recognition.face_locations(arnold_face_img)


    #loading photo where more than one face
    arnold_family_img = face_recognition.load_image_file("dataset/arnold_with_Family.jpeg")
    arnold_family_locations = face_recognition.face_locations(arnold_family_img)

    print(arnold_face_location)
    print(arnold_family_locations)
    print(f"Found {len(arnold_face_location)} face(s) in this photo")
    print(f"Found {len(arnold_family_locations)} face(s) in this photo")

    #drawing frames for faces in photo
    pil_img1 = Image.fromarray(arnold_face_img)
    draw1 = ImageDraw.Draw(pil_img1)
    for(top, right, bottom, left) in arnold_face_location:
        draw1.rectangle(((left,top ),(right, bottom)), outline=(255,255, 0), width=4 )
    del draw1
    pil_img1.save("dataset/new_arnold_img.jpeg")

    pil_img2 = Image.fromarray(arnold_family_img)
    draw2 = ImageDraw.Draw(pil_img2)
    for(top, right, bottom, left) in arnold_family_locations:
        draw2.rectangle(((left,top ),(right, bottom)), outline=(255,255, 0), width=4 )

    del draw2
    pil_img2.save("dataset/new_arnold_family_img.jpeg")



def extracting_faces(img_path):
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)
    number = len(faces_locations)
    for face_location in number:
        top, right, bottom, left = face_location
        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"faces_arnold_family/{number}_face_img.jpeg")

    return f"Found {number} face(s) in this photo"

def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]


    img2 = face_recognition.load_image_file(img2_path)
    img2_face_locations = face_recognition.face_locations(img2)
    faces_number = len(img2_face_locations)
    result = False
    for i in range (0, faces_number):
        img2_encodings = face_recognition.face_encodings(img2)[i]
        result = face_recognition.compare_faces([img1_encodings], img2_encodings)
        if result[0] == True:
            print(f"\nPerson found among {faces_number} face(s) in second photo")
            break
    if result == False:
        print("Person not found") 

def main():
    #face_rec()
    #extracting_faces("dataset/arnold_with_Family.jpeg")
    compare_faces("dataset/Arnold_4.jpeg", "dataset/arnold_with_wife.jpeg")

if __name__ == '__main__':
    main()
