import os
import sys
import face_recognition
import pickle

def train_model_by_img( name):
    if not os.path.exists('dataset1'):
        print("[ERROR] there is no dir 'dataset'")
        sys.exit()

    known_encodings = []
    images = os.listdir('dataset1')



    for (i, image) in enumerate (images):
        print(f'[+] processing img {i+1} : {image}  out of {len(images)} photos' )
        
        face_img = face_recognition.load_image_file(f'dataset1/{image}')
        face_enc = face_recognition.face_encodings(face_img)[0]        
        #print(face_enc)

        if len( known_encodings) == 0:
            known_encodings.append( face_enc)
        else:
            for item in range (0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])
                #print(result[0])
                
                if result[0]:
                    known_encodings.append(face_enc)
                    print('The same person')
                    break
                else:
                    print("Another person")
                    break
    #print(known_encodings)
    print(f'Length {len(known_encodings)}')

    data = {
        'name': name,
        'encodings': known_encodings
    }

    with open(f'{name}_encodings.pickle', 'wb') as file:
        file.write(pickle.dumps(data))
    
    
    return f'[INFO] File {name}_encodings_pickle successfully created'

def get_name_file (file_path):
    f = open(file_path)
    file_name = os.path.basename(f.name)
    print(file_name)

def main():
    print(train_model_by_img('Arnold'))

if __name__== '__main__':
    main()

