import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import cv2
import PIL

# def load_cached_model():
#     import shelve
#     cache = shelve.open('my_cache')
#     if 'model' in cache:
#         return cache['model']
#     model = load_model('model/xception_v1_15_0.975.h5')
#     cache['model'] = model
#     cache.close()
#     return model

model = load_model('./xception_v1_15_0.975.h5')

img = load_img('D:/Coding Challenges/Chess Website/model/data/bishop.png',target_size=(224,224))

def classify_cells(model, img):
    category_reference = {0: 'Black Bishop', 1: 'Black King', 2: 'Black Knight',3: 'Black Pawn',
                            4: 'Black Queen',5: 'Black Rook', 6: 'Empty Square', 7: 'White Bishop',
                            8: 'White King',9: 'White Knight', 10: 'White Pawn',11: 'White Queen',
                            12: 'White Rook'}

    img = cv2.imread('D:/Coding Challenges/Chess Website/model/data/wqueen.jpg')                       
    img = cv2.resize(img,(224, 224), interpolation=cv2.INTER_AREA)
    # # img = prepare_image('D:/Coding Challenges/Chess Website/model/data/bknight.jpg')
    # img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)

    #img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    # img = cv2.imread('D:/Coding Challenges/Chess Website/model/data/bknight.jpg')
    # img = np.asarray(img) 
    # img = img_to_array(img)
    img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    # img = preprocess_input(img)
    
    datagen = ImageDataGenerator(
        rotation_range=5,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')
    img_converted = datagen.flow(img,batch_size=1)

    out = model.predict(img_converted)

    print(out)
    top_pred = np.argmax(out)
    pred = category_reference[top_pred]

    print(pred)

    return pred


classify_cells(model,img)

