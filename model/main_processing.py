import base64
import numpy as np
import cv2
from collections import defaultdict
from rembg import remove
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import chess


#from website import views

model = load_model('model/xception_v1_15_0.975.h5')

def decode_image_base64(file64):
    encoded_data = file64.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    file = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return file
def image_resize(orgImage, resize_factor):
    dim = (int(orgImage.shape[1]*resize_factor), 
           int(orgImage.shape[0]*resize_factor))  # w, h
    return cv2.resize(orgImage, dim, cv2.INTER_AREA)
def segment_by_angle_kmeans(lines, k=2, **kwargs):

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

    labels, _ = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())

    return segmented
def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]
def order_points(pts):
    from functools import reduce
    import operator
    import math
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), pts[:,0]), [len(pts[:,0])] * 2))
    approx = (sorted(pts[:,0], key=lambda coord: (math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1])))))
    return np.array(approx)
def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections
def antialiasing(img,strength=13):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (strength,strength), 0)
    return blur
def image_crop(blur,img):
    _, im_bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(im_bw,(x,y),(x+w,y+h),(0,255,0),2)
    
    bw_crop = im_bw[y:y+h,x:x+w]
    img_crop = img[y:y+h,x:x+w]
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(bw_crop, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    epsilon = 0.5*cv2.arcLength(contour,True)
    for i in [x * 0.01 for x in range(0,100)]:
        approx = cv2.approxPolyDP(contour, i * epsilon, True)
        if approx.shape != (4,1,2):
            continue
        else:
            break
    approx = order_points(approx)

    corners = []
    for point in approx:
        x, y = point
        corners.append([x,y])
    
    return contours,bw_crop,img_crop,corners
def image_preprocessing(img):
    gray = antialiasing(img, strength=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) # <-----  
    morphed = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel, iterations=1)
    morphed = (morphed*255).astype(np.uint8)
    edged_wide = cv2.Canny(morphed, 10, 250, apertureSize=3)
    lines = cv2.HoughLines(edged_wide, 1, 1.8*np.pi /360, 120, None, 150,20)
    return lines
def nearest_common_demoninator(n, x):
  u = n % x > x // 2
  return n + (-1)**(1 - u) * abs(x * u - n % x)
def classify_cells(model, img):
    category_reference = {0: 'bb', 1: 'bk', 2: 'bn', 3: 'bp', 4: 'bq', 5: 'br', 6: 'em', 7: 'wb', 8: 'wk', 9: 'wn', 10: 'wp',
                          11: 'wq', 12: 'wr'}

    img = cv2.resize(img,(224, 224), interpolation=cv2.INTER_AREA)
    img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))

    datagen = ImageDataGenerator(
        rotation_range=5,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')
    img_converted = datagen.flow(img,batch_size=1)

    out = model.predict(img_converted)
    top_pred = np.argmax(out)
    pred = category_reference[top_pred]

    return pred
def board_to_fen(board):
    import io
    board = np.array(board).reshape((8,8))
        # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                c = cell[0]
                if c in ('w', 'b'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell[1].upper() if c == 'w' else cell[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' w KQkq - 0 1')
        return s.getvalue()
def image_import(img64):
    img = decode_image_base64(img64)  
    if img.shape[0] > 1000:
        img = image_resize(img,0.4)
     
    img = remove(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    blur = antialiasing(img)
    _,_,img_crop,corners = image_crop(blur,img)
    cropped_copy = img_crop.copy()
    h,w,_ = cropped_copy.shape
    return cropped_copy,corners,h,w
def image_warp(cropped_image):
    projection = np.array([[0, 0],[ w, 0], [w, h], [0, h]])
    poi = np.array([corners[0],corners[1],corners[2],corners[3]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(np.float32(poi),np.float32(projection))
    tf_out = cv2.warpPerspective(cropped_image,M,(w, h),flags=cv2.INTER_LINEAR)

    border = 15
    tf_crop = tf_out[0:h+border,border:w-border]
    return tf_crop
def image_split(cropped_copy):
    # make sure shape is divisible by 8:
    shape_h = nearest_common_demoninator(cropped_copy.shape[0],8)
    shape_w = nearest_common_demoninator(cropped_copy.shape[1],8)
    M = shape_h//8
    N = shape_w//8
    tiles = [cropped_copy[x:x+M,y:y+N] for x in range(0,shape_h,M) for y in range(0,shape_w,N)]
    return tiles
def run_file(test):
    cropped_copy,corners,h,w = image_import(test)
    tiles = image_split(cropped_copy)
    return tiles
def run_classification(img64):
    tiles = run_file(img64)
    grid = []
    for i in tiles:
        grid.append(classify_cells(model,i))

    fen = (board_to_fen(grid))
    board_svg = fen_to_image(fen)
    #board_image = cv2.imread(x)
    return board_svg
# Converts the FEN into a PNG file
def fen_to_image(fen):
    from chess import svg
    board = chess.Board(fen)
    current_board = svg.board(board=board)

    return current_board
