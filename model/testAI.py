from rembg import remove
import cv2

def imageResize(orgImage, resizeFact):
    dim = (int(orgImage.shape[1]*resizeFact), 
           int(orgImage.shape[0]*resizeFact))  # w, h
    return cv2.resize(orgImage, dim, cv2.INTER_AREA)

input_path = "D:/Coding Challenges/Chess Website/model/archive3/board/5.jpg" # input image path
output_path = 'output.png' # output image path

input = imageResize(cv2.imread(input_path),0.4)
output = remove(input)

# cv2.imshow('g',output)
# cv2.waitKey(0)



