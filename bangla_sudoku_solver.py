import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import imutils
from skimage.segmentation import clear_border

def find_rects(coords)  : 
	rect = np.zeros((4,2),dtype = "float32") 

	summation = coords.sum(axis = 1)

	rect[0]  = coords[np.argmin(summation)]
	rect[2]  = coords[np.argmax(summation)]

	difference = np.diff(coords,axis=1)
	rect[3] = coords[np.argmax(difference)]
	rect[1] = coords[np.argmin(difference)]

	return rect 


def bird_view_to_remove_background(image,coords) : 

	rect = find_rects(coords)
	(top_left,top_right,bottom_left,bottom_right) = rect 

	width_one = np.sqrt(np.power(top_left[0]-top_right[0],2) + np.power(top_right[0] - top_left[0],2))
	width_two= np.sqrt(np.power(bottom_left[0] - bottom_right[0],2) + np.power(bottom_right[0] - bottom_left[0],2))
	maximum_width = max(int(width_one),int(width_two))

	height_one = np.sqrt(np.power((top_left[0]-bottom_left[0]),2) + np.power(bottom_left[0] - top_left[0],2))
	height_two= np.sqrt(np.power(bottom_right[0] - top_right[0],2) + np.power(top_right[0] - bottom_right[0],2))
	maximum_height = max(int(height_one),int(height_two))

	to_be_drawn = np.array([[0,0],[maximum_width-1,0],[maximum_width-1,maximum_height-1],[0,maximum_height-1]],dtype = "float32")

	matrix = cv2.getPerspectiveTransform(rect,to_be_drawn)
	warped = cv2.warpPerspective(image,matrix,(maximum_width,maximum_height))

	return warped

def find_contours(image,original) : 

	contours = cv2.findContours(image.copy() , cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
	contours = imutils.grab_contours(contours)
	contours = sorted(contours , key = cv2.contourArea,reverse=True)

	sudoku_board_contour = None 

	for c in contours : 
		perimeter = cv2.arcLength(c,True)
		approximate = cv2.approxPolyDP(c,0.02 * perimeter,True)

		if len(approximate) == 4 :
			sudoku_board_contour = approximate
			break 

	output = original.copy()
	cv2.drawContours(output,[sudoku_board_contour],-1,(0,0,255),2)
	cv2.imshow("Contour on Image",output)
	cv2.waitKey(0)

	puzzle = bird_view_to_remove_background(image,sudoku_board_contour.reshape(4,2))

	return puzzle


#Extracting Digits from Sudoku Board 
def extract_digits_from_board(cell) : 
	individual_contours = cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	individual_contours = clear_border(individual_contours)
	individual_contours = cv2.findContours(cell,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
	individual_contours = imutils.grab_contours(individual_contours)
	

	if len(individual_contours) == 0 :
		return 0

	contours = max(individual_contours,key = cv2.contourArea)
	mask = np.zeros(cell.shape,dtype = "uint8")
	cv2.drawContours(mask,[contours],-1,255,-1)

	(height,width) = cell.shape

	digit = cv2.bitwise_and(cell,cell,mask=mask)
	return digit




#Original Image ==> Gray Scale
sudoku_orig = cv2.imread("sudoku.png")
sudoku = cv2.cvtColor(sudoku_orig,cv2.COLOR_BGR2GRAY)
cv2.imshow("Board",sudoku)
cv2.waitKey(0)

#Noise Reduction
gaussian_blur = cv2.GaussianBlur(sudoku,(1,1),cv2.BORDER_DEFAULT)
cv2.imshow("Noise Reduction",gaussian_blur)
cv2.waitKey(0)

#Inverse Binary Threshold
thresh,threshout = cv2.threshold(gaussian_blur,180,255,cv2.THRESH_BINARY_INV)
print(thresh)
cv2.imshow("Thresh Output",threshout)
cv2.waitKey() 

puzzle = find_contours(threshout,sudoku_orig)

cv2.imshow("Puzzle",puzzle)
cv2.waitKey(0)


cell_X = puzzle.shape[1]//9
cell_Y = puzzle.shape[0] // 9


print(cell_X)
print(cell_Y)

cell_coords = [] 

for y in range(0,9) : 

	row = [] 

	for x in range(0,9) :

		startX = x * cell_X
		startY= y * cell_Y
		endX = (x+1) * cell_X
		endY = (y+1) * cell_Y

		row.append([startX,startY,endX,endY])

		cell = puzzle[startY:endY,startX:endX]
		digit = extract_digits_from_board(cell) 

		cv2.imshow("digits",digit)
		cv2.waitKey(0)
