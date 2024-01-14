import pyautogui, sys
import numpy as np
import cv2
from mss import mss
from PIL import Image

#user created dependencies
from user_functionality import locateScreen

print(locateScreen())
#offsets
left, top, width, height = locateScreen()

top += height

#size of screen
width = 765
height = 1350
threshold = 0.95

bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}

sct = mss()

object_detector = cv2.createBackgroundSubtractorMOG2()

manual_roll = cv2.imread('assets/manual_roll.png', cv2.IMREAD_UNCHANGED)
auto_roll = cv2.imread('assets/auto_roll.png', cv2.IMREAD_UNCHANGED)


while True:
	sct_img = sct.grab(bounding_box)
	frame = np.array(sct_img)
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	cv2.imshow('screen', frame)


	mask = object_detector.apply(frame)
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#for cnt in contours:
	#	cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

	roll_status = cv2.matchTemplate(frame, manual_roll, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roll_status)

	if max_val >= threshold:
		pyautogui.mouseDown(button='left', x=max_loc[0]+left, y=max_loc[1]+top)



	
	#cv2.imshow('mask', mask)

	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
	
cv2.destroyAllWindows()
