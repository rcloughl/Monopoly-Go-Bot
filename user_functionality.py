import pyautogui, sys
import numpy as np
import cv2
from mss import mss
from PIL import Image

# locating the screen
def locateScreen():
	screen = pyautogui.locateOnScreen('assets/bluestacks_logo.png', confidence=0.95)
	if screen is None:
		print('Screen not found.')
		sys.exit()
	else:
		print('Screen found.')
		return screen