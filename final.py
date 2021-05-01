import Jetson.GPIO as g
import time
import cv2
import numpy
from tensorflow import keras
import threading
import socket

host = ''
port = 12345


model = keras.models.load_model('model.h5')

g.setmode(g.BOARD)
g.setwarnings(False)

class Motor:
	def __init__(self,fd,bk, lib, pulse = 0.02):
		self.spin = [[fd,bk],[0,0]]
		self.pwr = 0
		self.start = 0
		self.process = 0
		self.pulse = pulse
		self.lib = lib
		self.lib.setup(self.spin[0],self.lib.OUT)
		
	def setVel(self, pwr):
			
		
		if self.process == 0:
			self.pwr = pwr
			if self.pwr<0:
				self.spin[1] = [0,1]
			elif self.pwr == 0:
				self.spin[1] = [0,0]
			else:
				self.spin[1] = [1,0]
			
			if self.pwr < -1:
				self.pwr = -1
			elif self.pwr > 1:
				self.pwr = 1
			
			self.pwr = abs(self.pwr)
			self.pwr*=self.pulse
			self.start = time.time()
			self.process = 1
		if time.time()-self.start<self.pwr:
			self.status = 1
		else:
			self.status = 0
			if time.time()-self.start>=self.pulse:
				self.process = 0
		
		self.lib.output(self.spin[0],[self.spin[1][0]*self.status,self.spin[1][1]*self.status])

right = Motor(31, 33, g)
left = Motor(35, 37, g)


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = 0

prediction = 0


def init_camera():
	global cap
	print(gstreamer_pipeline(flip_method=0))
	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

init_camera()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen()
conn, addr = s.accept()
print('CONNECTED!')

time.sleep(10)

start = time.time()

def pic():
	global cap
	if cap.isOpened():
        # Window
		ret_val, img = cap.read()
		return img
            # Stop the program on the ESC key
	else:
		print("Unable to open camera")


def pred():
	global model, prediction, start, conn
	while time.time()-start<90:
		prediction = model.predict(numpy.array([pic()], dtype = float)/255., batch_size = 1)[0][0]
		conn.send(str(prediction).encode())

def drive():
	global prediction, start
	while time.time()-start<90:
		if prediction>=0.5:
			left.setVel(1)
			right.setVel(1)
		else:
			left.setVel(-1)
			right.setVel(1)




thread1 = threading.Thread(target = pred)
thread2 = threading.Thread(target = drive)

thread1.start()
thread2.start()

thread1.join()
thread2.join()
	
cap.release()
g.cleanup()
conn.close()
