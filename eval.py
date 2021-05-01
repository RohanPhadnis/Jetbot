from tensorflow import keras
import numpy as np
from PIL import Image
import os

model = keras.models.load_model('model.h5')
#print(model.summary())
#model.evaluate('Final_Images/Validation/Blocked')


for file in os.listdir('CSI_Camera'):
	'''image = Image.open('New_Raw/Free/free2'+str(n)+'.png')
	data = np.asarray(image, dtype = float)
	data/=255.
	image = Image.open('New_Raw/Blocked/blocked'+str(n)+'.png')
	data2 = np.asarray(image, dtype = float)
	data2/=255.
	predictions = model.predict(np.array([data, data2]), batch_size = 1)
	print(n, predictions)
	free.append(predictions[0])
	blocked.append(predictions[1])'''
	
	if '.png' in file:
		image = Image.open('CSI-Camera/'+file)
		data = np.asarray(image, dtype = float)
		data/=255.
		prediction = model.predict(np.array(data), batch_size = 1)
		print(prediction[0])

#print('blocked:', sum(blocked)/len(blocked))
#print('free:', sum(free)/len(free))
