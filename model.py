from tensorflow import keras

class myCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs = {}):
		if logs['acc']>=0.95:
			self.model.stop_training = True

cb = myCallback()

model = keras.models.Sequential([
	keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (720,1280,3)),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(32, (3,3), activation = 'relu'),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(64, (3,3), activation = 'relu'),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Flatten(),
	keras.layers.Dense(128, activation = 'relu'),
	keras.layers.Dense(1, activation = 'sigmoid')])

training = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255).flow_from_directory(
	'New_Raw',
	batch_size = 1,
	target_size = (1280,720),
	class_mode = 'binary')

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
print(model.summary())
history = model.fit_generator(
	training,
	steps_per_epoch = 200,
	verbose = 1,
	epochs = 50,
	callbacks = [cb])

model.save('model.h5')
