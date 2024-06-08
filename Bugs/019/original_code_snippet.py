x = []
for f in files:
        img = Image.open(f)
        img.load()
        data = np.asarray(img, dtype="int32")
        x.append(data)
x = np.array(x)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# other steps...