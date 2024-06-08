file_path = f"./STFT_spectra/STFT_spectra0.png"
image = io.read_file(file_path)
image = io.decode_png(image)
image = tf.image.convert_image_dtype(image, tf.float32)
image = tf.image.resize(image, [128,128])
print("----Here-----")
print(type(image))
print(image.shape)

# total = 100
# image_list = np.empty(shape=(total, 128, 128, 4))
# for i in tqdm(range(total)):
#     file_path = f"./STFT_spectra/STFT_spectra{i}.png"
#     image = io.read_file(file_path)
#     image = io.decode_png(image)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, [128, 128])
#     image_list[i] = image

# pickle.dump(image_list, open("image_list.p", "wb"))

labels = pickle.load(open(".././labels.p", "rb"))
fetched_image_list = pickle.load(open("../image_list.p", "rb"))
fetched_image_list = fetched_image_list.reshape(fetched_image_list.shape[0],
                                                fetched_image_list.shape[1],
                                                fetched_image_list.shape[2],
                                                fetched_image_list.shape[3],
                                                1)

dataset = tf.data.Dataset.from_tensor_slices((fetched_image_list, labels))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=(2,2), dilation_rate=(1,1), input_shape=(128,128,4,1), activation='relu'),
    tf.keras.layers.Conv2D(71, (3, 3), strides=(2,2), dilation_rate=(1,1), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 4), strides=(2,3), dilation_rate=(1,1),activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(2,2), dilation_rate=(1,1),activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 4), strides=(2, 3), dilation_rate=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), dilation_rate=(1, 1), activation='relu'),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])
