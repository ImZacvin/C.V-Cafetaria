# main.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Tentukan jumlah kelas
num_classes = 3  # Sesuaikan dengan jumlah jenis makanan

# Membuat model
model = create_model(num_classes)

# Mengompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mengatur direktori dataset
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# Membuat generator untuk augmentasi data pada saat training
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Memuat gambar dari folder training
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Memuat gambar dari folder validasi
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Melatih model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Menyimpan model
model.save('food_detection_model.h5')
