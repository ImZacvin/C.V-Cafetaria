# model.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_model(num_classes):
    # Memuat model pre-trained MobileNetV2 tanpa lapisan klasifikasi di atas
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Menambahkan lapisan klasifikasi kustom
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Membuat model akhir
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
