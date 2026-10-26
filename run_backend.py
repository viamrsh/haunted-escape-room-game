import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FONT_DIR = os.path.join(ASSET_DIR, "haunted text")

# Load fonts
fonts = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]

def generate_text_image(text, size=(300, 100), font_size=32):
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(random.choice(fonts), font_size)
    draw.text((10, 10), text, font=font, fill='white')
    return np.array(img)

def create_dataset(texts):
    images = []
    labels = []
    for i, text in enumerate(texts):
        img = generate_text_image(text)
        images.append(img)
        labels.append(i)
    return np.array(images), np.array(labels)

# Load and preprocess image data for GAN
def load_images(folder_path, target_size=(64, 64)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
            images.append(img_array)
    return np.array(images)

# GAN Models
def make_generator_model():
    model = keras.Sequential([
        keras.layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        
        keras.layers.Reshape((8, 8, 256)),
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        
        keras.layers.Conv2DTranspose(3, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def make_discriminator_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ])
    return model

# GAN training function
def train_gan(data_folder, output_folder, epochs=200, batch_size=64):
    os.makedirs(output_folder, exist_ok=True)
    
    images = load_images(data_folder)
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

    generator.save(os.path.join(output_folder, 'generator.h5'))
    discriminator.save(os.path.join(output_folder, 'discriminator.h5'))

# Text classification model
def create_text_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_text_model(images, labels, model_name):
    num_classes = len(set(labels))
    model = create_text_model(images.shape[1:], num_classes)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(images, labels, epochs=50, validation_split=0.2, verbose=1)
    
    model.save(os.path.join(MODEL_DIR, f"{model_name}.keras"))
    return history

# Puzzle Solver model
def make_puzzle_solver_model(input_size, hidden_size, output_size):
    model = keras.Sequential([
        keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
        keras.layers.Dense(hidden_size, activation='relu'),
        keras.layers.Dense(output_size, activation='softmax')
    ])
    return model

# Puzzle Solver training function
def train_puzzle_solver(data_folder, output_folder, input_size, hidden_size, output_size, epochs=50, batch_size=32):
    os.makedirs(output_folder, exist_ok=True)

    # Load puzzle data (you'll need to implement this based on your data format)
    # For now, we'll use random data
    X = np.random.rand(1000, input_size)
    y = np.random.randint(0, output_size, (1000,))

    model = make_puzzle_solver_model(input_size, hidden_size, output_size)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    model.save(os.path.join(output_folder, 'puzzle_solver.h5'))

def plot_history(history, title):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig(os.path.join(MODEL_DIR, f"{title}.png"))
    plt.close()

if __name__ == "__main__":
    # Train GAN models
    print("Training GAN models...")
    train_gan(os.path.join(ASSET_DIR, 'ghost'), os.path.join(MODEL_DIR, 'ghost_gan'))
    train_gan(os.path.join(ASSET_DIR, 'atmosphere'), os.path.join(MODEL_DIR, 'atmosphere_gan'))
    train_gan(os.path.join(ASSET_DIR, 'door'), os.path.join(MODEL_DIR, 'door_gan'))
    train_gan(os.path.join(ASSET_DIR, 'hallowean'), os.path.join(MODEL_DIR, 'halloween_gan'))
    train_gan(os.path.join(ASSET_DIR, 'haunted_text'), os.path.join(MODEL_DIR, 'haunted_text_gan'))

    # Train text classification models
    print("Training text classification models...")
    haunted_texts = ["Beware", "Danger", "Cursed", "Haunted"]
    ghost_dialogues = ["Who dares enter?", "Leave now!", "Your soul is mine"]
    riddles = ["What has keys but no locks?", "I'm tall when young and short when old"]

    haunted_images, haunted_labels = create_dataset(haunted_texts)
    ghost_images, ghost_labels = create_dataset(ghost_dialogues)
    riddle_images, riddle_labels = create_dataset(riddles)

    haunted_history = train_text_model(haunted_images, haunted_labels, "haunted_text_model")
    ghost_history = train_text_model(ghost_images, ghost_labels, "ghost_dialogue_model")
    riddle_history = train_text_model(riddle_images, riddle_labels, "riddle_model")

    # Train puzzle solvers
    print("Training puzzle solvers...")
    train_puzzle_solver(
        os.path.join(DATA_DIR, 'puzzles', 'Gravity Falls', 'puzzle_2x2'),
        os.path.join(MODEL_DIR, 'puzzle_solver_2x2'),
        input_size=4,
        hidden_size=64,
        output_size=4
    )
    train_puzzle_solver(
        os.path.join(DATA_DIR, 'puzzles', 'Gravity Falls', 'puzzle_4x4'),
        os.path.join(MODEL_DIR, 'puzzle_solver_4x4'),
        input_size=16,
        hidden_size=128,
        output_size=16
    )
    train_puzzle_solver(
        os.path.join(DATA_DIR, 'puzzles', 'Gravity Falls', 'puzzle_8x8'),
        os.path.join(MODEL_DIR, 'puzzle_solver_8x8'),
        input_size=64,
        hidden_size=256,
        output_size=64
    )

    # Plot training histories
    plot_history(haunted_history, 'Haunted Text Model Training')
    plot_history(ghost_history, 'Ghost Dialogue Model Training')
    plot_history(riddle_history, 'Riddle Model Training')

    print("All models trained and saved.")
