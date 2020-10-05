import os
import numpy as np
import matplotlib.pyplot as plt
import model_object


weight_file = 'F:/Projects/API_cycleGAN/saved_models/saveWeights_season/cyclegan_season_weights10'
cycle_gan_model.load_weights(weight_file).expect_partial()
print("Weights loaded successfully")

def get_prediction(image_path):
    _, ax = plt.subplots(4, 2, figsize=(10, 15))
    for i, img in enumerate(test_summer.take(4)):
        prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input image")
        ax[i, 0].set_title("Input image")
        ax[i, 1].set_title("Translated image")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")

        prediction = keras.preprocessing.image.array_to_img(prediction)
        prediction.save("predicted_img_{i}.png".format(i=i))
    plt.tight_layout()
    plt.show()
