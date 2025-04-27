from tensorflow.keras.models import load_model

# Load the old model
old_model = load_model("insta-fake-real.h5", compile=False)

# Save it in NEW format (SavedModel format)
old_model.save("insta-fake-real-new.keras", save_format="keras")
