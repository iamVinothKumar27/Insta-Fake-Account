from keras.models import load_model
# Step 1: Load the old model
old_model = load_model('insta-fake-real-new.keras', compile=False
# Step 2: Save the model in new .h5 format
old_model.save('insta-fake-real-new-fixed.h5', save_format='h5')
print("Model successfully converted and saved as insta-fake-real-new-fixed.h5")
