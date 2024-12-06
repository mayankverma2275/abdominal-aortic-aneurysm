img_path = r"D:\aortic aneurysm\img example.jpg"  # Use raw string (r) to handle backslashes in the path
img = image.load_img(img_path, target_size=(600, 600))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
preds = model.predict(img_array)
print('Predicted:', decode_predictions(preds, top=3)[0])
