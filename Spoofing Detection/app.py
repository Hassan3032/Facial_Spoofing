import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template
from gridfs import GridFS
from flask_pymongo import MongoClient
from keras.models import load_model

app = Flask(__name__)

mongo_client = MongoClient("mongodb://localhost:27017/facedetection")
db = mongo_client.get_database()
mangofs = GridFS(db,collection='Image Frames')
Imgrnd = db['Image Result']

model_path1 = os.path.join(os.path.dirname(__file__), 'mask_detector.h5')
model_path2 = os.path.join(os.path.dirname(__file__), 'Picture_In_Picture.h5')
model_path3 = os.path.join(os.path.dirname(__file__), 'Printed_Picture.h5')
model_path4 = os.path.join(os.path.dirname(__file__), 'depthMap.h5')

model1 = load_model(model_path1)
model2 = load_model(model_path2)
model3 = load_model(model_path3)
model4 = load_model(model_path4)

def convert_image_to_base64(base64_data):
    try:
        decoded_data = base64.b64decode(base64_data.split(',')[1])
        image = Image.open(BytesIO(decoded_data))
        image_new = BytesIO()
        image.save(image_new,format='png')
        image_new.seek(0)
        return image_new
    except Exception as e:
        print("Error decoding base64 data:", e)
        return None
    
def normalize_image(image, target_size=(224, 224)):
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize(target_size)
    image_array = np.array(image_resized)
    normalized_image = (image_array - 128.0) / 128.0
    return normalized_image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        image_data = data['image']
        image_new = convert_image_to_base64(image_data)
        
        frame_id = mangofs.put(image_new)
        Imgrnd.insert_one({
            'Frame_id': frame_id,
            'Results': None,
            'Mask Picture': None,
            'Screen In Screen': None,
            'Printed Picture': None,
            'Replay Picture': None})
        
        get_db_image = get_image(frame_id)
        normalized_image = normalize_image(get_db_image)

        results = []
        for idx, model in enumerate([model1, model2, model3, model4]):
            input_data = np.expand_dims(normalized_image, axis=0)
            try:
                prediction = model.predict(input_data)
                results.append(prediction)
            except Exception as e:
                print("Error predicting with Model", idx+1, ":", e)
        
        for idx, prediction in enumerate(results):
            try:
                col_name = 'Mask Picture' if idx == 0 else \
                           'Screen In Screen' if idx == 1 else \
                           'Printed Picture' if idx == 2 else \
                           'Replay Picture'
                    
                result_update = Imgrnd.update_one({
                    'Frame_id': frame_id}, 
                    {'$set': {col_name: float(prediction)}}) 
            except Exception as e:
                print("Error updating database:", e)

        pic_result = result_finalizer(frame_id) 
        return jsonify({'result': pic_result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

def get_image(frame_id):
    try:
        result_entry = Imgrnd.find_one({
            'Frame_id': frame_id,
            'Results': None,
            'Mask Picture': None,
            'Screen In Screen': None,
            'Printed Picture': None,
            'Replay Picture': None})

        if result_entry is None:
            return jsonify({'message': 'No image with null results found.'})

        grid_out = mangofs.get(frame_id)
        image_binary = grid_out.read()
        image = Image.open(BytesIO(image_binary))
        return image
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
def result_finalizer(frame_id):
    try:
        find_result = Imgrnd.find_one({
            'Frame_id': frame_id
        })
        if find_result is None:
            return jsonify({'message': 'Not found.'})
      
        s1_value = find_result.get('Mask Picture', 0)
        s2_value = find_result.get('Screen In Screen', 0)
        s3_value = find_result.get('Printed Picture', 0)
        s4_value = find_result.get('Replay Picture', 0)

        # Find the highest value and its corresponding column name
        max_value = max(s1_value, s2_value, s3_value, s4_value)
        if max_value == s1_value:
            results_value = 'Spoofed: Mask Picture'
        elif max_value == s2_value:
            results_value = 'Spoofed: Screen In Screen'
        elif max_value == s3_value:
            results_value = 'Spoofed: Printed Picture'
        else:
            results_value = 'Spoofed: Replay Picture'

        # Update the Results field in the document
        Imgrnd.update_one({'Frame_id': frame_id}, {'$set': {'Results': results_value}})

        return results_value
    
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)