import csv
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
import os
from ultralytics import YOLO
import werkzeug.utils

# Initialize YOLO model
model = YOLO('models/best.pt')

# Define the class names (this should match the class IDs in your YOLO model)
CLASS_NAMES = [
    'Front-windscreen-damage',
    'Headlight-damage',
    'Rear-windscreen-Damage',
    'Runningboard-Damage',
    'Sidemirror-Damage',
    'Taillight-Damage',
    'bonnet-dent',
    'boot-dent',
    'doorouter-dent',
    'fender-dent',
    'front-bumper-dent',
    'quaterpanel-dent',
    'rear-bumper-dent',
    'roof-dent'
]

# Mapping class names to part names in the CSV
CLASS_TO_PART_MAPPING = {
    'Front-windscreen-damage': 'Windscreen',
    'Headlight-damage': 'Headlight',
    'Rear-windscreen-Damage': 'Windscreen',  # Assuming rear windscreen maps to 'windscreen'
    'Runningboard-Damage': 'Running Board',
    'Sidemirror-Damage': 'Side Mirror',
    'Taillight-Damage': 'Taillight',
    'bonnet-dent': 'Hood',
    'boot-dent': 'Trunk',
    'doorouter-dent': 'Door',
    'fender-dent': 'Fender',  # This one seems unique, so assuming it maps to 'fender'
    'front-bumper-dent': 'Front Bumper',
    'quaterpanel-dent': 'Quarter Panel',  # Correct spelling: quarter
    'rear-bumper-dent': 'Rear Bumper',
    'roof-dent': 'Roof'
}

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load car brands and models from CSV
def load_car_brands_and_models():
    car_brands_and_models = {}
    with open('car_brand_models.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            brand = row['brand']
            model = row['model']
            if brand not in car_brands_and_models:
                car_brands_and_models[brand] = []
            car_brands_and_models[brand].append(model)
    return car_brands_and_models

# Load car brands and models
car_brands_and_models = load_car_brands_and_models()

# Load car brand models and prices from CSV
def load_prices():
    prices = []
    with open('prices.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prices.append({
                'brand': row['brand'],
                'model': row['model'],
                'part': row['part'],
                'damage_type': row['damage_type'],
                'price': float(row['price'])
            })
    return prices

# Fetch prices from CSV
prices = load_prices()

# Function to fetch claim amount based on brand, model, part, and damage type
def get_claim_amount(brand, model, part, damage_type):
    for price in prices:
        print(brand, model, part, damage_type)
        if (price['brand'] == brand and 
            price['model'] == model and 
            price['part'] == part and 
            price['damage_type'] == damage_type):
            return price['price']
    return 0  # Default if no match found

def predict_damage(image_path, selected_model, selected_brand):
    # Open the image and convert it to RGB
    img = Image.open(image_path).convert('RGB')
    results = model(img)  # Get predictions
    
    damage_details = []  # List to store damage details
    
    if results:
        result = results[0]  # The first result, as it's a list of results
        
        if hasattr(result, 'boxes'):
            preds = result.boxes.xyxy  # Get bounding box coordinates
            confidences = result.boxes.conf  # Get confidence scores
            classes = result.boxes.cls  # Get class ids

            # Prepare to draw on the image
            draw = ImageDraw.Draw(img)
            
            # Use a font if you want to add labels (you can also customize the font style)
            try:
                font = ImageFont.truetype("arial.ttf", size=20)  # Use a basic font
            except IOError:
                font = ImageFont.load_default()  # Fallback font if arial.ttf is not available

            # Loop through each prediction to extract damage details and draw the bounding boxes
            for idx in range(len(preds)):
                cls = int(classes[idx].item())  # Class id of the prediction
                confidence = confidences[idx].item()  # Confidence score

                if confidence > 0.5:  # Only consider predictions with high confidence
                    class_name = CLASS_NAMES[cls]  # Get the class name
                    
                    # Extract part name and damage type from the class name
                    if '-' in class_name:
                        part_name, damage_type = class_name.rsplit('-', 1)
                    else:
                        part_name = class_name
                        damage_type = 'unknown'  # Default if no damage type is found

                    # Map the class name to the corresponding part
                    mapped_part_name = CLASS_TO_PART_MAPPING.get(class_name, None)

                    if mapped_part_name:
                        # Get the claim amount based on the selected brand, model, mapped part, and damage type
                        claim_amount = get_claim_amount(selected_brand, selected_model, mapped_part_name, damage_type)

                        # Append the damage details to the list
                        damage_details.append({
                            'part': mapped_part_name,
                            'damage_type': damage_type,
                            'claim_amount': claim_amount,
                            'confidence': confidence,
                            'bbox': preds[idx].tolist()
                        })

                        # Draw the bounding box (xyxy format, (left, top, right, bottom))
                        left, top, right, bottom = preds[idx]
                        draw.rectangle([left, top, right, bottom], outline="red", width=3)
                        
                        # Add label (class name) above the box
                        label = f"{class_name} ({confidence:.2f})"

                        # Use textbbox to calculate the width and height of the text
                        bbox = draw.textbbox((left, top - 20), label, font=font)
                        text_width = bbox[2] - bbox[0]  # Width of the text
                        text_height = bbox[3] - bbox[1]  # Height of the text
                        
                        # Draw the background rectangle for the label
                        draw.rectangle([left, top - text_height, left + text_width, top], fill="red")
                        
                        # Draw the label text
                        draw.text((left, top - text_height), label, fill="white", font=font)

    # Save the predicted image with bounding boxes drawn
    predicted_image_path = os.path.join(RESULT_FOLDER, 'predicted_' + os.path.basename(image_path))
    print(f"Saving image to: {predicted_image_path}")  # Debugging print
    img.save(predicted_image_path)

    return damage_details, 'results/' + os.path.basename(predicted_image_path)  # Ensure correct folder path

@app.route('/', methods=['GET', 'POST'])
def claim_form():
    damage_details = None
    predicted_image_path = None
    models = []
    brand = request.form.get('brand', '')
    model_name = request.form.get('model', '')
    plate_number = request.form.get('plate', '')
    if brand in car_brands_and_models:
        models = car_brands_and_models[brand]

    if request.method == 'POST':
        plate_number = request.form['plate']
        file = request.files['image']

        if file:
            # Sanitize filename to avoid issues with special characters
            filename = werkzeug.utils.secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Now, call the damage prediction and get both damage details and predicted image path
            damage_details, predicted_image_path = predict_damage(file_path, model_name, brand)

    return render_template('index.html', damage_details=damage_details,
                           predicted_image_path=predicted_image_path,
                           car_brands_and_models=car_brands_and_models,
                           models=models, selected_brand=brand, selected_model=model_name, plate_number=plate_number)

if __name__ == '__main__':
    app.run(debug=True)
