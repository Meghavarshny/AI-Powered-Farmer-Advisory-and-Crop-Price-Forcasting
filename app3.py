import os
import datetime
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
from groq import Groq
import sqlite3

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_farm_app_key_change_in_production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# API Keys
WEATHER_API_KEY = 'c65022edc168e3d494ffe950131045b7'
GROQ_API_KEY = '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'  # Replace with your Groq API key

# Database file
DATABASE = 'farmassist.db'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/chat_images', exist_ok=True)

print("=" * 60)
print("🌾 FarmAssist - Smart Farming Platform")
print("=" * 60)
print("✅ Upload directory created:", UPLOAD_FOLDER)
print("✅ Images directory created: static/images")
print("✅ Chat images directory created: static/chat_images")
print("=" * 60)

# --- DATABASE SETUP ---
def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            user_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Products table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_username TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            quantity INTEGER NOT NULL,
            original_quantity INTEGER NOT NULL,
            quantity_type TEXT NOT NULL,
            price REAL NOT NULL,
            image TEXT,
            posted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_username) REFERENCES users(username)
        )
    ''')
    
    # Transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            farmer TEXT NOT NULL,
            buyer TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            quantity_type TEXT NOT NULL,
            price_per_unit REAL NOT NULL,
            total_amount REAL NOT NULL,
            payment_details TEXT,
            status TEXT DEFAULT 'pending',
            request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (farmer) REFERENCES users(username),
            FOREIGN KEY (buyer) REFERENCES users(username)
        )
    ''')
    
    # Chat messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            receiver TEXT NOT NULL,
            message TEXT NOT NULL,
            message_type TEXT DEFAULT 'text',
            timestamp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sender) REFERENCES users(username),
            FOREIGN KEY (receiver) REFERENCES users(username)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized: farmassist.db")
    print("=" * 60)

# Initialize database on startup
init_db()

# --- PYTORCH MODEL SETUP ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_model(model_name, num_classes):
    if model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# Load Model
try:
    best_model_path = 'best_efficientnet_model.pth'
    num_classes = 11
    best_model = create_model('efficientnet', num_classes)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model = best_model.to(device)
    best_model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    best_model = None

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]

# Disease information with pests and tips
disease_info = {
    'Apple___Apple_scab': {
        'pests': 'Apple scab fungus (Venturia inaequalis)',
        'tips': 'Remove fallen leaves, apply fungicides in spring, choose resistant varieties, ensure good air circulation'
    },
    'Apple___Black_rot': {
        'pests': 'Black rot fungus (Botryosphaeria obtusa)',
        'tips': 'Prune infected branches, remove mummified fruits, apply fungicides, maintain tree health'
    },
    'Apple___Cedar_apple_rust': {
        'pests': 'Cedar-apple rust fungus (Gymnosporangium juniperi-virginianae)',
        'tips': 'Remove nearby juniper trees, apply fungicides early season, use resistant varieties'
    },
    'Apple___healthy': {
        'pests': 'No disease detected',
        'tips': 'Continue regular monitoring, maintain proper irrigation and fertilization'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'pests': 'Cercospora zeae-maydis fungus',
        'tips': 'Rotate crops, use resistant hybrids, apply foliar fungicides, remove crop debris'
    },
    'Corn_(maize)___Common_rust_': {
        'pests': 'Puccinia sorghi fungus',
        'tips': 'Plant resistant hybrids, apply fungicides if severe, ensure proper plant spacing'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'pests': 'Exserohilum turcicum fungus',
        'tips': 'Use resistant varieties, rotate crops, apply fungicides preventively'
    },
    'Corn_(maize)___healthy': {
        'pests': 'No disease detected',
        'tips': 'Maintain current practices, monitor regularly for early disease detection'
    },
    'Potato___Early_blight': {
        'pests': 'Alternaria solani fungus',
        'tips': 'Remove infected leaves, apply copper-based fungicides, maintain soil fertility'
    },
    'Potato___Late_blight': {
        'pests': 'Phytophthora infestans',
        'tips': 'Apply fungicides regularly, ensure good drainage, destroy infected plants immediately'
    },
    'Potato___healthy': {
        'pests': 'No disease detected',
        'tips': 'Continue good agricultural practices, monitor weather conditions'
    }
}

def predict_image(image_path, model):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    class_idx = preds.item()
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()
    return class_idx, confidence

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- CROPS DATA ---
crops_data = [
    {
        "name": "Rice",
        "image": "https://images.unsplash.com/photo-1586201375761-83865001e31c?w=400",
        "nutrition": "Rich in carbohydrates, moderate protein, low fat. Source of B vitamins and minerals.",
        "suitable_land": "Clayey or loamy soils with high water retention. Requires flooded conditions.",
        "pests": "Brown Plant Hopper, Rice Gall Midge, Stem Borer",
        "sowing": "June-July for Kharif, Nov-Dec for Rabi",
        "irrigation": "Continuous flooding required during growth",
        "fertilizers": "Urea, DAP, Potash - Apply in 3 splits",
        "harvesting": "When 80% grains turn golden yellow"
    },
    {
        "name": "Wheat",
        "image": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
        "nutrition": "High in fiber, protein, and various minerals like selenium and manganese.",
        "suitable_land": "Well-drained fertile loam or clay loam soils. Moderate temperature.",
        "pests": "Aphids, Armyworm, Hessian Fly",
        "sowing": "October-November (Rabi season)",
        "irrigation": "4-6 irrigations during crop period",
        "fertilizers": "Nitrogen, Phosphorus, Potassium in recommended doses",
        "harvesting": "March-April when grains harden"
    },
    {
        "name": "Potato",
        "image": "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=400",
        "nutrition": "Good source of Vitamin C, Potassium, and Vitamin B6.",
        "suitable_land": "Sandy loam soil with good drainage and high organic matter.",
        "pests": "Colorado Potato Beetle, Potato Leafhopper, Late Blight",
        "sowing": "October for plains, March-April for hills",
        "irrigation": "Light and frequent irrigation",
        "fertilizers": "FYM, NPK in proper ratio",
        "harvesting": "90-120 days after planting"
    },
    {
        "name": "Maize (Corn)",
        "image": "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=400",
        "nutrition": "Rich in fiber, vitamins, minerals, and antioxidants.",
        "suitable_land": "Well-drained soils with rich organic content.",
        "pests": "Fall Armyworm, Corn Earworm, European Corn Borer",
        "sowing": "June-July (Kharif), Feb (Summer)",
        "irrigation": "Critical at flowering and grain filling",
        "fertilizers": "Nitrogen-rich fertilizers, Zinc sulfate",
        "harvesting": "80-110 days depending on variety"
    },
    {
        "name": "Cotton",
        "image": "https://images.unsplash.com/photo-1615485500834-bc10199bc743?w=400",
        "nutrition": "Used for fiber production, cottonseed oil is nutritious",
        "suitable_land": "Deep, well-drained black cotton soil",
        "pests": "Bollworm, Whitefly, Aphids",
        "sowing": "April-June",
        "irrigation": "8-10 irrigations during season",
        "fertilizers": "High nitrogen and potassium requirements",
        "harvesting": "Multiple pickings from October onwards"
    },
    {
        "name": "Sugarcane",
        "image": "https://images.unsplash.com/photo-1560582861-45078880e48e?w=400",
        "nutrition": "High sugar content, source of jaggery and sugar",
        "suitable_land": "Deep, fertile, well-drained loamy soil",
        "pests": "Top Borer, Root Borer, White Grub",
        "sowing": "February-March and September-October",
        "irrigation": "High water requirement, frequent irrigation",
        "fertilizers": "Heavy feeder - requires NPK in large quantities",
        "harvesting": "12-18 months after planting"
    }
]

# Government Schemes
schemes_data = [
    {
        "id": 1,
        "name": "PM-KISAN",
        "short_desc": "Direct income support of ₹6000/year to farmers",
        "full_desc": "Pradhan Mantri Kisan Samman Nidhi provides income support to all landholding farmers' families across the country with Rs 6000 per annum in three equal installments.",
        "eligibility": "All landholding farmers irrespective of size. Institutional landholders, income tax payers excluded.",
        "apply_link": "https://pmkisan.gov.in/"
    },
    {
        "id": 2,
        "name": "Kisan Credit Card",
        "short_desc": "Easy credit for agricultural needs",
        "full_desc": "KCC provides timely and adequate credit for agriculture and allied activities. Offers flexible credit limits based on operational land holdings, cropping pattern, and scale of finance.",
        "eligibility": "All farmers including tenant farmers, sharecroppers, and oral lessees",
        "apply_link": "https://www.nabard.org/content1.aspx?id=523&catid=8&mid=489"
    },
    {
        "id": 3,
        "name": "PM Fasal Bima Yojana",
        "short_desc": "Crop insurance against natural calamities",
        "full_desc": "Provides insurance coverage and financial support to farmers in case of crop failure. Low premium rates with high sum insured.",
        "eligibility": "All farmers growing notified crops in notified areas. Farmers paying premium can avail",
        "apply_link": "https://pmfby.gov.in/"
    },
    {
        "id": 4,
        "name": "Soil Health Card Scheme",
        "short_desc": "Free soil testing and recommendations",
        "full_desc": "Provides information to farmers on nutrient status of their soil along with recommendations on appropriate dosage of nutrients for improving soil health.",
        "eligibility": "All farmers can get their soil tested free of cost",
        "apply_link": "https://soilhealth.dac.gov.in/"
    },
    {
        "id": 5,
        "name": "Paramparagat Krishi Vikas Yojana",
        "short_desc": "Support for organic farming",
        "full_desc": "Promotes organic farming through cluster approach and PGS certification. Provides financial assistance of ₹50,000 per hectare over 3 years.",
        "eligibility": "Farmers willing to adopt organic farming in cluster approach",
        "apply_link": "https://pgsindia-ncof.gov.in/"
    },
    {
        "id": 6,
        "name": "PM Krishi Sinchayee Yojana",
        "short_desc": "Irrigation support and water conservation",
        "full_desc": "Aims to expand cultivated area with assured irrigation, improve water use efficiency, and adopt precision-irrigation technologies.",
        "eligibility": "All farmers including small and marginal farmers",
        "apply_link": "https://pmksy.gov.in/"
    }
]

# Loan Information
loans_data = [
    {
        "id": 1,
        "name": "Kisan Credit Card (KCC) Loan",
        "short_desc": "Flexible credit for crop production",
        "details": "Short-term credit for crop cultivation with flexible repayment. Interest subvention available. Credit limit up to ₹3 lakh at 7% interest.",
        "eligibility": "Farmers owning cultivable land. Tenant farmers, oral lessees, and sharecroppers also eligible.",
        "interest_rate": "7% (with interest subvention)",
        "loan_amount": "Based on scale of finance and cropping pattern"
    },
    {
        "id": 2,
        "name": "Agricultural Term Loan",
        "short_desc": "Long-term investment in farm assets",
        "details": "For purchase of tractors, construction of farm buildings, irrigation equipment, and other agricultural machinery. Repayment period up to 7-9 years.",
        "eligibility": "All farmers with owned/leased land. Good credit history required.",
        "interest_rate": "9-12% depending on bank",
        "loan_amount": "Up to ₹50 lakh or more for large investments"
    },
    {
        "id": 3,
        "name": "NABARD Dairy Loan",
        "short_desc": "Support for dairy farming activities",
        "details": "Financial assistance for purchase of milch animals, cattle shed construction, and dairy equipment. Special schemes for women farmers.",
        "eligibility": "Individual farmers, SHGs, cooperative societies involved in dairy",
        "interest_rate": "8-10% with subsidy benefits",
        "loan_amount": "₹30,000 to ₹20 lakh depending on project"
    },
    {
        "id": 4,
        "name": "Stand Up India Scheme",
        "short_desc": "Loans for SC/ST and women farmers",
        "details": "Facilitates bank loans for greenfield enterprises in agriculture sector. For setting up new ventures with composite loan of ₹10 lakh to ₹1 crore.",
        "eligibility": "SC/ST and women entrepreneurs. At least 51% shareholding required.",
        "interest_rate": "Bank's MCLR + 3% + tenor premium",
        "loan_amount": "₹10 lakh to ₹1 crore"
    },
    {
        "id": 5,
        "name": "Agricultural Gold Loan",
        "short_desc": "Quick loans against gold ornaments",
        "details": "Immediate loan disbursement against gold jewelry for agricultural purposes. No processing fee. Quick approval process.",
        "eligibility": "Any farmer with gold ornaments to pledge",
        "interest_rate": "7-11% depending on loan amount and bank",
        "loan_amount": "Up to 75% of gold value"
    }
]

# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        user_type = request.form['user_type']
        
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            flash('Username already exists!', 'error')
            conn.close()
        else:
            # Hash password and insert user
            password_hash = generate_password_hash(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, phone, user_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, password_hash, email, phone, user_type))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['username'] = username
            session['user_type'] = user['user_type']
            session['email'] = user['email']
            
            if user['user_type'] == 'farmer':
                return redirect(url_for('farmer_dashboard'))
            else:
                return redirect(url_for('buyer_dashboard'))
        else:
            flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/farmer/dashboard')
def farmer_dashboard():
    if 'username' not in session or session.get('user_type') != 'farmer':
        return redirect(url_for('login'))
    return render_template('farmer_dashboard.html', username=session['username'])

@app.route('/buyer/dashboard')
def buyer_dashboard():
    if 'username' not in session or session.get('user_type') != 'buyer':
        return redirect(url_for('login'))
    
    # Get all products that are in stock
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products WHERE quantity > 0 ORDER BY posted_date DESC')
    products = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return render_template('buyer_dashboard.html', username=session['username'], products=products)

# 1. CROP INFORMATION MODULE
@app.route('/crops')
def crops():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('crops.html', crops=crops_data)

# 2. WEATHER MODULE
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    weather_data = None
    forecast_data = []
    city_name = ""
    has_alert = False
    
    if request.method == 'POST':
        city = request.form['city']
        
        # Get Current Weather
        url_current = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url_current)
        
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                "city": data['name'],
                "temp": round(data['main']['temp']),
                "feels_like": round(data['main']['feels_like']),
                "humidity": data['main']['humidity'],
                "wind_speed": data['wind']['speed'],
                "desc": data['weather'][0]['description'].title(),
                "icon": data['weather'][0]['icon'],
                "pressure": data['main']['pressure']
            }
            
            # Get 5 Day Forecast
            url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
            fore_resp = requests.get(url_forecast)
            
            if fore_resp.status_code == 200:
                f_data = fore_resp.json()['list']
                seen_dates = set()
                
                for item in f_data:
                    date_txt = item['dt_txt']
                    date_str = date_txt.split(" ")[0]
                    
                    if "12:00:00" in date_txt and date_str not in seen_dates:
                        weather_condition = item['weather'][0]['main'].lower()
                        
                        # Check for severe weather
                        if weather_condition in ['rain', 'thunderstorm', 'snow'] or item['wind']['speed'] > 10:
                            has_alert = True
                        
                        forecast_data.append({
                            "date": datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d"),
                            "day": datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
                            "temp": round(item['main']['temp']),
                            "temp_min": round(item['main']['temp_min']),
                            "temp_max": round(item['main']['temp_max']),
                            "icon": item['weather'][0]['icon'],
                            "desc": item['weather'][0]['description'].title(),
                            "humidity": item['main']['humidity'],
                            "wind_speed": round(item['wind']['speed'], 1)
                        })
                        seen_dates.add(date_str)
                        
                        if len(forecast_data) >= 5:
                            break
            
            city_name = city
            
            # Send email alert if severe weather detected (simulation)
            if has_alert and session.get('user_type') == 'farmer':
                flash('⚠️ Weather Alert: Adverse weather conditions detected in the forecast. Please take necessary precautions!', 'warning')
        else:
            flash("City not found. Please try again.", 'error')
    
    return render_template('weather.html', weather=weather_data, forecast=forecast_data, city=city_name)

# 3. DISEASE DETECTION MODULE
@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if best_model:
                class_idx, confidence = predict_image(filepath, best_model)
                predicted_class = class_names[class_idx]
                info = disease_info[predicted_class]
                
                return render_template('disease_result.html', 
                                     filename=filename, 
                                     prediction=predicted_class.replace('_', ' '),
                                     confidence=round(confidence * 100, 2),
                                     pests=info['pests'],
                                     tips=info['tips'])
            else:
                flash('Model not loaded. Please check the model file.', 'error')
                return redirect(request.url)
    
    return render_template('disease.html')

# 4. GOVERNMENT SCHEMES MODULE
@app.route('/schemes')
def schemes():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('schemes.html', schemes=schemes_data)

# 5. MARKET PRICE & SELL PRODUCE MODULE
@app.route('/sell-produce', methods=['GET', 'POST'])
def sell_produce():
    if 'username' not in session or session.get('user_type') != 'farmer':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        product_name = request.form['product_name']
        description = request.form['description']
        quantity = int(request.form['quantity'])
        quantity_type = request.form['quantity_type']
        price = float(request.form['price'])
        
        product_image = 'https://images.unsplash.com/photo-1542838132-92c53300491e?w=400'
        
        if 'product_image' in request.files:
            file = request.files['product_image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                product_image = f'uploads/{filename}'
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO products (farmer_username, name, description, quantity, 
                                original_quantity, quantity_type, price, image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['username'], product_name, description, quantity, 
              quantity, quantity_type, price, product_image))
        conn.commit()
        conn.close()
        
        flash('Product published successfully!', 'success')
        return redirect(url_for('my_products'))
    
    return render_template('sell_produce.html')

@app.route('/my-products')
def my_products():
    if 'username' not in session or session.get('user_type') != 'farmer':
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products WHERE farmer_username = ? ORDER BY posted_date DESC', 
                   (session['username'],))
    products = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return render_template('my_products.html', products=products)

@app.route('/buy-request', methods=['POST'])
def buy_request():
    if 'username' not in session or session.get('user_type') != 'buyer':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    data = request.json
    product_id = data['product_id']
    quantity = int(data['quantity'])
    payment_details = data['payment_details']
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Find product
    cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
    product = cursor.fetchone()
    
    if not product:
        conn.close()
        return jsonify({'success': False, 'message': 'Product not found'})
    
    if product['quantity'] < quantity:
        conn.close()
        return jsonify({'success': False, 'message': 'Insufficient quantity available'})
    
    # Create transaction
    total_amount = product['price'] * quantity
    cursor.execute('''
        INSERT INTO transactions (product_id, product_name, farmer, buyer, quantity, 
                                quantity_type, price_per_unit, total_amount, payment_details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (product_id, product['name'], product['farmer_username'], session['username'],
          quantity, product['quantity_type'], product['price'], total_amount, payment_details))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Buy request sent successfully!'})

@app.route('/transactions')
def transactions():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user_type = session.get('user_type')
    
    conn = get_db()
    cursor = conn.cursor()
    
    if user_type == 'farmer':
        cursor.execute('SELECT * FROM transactions WHERE farmer = ? ORDER BY request_date DESC', (username,))
    else:
        cursor.execute('SELECT * FROM transactions WHERE buyer = ? ORDER BY request_date DESC', (username,))
    
    user_transactions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return render_template('transactions.html', transactions=user_transactions, user_type=user_type)

@app.route('/accept-transaction/<int:transaction_id>')
def accept_transaction(transaction_id):
    if 'username' not in session or session.get('user_type') != 'farmer':
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get transaction
    cursor.execute('SELECT * FROM transactions WHERE id = ?', (transaction_id,))
    transaction = cursor.fetchone()
    
    if transaction and transaction['farmer'] == session['username']:
        # Update transaction status
        cursor.execute('UPDATE transactions SET status = ? WHERE id = ?', ('accepted', transaction_id))
        
        # Update product quantity
        cursor.execute('''
            UPDATE products SET quantity = quantity - ? WHERE id = ?
        ''', (transaction['quantity'], transaction['product_id']))
        
        conn.commit()
        flash('Transaction accepted successfully!', 'success')
    
    conn.close()
    return redirect(url_for('transactions'))

# 6. LOAN INFORMATION MODULE
@app.route('/loans')
def loans():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('loans.html', loans=loans_data)

# 7. COMMUNITY FORUM MODULE
@app.route('/community')
def community():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get all farmers except current user
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT username, email FROM users 
        WHERE user_type = ? AND username != ?
        ORDER BY username
    ''', ('farmer', session['username']))
    
    farmers = [{'username': row['username'], 'email': row['email']} for row in cursor.fetchall()]
    conn.close()
    
    return render_template('community.html', farmers=farmers)

@app.route('/chat/<farmer_username>')
def chat(farmer_username):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE username = ?', (farmer_username,))
    
    if not cursor.fetchone():
        conn.close()
        flash('User not found', 'error')
        return redirect(url_for('community'))
    
    conn.close()
    return render_template('chat.html', chat_with=farmer_username)

@app.route('/get-messages/<farmer_username>')
def get_messages(farmer_username):
    if 'username' not in session:
        return jsonify([])
    
    user1 = session['username']
    user2 = farmer_username
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sender, message, message_type, timestamp 
        FROM chat_messages 
        WHERE (sender = ? AND receiver = ?) OR (sender = ? AND receiver = ?)
        ORDER BY created_at
    ''', (user1, user2, user2, user1))
    
    messages = []
    for row in cursor.fetchall():
        messages.append({
            'sender': row['sender'],
            'message': row['message'],
            'type': row['message_type'],
            'timestamp': row['timestamp']
        })
    
    conn.close()
    return jsonify(messages)

@app.route('/send-message', methods=['POST'])
def send_message():
    if 'username' not in session:
        return jsonify({'success': False})
    
    data = request.json
    receiver = data['receiver']
    message = data['message']
    message_type = data.get('type', 'text')  # 'text' or 'image'
    
    conn = get_db()
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().strftime("%H:%M")
    
    cursor.execute('''
        INSERT INTO chat_messages (sender, receiver, message, message_type, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (session['username'], receiver, message, message_type, timestamp))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/upload-chat-image', methods=['POST'])
def upload_chat_image():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Ensure directory exists
            chat_images_dir = os.path.join('static', 'chat_images')
            os.makedirs(chat_images_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(chat_images_dir, filename)
            
            # Save the file
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'image_url': f'chat_images/{filename}'
            })
        except Exception as e:
            print(f"Error saving image: {e}")
            return jsonify({'success': False, 'message': f'Error saving image: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Only PNG, JPG, JPEG allowed'})

# CHATBOT MODULE (Groq API)
@app.route('/chatbot')
def chatbot():
    if 'username' not in session or session.get('user_type') != 'farmer':
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/chat-api', methods=['POST'])
def chat_api():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    user_message = data.get('message', '')
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are Thamizh Bot (தமிழ் Bot), a helpful agricultural assistant for Indian farmers. You provide advice on farming, crops, weather, government schemes, loans, pest management, and general agriculture. Keep responses concise, practical, and farmer-friendly. Include Tamil terms where appropriate to connect with farmers."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        
        bot_response = completion.choices[0].message.content
        return jsonify({'response': bot_response})
        
    except Exception as e:
        return jsonify({'response': f'Sorry, I encountered an error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)