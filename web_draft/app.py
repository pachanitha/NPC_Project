#app.py
from pymongo import MongoClient
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from celery import Celery
import model
from celery.exceptions import Ignore
import logging
import datetime
import uuid  # For generating unique IDs
import os
import bcrypt

# MongoDB setup
uri = "mongodb+srv://pachanithafah:slN5OnTuC71f4S9o@cluster0.oqnm8.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true"
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("Connected to MongoDB!")
except Exception as e:
    print("Failed to connect to MongoDB:", e)

mydb = client["NPC_Web"]
mycol = mydb["DB_NPC"]
upload_sessions = mydb["upload_sessions"]
analysis_sessions = mydb["analysis_sessions"]
users_collection = mydb["users"]

# Setup Flask
app = Flask(__name__)
app.secret_key = '111'  # ควรเปลี่ยนเป็น key ที่ปลอดภัยกว่านี้ใน production

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['SESSION_TYPE'] = 'filesystem'

# Optional: Configure session settings for added security
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # True ถ้าใช้ HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize Celery with Flask app context
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@celery.task(bind=True)
def classify(self, code):
    """
    งาน Classification: เรียกใช้ model.classify_code(code)
    คืนค่าความเป็นไปได้ของ ChatGPT/Human
    """
    try:
        logger.info(f"Classifying code: {code[:50]}...")  # Log แค่ 50 ตัวแรกกันยาวเกิน
        probability_chatgpt, probability_human = model.classify_code(code)
        logger.info(f"ChatGPT probability: {probability_chatgpt}, Human probability: {probability_human}")
        return {
            'probability_chatgpt': probability_chatgpt * 100,
            'probability_human': probability_human * 100
        }
    except Exception as e:
        self.update_state(state='FAILURE', meta={'exc_message': str(e), 'exc_type': type(e).__name__})
        logger.error(f"Classification task failed: {e}")
        raise Ignore()

@celery.task(bind=True)
def generate_explanation(self, code, probability_chatgpt, probability_human, local_first, local_second, model_choice):
    """
    งาน Generate Explanation: เรียกใช้ model.gen_explanation(...)
    """
    try:
        if not local_first or not local_second:
            logger.error("Local neighborhoods are empty.")
            raise ValueError("Local neighborhoods are empty.")

        logger.debug(f"Starting explanation generation for model: {model_choice}")
        explanation_result = model.gen_explanation(
            code, probability_chatgpt, probability_human, local_first, local_second, model_choice
        )
        logger.info("Explanation generated successfully.")
        return {'explanation': explanation_result}
    except Exception as e:
        self.update_state(state='FAILURE', meta={'exc_message': str(e), 'exc_type': type(e).__name__})
        logger.error(f"Explanation generation failed: {e}")
        raise Ignore()

# Helper function for local neighborhood analysis
def local_neighborhood(code):
    """
    เรียกใช้งาน model.localneighborhood(...) สองครั้ง (prefix_0, prefix_1)
    """
    try:
        prefix_0 = '0_'
        prefix_1 = '1_'
        k = 4
        ratio = [70, 20, 10]

        local_first = model.localneighborhood(code, prefix_0, k, ratio)
        local_second = model.localneighborhood(code, prefix_1, k, ratio)

        return local_first, local_second
    except Exception as e:
        logger.error(f"Local neighborhood analysis failed: {e}")
        return None, None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    return render_template('analyze.html')

# In-memory cache to track ongoing requests
request_cache = {}

# Result route for classification and explanation
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        try:
            data = request.get_json()
            code = data.get('code', '').strip()
            model_choice = data.get('model', None)
            additional_check = data.get('additional_check', '')

            logger.info(f"Processing code for model: {model_choice}")

            # Validate input
            if not code:
                logger.warning("No code provided in the request.")
                return jsonify({'error': 'Code input is required'}), 400
            if model_choice not in ['ChatGPT', 'Gemini', 'LLaMa']:
                logger.warning(f"Invalid model choice: {model_choice}")
                return jsonify({'error': f'Invalid model choice: {model_choice}'}), 400

            # Generate a unique request_id
            request_id = uuid.uuid4().hex
            logger.info(f"Generated request_id: {request_id}")

            # Launch classification task asynchronously
            classification_task = classify.apply_async(args=[code])
            classification_result = classification_task.get(timeout=300)  # Wait up to 5 minutes

            probability_chatgpt = classification_result.get('probability_chatgpt', None)
            probability_human = classification_result.get('probability_human', None)

            if probability_chatgpt is None or probability_human is None:
                logger.error("Classification task did not return expected probabilities.")
                return jsonify({'error': 'Classification failed to return probabilities.'}), 500

            # Perform local neighborhood analysis
            local_first, local_second = local_neighborhood(code)
            if local_first is None or local_second is None:
                logger.error("Local neighborhood analysis failed.")
                return jsonify({'error': 'Local neighborhood analysis failed.'}), 500

            # Launch explanation task
            explanation_task = generate_explanation.apply_async(
                args=[code, probability_chatgpt, probability_human, local_first, local_second, model_choice]
            )
            explanation_result = explanation_task.get(timeout=300)  # Wait up to 5 minutes

            explanation = explanation_result.get('explanation', None)
            if explanation is None:
                logger.error("Explanation task did not return an explanation.")
                return jsonify({'error': 'Explanation generation failed.'}), 500

            # Insert all necessary fields into MongoDB
            user_email = session.get('user', 'Anonymous')  # Default to 'Anonymous' if not logged in
            mycol.insert_one({
                "request_id": request_id,
                "classification_task_id": classification_task.id,
                "explanation_task_id": explanation_task.id,
                "email": user_email,
                "code": code,
                "model_choice": model_choice,
                "probability_chatgpt": probability_chatgpt,
                "probability_human": probability_human,
                "classification": "🤖 ChatGPT-generated" if probability_chatgpt >= 50 else "👩🏻‍💻 Human",
                "explanation": explanation,
                "created_at": datetime.datetime.utcnow()
            })
            logger.info(f"Inserted result into MongoDB with request_id: {request_id}")

            # Store only request_id in session for feedback
            session['analysis_request_id'] = request_id

            # Return task IDs to frontend
            return jsonify({
                'classification_task_id': classification_task.id,
                'explanation_task_id': explanation_task.id,
                'request_id': request_id,
                'message': 'Tasks completed successfully.'
            }), 200

        except Exception as e:
            logger.error(f"Error submitting tasks: {str(e)}")
            return jsonify({'error': 'Failed to submit tasks. Please try again later.'}), 500

    return render_template('result.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()
            email = data.get('email', '').strip()
            password = data.get('password', '').strip()

            if not email or not password:
                return jsonify({'success': False, 'message': 'Email and password are required.'}), 400

            # Check if the user already exists
            existing_user = users_collection.find_one({'email': email})
            if existing_user:
                return jsonify({'success': False, 'message': 'Email already registered.'}), 400

            # Hash the password before storing
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Insert the new user
            new_user = {
                'email': email,
                'password': hashed_password.decode('utf-8'),
                'registered_at': datetime.datetime.utcnow()
            }
            users_collection.insert_one(new_user)

            return jsonify({'success': True, 'message': 'Registration successful! Please log in.'}), 200
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({'success': False, 'message': 'An error occurred during registration.'}), 500
    else:
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not email or not password:
            return jsonify({'success': False, 'message': 'Both email and password are required.'}), 400

        # Fetch user from MongoDB
        user = users_collection.find_one({'email': email})
        if not user:
            return jsonify({'success': False, 'message': 'User not found. Please register first.'}), 400

        # Verify password
        stored_hashed_password = user.get('password', '')
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
            return jsonify({'success': False, 'message': 'Incorrect password.'}), 400

        # Set user in session
        session['user'] = email
        logger.info(f"User {email} has logged in.")
        return jsonify({'success': True, 'message': f'Welcome, {email}!'}), 200
    else:
        return render_template('login.html')

@app.route('/logout', methods=['GET'])
def logout():
    if 'user' in session:
        user = session.pop('user', None)
        logger.info(f"User {user} has logged out.")
        return jsonify({'success': True, 'message': 'Logged out successfully.'}), 200
    else:
        return jsonify({'success': False, 'message': 'No user is currently logged in.'}), 400

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    user_email = session['user']
    return render_template('dashboard.html', email=user_email)

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = celery.AsyncResult(task_id)
    logger.debug(f"Task ID: {task_id}, State: {task.state}, Info: {task.info}")

    if task.state == 'PENDING':
        response = {'state': task.state, 'progress': 0, 'status': 'Task is in the queue.'}
    elif task.state == 'FAILURE':
        response = {'state': task.state, 'status': str(task.info)}
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'result': task.result}
    else:
        response = {
            'state': task.state,
            'progress': task.info.get('progress', 0),
            'status': task.info.get('status', '')
        }

    logger.debug(f"Response: {response}")
    return jsonify(response)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        logger.debug(f"Feedback received: {data}")

        request_id = data.get('request_id')
        upload_id = data.get('upload_id')
        ratings = data.get('ratings', {})
        feedback_additional = data.get('feedback_additional', '').strip()

        logger.debug(f"Request ID: {request_id}, Upload ID: {upload_id}, Ratings: {ratings}, Additional: {feedback_additional}")

        required_categories = [
            'overall',
            'conciseness',
            'clarity',
            'actionability',
            'selected_line_useful',
            'additional_code_useful'
        ]

        for category in required_categories:
            if category not in ratings:
                return jsonify({'error': f'Rating for "{category}" is required.'}), 400
            if not isinstance(ratings[category], int) or not (1 <= ratings[category] <= 5):
                return jsonify({'error': f'Rating for "{category}" must be an integer between 1 and 5.'}), 400

        if not request_id and not upload_id:
            return jsonify({'error': 'Invalid feedback submission. No identifier provided.'}), 400

        feedback_data = {
            "ratings": ratings,
            "feedback_additional": feedback_additional,
            "submitted_at": datetime.datetime.now(datetime.timezone.utc)
        }

        if request_id:
            result = mycol.find_one({"request_id": request_id})
            if not result:
                return jsonify({'error': 'Result not found. Please re-run the analysis.'}), 400
            mycol.update_one({"request_id": request_id}, {"$set": {"feedback": feedback_data}})
        elif upload_id:
            upload_session = upload_sessions.find_one({"upload_id": upload_id})
            if not upload_session:
                return jsonify({'error': 'Upload session not found. Please re-run the analysis.'}), 400
            upload_sessions.update_one({"upload_id": upload_id}, {"$set": {"feedback": feedback_data}})

        return jsonify({'success': True, 'message': 'Feedback submitted successfully!'}), 200

    except Exception as e:
        logger.error(f"Error during feedback submission: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

@app.route('/result/<classification_task_id>/<explanation_task_id>', methods=['GET'])
def show_result(classification_task_id, explanation_task_id):
    result = mycol.find_one({
        "classification_task_id": classification_task_id,
        "explanation_task_id": explanation_task_id
    })
    logger.debug(f"Fetched result from MongoDB: {result}")

    if not result:
        return jsonify({'error': 'Result not found. Please re-run the analysis.'}), 404

    probability_chatgpt = result.get('probability_chatgpt', "N/A")
    probability_human = result.get('probability_human', "N/A")
    explanation = result.get('explanation', "No explanation available.")
    code = result.get('code', "No code available.")
    email = result.get('email', 'Anonymous')

    return render_template(
        'result.html',
        classification_task_id=classification_task_id,
        explanation_task_id=explanation_task_id,
        probability_chatgpt=probability_chatgpt,
        probability_human=probability_human,
        explanation=explanation,
        code=code,
        request_id=result['request_id'],
        email=email
    )

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    processed_results = []

    if 'files[]' not in request.files:
        logger.error("No files uploaded.")
        return jsonify({'error': 'No files uploaded.'}), 400

    files = request.files.getlist('files[]')
    model_choice = request.form.get('model', None)
    additional_check = request.form.get('additional_check', '')
    logger.info(f"Received {len(files)} file(s) with model choice: {model_choice}")

    if model_choice not in ['ChatGPT', 'Gemini', 'LLaMa']:
        logger.error(f"Unsupported model choice: {model_choice}")
        return jsonify({'error': f"Unsupported model choice: {model_choice}"}), 400

    for file in files:
        filename = file.filename or f"default_file_name_{datetime.datetime.utcnow().timestamp()}"
        base_filename = os.path.basename(filename)

        logger.info(f"Processing file: {filename}")

        # Skip hidden files
        if base_filename.startswith('.'):
            logger.info(f"Skipping hidden file: {filename}")
            continue

        try:
            # Read file content
            file_content = file.read().decode('utf-8', errors='replace').strip()
            logger.info(f"File content read successfully for: {filename}")

            request_id = uuid.uuid4().hex

            # Classification
            classification_task = classify.apply_async(args=[file_content])
            classification_result = classification_task.get(timeout=300)
            probability_chatgpt = classification_result.get('probability_chatgpt', None)
            probability_human = classification_result.get('probability_human', None)

            if probability_chatgpt is None or probability_human is None:
                logger.error(f"Classification failed for file: {filename}")
                processed_results.append({
                    'file_name': base_filename,
                    'status': 'Error',
                    'error': 'Classification failed to return probabilities.'
                })
                continue

            # Local neighborhood analysis
            local_first, local_second = local_neighborhood(file_content)
            if local_first is None or local_second is None:
                logger.error(f"Local neighborhood analysis failed for file: {filename}")
                processed_results.append({
                    'file_name': base_filename,
                    'status': 'Error',
                    'error': 'Local neighborhood analysis failed.'
                })
                continue

            # Explanation
            explanation_task = generate_explanation.apply_async(
                args=[file_content, probability_chatgpt, probability_human, local_first, local_second, model_choice]
            )
            explanation_result = explanation_task.get(timeout=300)
            explanation = explanation_result.get('explanation', None)

            if explanation is None:
                logger.error(f"Explanation generation failed for file: {filename}")
                processed_results.append({
                    'file_name': base_filename,
                    'status': 'Error',
                    'error': 'Explanation generation failed.'
                })
                continue

            # Insert result to DB
            user_email = session.get('user', 'Anonymous')
            mycol.insert_one({
                "request_id": request_id,
                "classification_task_id": classification_task.id,
                "explanation_task_id": explanation_task.id,
                "email": user_email,
                "file_name": base_filename,
                "code": file_content,
                "model_choice": model_choice,
                "probability_chatgpt": probability_chatgpt,
                "probability_human": probability_human,
                "classification": "🤖 ChatGPT-generated" if probability_chatgpt >= 50 else "👩🏻‍💻 Human",
                "explanation": explanation,
                "created_at": datetime.datetime.utcnow()
            })
            logger.info(f"Inserted result into MongoDB for file: {filename}")

            processed_results.append({
                'file_name': base_filename,
                'request_id': request_id,
                'probability_chatgpt': round(probability_chatgpt, 2),
                'probability_human': round(probability_human, 2),
                'classification': "🤖 ChatGPT-generated" if probability_chatgpt >= 50 else "👩🏻‍💻 Human",
                'explanation': explanation,
                'status': 'Complete'
            })

        except Exception as e:
            processed_results.append({
                'file_name': base_filename,
                'status': 'Error',
                'error': str(e)
            })
            logger.error(f"Failed processing {filename}: {e}")

    # Insert upload session
    upload_id = uuid.uuid4().hex
    user_email = session.get('user', 'Anonymous')
    upload_sessions.insert_one({
        "upload_id": upload_id,
        "email": user_email,
        "processed_results": processed_results,
        "created_at": datetime.datetime.utcnow()
    })

    return jsonify({'upload_id': upload_id, 'message': 'Folder processed successfully.'}), 200

@app.route('/resultUploadFolder/<upload_id>')
def result_upload_folder(upload_id):
    upload_session = upload_sessions.find_one({"upload_id": upload_id})
    logger.debug(f"Fetched upload_session from MongoDB for upload_id {upload_id}: {upload_session}")

    if not upload_session:
        return jsonify({'error': 'Upload results not found.'}), 404

    email = upload_session.get('email', 'Anonymous')
    processed_results = upload_session.get('processed_results', [])

    return render_template('resultUploadFolder.html', 
                           email=email,
                           results=processed_results)

@app.route('/explanation/<request_id>', methods=['GET'])
def explanation(request_id):
    result = mycol.find_one({"request_id": request_id})
    logger.debug(f"Fetched result from MongoDB for request_id {request_id}: {result}")

    if not result:
        return jsonify({'error': 'Result not found for this request ID.'}), 404

    return render_template('explanation_folder.html', result=result, request_id=request_id)

if __name__ == '__main__':
    app.run(debug=True)


# cd /Users/fah/Downloads/senior_1/web_draft
# rm -rf myenv
# python3.11 -m venv myenv
# source myenv/bin/activate
# pip install --upgrade pip
# pip install -r requirements.txt
# pip install Flask

# Termianl1: redis-server (Redis Check: redis-cli ping, Stop server: redis-cli shutdown)
# Terminal2: python -m celery -A app.celery worker --loglevel=info
# Ternimal3: python app.py

# Stop the Worker: pkill -f "celery worker"


# brew install git-lfs  
# git lfs install
# git lfs status
# git lfs fetch --all

# git lfs pull

# cd graphcodebert-base
# git lfs pull

# brew services start mongodb-community@8.0
# python -m pip install "pymongo[srv]"==3.11
# cd /Users/fah/Downloads/senior_1/web_draft
# rm -rf myenv
# python3.11 -m venv myenv
# source myenv/bin/activate
# pip install --upgrade pip
# pip install -r requirements.txt
# pip install Flask

# Termianl1: redis-server (Redis Check: redis-cli ping, Stop server: redis-cli shutdown)
# Terminal2: python -m celery -A app.celery worker --loglevel=info
# Ternimal3: python app.py

# Stop the Worker: pkill -f "celery worker"


# brew install git-lfs  
# git lfs install
# git lfs status
# git lfs fetch --all

# git lfs pull

# cd graphcodebert-base
# git lfs pull

# brew services start mongodb-community@8.0
# python -m pip install "pymongo[srv]"==3.11