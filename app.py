from flask import Flask, render_template, redirect, url_for, request, jsonify, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
import datetime
from src.pipelines.Models import model_dict
from src.components.ChatBot import AIChatbot
from src.helper import datasets_path, json_load


import os
# from dotenv import load_dotenv
# load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URI")


app = Flask(__name__)

# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "database.db")}'


app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key_here"


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

migrate = Migrate(app, db)



class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    groq_api_key = db.Column(db.String(255), nullable=True) 

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)


class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(50), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now())

    user = db.relationship('User', backref=db.backref('sessions', lazy=True))


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('sessions.session_id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now())

    session = db.relationship('Session', backref=db.backref('chat_history', lazy=True))



@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('home.html', status = 200)
    return render_template('home.html', status = 400)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile', methods=['GET','POST'])
def profile():
    if "user_id" not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        new_api_key = request.form.get("groq_api_key")
        if new_api_key:
            user.groq_api_key = new_api_key
            db.session.commit()
            flash('API Key updated successfully.', 'success')
            return redirect(url_for('profile'))

    return render_template('profile.html', user=user)
    
    
    
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if not email or not password:
            flash('Email and Password are required', 'error')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User(email=email)
        new_user.set_password(password)  # Encrypt the password
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout', methods=['GET','POST'])
def logout():
    session.pop('user_id', None)
    return render_template('home.html',status=400)


@app.route('/save_api_key', methods=['POST'])
def save_api_key():
    if "user_id" not in session:
        return jsonify({'error': 'User not logged in'}), 403

    data = request.get_json()
    api_key = data.get('api_key')

    if not api_key:
        return jsonify({'error': 'API key missing'}), 400

    user = User.query.get(session['user_id'])
    if user:
        user.groq_api_key = api_key
        db.session.commit()
    
    return jsonify({'error': 'User not found'}), 404



@app.route('/create_new_session', methods=['POST'])
def create_new_session():
    if "user_id" not in session:
        return redirect(url_for('login'))
        
    session_id = request.form.get('session_id', '').strip()
    if not session_id:
        return redirect(url_for('chatbot'))
    
    # Check if the session already exists
    existing_session = Session.query.filter_by(session_id=session_id).first()
    if existing_session:
        return redirect(url_for('chatbot', session_id=session_id))
    
    # Create a new session
    new_session = Session(user_id=session['user_id'], session_id=session_id)
    db.session.add(new_session)
    db.session.commit()
    return redirect(url_for('chatbot', session_id=session_id))

    
def get_chat_history(session_id):
    if not session_id:
        return []

    history = ChatHistory.query.filter_by(session_id=session_id).order_by(ChatHistory.timestamp).all()

    formatted_history = []
    for chat in history:
        formatted_history.append({"role": "user", "content": chat.user_message})
        formatted_history.append({"role": "assistant", "content": chat.bot_response})
    
    return formatted_history
    
def get_user_sessions():
    return Session.query.filter_by(user_id=session['user_id']).all()

@app.route('/chatbot', methods=['GET','POST'])
def chatbot():
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    if not user or not user.groq_api_key:
        return redirect(url_for('profile'))

    
    api_key = user.groq_api_key
    current_session_id = request.args.get('session_id',None)
    
    if request.method == 'POST':
        question = request.form['question']
        session_id = request.form['session_id'] or current_session_id
        
        if not session_id or not question:
            return redirect(url_for('chatbot', session_id=session_id) if session_id else url_for('chatbot'))
        
        chatbot_model = AIChatbot(api_key)
            
        history = get_chat_history(session_id)
        
        
        bot_response = chatbot_model.generate_response(question, history)
        
        chat_entry = ChatHistory(session_id=session_id, user_message=question, bot_response=bot_response)
        db.session.add(chat_entry)
        db.session.commit()
        
        return redirect(url_for('chatbot', session_id=session_id) if session_id else url_for('chatbot'))
    
    chat_history = get_chat_history(current_session_id) if current_session_id else []
    
    return render_template('chatbot.html', user_sessions=get_user_sessions(), current_session_id=current_session_id, chat_history=chat_history, show_api_popup=False, groq_api_key=api_key)




@app.route('/explore_datasets')
def explore_datasets():
    return render_template('explore_datasets.html', dataset_dict = datasets_path(), count = enumerate)


@app.route('/explore_models')
def explore_models():
    return render_template('explore_models.html', model_dict=model_dict, count = enumerate)


@app.route('/explore_categories')
def explore_categories():
    cat = json_load('artifacts/def_types.json')
    return render_template('explore_categories.html', categories = datasets_path().keys(), count = enumerate, cat = cat)





@app.route('/model/<model>')
def model(model):
    types = request.args.get('types')
    return render_template('model.html', name=model, types=types)


@app.route('/dataset/<dataset>')
def dataset(dataset):
    types = request.args.get('types')
    return render_template('dataset.html', name=dataset, types=types)


@app.route('/category/<category>')
def category(category):
    types = request.args.get('types')
    return render_template('category.html', name=category, types=types)


@app.route('/explore')
def explore():
    option = request.args.get('option')      ## eg: 'explore_datasets' 
    name = request.args.get('name')          ## eg: 'iris' or 'classification'
    which = request.args.get('which')        ## eg: is it dataset or model
    types = request.args.get('types')        ## eg: 'classification' or 'regression'
    
    if "user_id" not in session:
        return redirect(url_for('login'))
    
    return redirect(f"http://98.84.178.48:8501/?page={which}&select={option}&name={name}&types={types}&session={session['user_id']}")
    # return redirect(f"http://localhost:8501/?page={which}&select={option}&name={name}&types={types}&session={session['user_id']}")


if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)