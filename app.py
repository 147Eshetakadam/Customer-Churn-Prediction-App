# app.py
# Main Flask application entry point for Customer Churn Prediction System

import os
import json
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash)
from werkzeug.utils import secure_filename

from utils.preprocessor import load_and_preprocess, preprocess_single
from utils.trainer import train_model, predict_single
from utils.visualizer import (get_churn_distribution, get_confusion_matrix_data,
                               get_feature_importance_chart, get_tenure_churn_chart,
                               get_monthly_charges_chart)

app = Flask(__name__)
app.secret_key = 'churn_predict_secret_2025'

# ─── Config ───────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# In-memory storage for current session data
_app_state = {
    'df_raw': None,
    'df_processed': None,
    'feature_names': None,
    'last_metrics': None,
    'dataset_stats': None
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Auth (Basic) ─────────────────────────────────────────────────────────────
USERS = {'admin': 'churn2025', 'analyst': 'telecom123'}


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if USERS.get(username) == password:
            session['user'] = username
            return redirect(url_for('index'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ─── Main Routes ──────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html',
                           user=session.get('user'),
                           has_data=_app_state['df_raw'] is not None,
                           has_model=os.path.exists('model/active_model.pkl'),
                           stats=_app_state['dataset_stats'])


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files allowed'}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess
            df_raw = pd.read_csv(filepath)
            df_proc, X, y, feature_names = load_and_preprocess(filepath)

            _app_state['df_raw'] = df_raw
            _app_state['df_processed'] = df_proc
            _app_state['feature_names'] = feature_names
            _app_state['filepath'] = filepath

            # Dataset stats
            churn_col = 'Churn'
            churn_count = 0
            churn_pct = 0
            if churn_col in df_raw.columns:
                churn_series = df_raw[churn_col].map({'Yes': 1, 'No': 0}).fillna(df_raw[churn_col])
                churn_count = int(pd.to_numeric(churn_series, errors='coerce').sum())
                churn_pct = round(churn_count / len(df_raw) * 100, 1)

            _app_state['dataset_stats'] = {
                'total_records': len(df_raw),
                'total_features': len(df_raw.columns),
                'churn_count': churn_count,
                'churn_pct': churn_pct,
                'missing_values': int(df_raw.isnull().sum().sum()),
                'processed_features': len(feature_names)
            }

            # Preview (first 10 rows)
            preview = df_raw.head(10).to_dict(orient='records')
            columns = list(df_raw.columns)

            return jsonify({
                'success': True,
                'stats': _app_state['dataset_stats'],
                'preview': preview,
                'columns': columns
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('upload.html', user=session.get('user'),
                           stats=_app_state['dataset_stats'],
                           has_data=_app_state['df_raw'] is not None)


@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    if request.method == 'POST':
        if _app_state['df_processed'] is None:
            return jsonify({'error': 'No dataset loaded. Please upload data first.'}), 400

        model_type = request.json.get('model_type', 'random_forest')
        test_size = float(request.json.get('test_size', 0.2))

        try:
            df = _app_state['df_processed']
            X = df.drop(columns=['Churn']) if 'Churn' in df.columns else df
            y = df['Churn'] if 'Churn' in df.columns else None

            if y is None:
                return jsonify({'error': 'No Churn column in dataset'}), 400

            metrics = train_model(X, y, model_type=model_type, test_size=test_size)
            _app_state['last_metrics'] = metrics

            return jsonify({'success': True, 'metrics': metrics})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('train.html', user=session.get('user'),
                           has_data=_app_state['df_raw'] is not None,
                           metrics=_app_state['last_metrics'])


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            form_data = request.json
            X_proc = preprocess_single(form_data)
            prediction, probability = predict_single(X_proc)
            result = {
                'churn': bool(prediction),
                'label': 'CHURN' if prediction else 'NO CHURN',
                'probability': probability,
                'risk_level': 'High' if probability >= 70 else ('Medium' if probability >= 40 else 'Low')
            }
            return jsonify({'success': True, 'result': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('predict.html', user=session.get('user'),
                           has_model=os.path.exists('model/active_model.pkl'))


@app.route('/dashboard')
@login_required
def dashboard():
    charts = {}
    if _app_state['df_processed'] is not None:
        df = _app_state['df_processed']
        df_raw = _app_state['df_raw']
        charts['churn_dist'] = get_churn_distribution(df)
        charts['tenure_churn'] = get_tenure_churn_chart(df_raw if df_raw is not None else df)
        charts['monthly_charges'] = get_monthly_charges_chart(df_raw if df_raw is not None else df)

    if _app_state['last_metrics']:
        m = _app_state['last_metrics']
        charts['confusion_matrix'] = get_confusion_matrix_data(m['confusion_matrix'])
        if 'feature_importance' in m:
            charts['feature_importance'] = get_feature_importance_chart(m['feature_importance'])

    return render_template('dashboard.html',
                           user=session.get('user'),
                           charts_json=json.dumps(charts),
                           metrics=_app_state['last_metrics'],
                           stats=_app_state['dataset_stats'],
                           has_data=_app_state['df_raw'] is not None)


# if __name__ == '__main__':
#     os.makedirs('model', exist_ok=True)
#     print("\n" + "="*50)
#     print("  Customer Churn Prediction System")
#     print("  Running at: http://localhost:5000")
#     print("  Login: admin / churn2025")
#     print("="*50 + "\n")
#     app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
