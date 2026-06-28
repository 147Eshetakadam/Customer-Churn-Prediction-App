# run.py
from app import app

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Customer Churn Prediction System")
    print("  Running at: http://localhost:5001")
    print("  Login: admin / churn2025")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)