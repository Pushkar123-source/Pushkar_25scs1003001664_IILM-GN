Flask Backend API for Drug Classification
RESTful API endpoints for drug type prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import DrugClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize classifier
classifier = DrugClassifier()

# Global variable to store dataset stats
dataset_stats = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.model is not None
    })

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load pre-trained model"""
    try:
        model_path = request.json.get('model_path', '../models/')
        classifier.load_model(model_path)
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully',
            'classes': classifier.label_encoder.classes_.tolist()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model on dataset"""
    try:
        csv_path = request.json.get('csv_path', '../onemg.csv')
        accuracy = classifier.train(csv_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'classes': classifier.label_encoder.classes_.tolist()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict drug type from text description
    Request body: { "text": "drug description..." }
    """
    try:
        if classifier.model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please load or train the model first.'
            }), 400
        
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text field is required'
            }), 400
        
        result = classifier.predict(text)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple texts
    Request body: { "texts": ["text1", "text2", ...] }
    """
    try:
        if classifier.model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please load or train the model first.'
            }), 400
        
        data = request.json
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'success': False,
                'error': 'Texts array is required'
            }), 400
        
        results = []
        for text in texts:
            result = classifier.predict(text)
            results.append(result)
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset-stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        csv_path = '../onemg.csv'
        df = pd.read_csv(csv_path)
        
        stats = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'drug_types': df['Drug_Type'].value_counts().to_dict(),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        if classifier.model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 400
        
        info = {
            'model_type': 'Multinomial Naive Bayes',
            'vectorizer': 'TF-IDF',
            'classes': classifier.label_encoder.classes_.tolist(),
            'num_classes': len(classifier.label_encoder.classes_),
            'vocabulary_size': len(classifier.tfidf_vectorizer.vocabulary_),
            'features': classifier.tfidf_vectorizer.max_features
        }
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sample-drugs', methods=['GET'])
def get_sample_drugs():
    """Get sample drug data"""
    try:
        csv_path = '../onemg.csv'
        df = pd.read_csv(csv_path)
        
        # Get 10 random samples
        samples = df.sample(n=min(10, len(df)))[
            ['Drug_Name', 'Drug_Type', 'Product_Introduction', 'Uses']
        ].to_dict('records')
        
        return jsonify({
            'success': True,
            'samples': samples
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask Backend Server...")
    print("Loading model...")
    
    try:
        # Try to load existing model
        classifier.load_model('../models/')
        print("Model loaded successfully!")
    except:
        print("No pre-trained model found. Please train the model first.")
    
    print("\nAPI Endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/train - Train model")
    print("  POST /api/load-model - Load pre-trained model")
    print("  POST /api/predict - Predict drug type")
    print("  POST /api/batch-predict - Batch prediction")
    print("  GET  /api/dataset-stats - Dataset statistics")
    print("  GET  /api/model-info - Model information")
    print("  GET  /api/sample-drugs - Sample drug data")
    print("\nServer running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)