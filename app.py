import os
import json
import time
import threading
import logging
from flask import Flask, render_template, jsonify, request
from engine import engine

app = Flask(__name__)

# Background auto-updater thread (runs every hour)
def background_hourly_updater():
    logging.info("Starting background hourly updater thread...")
    while True:
        try:
            # Refresh every 3600 seconds (1 hour)
            time.sleep(3600)
            logging.info("Hourly timer triggered: refreshing crude oil intelligence...")
            engine.refresh(force=True)
        except Exception as e:
            logging.error(f"Error in background updater: {e}")
            time.sleep(60)

# Start background thread as daemon
updater_thread = threading.Thread(target=background_hourly_updater, daemon=True)
updater_thread.start()

@app.route('/')
def index():
    return render_template('minimal.html')

@app.route('/classic')
def classic():
    return render_template('index.html')

@app.route('/ml')
def ml_pipeline():
    return render_template('ml.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        force = request.args.get('force', 'false').lower() == 'true'
        data = engine.refresh(force=force)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logging.error(f"API Error in get_data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/refresh', methods=['POST', 'GET'])
def trigger_refresh():
    try:
        data = engine.refresh(force=True)
        return jsonify({"success": True, "data": data, "message": "Market data refreshed successfully"})
    except Exception as e:
        logging.error(f"API Error in trigger_refresh: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/metrics')
def get_metrics():
    """Serve the latest offline backtest summary (run `python -m evaluation.run_evaluation`)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'evaluation_results.json')
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "No evaluation yet. Run: python -m evaluation.run_evaluation"}), 404
    with open(path) as f:
        return jsonify({"success": True, "data": json.load(f)})

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "service": "crude-oil-intelligence-engine"})

if __name__ == '__main__':
    # Initial load on startup
    logging.info("Pre-warming engine cache...")
    try:
        engine.refresh(force=True)
    except Exception as err:
        logging.warning(f"Initial warm-up warning: {err}")
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
