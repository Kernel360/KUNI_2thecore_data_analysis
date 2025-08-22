# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based data analysis project for KUNI 2thecore that includes MySQL database connectivity and data analysis capabilities. The project uses Python 3.13.5 and focuses on data loading and analysis workflows.

## Environment Setup

The project requires a virtual environment with dependencies installed from `requirements.txt`:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

## Database Configuration

The project uses MySQL with SQLAlchemy and requires a `.env` file with database credentials:
- DB_HOST
- DB_USER  
- DB_PASSWORD
- DB_NAME
- DB_PORT

Database connection and data loading functionality is centralized in `src/data_loader.py`.

## Project Structure

- `src/data_loader.py` - Core database connection and data loading utilities
  - `get_db_connection()` - Creates SQLAlchemy engine from .env credentials
  - `get_data_from_db(query)` - Executes SQL queries and returns pandas DataFrames
- `verify_setup.py` - Environment verification script for testing library imports
- `requirements.txt` - Comprehensive dependency list including pandas, numpy, scikit-learn, matplotlib, seaborn, SQLAlchemy, mysql-connector-python, and Jupyter

## Development Workflow

Since this is a data analysis project with Flask web server capabilities, use these commands:

```bash
# Test database connection and data loading
python src/data_loader.py

# Verify environment setup
python verify_setup.py

# Start Flask web server
python app.py
# or
python run_server.py

# Start Jupyter for analysis
jupyter lab
```

## Flask Web Server

The project includes a Flask-based REST API server for data analysis:

- **Main Application**: `app.py` - Core Flask application with REST API endpoints
- **Server Runner**: `run_server.py` - Development server startup script
- **Base URL**: `http://localhost:5000`

### API Endpoints

- `GET /` - API information and available endpoints
- `POST /api/data` - Execute SQL queries and return results as JSON
- `GET /api/health` - Health check and database connectivity status

### API Usage Example

```bash
# Health check
curl http://localhost:5000/api/health

# Execute a query
curl -X POST http://localhost:5000/api/data \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM car LIMIT 5;"}'
```

## Key Libraries and Dependencies

The project includes a full data science stack:
- **Database**: SQLAlchemy, mysql-connector-python, python-dotenv
- **Data Analysis**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn  
- **Visualization**: matplotlib, seaborn
- **Web Server**: Flask, Flask-CORS, Flask-RESTful, Werkzeug
- **Jupyter**: jupyterlab, ipykernel
- **Development**: All supporting libraries for data analysis workflows