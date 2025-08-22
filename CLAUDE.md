# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based data analysis project for KUNI 2thecore that provides data analysis reports for car rental companies. The project analyzes rental car usage patterns, brand preferences, seasonal trends, and location optimization using MySQL database connectivity and advanced data analysis capabilities.

### Business Purpose
The system provides data-driven insights to rental car companies using two main data sources:
- **car** table: Vehicle information including model, brand, year, type, status, and location
- **drivelog** table: Driving records with trip details, timestamps, locations, and events

### Core Services Provided
1. **Seasonal Brand/Vehicle Preference Analysis** - Monthly and seasonal analysis of preferred brands and vehicles with seasonality evaluation
2. **Trend Analysis** - Year-over-year changes in brand and vehicle preferences 
3. **Daily Vehicle Usage Forecasting** - Daily operational vehicle count analysis with linear regression forecasting (1 week to 1 month ahead)
4. **Location Clustering Analysis** - Regional importance quantification for optimal rental location placement

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

### Database Schema

#### car table
- `car_id` (Primary Key): Unique vehicle identifier
- `model`: Vehicle model name (e.g., K3, 투싼, 아이오닉 5)
- `brand`: Manufacturer (기아, 현대, 제네시스)
- `status`: Current status (IDLE, MAINTENANCE)
- `car_year`: Manufacturing year
- `car_type`: Vehicle category (소형, 중형, 대형, SUV)
- `car_number`: License plate number
- `sum_dist`: Total distance traveled (km)
- `login_id`: Associated user/company ID
- `last_latitude`, `last_longitude`: Current GPS coordinates

#### drivelog table  
- `drive_log_id` (Primary Key): Unique trip identifier
- `car_id` (Foreign Key): References car table
- `drive_dist`: Trip distance (km)
- `start_point`, `end_point`: Location names
- `start_latitude`, `start_longitude`: Trip origin coordinates
- `end_latitude`, `end_longitude`: Trip destination coordinates  
- `start_time`, `end_time`: Trip timestamps
- `created_at`: Record creation timestamp
- `model`, `brand`: Denormalized vehicle information
- `memo`: Trip notes/events (급감속, 과열 경고, etc.)
- `status`: Trip status

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

### Core Analysis API Endpoints

- `GET /` - API information and available endpoints  
- `POST /api/data` - Execute SQL queries and return results as JSON
- `GET /api/health` - Health check and database connectivity status

#### Planned Data Analysis Endpoints
- `GET /api/analysis/seasonal-preferences` - Monthly/seasonal brand and vehicle preference analysis
- `GET /api/analysis/trend-analysis` - Year-over-year brand/vehicle preference trends
- `GET /api/analysis/daily-usage-forecast` - Daily vehicle usage counts with forecasting
- `GET /api/analysis/location-clustering` - Regional importance analysis for location optimization

### API Usage Example

```bash
# Health check
curl http://localhost:5000/api/health

# Execute a query
curl -X POST http://localhost:5000/api/data \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM car LIMIT 5;"}'
```

## Data Analysis Services

### 1. Seasonal Brand/Vehicle Preference Analysis
**Purpose**: Analyze monthly and seasonal patterns in brand and vehicle preferences
- Calculate preference ratios by month/season for each brand (기아, 현대, 제네시스)
- Evaluate seasonality strength using statistical measures
- Identify peak seasons for specific vehicle types (소형, 중형, 대형, SUV)
- Generate seasonal preference rankings and trends

**Key Metrics**:
- Monthly/seasonal rental frequency by brand and model
- Seasonality index calculation
- Brand market share variations across seasons
- Vehicle type preference patterns

### 2. Year-over-Year Trend Analysis  
**Purpose**: Track long-term changes in brand and vehicle preferences over multiple years
- Compare brand preference shifts year-over-year
- Identify emerging trends in vehicle type preferences
- Analyze market share evolution for each manufacturer
- Detect preference pattern changes over time

**Key Metrics**:
- Annual growth rates by brand
- Market share trend analysis
- Vehicle type adoption patterns
- Preference volatility measurements

### 3. Daily Vehicle Usage Forecasting
**Purpose**: Provide operational insights and future demand prediction
- Analyze daily operational vehicle counts over specified periods
- Generate visual graphs of usage patterns
- Implement linear regression forecasting for 1-week to 1-month ahead
- Create interactive dashboards with forecast visualizations

**Key Features**:
- Real-time daily usage tracking
- Seasonal adjustment in forecasting models
- Confidence intervals for predictions
- Visual trend analysis with matplotlib/seaborn

### 4. Location Clustering and Regional Analysis
**Purpose**: Optimize rental location placement through data-driven regional analysis
- Perform clustering analysis on trip start/end coordinates
- Quantify regional importance based on usage density
- Identify optimal locations for new rental stations
- Analyze geographical usage patterns and accessibility

**Key Techniques**:
- K-means clustering on geographical coordinates
- Regional demand density analysis
- Distance-based accessibility scoring
- Heatmap visualizations for location insights

## Key Libraries and Dependencies

The project includes a full data science stack:
- **Database**: SQLAlchemy, mysql-connector-python, python-dotenv
- **Data Analysis**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn (clustering, regression)
- **Visualization**: matplotlib, seaborn, plotly (for interactive graphs)
- **Geospatial**: geopy (for location analysis)
- **Web Server**: Flask, Flask-CORS, Flask-RESTful, Werkzeug
- **Jupyter**: jupyterlab, ipykernel
- **Development**: All supporting libraries for data analysis workflows