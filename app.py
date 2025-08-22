from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
from flask_restful import Api, Resource
import json
from src.data_loader import get_data_from_db, get_db_connection
from src.simple_preference_analysis import create_simple_preference_api
from src.simple_trend_analysis import create_simple_trend_api
from src.services.daily_forecast import create_daily_forecast_api
from src.services.region_clustering import create_region_clustering_api

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
CORS(app)
api = Api(app)

class DataAnalysisAPI(Resource):
    def get(self):
        return {
            "message": "KUNI 2thecore Data Analysis API",
            "status": "running",
            "endpoints": {
                "/": "API 정보",
                "/api/data": "데이터 조회 (POST)",
                "/api/health": "헬스 체크",
                "/api/analysis/preference-by-period": "선호도 분석 (GET)",
                "/api/analysis/yearly-trend": "연도별 트렌드 분석 (GET)"
            }
        }

class DataQueryAPI(Resource):
    def post(self):
        try:
            data = request.get_json()
            if not data or 'query' not in data:
                return {"error": "쿼리가 필요합니다. 'query' 필드를 포함해주세요."}, 400
            
            query = data['query']
            result_df = get_data_from_db(query)
            
            if result_df is not None:
                result_dict = result_df.to_dict('records')
                return {
                    "success": True,
                    "data": result_dict,
                    "row_count": len(result_dict)
                }
            else:
                return {"error": "쿼리 실행 중 오류가 발생했습니다."}, 500
                
        except Exception as e:
            return {"error": f"요청 처리 중 오류가 발생했습니다: {str(e)}"}, 500

class HealthCheckAPI(Resource):
    def get(self):
        try:
            engine = get_db_connection()
            with engine.connect() as connection:
                connection.execute("SELECT 1")
            return {
                "status": "healthy",
                "database": "connected",
                "message": "시스템이 정상적으로 작동 중입니다."
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }, 503

# 간소화된 선호도 분석 API 클래스
class PreferenceAnalysisAPI(Resource):
    def get(self):
        return create_simple_preference_api()()

# 간소화된 트렌드 분석 API 클래스  
class TrendAnalysisAPI(Resource):
    def get(self):
        return create_simple_trend_api()()

class DailyForecastAPI(Resource):
    def get(self):
        return create_daily_forecast_api()()

class RegionClusteringAPI(Resource):
    def get(self):
        return create_region_clustering_api()()

api.add_resource(DataAnalysisAPI, '/')
api.add_resource(PreferenceAnalysisAPI, '/api/analysis/period')
api.add_resource(TrendAnalysisAPI, '/api/analysis/trend')
api.add_resource(DailyForecastAPI, '/api/forecast/daily')
api.add_resource(RegionClusteringAPI, '/api/clustering/regions')

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)