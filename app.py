from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import json
from src.data_loader import get_data_from_db, get_db_connection

app = Flask(__name__)
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
                "/api/health": "헬스 체크"
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

api.add_resource(DataAnalysisAPI, '/')
api.add_resource(DataQueryAPI, '/api/data')
api.add_resource(HealthCheckAPI, '/api/health')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)