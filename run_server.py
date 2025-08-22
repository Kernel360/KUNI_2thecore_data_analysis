#!/usr/bin/env python3

import os
import sys
from app import app

if __name__ == '__main__':
    print("=== KUNI 2thecore Data Analysis Server ===")
    print("Starting Flask development server...")
    print("Server will be available at: http://localhost:5000")
    print("API endpoints:")
    print("  GET  /                - API 정보")
    print("  POST /api/data        - 데이터 쿼리 실행")
    print("  GET  /api/health      - 헬스 체크")
    print("=====================================")
    
    app.run(debug=True, host='0.0.0.0', port=5000)