"""
간소화된 선호도 분석 모듈
sklearn을 활용한 간단하고 효율적인 선호도 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import chi2_contingency
from typing import Dict, List, Any, Optional
import base64
import io
import logging

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class SimplePreferenceAnalyzer:
    """간소화된 선호도 분석 클래스"""
    
    def __init__(self):
        self.brand_colors = {'현대': '#1f77b4', '기아': '#ff7f0e', '제네시스': '#2ca02c'}
        self.season_names = {1: '봄', 2: '여름', 3: '가을', 4: '겨울'}
        
    def analyze_preferences(self, year: Optional[str] = None, period_type: str = 'month') -> Dict[str, Any]:
        """메인 분석 함수"""
        try:
            # 데이터 로드
            df = self._load_data(year)
            if df.empty:
                return {"success": False, "message": "분석할 데이터가 없습니다.", "visualizations": {}}
            
            # 시각화 생성
            visualizations = self._create_all_charts(df, period_type)
            
            return {
                "success": True,
                "message": f"{period_type} 선호도 분석이 완료되었습니다.",
                "visualizations": visualizations
            }
            
        except Exception as e:
            logger.error(f"분석 중 오류: {str(e)}")
            return {
                "success": False,
                "message": f"분석 중 오류가 발생했습니다: {str(e)}",
                "visualizations": {}
            }
    
    def _load_data(self, year: Optional[str]) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        from .data_loader import get_data_from_db
        
        query = """
        SELECT dl.start_time, dl.brand, dl.model, c.car_type
        FROM drivelog dl
        JOIN car c ON dl.car_id = c.car_id
        WHERE dl.start_time IS NOT NULL
        """
        
        if year:
            query += f" AND YEAR(dl.start_time) = {year}"
        
        df = get_data_from_db(query)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 기본 전처리
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['year'] = df['start_time'].dt.year
        df['month'] = df['start_time'].dt.month
        df['season'] = df['month'].map({12: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 
                                       6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
        
        return df.dropna()
    
    def _create_all_charts(self, df: pd.DataFrame, period_type: str) -> Dict[str, str]:
        """모든 차트 생성"""
        charts = {}
        
        try:
            charts['brand_period_heatmap'] = self._create_heatmap(df, period_type)
        except Exception as e:
            logger.error(f"히트맵 생성 오류: {e}")
            charts['brand_period_heatmap'] = ""
        
        try:
            charts['market_share_pie'] = self._create_pie_chart(df)
        except Exception as e:
            logger.error(f"파이차트 생성 오류: {e}")
            charts['market_share_pie'] = ""
        
        try:
            charts['brand_preference_line'] = self._create_line_chart(df, period_type)
        except Exception as e:
            logger.error(f"라인차트 생성 오류: {e}")
            charts['brand_preference_line'] = ""
        
        try:
            charts['seasonality_strength_bar'] = self._create_seasonality_chart(df)
        except Exception as e:
            logger.error(f"계절성 차트 생성 오류: {e}")
            charts['seasonality_strength_bar'] = ""
        
        try:
            charts['statistical_comparison'] = self._create_statistical_chart(df, period_type)
        except Exception as e:
            logger.error(f"통계 차트 생성 오류: {e}")
            charts['statistical_comparison'] = ""
        
        return charts
    
    def _create_heatmap(self, df: pd.DataFrame, period_type: str) -> str:
        """브랜드별 기간별 히트맵"""
        period_col = 'month' if period_type == 'month' else 'season'
        crosstab = pd.crosstab(df['brand'], df[period_col], normalize='columns')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(crosstab, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'브랜드별 {period_type} 선호도')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_pie_chart(self, df: pd.DataFrame) -> str:
        """브랜드별 시장 점유율 파이차트"""
        brand_counts = df['brand'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = [self.brand_colors.get(b, 'gray') for b in brand_counts.index]
        
        ax.pie(brand_counts.values, labels=brand_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('브랜드별 시장 점유율')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_line_chart(self, df: pd.DataFrame, period_type: str) -> str:
        """브랜드별 기간별 트렌드 라인차트"""
        period_col = 'month' if period_type == 'month' else 'season'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand]
            period_counts = brand_data.groupby(period_col).size()
            
            # 전체 기간 채우기
            full_range = range(1, 13) if period_type == 'month' else range(1, 5)
            period_counts = period_counts.reindex(full_range, fill_value=0)
            
            ax.plot(period_counts.index, period_counts.values, 
                   marker='o', label=brand, linewidth=2,
                   color=self.brand_colors.get(brand, 'gray'))
        
        ax.set_title(f'브랜드별 {period_type} 트렌드')
        ax.set_xlabel('기간')
        ax.set_ylabel('운행 건수')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_seasonality_chart(self, df: pd.DataFrame) -> str:
        """브랜드별 계절성 강도 차트"""
        seasonality_scores = {}
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand]
            seasonal_counts = brand_data.groupby('season').size()
            
            if len(seasonal_counts) > 1:
                cv = seasonal_counts.std() / seasonal_counts.mean()
                seasonality_scores[brand] = cv
        
        if not seasonality_scores:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, '계절성 데이터 부족', ha='center', va='center')
            return self._fig_to_base64(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        brands = list(seasonality_scores.keys())
        scores = list(seasonality_scores.values())
        colors = [self.brand_colors.get(b, 'gray') for b in brands]
        
        bars = ax.bar(brands, scores, color=colors, alpha=0.7)
        
        # 값 표시
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom')
        
        ax.set_title('브랜드별 계절성 강도')
        ax.set_ylabel('계절성 강도 (변동계수)')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_statistical_chart(self, df: pd.DataFrame, period_type: str) -> str:
        """통계적 유의성 차트"""
        period_col = 'season' if period_type == 'season' else 'month'
        crosstab = pd.crosstab(df['brand'], df[period_col])
        
        # 카이제곱 검정
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 관측값
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'관측값 (χ² = {chi2:.2f})')
        
        # 기댓값
        sns.heatmap(expected, annot=True, fmt='.1f', cmap='Reds', ax=ax2)
        ax2.set_title(f'기댓값 (p = {p_value:.4f})')
        
        # 유의성 표시
        significance = "통계적으로 유의함" if p_value < 0.05 else "통계적으로 유의하지 않음"
        fig.suptitle(f'브랜드-기간 연관성 분석: {significance}')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Figure를 base64로 변환"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return image_base64


# Flask API 엔드포인트
def create_simple_preference_api():
    """간소화된 API 생성"""
    from flask import request, jsonify
    
    analyzer = SimplePreferenceAnalyzer()
    
    def preference_by_period():
        """API 엔드포인트"""
        try:
            year = request.args.get('year', None)
            period_type = request.args.get('period_type', 'month')
            
            # 파라미터 검증
            if period_type not in ['month', 'season']:
                return jsonify({
                    "success": False,
                    "message": "period_type은 'month' 또는 'season' 중 하나여야 합니다.",
                    "visualizations": {}
                }), 400
            
            # 분석 실행
            result = analyzer.analyze_preferences(year, period_type)
            
            return jsonify(result), 200 if result['success'] else 400
            
        except Exception as e:
            logger.error(f"API 오류: {str(e)}")
            return jsonify({
                "success": False,
                "message": "서버 내부 오류가 발생했습니다.",
                "visualizations": {}
            }), 500
    
    return preference_by_period