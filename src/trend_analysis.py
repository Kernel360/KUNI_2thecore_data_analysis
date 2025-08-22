"""
연도별 트렌드 분석 모듈
브랜드와 차량 모델의 시간적 변화 패턴과 트렌드를 분석하는 기능 제공
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import linregress, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """연도별 트렌드 분석 클래스"""
    
    def __init__(self):
        self.trend_methods = {
            'linear': self._linear_trend,
            'polynomial': self._polynomial_trend,
            'exponential': self._exponential_trend
        }
        
        self.trend_strength_thresholds = {
            'strong_growth': 0.15,      # 15% 이상 성장
            'moderate_growth': 0.05,    # 5% 이상 성장
            'stable': 0.05,             # -5% ~ 5% 변화
            'moderate_decline': -0.15,  # -15% 이상 감소
            'strong_decline': float('-inf')  # -15% 미만 감소
        }
    
    def analyze_brand_trends(self, df: pd.DataFrame, start_year: Optional[int] = None,
                           end_year: Optional[int] = None) -> Dict[str, Any]:
        """
        브랜드별 연도별 트렌드 분석
        
        Args:
            df: 분석할 데이터프레임
            start_year: 분석 시작 연도 (None이면 데이터 최소 연도)
            end_year: 분석 종료 연도 (None이면 데이터 최대 연도)
            
        Returns:
            브랜드별 트렌드 분석 결과
        """
        try:
            # 데이터 전처리
            df_clean = self._preprocess_trend_data(df, start_year, end_year)
            
            if df_clean.empty:
                return self._empty_result("브랜드 트렌드 분석")
            
            results = {}
            
            # 전체 트렌드 분석
            results['overall_trends'] = self._analyze_overall_market_trends(df_clean)
            
            # 브랜드별 상세 트렌드
            results['brand_details'] = {}
            for brand in df_clean['brand'].unique():
                brand_data = df_clean[df_clean['brand'] == brand]
                results['brand_details'][brand] = self._analyze_single_brand_trend(brand_data, brand)
            
            # 브랜드 간 비교 분석
            results['brand_comparison'] = self._compare_brand_trends(df_clean)
            
            # 시장 점유율 변화 분석
            results['market_share_evolution'] = self._analyze_market_share_evolution(df_clean)
            
            # 트렌드 예측
            results['trend_predictions'] = self._predict_future_trends(df_clean)
            
            return {
                'analysis_type': '브랜드 트렌드 분석',
                'analysis_period': {
                    'start_year': int(df_clean['year'].min()),
                    'end_year': int(df_clean['year'].max()),
                    'total_years': int(df_clean['year'].nunique())
                },
                'total_records': len(df_clean),
                'results': results,
                'methodology': self._get_trend_methodology(),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"브랜드 트렌드 분석 중 오류 발생: {str(e)}")
            raise
    
    def analyze_model_trends(self, df: pd.DataFrame, top_n: int = 15,
                           start_year: Optional[int] = None,
                           end_year: Optional[int] = None) -> Dict[str, Any]:
        """
        모델별 연도별 트렌드 분석
        
        Args:
            df: 분석할 데이터프레임
            top_n: 분석할 상위 모델 개수
            start_year: 분석 시작 연도
            end_year: 분석 종료 연도
            
        Returns:
            모델별 트렌드 분석 결과
        """
        try:
            df_clean = self._preprocess_trend_data(df, start_year, end_year)
            
            if df_clean.empty:
                return self._empty_result("모델 트렌드 분석")
            
            # 상위 모델 선별
            model_counts = df_clean['model'].value_counts()
            top_models = model_counts.head(top_n).index.tolist()
            
            results = {}
            
            # 모델별 상세 분석
            results['model_details'] = {}
            for model in top_models:
                model_data = df_clean[df_clean['model'] == model]
                if len(model_data) >= 5:  # 최소 5개 레코드 필요
                    results['model_details'][model] = self._analyze_single_model_trend(model_data, model)
            
            # 모델 라이프사이클 분석
            results['model_lifecycle'] = self._analyze_model_lifecycle(df_clean, top_models)
            
            # 신규/퇴출 모델 분석
            results['model_introduction_retirement'] = self._analyze_model_introduction_retirement(df_clean)
            
            # 브랜드별 모델 포트폴리오 변화
            results['brand_portfolio_evolution'] = self._analyze_brand_portfolio_evolution(df_clean)
            
            return {
                'analysis_type': '모델 트렌드 분석',
                'analysis_period': {
                    'start_year': int(df_clean['year'].min()),
                    'end_year': int(df_clean['year'].max()),
                    'total_years': int(df_clean['year'].nunique())
                },
                'top_models_analyzed': top_n,
                'total_records': len(df_clean),
                'results': results,
                'methodology': self._get_trend_methodology(),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"모델 트렌드 분석 중 오류 발생: {str(e)}")
            raise
    
    def analyze_preference_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        선호도 진화 분석 (계절성과 트렌드의 결합)
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            선호도 진화 분석 결과
        """
        try:
            df_clean = self._preprocess_trend_data(df)
            
            if df_clean.empty:
                return self._empty_result("선호도 진화 분석")
            
            results = {}
            
            # 계절성 변화 트렌드
            results['seasonality_evolution'] = self._analyze_seasonality_evolution(df_clean)
            
            # 선호도 변동성 변화
            results['preference_volatility_trends'] = self._analyze_preference_volatility_trends(df_clean)
            
            # 차량 유형별 트렌드
            results['vehicle_type_trends'] = self._analyze_vehicle_type_trends(df_clean)
            
            # 연식별 선호도 변화
            results['vehicle_age_preference_trends'] = self._analyze_vehicle_age_trends(df_clean)
            
            return {
                'analysis_type': '선호도 진화 분석',
                'analysis_period': {
                    'start_year': int(df_clean['year'].min()),
                    'end_year': int(df_clean['year'].max()),
                    'total_years': int(df_clean['year'].nunique())
                },
                'total_records': len(df_clean),
                'results': results,
                'methodology': self._get_trend_methodology(),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"선호도 진화 분석 중 오류 발생: {str(e)}")
            raise
    
    def _preprocess_trend_data(self, df: pd.DataFrame, start_year: Optional[int] = None,
                             end_year: Optional[int] = None) -> pd.DataFrame:
        """트렌드 분석용 데이터 전처리"""
        df_clean = df.dropna(subset=['start_time', 'brand']).copy()
        
        if df_clean.empty:
            return df_clean
        
        # 시간 변수 생성
        df_clean['start_time'] = pd.to_datetime(df_clean['start_time'])
        df_clean['year'] = df_clean['start_time'].dt.year
        df_clean['month'] = df_clean['start_time'].dt.month
        df_clean['season'] = df_clean['month'].map({
            12: 4, 1: 4, 2: 4,  # 겨울
            3: 1, 4: 1, 5: 1,   # 봄
            6: 2, 7: 2, 8: 2,   # 여름
            9: 3, 10: 3, 11: 3  # 가을
        })
        
        # 연도 필터링
        if start_year:
            df_clean = df_clean[df_clean['year'] >= start_year]
        if end_year:
            df_clean = df_clean[df_clean['year'] <= end_year]
        
        return df_clean
    
    def _analyze_overall_market_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """전체 시장 트렌드 분석"""
        # 연도별 총 운행 건수
        yearly_totals = df.groupby('year').size()
        
        # 선형 트렌드 분석
        years = yearly_totals.index.values
        counts = yearly_totals.values
        
        if len(years) < 2:
            return {'error': '트렌드 분석을 위한 연도가 부족합니다'}
        
        # 다양한 트렌드 모델 적용
        trend_results = {}
        for method_name, method_func in self.trend_methods.items():
            try:
                trend_results[method_name] = method_func(years, counts)
            except Exception as e:
                trend_results[method_name] = {'error': str(e)}
        
        # 최적 모델 선택
        best_model = self._select_best_trend_model(trend_results)
        
        # 성장률 계산
        if len(counts) >= 2:
            total_growth_rate = (counts[-1] - counts[0]) / counts[0]
            cagr = ((counts[-1] / counts[0]) ** (1 / (len(counts) - 1))) - 1
        else:
            total_growth_rate = 0
            cagr = 0
        
        return {
            'yearly_totals': yearly_totals.to_dict(),
            'trend_models': trend_results,
            'best_model': best_model,
            'growth_metrics': {
                'total_growth_rate': float(total_growth_rate),
                'cagr': float(cagr),
                'trend_classification': self._classify_trend(cagr)
            }
        }
    
    def _analyze_single_brand_trend(self, brand_data: pd.DataFrame, brand: str) -> Dict[str, Any]:
        """개별 브랜드 트렌드 분석"""
        yearly_counts = brand_data.groupby('year').size()
        
        if len(yearly_counts) < 2:
            return {'brand': brand, 'error': '트렌드 분석을 위한 연도가 부족합니다'}
        
        years = yearly_counts.index.values
        counts = yearly_counts.values
        
        # 트렌드 분석
        trend_results = {}
        for method_name, method_func in self.trend_methods.items():
            try:
                trend_results[method_name] = method_func(years, counts)
            except Exception as e:
                trend_results[method_name] = {'error': str(e)}
        
        # 성장률 및 변동성 계산
        total_growth_rate = (counts[-1] - counts[0]) / counts[0] if counts[0] > 0 else 0
        cagr = ((counts[-1] / counts[0]) ** (1 / (len(counts) - 1))) - 1 if counts[0] > 0 and len(counts) > 1 else 0
        volatility = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        
        # 추세 방향성 검정
        trend_tests = self._perform_trend_tests(years, counts)
        
        # 변곡점 탐지
        change_points = self._detect_change_points(yearly_counts)
        
        return {
            'brand': brand,
            'yearly_counts': yearly_counts.to_dict(),
            'trend_analysis': trend_results,
            'growth_metrics': {
                'total_growth_rate': float(total_growth_rate),
                'cagr': float(cagr),
                'volatility': float(volatility),
                'trend_classification': self._classify_trend(cagr)
            },
            'statistical_tests': trend_tests,
            'change_points': change_points
        }
    
    def _analyze_single_model_trend(self, model_data: pd.DataFrame, model: str) -> Dict[str, Any]:
        """개별 모델 트렌드 분석"""
        yearly_counts = model_data.groupby('year').size()
        brand = model_data['brand'].iloc[0]
        
        if len(yearly_counts) < 2:
            return {'model': model, 'brand': brand, 'error': '트렌드 분석을 위한 연도가 부족합니다'}
        
        years = yearly_counts.index.values
        counts = yearly_counts.values
        
        # 기본 트렌드 분석
        slope, intercept, r_value, p_value, std_err = linregress(years, counts)
        
        # 모델 생명주기 단계 판단
        lifecycle_stage = self._determine_lifecycle_stage(yearly_counts)
        
        # 시장 점유율 변화
        model_market_share_trend = self._calculate_model_market_share_trend(model_data)
        
        return {
            'model': model,
            'brand': brand,
            'yearly_counts': yearly_counts.to_dict(),
            'trend_metrics': {
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'lifecycle_stage': lifecycle_stage,
            'market_share_trend': model_market_share_trend
        }
    
    def _compare_brand_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """브랜드 간 트렌드 비교"""
        brand_growth_rates = {}
        brand_volatilities = {}
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand]
            yearly_counts = brand_data.groupby('year').size()
            
            if len(yearly_counts) >= 2:
                counts = yearly_counts.values
                cagr = ((counts[-1] / counts[0]) ** (1 / (len(counts) - 1))) - 1 if counts[0] > 0 else 0
                volatility = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
                
                brand_growth_rates[brand] = float(cagr)
                brand_volatilities[brand] = float(volatility)
        
        # 브랜드 순위
        growth_ranking = sorted(brand_growth_rates.items(), key=lambda x: x[1], reverse=True)
        stability_ranking = sorted(brand_volatilities.items(), key=lambda x: x[1])  # 낮은 변동성이 안정적
        
        return {
            'growth_rates': brand_growth_rates,
            'volatilities': brand_volatilities,
            'growth_ranking': growth_ranking,
            'stability_ranking': stability_ranking,
            'performance_matrix': self._create_performance_matrix(brand_growth_rates, brand_volatilities)
        }
    
    def _analyze_market_share_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시장 점유율 진화 분석"""
        yearly_brand_shares = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            brand_counts = year_data['brand'].value_counts()
            total_count = len(year_data)
            
            year_shares = {}
            for brand, count in brand_counts.items():
                year_shares[brand] = count / total_count
            
            yearly_brand_shares[str(year)] = year_shares
        
        # 점유율 트렌드 분석
        share_trends = {}
        for brand in df['brand'].unique():
            brand_shares = []
            years = []
            
            for year_str, shares in yearly_brand_shares.items():
                if brand in shares:
                    brand_shares.append(shares[brand])
                    years.append(int(year_str))
                else:
                    brand_shares.append(0)
                    years.append(int(year_str))
            
            if len(brand_shares) >= 2:
                slope, _, r_value, p_value, _ = linregress(years, brand_shares)
                share_trends[brand] = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
        
        return {
            'yearly_market_shares': yearly_brand_shares,
            'market_share_trends': share_trends
        }
    
    def _predict_future_trends(self, df: pd.DataFrame, forecast_years: int = 2) -> Dict[str, Any]:
        """미래 트렌드 예측"""
        predictions = {}
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand]
            yearly_counts = brand_data.groupby('year').size()
            
            if len(yearly_counts) >= 3:  # 최소 3년 데이터 필요
                years = yearly_counts.index.values.reshape(-1, 1)
                counts = yearly_counts.values
                
                # 선형 회귀 모델
                model = LinearRegression()
                model.fit(years, counts)
                
                # 미래 연도 예측
                future_years = np.array(range(int(yearly_counts.index.max() + 1), 
                                            int(yearly_counts.index.max() + forecast_years + 1))).reshape(-1, 1)
                future_predictions = model.predict(future_years)
                
                predictions[brand] = {
                    'model_score': float(model.score(years, counts)),
                    'predictions': {
                        str(int(year[0])): max(0, float(pred))  # 음수 방지
                        for year, pred in zip(future_years, future_predictions)
                    }
                }
        
        return predictions
    
    def _analyze_seasonality_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """계절성 변화 분석"""
        seasonality_by_year = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            
            brand_seasonality = {}
            for brand in year_data['brand'].unique():
                brand_year_data = year_data[year_data['brand'] == brand]
                seasonal_counts = brand_year_data.groupby('season').size()
                
                if len(seasonal_counts) > 1:
                    cv = seasonal_counts.std() / seasonal_counts.mean()
                    brand_seasonality[brand] = float(cv)
            
            seasonality_by_year[str(year)] = brand_seasonality
        
        # 계절성 트렌드 계산
        seasonality_trends = {}
        for brand in df['brand'].unique():
            brand_seasonality_values = []
            years = []
            
            for year_str, year_seasonality in seasonality_by_year.items():
                if brand in year_seasonality:
                    brand_seasonality_values.append(year_seasonality[brand])
                    years.append(int(year_str))
            
            if len(brand_seasonality_values) >= 2:
                slope, _, r_value, p_value, _ = linregress(years, brand_seasonality_values)
                seasonality_trends[brand] = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'interpretation': 'increasing_seasonality' if slope > 0 else 'decreasing_seasonality' if slope < 0 else 'stable_seasonality'
                }
        
        return {
            'yearly_seasonality': seasonality_by_year,
            'seasonality_trends': seasonality_trends
        }
    
    def _linear_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """선형 트렌드 분석"""
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        return {
            'model_type': 'linear',
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'significant': p_value < 0.05
        }
    
    def _polynomial_trend(self, x: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict[str, Any]:
        """다항식 트렌드 분석"""
        try:
            poly_features = PolynomialFeatures(degree=degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(x_poly, y)
            
            y_pred = model.predict(x_poly)
            r2 = r2_score(y, y_pred)
            
            return {
                'model_type': f'polynomial_degree_{degree}',
                'r_squared': float(r2),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            }
        except Exception as e:
            return {'model_type': f'polynomial_degree_{degree}', 'error': str(e)}
    
    def _exponential_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """지수 트렌드 분석"""
        try:
            # 로그 변환 후 선형 회귀
            y_positive = np.maximum(y, 1e-10)  # 0 방지
            log_y = np.log(y_positive)
            
            slope, intercept, r_value, p_value, std_err = linregress(x, log_y)
            
            return {
                'model_type': 'exponential',
                'growth_rate': float(np.exp(slope) - 1),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except Exception as e:
            return {'model_type': 'exponential', 'error': str(e)}
    
    def _select_best_trend_model(self, trend_results: Dict[str, Any]) -> str:
        """최적 트렌드 모델 선택"""
        valid_models = {k: v for k, v in trend_results.items() if 'error' not in v}
        
        if not valid_models:
            return 'none'
        
        # R² 값이 가장 높은 모델 선택
        best_model = max(valid_models.items(), key=lambda x: x[1].get('r_squared', 0))
        return best_model[0]
    
    def _classify_trend(self, cagr: float) -> str:
        """트렌드 분류"""
        if cagr >= self.trend_strength_thresholds['strong_growth']:
            return 'strong_growth'
        elif cagr >= self.trend_strength_thresholds['moderate_growth']:
            return 'moderate_growth'
        elif cagr >= self.trend_strength_thresholds['moderate_decline']:
            return 'stable'
        elif cagr >= self.trend_strength_thresholds['strong_decline']:
            return 'moderate_decline'
        else:
            return 'strong_decline'
    
    def _perform_trend_tests(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """트렌드 통계 검정"""
        tests = {}
        
        # Mann-Kendall 트렌드 검정
        try:
            tau, p_value = kendalltau(x, y)
            tests['mann_kendall'] = {
                'tau': float(tau),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'trend': 'increasing' if tau > 0 else 'decreasing' if tau < 0 else 'no_trend'
            }
        except Exception as e:
            tests['mann_kendall'] = {'error': str(e)}
        
        # Spearman 상관계수
        try:
            rho, p_value = spearmanr(x, y)
            tests['spearman'] = {
                'rho': float(rho),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except Exception as e:
            tests['spearman'] = {'error': str(e)}
        
        return tests
    
    def _detect_change_points(self, time_series: pd.Series) -> List[Dict[str, Any]]:
        """변곡점 탐지"""
        change_points = []
        
        if len(time_series) < 4:
            return change_points
        
        # 간단한 변곡점 탐지 (2차 차분)
        values = time_series.values
        second_diff = np.diff(values, n=2)
        
        # 임계값을 넘는 변화점 탐지
        threshold = np.std(second_diff) * 2
        
        for i, diff in enumerate(second_diff):
            if abs(diff) > threshold:
                change_points.append({
                    'year': int(time_series.index[i + 2]),
                    'change_magnitude': float(diff),
                    'change_type': 'acceleration' if diff > 0 else 'deceleration'
                })
        
        return change_points
    
    def _determine_lifecycle_stage(self, time_series: pd.Series) -> str:
        """모델 생명주기 단계 판단"""
        if len(time_series) < 3:
            return 'insufficient_data'
        
        values = time_series.values
        first_half_mean = np.mean(values[:len(values)//2])
        second_half_mean = np.mean(values[len(values)//2:])
        
        growth_rate = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        
        if growth_rate > 0.2:
            return 'growth'
        elif growth_rate > -0.2:
            return 'maturity'
        else:
            return 'decline'
    
    def _calculate_model_market_share_trend(self, model_data: pd.DataFrame) -> Dict[str, Any]:
        """모델 시장 점유율 트렌드 계산"""
        # 간단한 구현 - 실제로는 전체 시장 데이터와 비교 필요
        yearly_counts = model_data.groupby('year').size()
        
        if len(yearly_counts) >= 2:
            slope, _, r_value, p_value, _ = linregress(yearly_counts.index.values, yearly_counts.values)
            return {
                'trend_slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            }
        else:
            return {'error': 'insufficient_data'}
    
    def _create_performance_matrix(self, growth_rates: Dict[str, float], 
                                 volatilities: Dict[str, float]) -> Dict[str, str]:
        """성과 매트릭스 생성 (성장률 vs 안정성)"""
        performance_matrix = {}
        
        growth_median = np.median(list(growth_rates.values()))
        volatility_median = np.median(list(volatilities.values()))
        
        for brand in growth_rates.keys():
            growth = growth_rates[brand]
            volatility = volatilities[brand]
            
            if growth >= growth_median and volatility <= volatility_median:
                category = 'star'  # 높은 성장, 낮은 변동성
            elif growth >= growth_median and volatility > volatility_median:
                category = 'question_mark'  # 높은 성장, 높은 변동성
            elif growth < growth_median and volatility <= volatility_median:
                category = 'cash_cow'  # 낮은 성장, 낮은 변동성
            else:
                category = 'dog'  # 낮은 성장, 높은 변동성
            
            performance_matrix[brand] = category
        
        return performance_matrix
    
    def _analyze_model_lifecycle(self, df: pd.DataFrame, models: List[str]) -> Dict[str, Any]:
        """모델 라이프사이클 분석"""
        lifecycle_analysis = {}
        
        for model in models:
            model_data = df[df['model'] == model]
            yearly_counts = model_data.groupby('year').size()
            
            if len(yearly_counts) >= 2:
                lifecycle_stage = self._determine_lifecycle_stage(yearly_counts)
                
                lifecycle_analysis[model] = {
                    'stage': lifecycle_stage,
                    'first_appearance': int(yearly_counts.index.min()),
                    'peak_year': int(yearly_counts.idxmax()),
                    'peak_count': int(yearly_counts.max()),
                    'latest_count': int(yearly_counts.iloc[-1])
                }
        
        return lifecycle_analysis
    
    def _analyze_model_introduction_retirement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """신규/퇴출 모델 분석"""
        all_years = sorted(df['year'].unique())
        
        if len(all_years) < 2:
            return {'error': '연도 데이터 부족'}
        
        # 연도별 모델 목록
        yearly_models = {}
        for year in all_years:
            yearly_models[year] = set(df[df['year'] == year]['model'].unique())
        
        # 신규 진입 모델
        new_models_by_year = {}
        for i, year in enumerate(all_years[1:], 1):
            prev_year = all_years[i-1]
            new_models = yearly_models[year] - yearly_models[prev_year]
            new_models_by_year[str(year)] = list(new_models)
        
        # 퇴출 모델
        retired_models_by_year = {}
        for i, year in enumerate(all_years[1:], 1):
            prev_year = all_years[i-1]
            retired_models = yearly_models[prev_year] - yearly_models[year]
            retired_models_by_year[str(year)] = list(retired_models)
        
        return {
            'new_models_by_year': new_models_by_year,
            'retired_models_by_year': retired_models_by_year,
            'model_turnover_rate': self._calculate_model_turnover_rate(yearly_models)
        }
    
    def _analyze_brand_portfolio_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """브랜드별 모델 포트폴리오 변화"""
        portfolio_evolution = {}
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand]
            yearly_model_counts = brand_data.groupby(['year', 'model']).size().unstack(fill_value=0)
            
            portfolio_evolution[brand] = {
                'model_diversity_by_year': yearly_model_counts.apply(lambda x: (x > 0).sum(), axis=1).to_dict(),
                'total_models': len(brand_data['model'].unique())
            }
        
        return portfolio_evolution
    
    def _analyze_preference_volatility_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """선호도 변동성 트렌드"""
        volatility_by_year = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            brand_counts = year_data['brand'].value_counts()
            
            if len(brand_counts) > 1:
                volatility = brand_counts.std() / brand_counts.mean()
                volatility_by_year[str(year)] = float(volatility)
        
        # 변동성 트렌드
        if len(volatility_by_year) >= 2:
            years = [int(y) for y in volatility_by_year.keys()]
            volatilities = list(volatility_by_year.values())
            
            slope, _, r_value, p_value, _ = linregress(years, volatilities)
            
            return {
                'yearly_volatility': volatility_by_year,
                'volatility_trend': {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'interpretation': 'increasing_volatility' if slope > 0 else 'decreasing_volatility'
                }
            }
        
        return {'yearly_volatility': volatility_by_year}
    
    def _analyze_vehicle_type_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """차량 유형별 트렌드"""
        if 'car_type' not in df.columns:
            return {'error': 'car_type 컬럼이 없습니다'}
        
        type_trends = {}
        
        for car_type in df['car_type'].unique():
            type_data = df[df['car_type'] == car_type]
            yearly_counts = type_data.groupby('year').size()
            
            if len(yearly_counts) >= 2:
                years = yearly_counts.index.values
                counts = yearly_counts.values
                
                slope, _, r_value, p_value, _ = linregress(years, counts)
                cagr = ((counts[-1] / counts[0]) ** (1 / (len(counts) - 1))) - 1 if counts[0] > 0 else 0
                
                type_trends[car_type] = {
                    'yearly_counts': yearly_counts.to_dict(),
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'cagr': float(cagr),
                    'trend_classification': self._classify_trend(cagr)
                }
        
        return type_trends
    
    def _analyze_vehicle_age_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """차량 연식별 선호도 트렌드"""
        if 'car_year' not in df.columns:
            return {'error': 'car_year 컬럼이 없습니다'}
        
        # 현재 연도 기준 차량 나이 계산
        current_year = df['year'].max()
        df = df.copy()
        df['vehicle_age'] = current_year - df['car_year']
        
        # 연령대별 분류
        df['age_category'] = pd.cut(df['vehicle_age'], 
                                  bins=[-float('inf'), 2, 5, 10, float('inf')],
                                  labels=['신차(0-2년)', '준신차(3-5년)', '중고차(6-10년)', '노후차(11년+)'])
        
        age_trends = {}
        for age_cat in df['age_category'].unique():
            if pd.isna(age_cat):
                continue
                
            age_data = df[df['age_category'] == age_cat]
            yearly_counts = age_data.groupby('year').size()
            
            if len(yearly_counts) >= 2:
                years = yearly_counts.index.values
                counts = yearly_counts.values
                
                slope, _, r_value, p_value, _ = linregress(years, counts)
                
                age_trends[str(age_cat)] = {
                    'yearly_counts': yearly_counts.to_dict(),
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing'
                }
        
        return age_trends
    
    def _calculate_model_turnover_rate(self, yearly_models: Dict[int, set]) -> Dict[str, float]:
        """모델 교체율 계산"""
        years = sorted(yearly_models.keys())
        turnover_rates = {}
        
        for i in range(1, len(years)):
            current_year = years[i]
            prev_year = years[i-1]
            
            prev_models = yearly_models[prev_year]
            current_models = yearly_models[current_year]
            
            if len(prev_models) > 0:
                # 신규 모델 비율
                new_models = current_models - prev_models
                turnover_rate = len(new_models) / len(prev_models)
                turnover_rates[str(current_year)] = float(turnover_rate)
        
        return turnover_rates
    
    def _empty_result(self, analysis_type: str) -> Dict[str, Any]:
        """빈 결과 반환"""
        return {
            'error': f'{analysis_type}에 충분한 데이터가 없습니다.',
            'data_available': False,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _get_trend_methodology(self) -> Dict[str, Any]:
        """분석 방법론 정보"""
        return {
            'trend_models': ['linear', 'polynomial', 'exponential'],
            'statistical_tests': ['mann_kendall', 'spearman_correlation', 'linear_regression'],
            'metrics': ['cagr', 'total_growth_rate', 'volatility', 'r_squared'],
            'trend_classifications': list(self.trend_strength_thresholds.keys()),
            'lifecycle_stages': ['growth', 'maturity', 'decline'],
            'change_point_detection': 'second_difference_method'
        }