"""
고도화된 계절성 분석 모듈
통계적으로 엄밀한 계절성 분석 및 선호도 패턴 분석 기능 제공
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class SeasonalityAnalyzer:
    """고도화된 계절성 분석 클래스"""
    
    def __init__(self):
        self.season_mapping = {
            12: 4, 1: 4, 2: 4,  # 겨울
            3: 1, 4: 1, 5: 1,   # 봄
            6: 2, 7: 2, 8: 2,   # 여름
            9: 3, 10: 3, 11: 3  # 가을
        }
        
        self.season_names = {1: '봄', 2: '여름', 3: '가을', 4: '겨울'}
        self.month_names = {
            1: '1월', 2: '2월', 3: '3월', 4: '4월', 5: '5월', 6: '6월',
            7: '7월', 8: '8월', 9: '9월', 10: '10월', 11: '11월', 12: '12월'
        }
    
    def analyze_brand_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        브랜드별 계절성 종합 분석
        
        Args:
            df: 운행 데이터프레임 (start_time, brand, model 등 포함)
            
        Returns:
            브랜드별 계절성 분석 결과
        """
        try:
            # 데이터 전처리
            df_clean = self._preprocess_temporal_data(df)
            
            if df_clean.empty:
                return self._empty_result("브랜드 계절성 분석")
            
            # 브랜드별 분석
            brand_results = {}
            overall_results = {}
            
            for brand in df_clean['brand'].unique():
                brand_data = df_clean[df_clean['brand'] == brand]
                brand_results[brand] = self._analyze_single_brand_seasonality(brand_data)
            
            # 전체 브랜드 비교 분석
            overall_results = self._analyze_overall_brand_patterns(df_clean)
            
            return {
                'brand_analysis': brand_results,
                'overall_patterns': overall_results,
                'methodology': self._get_methodology_info(),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"브랜드 계절성 분석 중 오류 발생: {str(e)}")
            raise
    
    def analyze_model_seasonality(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        """
        모델별 계절성 분석 (상위 N개 모델 대상)
        
        Args:
            df: 운행 데이터프레임
            top_n: 분석할 상위 모델 개수
            
        Returns:
            모델별 계절성 분석 결과
        """
        try:
            df_clean = self._preprocess_temporal_data(df)
            
            if df_clean.empty:
                return self._empty_result("모델 계절성 분석")
            
            # 상위 N개 모델 선별
            model_counts = df_clean['model'].value_counts()
            top_models = model_counts.head(top_n).index.tolist()
            
            model_results = {}
            for model in top_models:
                model_data = df_clean[df_clean['model'] == model]
                if len(model_data) >= 12:  # 최소 12개 레코드 필요
                    model_results[model] = self._analyze_single_model_seasonality(model_data)
            
            return {
                'model_analysis': model_results,
                'model_ranking': model_counts.head(top_n).to_dict(),
                'methodology': self._get_methodology_info(),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"모델 계절성 분석 중 오류 발생: {str(e)}")
            raise
    
    def calculate_preference_metrics(self, df: pd.DataFrame, period_type: str = 'month') -> Dict[str, Any]:
        """
        선호도 지표 계산 (월별/계절별)
        
        Args:
            df: 운행 데이터프레임
            period_type: 'month' 또는 'season'
            
        Returns:
            선호도 지표 분석 결과
        """
        try:
            df_clean = self._preprocess_temporal_data(df)
            
            if df_clean.empty:
                return self._empty_result("선호도 지표 분석")
            
            period_col = period_type
            
            # 기간별 집계
            if period_type == 'month':
                period_data = df_clean.groupby(['brand', 'month']).size().reset_index(name='count')
                period_totals = df_clean.groupby('month').size()
            else:  # season
                period_data = df_clean.groupby(['brand', 'season']).size().reset_index(name='count')
                period_totals = df_clean.groupby('season').size()
            
            # 시장 점유율 계산
            market_share = self._calculate_market_share(period_data, period_totals, period_type)
            
            # 선호도 순위 계산
            preference_ranking = self._calculate_preference_ranking(period_data, period_type)
            
            # 변동성 분석
            volatility_analysis = self._calculate_preference_volatility(period_data, period_type)
            
            return {
                'market_share': market_share,
                'preference_ranking': preference_ranking,
                'volatility_analysis': volatility_analysis,
                'period_totals': period_totals.to_dict(),
                'methodology': self._get_methodology_info()
            }
            
        except Exception as e:
            logger.error(f"선호도 지표 계산 중 오류 발생: {str(e)}")
            raise
    
    def _preprocess_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 데이터 전처리"""
        df_clean = df.dropna(subset=['start_time', 'brand']).copy()
        
        if df_clean.empty:
            return df_clean
        
        # 시간 변수 생성
        df_clean['start_time'] = pd.to_datetime(df_clean['start_time'])
        df_clean['year'] = df_clean['start_time'].dt.year
        df_clean['month'] = df_clean['start_time'].dt.month
        df_clean['season'] = df_clean['month'].map(self.season_mapping)
        df_clean['weekday'] = df_clean['start_time'].dt.weekday
        df_clean['hour'] = df_clean['start_time'].dt.hour
        
        return df_clean
    
    def _analyze_single_brand_seasonality(self, brand_data: pd.DataFrame) -> Dict[str, Any]:
        """개별 브랜드 계절성 분석"""
        brand_name = brand_data['brand'].iloc[0]
        
        # 월별/계절별 집계
        monthly_counts = brand_data.groupby('month').size()
        seasonal_counts = brand_data.groupby('season').size()
        
        # 계절성 강도 계산
        seasonality_metrics = self._calculate_seasonality_strength(monthly_counts, seasonal_counts)
        
        # 통계적 유의성 검정
        significance_tests = self._perform_seasonality_tests(monthly_counts, seasonal_counts)
        
        # 계절별 특성
        seasonal_characteristics = self._analyze_seasonal_characteristics(brand_data)
        
        return {
            'brand': brand_name,
            'total_records': len(brand_data),
            'monthly_distribution': monthly_counts.to_dict(),
            'seasonal_distribution': seasonal_counts.to_dict(),
            'seasonality_metrics': seasonality_metrics,
            'significance_tests': significance_tests,
            'seasonal_characteristics': seasonal_characteristics
        }
    
    def _analyze_single_model_seasonality(self, model_data: pd.DataFrame) -> Dict[str, Any]:
        """개별 모델 계절성 분석"""
        model_name = model_data['model'].iloc[0]
        brand_name = model_data['brand'].iloc[0]
        
        # 월별/계절별 집계
        monthly_counts = model_data.groupby('month').size()
        seasonal_counts = model_data.groupby('season').size()
        
        # 계절성 강도 계산
        seasonality_metrics = self._calculate_seasonality_strength(monthly_counts, seasonal_counts)
        
        # 통계적 유의성 검정
        significance_tests = self._perform_seasonality_tests(monthly_counts, seasonal_counts)
        
        return {
            'model': model_name,
            'brand': brand_name,
            'total_records': len(model_data),
            'monthly_distribution': monthly_counts.to_dict(),
            'seasonal_distribution': seasonal_counts.to_dict(),
            'seasonality_metrics': seasonality_metrics,
            'significance_tests': significance_tests
        }
    
    def _calculate_seasonality_strength(self, monthly_counts: pd.Series, seasonal_counts: pd.Series) -> Dict[str, float]:
        """다양한 계절성 강도 지표 계산"""
        metrics = {}
        
        # 1. 변동계수 (Coefficient of Variation)
        metrics['monthly_cv'] = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
        metrics['seasonal_cv'] = seasonal_counts.std() / seasonal_counts.mean() if seasonal_counts.mean() > 0 else 0
        
        # 2. 계절성 지수 (Seasonal Index)
        annual_mean = monthly_counts.mean()
        seasonal_indices = {}
        for season in [1, 2, 3, 4]:
            season_months = [m for m, s in self.season_mapping.items() if s == season]
            season_mean = monthly_counts.reindex(season_months).mean()
            seasonal_indices[season] = (season_mean / annual_mean * 100) if annual_mean > 0 else 100
        
        metrics['seasonal_indices'] = seasonal_indices
        
        # 3. 범위 기반 계절성 (Range-based seasonality)
        metrics['monthly_range_ratio'] = (monthly_counts.max() - monthly_counts.min()) / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
        metrics['seasonal_range_ratio'] = (seasonal_counts.max() - seasonal_counts.min()) / seasonal_counts.mean() if seasonal_counts.mean() > 0 else 0
        
        # 4. 엔트로피 기반 계절성
        monthly_probs = monthly_counts / monthly_counts.sum()
        monthly_entropy = -np.sum(monthly_probs * np.log2(monthly_probs + 1e-10))
        metrics['monthly_entropy'] = monthly_entropy
        metrics['seasonality_from_entropy'] = 1 - (monthly_entropy / np.log2(12))  # 정규화된 계절성
        
        return metrics
    
    def _perform_seasonality_tests(self, monthly_counts: pd.Series, seasonal_counts: pd.Series) -> Dict[str, Any]:
        """통계적 유의성 검정"""
        tests = {}
        
        # 1. 카이제곱 적합도 검정 (균등분포 대비)
        try:
            # 월별 균등분포 검정
            expected_monthly = np.full(12, monthly_counts.sum() / 12)
            observed_monthly = monthly_counts.reindex(range(1, 13), fill_value=0)
            chi2_monthly, p_monthly = stats.chisquare(observed_monthly, expected_monthly)
            
            tests['monthly_uniformity_test'] = {
                'chi2_statistic': float(chi2_monthly),
                'p_value': float(p_monthly),
                'significant': p_monthly < 0.05,
                'interpretation': 'significant_seasonality' if p_monthly < 0.05 else 'no_significant_seasonality'
            }
            
            # 계절별 균등분포 검정
            expected_seasonal = np.full(4, seasonal_counts.sum() / 4)
            observed_seasonal = seasonal_counts.reindex([1, 2, 3, 4], fill_value=0)
            chi2_seasonal, p_seasonal = stats.chisquare(observed_seasonal, expected_seasonal)
            
            tests['seasonal_uniformity_test'] = {
                'chi2_statistic': float(chi2_seasonal),
                'p_value': float(p_seasonal),
                'significant': p_seasonal < 0.05,
                'interpretation': 'significant_seasonality' if p_seasonal < 0.05 else 'no_significant_seasonality'
            }
            
        except Exception as e:
            logger.warning(f"통계 검정 중 오류: {str(e)}")
            tests['error'] = str(e)
        
        # 2. 콜모고로프-스미르노프 검정 (연속성 검정)
        try:
            uniform_cdf = np.arange(1, 13) / 12
            monthly_cdf = monthly_counts.cumsum() / monthly_counts.sum()
            ks_stat, ks_p = stats.kstest(monthly_cdf, uniform_cdf)
            
            tests['ks_uniformity_test'] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(ks_p),
                'significant': ks_p < 0.05
            }
            
        except Exception as e:
            logger.warning(f"KS 검정 중 오류: {str(e)}")
        
        return tests
    
    def _analyze_overall_brand_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """전체 브랜드 패턴 비교 분석"""
        # 브랜드별 계절별 교차표 생성
        contingency_table = pd.crosstab(df['brand'], df['season'])
        
        # 카이제곱 독립성 검정
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # 크래머의 V (연관성 강도)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        # 브랜드 간 계절성 차이
        brand_seasonality_comparison = self._compare_brand_seasonality(df)
        
        return {
            'contingency_table': contingency_table.to_dict(),
            'independence_test': {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05,
                'cramers_v': float(cramers_v),
                'effect_size': self._interpret_cramers_v(cramers_v)
            },
            'brand_seasonality_comparison': brand_seasonality_comparison
        }
    
    def _compare_brand_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """브랜드 간 계절성 차이 비교"""
        brand_seasonality = {}
        
        for brand in df['brand'].unique():
            brand_data = df[df['brand'] == brand]
            seasonal_counts = brand_data.groupby('season').size()
            
            # 계절성 강도 계산
            cv = seasonal_counts.std() / seasonal_counts.mean() if seasonal_counts.mean() > 0 else 0
            
            # 가장 선호되는 계절
            peak_season = seasonal_counts.idxmax()
            peak_ratio = seasonal_counts.max() / seasonal_counts.sum()
            
            brand_seasonality[brand] = {
                'seasonality_strength': float(cv),
                'peak_season': int(peak_season),
                'peak_season_name': self.season_names[peak_season],
                'peak_ratio': float(peak_ratio),
                'seasonal_distribution': seasonal_counts.to_dict()
            }
        
        return brand_seasonality
    
    def _calculate_market_share(self, period_data: pd.DataFrame, period_totals: pd.Series, period_type: str) -> Dict[str, Any]:
        """시장 점유율 계산"""
        market_share = {}
        
        for brand in period_data['brand'].unique():
            brand_data = period_data[period_data['brand'] == brand]
            
            brand_share = {}
            for _, row in brand_data.iterrows():
                period_val = row[period_type]
                period_name = self.month_names[period_val] if period_type == 'month' else self.season_names[period_val]
                
                total_for_period = period_totals[period_val]
                share = row['count'] / total_for_period if total_for_period > 0 else 0
                brand_share[period_name] = float(share)
            
            market_share[brand] = brand_share
        
        return market_share
    
    def _calculate_preference_ranking(self, period_data: pd.DataFrame, period_type: str) -> Dict[str, Any]:
        """선호도 순위 계산"""
        rankings = {}
        
        periods = period_data[period_type].unique()
        
        for period in periods:
            period_name = self.month_names[period] if period_type == 'month' else self.season_names[period]
            period_brands = period_data[period_data[period_type] == period].sort_values('count', ascending=False)
            
            ranking = []
            for i, (_, row) in enumerate(period_brands.iterrows(), 1):
                ranking.append({
                    'rank': i,
                    'brand': row['brand'],
                    'count': int(row['count']),
                    'rank_change': None  # 추후 시계열 분석에서 계산 가능
                })
            
            rankings[period_name] = ranking
        
        return rankings
    
    def _calculate_preference_volatility(self, period_data: pd.DataFrame, period_type: str) -> Dict[str, Any]:
        """선호도 변동성 분석"""
        volatility = {}
        
        for brand in period_data['brand'].unique():
            brand_data = period_data[period_data['brand'] == brand]
            
            # 기간별 점유율 계산
            brand_counts = brand_data.set_index(period_type)['count']
            
            # 변동성 지표
            cv = brand_counts.std() / brand_counts.mean() if brand_counts.mean() > 0 else 0
            volatility_range = (brand_counts.max() - brand_counts.min()) / brand_counts.mean() if brand_counts.mean() > 0 else 0
            
            volatility[brand] = {
                'coefficient_of_variation': float(cv),
                'range_ratio': float(volatility_range),
                'stability_score': float(1 / (1 + cv)),  # 높을수록 안정적
                'volatility_level': self._classify_volatility(cv)
            }
        
        return volatility
    
    def _classify_volatility(self, cv: float) -> str:
        """변동성 수준 분류"""
        if cv < 0.1:
            return 'very_stable'
        elif cv < 0.3:
            return 'stable'
        elif cv < 0.5:
            return 'moderate'
        elif cv < 0.8:
            return 'volatile'
        else:
            return 'very_volatile'
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """크래머의 V 해석"""
        if cramers_v < 0.1:
            return 'negligible'
        elif cramers_v < 0.3:
            return 'small'
        elif cramers_v < 0.5:
            return 'medium'
        else:
            return 'large'
    
    def _analyze_seasonal_characteristics(self, brand_data: pd.DataFrame) -> Dict[str, Any]:
        """계절별 특성 분석"""
        characteristics = {}
        
        for season in [1, 2, 3, 4]:
            season_data = brand_data[brand_data['season'] == season]
            
            if len(season_data) > 0:
                # 시간대별 분포
                hourly_dist = season_data['hour'].value_counts().sort_index()
                
                # 요일별 분포  
                weekday_dist = season_data['weekday'].value_counts().sort_index()
                
                characteristics[self.season_names[season]] = {
                    'total_trips': len(season_data),
                    'avg_trips_per_day': len(season_data) / 90 if len(season_data) > 0 else 0,  # 대략 계절당 90일
                    'peak_hour': int(hourly_dist.idxmax()) if len(hourly_dist) > 0 else None,
                    'peak_weekday': int(weekday_dist.idxmax()) if len(weekday_dist) > 0 else None,
                    'hourly_distribution': hourly_dist.head(5).to_dict(),
                    'weekday_distribution': weekday_dist.to_dict()
                }
        
        return characteristics
    
    def _empty_result(self, analysis_type: str) -> Dict[str, Any]:
        """빈 결과 반환"""
        return {
            'error': f'{analysis_type}에 충분한 데이터가 없습니다.',
            'data_available': False,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _get_methodology_info(self) -> Dict[str, Any]:
        """분석 방법론 정보"""
        return {
            'seasonality_metrics': [
                'coefficient_of_variation',
                'seasonal_indices',
                'range_ratio',
                'entropy_based_seasonality'
            ],
            'statistical_tests': [
                'chi_square_goodness_of_fit',
                'chi_square_independence',
                'kolmogorov_smirnov'
            ],
            'season_definition': {
                '봄': [3, 4, 5],
                '여름': [6, 7, 8], 
                '가을': [9, 10, 11],
                '겨울': [12, 1, 2]
            },
            'significance_level': 0.05,
            'effect_size_measure': 'cramers_v'
        }