"""
데이터 품질 검증 모듈
렌트카 데이터의 품질을 검증하고 분석 신뢰성을 확보하는 기능 제공
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """데이터 품질 검증 클래스"""
    
    def __init__(self):
        self.quality_thresholds = {
            'missing_data_threshold': 0.05,  # 5% 이하 결측치 허용
            'duplicate_threshold': 0.01,     # 1% 이하 중복 허용
            'seasonal_balance_threshold': 0.5,  # 계절별 균형 기준
            'min_records_per_month': 10,     # 월별 최소 레코드 수
            'min_total_months': 6            # 최소 분석 기간 (개월)
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 데이터 품질 검증
        
        Args:
            df: 검증할 데이터프레임
            
        Returns:
            품질 검증 결과 딕셔너리
        """
        try:
            # 기본 데이터 검증
            basic_checks = self._basic_data_checks(df)
            
            # 시간적 품질 검증
            temporal_checks = self._temporal_quality_checks(df)
            
            # 비즈니스 로직 검증
            business_checks = self._business_logic_checks(df)
            
            # 전체 품질 점수 계산
            quality_score = self._calculate_quality_score(
                basic_checks, temporal_checks, business_checks
            )
            
            return {
                'quality_score': quality_score,
                'basic_checks': basic_checks,
                'temporal_checks': temporal_checks,
                'business_checks': business_checks,
                'recommendations': self._generate_recommendations(
                    basic_checks, temporal_checks, business_checks
                ),
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"데이터 품질 검증 중 오류 발생: {str(e)}")
            raise
    
    def _basic_data_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기본 데이터 품질 검사"""
        total_records = len(df)
        
        return {
            'total_records': total_records,
            'missing_start_time': df['start_time'].isnull().sum(),
            'missing_brand': df['brand'].isnull().sum(),
            'missing_model': df['model'].isnull().sum(),
            'duplicate_records': df.duplicated().sum(),
            'missing_data_ratio': df.isnull().sum().sum() / (total_records * len(df.columns)),
            'duplicate_ratio': df.duplicated().sum() / total_records if total_records > 0 else 0
        }
    
    def _temporal_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시간적 데이터 품질 검사"""
        if df.empty or df['start_time'].isnull().all():
            return {
                'temporal_coverage': None,
                'monthly_distribution': {},
                'seasonal_balance': None,
                'data_continuity': False
            }
        
        # 시간 데이터 처리
        df_temp = df.dropna(subset=['start_time']).copy()
        df_temp['start_time'] = pd.to_datetime(df_temp['start_time'])
        
        # 시간적 커버리지
        start_date = df_temp['start_time'].min()
        end_date = df_temp['start_time'].max()
        total_months = len(df_temp['start_time'].dt.to_period('M').unique())
        
        # 월별 분포
        monthly_dist = df_temp.groupby(df_temp['start_time'].dt.month).size()
        
        # 계절별 균형
        df_temp['season'] = df_temp['start_time'].dt.month.map({
            12: 4, 1: 4, 2: 4,  # 겨울
            3: 1, 4: 1, 5: 1,   # 봄
            6: 2, 7: 2, 8: 2,   # 여름
            9: 3, 10: 3, 11: 3  # 가을
        })
        
        seasonal_dist = df_temp.groupby('season').size()
        seasonal_balance = seasonal_dist.std() / seasonal_dist.mean() if seasonal_dist.mean() > 0 else float('inf')
        
        return {
            'temporal_coverage': {
                'start_date': start_date.isoformat() if pd.notna(start_date) else None,
                'end_date': end_date.isoformat() if pd.notna(end_date) else None,
                'total_months': total_months,
                'period_days': (end_date - start_date).days if pd.notna(start_date) and pd.notna(end_date) else 0
            },
            'monthly_distribution': monthly_dist.to_dict(),
            'seasonal_balance': seasonal_balance,
            'data_continuity': total_months >= self.quality_thresholds['min_total_months']
        }
    
    def _business_logic_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """비즈니스 로직 검증"""
        # 브랜드별 레코드 수
        brand_counts = df['brand'].value_counts()
        
        # 모델별 레코드 수
        model_counts = df['model'].value_counts()
        
        # 차량 ID별 레코드 수
        car_id_counts = df['car_id'].value_counts()
        
        return {
            'brand_distribution': brand_counts.to_dict(),
            'model_distribution': model_counts.head(10).to_dict(),  # 상위 10개만
            'unique_brands': len(brand_counts),
            'unique_models': len(model_counts),
            'unique_cars': len(car_id_counts),
            'avg_records_per_car': car_id_counts.mean() if len(car_id_counts) > 0 else 0,
            'records_balance': {
                'brand_balance': brand_counts.std() / brand_counts.mean() if brand_counts.mean() > 0 else float('inf'),
                'min_brand_records': brand_counts.min() if len(brand_counts) > 0 else 0,
                'max_brand_records': brand_counts.max() if len(brand_counts) > 0 else 0
            }
        }
    
    def _calculate_quality_score(self, basic: Dict, temporal: Dict, business: Dict) -> float:
        """전체 품질 점수 계산 (0-1 범위)"""
        score = 1.0
        
        # 기본 품질 점수 (40% 가중치)
        if basic['missing_data_ratio'] > self.quality_thresholds['missing_data_threshold']:
            score -= 0.4 * (basic['missing_data_ratio'] - self.quality_thresholds['missing_data_threshold'])
        
        if basic['duplicate_ratio'] > self.quality_thresholds['duplicate_threshold']:
            score -= 0.1 * (basic['duplicate_ratio'] - self.quality_thresholds['duplicate_threshold'])
        
        # 시간적 품질 점수 (30% 가중치)
        if temporal['temporal_coverage'] and temporal['temporal_coverage']['total_months'] < self.quality_thresholds['min_total_months']:
            score -= 0.3
        
        if temporal['seasonal_balance'] and temporal['seasonal_balance'] > self.quality_thresholds['seasonal_balance_threshold']:
            score -= 0.2
        
        # 비즈니스 품질 점수 (30% 가중치)
        if business['unique_brands'] < 2:  # 최소 2개 브랜드 필요
            score -= 0.2
        
        if business['avg_records_per_car'] < 5:  # 차량당 최소 5개 레코드
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, basic: Dict, temporal: Dict, business: Dict) -> List[str]:
        """데이터 품질 개선 권장사항 생성"""
        recommendations = []
        
        if basic['missing_data_ratio'] > self.quality_thresholds['missing_data_threshold']:
            recommendations.append(f"결측치 비율이 {basic['missing_data_ratio']:.2%}로 높습니다. 데이터 수집 과정을 점검하세요.")
        
        if basic['duplicate_ratio'] > self.quality_thresholds['duplicate_threshold']:
            recommendations.append(f"중복 레코드가 {basic['duplicate_ratio']:.2%} 발견되었습니다. 데이터 정제가 필요합니다.")
        
        if temporal['temporal_coverage'] and temporal['temporal_coverage']['total_months'] < self.quality_thresholds['min_total_months']:
            recommendations.append(f"분석 기간이 {temporal['temporal_coverage']['total_months']}개월로 짧습니다. 최소 {self.quality_thresholds['min_total_months']}개월 이상의 데이터가 필요합니다.")
        
        if temporal['seasonal_balance'] and temporal['seasonal_balance'] > self.quality_thresholds['seasonal_balance_threshold']:
            recommendations.append("계절별 데이터 불균형이 심합니다. 계절성 분석 결과 해석 시 주의가 필요합니다.")
        
        if business['unique_brands'] < 3:
            recommendations.append("브랜드 다양성이 부족합니다. 브랜드별 비교 분석에 제한이 있을 수 있습니다.")
        
        if not recommendations:
            recommendations.append("데이터 품질이 양호합니다. 신뢰할 수 있는 분석 결과를 기대할 수 있습니다.")
        
        return recommendations


def validate_input_parameters(year: str = None, period_type: str = 'month') -> Dict[str, Any]:
    """
    API 입력 파라미터 검증
    
    Args:
        year: 분석 연도 (선택적)
        period_type: 기간 타입 ('month' 또는 'season')
        
    Returns:
        검증 결과 및 정제된 파라미터
    """
    errors = []
    warnings = []
    
    # period_type 검증
    valid_period_types = ['month', 'season']
    if period_type not in valid_period_types:
        errors.append(f"period_type은 {valid_period_types} 중 하나여야 합니다.")
        period_type = 'month'  # 기본값으로 설정
    
    # year 검증
    current_year = datetime.now().year
    if year:
        try:
            year_int = int(year)
            if year_int < 2020 or year_int > current_year + 1:
                warnings.append(f"연도 {year_int}는 분석 범위({2020}-{current_year+1})를 벗어납니다.")
        except ValueError:
            errors.append("연도는 숫자여야 합니다.")
            year = None
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'cleaned_params': {
            'year': year,
            'period_type': period_type
        }
    }