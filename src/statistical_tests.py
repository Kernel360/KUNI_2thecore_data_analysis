"""
통계적 유의성 검정 모듈
데이터 분석의 신뢰성을 보장하기 위한 다양한 통계 검정 기능 제공
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, shapiro, levene
import warnings
warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger(__name__)


class StatisticalTestSuite:
    """통계적 검정 도구 모음"""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.test_results_template = {
            'test_name': '',
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'effect_size': None,
            'interpretation': '',
            'assumptions_met': True,
            'warnings': []
        }
    
    def comprehensive_preference_test(self, df: pd.DataFrame, groupby_col: str = 'brand', 
                                    period_col: str = 'season') -> Dict[str, Any]:
        """
        선호도 분석을 위한 종합적 통계 검정
        
        Args:
            df: 분석할 데이터프레임
            groupby_col: 그룹핑 컬럼 (brand, model 등)
            period_col: 기간 컬럼 (season, month 등)
            
        Returns:
            종합적 통계 검정 결과
        """
        try:
            results = {
                'summary': {},
                'independence_tests': {},
                'homogeneity_tests': {},
                'normality_tests': {},
                'post_hoc_tests': {},
                'effect_size_measures': {},
                'recommendations': []
            }
            
            # 기본 요약 통계
            results['summary'] = self._generate_summary_stats(df, groupby_col, period_col)
            
            # 독립성 검정
            results['independence_tests'] = self._test_independence(df, groupby_col, period_col)
            
            # 동질성 검정
            results['homogeneity_tests'] = self._test_homogeneity(df, groupby_col, period_col)
            
            # 정규성 검정
            results['normality_tests'] = self._test_normality(df, groupby_col, period_col)
            
            # 사후 검정 (유의한 결과가 있을 경우)
            if results['independence_tests']['chi_square']['significant']:
                results['post_hoc_tests'] = self._perform_post_hoc_tests(df, groupby_col, period_col)
            
            # 효과 크기 측정
            results['effect_size_measures'] = self._calculate_effect_sizes(df, groupby_col, period_col)
            
            # 분석 결과 기반 권장사항
            results['recommendations'] = self._generate_statistical_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"종합 통계 검정 중 오류 발생: {str(e)}")
            raise
    
    def seasonality_significance_test(self, time_series: pd.Series, period: int = 12) -> Dict[str, Any]:
        """
        계절성 유의성 검정
        
        Args:
            time_series: 시계열 데이터
            period: 계절 주기 (월별=12, 분기별=4)
            
        Returns:
            계절성 검정 결과
        """
        try:
            results = {}
            
            # 1. Friedman 검정 (비모수 계절성 검정)
            if len(time_series) >= period * 2:  # 최소 2주기 필요
                results['friedman_test'] = self._friedman_seasonality_test(time_series, period)
            
            # 2. Kruskal-Wallis 검정 (계절별 그룹 차이)
            results['kruskal_wallis_test'] = self._kruskal_wallis_seasonal_test(time_series, period)
            
            # 3. 순환성 검정 (Ljung-Box)
            results['ljung_box_test'] = self._ljung_box_test(time_series)
            
            # 4. 계절성 강도 검정
            results['seasonality_strength_test'] = self._seasonality_strength_test(time_series, period)
            
            return results
            
        except Exception as e:
            logger.error(f"계절성 검정 중 오류 발생: {str(e)}")
            raise
    
    def brand_comparison_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        브랜드 간 비교 검정
        
        Args:
            df: 브랜드 데이터가 포함된 데이터프레임
            
        Returns:
            브랜드 비교 검정 결과
        """
        try:
            results = {}
            
            brands = df['brand'].unique()
            
            if len(brands) < 2:
                return {'error': '비교할 브랜드가 충분하지 않습니다 (최소 2개 필요)'}
            
            # 브랜드별 데이터 분리
            brand_data = {brand: df[df['brand'] == brand] for brand in brands}
            
            # 1. 전체 브랜드 비교 (ANOVA 또는 Kruskal-Wallis)
            results['overall_comparison'] = self._compare_all_brands(brand_data)
            
            # 2. 쌍별 비교 (Pairwise comparisons)
            results['pairwise_comparisons'] = self._pairwise_brand_comparisons(brand_data)
            
            # 3. 시장 점유율 검정
            results['market_share_tests'] = self._market_share_tests(brand_data)
            
            return results
            
        except Exception as e:
            logger.error(f"브랜드 비교 검정 중 오류 발생: {str(e)}")
            raise
    
    def _generate_summary_stats(self, df: pd.DataFrame, groupby_col: str, period_col: str) -> Dict[str, Any]:
        """요약 통계 생성"""
        # 교차표 생성
        contingency = pd.crosstab(df[groupby_col], df[period_col])
        
        return {
            'total_observations': len(df),
            'unique_groups': len(df[groupby_col].unique()),
            'unique_periods': len(df[period_col].unique()),
            'contingency_table': contingency.to_dict(),
            'row_totals': contingency.sum(axis=1).to_dict(),
            'column_totals': contingency.sum(axis=0).to_dict(),
            'expected_cell_count': len(df) / (len(df[groupby_col].unique()) * len(df[period_col].unique())),
            'min_cell_count': contingency.min().min(),
            'cells_below_5': (contingency < 5).sum().sum()
        }
    
    def _test_independence(self, df: pd.DataFrame, groupby_col: str, period_col: str) -> Dict[str, Any]:
        """독립성 검정"""
        results = {}
        
        # 교차표 생성
        contingency = pd.crosstab(df[groupby_col], df[period_col])
        
        # 카이제곱 검정
        chi2_result = self._chi_square_test(contingency)
        results['chi_square'] = chi2_result
        
        # Fisher의 정확검정 (2x2 테이블일 경우)
        if contingency.shape == (2, 2):
            results['fisher_exact'] = self._fisher_exact_test(contingency)
        
        # G-test (우도비 검정)
        results['g_test'] = self._g_test(contingency)
        
        return results
    
    def _test_homogeneity(self, df: pd.DataFrame, groupby_col: str, period_col: str) -> Dict[str, Any]:
        """동질성 검정"""
        results = {}
        
        # 그룹별 데이터 분리
        groups = [df[df[groupby_col] == group][period_col] for group in df[groupby_col].unique()]
        
        # Levene 검정 (등분산성)
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            try:
                levene_stat, levene_p = levene(*groups)
                results['levene_test'] = {
                    'test_name': 'Levene Test',
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p),
                    'significant': levene_p < self.alpha,
                    'interpretation': 'Unequal variances' if levene_p < self.alpha else 'Equal variances assumed'
                }
            except Exception as e:
                results['levene_test'] = {'error': str(e)}
        
        # Bartlett 검정 (정규성 가정하에 등분산성)
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            try:
                bartlett_stat, bartlett_p = stats.bartlett(*groups)
                results['bartlett_test'] = {
                    'test_name': 'Bartlett Test',
                    'statistic': float(bartlett_stat),
                    'p_value': float(bartlett_p),
                    'significant': bartlett_p < self.alpha,
                    'interpretation': 'Unequal variances' if bartlett_p < self.alpha else 'Equal variances assumed'
                }
            except Exception as e:
                results['bartlett_test'] = {'error': str(e)}
        
        return results
    
    def _test_normality(self, df: pd.DataFrame, groupby_col: str, period_col: str) -> Dict[str, Any]:
        """정규성 검정"""
        results = {}
        
        for group in df[groupby_col].unique():
            group_data = df[df[groupby_col] == group][period_col]
            
            if len(group_data) < 3:
                continue
                
            group_results = {}
            
            # Shapiro-Wilk 검정 (n < 5000)
            if len(group_data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(group_data)
                    group_results['shapiro_wilk'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'significant': shapiro_p < self.alpha,
                        'normal': shapiro_p >= self.alpha
                    }
                except Exception as e:
                    group_results['shapiro_wilk'] = {'error': str(e)}
            
            # D'Agostino-Pearson 검정
            if len(group_data) >= 8:
                try:
                    dagostino_stat, dagostino_p = normaltest(group_data)
                    group_results['dagostino_pearson'] = {
                        'statistic': float(dagostino_stat),
                        'p_value': float(dagostino_p),
                        'significant': dagostino_p < self.alpha,
                        'normal': dagostino_p >= self.alpha
                    }
                except Exception as e:
                    group_results['dagostino_pearson'] = {'error': str(e)}
            
            results[str(group)] = group_results
        
        return results
    
    def _chi_square_test(self, contingency: pd.DataFrame) -> Dict[str, Any]:
        """카이제곱 검정"""
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # 효과 크기 (Cramer's V)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            # 가정 확인
            assumptions_met = (expected >= 5).all().all()
            low_expected_count = (expected < 5).sum().sum()
            
            return {
                'test_name': 'Chi-Square Test of Independence',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < self.alpha,
                'effect_size': float(cramers_v),
                'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
                'assumptions_met': assumptions_met,
                'cells_below_expected': int(low_expected_count),
                'warnings': [] if assumptions_met else ['일부 셀의 기대빈도가 5 미만입니다'],
                'interpretation': self._interpret_chi_square(p_value, cramers_v)
            }
            
        except Exception as e:
            return {'test_name': 'Chi-Square Test', 'error': str(e)}
    
    def _fisher_exact_test(self, contingency: pd.DataFrame) -> Dict[str, Any]:
        """Fisher의 정확검정"""
        try:
            table = contingency.values
            odds_ratio, p_value = stats.fisher_exact(table)
            
            return {
                'test_name': 'Fisher Exact Test',
                'odds_ratio': float(odds_ratio),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'interpretation': f'Odds ratio: {odds_ratio:.3f}'
            }
            
        except Exception as e:
            return {'test_name': 'Fisher Exact Test', 'error': str(e)}
    
    def _g_test(self, contingency: pd.DataFrame) -> Dict[str, Any]:
        """G-test (우도비 검정)"""
        try:
            observed = contingency.values
            expected = stats.contingency.expected_freq(observed)
            
            # G-statistic 계산
            mask = observed > 0
            g_stat = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
            
            dof = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
            p_value = 1 - stats.chi2.cdf(g_stat, dof)
            
            return {
                'test_name': 'G-test (Log-likelihood ratio)',
                'statistic': float(g_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < self.alpha,
                'interpretation': 'G-test는 큰 표본에서 카이제곱 검정보다 정확합니다'
            }
            
        except Exception as e:
            return {'test_name': 'G-test', 'error': str(e)}
    
    def _perform_post_hoc_tests(self, df: pd.DataFrame, groupby_col: str, period_col: str) -> Dict[str, Any]:
        """사후 검정"""
        results = {}
        
        # 잔차 분석
        contingency = pd.crosstab(df[groupby_col], df[period_col])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # 표준화 잔차 계산
        residuals = (contingency - expected) / np.sqrt(expected * (1 - contingency.sum(axis=1).values[:, np.newaxis] / contingency.sum().sum()) * (1 - contingency.sum(axis=0).values / contingency.sum().sum()))
        
        results['standardized_residuals'] = {
            'residuals': residuals.to_dict(),
            'significant_cells': self._identify_significant_cells(residuals),
            'interpretation': '표준화 잔차의 절댓값이 2 이상인 셀은 통계적으로 유의한 기여를 합니다'
        }
        
        return results
    
    def _calculate_effect_sizes(self, df: pd.DataFrame, groupby_col: str, period_col: str) -> Dict[str, Any]:
        """효과 크기 계산"""
        contingency = pd.crosstab(df[groupby_col], df[period_col])
        n = contingency.sum().sum()
        
        # 카이제곱 통계량
        chi2, _, _, _ = chi2_contingency(contingency)
        
        # 다양한 효과 크기 지표
        results = {
            'cramers_v': float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))),
            'phi_coefficient': float(np.sqrt(chi2 / n)) if contingency.shape == (2, 2) else None,
            'contingency_coefficient': float(np.sqrt(chi2 / (chi2 + n))),
            'tschuprows_t': float(np.sqrt(chi2 / (n * np.sqrt((contingency.shape[0] - 1) * (contingency.shape[1] - 1)))))
        }
        
        # 해석 추가
        results['cramers_v_interpretation'] = self._interpret_cramers_v(results['cramers_v'])
        
        return results
    
    def _friedman_seasonality_test(self, time_series: pd.Series, period: int) -> Dict[str, Any]:
        """Friedman 계절성 검정"""
        try:
            # 데이터를 기간별로 재구성
            data_matrix = []
            for i in range(0, len(time_series) - period + 1, period):
                if i + period <= len(time_series):
                    data_matrix.append(time_series.iloc[i:i+period].values)
            
            if len(data_matrix) < 2:
                return {'error': 'Friedman 검정을 위한 데이터가 부족합니다'}
            
            # Friedman 검정 수행
            stat, p_value = stats.friedmanchisquare(*np.array(data_matrix).T)
            
            return {
                'test_name': 'Friedman Test for Seasonality',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'interpretation': '계절성이 있음' if p_value < self.alpha else '계절성이 없음'
            }
            
        except Exception as e:
            return {'test_name': 'Friedman Test', 'error': str(e)}
    
    def _kruskal_wallis_seasonal_test(self, time_series: pd.Series, period: int) -> Dict[str, Any]:
        """Kruskal-Wallis 계절 검정"""
        try:
            # 계절별 그룹 생성
            seasonal_groups = []
            for season in range(period):
                seasonal_data = time_series.iloc[season::period]
                if len(seasonal_data) > 0:
                    seasonal_groups.append(seasonal_data.values)
            
            if len(seasonal_groups) < 2:
                return {'error': 'Kruskal-Wallis 검정을 위한 그룹이 부족합니다'}
            
            # Kruskal-Wallis 검정
            stat, p_value = stats.kruskal(*seasonal_groups)
            
            return {
                'test_name': 'Kruskal-Wallis Test for Seasonal Differences',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'interpretation': '계절별 차이가 있음' if p_value < self.alpha else '계절별 차이가 없음'
            }
            
        except Exception as e:
            return {'test_name': 'Kruskal-Wallis Test', 'error': str(e)}
    
    def _ljung_box_test(self, time_series: pd.Series, lags: int = 10) -> Dict[str, Any]:
        """Ljung-Box 검정 (자기상관 검정)"""
        try:
            # 간단한 Ljung-Box 구현
            n = len(time_series)
            acf_values = []
            
            for lag in range(1, min(lags + 1, n // 4)):
                if lag < n:
                    correlation = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                    if not np.isnan(correlation):
                        acf_values.append(correlation ** 2)
            
            if not acf_values:
                return {'error': 'Ljung-Box 검정을 수행할 수 없습니다'}
            
            # Ljung-Box 통계량
            lb_stat = n * (n + 2) * sum(acf / (n - k) for k, acf in enumerate(acf_values, 1))
            p_value = 1 - stats.chi2.cdf(lb_stat, len(acf_values))
            
            return {
                'test_name': 'Ljung-Box Test for Autocorrelation',
                'statistic': float(lb_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'interpretation': '자기상관이 있음' if p_value < self.alpha else '자기상관이 없음'
            }
            
        except Exception as e:
            return {'test_name': 'Ljung-Box Test', 'error': str(e)}
    
    def _seasonality_strength_test(self, time_series: pd.Series, period: int) -> Dict[str, Any]:
        """계절성 강도 검정"""
        try:
            # 계절별 분산 vs 전체 분산
            seasonal_vars = []
            
            for season in range(period):
                seasonal_data = time_series.iloc[season::period]
                if len(seasonal_data) > 1:
                    seasonal_vars.append(seasonal_data.var())
            
            if not seasonal_vars:
                return {'error': '계절성 강도 계산을 위한 데이터가 부족합니다'}
            
            within_season_var = np.mean(seasonal_vars)
            total_var = time_series.var()
            
            # F-test 유사한 통계량
            strength_ratio = total_var / within_season_var if within_season_var > 0 else 0
            
            return {
                'test_name': 'Seasonality Strength Test',
                'strength_ratio': float(strength_ratio),
                'within_season_variance': float(within_season_var),
                'total_variance': float(total_var),
                'seasonality_strength': float(1 - within_season_var / total_var) if total_var > 0 else 0,
                'interpretation': '강한 계절성' if strength_ratio > 2 else '약한 계절성'
            }
            
        except Exception as e:
            return {'test_name': 'Seasonality Strength Test', 'error': str(e)}
    
    def _compare_all_brands(self, brand_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """전체 브랜드 비교"""
        # 각 브랜드의 계절별 평균 사용량 비교
        brand_seasonal_means = []
        brand_names = []
        
        for brand, data in brand_data.items():
            if 'season' in data.columns and len(data) > 0:
                seasonal_means = data.groupby('season').size()
                brand_seasonal_means.append(seasonal_means.values)
                brand_names.append(brand)
        
        if len(brand_seasonal_means) < 2:
            return {'error': '비교할 브랜드 데이터가 부족합니다'}
        
        try:
            # Kruskal-Wallis 검정
            stat, p_value = stats.kruskal(*brand_seasonal_means)
            
            return {
                'test_name': 'Kruskal-Wallis Test for Brand Comparison',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'brands_compared': brand_names,
                'interpretation': '브랜드 간 유의한 차이가 있음' if p_value < self.alpha else '브랜드 간 유의한 차이가 없음'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _pairwise_brand_comparisons(self, brand_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """쌍별 브랜드 비교"""
        results = {}
        brands = list(brand_data.keys())
        
        for i in range(len(brands)):
            for j in range(i + 1, len(brands)):
                brand1, brand2 = brands[i], brands[j]
                
                try:
                    # Mann-Whitney U 검정
                    data1 = brand_data[brand1]['season'] if 'season' in brand_data[brand1].columns else []
                    data2 = brand_data[brand2]['season'] if 'season' in brand_data[brand2].columns else []
                    
                    if len(data1) > 0 and len(data2) > 0:
                        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        results[f'{brand1}_vs_{brand2}'] = {
                            'test_name': 'Mann-Whitney U Test',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < self.alpha,
                            'interpretation': f'{brand1}과 {brand2} 간 유의한 차이가 있음' if p_value < self.alpha else f'{brand1}과 {brand2} 간 유의한 차이가 없음'
                        }
                        
                except Exception as e:
                    results[f'{brand1}_vs_{brand2}'] = {'error': str(e)}
        
        return results
    
    def _market_share_tests(self, brand_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """시장 점유율 검정"""
        brand_counts = {brand: len(data) for brand, data in brand_data.items()}
        total_count = sum(brand_counts.values())
        
        # 균등분포에 대한 검정
        expected_equal = total_count / len(brand_counts)
        observed = list(brand_counts.values())
        expected = [expected_equal] * len(brand_counts)
        
        try:
            chi2_stat, p_value = stats.chisquare(observed, expected)
            
            return {
                'equal_share_test': {
                    'test_name': 'Chi-square goodness-of-fit test for equal market share',
                    'statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'significant': p_value < self.alpha,
                    'market_shares': {brand: count/total_count for brand, count in brand_counts.items()},
                    'interpretation': '시장점유율이 균등하지 않음' if p_value < self.alpha else '시장점유율이 균등함'
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _identify_significant_cells(self, residuals: pd.DataFrame) -> Dict[str, Any]:
        """유의한 셀 식별"""
        significant_cells = {}
        
        for row in residuals.index:
            for col in residuals.columns:
                residual_value = residuals.loc[row, col]
                if abs(residual_value) > 2:  # |z| > 2는 p < 0.05에 해당
                    significant_cells[f'{row}_{col}'] = {
                        'residual': float(residual_value),
                        'interpretation': 'Over-represented' if residual_value > 0 else 'Under-represented'
                    }
        
        return significant_cells
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Cramer's V 해석"""
        if cramers_v < 0.1:
            return 'negligible association'
        elif cramers_v < 0.3:
            return 'small association'
        elif cramers_v < 0.5:
            return 'medium association'
        else:
            return 'large association'
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """카이제곱 검정 결과 해석"""
        if p_value < self.alpha:
            effect_size = self._interpret_cramers_v(cramers_v)
            return f'통계적으로 유의한 연관성이 있으며, 효과 크기는 {effect_size}입니다.'
        else:
            return '통계적으로 유의한 연관성이 없습니다.'
    
    def _generate_statistical_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """통계 분석 결과 기반 권장사항"""
        recommendations = []
        
        # 독립성 검정 결과 기반
        if 'chi_square' in results.get('independence_tests', {}):
            chi_square_result = results['independence_tests']['chi_square']
            if chi_square_result.get('significant', False):
                effect_size = chi_square_result.get('effect_size', 0)
                if effect_size > 0.3:
                    recommendations.append('브랜드와 계절 간에 강한 연관성이 있습니다. 계절별 마케팅 전략을 차별화하는 것을 권장합니다.')
                else:
                    recommendations.append('브랜드와 계절 간에 약한 연관성이 있습니다. 추가적인 분석을 통해 패턴을 확인하세요.')
        
        # 데이터 품질 관련
        summary = results.get('summary', {})
        if summary.get('cells_below_5', 0) > 0:
            recommendations.append('일부 셀의 관측치가 적습니다. 데이터를 더 수집하거나 카테고리를 통합하는 것을 고려하세요.')
        
        # 정규성 검정 결과 기반
        normality_results = results.get('normality_tests', {})
        non_normal_groups = [group for group, tests in normality_results.items() 
                           if any(test.get('normal', True) == False for test in tests.values() if isinstance(test, dict))]
        
        if non_normal_groups:
            recommendations.append('일부 그룹이 정규분포를 따르지 않습니다. 비모수 검정 사용을 고려하세요.')
        
        if not recommendations:
            recommendations.append('통계적 검정 결과가 신뢰할 만합니다. 결과를 바탕으로 의사결정을 진행하세요.')
        
        return recommendations