# 테스트 파일
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn as sns

print("--- 라이브러리 임포트 성공 ---")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

# 간단한 데이터프레임 생성 테스트
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
})
print("\n--- Pandas 데이터프레임 생성 성공 ---")
print(df.head())

print("\n환경 검증 완료. 주요 라이브러리가 모두 정상적으로 동작합니다.")
