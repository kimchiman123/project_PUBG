from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode
from pyspark.ml.feature import VectorAssembler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# PySpark 세션 생성
spark = SparkSession.builder.appName("PUBG_Analysis").getOrCreate()

# JSON 파일 로드
df_normal = spark.read.json('PUBG_player_squad.json', multiLine=True)
df_banned = spark.read.json('PUBG_ban_player_squad.json', multiLine=True)

# participants 배열을 explode하여 개별 플레이어 데이터를 추출
df_normal = df_normal.withColumn("participant", explode(col("participants")))
df_banned = df_banned.withColumn("participant", explode(col("participants")))

# 필요한 특성들을 participants 내부의 속성에서 추출
df_normal = df_normal.select(col("participant.*")).withColumn("banned", lit(0))
df_banned = df_banned.select(col("participant.*")).withColumn("banned", lit(1))

# 두 데이터셋을 결합
df = df_normal.union(df_banned)

# 필요한 특성 선택
features = ['DBNOs', 'assists', 'boosts', 'damageDealt', 'headshotKills', 
            'killPlace', 'killStreaks', 'kills', 'longestKill', 'rideDistance',
            'roadKills', 'timeSurvived', 'vehicleDestroys', 'winPlace']

# 결측값 제거
df = df.na.drop(subset=features)

# 역수 값 적용
df = df.withColumn('winPlace', 1 / col('winPlace')).withColumn('killPlace', 1 / col('killPlace'))

# 샘플링 전 데이터 개수 확인
total_count = df.count()
count_banned = df.filter(col('banned') == 1).count()
count_normal = df.filter(col('banned') == 0).count()

print(f"총 데이터 개수: {total_count}")
print(f"정상 유저(banned=0) 개수: {count_normal}")
print(f"어뷰징 유저(banned=1) 개수: {count_banned}")

# 과대 샘플링 (소수 클래스 데이터 복제)
oversampling_factor = count_normal / count_banned  # 정상 유저와 어뷰징 유저의 비율
df_banned_oversampled = df.filter(col('banned') == 1).sample(withReplacement=True, fraction=oversampling_factor)

# 다수 클래스와 소수 클래스를 결합하여 균형 잡힌 데이터셋 생성
df_balanced = df.filter(col('banned') == 0).union(df_banned_oversampled)

# 샘플링 후 데이터 개수 확인
balanced_total_count = df_balanced.count()
balanced_banned_count = df_balanced.filter(col('banned') == 1).count()
balanced_normal_count = df_balanced.filter(col('banned') == 0).count()

print(f"샘플링 후 총 데이터 개수: {balanced_total_count}")
print(f"샘플링 후 정상 유저(banned=0) 개수: {balanced_normal_count}")
print(f"샘플링 후 어뷰징 유저(banned=1) 개수: {balanced_banned_count}")

# PySpark 데이터프레임을 Pandas로 변환
df_balanced_pd = df_balanced.select(features + ['banned']).toPandas()

# 특성과 레이블을 분리
X = df_balanced_pd[features]
y = df_balanced_pd['banned']

# 데이터 분할 (훈련 및 테스트 데이터셋)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTE 오버샘플링 적용 (훈련 데이터에만 적용)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

# XGBoost 모델 정의
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 랜덤 서치 객체 정의 (5-fold 교차 검증 적용, n_iter로 시도할 조합 수를 제한)
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, 
                                   scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, n_iter=50)  # n_iter = 시도할 조합 수

# 하이퍼파라미터 튜닝 적용 (훈련 데이터에 대해 수행)
random_search.fit(X_train_resampled, y_train_resampled)

# 최적 하이퍼파라미터 확인
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best ROC AUC: {random_search.best_score_}")

# 최적 하이퍼파라미터를 적용한 모델로 테스트 데이터에 대해 예측
best_xgb_model = random_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)
y_prob = best_xgb_model.predict_proba(X_test)[:, 1]  # 양성 클래스 확률

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
error_rate = 1 - accuracy

print(f"Test Accuracy: {accuracy}")
print(f"Test ROC AUC: {roc_auc}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")
print(f"Test Error Rate: {error_rate}")

# ROC Curve 그리기
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# 모델 저장
joblib.dump(best_xgb_model, 'xgb_model_final3.pki')

print('XGB모델이 PKI파일로 저장되었습니다.')