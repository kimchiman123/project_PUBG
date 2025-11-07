from flask import Flask, request, jsonify, render_template, send_file
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import MinMaxScaler
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 대신 Agg 백엔드 사용
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import io

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 환경에서 Malgun Gothic 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

app = Flask(__name__)

# 모델 로드
xgb_model_path = 'xgb_model_prot.pkl'
rf_model_path = 'random_forest_model.pkl'
isolation_model = 'final_isolation_forest_model_double_scaled.pkl'

try:
    with open(xgb_model_path, 'rb') as file:
        xgb_model = joblib.load(file)
    print("XGBoost 모델이 성공적으로 로드되었습니다.")
except (joblib.UnpicklingError, FileNotFoundError) as e:
    print(f"XGBoost 모델 로드 실패: {e}")

try:
    with open(rf_model_path, 'rb') as file:
        rf_model = joblib.load(file)
    print("랜덤 포레스트 모델이 성공적으로 로드되었습니다.")
except (joblib.UnpicklingError, FileNotFoundError) as e:
    print(f"랜덤 포레스트 모델 로드 실패: {e}")

try:
    with open(isolation_model, 'rb') as file:
        isolation_forest_model = joblib.load(file)
    print("Isolation Forest 모델이 성공적으로 로드되었습니다.")
except (joblib.UnpicklingError, FileNotFoundError) as e:
    print(f"Isolation Forest 모델 로드 실패: {e}")

# PUBG API 설정
API_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiI5NGI4M2M2MC01MTc4LTAxM2QtMjE2YS0xZTk3NmMxZGFkMzUiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNzI1OTU1MTkxLCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6InNhbmcifQ.-8ErzX775Yhl5hrZanUz64CR9fFhMtRt5EzjRsMeiHE' 
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Accept': 'application/vnd.api+json'
}

FEATURES = ['DBNOs', 'assists','boosts','damageDealt', 'headshotKills', 
            'killPlace','killStreaks', 'kills','longestKill', 'rideDistance',
            'timeSurvived', 'winPlace','heals','walkDistance']

# 플레이어 데이터 요청
def get_player_data(player_name):
    url = f'https://api.pubg.com/shards/steam/players?filter[playerNames]={player_name}'
    response = requests.get(url, headers=HEADERS)
    return response.json() if response.status_code == 200 else None

# 매치 데이터 요청
def get_match_data(match_id):
    response = requests.get(f'https://api.pubg.com/shards/steam/matches/{match_id}', headers=HEADERS)
    return response.json() if response.status_code == 200 else None

# 매치 ID 추출
def extract_match_ids(player_data, max_matches=50):
    match_ids = []
    if player_data and 'data' in player_data and len(player_data['data']) > 0:
        player = player_data['data'][0]
        matches_data = player['relationships']['matches']['data']
        match_ids = [match['id'] for match in matches_data[:max_matches]]
    return match_ids

# 참가자 데이터 필터링
def filter_participant_data(participants, account_id):
    return [p for p in participants if p['playerId'] == account_id and p['damageDealt'] > 0 and p['timeSurvived'] >= 240]

# 플레이어 데이터 처리
def process_player_data(player_name):
    player_data = get_player_data(player_name)
    if player_data:
        match_ids = extract_match_ids(player_data, max_matches=50)
        account_id = player_data['data'][0]['id']
        all_processed_matches = []

        for match_id in match_ids:
            match_data = get_match_data(match_id)
            if match_data:
                game_mode = match_data['data']['attributes']['gameMode']
                if game_mode in ["squad", "squad-fpp"]:
                    participants = [item['attributes']['stats'] for item in match_data['included'] if item['type'] == 'participant']
                    filtered_data = filter_participant_data(participants, account_id)
                    if filtered_data:
                        for participant in filtered_data:
                            participant['match_id'] = match_id
                            participant['gameMode'] = game_mode
                        all_processed_matches.extend(filtered_data)
        
        if all_processed_matches:
            return pd.DataFrame(all_processed_matches)
    return None

# XGBoost 예측
def xgb_predict(df):
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    y_pred_proba = xgb_model.predict_proba(X)[:, 1]
    return y_pred_proba

# 매치 시간에 따른 예측 확률 변화 추세 시각화 xgboost
def plot_probability_trend(df):
    df = df.sort_values(by='timeSurvived')
    plt.figure(figsize=(10, 6))
    plt.plot(df['timeSurvived'], df['prediction_probability'], marker='o')
    plt.xlabel('매치 시간 (초)')
    plt.ylabel('예측 확률')
    plt.title('매치 시간에 따른 XGBoost 예측 확률 변화 추세')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# 랜덤 포레스트 예측
def rf_predict(df):
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    y_pred_proba = rf_model.predict_proba(X)[:, 1]
    return y_pred_proba

# Isolation Forest 예측 (이상치 감지)
def isolation_forest_predict(df):
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()
    x1 = robust_scaler.fit_transform(X)
    x2 = standard_scaler.fit_transform(x1)
    y_pred_score = isolation_forest_model.decision_function(x2)  # decision_function 사용
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_pred_score_normalized = scaler.fit_transform(y_pred_score.reshape(-1, 1)).flatten()
    return y_pred_score_normalized

@app.route('/plot_trend')
def plot_trend():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['prediction_probability'] = xgb_predict(matches_df)
        img = plot_probability_trend(matches_df)
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."

# 다양한 특징에 따른 이상 탐지 시도
def flag_anomalies(df, feature, threshold_percentile=99):
    threshold_value = np.percentile(df[feature], threshold_percentile)
    df[f'{feature}_anomaly'] = df[feature] >= threshold_value
    return df

def detect_anomalies(df):
    features_to_check = ['damageDealt', 'kills', 'longestKill']
    for feature in features_to_check:
        df = flag_anomalies(df, feature)
    return df

# 상세 보고서 생성 및 다운로드
def export_report(df, player_name):
    filename = f'{player_name}_report.csv'
    df.to_csv(filename, index=False)
    return filename

# XGBoost시각화 함수
def create_visualization(df):
    labels = ['일반', '의심', '확실']
    counts = [
        len(df[df['prediction'] == 0]),
        len(df[(df['prediction'] == 1) & (df['prediction_probability'] < 0.7)]),
        len(df[(df['prediction'] == 1) & (df['prediction_probability'] >= 0.7)])
    ]


    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'orange', 'red'])
    plt.xlabel('유저 상태')
    plt.ylabel('매치 수')
    plt.title("XGBoost유저 상태 분포")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

    
# Isolation Forest 점수에 따른 매치 시간 추세 시각화 함수
@app.route('/plot_trend_isolation')
def plot_isolation_trend():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['isolation_forest_score'] = isolation_forest_predict(matches_df)
        img = plot_probability_trend_isolation(matches_df)
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."


# 랜덤 포레스트 예측 확률을 추가한 상세 보고서 생성 및 다운로드
@app.route('/download_report_rf')
def download_report_rf():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['prediction_probability_rf'] = rf_predict(matches_df)
        matches_df['prediction_rf'] = matches_df['prediction_probability_rf'] >= 0.5
        matches_df = detect_anomalies(matches_df)  # 이상 탐지 수행
        filename = export_report(matches_df, f'{player_name}_rf')
        return send_file(filename, as_attachment=True)
    return "데이터를 찾을 수 없습니다."

# XGBoost 예측 확률을 추가한 상세 보고서 생성 및 다운로드
@app.route('/download_report')
def download_report():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['prediction_probability'] = xgb_predict(matches_df)
        matches_df['prediction'] = matches_df['prediction_probability'] >= 0.5
        matches_df = detect_anomalies(matches_df)
        filename = export_report(matches_df, player_name)
        return send_file(filename, as_attachment=True)
    return "데이터를 찾을 수 없습니다."

@app.route('/download_report_isolation')
def download_report_isolation():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        # Isolation Forest 점수 계산
        matches_df['isolation_forest_score'] = isolation_forest_predict(matches_df)
        
        matches_df['isolation_anomaly'] = matches_df['isolation_forest_score'] > 0.8
        
        # 추가적인 이상 탐지 수행
        matches_df = detect_anomalies(matches_df)
        
        # 보고서 파일 생성 및 다운로드
        filename = export_report(matches_df, f'{player_name}_isolation')
        return send_file(filename, as_attachment=True)
    return "데이터를 찾을 수 없습니다."

# 랜덤 포레스트 전용 시각화 함수
def create_rf_visualization(df):
    labels = ['일반', '의심', '확실']
    
    counts = [
        len(df[df['prediction_rf'] == 0]),
        len(df[(df['prediction_rf'] == 1) & (df['prediction_probability_rf'] < 0.7)]),
        len(df[(df['prediction_rf'] == 1) & (df['prediction_probability_rf'] >= 0.7)])
    ]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'orange', 'red'])
    plt.xlabel('유저 상태')
    plt.ylabel('매치 수')
    plt.title("랜덤 포레스트 유저 상태 분포")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img



# 랜덤 포레스트 예측 확률을 이용한 매치 시간에 따른 예측 확률 변화 추세 시각화
@app.route('/plot_trend_rf')
def plot_trend_rf():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['prediction_probability_rf'] = rf_predict(matches_df)
        img = plot_probability_trend_rf(matches_df)
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."

# 랜덤 포레스트 예측 확률에 따른 시각화 함수
def plot_probability_trend_rf(df):
    df = df.sort_values(by='timeSurvived')
    plt.figure(figsize=(10, 6))
    plt.plot(df['timeSurvived'], df['prediction_probability_rf'], marker='o', color='purple')
    plt.xlabel('매치 시간 (초)')
    plt.ylabel('랜덤 포레스트 예측 확률')
    plt.title('매치 시간에 따른 랜덤 포레스트 예측 확률 변화 추세')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# Isolation Forest 시각화함수 
def create_isolation_visualization(df):
    labels = ['일반', '의심', '확실']
    counts = [
       len(df[df['isolation_forest_score'] < 0.5]),
       len(df[(df['isolation_forest_score'] >= 0.5) & (df['isolation_forest_score'] <= 0.8)]),
       len(df[df['isolation_forest_score'] > 0.8])

    ]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'orange', 'red'])
    plt.xlabel('유저 상태')
    plt.ylabel('매치 수')
    plt.title("Isolation Forest 유저 상태 분포")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img


# Isolation Forest 예측 점수에 따른 상태 분류 및 시각화 함수
def plot_probability_trend_isolation(df):
    df = df.sort_values(by='timeSurvived')
    plt.figure(figsize=(10, 6))
    plt.plot(df['timeSurvived'], df['isolation_forest_score'], marker='o', color='cyan')
    plt.xlabel('매치 시간 (초)')
    plt.ylabel('Isolation Forest 점수')
    plt.title('매치 시간에 따른 Isolation Forest 점수 변화 추세')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    player_name = request.form['player_name']
    
    # 플레이어 데이터를 처리하여 매치 데이터로 변환
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        # 모델 예측 수행
        y_pred = xgb_predict(matches_df)
        y_pred_rf = rf_predict(matches_df)
        isolation_scores = isolation_forest_predict(matches_df)

        print("Isolation Forest 점수 확인:", isolation_scores)  # 점수 출력

        # 분석된 매치 수 계산
        match_count = len(matches_df)

        # 각 모델의 평균 예측 확률 및 긍정 예측 비율 계산
        avg_pred_xgb = float(round(y_pred.mean(), 2))
        avg_pred_rf = float(round(y_pred_rf.mean(), 2))
        avg_isolation_score = float(round(isolation_scores.mean(), 2))

        positive_ratio_xgb = float(round(sum(y_pred > 0.5) / match_count, 2))
        positive_ratio_rf = float(round(sum(y_pred_rf > 0.5) / match_count, 2))
        anomaly_ratio_isolation = float(round((isolation_scores > 0.3).mean(), 2))
        
        # 종합 결과 반환
        result = {
            'summary': {
                'model1': {
                    'avg_prediction': avg_pred_xgb,         # 모델 1의 평균 예측 확률
                    'positive_ratio': positive_ratio_xgb    # 모델 1의 긍정 예측 비율
                },
                'model2': {
                    'avg_prediction': avg_pred_rf,          # 모델 2의 평균 예측 확률
                    'positive_ratio': positive_ratio_rf     # 모델 2의 긍정 예측 비율
                },
                'model3': {
                    'average_score': avg_isolation_score,   # Isolation Forest의 평균 점수
                    'anomaly_ratio': anomaly_ratio_isolation # Isolation Forest의 이상치 비율
                },
                'match_count': match_count                 # 총 매치 수
            }
        }
        
        return jsonify(status='success', data=result)
    else:
        error_message = 'Failed to process player data or no matches found'
        return jsonify(status='error', message=error_message)

@app.route('/plot') #XGBoost
def plot():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        # 예측 확률 추가
        matches_df['prediction_probability'] = xgb_predict(matches_df)
        # 예측 결과를 이진 값으로 변환하여 추가 (0 또는 1로 설정)
        matches_df['prediction'] = matches_df['prediction_probability'] >= 0.5

        # 시각화 생성
        img = create_visualization(matches_df)
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."


@app.route('/plot_rf') #랜덤포레스트
def plot_rf():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['prediction_probability_rf'] = rf_predict(matches_df)
        matches_df['prediction_rf'] = matches_df['prediction_probability_rf'] >= 0.5

        # 랜덤 포레스트 시각화 생성
        img = create_rf_visualization(matches_df)
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."

@app.route('/plot_trend_isolation') #isolation
def plot_trend_isolation():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        matches_df['isolation_forest_score'] = isolation_forest_predict(matches_df)
        img = plot_probability_trend_isolation(matches_df)
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."

@app.route('/plot_isolation_status')
def plot_isolation_status():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    if matches_df is not None:
        # Isolation Forest 점수 계산 후 상태별로 분류
        matches_df['isolation_forest_score'] = isolation_forest_predict(matches_df)
        img = create_isolation_visualization(matches_df)  # 막대그래프 함수 호출
        return send_file(img, mimetype='image/png')
    return "데이터를 찾을 수 없습니다."





# 매치별 예측 확률 데이터를 반환하는 엔드포인트
@app.route('/detailed_predictions', methods=['GET'])
def detailed_predictions():
    player_name = request.args.get('player_name')
    matches_df = process_player_data(player_name)
    
    if matches_df is not None:
        # 예측 확률 계산
        matches_df['xgb_prediction'] = xgb_predict(matches_df)
        matches_df['rf_prediction'] = rf_predict(matches_df)
        matches_df['isolation_forest_score'] = isolation_forest_predict(matches_df)

        # 매치별 예측 확률 데이터를 딕셔너리 리스트로 변환
        match_predictions = matches_df[['match_id', 'timeSurvived', 'xgb_prediction', 'rf_prediction','isolation_forest_score']].to_dict(orient='records')
        
        # JSON 형식으로 반환
        return jsonify(status='success', data=match_predictions)
    else:
        return jsonify(status='error', message="데이터를 찾을 수 없습니다.")

if __name__ == '__main__':
    app.run(debug=True)