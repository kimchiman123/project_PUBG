import requests
import json
import os
import time
import argparse
from typing import List, Dict, Optional, Set


# ************ GIT에올리는 경우 반드시 키 값을 제거************
API_KEY = ""
OUTPUT_DIR = "match_data"

# --- 보조 함수 ---

def get_api_headers(api_key: str) -> Dict[str, str]:
    """API 요청에 필요한 헤더를 반환합니다."""
    return {
        'accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {api_key}'
    }

def make_request(url: str, headers: Dict[str, str], params: Optional[Dict] = None) -> Optional[Dict]:
    """지정된 URL에 GET 요청을 보내고 JSON 응답을 반환합니다."""
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 4xx 또는 5xx 상태 코드에 대해 예외를 발생시킵니다.
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        return None

# --- 파이프라인 단계 ---

def get_leaderboard_player_ids(platform: str, game_mode: str, headers: Dict[str, str]) -> List[str]:
    """리더보드 데이터를 가져와 플레이어 계정 ID를 추출합니다."""
    print(f"1단계: [{platform}/{game_mode}] 리더보드를 조회하여 플레이어 ID를 가져옵니다...")
    url = f"https://api.pubg.com/shards/{platform}/leaderboards/division.bro.official.pc-2018-38/{game_mode}"
    data = make_request(url, headers)
    
    if not data or 'included' not in data:
        print("리더보드 데이터를 가져오지 못했거나 'included'된 플레이어가 없습니다.")
        return []
        
    player_ids = [
        item['id'] 
        for item in data['included'] 
        if item['type'] == 'player'
    ]
    print(f"리더보드에서 {len(player_ids)}명의 플레이어를 찾았습니다.")
    return player_ids

def get_match_ids_from_players(player_ids: List[str], platform: str, user_type: str, headers: Dict[str, str]) -> Set[str]:
    """
    플레이어 데이터를 배치로 가져와 밴된 플레이어를 필터링하고 매치 ID를 수집합니다.
    """
    print(f"2단계: [{user_type}] 플레이어 데이터를 조회하여 매치 ID를 수집합니다...")
    all_match_ids = set()
    batch_size = 10  # 플레이어 엔드포인트는 요청당 최대 10개의 ID를 지원합니다.

    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i+batch_size]
        print(f"플레이어 배치 처리 중 {i//batch_size + 1}/{(len(player_ids) + batch_size - 1)//batch_size}...")
        
        params = {'filter[playerIds]': ",".join(batch)}
        url = f"https://api.pubg.com/shards/{platform}/players"
        player_data = make_request(url, headers, params)

        if not player_data or 'data' not in player_data:
            continue

        for player in player_data['data']:
            is_banned = player.get('attributes', {}).get('banType', 'Innocent') != 'Innocent'
            
            # 설정에 따라 유저 필터링
            if user_type == 'normal' and is_banned:
                print(f"정상 유저 필터: 밴된 플레이어({player['id']})를 건너뜁니다.")
                continue
            if user_type == 'abnormal' and not is_banned:
                print(f"비정상 유저 필터: 정상 플레이어({player['id']})를 건너뜁니다.")
                continue
            
            # 매치 ID 수집
            matches = player.get('relationships', {}).get('matches', {}).get('data', [])
            for match in matches:
                all_match_ids.add(match['id'])
        
        # 동일하게 API제한으로 느리게 설정 
        time.sleep(5) # API 속도 제한을 준수합니다.

    print(f"총 {len(all_match_ids)}개의 고유한 매치 ID를 찾았습니다.")
    return all_match_ids

def download_match_data(match_ids: Set[str], platform: str, game_mode: str, user_type: str, headers: Dict[str, str], num_matches: int):
    """
    지정된 수의 매치에 대한 데이터를 다운로드하고 저장합니다.
    이미 다운로드된 매치는 건너뜁니다.
    """
    print(f"3단계: 최대 {num_matches}개의 매치 데이터를 다운로드합니다...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    processed_count = 0
    match_list = list(match_ids)

    for i, match_id in enumerate(match_list):
        if processed_count >= num_matches:
            print(f"요청한 매치 수({num_matches})에 도달했습니다.")
            break

        print(f"매치 처리 중 {i+1}/{len(match_list)} (ID: {match_id})")
        output_path = os.path.join(OUTPUT_DIR, f"{match_id}.json")

        if os.path.exists(output_path):
            print(f"{match_id}의 매치 데이터가 이미 존재합니다. 건너뜁니다.")
            continue

        url = f"https://api.pubg.com/shards/{platform}/matches/{match_id}"
        match_data = make_request(url, headers)

        if match_data:
            # 파이프라인 설정 정보 추가
            match_data['pipeline_settings'] = {
                'platform': platform,
                'game_mode': game_mode,
                'user_type': user_type
            }
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(match_data, f, indent=4, ensure_ascii=False)
                print(f"매치 데이터를 {output_path}에 성공적으로 저장했습니다.")
                processed_count += 1
            except IOError as e:
                print(f"{output_path} 파일 쓰기 실패: {e}")
        
        # 현재 1분 10회 요청 제한으로 인해, 매우 느리게 작동 
        time.sleep(5)

    print(f"다운로드를 완료했습니다. 이번 실행에서 처리된 총 매치 수: {processed_count}")


# --- 메인 실행 ---

def main():
    """PUBG 데이터 파이프라인을 실행하는 메인 함수"""
    parser = argparse.ArgumentParser(description="PUBG 데이터 수집 파이프라인")
    
    parser.add_argument(
        '--user_type',
        type=str,
        default='normal',
        choices=['normal', 'abnormal'],
        help='수집할 유저 유형 (normal: 정상, abnormal: 밴)'
    )
    parser.add_argument(
        '--platform',
        type=str,
        default='pc-kakao', # krjp서버는 오류가 떠서, 카카오를 기본으로 
        choices=['pc-krjp', 'pc-kakao'], # 스팀서버(한국, 일본), 카카오
        help='데이터를 수집할 서버 (steam, kakao)'
    )
    parser.add_argument(
        '--game_mode',
        type=str,
        default='squad',
        choices=['solo', 'duo', 'squad'],
        help='리더보드를 조회할 게임 모드 (solo, duo, squad)'
    )
    parser.add_argument(
        '--num_matches',
        type=int,
        default=50,
        help='새로 다운로드할 최대 매치 수'
    )
    
    args = parser.parse_args()

    print("--- PUBG 데이터 파이프라인 시작 ---")
    print(f"설정: 유저 유형={args.user_type}, 서버={args.platform}, 게임 모드={args.game_mode}, 매치 수={args.num_matches}")
    
    if not API_KEY or API_KEY == "여기에_API_키를_붙여넣으세요":
        print("오류: 스크립트 상단의 API_KEY 변수에 API 키를 입력해주세요.")
        return

    headers = get_api_headers(API_KEY)

    # 1단계: 리더보드에서 플레이어 ID 가져오기
    player_ids = get_leaderboard_player_ids(args.platform, args.game_mode, headers)
    if not player_ids:
        print("플레이어 ID를 찾을 수 없어 파이프라인을 중지합니다.")
        return

    # 2단계: 설정된 유저 유형에 따라 매치 ID 가져오기
    match_ids = get_match_ids_from_players(player_ids, args.platform, args.user_type, headers)
    if not match_ids:
        print("매치 ID를 찾을 수 없어 파이프라인을 중지합니다.")
        return

    # 3단계: 매치 데이터 다운로드
    download_match_data(match_ids, args.platform, args.game_mode, args.user_type, headers, args.num_matches)

    print("--- PUBG 데이터 파이프라인 종료 ---")

if __name__ == "__main__":
    main()