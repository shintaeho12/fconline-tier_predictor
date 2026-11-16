import streamlit as st
import requests
import pandas as pd
import joblib
import time

API_KEY = st.text_input(
    "🔑 NEXON Open API Key를 입력하세요",
    type="password",
    placeholder="test_... 또는 live_..."
)


st.set_page_config(
    page_title="FC 온라인 AI 티어 분석기",
    page_icon="⚽",
    layout="wide"
)

MODEL_PATH = "tier_model.pkl"
API_HOST = ".".join(["open", "api", "nexon", "com"])
API_PREFIX = "https" + "://" + API_HOST + "/fconline/v1"

DIVISION_MAP = {
    800: "슈퍼챔피언스", 900: "챔피언스", 1000: "슈퍼챌린저",
    1100: "챌린저 1부", 1200: "챌린저 2부", 1300: "챌린저 3부",
    1400: "마스터 1부", 1500: "마스터 2부", 1600: "마스터 3부",
    1700: "월드클래스 1부", 1800: "월드클래스 1부", 1900: "월드클래스 1부", 2000: "월드클래스 1부",
    2100: "월드클래스 2부", 2200: "월드클래스 3부",
    2300: "프로 1부", 2400: "프로 2부", 2500: "프로 3부",
    2600: "세미프로 1부", 2700: "세미프로 2부", 2800: "세미프로 3부",
    2900: "유망주 1부", 3000: "유망주 2부", 3100: "유망주 3부",
}

# ==========================================
# 2. 모델 로드 및 파생 변수 생성 함수
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def create_features(df):
    data = df.copy()
    data["pass_success_rate"] = data["pass_success"] / (data["pass_try"] + 1e-5)
    data["through_pass_ratio"] = data["through_pass_try"] / (data["pass_try"] + 1e-5)
    data["lobbed_through_ratio"] = data["lobbed_through_try"] / (data["pass_try"] + 1e-5)
    data["effective_shoot_ratio"] = data["effective_shoot"] / (data["shoot_total"] + 1e-5)
    data["goal_conversion_rate"] = data["goal_total"] / (data["effective_shoot"] + 1e-5)
    data["tackle_success_rate"] = data["tackle_success"] / (data["tackle_try"] + 1e-5)
    data["block_success_rate"] = data["block_success"] / (data["block_try"] + 1e-5)
    data["dribble_per_possession"] = data["dribble"] / (data["possession"] + 1e-5)
    data["shoot_per_possession"] = data["shoot_total"] / (data["possession"] + 1e-5)

    feature_cols = [
        "possession", "dribble", "foul", "corner_kick", "offside_count",
        "shoot_total", "effective_shoot", "goal_total",
        "pass_try", "pass_success", "short_pass_try", "through_pass_try", "lobbed_through_try",
        "tackle_try", "tackle_success", "block_try", "block_success",
        "pass_success_rate", "through_pass_ratio", "lobbed_through_ratio",
        "effective_shoot_ratio", "goal_conversion_rate",
        "tackle_success_rate", "block_success_rate",
        "dribble_per_possession", "shoot_per_possession"
    ]
    return data[feature_cols], data

# ==========================================
# 3. API 데이터 조회 함수 (1시간 캐싱)
# ==========================================
def safe_request(url, headers, params=None):
    for attempt in range(3):
        try:
            return requests.get(url, headers=headers, params=params, timeout=20)
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(1.0)
                continue
            raise ValueError("넥슨 API 서버 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cached_user_matches(nickname, api_key, max_matches=5):
    headers = {"x-nxopen-api-key": api_key.strip()}

    id_res = safe_request(f"{API_PREFIX}/id", headers=headers, params={"nickname": nickname})
    if id_res.status_code == 429:
        raise ValueError("오늘의 API 일일 호출 한도가 초과되었습니다. 오른쪽 '스탯 직접 입력 시뮬레이터' 탭을 이용해 주세요.")
    if id_res.status_code in (401, 403):
        raise ValueError("API Key가 유효하지 않습니다. 코드 상단의 API_KEY를 확인해 주세요.")
    if id_res.status_code != 200:
        raise ValueError(f"'{nickname}' 구단주를 찾을 수 없습니다.")
    ouid = id_res.json().get("ouid")
    time.sleep(0.2)

    div_res = safe_request(f"{API_PREFIX}/user/maxdivision", headers=headers, params={"ouid": ouid})
    actual_div = None
    if div_res.status_code == 200:
        for item in div_res.json():
            if item.get("matchType") == 50:
                actual_div = item.get("division")
                break
    time.sleep(0.2)

    match_res = safe_request(
        f"{API_PREFIX}/user/match",
        headers=headers,
        params={"ouid": ouid, "matchtype": 50, "offset": 0, "limit": 7}
    )
    if match_res.status_code != 200 or not match_res.json():
        raise ValueError("최근 공식경기 기록이 존재하지 않습니다.")
    time.sleep(0.2)

    valid_rows = []
    for match_id in match_res.json():
        if len(valid_rows) >= max_matches:
            break
        detail_res = safe_request(f"{API_PREFIX}/match-detail", headers=headers, params={"matchid": match_id})
        if detail_res.status_code != 200:
            continue

        for info in detail_res.json().get("matchInfo", []):
            if info.get("ouid") == ouid:
                m_detail = info.get("matchDetail", {})
                if m_detail.get("matchEndType") == 0 and info.get("pass", {}).get("passTry", 0) >= 20:
                    shoot = info.get("shoot", {})
                    pass_info = info.get("pass", {})
                    defence = info.get("defence", {})

                    valid_rows.append({
                        "possession": m_detail.get("possession", 0),
                        "dribble": m_detail.get("dribble", 0),
                        "foul": m_detail.get("foul", 0),
                        "corner_kick": m_detail.get("cornerKick", 0),
                        "offside_count": m_detail.get("offsideCount", 0),
                        "shoot_total": shoot.get("shootTotal", 0),
                        "effective_shoot": shoot.get("effectiveShootTotal", 0),
                        "goal_total": shoot.get("goalTotal", 0),
                        "pass_try": pass_info.get("passTry", 0),
                        "pass_success": pass_info.get("passSuccess", 0),
                        "short_pass_try": pass_info.get("shortPassTry", 0),
                        "through_pass_try": pass_info.get("throughPassTry", 0),
                        "lobbed_through_try": pass_info.get("lobbedThroughPassTry", 0),
                        "tackle_try": defence.get("tackleTry", 0),
                        "tackle_success": defence.get("tackleSuccess", 0),
                        "block_try": defence.get("blockTry", 0),
                        "block_success": defence.get("blockSuccess", 0),
                    })
        time.sleep(0.2)

    if not valid_rows:
        raise ValueError("분석 가능한 정상 종료 공식경기 기록이 부족합니다.")

    avg_df = pd.DataFrame(valid_rows).mean().to_frame().T
    return avg_df, len(valid_rows), actual_div

# ==========================================
# 4. 웹 UI 렌더링
# ==========================================
st.title("⚽ FC 온라인 AI 공식경기 티어 유추 서비스")
st.markdown("Random Forest 머신러닝 모델이 경기 세부 지표(패스 효율, 스루패스 비중, 슈팅 정교함 등)를 분석하여 **챌린저 3부 이상 상위권 플레이 스타일**인지 판별합니다.")

model = load_model()

tab1, tab2 = st.tabs(["🔍 구단주 닉네임 검색 분석", "🎛️ 스탯 직접 입력 시뮬레이터"])

# ------------------------------------------
# TAB 1: 구단주 닉네임 검색 분석
# ------------------------------------------
with tab1:
    st.subheader("구단주 닉네임으로 최근 공식경기 5판 평균 분석")
    nickname_input = st.text_input("분석할 구단주 닉네임 입력", placeholder="구단주 닉네임을 입력하세요")
    run_btn = st.button("🚀 AI 티어 분석 실행", type="primary")

    if run_btn:
        if not API_KEY or "여기에_본인의" in API_KEY:
            st.error("app.py 코드 상단의 API_KEY 변수에 본인의 NEXON Open API Key를 먼저 입력해 주세요.")
        elif not nickname_input.strip():
            st.warning("구단주 닉네임을 입력해 주세요.")
        else:
            try:
                with st.spinner(f"'{nickname_input}' 구단주의 최근 공식경기 데이터를 분석 중입니다..."):
                    avg_df, match_cnt, actual_div = fetch_cached_user_matches(nickname_input.strip(), API_KEY, max_matches=5)
                    X_input, full_df = create_features(avg_df)

                    # 대괄호 인덱스 없이 .item()과 .ravel()로 안전하게 단일 값 추출
                    pred_class = int(model.predict(X_input).item())
                    proba_list = model.predict_proba(X_input).ravel().tolist()
                    high_prob = float(proba_list.pop()) * 100

                pred_label = "🏆 상위권 (챌린저1↑)" if pred_class == 1 else "⚽ 일반권 (챌린저2↓)"
                actual_label = DIVISION_MAP.get(actual_div, f"코드 {actual_div}" if actual_div else "기록 없음")

                st.success(f"분석 완료! (최근 정상 공식경기 {match_cnt}판 평균 데이터 기준)")
                c1, c2, c3 = st.columns(3)
                c1.metric("AI 예측 티어 그룹", pred_label)
                c2.metric("상위권(챌린저 1부↑) 확률", f"{high_prob:.1f}%")
                c3.metric("실제 공식경기 최고 등급", actual_label)

                st.progress(int(high_prob))
                st.divider()

                row_vals = full_df.squeeze()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("스루패스 비중", f"{row_vals['through_pass_ratio'] * 100:.1f}%")
                m2.metric("패스 성공률", f"{row_vals['pass_success_rate'] * 100:.1f}%")
                m3.metric("유효슈팅 비율", f"{row_vals['effective_shoot_ratio'] * 100:.1f}%")
                m4.metric("평균 점유율", f"{row_vals['possession']:.1f}%")

            except Exception as e:
                st.error(f"분석 실패: {e}")

with tab2:
    st.subheader("내 경기 스탯으로 직접 테스트해 보기")
    st.markdown("슬라이더를 조절해 경기 기록을 입력하면 학습된 `tier_model.pkl`이 실시간으로 티어를 예측합니다.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sim_poss = st.slider("점유율 (%)", 20, 80, 52)
        sim_pass_try = st.slider("전체 패스 시도", 30, 200, 110)
        sim_pass_acc = st.slider("패스 성공률 (%)", 50, 98, 90)
    with col_b:
        sim_through_try = st.slider("스루패스 시도", 0, 50, 13)
        sim_shoot_total = st.slider("전체 슈팅 수", 1, 25, 9)
        sim_eff_shoot = st.slider("유효 슈팅 수", 0, sim_shoot_total, min(6, sim_shoot_total))
    with col_c:
        sim_goal = st.slider("득점 수", 0, sim_eff_shoot, min(2, sim_eff_shoot))
        sim_tackle_try = st.slider("태클 시도", 0, 30, 10)
        sim_tackle_suc = st.slider("태클 성공", 0, sim_tackle_try, min(7, sim_tackle_try))

    sim_row = pd.DataFrame([{
        "possession": sim_poss,
        "dribble": int(sim_poss * 1.4),
        "foul": 1,
        "corner_kick": 2,
        "offside_count": 1,
        "shoot_total": sim_shoot_total,
        "effective_shoot": sim_eff_shoot,
        "goal_total": sim_goal,
        "pass_try": sim_pass_try,
        "pass_success": int(sim_pass_try * (sim_pass_acc / 100)),
        "short_pass_try": int(sim_pass_try * 0.8),
        "through_pass_try": sim_through_try,
        "lobbed_through_try": int(sim_through_try * 0.2),
        "tackle_try": sim_tackle_try,
        "tackle_success": sim_tackle_suc,
        "block_try": 5,
        "block_success": 3,
    }])

    X_sim, _ = create_features(sim_row)
    sim_pred = int(model.predict(X_sim).item())
    sim_prob = float(model.predict_proba(X_sim).ravel().tolist().pop()) * 100

    st.divider()
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("시뮬레이션 AI 예측 결과", "🏆 상위권 (챌린저3↑)" if sim_pred == 1 else "⚽ 일반권 (마스터1↓)")
    res_col2.metric("상위권 플레이 스타일 일치도", f"{sim_prob:.1f}%")


st.divider()
st.caption("Data based on NEXON Open API")
