import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =============================================================================
# 운영체제별 한글 폰트 설정 (matplotlib에서 한글이 깨지지 않도록)
# =============================================================================
import platform

def set_korean_font():
    """운영체제에 따라 적절한 한글 폰트 설정"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':  # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux 또는 기타
        plt.rcParams['font.family'] = 'DejaVu Sans'

    # 마이너스 폰트 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정 적용
set_korean_font()




# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="Kaggle Competition Dashboard",           # 브라우저 탭에 표시될 제목
    page_icon="🏆",                    # 브라우저 탭 아이콘
    layout="wide",                     # 와이드 레이아웃: 화면 전체 폭 사용
    initial_sidebar_state="expanded"   # 페이지 로드 시 사이드바 펼쳐진 상태로 시작
)


# # 🔧 캐글 블루 태그 색상 재정의
# st.markdown("""
#     <style>
#     /* 태그 박스 스타일 */
#     .stMultiSelect [data-baseweb="tag"] {
#         background-color: #20BEFF !important;
#         color: white !important;
#         border-radius: 8px !important;
#         padding: 4px 8px !important;
#         font-weight: 500 !important;
#     }
#     /* X 아이콘 색상 */
#     .stMultiSelect [data-baseweb="tag"] svg {
#         fill: white !important;
#     }
#     </style>
# """, unsafe_allow_html=True)



# =============================================================================
# 데이터 생성 함수
# =============================================================================
@st.cache_data  # 데이터 캐싱: 함수 실행 결과를 메모리에 저장하여 재실행 방지
def load_data():
    comp1 = pd.read_parquet('competition_1.parquet')
    comp_date = pd.read_parquet('competition_date.parquet')
    org = pd.read_parquet('Organization_clean.parquet')
    return comp1, comp_date, org

# 데이터 로드 (캐싱된 함수 호출)
comp1, comp_date, org = load_data()

# =============================================================================
# 사이드바 구성
# =============================================================================
with st.sidebar:
    # 로고 이미지 표시 시도, 실패 시 텍스트로 대체
    try:
        st.image("kaggle-logo-transparent-300.png", width=200)
    except:
        st.markdown("🤼 **캐글 대회 대시보드**")


    st.divider()  # 구분선 추가

    # -------------------------------------------------------------------------
    # 알고리즘 필터 섹션
    # -------------------------------------------------------------------------
    st.subheader("평가 기준 필터")
    
    # 데이터에서 고유한 카테고리 목록 추출
    all_categories_algo = comp1['AlgorithmCategory'].unique()
    
    # 다중 선택 위젯: 기본값으로 모든 카테고리 선택
    selected_algo_categories = st.multiselect(
        "카테고리 선택",
        options=all_categories_algo,
        default=all_categories_algo,                    # 모든 카테고리를 기본 선택
        help="분석할 카테고리를 선택하세요"        # 도움말 텍스트
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 카테고리 필터 섹션
    # -------------------------------------------------------------------------
    st.subheader("대회 유형 필터")
    
    # 데이터에서 고유한 카테고리 목록 추출
    all_categories_host = comp1['HostSegmentTitle'].unique()
    
    # 다중 선택 위젯: 기본값으로 모든 카테고리 선택
    selected_host_categories = st.multiselect(
        "카테고리 선택",
        options=all_categories_host,
        default=all_categories_host,                    # 모든 카테고리를 기본 선택
        help="분석할 카테고리를 선택하세요"        # 도움말 텍스트
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 상금 필터 섹션
    # -------------------------------------------------------------------------
    st.subheader("💸 상금 구간 필터")
    
    def make_rewardgroup(x):
        if x < 10:
            return 'None'
        elif x < 5000:
            return '10~4999'
        elif x < 10000:
            return '5000~9999'
        elif x < 50000:
            return '10000~49999'
        else:
            return '50000+'

    comp1['RewardGroup'] = comp1['RewardQuantity'].apply(make_rewardgroup)

    # 데이터에서 고유한 지역 목록 추출
    all_rewards = sorted(comp1['RewardGroup'].unique().tolist())
    options_with_all = ['전체'] + all_rewards

    selected_reward = st.multiselect(
        "상금 구간 선택",
        options=options_with_all,
        default=['전체']
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 연도 필터 섹션
    # -------------------------------------------------------------------------
    st.subheader("📅 연도 필터")
    
    comp1['Year'] = comp1['EnabledDate'].dt.year

    # 대회 Id 17374번의 시작 날짜 변경 
    comp1.loc[comp1['CompetitionId'] == 17374, 'EnabledDate'] = pd.to_datetime('2019-11-20')

    # 시계열 분석시 시작 날짜 파악 불가능한 유일한 대회 -> Id 31017번 대회 제외
    comp1 = comp1[comp1['CompetitionId'] != 31017]

    # 데이터에서 고유한 지역 목록 추출
    all_years = comp1['Year'].unique()
    min_year = all_years.min()
    max_year = all_years.max()
    
    # 다중 선택 위젯: 기본값으로 모든 지역 선택
    selected_year_range = st.slider(
        "연도 범위 선택",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),  # ✅ 전체 범위를 기본값으로 설정
        step=1
    )

    st.divider()
    # -------------------------------------------------------------------------
    # 액션 버튼 섹션
    # -------------------------------------------------------------------------
    
    # 필터 초기화 버튼
    if st.button("🔄 필터 초기화", use_container_width=True):
        st.session_state.clear()  # 모든 세션 상태 삭제
        st.rerun()  # 페이지 새로고침

# =============================================================================
# 메인 대시보드 영역
# =============================================================================

# -------------------------------------------------------------------------
# 데이터 필터링
# -------------------------------------------------------------------------
# 사용자가 선택한 조건에 따라 원본 데이터를 필터링
filtered_df = comp1[
    (comp1['AlgorithmCategory'].isin(selected_algo_categories)) &              # 알고리즘 필터
    (comp1['HostSegmentTitle'].isin(selected_host_categories))  &               # Host 필터
    (comp1['Year'] >= selected_year_range[0]) &                      # 시작일 이후
    (comp1['Year'] <= selected_year_range[1])                         # 종료일 이전
]

   # 전체 선택이면 전체 사용
if '전체' not in selected_reward:
    filtered_df = filtered_df[filtered_df['RewardGroup'].isin(selected_reward)]


# -------------------------------------------------------------------------
# 헤더 및 기본 정보
# -------------------------------------------------------------------------
st.title("📊 캐글 대회 대시보드")
st.markdown(f"**분석 기간**: {selected_year_range[0]} ~ {selected_year_range[1]} | **데이터 건수**: {len(filtered_df):,}건")

# 필터링 결과 데이터가 없는 경우 경고 메시지 표시 후 중단
if len(filtered_df) == 0:
    st.warning("⚠️ 선택한 조건에 해당하는 데이터가 없습니다. 필터 조건을 확인해주세요.")
    st.stop()  # 이후 코드 실행 중단

# -------------------------------------------------------------------------
# KPI 메트릭 카드 섹션
# -------------------------------------------------------------------------
# 4개의 컬럼으로 주요 지표들을 나란히 배치
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    # 현재 활성화된 대회 수
    comp_big1 = filtered_df[filtered_df['DeadlineStatus'] != 'Closed']
    st.metric(
        "현재 활성화된 대회 수",
        f"{comp_big1.shape[0]:,.0f}개"           # 천단위 구분기호 포함
    )

with col2:
    # 대회 전체 평균 경쟁률
    comp_big2 = round(filtered_df['CompetitionRate'].median(),2)
    st.metric(
        "대회 경쟁률 중앙값",
        f"{comp_big2:,.0f} : 1"  
    )

with col3:
    # 평균 대회 상금 (상금 100달러 이상 대회 한정)
    comp_plus100 = filtered_df[
        (filtered_df['RewardQuantity'] >= 100) & 
        (~filtered_df['RewardType'].isin(['EUR', 'GBP']))]
    
    comp_big4 = round(comp_plus100['RewardQuantity'].mean(),0)
    
    st.metric(
        "평균 대회 상금",
        f"${comp_big4:,.0f}",
        "상금 100달러 이상 대회 한정"
    )

with col4:
    # 대회 기간 중앙값
    comp_big5 = round(filtered_df['Duration'].median(),0)

    st.metric(
    "대회 기간 중앙값",
    f"{comp_big5:,.0f}일"
    )

with col5:
    # 대회 참가팀 중앙값
    comp_big6 = round(filtered_df['TotalTeams'].mean(),2)

    st.metric(
    "대회 참가팀 중앙값",
    f"{comp_big6:,.0f}팀"
    )

    # with tab3:
    #     comp_big7 = round(filtered_df['TotalSubmissions'].mean(),2)

    #     st.metric(
    #     "대회 1개당 평균 제출 수",
    #     f"{comp_big7:,}회"
    #     )

st.divider()

# -------------------------------------------------------------------------
# 차트 영역 (2개 컬럼으로 구성)
# -------------------------------------------------------------------------

col1, col2 = st.columns([1, 1])  # 좌우 비율 1:1

from sklearn.preprocessing import MinMaxScaler

# ▣ 좌측: 단독 레이더 차트
with col1:
    st.subheader("📌 대회 유형별 특징 차트")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["레이더 차트", "데이터 원본"])


    with tab1:
        comp_radar = filtered_df.groupby('HostSegmentTitle').agg({
            'CompetitionId': 'count',
            'RewardQuantity': 'mean',
            'CompetitionRate': 'mean',
            'TotalTeams': 'mean',
            'Duration': 'mean'
        }).reset_index()

        comp_radar.rename(columns={
            'HostSegmentTitle': '대회 유형',
            'CompetitionId': '대회 수',
            'RewardQuantity': '평균 상금',
            'CompetitionRate': '평균 경쟁률',
            'TotalTeams': '평균 참가팀',
            'Duration': '평균 대회기간'           
            }, inplace=True)

        # 정규화
        radar_data = comp_radar.set_index('대회 유형')
        scaler = MinMaxScaler()
        radar_normalized = pd.DataFrame(
            scaler.fit_transform(radar_data),
            index=radar_data.index,
            columns=radar_data.columns
        )

        # Plotly Radar Chart
        fig = go.Figure()

        for idx, row in radar_normalized.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.tolist() + [row.tolist()[0]],  # 닫기 위해 첫 값 추가
                theta=radar_normalized.columns.tolist() + [radar_normalized.columns.tolist()[0]],
                fill='toself',
                name=idx
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            legend=dict(
                x=-0.2,
                y=1.05,
                xanchor='left',
                yanchor='top'
            ),
            width=850,
            height=750,
            margin=dict(t=30, b=30, l=30, r=80)
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        comp_radar_tab2 = comp_radar.copy()
        comp_radar_tab2.rename(columns={
            '대회 수': '대회 수 (개)',
            '평균 상금': '평균 상금 (USD)',
            '평균 경쟁률': '평균 경쟁률 (:1)',
            '평균 참가팀': '평균 참가팀 (팀)',
            '평균 대회기간': '평균 대회기간 (일)'           
            }, inplace=True)
        comp_radar_tab2 = comp_radar_tab2.sort_values("대회 수 (개)", ascending=False)
        comp_radar_tab2.reset_index(drop=True, inplace=True)
        comp_radar_tab2.index += 1  # 인덱스 1부터 시작

        st.subheader("데이터 원본")
        st.dataframe(comp_radar_tab2.style.format({
        "대회 수 (개)": "{:,.0f}",       # 정수 (천 단위 콤마 포함)
        "평균 참가팀 (팀)": "{:,.0f}",   # 정수
        "평균 상금 (USD)": "{:,.0f}",     # 정수 (천 단위 콤마 포함)
        "평균 경쟁률 (:1)": "{:.2f}",    # 소수점 2자리
        "평균 대회기간 (일)": "{:,.0f}"   # 정수 (천 단위 콤마 포함)
        }))



## ----------------------------------------------------------------------

# ▣ 우측 상단: 탭 2개 (그래프)
with col2:
    with st.container():
        st.subheader("📈 연도별 트렌드")
        tab1, tab2, tab3 = st.tabs(["1.평가 기준", "1-1.데이터 원본", "2.누적 대회 수 & 상금"])

        with tab1:
            pivot_cat = filtered_df.pivot_table(
                index='AlgorithmCategory',
                columns='Year',
                values='CompetitionId',  
                aggfunc='count',
                fill_value=0
            )

            pivot_cat_sorted = pivot_cat.loc[pivot_cat.sum(axis=1).sort_values(ascending=False).index]
            pivot_cat_sorted['Total'] = pivot_cat_sorted.sum(axis=1)

            plot_trend_cat = pivot_cat_sorted.drop(columns='Total').head(10)
            if 2025 in plot_trend_cat.columns:
                plot_trend_cat = plot_trend_cat.drop(columns=[2025])

            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))
            for idx, row in plot_trend_cat.iterrows():
                ax.plot(row.index, row.values, label=idx)

            ax.set_title('Top 10 Evaluation Algorithms Usage Over Years 2010~2024')
            ax.set_xlabel('Year')
            ax.set_ylabel('Algorithm Counts')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig) 

        with tab2:
            st.subheader("📄 알고리즘 카테고리 연도별 대회 수 (원본 테이블)")
            st.dataframe(pivot_cat_sorted, use_container_width=True)
        
        with tab3:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 연도별 대회 수 / 상금 집계
            plot_4_1 = filtered_df.groupby('Year')['CompetitionId'].count().reset_index()
            plot_4_2 = filtered_df.groupby('Year')['RewardQuantity'].sum().reset_index()

            # 누적 상금 처리
            plot_4_2['RewardQuantity'] = plot_4_2['RewardQuantity'].cumsum()

            # 연도 정수형 변환 및 정렬
            plot_4_1['Year'] = plot_4_1['Year'].astype(int)
            plot_4_2['Year'] = plot_4_2['Year'].astype(int)
            plot_4_1 = plot_4_1[plot_4_1['Year'] < 2025]
            plot_4_2 = plot_4_2[plot_4_2['Year'] < 2025]
            plot_4_1 = plot_4_1.sort_values('Year')
            plot_4_2 = plot_4_2.sort_values('Year')

            # 문자열형 연도 생성 (x축 표기용)
            plot_4_1['Year_str'] = plot_4_1['Year'].astype(str)
            plot_4_2['Year_str'] = plot_4_2['Year'].astype(str)

            # 대회 수: 막대그래프
            sns.barplot(data=plot_4_1, x='Year_str', y='CompetitionId', ax=ax1, color='skyblue')
            ax1.set_ylabel("Annual Competition Count", fontsize=12)

            # 상금: 꺾은선 (누적)
            ax2 = ax1.twinx()
            sns.lineplot(data=plot_4_2, x='Year_str', y='RewardQuantity', marker='o', color='crimson', ax=ax2)
            ax2.set_ylabel("Cumulative Reward Quantity", fontsize=12)
            ax2.grid(False)

            # 기타 스타일
            ax1.set_xlabel("Year", fontsize=12)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            plt.title("Annual Competitions and Cumulative Rewards", fontsize=14)
            plt.legend()
            plt.tight_layout()

            st.pyplot(fig)
            

    st.divider()

## ----------------------------------------------------------------------

    # ▣ 우측 하단: 탭 2개 (표)
    with st.container():
        st.subheader("📋 Top10 기관 차트")
        tab3, tab4 = st.tabs(["🏆 대회 수 Top10", "💰 상금 규모 Top10"])

        with tab3:
            # 1. 필터링된 데이터에서 대회 수 집계
            comp_org = filtered_df.groupby('OrganizationId')['CompetitionId'].count().reset_index()
            comp_org = comp_org[comp_org['OrganizationId'] != 0]
            comp_org.rename(columns={'CompetitionId': 'NumberOfCompetitions'}, inplace=True)

            # 2. 대회 수 + 조직 정보 조인
            top_comp_org = pd.merge(comp_org, org, how='inner', left_on='OrganizationId', right_on='Id')
            top_comp_org.drop(columns='Id', axis=1, inplace=True)

            # 3. 평균 상금 집계 (필터 적용)
            comp_org_mean = filtered_df.groupby('OrganizationId')['RewardQuantity'].mean().reset_index()
            comp_org_mean.rename(columns={'RewardQuantity': 'MeanPrize'}, inplace=True)
            comp_org_mean['MeanPrize'] = comp_org_mean['MeanPrize'].astype(int)

            # 4. 평균 상금 + 조직 정보 조인
            top_comp_org_mean = pd.merge(comp_org_mean, org, how='inner', left_on='OrganizationId', right_on='Id')
            top_comp_org_mean.drop(columns='Id', axis=1, inplace=True)

            # 5. 두 정보 병합
            top10_comp_org_mean = pd.merge(comp_org_mean, top_comp_org, how='inner', on='OrganizationId')       

            # 6. 최종본 컬럼 이름 한글화
            top10_comp_org_mean.rename(columns={
            'Name_clean': '기관명',
            'NumberOfCompetitions': '대회 수 (개)',
            'MeanPrize': '평균 상금 (USD)',
            'industry': '산업군'        
            }, inplace=True)

            # 7. 최종 정리
            top10_table = top10_comp_org_mean[['기관명', '대회 수 (개)', '평균 상금 (USD)', '산업군']]\
                            .sort_values(by='대회 수 (개)', ascending=False).head(10).round(0).reset_index(drop=True)
            top10_table.index += 1

            # 8. 출력
            st.dataframe(top10_table.style.format({
                "평균 상금 (USD)": "{:,.0f}"   # 정수 (천 단위 콤마 포함)
                }))


        with tab4:
            # 1. 기관별 대회 수 집계
            comp_org2 = filtered_df.groupby('OrganizationId')['CompetitionId'].count().reset_index()
            comp_org2 = comp_org2[comp_org2['OrganizationId'] != 0]
            comp_org2.rename(columns={'CompetitionId': 'NumberOfCompetitions'}, inplace=True)

            # 2. 대회 수 기준 상위 기관 병합
            top_comp_org2 = pd.merge(comp_org2, org, how='inner', left_on='OrganizationId', right_on='Id')
            top_comp_org2.drop(columns='Id', axis=1, inplace=True)

            # 3. 기관별 총 상금 집계
            comp_org_sum = filtered_df.groupby('OrganizationId')['RewardQuantity'].sum().reset_index()
            comp_org_sum.rename(columns={'RewardQuantity': 'TotalPrize'}, inplace=True)

            # 4. 총 상금 기준 상위 기관 병합
            top_comp_org_sum = pd.merge(comp_org_sum, org, how='inner', left_on='OrganizationId', right_on='Id')
            top_comp_org_sum.drop(columns='Id', axis=1, inplace=True)
            top_comp_org_sum['TotalPrize'] = top_comp_org_sum['TotalPrize'].astype(int)

            # 5. 최종 병합 
            top10_comp_org_sum = pd.merge(comp_org_sum, top_comp_org2, how='inner', on='OrganizationId')

            # 6. 최종본 컬럼 이름 한글화
            top10_comp_org_sum.rename(columns={
            'Name_clean': '기관명',
            'NumberOfCompetitions': '대회 수 (개)',
            'TotalPrize': '총 상금 (USD)',
            'industry': '산업군'        
            }, inplace=True)

            # 7. 최종 정리
            top10sum_table = top10_comp_org_sum[
                ['기관명', '대회 수 (개)', '총 상금 (USD)', '산업군']
                ].sort_values(by='총 상금 (USD)', ascending=False).head(10).round(0).reset_index(drop=True)
            top10sum_table.index += 1
            
            # 8. 출력
            st.dataframe(top10sum_table.style.format({
                "총 상금 (USD)": "{:,.0f}"   # 정수 (천 단위 콤마 포함)
                }))
            