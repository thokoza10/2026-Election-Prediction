import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="eThekwini Election Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.12), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(16, 185, 129, 0.08), transparent 25%),
        #080b12;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background: #0b0f18;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.hero {
    padding: 2.5rem;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(15,23,42,0.96), rgba(17,24,39,0.92));
    border: 1px solid #273449;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -100px;
    top: -120px;
    border-radius: 50%;
    background: rgba(37,99,235,0.12);
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 850px;
    line-height: 1.7;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(37,99,235,0.14);
    border: 1px solid rgba(96,165,250,0.3);
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(145deg, #111827, #0d131f);
    border: 1px solid #253247;
    border-radius: 18px;
    padding: 1.35rem;
    min-height: 145px;
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #3b82f6;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.4rem;
    color: #f8fafc;
}

.metric-description {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.35rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 1.8rem;
    margin-bottom: 0.7rem;
}

.section-subtitle {
    color: #94a3b8;
    margin-bottom: 1rem;
}

.info-box {
    background: rgba(15,23,42,0.72);
    border: 1px solid #253247;
    border-radius: 16px;
    padding: 1.2rem;
    margin: 0.7rem 0;
}

.info-box strong {
    color: #f8fafc;
}

.warning-box {
    background: rgba(120,53,15,0.16);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 16px;
    padding: 1.1rem;
    color: #fde68a;
}

.success-box {
    background: rgba(6,78,59,0.18);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 1.1rem;
    color: #a7f3d0;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1f2937;
    color: #64748b;
    font-size: 0.78rem;
    text-align: center;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #253247;
    border-radius: 12px;
}

button[kind="primary"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_data():

    df1 = pd.read_csv("ETH.csv")
    df2 = pd.read_csv("ETH (1).csv")

    df_2021_pr = df1[df1["BallotType"] == "PR"].copy()
    df_2021_ward = df1[df1["BallotType"] == "Ward"].copy()

    df_2016_pr = df2[df2["BallotType"] == "PR"].copy()
    df_2016_ward = df2[df2["BallotType"] == "Ward"].copy()

    datasets = [
        df_2016_pr,
        df_2016_ward,
        df_2021_pr,
        df_2021_ward
    ]

    for df in datasets:

        df["Province"] = df["Province"].astype(str).str.strip()
        df["Municipality"] = df["Municipality"].astype(str).str.strip()
        df["Ward"] = df["Ward"].astype(str).str.strip()
        df["VotingStationName"] = (
            df["VotingStationName"].astype(str).str.strip()
        )
        df["PartyName"] = df["PartyName"].astype(str).str.strip()
        df["BallotType"] = df["BallotType"].astype(str).str.strip()

        df["DateGenerated"] = pd.to_datetime(
            df["DateGenerated"],
            errors="coerce"
        )

        for column in [
            "VotingDistrict",
            "RegisteredVoters",
            "SpoiltVotes",
            "TotalValidVotes"
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df_2016_pr["ElectionYear"] = 2016
    df_2016_ward["ElectionYear"] = 2016
    df_2021_pr["ElectionYear"] = 2021
    df_2021_ward["ElectionYear"] = 2021

    return (
        df_2016_pr,
        df_2016_ward,
        df_2021_pr,
        df_2021_ward
    )


@st.cache_data
def prepare_analysis(
    df_2016_pr,
    df_2016_ward,
    df_2021_pr,
    df_2021_ward
):

    common_wards = sorted(
        set(df_2016_ward["Ward"].unique())
        &
        set(df_2021_ward["Ward"].unique())
    )

    df_2016_pr_common = df_2016_pr[
        df_2016_pr["Ward"].isin(common_wards)
    ].copy()

    df_2021_pr_common = df_2021_pr[
        df_2021_pr["Ward"].isin(common_wards)
    ].copy()

    pr_2016_ward = (
        df_2016_pr_common
        .groupby(["Ward", "PartyName"], as_index=False)
        ["TotalValidVotes"]
        .sum()
    )

    pr_2021_ward = (
        df_2021_pr_common
        .groupby(["Ward", "PartyName"], as_index=False)
        ["TotalValidVotes"]
        .sum()
    )

    pr_2016_totals = (
        pr_2016_ward
        .groupby("Ward", as_index=False)
        ["TotalValidVotes"]
        .sum()
        .rename(
            columns={
                "TotalValidVotes":
                "WardTotalVotes"
            }
        )
    )

    pr_2021_totals = (
        pr_2021_ward
        .groupby("Ward", as_index=False)
        ["TotalValidVotes"]
        .sum()
        .rename(
            columns={
                "TotalValidVotes":
                "WardTotalVotes"
            }
        )
    )

    pr_2016_ward = pr_2016_ward.merge(
        pr_2016_totals,
        on="Ward",
        how="left"
    )

    pr_2021_ward = pr_2021_ward.merge(
        pr_2021_totals,
        on="Ward",
        how="left"
    )

    pr_2016_ward["VoteShare"] = (
        pr_2016_ward["TotalValidVotes"]
        /
        pr_2016_ward["WardTotalVotes"]
    ) * 100

    pr_2021_ward["VoteShare"] = (
        pr_2021_ward["TotalValidVotes"]
        /
        pr_2021_ward["WardTotalVotes"]
    ) * 100

    ml_data = (
        pr_2016_ward[
            [
                "Ward",
                "PartyName",
                "TotalValidVotes",
                "VoteShare",
                "WardTotalVotes"
            ]
        ]
        .rename(
            columns={
                "TotalValidVotes": "Votes_2016",
                "VoteShare": "VoteShare_2016",
                "WardTotalVotes": "WardTotalVotes_2016"
            }
        )
    )

    ml_data_2021 = (
        pr_2021_ward[
            [
                "Ward",
                "PartyName",
                "TotalValidVotes",
                "VoteShare",
                "WardTotalVotes"
            ]
        ]
        .rename(
            columns={
                "TotalValidVotes": "Votes_2021",
                "VoteShare": "VoteShare_2021",
                "WardTotalVotes": "WardTotalVotes_2021"
            }
        )
    )

    ml_data = (
        ml_data
        .merge(
            ml_data_2021,
            on=["Ward", "PartyName"],
            how="outer"
        )
        .fillna(0)
    )

    ml_data["VoteChange_2016_2021"] = (
        ml_data["Votes_2021"]
        -
        ml_data["Votes_2016"]
    )

    ml_data["VoteShareChange_2016_2021"] = (
        ml_data["VoteShare_2021"]
        -
        ml_data["VoteShare_2016"]
    )

    return (
        common_wards,
        pr_2016_ward,
        pr_2021_ward,
        ml_data
    )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_or_train_model(ml_data):

    model_file = "eThekwini_2026_election_model.pkl"

    if os.path.exists(model_file):

        package = joblib.load(model_file)

        model = package["model"]
        feature_columns = package["feature_columns"]

        encoded = pd.get_dummies(
            ml_data,
            columns=["Ward"],
            dtype=int
        )

        encoded = encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        X = encoded[
            feature_columns
        ]

        y = ml_data["VoteShare_2021"]

        return model, feature_columns, X, y, True

    encoded = pd.get_dummies(
        ml_data,
        columns=["Ward"],
        dtype=int
    )

    feature_columns = [
        "Votes_2016",
        "VoteShare_2016",
        "WardTotalVotes_2016"
    ] + [
        c for c in encoded.columns
        if c.startswith("Ward_")
    ]

    X = encoded[feature_columns]
    y = ml_data["VoteShare_2021"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns
        },
        model_file
    )

    return model, feature_columns, X, y, False


# ============================================================
# FORECAST
# ============================================================

@st.cache_data
def create_forecast(
    pr_2021_ward,
    _model,
    feature_columns
):

    forecast = pr_2021_ward[
        [
            "Ward",
            "PartyName",
            "TotalValidVotes",
            "VoteShare",
            "WardTotalVotes"
        ]
    ].copy()

    forecast = forecast.rename(
        columns={
            "TotalValidVotes": "Votes_2016",
            "VoteShare": "VoteShare_2016",
            "WardTotalVotes":
            "WardTotalVotes_2016"
        }
    )

    encoded = pd.get_dummies(
        forecast,
        columns=["Ward"],
        dtype=int
    )

    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    forecast[
        "PredictedVoteShare_2026"
    ] = _model.predict(encoded)

    forecast[
        "PredictedVoteShare_2026"
    ] = forecast[
        "PredictedVoteShare_2026"
    ].clip(lower=0)

    ward_prediction_totals = (
        forecast
        .groupby("Ward")
        ["PredictedVoteShare_2026"]
        .transform("sum")
    )

    forecast[
        "PredictedVoteShare_2026_Normalized"
    ] = (
        forecast["PredictedVoteShare_2026"]
        /
        ward_prediction_totals
    ) * 100

    return forecast


# ============================================================
# TURNOUT
# ============================================================

@st.cache_data
def create_turnout(
    df_2016_pr,
    df_2021_pr
):

    turnout_2016 = (
        df_2016_pr
        .groupby(
            [
                "Ward",
                "VotingDistrict",
                "VotingStationName"
            ],
            as_index=False
        )
        .agg(
            RegisteredVoters=(
                "RegisteredVoters",
                "first"
            ),
            SpoiltVotes=(
                "SpoiltVotes",
                "first"
            ),
            ValidVotes=(
                "TotalValidVotes",
                "sum"
            )
        )
    )

    turnout_2021 = (
        df_2021_pr
        .groupby(
            [
                "Ward",
                "VotingDistrict",
                "VotingStationName"
            ],
            as_index=False
        )
        .agg(
            RegisteredVoters=(
                "RegisteredVoters",
                "first"
            ),
            SpoiltVotes=(
                "SpoiltVotes",
                "first"
            ),
            ValidVotes=(
                "TotalValidVotes",
                "sum"
            )
        )
    )

    turnout_2016["TurnoutRate"] = (
        (
            turnout_2016["ValidVotes"]
            +
            turnout_2016["SpoiltVotes"]
        )
        /
        turnout_2016["RegisteredVoters"]
    ) * 100

    turnout_2021["TurnoutRate"] = (
        (
            turnout_2021["ValidVotes"]
            +
            turnout_2021["SpoiltVotes"]
        )
        /
        turnout_2021["RegisteredVoters"]
    ) * 100

    ward_2016 = (
        turnout_2016
        .groupby("Ward", as_index=False)
        .agg(
            RegisteredVoters_2016=(
                "RegisteredVoters",
                "sum"
            ),
            ValidVotes_2016=(
                "ValidVotes",
                "sum"
            ),
            SpoiltVotes_2016=(
                "SpoiltVotes",
                "sum"
            )
        )
    )

    ward_2021 = (
        turnout_2021
        .groupby("Ward", as_index=False)
        .agg(
            RegisteredVoters_2021=(
                "RegisteredVoters",
                "sum"
            ),
            ValidVotes_2021=(
                "ValidVotes",
                "sum"
            ),
            SpoiltVotes_2021=(
                "SpoiltVotes",
                "sum"
            )
        )
    )

    ward_2016["TurnoutRate_2016"] = (
        (
            ward_2016["ValidVotes_2016"]
            +
            ward_2016["SpoiltVotes_2016"]
        )
        /
        ward_2016["RegisteredVoters_2016"]
    ) * 100

    ward_2021["TurnoutRate_2021"] = (
        (
            ward_2021["ValidVotes_2021"]
            +
            ward_2021["SpoiltVotes_2021"]
        )
        /
        ward_2021["RegisteredVoters_2021"]
    ) * 100

    comparison = ward_2016.merge(
        ward_2021,
        on="Ward",
        how="inner"
    )

    comparison["TurnoutChange"] = (
        comparison["TurnoutRate_2021"]
        -
        comparison["TurnoutRate_2016"]
    )

    return comparison


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        df_2016_pr,
        df_2016_ward,
        df_2021_pr,
        df_2021_ward
    ) = load_data()

    (
        common_wards,
        pr_2016_ward,
        pr_2021_ward,
        ml_data
    ) = prepare_analysis(
        df_2016_pr,
        df_2016_ward,
        df_2021_pr,
        df_2021_ward
    )

    (
        final_model,
        feature_columns,
        X_model,
        y_model,
        loaded_from_file
    ) = load_or_train_model(
        ml_data
    )

    forecast_2026 = create_forecast(
        pr_2021_ward,
        final_model,
        feature_columns
    )

    turnout_comparison = create_turnout(
        df_2016_pr,
        df_2021_pr
    )

except Exception as e:

    st.error(
        "The application could not load the project data."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.20,
    random_state=42
)

test_predictions = final_model.predict(X_test)
train_predictions = final_model.predict(X_train)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

train_r2 = r2_score(
    y_train,
    train_predictions
)


# ============================================================
# OVERALL FORECAST
# ============================================================

overall_2026 = (
    forecast_2026
    .groupby("PartyName")
    ["PredictedVoteShare_2026_Normalized"]
    .mean()
    .sort_values(
        ascending=False
    )
    .reset_index()
)

overall_2026 = overall_2026.rename(
    columns={
        "PredictedVoteShare_2026_Normalized":
        "AveragePredictedVoteShare_2026"
    }
)

average_turnout_2016 = (
    turnout_comparison[
        "TurnoutRate_2016"
    ].mean()
)

average_turnout_2021 = (
    turnout_comparison[
        "TurnoutRate_2021"
    ].mean()
)

turnout_ratio = (
    average_turnout_2021
    /
    average_turnout_2016
)

projected_turnout_2026 = (
    average_turnout_2021
    *
    turnout_ratio
)

total_registered_2021 = (
    turnout_comparison[
        "RegisteredVoters_2021"
    ].sum()
)

estimated_voters_2026 = (
    total_registered_2021
    *
    projected_turnout_2026
    /
    100
)

overall_2026[
    "ProjectedVotes_2026"
] = (
    overall_2026[
        "AveragePredictedVoteShare_2026"
    ]
    /
    overall_2026[
        "AveragePredictedVoteShare_2026"
    ].sum()
) * estimated_voters_2026

overall_2026[
    "ProjectedVotes_2026"
] = (
    overall_2026[
        "ProjectedVotes_2026"
    ]
    .round()
    .astype(int)
)

overall_2026[
    "ProjectedShare_2026"
] = (
    overall_2026[
        "ProjectedVotes_2026"
    ]
    /
    overall_2026[
        "ProjectedVotes_2026"
    ].sum()
) * 100


# ============================================================
# HISTORICAL PARTY TOTALS
# ============================================================

party_2016 = (
    pr_2016_ward
    .groupby("PartyName")[
        "TotalValidVotes"
    ]
    .sum()
    .reset_index()
)

party_2021 = (
    pr_2021_ward
    .groupby("PartyName")[
        "TotalValidVotes"
    ]
    .sum()
    .reset_index()
)

party_history = party_2016.merge(
    party_2021,
    on="PartyName",
    how="outer",
    suffixes=(
        "_2016",
        "_2021"
    )
).fillna(0)

party_history["Change"] = (
    party_history[
        "TotalValidVotes_2021"
    ]
    -
    party_history[
        "TotalValidVotes_2016"
    ]
)

party_history = party_history.sort_values(
    "TotalValidVotes_2021",
    ascending=False
)


# ============================================================
# WARD WINNERS
# ============================================================

winners_2016 = (
    pr_2016_ward.loc[
        pr_2016_ward
        .groupby("Ward")[
            "VoteShare"
        ]
        .idxmax()
    ][
        [
            "Ward",
            "PartyName",
            "VoteShare"
        ]
    ]
    .rename(
        columns={
            "PartyName":
            "Winner_2016",
            "VoteShare":
            "WinnerShare_2016"
        }
    )
)

winners_2021 = (
    pr_2021_ward.loc[
        pr_2021_ward
        .groupby("Ward")[
            "VoteShare"
        ]
        .idxmax()
    ][
        [
            "Ward",
            "PartyName",
            "VoteShare"
        ]
    ]
    .rename(
        columns={
            "PartyName":
            "Winner_2021",
            "VoteShare":
            "WinnerShare_2021"
        }
    )
)

ward_change = (
    pr_2016_ward[
        [
            "Ward",
            "PartyName",
            "VoteShare"
        ]
    ]
    .rename(
        columns={
            "VoteShare":
            "VoteShare_2016"
        }
    )
    .merge(
        pr_2021_ward[
            [
                "Ward",
                "PartyName",
                "VoteShare"
            ]
        ].rename(
            columns={
                "VoteShare":
                "VoteShare_2021"
            }
        ),
        on=[
            "Ward",
            "PartyName"
        ],
        how="outer"
    )
    .fillna(0)
)

ward_change["VoteShareChange"] = (
    ward_change[
        "VoteShare_2021"
    ]
    -
    ward_change[
        "VoteShare_2016"
    ]
)

ward_change_summary = (
    ward_change
    .assign(
        AbsoluteChange=lambda x:
        x["VoteShareChange"].abs()
    )
    .groupby("Ward")
    .agg(
        MaxAbsoluteChange=(
            "AbsoluteChange",
            "max"
        ),
        TotalChange=(
            "VoteShareChange",
            lambda x: x.abs().sum()
        )
    )
    .reset_index()
)

ward_winners_2026 = forecast_2026.loc[
    forecast_2026
    .groupby("Ward")[
        "PredictedVoteShare_2026_Normalized"
    ]
    .idxmax()
][
    [
        "Ward",
        "PartyName",
        "PredictedVoteShare_2026_Normalized"
    ]
].rename(
    columns={
        "PartyName":
        "PredictedWinner_2026"
    }
)

ward_analysis = (
    ward_change_summary
    .merge(
        winners_2016,
        on="Ward",
        how="left"
    )
    .merge(
        winners_2021,
        on="Ward",
        how="left"
    )
    .merge(
        ward_winners_2026,
        on="Ward",
        how="left"
    )
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:1.35rem;
        font-weight:800;
        margin-bottom:0.2rem;
    ">
        📊 ETH INTELLIGENCE
    </div>
    <div style="
        color:#64748b;
        font-size:0.78rem;
        margin-bottom:1.5rem;
    ">
        Election Analytics Platform
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Historical Analysis",
        "2026 Forecast",
        "Ward Intelligence",
        "Turnout",
        "Model Performance",
        "Data & Methodology"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    **Municipality**

    ETH - eThekwini

    **Comparable wards**

    {len(common_wards)}

    **Model**

    Random Forest Regressor

    **Evaluation**

    {test_r2 * 100:.2f}% R²
    """
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Academic forecasting application • 2016–2021 historical data"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            DATA SCIENCE • MACHINE LEARNING • FORECASTING
        </div>

        <div class="hero-title">
            eThekwini Election Intelligence
        </div>

        <div class="hero-subtitle">
            An interactive analytical platform examining historical
            election behaviour in eThekwini Metropolitan Municipality
            and presenting model-generated 2026 projections from
            historical voting and participation patterns.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Key indicators from the complete forecasting pipeline.'
        '</div>',
        unsafe_allow_html=True
    )

    top_party = overall_2026.iloc[0]

    cols = st.columns(5)

    metrics = [
        (
            "MODEL R²",
            f"{test_r2 * 100:.2f}%",
            "Test-set performance"
        ),
        (
            "2016 PR",
            f"{average_turnout_2016:.2f}%",
            "Participation proxy"
        ),
        (
            "2021 PR",
            f"{average_turnout_2021:.2f}%",
            "Participation proxy"
        ),
        (
            "2026 PR",
            f"{projected_turnout_2026:.2f}%",
            "Model-derived estimate"
        ),
        (
            "2026 PARTICIPANTS",
            f"{estimated_voters_2026:,.0f}",
            "Estimated voters"
        )
    ]

    for col, item in zip(cols, metrics):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {item[0]}
                    </div>

                    <div class="metric-value">
                        {item[1]}
                    </div>

                    <div class="metric-description">
                        {item[2]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">'
        '2026 Model-Generated Party Projection'
        '</div>',
        unsafe_allow_html=True
    )

    top10 = overall_2026.head(10)

    fig = px.bar(
        top10,
        x="ProjectedShare_2026",
        y="PartyName",
        orientation="h",
        text="ProjectedShare_2026",
        template="plotly_dark"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=550,
        xaxis_title="Projected Share (%)",
        yaxis_title="",
        yaxis=dict(
            categoryorder="total ascending"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=10,
            r=40,
            t=30,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="info-box">
        <strong>Interpretation</strong><br><br>
        The chart above presents the model's projected 2026
        party-share distribution. These values are model-generated
        estimates based on the historical data and assumptions used
        in this project. They should not be interpreted as actual
        election results.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

elif page == "Historical Analysis":

    st.markdown(
        '<div class="section-title">Historical Election Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Comparison of PR voting patterns across the 110 geographically '
        'comparable wards.'
        '</div>',
        unsafe_allow_html=True
    )

    top_history = party_history.head(10).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_history["PartyName"],
            y=top_history[
                "TotalValidVotes_2016"
            ],
            name="2016"
        )
    )

    fig.add_trace(
        go.Bar(
            x=top_history["PartyName"],
            y=top_history[
                "TotalValidVotes_2021"
            ],
            name="2021"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=550,
        xaxis_tickangle=-35,
        yaxis_title="Valid Votes",
        xaxis_title="Party"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Historical Party Table'
        '</div>',
        unsafe_allow_html=True
    )

    display_history = party_history.copy()

    display_history[
        "TotalValidVotes_2016"
    ] = display_history[
        "TotalValidVotes_2016"
    ].astype(int)

    display_history[
        "TotalValidVotes_2021"
    ] = display_history[
        "TotalValidVotes_2021"
    ].astype(int)

    st.dataframe(
        display_history.head(20),
        use_container_width=True,
        hide_index=True
    )

    csv = display_history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download historical party analysis",
        csv,
        "historical_party_analysis.csv",
        "text/csv"
    )


# ============================================================
# 2026 FORECAST
# ============================================================

elif page == "2026 Forecast":

    st.markdown(
        '<div class="section-title">'
        '2026 Forecast Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore the model-generated 2026 party projections interactively.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        party_list = overall_2026[
            "PartyName"
        ].tolist()

        selected_party = st.selectbox(
            "Select a party",
            party_list
        )

    selected_row = overall_2026[
        overall_2026["PartyName"]
        ==
        selected_party
    ].iloc[0]

    with col2:

        st.markdown(
            f"""
            <div class="success-box">
            <strong>{selected_party}</strong><br>
            Projected share:
            <strong>
            {selected_row["ProjectedShare_2026"]:.2f}%
            </strong><br>
            Projected votes:
            <strong>
            {selected_row["ProjectedVotes_2026"]:,.0f}
            </strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">'
        'Projected Party Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    forecast_chart = overall_2026.head(15)

    fig = px.bar(
        forecast_chart,
        x="PartyName",
        y="ProjectedShare_2026",
        text="ProjectedShare_2026",
        template="plotly_dark"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=600,
        xaxis_tickangle=-40,
        yaxis_title="Projected Share (%)",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Complete Projection'
        '</div>',
        unsafe_allow_html=True
    )

    final_table = overall_2026[
        [
            "PartyName",
            "ProjectedVotes_2026",
            "ProjectedShare_2026"
        ]
    ].copy()

    final_table[
        "ProjectedVotes_2026"
    ] = final_table[
        "ProjectedVotes_2026"
    ].map(
        lambda x: f"{x:,}"
    )

    final_table[
        "ProjectedShare_2026"
    ] = final_table[
        "ProjectedShare_2026"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        final_table,
        use_container_width=True,
        hide_index=True
    )

    csv = overall_2026.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download 2026 projection",
        csv,
        "eThekwini_2026_forecast.csv",
        "text/csv"
    )


# ============================================================
# WARD INTELLIGENCE
# ============================================================

elif page == "Ward Intelligence":

    st.markdown(
        '<div class="section-title">'
        'Ward Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Compare historical ward behaviour with the model-generated '
        '2026 projection.'
        '</div>',
        unsafe_allow_html=True
    )

    selected_ward = st.selectbox(
        "Select ward",
        sorted(common_wards)
    )

    ward_rows = ward_analysis[
        ward_analysis["Ward"]
        ==
        selected_ward
    ]

    forecast_rows = forecast_2026[
        forecast_2026["Ward"]
        ==
        selected_ward
    ].sort_values(
        "PredictedVoteShare_2026_Normalized",
        ascending=False
    )

    if not ward_rows.empty:

        row = ward_rows.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "2016",
                row["Winner_2016"],
                f'{row["WinnerShare_2016"]:.2f}%'
            )

        with c2:
            st.metric(
                "2021",
                row["Winner_2021"],
                f'{row["WinnerShare_2021"]:.2f}%'
            )

        with c3:
            st.metric(
                "2026 Projection",
                row["PredictedWinner_2026"],
                f'{row["PredictedVoteShare_2026_Normalized"]:.2f}%'
            )

        with c4:
            st.metric(
                "Largest Historical Change",
                f'{row["MaxAbsoluteChange"]:.2f} pp'
            )

    st.markdown(
        '<div class="section-title">'
        'Selected Ward: 2026 Party Projection'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        forecast_rows.head(10),
        x="PredictedVoteShare_2026_Normalized",
        y="PartyName",
        orientation="h",
        text="PredictedVoteShare_2026_Normalized",
        template="plotly_dark"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Projected Share (%)",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Three Exam Focus Wards'
        '</div>',
        unsafe_allow_html=True
    )

    selected_three = [
        "Ward 59500028",
        "Ward 59500052",
        "Ward 59500104"
    ]

    three_wards = ward_analysis[
        ward_analysis["Ward"].isin(
            selected_three
        )
    ][
        [
            "Ward",
            "Winner_2016",
            "WinnerShare_2016",
            "Winner_2021",
            "WinnerShare_2021",
            "MaxAbsoluteChange",
            "PredictedWinner_2026",
            "PredictedVoteShare_2026_Normalized"
        ]
    ].copy()

    st.dataframe(
        three_wards,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TURNOUT
# ============================================================

elif page == "Turnout":

    st.markdown(
        '<div class="section-title">'
        'Participation & Turnout Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">
        <strong>Important methodological note</strong><br><br>
        The participation measure used here is a PR-ballot participation
        proxy calculated from valid votes plus spoilt votes divided by
        registered voters. It should not automatically be interpreted
        as total participation across every municipal ballot type.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "2016 PR participation",
            f"{average_turnout_2016:.2f}%"
        )

    with c2:
        st.metric(
            "2021 PR participation",
            f"{average_turnout_2021:.2f}%"
        )

    with c3:
        st.metric(
            "2026 projected PR participation",
            f"{projected_turnout_2026:.2f}%"
        )

    st.markdown(
        '<div class="section-title">'
        'Historical Participation'
        '</div>',
        unsafe_allow_html=True
    )

    turnout_plot = turnout_comparison[
        [
            "Ward",
            "TurnoutRate_2016",
            "TurnoutRate_2021"
        ]
    ].melt(
        id_vars="Ward",
        var_name="Election",
        value_name="Participation"
    )

    turnout_plot["Election"] = (
        turnout_plot["Election"]
        .str.replace(
            "TurnoutRate_",
            ""
        )
    )

    fig = px.box(
        turnout_plot,
        x="Election",
        y="Participation",
        points="outliers",
        template="plotly_dark"
    )

    fig.update_layout(
        height=450,
        yaxis_title="Participation (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Estimated 2026 Participation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                ESTIMATED PARTICIPATING VOTERS
            </div>

            <div class="metric-value">
                {estimated_voters_2026:,.0f}
            </div>

            <div class="metric-description">
                Derived from the 2021 registered-voter base and
                the projected 2026 PR participation rate.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Ward Participation Change'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(
        turnout_comparison,
        x="TurnoutChange",
        nbins=25,
        template="plotly_dark"
    )

    fig.update_layout(
        height=420,
        xaxis_title="Change in participation (percentage points)",
        yaxis_title="Number of wards"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Random Forest Regression used to model party vote share.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Test R²",
            f"{test_r2 * 100:.2f}%"
        )

    with c2:
        st.metric(
            "Test MAE",
            f"{test_mae:.4f}"
        )

    with c3:
        st.metric(
            "Test RMSE",
            f"{test_rmse:.4f}"
        )

    with c4:
        st.metric(
            "Training R²",
            f"{train_r2 * 100:.2f}%"
        )

    st.markdown(
        '<div class="success-box">'
        '<strong>Model evaluation</strong><br><br>'
        f'The model achieved a test-set R² of '
        f'<strong>{test_r2 * 100:.2f}%</strong>. '
        'For this regression target, R², MAE and RMSE are used '
        'as the primary evaluation measures.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Actual vs Predicted Vote Share'
        '</div>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": test_predictions
    })

    fig = px.scatter(
        comparison,
        x="Actual",
        y="Predicted",
        opacity=0.55,
        template="plotly_dark"
    )

    minimum = min(
        comparison["Actual"].min(),
        comparison["Predicted"].min()
    )

    maximum = max(
        comparison["Actual"].max(),
        comparison["Predicted"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect prediction"
        )
    )

    fig.update_layout(
        height=520,
        xaxis_title="Actual vote share",
        yaxis_title="Predicted vote share"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        'Model Configuration'
        '</div>',
        unsafe_allow_html=True
    )

    model_info = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Estimators",
            "Maximum depth",
            "Minimum samples per leaf",
            "Train/Test split",
            "Random state",
            "Features"
        ],
        "Value": [
            "Random Forest Regressor",
            "200",
            "10",
            "2",
            "80% / 20%",
            "42",
            str(len(feature_columns))
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA & METHODOLOGY
# ============================================================

elif page == "Data & Methodology":

    st.markdown(
        '<div class="section-title">'
        'Data & Methodology'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <strong>Historical data</strong><br><br>
        Two election datasets were used: 2016 and 2021 eThekwini
        election records. Each dataset contains PR and Ward ballot
        records together with voting districts, voting stations,
        registered voters, spoilt votes, party names and valid votes.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "2016 records",
            f"{len(df_2016_pr) + len(df_2016_ward):,}"
        )

    with c2:
        st.metric(
            "2021 records",
            f"{len(df_2021_pr) + len(df_2021_ward):,}"
        )

    with c3:
        st.metric(
            "Common wards",
            len(common_wards)
        )

    with c4:
        st.metric(
            "Model features",
            len(feature_columns)
        )

    st.markdown(
        '<div class="section-title">'
        'Geographic Alignment'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The 2016 and 2021 datasets contained 110 wards in common.
        One additional ward appeared in 2021 only, so the 110 common
        wards were used for comparable historical analysis and modelling.
        """
    )

    st.markdown(
        '<div class="section-title">'
        'Data Quality Findings'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">
        <strong>2016 source-data inconsistency</strong><br><br>
        One 2016 station contained valid party votes greater than its
        registered-voter count. The record was retained rather than
        silently deleted because it represents a source-data issue that
        should be documented and investigated rather than hidden.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Forecasting Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        <strong>1. Raw data</strong><br>
        2016 and 2021 eThekwini election datasets

        <br><br>

        <strong>2. Cleaning</strong><br>
        Missing-value checks, duplicate checks, text standardisation,
        numeric conversion and data-quality validation

        <br><br>

        <strong>3. Geographic alignment</strong><br>
        110 common wards were selected for comparable modelling

        <br><br>

        <strong>4. Feature engineering</strong><br>
        Historical votes, vote share and ward-level historical totals

        <br><br>

        <strong>5. Machine learning</strong><br>
        Random Forest Regression

        <br><br>

        <strong>6. Evaluation</strong><br>
        80/20 train-test split using MAE, RMSE and R²

        <br><br>

        <strong>7. Forecast</strong><br>
        Model-generated 2026 vote-share projections

        <br><br>

        <strong>8. Participation estimate</strong><br>
        Historical PR participation used to derive a 2026 participation
        scenario and estimated number of participating voters

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Important Limitations'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">

        • The 2026 figures are model-generated projections, not actual
        election results.

        <br><br>

        • The model is based on only two historical election periods,
        so long-term political and demographic changes are not fully
        represented.

        <br><br>

        • The participation figure is specifically a PR-ballot
        participation proxy.

        <br><br>

        • Geographic boundaries and voting districts can change between
        elections, which is why common wards were used.

        <br><br>

        • Statistical model performance should not be interpreted as a
        guarantee that future real-world results will match the model.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        eThekwini Election Intelligence Dashboard<br>

        Historical analysis • Machine learning • Forecasting •
        Interactive analytics

        <br><br>

        Academic project — projections are model-generated estimates
        and should not be interpreted as actual election results.

    </div>
    """,
    unsafe_allow_html=True
)