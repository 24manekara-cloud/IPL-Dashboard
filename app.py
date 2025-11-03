import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="🏏 IPL Interactive Dashboard",
    layout="wide",
    page_icon="🏏"
)

# =================== DARK THEME CSS ===================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0f172a, #1e293b);
    color: #f8fafc;
    font-family: "Poppins", sans-serif;
}
h1, h2, h3, h4 {
    color: #38bdf8;
}
div[data-testid="stMetricValue"] {
    color: #22c55e;
}
[data-testid="stMetric"] {
    background-color: rgba(30, 41, 59, 0.6);
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(56,189,248,0.3);
}
hr {
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# =================== TITLE ===================
st.title("🏏 IPL Interactive Analytics Dashboard")
st.markdown("### Local CSV • ML Predictions • Player Comparison")

# =================== FILE UPLOAD ===================
match_file = st.file_uploader("📂 Upload IPL Match Data CSV", type=["csv"])
if match_file is None:
    st.info("Upload a CSV with: match_id, date, team1, team2, venue, runs_team1, runs_team2")
    st.stop()

df = pd.read_csv(match_file, parse_dates=["date"], dayfirst=True)
df.columns = [c.strip().lower() for c in df.columns]

# =================== BASIC VALIDATION ===================
required_cols = ["match_id", "date", "team1", "team2", "venue", "runs_team1", "runs_team2"]
if not set(required_cols).issubset(df.columns):
    st.error(f"Missing columns: {set(required_cols) - set(df.columns)}")
    st.stop()

# =================== ADD CALCULATED FIELDS ===================
df["winner"] = df.apply(lambda r: r["team1"] if r["runs_team1"] > r["runs_team2"] else r["team2"], axis=1)
df["margin"] = (df["runs_team1"] - df["runs_team2"]).abs()

# =================== SIDEBAR ===================
st.sidebar.header("🏟️ Filters")
teams = sorted(set(df["team1"]) | set(df["team2"]))
team = st.sidebar.selectbox("Select Team", teams)

# =================== FILTER DATA ===================
team_df = df[(df.team1 == team) | (df.team2 == team)].copy()
team_df["team_runs"] = team_df.apply(lambda r: r["runs_team1"] if r["team1"] == team else r["runs_team2"], axis=1)
team_df = team_df.sort_values("date")

# =================== TABS ===================
tabs = st.tabs(["📈 Team Performance", "🧩 Player Comparison", "🏅 Top Performers"])

# ========== TAB 1: TEAM PERFORMANCE ==========
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Matches Played", len(team_df))
    col2.metric("Total Runs", team_df["team_runs"].sum())
    col3.metric("Highest Score", team_df["team_runs"].max())

    st.subheader(f"📊 {team} Runs Over Time")
    fig1 = px.line(
        team_df,
        x="date",
        y="team_runs",
        markers=True,
        title=f"{team} Performance Trend",
        color_discrete_sequence=["#38bdf8"]
    )
    fig1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

    # Rolling average
    st.subheader("📈 Rolling Average (Last 5 Matches)")
    team_df["rolling_avg"] = team_df["team_runs"].rolling(5).mean()
    fig2 = px.line(
        team_df,
        x="date",
        y=["team_runs", "rolling_avg"],
        labels={"value": "Runs"},
        color_discrete_map={"team_runs": "#22c55e", "rolling_avg": "#facc15"},
        title=f"{team} - Runs vs Rolling Average"
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # Venue performance
    st.subheader("🏟️ Average Runs by Venue")
    venue_avg = team_df.groupby("venue")["team_runs"].mean().sort_values(ascending=False)
    fig3 = px.bar(
        venue_avg,
        x=venue_avg.index,
        y=venue_avg.values,
        color=venue_avg.values,
        color_continuous_scale="Blues",
        title=f"{team} Average Runs per Venue"
    )
    fig3.update_layout(template="plotly_dark", xaxis_title="Venue", yaxis_title="Average Runs")
    st.plotly_chart(fig3, use_container_width=True)

    # Outliers
    st.subheader("🚨 Outlier Matches")
    z = np.abs(stats.zscore(team_df["team_runs"]))
    outliers = team_df[z > 2]
    if not outliers.empty:
        st.warning(f"{len(outliers)} outlier matches found!")
        st.dataframe(outliers[["date", "venue", "team_runs"]])
    else:
        st.info("No outliers detected.")

    # ML Prediction
    if len(team_df) >= 5:
        X = np.arange(len(team_df)).reshape(-1, 1)
        y = team_df["team_runs"]
        model = LinearRegression().fit(X, y)
        prediction = model.predict([[len(team_df)]])[0]
        st.success(f"🤖 Predicted Next Match Runs: {prediction:.1f}")
    else:
        st.info("Need at least 5 matches for prediction.")

# ========== TAB 2: PLAYER COMPARISON ==========
with tabs[1]:
    st.header("🧩 Player Comparison")

    player_file = st.file_uploader("📂 Upload Player Stats CSV", type=["csv"], key="p_upload")
    if player_file:
        players = pd.read_csv(player_file)
        required = {"player_name", "team", "runs", "wickets", "strike_rate", "economy"}
        if not required.issubset(players.columns):
            st.error(f"Missing columns: {required - set(players.columns)}")
        else:
            p1, p2 = st.columns(2)
            player1 = p1.selectbox("Select Player 1", sorted(players["player_name"].unique()))
            player2 = p2.selectbox("Select Player 2", sorted(players["player_name"].unique()))

            df1 = players[players["player_name"] == player1].iloc[0]
            df2 = players[players["player_name"] == player2].iloc[0]

            compare = pd.DataFrame({
                "Metric": ["Runs", "Wickets", "Strike Rate", "Economy"],
                player1: [df1["runs"], df1["wickets"], df1["strike_rate"], df1["economy"]],
                player2: [df2["runs"], df2["wickets"], df2["strike_rate"], df2["economy"]],
            })

            st.subheader(f"⚔️ {player1} vs {player2}")
            fig = px.bar(
                compare,
                x="Metric",
                y=[player1, player2],
                barmode="group",
                color_discrete_sequence=["#38bdf8", "#22c55e"]
            )
            fig.update_layout(template="plotly_dark", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a CSV with player stats.")

# ========== TAB 3: TOP PERFORMERS ==========
with tabs[2]:
    st.header("🏅 Top Performer Analysis")
    if player_file:
        players["bat_score"] = (
            (players["runs"] / players["runs"].max()) * 0.6 +
            (players["strike_rate"] / players["strike_rate"].max()) * 0.4
        )
        players["bowl_score"] = (
            (players["wickets"] / players["wickets"].max()) * 0.7 +
            ((players["economy"].min() / players["economy"]) * 0.3)
        )
        players["performance_index"] = (players["bat_score"] + players["bowl_score"]) / 2

        top_players = players.sort_values("performance_index", ascending=False).head(5)
        st.dataframe(
            top_players[["player_name", "team", "runs", "wickets", "strike_rate", "economy", "performance_index"]]
        )

        fig_top = px.bar(
            top_players,
            x="player_name",
            y="performance_index",
            color="team",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="🔥 Top 5 Performers"
        )
        fig_top.update_layout(template="plotly_dark")
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("Upload a player stats CSV to view top performers.")
