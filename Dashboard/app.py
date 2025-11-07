import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Hierarchical MARL Smart Grid Dashboard")

# ---------------------------
# Helper functions
# ---------------------------
def parse_signal(x):
    """Parse agg_signal which may be stringified list or actual list/array."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [np.nan, np.nan]
    elif isinstance(x, (list, np.ndarray)):
        return x
    else:
        return [np.nan, np.nan]

def contiguous_ranges_from_bool_series(df_ts, bool_col='dr_event', time_col='time_step'):
    """
    Given a dataframe with ordered time_col and a boolean column,
    return list of (start, end) ranges where bool_col == True contiguously.
    """
    ranges = []
    if df_ts.empty:
        return ranges
    series = df_ts[[time_col, bool_col]].sort_values(time_col)
    in_range = False
    start = None
    prev_t = None
    for _, row in series.iterrows():
        t = row[time_col]
        v = bool(row[bool_col])
        if v and not in_range:
            in_range = True
            start = t
            prev_t = t
        elif v and in_range:
            prev_t = t
        elif not v and in_range:
            ranges.append((start, prev_t))
            in_range = False
            start = None
            prev_t = None
    if in_range and start is not None:
        ranges.append((start, prev_t))
    return ranges

def compute_baseline_per_consumer(df, consumer_id, dr_flag_col='dr_event', demand_col='net_electricity_consumption'):
    """
    Baseline = mean consumption for this consumer during non-DR timesteps in the same episode.
    Fallback: mean consumption across all timesteps for the consumer in episode.
    """
    sub = df[df['consumer_id'] == consumer_id]
    non_dr = sub[sub[dr_flag_col] == 0]
    if not non_dr.empty:
        base = non_dr[demand_col].mean()
        if np.isfinite(base):
            return base
    # fallback to overall mean for consumer
    if not sub.empty:
        base = sub[demand_col].mean()
        if np.isfinite(base):
            return base
    # final fallback
    return 0.0

def recommended_action_for_row(row, soc_threshold=0.1):
    """
    Return a short recommendation string and numeric recommended action.
    Simple rule:
      - If SOC below threshold -> Hold
      - If DR requested (agg_reduction_target > 0):
          -> If SOC ok -> Discharge (-1) to meet reduction
          -> If SOC low -> Try partial reduction / Hold
      - Else (no DR): Charge (+1) if SOC below 0.95 else Hold
    """
    if pd.isna(row.get('agg_reduction_target', np.nan)):
        return "Hold", 0
    if row.get('soc_after_action', 1) < soc_threshold:
        return "Hold (Low SOC)", 0
    if row['agg_reduction_target'] > 0:
        # prefer discharge to meet reduction
        return f"Discharge (-1) to try reduce {row['agg_reduction_target']:.3f} kW", -1
    else:
        # no DR, prefer charge if capacity
        if row.get('soc_after_action', 0) < 0.95:
            return "Charge (+1)", 1
        else:
            return "Hold", 0

# ---------------------------
# Page sidebar controls
# ---------------------------
st.sidebar.title("Settings & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV (sample.csv format)", type=['csv'])
use_uploaded = uploaded_file is not None

price_thr_slider = st.sidebar.slider("Optional price signal threshold (use 0 to disable)", 0.0, 5.0, 0.0, step=0.01)
incentive_thr_slider = st.sidebar.slider("Optional incentive signal threshold (use 0 to disable)", 0.0, 5.0, 0.0, step=0.01)
soc_safety_threshold = st.sidebar.slider("SOC safety threshold (avoid discharging below)", 0.0, 0.5, 0.1, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**DR detection logic**: DR is considered active if `agg_reduction_target` > 0 **or** signals exceed threshold sliders (if set).")
st.sidebar.markdown("Adjust thresholds to suit your signal scale.")

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

try:
    if use_uploaded:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_csv_from_path('sample.csv')
except FileNotFoundError:
    st.error("sample.csv not found. Upload a CSV or place sample.csv in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# ---------------------------
# Validate columns
# ---------------------------
required_cols = ['episode', 'time_step', 'consumer_id', 'consumer_type',
                 'total_demand', 'net_electricity_consumption',
                 'soc_after_action', 'agg_signal', 'agg_reduction_target',
                 'action', 'reward']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"Missing columns (required): {', '.join(missing)}. Some features may not work.")
    # but continue — code will try to be robust

# ---------------------------
# Parse agg_signal
# ---------------------------
# create price_signal and incentive_signal (best-effort)
if 'agg_signal' in df.columns:
    df['agg_signal_parsed'] = df['agg_signal'].apply(parse_signal)
    # ensure length >= 2
    parsed = pd.DataFrame(df['agg_signal_parsed'].tolist(), index=df.index)
    # handle if parsed has only one column etc.
    if parsed.shape[1] >= 2:
        df['price_signal'] = pd.to_numeric(parsed[0], errors='coerce').fillna(0.0)
        df['incentive_signal'] = pd.to_numeric(parsed[1], errors='coerce').fillna(0.0)
    elif parsed.shape[1] == 1:
        df['price_signal'] = pd.to_numeric(parsed[0], errors='coerce').fillna(0.0)
        df['incentive_signal'] = 0.0
    else:
        df['price_signal'] = 0.0
        df['incentive_signal'] = 0.0
else:
    df['price_signal'] = 0.0
    df['incentive_signal'] = 0.0

# Ensure numeric types for important columns
for c in ['total_demand', 'net_electricity_consumption', 'soc_after_action', 'agg_reduction_target', 'action', 'reward']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------------------------
# Sidebar: select episode (if present)
# ---------------------------
st.title("🤖 Hierarchical MARL Smart Grid: Performance Analysis Dashboard")
if 'episode' in df.columns:
    episodes = sorted(df['episode'].dropna().unique().tolist())
    if episodes:
        selected_episode = st.sidebar.selectbox("Select Episode", episodes, index=len(episodes)-1)
        df = df[df['episode'] == selected_episode].copy()
else:
    selected_episode = None

# ensure time_step sorted numeric
if 'time_step' in df.columns:
    df = df.sort_values('time_step').reset_index(drop=True)
else:
    st.warning("No time_step column found — visualizations may be incorrect.")

st.markdown("---")

# ---------------------------
# Compute DR event flag
# ---------------------------
# Primary DR trigger: agg_reduction_target > 0 (preferred)
# --- SAFE HANDLING for agg_reduction_target ---
if 'agg_reduction_target' in df.columns:
    # convert to numeric if column exists
    df['agg_reduction_target'] = pd.to_numeric(df['agg_reduction_target'], errors='coerce').fillna(0.0)
else:
    # if column missing, create a new column with zeros
    df['agg_reduction_target'] = pd.Series(0.0, index=df.index)


# optional signal-based DR triggers
price_thr = price_thr_slider if price_thr_slider > 0 else None
incentive_thr = incentive_thr_slider if incentive_thr_slider > 0 else None

def detect_dr(row):
    if row.get('agg_reduction_target', 0) > 0:
        return 1
    if price_thr is not None and row.get('price_signal', 0) >= price_thr:
        return 1
    if incentive_thr is not None and row.get('incentive_signal', 0) >= incentive_thr:
        return 1
    return 0

df['dr_event'] = df.apply(detect_dr, axis=1)

# ---------------------------
# System-wide statistics & plots
# ---------------------------
st.header("1️⃣ System Overview")

# Total electricity consumption: show system total_demand over time
if 'time_step' in df.columns and 'total_demand' in df.columns:
    system_demand = df.groupby('time_step', as_index=False)['total_demand'].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Grid Demand", f"{system_demand['total_demand'].mean():.2f} kW")
    c2.metric("Peak Demand", f"{system_demand['total_demand'].max():.2f} kW")
    c3.metric("Avg Reward (all agents)", f"{df['reward'].mean():.3f}")

    fig = px.line(system_demand, x='time_step', y='total_demand', title="Total System Demand (kW)")
    # highlight DR periods on the system plot
    ranges = contiguous_ranges_from_bool_series(df[['time_step','dr_event']].drop_duplicates().sort_values('time_step'),
                                                bool_col='dr_event', time_col='time_step')
    for (s, e) in ranges:
        fig.add_vrect(x0=s, x1=e, fillcolor="LightSalmon", opacity=0.3, line_width=0)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No total_demand or time_step column for system-level plot.")

st.markdown("---")

# ---------------------------
# Signals overview
# ---------------------------
st.header("2️⃣ Aggregator Signals & DR Timeline")
if 'time_step' in df.columns:
    signals_cols = []
    if 'price_signal' in df.columns:
        signals_cols.append('price_signal')
    if 'incentive_signal' in df.columns:
        signals_cols.append('incentive_signal')
    if signals_cols:
        fig2 = px.line(df, x='time_step', y=signals_cols, title="Aggregator Signals Over Time",
                       labels={'value':'Signal','variable':'Signal Type'})
        # shade DR windows
        for (s,e) in contiguous_ranges_from_bool_series(df[['time_step','dr_event']].drop_duplicates().sort_values('time_step'),
                                                        bool_col='dr_event', time_col='time_step'):
            fig2.add_vrect(x0=s, x1=e, fillcolor="LightGreen", opacity=0.25, line_width=0)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No price/incentive signals parsed.")
else:
    st.info("No time_step column to show signals timeline.")

st.markdown("---")

# ---------------------------
# Episode-wide DR summary per consumer
# ---------------------------
st.header("3️⃣ Episode-wide DR Summary (per consumer)")

# compute baseline per consumer (non-DR mean) and actual reduction per row
consumer_ids = df['consumer_id'].unique().tolist() if 'consumer_id' in df.columns else []
baseline_map = {}
for cid in consumer_ids:
    baseline_map[cid] = compute_baseline_per_consumer(df, cid, dr_flag_col='dr_event',
                                                     demand_col='net_electricity_consumption')

# compute actual_reduction and response_gap
def compute_reductions(row):
    cid = row.get('consumer_id')
    base = baseline_map.get(cid, 0.0)
    actual = row.get('net_electricity_consumption', 0.0)
    actual_reduction = base - actual  # positive means reduced consumption relative to baseline
    # clamp small negatives to zero to avoid tiny floating noise
    if actual_reduction < 0 and abs(actual_reduction) < 1e-6:
        actual_reduction = 0.0
    target = row.get('agg_reduction_target', 0.0)
    response_gap = target - actual_reduction  # positive => shortfall, negative => over-fulfilled
    return pd.Series({
        'baseline_consumption': base,
        'actual_reduction': actual_reduction,
        'response_gap': response_gap
    })

if 'net_electricity_consumption' in df.columns:
    reductions = df.apply(compute_reductions, axis=1)
    df = pd.concat([df, reductions], axis=1)
else:
    df['baseline_consumption'] = 0.0
    df['actual_reduction'] = 0.0
    df['response_gap'] = 0.0

# Summarize per consumer
summary_cols = []
if consumer_ids:
    dr_summary_all = df.groupby('consumer_id').agg(
        dr_events=('dr_event', 'sum'),
        total_timesteps=('time_step', 'count'),
        avg_target=('agg_reduction_target', 'mean'),
        avg_actual_reduction=('actual_reduction', 'mean'),
        avg_response_gap=('response_gap', 'mean'),
        pct_timesteps_dr=('dr_event', lambda x: f"{x.sum()/len(x)*100:.1f}%"),
        avg_reward=('reward', 'mean'),
        mean_soc=('soc_after_action', 'mean')
    ).reset_index()
    st.dataframe(dr_summary_all, use_container_width=True)
else:
    st.info("No consumer_id column found to compute per-consumer summary.")

st.markdown("""
**Legend:**  
- `avg_target` = average aggregator reduction request (kW) for the consumer in this episode.  
- `avg_actual_reduction` = average achieved reduction relative to baseline (kW).  
- `avg_response_gap` = avg(target - actual). Positive = consumer on average fell short of the requested reduction.
""")
st.markdown("---")

# ---------------------------
# Consumer-level detailed analysis & recommendation
# ---------------------------
st.header("4️⃣ Individual Consumer Analysis & Recommendations")

if consumer_ids:
    selected_consumer = st.selectbox("Select consumer", consumer_ids)
    df_cons = df[df['consumer_id'] == selected_consumer].copy().reset_index(drop=True)

    st.subheader(f"Consumer {selected_consumer} — Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Reward", f"{df_cons['reward'].mean():.3f}")
    c2.metric("Avg SOC", f"{df_cons['soc_after_action'].mean():.2f}")
    c3.metric("Avg Actual Reduction (kW)", f"{df_cons['actual_reduction'].mean():.3f}")

    # consumption time-series
    if 'time_step' in df_cons.columns and 'net_electricity_consumption' in df_cons.columns:
        fig_c = px.line(df_cons, x='time_step', y='net_electricity_consumption',
                        title=f"Net Electricity Consumption — {selected_consumer}")
        # shade DR windows
        ranges = contiguous_ranges_from_bool_series(df_cons[['time_step','dr_event']].drop_duplicates().sort_values('time_step'),
                                                    bool_col='dr_event', time_col='time_step')
        for (s,e) in ranges:
            fig_c.add_vrect(x0=s, x1=e, fillcolor="LightSalmon", opacity=0.25, line_width=0)
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No per-consumer consumption/time_step data to plot.")

    # Action vs Price scatter with incentive bubble
    if all(col in df_cons.columns for col in ['price_signal','action','computed_incentive'] ) is False:
        # compute computed_incentive if not present (same rule as earlier)
        df_cons['computed_incentive'] = np.where(
            df_cons['action'] < 0,
            np.abs(df_cons['incentive_signal'] * 0.8),
            np.abs(df_cons['price_signal'] * 0.4)
        )
    fig_sc = px.scatter(df_cons, x='price_signal', y='action',
                        size='computed_incentive' if 'computed_incentive' in df_cons.columns else None,
                        color=df_cons['dr_event'].map({1:'DR active',0:'Normal'}),
                        hover_data=['time_step','reward','soc_after_action','agg_reduction_target','actual_reduction'],
                        title="Action vs Price Signal (bubble = incentive magnitude; color = DR active)")
    st.plotly_chart(fig_sc, use_container_width=True)

    # DR tracking table for selected consumer
    st.subheader("DR Periods & Compliance (selected consumer)")
    if not df_cons.empty:
        # compute recommended action per row
        recs = df_cons.apply(lambda r: recommended_action_for_row(r, soc_threshold=soc_safety_threshold), axis=1)
        df_cons[['recommendation_text','recommendation_numeric']] = pd.DataFrame(recs.tolist(), index=df_cons.index)

        display_cols = ['time_step', 'dr_event', 'agg_reduction_target', 'baseline_consumption',
                        'net_electricity_consumption', 'actual_reduction', 'response_gap',
                        'action', 'reward', 'soc_after_action', 'recommendation_text']
        existing_cols = [c for c in display_cols if c in df_cons.columns]
        st.dataframe(df_cons[existing_cols].sort_values('time_step'), use_container_width=True)

        # Summarize selected consumer performance in DR vs Normal
        by_dr = df_cons.groupby('dr_event').agg(
            avg_target=('agg_reduction_target','mean'),
            avg_actual_red=('actual_reduction','mean'),
            avg_gap=('response_gap','mean'),
            avg_reward=('reward','mean'),
            count=('time_step','count')
        ).reset_index()
        by_dr['dr_event'] = by_dr['dr_event'].map({0:'Normal', 1:'DR'})
        st.table(by_dr)

        # Recommendation summary
        st.subheader("Recommended Policy Summary (selected consumer)")
        # If DR occurs at least some times, recommend a DR-aware policy
        total_dr = int(df_cons['dr_event'].sum())
        if total_dr > 0:
            avg_target = df_cons.loc[df_cons['dr_event']==1, 'agg_reduction_target'].mean()
            avg_gap = df_cons.loc[df_cons['dr_event']==1, 'response_gap'].mean()
            rec_action = df_cons.loc[df_cons['dr_event']==1].apply(lambda r: recommended_action_for_row(r, soc_threshold=soc_safety_threshold)[0], axis=1)
            # Most common recommended action during DR
            most_common_rec = rec_action.mode().iloc[0] if not rec_action.mode().empty else "Discharge (-1)"
            st.markdown(f"""
            - DR was active for **{total_dr}** timesteps for this consumer.  
            - Avg aggregator reduction request during DR: **{avg_target:.3f} kW**.  
            - Avg response gap (target - actual): **{avg_gap:.3f} kW** (positive = shortfall).  
            - **Recommended action during DR:** **{most_common_rec}**  
            """)
        else:
            st.markdown("No DR events detected for this consumer in the selected episode. Recommended default policy: charge when SOC < 0.95, otherwise hold.")

    else:
        st.info("No records for the selected consumer.")

    st.markdown("---")

    # Bulk recommendations for all consumers (table)
    st.header("5️⃣ Per-Consumer Recommended Action Summary (episode)")
    recs_all = df.apply(lambda r: recommended_action_for_row(r, soc_threshold=soc_safety_threshold)[0], axis=1)
    df['recommendation_text'] = recs_all
    per_cons_rec = df.groupby('consumer_id').agg(
        avg_target=('agg_reduction_target','mean'),
        avg_actual_reduction=('actual_reduction','mean'),
        avg_gap=('response_gap','mean'),
        pct_dr=('dr_event', lambda x: f"{x.sum()/len(x)*100:.1f}%"),
        most_common_rec=('recommendation_text', lambda x: x.mode().iloc[0] if not x.mode().empty else "Hold")
    ).reset_index()
    st.dataframe(per_cons_rec, use_container_width=True)
else:
    st.info("No consumers (consumer_id) found in data to analyze.")

st.markdown("---")
st.success("✅ Dashboard updated — showing total consumption, DR activation, per-consumer target vs actual, and recommended optimal actions.")
st.markdown("If you want, I can: (a) adapt DR baseline calculation, (b) add export (CSV) of recommendations, or (c) add per-consumer interactive sliders for custom targets.")
