from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analyze import fit_evaluate, load_data, review_metrics

st.set_page_config(page_title="Signal & Shield | Card Fraud Analytics", page_icon="◈", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp{background:#0a1018;color:#e9eff7;font-family:'DM Sans',sans-serif}
[data-testid='stSidebar']{background:#101923;border-right:1px solid #243241}
h1,h2,h3{letter-spacing:-.04em}h1{font-size:2.5rem!important}
.eyebrow{color:#51e0b0;font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.17em;text-transform:uppercase}
.subtle{color:#93a4b6;font-size:.92rem}
div[data-testid='stMetric']{background:#111c27;border:1px solid #223344;padding:16px 18px;border-radius:12px}
div[data-testid='stMetricLabel']{color:#93a4b6}div[data-testid='stMetricValue']{color:#f3f7fb}
</style>""", unsafe_allow_html=True)

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

@st.cache_data(show_spinner="Loading 284k transactions…")
def get_data(path: str):
    return load_data(path)

@st.cache_resource(show_spinner="Training chronological fraud baseline…")
def get_analysis(data: pd.DataFrame):
    return fit_evaluate(data)

if not DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Place the supplied creditcard.csv in `data/raw/` and refresh.")
    st.stop()

data = get_data(str(DATA_PATH))
result = get_analysis(data)
holdout = result["test"].copy()
metrics = result["metrics"]

st.markdown('<div class="eyebrow">Risk intelligence / card fraud case study</div>', unsafe_allow_html=True)
st.title("Signal & Shield")
st.markdown('<div class="subtle">A chronological review of anonymized card transactions: rank suspicious activity, size the review queue, and quantify the fraud captured.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Review strategy")
    score_label = st.radio("Ranking approach", ["Logistic risk model", "Amount-only baseline"])
    score_col = "risk_score" if score_label == "Logistic risk model" else "amount_only_score"
    review_share = st.slider("Share of holdout to review", min_value=0.05, max_value=5.0, value=1.0, step=0.05, format="%.2f%%") / 100
    st.caption("The slider represents the share of transactions sent for retrospective review. It is not a calibrated probability cutoff.")

op = review_metrics(holdout, score_col, review_share)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{metrics['total_rows']:,}")
c2.metric("Holdout fraud labels", f"{int(holdout['Class'].sum()):,}", f"{metrics['fraud_rate']:.3%} of holdout")
c3.metric("Review queue", f"{op['alerts']:,}", f"{review_share:.2%} of holdout")
c4.metric("Fraud recall", f"{op['recall']:.1%}")
c5.metric("Fraud amount captured", f"{op['captured_amount']:,.2f}", f"{op['captured_amount_share']:.1%} of holdout fraud amount")
st.caption("The final 20% of elapsed transaction time is held out. Class is used only as the target and never as a model input.")

overview, model_tab, queue_tab = st.tabs(["Exposure overview", "Model & review trade-offs", "Review queue"])

with overview:
    left, right = st.columns([1.25, 1])
    with left:
        timeline = holdout.copy()
        timeline["time_bin"] = (timeline["Time"] // 1800).astype(int)
        grouped = timeline.groupby("time_bin").agg(transactions=("Class", "size"), fraud=("Class", "sum"), fraud_amount=("Amount", lambda x: x[timeline.loc[x.index, "Class"].eq(1)].sum())).reset_index()
        grouped["fraud_rate"] = grouped.fraud / grouped.transactions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grouped.time_bin * .5, y=grouped.transactions, name="Transactions", mode="lines", line=dict(color="#52687f", width=1.5)))
        fig.add_trace(go.Bar(x=grouped.time_bin * .5, y=grouped.fraud, name="Fraud labels", marker_color="#51e0b0", opacity=.82, yaxis="y2"))
        fig.update_layout(title="Holdout activity over elapsed time", xaxis_title="Hours since first recorded transaction", yaxis_title="Transactions", yaxis2=dict(title="Fraud cases", overlaying="y", side="right", showgrid=False), height=360, margin=dict(l=10,r=15,t=55,b=20), paper_bgcolor="#0a1018", plot_bgcolor="#0a1018", font_color="#dce7f1", xaxis=dict(gridcolor="#1c2935"), yaxis=dict(gridcolor="#1c2935"), legend=dict(orientation="h",y=1.14))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        amount_frame = holdout.copy()
        amount_frame["label"] = amount_frame["Class"].map({0:"Legitimate",1:"Fraud"})
        fig = px.histogram(amount_frame, x="Amount", color="label", histnorm="probability density", barmode="overlay", nbins=60, opacity=.72, title="Transaction amount distribution", color_discrete_map={"Legitimate":"#6683a2","Fraud":"#51e0b0"})
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=55,b=20), paper_bgcolor="#0a1018", plot_bgcolor="#0a1018", font_color="#dce7f1", xaxis=dict(gridcolor="#1c2935", title="Amount (dataset units)"), yaxis=dict(gridcolor="#1c2935"), legend=dict(orientation="h",y=1.14))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Holdout snapshot")
    a,b,c = st.columns(3)
    a.metric("Holdout transactions", f"{metrics['test_rows']:,}")
    b.metric("Holdout fraud rate", f"{metrics['fraud_rate']:.3%}")
    c.metric("Fraud amount in holdout", f"{op['total_fraud_amount']:,.2f}", help="Sum of Amount for Class=1 rows; amount units are inherited from the source dataset.")

with model_tab:
    left, right = st.columns([1.2,1])
    with left:
        curve_points=[]
        for col,label in [("risk_score","Logistic risk model"),("amount_only_score","Amount-only baseline")]:
            ordered=holdout.sort_values(col,ascending=False)
            fraud_total=max(1,int(ordered["Class"].sum()))
            for share in np.geomspace(.0005,.20,45):
                n=max(1,int(np.ceil(len(ordered)*share))); top=ordered.head(n)
                tp=int(top["Class"].sum())
                curve_points.append({"Recall":tp/fraud_total,"Precision":tp/n,"Approach":label,"Review share":share})
        curve=pd.DataFrame(curve_points)
        fig=px.line(curve,x="Recall",y="Precision",color="Approach",hover_data={"Review share":":.2%"},title="Precision–recall at different review budgets",color_discrete_map={"Logistic risk model":"#51e0b0","Amount-only baseline":"#6683a2"})
        fig.add_trace(go.Scatter(x=[op["recall"]],y=[op["precision"]],mode="markers",name="Selected operating point",marker=dict(size=12,color="#ffbd69",line=dict(width=1,color="#fff"))))
        fig.update_layout(height=400,margin=dict(l=10,r=10,t=55,b=20),paper_bgcolor="#0a1018",plot_bgcolor="#0a1018",font_color="#dce7f1",xaxis=dict(tickformat=".0%",gridcolor="#1c2935",title="Fraud recall"),yaxis=dict(tickformat=".0%",gridcolor="#1c2935",title="Alert precision"),legend=dict(orientation="h",y=1.15))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("#### Chronological holdout quality")
        st.metric("PR-AUC",f"{metrics['pr_auc']:.4f}",help="Average precision on the final 20% of elapsed time.")
        st.metric("ROC-AUC",f"{metrics['roc_auc']:.4f}")
        st.metric("Training / holdout",f"{metrics['train_rows']:,} / {metrics['test_rows']:,}")
        st.markdown("#### Selected review budget")
        st.write(f"**{score_label}** · {op['alerts']:,} reviewed transactions")
        st.write(f"Precision **{op['precision']:.2%}** · Recall **{op['recall']:.2%}**")
        st.write(f"Fraud caught **{op['true_positives']:,}** · False alerts **{op['false_positives']:,}** · Missed fraud **{op['missed_fraud']:,}")
    st.info("V1–V28 are anonymized transformed features, so their names do not map directly to customer or merchant behaviors. Class-weighting changes score calibration: the review-share control selects a ranked queue instead of treating scores as real-world probabilities.")

with queue_tab:
    st.markdown("#### Highest-ranked holdout transactions")
    queue=op["queue"].copy()
    queue["review_rank"]=np.arange(1,len(queue)+1)
    columns=["review_rank","Time","Amount",*([f"V{i}" for i in range(1,29)]),"Class"]
    display=queue[columns].rename(columns={"Class":"Class (retrospective label)"})
    st.dataframe(display,hide_index=True,use_container_width=True,height=440)
    st.download_button("Download review queue CSV",display.to_csv(index=False).encode(),"fraud_review_queue.csv","text/csv")
    st.caption("Retrospective Class labels are included for evaluation. In an operational review queue, the label would be unavailable at decision time.")

st.divider()
st.markdown('<div class="eyebrow">Data: supplied creditcard.csv · chronological evaluation · anonymized PCA features</div>',unsafe_allow_html=True)
st.caption("Python · pandas · scikit-learn · Streamlit · Plotly. Amounts retain the source dataset units. The holdout is one time period from one dataset, not evidence of production performance.")
