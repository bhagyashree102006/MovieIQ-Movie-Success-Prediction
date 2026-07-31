# ==========================================================
# MovieIQ - Professional Movie Analytics Dashboard
# Internship Project
# Part 1
# ==========================================================

# ----------------------------
# IMPORT LIBRARIES
# ----------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import ast
import os
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:1rem;
}

h1{
    color:#00E5FF;
    text-align:center;
    font-weight:bold;
}

h2,h3{
    color:white;
}

div[data-testid="metric-container"]{
    background:#1B1F2A;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 0px 12px rgba(0,229,255,0.15);
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_movies.csv")

df = load_data()

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")

model = load_model()

# ----------------------------
# EXTRACT GENRE
# ----------------------------

def extract_genre(x):

    try:

        genres = ast.literal_eval(x)

        if len(genres) > 0:
            return genres[0]["name"]

        return "Unknown"

    except:
        return "Unknown"

df["genre"] = df["genres"].apply(extract_genre)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("🎬 MovieIQ")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(

    "",

    [

        "🏠 Dashboard",

        "📊 Dataset",

        "📈 EDA",

        "📉 Statistical Tests",

        "🤖 Prediction",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

# ----------------------------
# FILTERS
# ----------------------------

st.sidebar.subheader("🎯 Filters")

genre_list = sorted(df["genre"].unique())

selected_genre = st.sidebar.selectbox(

    "Genre",

    ["All"] + genre_list

)

minimum_vote = st.sidebar.slider(

    "Minimum Vote Average",

    float(df["vote_average"].min()),

    float(df["vote_average"].max()),

    5.0

)

filtered_df = df.copy()

if selected_genre != "All":

    filtered_df = filtered_df[
        filtered_df["genre"] == selected_genre
    ]

filtered_df = filtered_df[
    filtered_df["vote_average"] >= minimum_vote
]

# ----------------------------
# PAGE TITLE
# ----------------------------

st.title("🎬 MovieIQ")

st.markdown(
"""
### Predictive Analytics on Film Success

Professional Machine Learning Dashboard
"""
)
# ==========================================================
# PART 2 - DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    # ----------------------------
    # Dashboard Title
    # ----------------------------

    st.header("📊 Dashboard Overview")

    st.write("Welcome to the MovieIQ Analytics Dashboard.")

    # ----------------------------
    # Calculate KPIs
    # ----------------------------

    total_movies = len(filtered_df)

    successful_movies = filtered_df["success"].sum()

    success_rate = (successful_movies / total_movies) * 100 if total_movies > 0 else 0

    avg_budget = filtered_df["budget"].mean()

    avg_revenue = filtered_df["revenue"].mean()

    avg_rating = filtered_df["vote_average"].mean()

    avg_popularity = filtered_df["popularity"].mean()

    # ----------------------------
    # KPI Cards
    # ----------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🎬 Total Movies",
        f"{total_movies:,}"
    )

    col2.metric(
        "✅ Success Rate",
        f"{success_rate:.1f}%"
    )

    col3.metric(
        "⭐ Average Rating",
        f"{avg_rating:.2f}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "💰 Avg Budget",
        f"${avg_budget:,.0f}"
    )

    col5.metric(
        "💵 Avg Revenue",
        f"${avg_revenue:,.0f}"
    )

    col6.metric(
        "🔥 Avg Popularity",
        f"{avg_popularity:.2f}"
    )

    st.markdown("---")

    # =====================================================
    # Budget vs Revenue
    # =====================================================

    st.subheader("💰 Budget vs Revenue")

    fig = px.scatter(
        filtered_df,
        x="budget",
        y="revenue",
        color="success",
        size="popularity",
        hover_name="title",
        title="Budget vs Revenue",
        template="plotly_dark"
    )

    fig.update_layout(height=550)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Row 2 Charts
    # =====================================================

    left, right = st.columns(2)

    # ----------------------------
    # Success Distribution
    # ----------------------------

    with left:

        st.subheader("🎯 Success Distribution")

        success_counts = (
            filtered_df["success"]
            .map({1: "Successful", 0: "Not Successful"})
            .value_counts()
        )

        fig = px.pie(
            values=success_counts.values,
            names=success_counts.index,
            hole=0.45,
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------
    # Top Genres
    # ----------------------------

    with right:

        st.subheader("🎭 Genre Distribution")

        genre_count = filtered_df["genre"].value_counts()

        fig = px.bar(
            x=genre_count.index,
            y=genre_count.values,
            color=genre_count.values,
            labels={
                "x": "Genre",
                "y": "Movies"
            },
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # Row 3 Charts
    # =====================================================

    left, right = st.columns(2)

    # ----------------------------
    # Revenue Histogram
    # ----------------------------

    with left:

        st.subheader("💵 Revenue Distribution")

        fig = px.histogram(
            filtered_df,
            x="revenue",
            nbins=30,
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------
    # Popularity Histogram
    # ----------------------------

    with right:

        st.subheader("🔥 Popularity Distribution")

        fig = px.histogram(
            filtered_df,
            x="popularity",
            nbins=25,
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        # ==========================================================
# PART 3 - DATASET EXPLORER
# ==========================================================

elif page == "📊 Dataset":

    st.header("📋 Movie Dataset Explorer")

    st.markdown(
        """
        Explore the filtered movie dataset.
        Search movies, view statistics, and download the filtered data.
        """
    )

    st.markdown("---")

    # -----------------------------------------
    # Search Box
    # -----------------------------------------

    search = st.text_input(
        "🔍 Search Movie Title"
    )

    dataset = filtered_df.copy()

    if search:

        dataset = dataset[
            dataset["title"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # -----------------------------------------
    # Dataset Statistics
    # -----------------------------------------

    st.subheader("📊 Dataset Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Movies",
        len(dataset)
    )

    c2.metric(
        "Genres",
        dataset["genre"].nunique()
    )

    c3.metric(
        "Average Runtime",
        f"{dataset['runtime'].mean():.1f} min"
    )

    c4.metric(
        "Average Rating",
        f"{dataset['vote_average'].mean():.2f}"
    )

    st.markdown("---")

    # -----------------------------------------
    # Dataset Preview
    # -----------------------------------------

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        dataset,
        use_container_width=True,
        height=500
    )

    st.markdown("---")

    # -----------------------------------------
    # Missing Values
    # -----------------------------------------

    st.subheader("❗ Missing Values")

    missing = pd.DataFrame({

        "Column": dataset.columns,

        "Missing Values": dataset.isnull().sum().values

    })

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------
    # Summary Statistics
    # -----------------------------------------

    st.subheader("📈 Numerical Summary")

    st.dataframe(
        dataset.describe(),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------
    # Download Button
    # -----------------------------------------

    st.download_button(

        label="⬇ Download Filtered Dataset",

        data=dataset.to_csv(index=False),

        file_name="filtered_movies.csv",

        mime="text/csv"

    )

    st.success("Dataset Ready for Download ✅")
    # ==========================================================
# PART 4 - EDA PAGE
# ==========================================================

elif page == "📈 EDA":

    st.header("📈 Exploratory Data Analysis")

    st.write("Analyze movie trends using interactive visualizations.")

    st.markdown("---")

    # -------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------

    st.subheader("📊 Correlation Heatmap")

    corr = filtered_df[
        [
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "success"
        ]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Between Numerical Features"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Higher correlation values indicate stronger relationships between variables."
    )

    st.markdown("---")

    # -------------------------------------------------
    # Runtime Distribution
    # -------------------------------------------------

    st.subheader("⏱ Runtime Distribution")

    fig = px.histogram(
        filtered_df,
        x="runtime",
        nbins=25,
        color="success",
        template="plotly_dark",
        title="Movie Runtime Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This chart shows how movie runtimes are distributed."
    )

    st.markdown("---")

    # -------------------------------------------------
    # Vote Average Distribution
    # -------------------------------------------------

    st.subheader("⭐ Vote Average Distribution")

    fig = px.box(
        filtered_df,
        y="vote_average",
        color="success",
        template="plotly_dark",
        title="Vote Average by Success"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Successful movies generally tend to receive higher ratings."
    )

    st.markdown("---")

    # -------------------------------------------------
    # Popularity vs Rating
    # -------------------------------------------------

    st.subheader("🔥 Popularity vs Rating")

    fig = px.scatter(
        filtered_df,
        x="popularity",
        y="vote_average",
        color="success",
        size="budget",
        hover_name="title",
        template="plotly_dark",
        title="Popularity vs Vote Average"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Highly popular movies often receive higher ratings, though there are exceptions."
    )

    st.markdown("---")

    # -------------------------------------------------
    # Budget vs Revenue
    # -------------------------------------------------

    st.subheader("💰 Budget vs Revenue")

    fig = px.scatter(
        filtered_df,
        x="budget",
        y="revenue",
        color="genre",
        size="popularity",
        hover_name="title",
        template="plotly_dark",
        title="Budget vs Revenue by Genre"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Movies with larger budgets often generate higher revenue, but this is not always guaranteed."
    )

    st.markdown("---")

    # -------------------------------------------------
    # Top Genres
    # -------------------------------------------------

    st.subheader("🎭 Genre Distribution")

    genre_count = filtered_df["genre"].value_counts()

    fig = px.bar(
        x=genre_count.index,
        y=genre_count.values,
        color=genre_count.values,
        labels={
            "x": "Genre",
            "y": "Number of Movies"
        },
        title="Movies by Genre",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("EDA Completed Successfully ✅")
    # ==========================================================
# PART 5 - STATISTICAL TESTS
# ==========================================================

elif page == "📉 Statistical Tests":

    st.header("📉 Statistical Analysis")

    st.markdown("""
    This section performs statistical tests to analyze
    relationships between movie features and movie success.
    """)

    st.markdown("---")

    # ============================================
    # Split Data
    # ============================================

    success_movies = filtered_df[filtered_df["success"] == 1]

    failure_movies = filtered_df[filtered_df["success"] == 0]

    # ============================================
    # T-Test
    # ============================================

    st.subheader("📊 Independent T-Test")

    t_stat, p_value = ttest_ind(
        success_movies["vote_average"],
        failure_movies["vote_average"],
        equal_var=False
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "T Statistic",
        f"{t_stat:.4f}"
    )

    col2.metric(
        "P Value",
        f"{p_value:.6f}"
    )

    if p_value < 0.05:

        st.success(
            "✅ Since p-value < 0.05, there is a statistically significant difference in vote averages."
        )

    else:

        st.warning(
            "⚠ Since p-value ≥ 0.05, there is no statistically significant difference."
        )

    st.markdown("---")

    # ============================================
    # Chi-Square Test
    # ============================================

    st.subheader("📈 Chi-Square Test")

    contingency_table = pd.crosstab(
        filtered_df["genre"],
        filtered_df["success"]
    )

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Chi-Square",
        f"{chi2:.3f}"
    )

    col2.metric(
        "Degrees of Freedom",
        dof
    )

    col3.metric(
        "P Value",
        f"{p:.6f}"
    )

    if p < 0.05:

        st.success(
            "✅ Genre and movie success are statistically associated."
        )

    else:

        st.warning(
            "⚠ No significant association between genre and success."
        )

    st.markdown("---")

    # ============================================
    # Correlation Table
    # ============================================

    st.subheader("📋 Correlation Matrix")

    corr = filtered_df[
        [
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "success"
        ]
    ].corr()

    st.dataframe(
        corr.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # Summary
    # ============================================

    st.subheader("📌 Statistical Summary")

    st.info(f"""
**Independent T-Test**

• T Statistic : {t_stat:.4f}

• P Value : {p_value:.6f}

-------------------------------------------------

**Chi-Square Test**

• Chi Square : {chi2:.3f}

• Degrees of Freedom : {dof}

• P Value : {p:.6f}
""")

    st.success("Statistical Analysis Completed Successfully ✅")
    # ==========================================================
# PART 6 - MOVIE SUCCESS PREDICTION
# ==========================================================

elif page == "🤖 Prediction":

    st.header("🤖 Movie Success Prediction")

    st.write(
        "Enter the movie details below to predict whether the movie is likely to be successful."
    )

    st.markdown("---")

    # ===========================================
    # Input Columns
    # ===========================================

    left, right = st.columns(2)

    with left:

        budget = st.number_input(
            "💰 Budget ($)",
            min_value=0.0,
            value=100000000.0,
            step=1000000.0
        )

        runtime = st.number_input(
            "🎬 Runtime (Minutes)",
            min_value=30,
            max_value=300,
            value=120
        )

    with right:

        popularity = st.number_input(
            "🔥 Popularity",
            min_value=0.0,
            value=50.0
        )

        vote_average = st.slider(
            "⭐ Vote Average",
            0.0,
            10.0,
            6.5
        )

    st.markdown("---")

    # ===========================================
    # Prediction Button
    # ===========================================

    if st.button("🎯 Predict Movie Success", use_container_width=True):

        sample = pd.DataFrame({

            "budget": [budget],

            "popularity": [popularity],

            "runtime": [runtime],

            "vote_average": [vote_average]

        })

        prediction = model.predict(sample)[0]

        probability = model.predict_proba(sample)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("🎉 This movie is likely to be SUCCESSFUL!")

        else:

            st.error("❌ This movie is likely to be NOT SUCCESSFUL.")

        # ===========================================
        # Confidence
        # ===========================================

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        # ===========================================
        # Probability Chart
        # ===========================================

        probability_df = pd.DataFrame({

            "Result": ["Not Successful", "Successful"],

            "Probability": probability

        })

        fig = px.bar(

            probability_df,

            x="Result",

            y="Probability",

            color="Result",

            text_auto=".2f",

            template="plotly_dark",

            title="Prediction Probability"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ===========================================
        # Recommendation
        # ===========================================

        st.subheader("💡 Recommendation")

        if prediction == 1:

            st.info("""
### Recommendation

The model predicts that this movie has a **high probability of success**.

Possible reasons:

- Good audience rating
- Strong popularity
- Balanced runtime
- Suitable budget
""")

        else:

            st.warning("""
### Recommendation

The model predicts that this movie has a **lower probability of success**.

Possible improvements:

- Increase audience engagement
- Improve ratings
- Better marketing
- Optimize production budget
""")

        st.success("Prediction Completed Successfully ✅")
        # ==========================================================
# PART 7 - ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    st.header("ℹ About MovieIQ")

    st.markdown("""
    ## 🎬 MovieIQ - Predictive Analytics on Film Success

    MovieIQ is a Machine Learning based web application developed
    to predict whether a movie is likely to become successful.

    The project combines Data Analytics,
    Machine Learning and Interactive Visualization
    using Streamlit.
    """)

    st.markdown("---")

    # ==========================================
    # Project Objective
    # ==========================================

    st.subheader("🎯 Project Objective")

    st.info("""
Predict the success of movies using historical movie data and
provide interactive visualizations for better business insights.
""")

    # ==========================================
    # Technology Stack
    # ==========================================

    st.subheader("🛠 Technology Stack")

    tech = pd.DataFrame({

        "Technology":[

            "Python",

            "Pandas",

            "NumPy",

            "Plotly",

            "Scikit-Learn",

            "SciPy",

            "Joblib",

            "Streamlit"

        ],

        "Purpose":[

            "Programming",

            "Data Analysis",

            "Numerical Computing",

            "Interactive Visualization",

            "Machine Learning",

            "Statistical Testing",

            "Model Saving",

            "Dashboard"

        ]

    })

    st.dataframe(
        tech,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # Machine Learning Model
    # ==========================================

    st.subheader("🤖 Machine Learning Model")

    st.success("""
Model Used :

✔ Random Forest Classifier

Reason:

• High Accuracy

• Handles Non-linear Data

• Robust against Overfitting

• Easy Feature Importance Analysis
""")

    st.markdown("---")

    # ==========================================
    # Dataset Information
    # ==========================================

    st.subheader("📊 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric(
        "Movies",
        len(df)
    )

    col2.metric(
        "Features",
        len(df.columns)
    )

    st.markdown("---")

    # ==========================================
    # Features Used
    # ==========================================

    st.subheader("📌 Features Used")

    st.write("""
- Budget
- Popularity
- Runtime
- Vote Average
- Revenue
- Genre
""")

    st.markdown("---")

    # ==========================================
    # Footer
    # ==========================================

    st.success("🎉 Internship Project Completed Successfully!")

    st.markdown(
        """
---
### 👨‍💻 Developed Using

- Python
- Streamlit
- Plotly
- Scikit-Learn

**MovieIQ © 2026**
"""
    )