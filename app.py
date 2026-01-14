"""
Streamlit app for exploring Gen-AI policies and comparing user-uploaded policy.
"""

# from copyreg 
import pickle
# import pypickle
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from wordcloud import WordCloud, STOPWORDS

from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import circlify
pkl_path = "save/corex_results.pkl"
from ollama import chat
from ollama import Client
from ollama import generate








try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    st.error("scikit-learn is required. Install with `pip install scikit-learn`.")
    raise

# readability
try:
    import textstat
    HAS_TEXTSTAT = True
except Exception:
    HAS_TEXTSTAT = False

# docx and pdf reading
try:
    import docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

try:
    import PyPDF2
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# sentence-transformers  (for semantic embeddings if available)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.feature_extraction.text import CountVectorizer as _CV
    from plotly.colors import qualitative
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

st.set_page_config(page_title="GenAI Policy Explorer", layout="wide")

# Utilities
@st.cache_data
def load_dataset_from_mnt() -> Optional[pd.DataFrame]:
    # Search local repo `data/` first, then fallback to `/mnt/data`
    candidates = []
    repo_data = Path(__file__).parent / "data"
    if repo_data.exists():
        for ext in ("csv", "json", "parquet", "xlsx"):
            for p in repo_data.glob(f"*.{ext}"):
                candidates.append(p)
    data_dir = Path("/mnt/data")
    if data_dir.exists():
        for ext in ("csv", "json", "parquet", "xlsx"):
            for p in data_dir.glob(f"*.{ext}"):
                # avoid duplicates if same filename present in both
                if p not in candidates:
                    candidates.append(p)
    # Prefer the known dataset name if present
    pref_name = "UK-60-HEI-policies_with_file_text"
    pref = [p for p in candidates if pref_name in p.name]
    # Prefer files with 'policy' in the name otherwise
    if not pref:
        pref = [p for p in candidates if "policy" in p.name.lower()]
    candidates = pref or candidates
    if not candidates:
        return None
    # pick first candidate
    p = candidates[0]
    try:
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix == ".json":
            df = pd.read_json(p, lines=False)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix in (".xls", ".xlsx"):
            df = pd.read_excel(p)
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to read {p.name}: {e}")
        return None
    # Normalize column names (strip whitespace) and map common columns from the provided CSV
    df.columns = [c.strip() for c in df.columns]
    # explicit mapping for known CSV layout: Rank, Name, Filename, PolicyURL, Policy Text
    col_map = {}
    for c in df.columns:
        lc = c.lower().replace(' ', '')
        if lc == 'policytext' or ('policy' in lc and 'text' in lc) or lc.endswith('policytext'):
            col_map[c] = 'policy_text'
        elif lc == 'name' or lc == 'university' or 'institution' in lc:
            # map 'Name' -> 'university'
            col_map[c] = 'university'
        elif lc in ('policyurl', 'policy_url', 'url'):
            col_map[c] = 'url'
        elif lc == 'filename':
            col_map[c] = 'filename'
    if col_map:
        df = df.rename(columns=col_map)
    # Ensure columns
    if 'policy_text' not in df.columns:
        # try guesses
        txt_cols = [c for c in df.columns if 'text' in c.lower() or 'policy' in c.lower() or 'content' in c.lower()]
        if txt_cols:
            df = df.rename(columns={txt_cols[0]: 'policy_text'})
        else:
            st.warning(f"Dataset {p.name} doesn't contain a 'policy_text' column. Found columns: {df.columns.tolist()}")
            return None
    if 'university' not in df.columns:
        # attempt guess (include common 'name' column)
        uni_cols = [c for c in df.columns if 'univ' in c.lower() or 'institution' in c.lower() or 'school' in c.lower() or c.lower() == 'name']
        if uni_cols:
            df = df.rename(columns={uni_cols[0]: 'university'})
        else:
            # create index-based names
            df['university'] = df.index.astype(str)
    # fillna for text
    df['policy_text'] = df['policy_text'].fillna('').astype(str)
    df['university'] = df['university'].fillna('').astype(str)
    return df

def read_pdf(file_stream: bytes) -> str:
    if not HAS_PDF:
        raise RuntimeError("PyPDF2 not installed")
    text = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_stream))
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text)

def read_docx(file_stream: bytes) -> str:
    if not HAS_DOCX:
        raise RuntimeError("python-docx not installed")
    doc = docx.Document(io.BytesIO(file_stream))
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)

def read_uploaded_file(uploaded) -> Tuple[str, str]:
    """Return (filename, text)"""
    if uploaded is None:
        return "", ""
    fname = uploaded.name
    raw = uploaded.read()
    if fname.lower().endswith(".txt"):
        text = raw.decode('utf-8', errors='ignore')
    elif fname.lower().endswith(".pdf"):
        text = read_pdf(raw)
    elif fname.lower().endswith(".docx"):
        text = read_docx(raw)
    else:
        # try decode
        try:
            text = raw.decode('utf-8', errors='ignore')
        except Exception:
            text = ""
    return fname, text

# Text metric functions
def basic_stats(text: str) -> Dict[str, float]:
    s = text.strip()
    words = re.findall(r"\w+", s)
    sentences = re.split(r'[.!?]+', s)
    sentences = [se for se in sentences if se.strip()]
    return {
        "chars": len(s),
        "words": len(words),
        "sentences": len(sentences),
        "avg_words_per_sentence": (len(words) / max(1, len(sentences))),
    }

def readability_metrics(text: str) -> Dict[str, float]:
    if not HAS_TEXTSTAT:
        # fallback approximate
        bs = basic_stats(text)
        fk = None
        flesch = None
    else:
        try:
            fk = textstat.flesch_kincaid_grade(text)
            flesch = textstat.flesch_reading_ease(text)
        except Exception:
            fk = None
            flesch = None
    return {"flesch_kincaid_grade": fk, "flesch_reading_ease": flesch}

def keyword_counts(text: str, keywords: List[str]) -> Dict[str,int]:
    lc = text.lower()
    return {k: lc.count(k.lower()) for k in keywords}

# Similarity helpers
@st.cache_data
def build_tfidf_matrix(corpus: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

@st.cache_data
def build_sbert_model(name='all-MiniLM-L6-v2'):
    if not HAS_SBERT:
        return None
    model = SentenceTransformer(name)
    return model

def compute_embedding_sim(model, corpus_texts: List[str], query_text: str) -> np.ndarray:
    # returns cosine similarities between query and corpus
    texts = corpus_texts + [query_text]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    corpus_embs = embs[:-1]
    query_emb = embs[-1].reshape(1, -1)
    sim = cosine_similarity(corpus_embs, query_emb).flatten()
    return sim

@st.cache_data
def compute_corpus_metrics(df: pd.DataFrame) -> pd.DataFrame:
    keywords = ["student", "staff", "assessment", "plagiarism", "ai", "attribution", "data", "consent", "privacy", "third party", "model", "training"]
    rows = []
    for _, r in df.iterrows():
        text = str(r.get('policy_text','') or '')
        bs = basic_stats(text)
        rd = readability_metrics(text)
        kc = keyword_counts(text, keywords)
        row = {
            "university": r.get('university', ''),
            "chars": bs['chars'],
            "words": bs['words'],
            "sentences": bs['sentences'],
            "avg_words_per_sentence": bs['avg_words_per_sentence'],
            **{f"kw_{k}": kc[k] for k in keywords},
            "flesch_kincaid": rd['flesch_kincaid_grade'],
            "flesch_reading_ease": rd['flesch_reading_ease'],
        }
        rows.append(row)
    return pd.DataFrame(rows)

def generate_word_cloud(texts: pd.Series, name):
    combined_text = " ".join(texts.astype(str).tolist())
    # extend stopwords with some common artifacts from scraped policies
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 'ac', 'co', 'http', 'www','use' , 'will', 'using', 'must', 'com', 'org', 'edu', 'page', 'policy', 'policytext', 'may', 'take', 'top'])
    #put all single letters into stopwords
    stopwords.update(list("abcdefghijklmnopqrstuvwxyz"))
    wc = WordCloud(width=1200, height=600, background_color='white',
                   stopwords=stopwords, collocations=False).generate(combined_text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    # ax.set_title(f'Word Cloud for {name}', fontsize=16)

    # Display in streamlit
    st.pyplot(fig)
    plt.close(fig)

    # # allow download as image
    # buf = io.BytesIO()
    # wc.to_image().save(buf, format='PNG')
    # st.download_button("download", buf.getvalue(), "wordcloud.png")

anchors_old = [
        ["ethics", "ethical", "fair"],               # Topic 1: Ethics
        ["assessment", "exam", "grading", "quiz"],   # Topic 2: Assessment
        ["privacy", "security"],                     # Topic 3: Data Privacy
        ["student", "staff", "faculty"],             # Topic 4: Stakeholders
        ["inclusion", "equity", "accessibility"]   # Topic 5: Inclusivity
    ]
alexes_3_discourses = [
        ["Usability", "adoption", "readiness", "development"],
        ["Fair", "compliance", "ethical", "risk", "biases", "marginalise"],
        ["Effective", "Measure", "Performance", "facilitate", "teaching"]
    ]
topics_from_thematic_analysis = [
        ["resources", "training",],
        ["disability", "accessible"],
        ["acknowledge", "cite", "reference"],
        ["appropriate", "allowed", "ethical", "transparency", "use", "integrity"],
        ["risk", "privacy", "error", "inaccurate", "hallucination", "copyright" ],
        ["misconduct"],
        ["detection", "plagiarism"]
    ]

def chunk_policy(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def run_corex(policies, anchors):
    # policies = df['policy_text']
    # Split policies into equal-length chunks (e.g., 50 words)
    opt_chunk_size=60
    chunk_size=opt_chunk_size
    # st.text(f"Using chunk size = {chunk_size} words")
    # chunk_size=100 --- IGNORE ---

    chunks = []
    for policy in policies:
        chunks.extend(chunk_policy(policy, chunk_size))

    # Prepare document-term matrix for chunks
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(chunks)
    words = vectorizer.get_feature_names_out()


    corex_model = ct.Corex(n_hidden=n_topics, words=words, seed=42)
    corex_model.fit(doc_term_matrix, words=words, anchors=anchors, anchor_strength=5)
    # corex_model.fit(doc_term_matrix, words=words)

    # Get topic distribution for each chunk
    corex_topic_dist = corex_model.transform(doc_term_matrix, words=words)

    # Aggregate chunk-level topic assignments to policy-level by mean
    chunks_per_policy = [len(chunk_policy(policy, chunk_size)) for policy in policies]
    chunk_indices = np.repeat(np.arange(len(policies)), chunks_per_policy)

    # Create a DataFrame for chunk topic assignments
    corex_chunk_df = pd.DataFrame(corex_topic_dist, columns=[f'CorEx_topic_{i}' for i in range(n_topics)])
    corex_chunk_df['policy_idx'] = chunk_indices

    # Compute mean topic assignment for each policy
    corex_policy_topic_means = corex_chunk_df.groupby('policy_idx').mean()
    date_run= date.today()
    #save corex results to pickle
    # with open("corex_results.pkl", "wb") as f:
    #     pickle.dump((corex_model, doc_term_matrix, corex_policy_topic_means), f)
    #option to download pickle file
    
    # with st.sidebar.expander("Advanced Options", expanded=False):
    #     #dump (corex_model, doc_term_matrix, corex_policy_topic_means as pickle file corecx_results_DATE.pkl
    #     pkl_path = f"save/corex_results_{date_run}.pkl"
    #     with open(pkl_path, "wb") as f:
    #         pickle.dump((corex_model, doc_term_matrix, corex_policy_topic_means), f)
    #     st.download_button("Download CorEx results", data=open(pkl_path, "rb").read(), file_name=f"corex_results_{date_run}.pkl")
    return corex_model, doc_term_matrix, corex_policy_topic_means

def corexResults_piechart(corex_model, idx, numTopicsPerGroup=3):
# n_topics=len(anchors)+1
    corex_topics = [f'CorEx_topic_{i}' for i in range(n_topics)]
    # CorEx topic values for this policy (uses existing CorEx_topic_* columns)
    corex_topic_cols = corex_topics  # provided in notebook as a list of column names
    corex_vals = df.loc[idx, corex_topic_cols].astype(float).values

    # fig=corexResults_piechart(corex_model, numTopicsPerGroup=3)

    if corex_vals.sum() == 0:
            # no topics matched -- show a single grey slice
            fig = go.Figure(data=[go.Pie(
            labels=['No matching CorEx topics'], 
            values=[1], 
            marker=dict(colors=['lightgrey'])
            )])
    else:
        # get first 3 words for each topic from the fitted model and display as pie chart
        labels = []
        legend_labels = []
        # group=[]
        if True: #'corex_model' in globals():          
            topics_top3 = corex_model.get_topics(n_words=numTopicsPerGroup)
            for i, t in enumerate(topics_top3[:len(corex_vals)]):
                if t:
                    words = [w for w, *rest in t][:numTopicsPerGroup]  # take top N words
                    label = ', '.join(words)
                    labels.append(label)
                    legend_labels.append(f"Group{i+1} ({corex_vals[i]/corex_vals.sum()*100:.1f}%): {label}")
                    # group.append([f"Group{i+1}"])
                else:
                    labels.append(f"Group{i+1}")
                    legend_labels.append(f"Group{i+1} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)")
            # print(legend_labels)
        # else:
        #     labels = [f"Group{i+1}" for i in range(len(corex_vals))]
        #     legend_labels = [f"Group{i+1} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)" for i in range(len(corex_vals))]


        # Create pie chart with Plotly
        fig = go.Figure(data=[go.Pie(
        labels=legend_labels,
        values=corex_vals,
        textinfo='percent',
        # textinfo='label+percent',
        textposition='inside',
        hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<br>Percent: %{percent}<extra></extra>'
        )])
        
    fig.update_layout(
        title=f"Topics found in {uni_choice}'s policy",
        showlegend=True,
        height=600
    )
    return fig

def scatterPlot2col(
    df, 
    sel_row, 
    uni_choice, 
    metric_func, 
    metric_label, 
    color="#fefae0",
    value_format="{:.2f}"):
    """
    Generic metric plotter for comparing a numeric text metric across universities.
    """

    colE, colF = st.columns([1, 5])

    with colF:
        # Compute metric for all universities
        all_metrics = df['policy_text'].apply(lambda t: metric_func(str(t))).values
        avg_metric = all_metrics.mean()
        sel_metric = metric_func(str(sel_row.get('policy_text', '')))

        # Data for scatter plot
        plot_df = pd.DataFrame({
            'metric_value': all_metrics,
            'university': df['university'],
            'y': [0] * len(all_metrics)
        })

        # Create scatter plot
        fig = go.Figure()
        fig.update_layout(plot_bgcolor=color, paper_bgcolor=color)

        # Add all universities
        fig.add_trace(go.Scatter(
            x=plot_df['metric_value'],
            y=plot_df['y'],
            mode='markers',
            marker=dict(color='steelblue', size=8, opacity=0.7),
            text=plot_df['university'],
            hovertemplate=f"<b>%{{text}}</b><br>{metric_label}: %{{x:.2f}}<extra></extra>",
            name='All Universities'
        ))

        # Highlight selected university
        fig.add_trace(go.Scatter(
            x=[sel_metric],
            y=[0],
            mode='markers',
            marker=dict(color='red', size=12),
            text=[uni_choice],
            hovertemplate=f"<b>%{{text}}</b><br>{metric_label}: %{{x:.2f}}<extra></extra>",
            name=f'{uni_choice}'
        ))

        # Add average line
        fig.add_vline(
            x=avg_metric,
            line_dash="dash",
            line_color="orange",
            annotation_text="Average",
            annotation_position="top"
        )

        # Layout styling
        fig.update_layout(
            xaxis_title=metric_label,
            yaxis=dict(visible=False),
            height=300,
            showlegend=True,
            hovermode='closest'
        )

        st.plotly_chart(fig)#, width='stretch')

    with colE:
        # Metric display
        try:
            sel_fmt = value_format.format(sel_metric)
        except:
            sel_fmt = str(sel_metric)

        st.markdown(
            f"""
            <div style='display: flex; flex-direction: column; justify-content: center; 
                        align-items: center; height: 100%; width: 100%; 
                        text-align: center; padding: 1rem; box-sizing: border-box;'>
                <p style='font-size: clamp(10px, 1.5vw, 15px);
                        font-weight: 600; 
                        margin-bottom: 0;'>{metric_label}:</p>
                <p style='font-size: clamp(20px, 5vw, 40px);
                        font-weight: 700; 
                        margin-top: 0;
                        line-height: 1;'> {sel_fmt} </p>
            </div>
            """,
            unsafe_allow_html=True
        )

def tfidf_similarity(corpus_texts, corpus_meta, uploaded_text, build_tfidf_matrix_func):


    #             vectorizer, X = build_tfidf_matrix(corpus_texts + [uploaded_text])
    #             qvec = X[-1]
    #             corpus_X = X[:-1]
    #             sims = cosine_similarity(corpus_X, qvec).flatten()
    #             sim_df = pd.DataFrame({
    #                 "university": corpus_meta,
    #                 "similarity": sims,
    #                 "words": [len(re.findall(r"\w+", t)) for t in corpus_texts],
    #                 "chars": [len(t) for t in corpus_texts]
    #             }).sort_values("similarity", ascending=False).reset_index(drop=True)
    #             st.dataframe(sim_df.head(20))
    """
    Compare an uploaded text to a corpus of documents using TF-IDF cosine similarity.

    Parameters
    ----------
    corpus_texts : list[str]
        List of corpus documents (text only).
    corpus_meta : list[str]
        Metadata for each document (e.g., university names).
    uploaded_text : str
        The uploaded text to compare against the corpus.
    build_tfidf_matrix_func : callable
        Function that takes a list of texts and returns (vectorizer, X_matrix).
    top_n : int, optional
        Number of top similar documents to display or return (default=20).
    show_in_streamlit : bool, optional
        If True, displays dataframe in Streamlit; otherwise returns it.

    Returns
    -------
    sim_df : pd.DataFrame
        Sorted DataFrame with columns: university, similarity, words, chars.
    """

    # Build TF-IDF representation
    vectorizer, X = build_tfidf_matrix_func(corpus_texts + [uploaded_text])
    qvec = X[-1]           # query (uploaded)
    corpus_X = X[:-1]      # corpus
    sims = cosine_similarity(corpus_X, qvec).flatten()

    # Build similarity dataframe
    sim_df = pd.DataFrame({
        "university": corpus_meta,
        "similarity": sims,
        "words": [len(re.findall(r"\w+", t)) for t in corpus_texts],
        "chars": [len(t) for t in corpus_texts]
    }).sort_values("similarity", ascending=False).reset_index(drop=True)


    return sim_df

#------------------------------------------------------------------------------------------------
# MAIN-------------------------------------------------------------------------------------------
# Streamlit App Configuration
st.set_page_config(page_title="Gen-AI Policy Explorer", layout="wide", initial_sidebar_state="expanded")

applogo = "img/pstapplogo.png" 
if os.path.exists(applogo):
    st.sidebar.image(applogo, width=70)
st.sidebar.header("HEI Gen AI Policy Toolkit")

# Utilities (loading dataset, sidebar summary, uploader, computing metrics_df)
df = load_dataset_from_mnt()
if df is None:
    st.sidebar.warning("No dataset automatically found in /mnt/data. You can still upload a single policy to compare against an empty corpus.")
    df = pd.DataFrame(columns=['university', 'policy_text'])

# show basic corpus summary
# st.sidebar.markdown("---")
# st.sidebar.subheader("Corpus summary")
# st.sidebar.write(f"Policies loaded: **{len(df)}**")
if len(df) > 0:
    avg_len = int(df['policy_text'].str.len().mean())
    # st.sidebar.write(f"Avg policy length (chars): **{avg_len}**")


# Precompute metrics for corpus
# st.sidebar.markdown("---")
# compute_button = st.sidebar.button("(Re)compute corpus metrics")


if len(df) > 0:
    with st.spinner("Computing corpus metrics..."):
        metrics_df = compute_corpus_metrics(df)
else:
    metrics_df = pd.DataFrame()

# Replace previous two-column UI with a mode-based layout
# Add a sidebar mode selector: Explore, Upload, About
st.sidebar.divider()
mode = st.sidebar.radio("Select an option:", options=["About", "Explore", "Analyse", "Upload", "Enforceablity", "Contact Us"], index=0)


#------------------------------------------------------------------------------------------------
# EXPLORE------------------------------------------------------------------------------------------
if mode == "Contact Us":
    st.header("Contact Us")
    st.markdown("""
    For suggestions, feedback, or complaints regarding the HEI Gen AI Policy Toolkit, please reach out to us through the following channels:
    - **Email**: to add.    
    - **Project Website**: to add
    """)
elif mode == "Explore":
    # st.header("Exploring UK HEI policies for Generative AI use")
    # prepare display_df for listing/searching (same logic as before)
    display_df = df.copy()
    if 'policy_text' in display_df.columns:
        display_df['words'] = display_df['policy_text'].apply(lambda t: len(re.findall(r"\w+", str(t))))
        display_df['chars'] = display_df['policy_text'].apply(lambda t: len(str(t)))

    #ideas: 
    # show list of universities as containers with name of university and explore button
    # [done] show logos with university names
    # show logos on UK map

    # search box and min words filter visible in main pane
    q = st.text_input("Search universities or policy text (regex can be used)", value="")



    # min_words = st.slider("Min words", min_value=0, max_value=5000, value=0, step=50)
    min_words = 0
    if q:
        try:
            mask = display_df['university'].str.contains(q, case=False, na=False) | display_df['policy_text'].str.contains(q, case=False, na=False, regex=True)
        except Exception:
            mask = display_df['university'].str.contains(q, case=False, na=False)
        display_df = display_df[mask]
    display_df = display_df[display_df.get('words', 0) >= min_words]
    
    #remove URLs from text in display_df
    display_df['policy_text'] = display_df['policy_text'].apply(lambda t: re.sub(r'http\S+|www\.\S+', '', str(t)))

    #remove individual numbers from text in display_df
    display_df['policy_text'] = display_df['policy_text'].apply(lambda t: re.sub(r'\b\d+\b', '', str(t)))

    #remove single letter words from text in display_df
    display_df['policy_text'] = display_df['policy_text'].apply(lambda t: re.sub(r'\b[a-zA-Z]\b', '', str(t)))

    #remove any words with numbers in them from text in display_df
    display_df['policy_text'] = display_df['policy_text'].apply(lambda t: re.sub(r'\b\w*\d\w*\b', '', str(t)))

    #remove word = "st" from text in display_df
    display_df['policy_text'] = display_df['policy_text'].apply(lambda t: re.sub(r'\b(st)\b', '', str(t)))

    if len(metrics_df) == 0:
        st.info("No corpus metrics available. Add a dataset file to /mnt/data or upload one policy to compare.")
    else:
        # st.subheader(f"Policies loaded: {len(display_df)}")
        st.badge(f"Policies loaded: {len(display_df)}", icon=":material/check:", color="green")
        st.dataframe(display_df[['university', 'words', 'chars']].head(200))
        # st.dataframe(display_df[['university', 'words', 'chars', 'flesch_kincaid']].head(200))
            #basic stats
        # with st.expander("Statistics & evaluation overview", expanded=True):
        # # st.markdown("**Basic statistics across universities**")
        #     # Top longest policies
        #     top_long = metrics_df.sort_values("words", ascending=False).head(10)
        #     st.write("Top 10 longest (by words):")
        #     st.table(top_long[['university','words','chars','flesch_kincaid']].reset_index(drop=True))

            # Keyword aggregation
        # kw_cols = [c for c in metrics_df.columns if c.startswith('kw_')]
        # if kw_cols:
        #     kw_sum = metrics_df[kw_cols].sum().sort_values(ascending=False)
        #     with st.expander("Keywords", expanded=True):
        #         # col1, col2 = st.columns([1,1])
        #         # with col1:
        #         #     # st.subheader("Keyword mentions across universities (counts)")
        #         #     st.table(kw_sum.rename_axis('keyword').reset_index().rename(columns={0:'count'}))
        #         # with col2:
        #         #     #show top keywords as a plotly donut chart
        #         #     fig = px.pie(values=kw_sum.values, names=kw_sum.index, title='Top Keywords')
        #         #     st.plotly_chart(fig)



        #         #load keywords into multiselect for further analysis, if a user removes a keyword, update the bar chart
        #         selected_keywords = st.multiselect(
        #             "Select keywords to analyze", 
        #             options=kw_sum.index.str.replace('kw_', '').tolist(), 
        #             default=kw_sum.index.str.replace('kw_', '').tolist()
        #         )
        #         #st badge to show keywords selected
        #         st.badge(f"Keywords selected: {len(selected_keywords)}", icon=":material/tag:", color="blue")

        #         if selected_keywords:
        #             # Filter the keyword data based on selection
        #             selected_kw_cols = [f'kw_{kw}' for kw in selected_keywords]
        #             filtered_kw_sum = metrics_df[selected_kw_cols].sum().sort_values(ascending=False)
                    
        #             # Update the bar chart with selected keywords
        #             filtered_kw_df = filtered_kw_sum.reset_index()
        #             filtered_kw_df.columns = ['keyword', 'count']
        #             filtered_kw_df['keyword'] = filtered_kw_df['keyword'].str.replace('kw_', '')
                    
        #             fig_filtered = px.bar(filtered_kw_df, x='keyword', y='count')
        #             # , 
        #                                 #  title=f'Selected Keywords ({len(selected_keywords)} selected)')
        #             st.plotly_chart(fig_filtered)
        #         else:
        #             st.info("Select keywords to display in the chart")

       #------- Word Frequency Analysis
        # Word frequency bar chart for the current corpus
        texts = display_df['policy_text'].astype(str).tolist()
        if not any(t.strip() for t in texts):
            st.info("No text available to compute word frequencies.")
        else:
            with st.expander(f"Most Frequent Words in Selected Policies", expanded=True):
                vec = CountVectorizer(stop_words='english', token_pattern=r"(?u)\b\w+\b", lowercase=True)
                X = vec.fit_transform(texts)
                freqs = np.asarray(X.sum(axis=0)).ravel()
                terms = vec.get_feature_names_out()
                words_df = pd.DataFrame({"word": terms, "count": freqs}).sort_values("count", ascending=False).reset_index(drop=True)

                # allow selecting a range of word ranks to plot (default 0..50)
                max_words = len(words_df)
                max_words= 500
                default_end = min(50, max_words)
                start, end = st.slider(
                    "Select word rank range to plot (start, end)",
                    min_value=0,
                    max_value=max_words,
                    value=(0, default_end),
                    step=1
                )
                # Ensure valid slicing
                if end <= start:
                    end = min(start + 1, max_words)
                words_df = words_df.iloc[start:end].reset_index(drop=True)
                top_n = len(words_df)

            # with st.expander(f"Top {top_n} most frequent words", expanded=True):
                fig = px.bar(words_df.head(top_n), x="word", y="count", title=f"Word Frequencies", labels={"count":"Frequency","word":"Word"})
                fig.update_layout(xaxis_tickangle=-45, height=420)
                st.plotly_chart(fig)
                st.dataframe(words_df.head(200), width='stretch')

            # allow download of the full frequency table
            # csv_bytes = words_df.to_csv(index=False).encode("utf-8")
            # st.download_button("Download word frequencies (CSV)", data=csv_bytes, file_name="word_frequencies.csv", mime="text/csv")

        # with st.expander("Keyword Analysis", expanded=True):


        # #add selected keywords as input chips and show their counts across corpus as graph
        # selected_keywords = st.multiselect("Select keywords to analyze", options=kw_sum.index.tolist())
        # if selected_keywords:
        #     filtered_df = display_df[display_df['policy_text'].str.contains('|'.join(selected_keywords), case=False, na=False)]
        #     st.bar_chart(filtered_df['policy_text'].str.split().str.len())
                # st.subheader("Combined word cloud:")
        # Generate and display word cloud
    # generate_word_cloud(display_df['policy_text'])
    
        # Generate and display word cloud
    with st.expander(f"Word Cloud for Selected Policies", expanded=True):
        try:
            wordcloud = generate_word_cloud(display_df['policy_text'],"Combined Policies")
            st.image(wordcloud, width='stretch')
        except Exception as e:
            pass
            # st.error(f"Error generating word cloud: {e}")

            # st.subheader("CorEx Topic Modeling:")
    st.divider()
    
    anchors = topics_from_thematic_analysis
    n_topics = len(anchors)+1 # n_topics=14
    policies = display_df['policy_text']
    if os.path.exists("save/corex_results.pkl"):
        with open("save/corex_results.pkl", "rb") as f:
            corex_model, doc_term_matrix, corex_policy_topic_means = pickle.load(f)
            print(f"Loaded corex results from pickle -> in main")
    else:   
        corex_model, doc_term_matrix, corex_policy_topic_means = run_corex(policies, anchors=anchors)
        print("Generated new corex results -> in main")
    for i in range(n_topics):
        df[f'CorEx_topic_{i}'] = corex_policy_topic_means[f'CorEx_topic_{i}'].values



    with st.expander("Common Topics in ALL policies", expanded=True):
        # show topics as a bubble chart, with each bubble represeting the size of topic in the corpora 
        topic_sizes = df[[f'CorEx_topic_{i}' for i in range(n_topics)]].sum()
        # fig = px.scatter(x=topic_sizes.index, y=topic_sizes.values, size=topic_sizes.values, title="CorEx Topic Sizes")
        # st.plotly_chart(fig)

        #show topics as circles using  circlify

        circles = circlify.circlify(
            topic_sizes.values.tolist(), 
            show_enclosure=False, 
            target_enclosure=circlify.Circle(x=0, y=0, r=1)
        )

        # Get first 6 words for each topic from the CorEx model
        num_topics_to_show = 6#len(topic_sizes)
        if 'corex_model' in globals():
            topics_top3 = corex_model.get_topics(n_words=num_topics_to_show)
            topic_labels = []
            for i, t in enumerate(topics_top3[:len(topic_sizes)]):
                if t:
                    words = [w for w, *rest in t][:num_topics_to_show]  # take top 6 words
                    topic_labels.append(f'\n'.join(words))
                    # topic_labels.append(f"Group {i+1}:\n" + '\n'.join(words))
                else:
                    topic_labels.append(f" ")
                    # topic_labels.append(f"Group {i+1}")
        else:
            topic_labels = [f"Group {i+1}" for i in range(len(topic_sizes))]
            # topic_labels = [f"Group {i+1}" for i in range(len(topic_sizes))]
        
        fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(6,2))
        # Adjust figure size for landscape mode
        fig.set_size_inches(10, 6)
        ax.axis('off')
        
        # import matplotlib.colors as mcolors
        
        # Generate random light colors
        light_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                    'lightpink', 'lightgray', 'lightcyan', 'lavender',
                    'mistyrose', 'honeydew', 'aliceblue', 'seashell',
                    'linen', 'oldlace', 'floralwhite', 'mintcream']
        
        # Manually draw circles using matplotlib
        for i, circle in enumerate(circles):
            color = light_colors[i % len(light_colors)]
            patch = plt.Circle((circle.x, circle.y), circle.r, alpha=0.6, 
                    facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(patch)
        
        # Add text labels inside each circle with word wrapping to fit
        for circle, label in zip(circles, topic_labels):
            # Calculate font size based on circle radius to ensure text fits
            # font_size = min(8, max(4, circle.r * 10))
            font_size = min(12, max(6, circle.r * 20))
            ax.text(circle.x, circle.y, label, ha='center', va='center', 
            fontsize=font_size, wrap=True)
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')

        st.pyplot(fig)

    # with st.expander("Topics found", expanded=False):
    #     fig= corexResults_piechart(corex_model, numTopicsPerGroup=3)
    #     st.plotly_chart(fig)


    selected_policies = display_df['policy_text']
    if len(selected_policies) != len(policies) and len(selected_policies) > 0:  
        with st.expander("Common Topics in SELECTED policies", expanded=True):  
            st.badge(f"Policies selected: {len(display_df)}", icon=":material/check:", color="green")
            st.info("Note: as you have selected a subset of policies, click the button below to analyze topics for the selected policies only.")
            if st.button("Analyze topics for selected policies"):
                # if selected policies are not zero, rerun corex on selected policies
                selected_policies = display_df['policy_text']
                if len(selected_policies) == 0:
                    st.info("No policies selected to analyze.")
                else:   
                    corex_model_sel, doc_term_matrix_sel, corex_policy_topic_means_sel = run_corex(selected_policies, anchors=anchors)
                    print("Generated new corex results for selected policies -> in main")
                    for i in range(n_topics):
                        display_df[f'CorEx_topic_{i}'] = corex_policy_topic_means_sel[f'CorEx_topic_{i}'].values
                    
                    # show topics as a bubble chart, with each bubble represeting the size of topic in the corpora 
                    topic_sizes_sel = display_df[[f'CorEx_topic_{i}' for i in range(n_topics)]].sum()
                    # fig = px.scatter(x=topic_sizes.index, y=topic_sizes.values, size=topic_sizes.values, title="CorEx Topic Sizes")
                    # st.plotly_chart(fig)

                    #show topics as circles using  circlify

                    circles = circlify.circlify(
                        topic_sizes_sel.values.tolist(), 
                        show_enclosure=False, 
                        target_enclosure=circlify.Circle(x=0, y=0, r=1)
                    )

                    # Get first 6 words for each topic from the CorEx model
                    num_topics_to_show = 6#len(topic_sizes)
                    if 'corex_model_sel' in globals():
                        topics_top3 = corex_model_sel.get_topics(n_words=num_topics_to_show)
                        topic_labels = []
                        for i, t in enumerate(topics_top3[:len(topic_sizes)]):
                            if t:
                                words = [w for w, *rest in t][:num_topics_to_show]  # take top 6 words
                                topic_labels.append(f'\n'.join(words))
                                # topic_labels.append(f"Group {i+1}:\n" + '\n'.join(words))
                            else:
                                topic_labels.append(f" ")
                                # topic_labels.append(f"Group {i+1}")
                    else:
                        topic_labels = [f"Group {i+1}" for i in range(len(topic_sizes))]
                        # topic_labels = [f"Group {i+1}" for i in range(len(topic_sizes))]
                    
                    fig, ax = plt.subplots()
                    # fig, ax = plt.subplots(figsize=(6,2))
                    # Adjust figure size for landscape mode
                    fig.set_size_inches(10, 6)
                    ax.axis('off')
                    
                    # import matplotlib.colors as mcolors
                    
                    # Generate random light colors
                    light_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                                'lightpink', 'lightgray', 'lightcyan', 'lavender',
                                'mistyrose', 'honeydew', 'aliceblue', 'seashell',
                                'linen', 'oldlace', 'floralwhite', 'mintcream']
                    
                    # Manually draw circles using matplotlib
                    for i, circle in enumerate(circles):
                        color = light_colors[i % len(light_colors)]
                        patch = plt.Circle((circle.x, circle.y), circle.r, alpha=0.6, 
                                facecolor=color, edgecolor='black', linewidth=0.5)
                        ax.add_patch(patch)
                    
                    # Add text labels inside each circle with word wrapping to fit
                    for circle, label in zip(circles, topic_labels):
                        # Calculate font size based on circle radius to ensure text fits
                        # font_size = min(8, max(4, circle.r * 10))
                        font_size = min(12, max(6, circle.r * 20))
                        ax.text(circle.x, circle.y, label, ha='center', va='center', 
                        fontsize=font_size, wrap=True)
                    
                    # Set equal aspect ratio and limits
                    ax.set_xlim(-1.1, 1.1)
                    ax.set_ylim(-1.1, 1.1)
                    ax.set_aspect('equal')

                    st.pyplot(fig)
      

    # with st.expander(":blue-background[Full List of Topics Found]", expanded=False):
    #         # Print top words for each topic
    #     for i, topic in enumerate(corex_model.get_topics(n_words=10)):
    #         st.text(f"Group {i+1}: {[w for w, _, _ in topic]}")
    #     # st.text(df[[f'CorEx_topic_{i}' for i in range(n_topics)]].head())
    with st.expander("Advance Options", expanded=False):
        # st.download_button("Download similarity table (CSV)", data=csv, file_name="similarity_results.csv", mime="text/csv")
        st.download_button("Download metrics (CSV)", data=metrics_df.to_csv(index=False).encode('utf-8'), file_name="corpus_metrics.csv", mime="text/csv")
        # force recompute button for corex
            #         # Print top words for each topic
        st.text("CorEx Topics:")
        for i, topic in enumerate(corex_model.get_topics(n_words=10)):
            st.text(f"Group {i+1}: {[w for w, _, _ in topic]}")
        # st.text(df[[f'CorEx_topic_{i}' for i in range(n_topics)]].head())
        #dump (corex_model, doc_term_matrix, corex_policy_topic_means as pickle file corecx_results_DATE.pkl
        date_run= date.today()
        pkl_path = f"save/corex_results_{date_run}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump((corex_model, doc_term_matrix, corex_policy_topic_means), f)
        st.download_button("Download CorEx results (all policies)", data=open(pkl_path, "rb").read(), file_name=f"corex_results_{date_run}.pkl")

        if len(selected_policies) != len(policies) and len(selected_policies) > 0 and 'corex_model_sel' in globals():  
            #dump (corex_model_sel, doc_term_matrix_sel, corex_policy_topic_means_sel as pickle file corecx_results_SELECTED_DATE.pkl
            date_run= date.today()
            pkl_path_sel = f"save/corex_results_SELECTED_{date_run}.pkl"
            with open(pkl_path_sel, "wb") as f:
                pickle.dump((corex_model_sel, doc_term_matrix_sel, corex_policy_topic_means_sel), f)
            st.download_button("Download CorEx results (selected policies)", data=open(pkl_path_sel, "rb").read(), file_name=f"corex_results_SELECTED_{date_run}.pkl")
        
             
        if st.button("Force Recompute CorEx Topic Modeling for ALL policies"):
            corex_model, doc_term_matrix, corex_policy_topic_means = run_corex(policies, anchors=anchors)
            
            st.success("Recomputed CorEx Topic Modeling.")
            date_run= date.today()
            pkl_path = f"save/corex_results_{date_run}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump((corex_model, doc_term_matrix, corex_policy_topic_means), f)
            st.download_button("Download Recomputed CorEx results", data=open(pkl_path, "rb").read(), file_name=f"corex_results_{date_run}.pkl")

# #------------------------------------------------------------------------------------------------
# # ANALYSE------------------------------------------------------------------------------------------
elif mode == "Analyse":
    # st.text("select or type university name to analyze")

    # # prepare display_df for listing/searching  
    display_df = df.copy()

    uni_options = [""] + display_df['university'].tolist()
    # uni_options = ["All universities"] + display_df['university'].tolist()
    uni_choice = st.selectbox("Select or type university name to analyze", options=uni_options, index=0)    
    sel_df = df[df['university'] == uni_choice]

    if sel_df.empty:
        st.warning(" ")
        # st.warning("Selected university not found in dataset.")
    else:
        sel_row = sel_df.iloc[0]
        #if logo of university exists in img/logos/filename.png display it
        logofilename = os.path.splitext(sel_row.get('filename','').strip())[0]
        logo_path = f"img/{logofilename}.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=300)
        st.subheader(f"{uni_choice}")

        if sel_row.get('url'):
            st.write("Policy URL:", sel_row.get('url'))

        # st.markdown("**Raw policy text**")
        with st.expander("Raw policy text", expanded=True):
            st.text_area("Raw policy text", value=sel_row.get('policy_text',''), height=300, label_visibility="collapsed") 
        idx= sel_row.name   
        col1, col2 = st.columns([2,1])
            
        with col1:
            st.markdown(f"**Wordcloud for {uni_choice}**")
            wordcloud = generate_word_cloud(sel_df['policy_text'], name=uni_choice)
            # get row number for university
            
        with col2:
            st.markdown("**Stats**")
            bs = basic_stats(str(sel_row.get('policy_text','')))
            rd = readability_metrics(str(sel_row.get('policy_text','')))
            st.write(pd.DataFrame([ {**bs, **rd} ]).T.rename(columns={0:"value"}))
            
            
        with st.expander("General Statistics", expanded=True):
                scatterPlot2col(df, sel_row, uni_choice, 
                           lambda t: len(re.findall(r"\w+", str(t))), 
                           "Word Count", "#EDF1FD", "{:,.0f}")
                scatterPlot2col(df, sel_row, uni_choice, 
                           lambda t: basic_stats(t)['avg_words_per_sentence'], 
                           "Words/Sentence", "#F8EDFD")
                scatterPlot2col(df, sel_row, uni_choice, 
                           lambda t: textstat.flesch_reading_ease(t), 
                           "Reading Ease", "#F2FDED")
                
        


        # topic modeling with CorEx, pie chart of topic distribution
        anchors = topics_from_thematic_analysis
        n_topics = len(anchors)+1  # n_topics=14
        policies = df['policy_text']
        # corex_model, doc_term_matrix, corex_policy_topic_means= run_corex(policies, anchors=anchors)
        if os.path.exists("save/corex_results.pkl"):
            with open("save/corex_results.pkl", "rb") as f:
                corex_model, doc_term_matrix, corex_policy_topic_means = pickle.load(f)
                print("Loaded corex results from pickle -> in analyse")
        else:   
            corex_model, doc_term_matrix, corex_policy_topic_means = run_corex(policies, anchors=anchors)
            print("Generated new corex results -> in analyse")


        #     # Add topic distribution to df1
        for i in range(n_topics):
            df[f'CorEx_topic_{i}'] = corex_policy_topic_means[f'CorEx_topic_{i}'].values

        fig= corexResults_piechart(corex_model, idx, numTopicsPerGroup=9)

        # Pie chart of CorEx topic distribution for this policy (include first 3 words of each topic)
        # if corex_vals.sum() == 0:
        #     # no topics matched -- show a single grey slice
        #     fig = go.Figure(data=[go.Pie(
        #     labels=['No matching CorEx topics'], 
        #     values=[1], 
        #     marker=dict(colors=['lightgrey'])
        #     )])
        # else:
        #     # get first 3 words for each topic from the fitted model and display as pie chart
        #     labels = []
        #     legend_labels = []
        #     if 'corex_model' in globals():          
        #         topics_top3 = corex_model.get_topics(n_words=5)
        #         for i, t in enumerate(topics_top3[:len(corex_vals)]):
        #             if t:
        #                 words = [w for w, *rest in t][:3]  # take top 3 words
        #                 label = ', '.join(words)
        #                 labels.append(label)
        #                 legend_labels.append(f"Group{i+1}: {label} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)")
        #             else:
        #                 labels.append(f"Group{i+1}")
        #                 legend_labels.append(f"Group{i+1} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)")
        #     else:
        #         labels = [f"Group{i+1}" for i in range(len(corex_vals))]
        #         legend_labels = [f"Group{i+1} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)" for i in range(len(corex_vals))]

        #     # Expose the verbose legend labels for use elsewhere (e.g. display the key below the chart)
        #     pie_legend_labels = legend_labels
            
        #     # Create pie chart with Plotly
        #     fig = go.Figure(data=[go.Pie(
        #     labels=labels,
        #     values=corex_vals,
        #     textinfo='label+percent',
        #     textposition='inside',
        #     hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<br>Percent: %{percent}<extra></extra>'
        #     )])
            
        # fig.update_layout(
        #     title=f"Topics found in {uni_choice}'s policy",
        #     showlegend=True,
        #     height=600
        # )
        
        with st.expander(f"Topics found in {uni_choice}'s policy", expanded=True):
            st.plotly_chart(fig)#, width='stretch')


        # # Pie chart of CorEx topic distribution for this policy (include first 3 words of each topic)
        # plt.figure(figsize=(6,6))
        # if corex_vals.sum() == 0:
        #     # no topics matched -- show a single grey slice
        #     plt.pie([1], labels=['No matching CorEx topics'], colors=['lightgrey'], autopct='%1.1f%%', startangle=140)
        # else:
        #     # try to get the first word for each CorEx topic from the fitted model
        #     first_words = []
        #     if 'corex_model' in globals():
        #         topics = corex_model.get_topics(n_words=1)
        #                 # topics is a list where each element is a list/tuple of (word, score, ...)
        #         for t in topics[:len(corex_vals)]:
        #             first_words.append(t[0][0] if t and len(t) > 0 else '')
        #     else:
        #         first_words = [''] * len(corex_vals)

        #     # build labels with the top 3 words for each CorEx topic (fallback to existing first_words)
        #     if 'corex_model' in globals():
        #         topics_top3 = corex_model.get_topics(n_words=5)
        #         label_words = []
        #         for t in topics_top3[:len(corex_vals)]:
        #             if t:
        #                         # each t is list of tuples like (word, score, ...)
        #                 words = [w for w, *rest in t][:3]# take top 3 words
        #                 label_words.append(', '.join(words))
        #             else:
        #                 label_words.append('')
        #     else:
        #         label_words = [fw or '' for fw in first_words]

        #     # Create pie chart without labels
        #     colors = plt.cm.Set3(np.linspace(0, 1, len(corex_vals)))
        #     wedges, texts, autotexts = plt.pie(corex_vals, autopct='%1.1f%%', startangle=140, colors=colors)

        #     # Create separate legend/key
        #     labels = [f"Group{i+1}: {label_words[i]} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)" if label_words[i] else f"Group{i+1} ({corex_vals[i]/corex_vals.sum()*100:.1f}%)" for i in range(len(corex_vals))]
        #     plt.legend(wedges, labels, title="Topics", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        # # plt.title(f"CorEx topic distribution for policy {idx}")
        # with st.expander(f"Topics found in {uni_choice}'s policy", expanded=True):
        #     st.pyplot(plt)
        # plt.close()



#-------------------------------------------------------------------------------------------------
# UPLOAD------------------------------------------------------------------------------------------
elif mode == "Upload":
    st.header("Upload & Compare")
    # upload user policy
    uploaded = st.file_uploader("Upload your policy (txt / pdf / docx)", type=['txt','pdf','docx'])
    uploaded_name, uploaded_text = read_uploaded_file(uploaded) if uploaded else ("", "")

    if uploaded:
        st.success(f"Uploaded: {uploaded_name}")
        with st.expander("Uploaded policy text", expanded=True):
            st.text_area("Uploaded policy text", value=uploaded_text, height=250, label_visibility="collapsed")
        # st.markdown("**Uploaded Policy Text**")
        # st.text_area("Uploaded policy text", value=uploaded_text, height=250, label_visibility="collapsed")

        with st.expander("Word Cloud for Uploaded Policy", expanded=True):
            wordcloud = generate_word_cloud(pd.Series([uploaded_text]), "Uploaded Policy")

        up_bs = basic_stats(uploaded_text)
        up_rd = readability_metrics(uploaded_text)
        with st.expander("Uploaded Policy's Stats", expanded=True):
            # st.subheader("Uploaded policy metrics")
            st.write(pd.DataFrame([{**up_bs, **up_rd}]).T.rename(columns={0:"value"}))



        corpus_texts = df['policy_text'].astype(str).tolist()
        corpus_meta = df['university'].astype(str).tolist()
        if len(corpus_texts) == 0:
            st.warning("No corpus policies available to compare against.")
        else:
            with st.expander("Similarity with other Policies", expanded=True):
            # st.markdown("**Similarity with other university policies (TF-IDF cosine)**")
                # vectorizer, X = build_tfidf_matrix(corpus_texts + [uploaded_text])
                # qvec = X[-1]
                # corpus_X = X[:-1]
                # sims = cosine_similarity(corpus_X, qvec).flatten()
                # sim_df = pd.DataFrame({
                #     "university": corpus_meta,
                #     "similarity": sims,
                #     "words": [len(re.findall(r"\w+", t)) for t in corpus_texts],
                #     "chars": [len(t) for t in corpus_texts]
                # }).sort_values("similarity", ascending=False).reset_index(drop=True)
                sim_df = tfidf_similarity(
                    corpus_texts=df['policy_text'].tolist(),
                    corpus_meta=df['university'].tolist(),
                    uploaded_text=uploaded_text,
                    build_tfidf_matrix_func=build_tfidf_matrix  # your existing TF-IDF builder
                )
                st.dataframe(sim_df.head(20))
       
            #add scatterPlot2col for similarity scores, based on TF-IDF similarity, as in sim_df, sims
            # Pre-build the TF-IDF matrix with all texts to ensure compatible dimensions
            all_texts = corpus_texts + [uploaded_text]
            vectorizer, X_all = build_tfidf_matrix(all_texts)
            uploaded_vec = X_all[-1]
            
            # scatterPlot2col(df, uploaded_text, "Uploaded Policy", 
            #                lambda t: cosine_similarity(
            #                    vectorizer.transform([str(t)]),
            #                    uploaded_vec.reshape(1, -1)
            #                ).flatten()[0] if str(t).strip() else 0.0,
            #                "Similarity Score", "#FAF7E6", "{:.3f}")
            
            # scatterPlot2col(df, sel_row, uni_choice, 
            #                lambda t: len(re.findall(r"\w+", str(t))), 
            #                "Word Count", "#EDF1FD", "{:,.0f}")



            # if HAS_SBERT:
            #     st.markdown("**Semantic similarity (sentence-transformers)**")
            #     with st.spinner("Computing SBERT embeddings..."):
            #         model = build_sbert_model()
            #         if model is not None:
            #             emb_sims = compute_embedding_sim(model, corpus_texts, uploaded_text)
            #             sim_df['sbert_similarity'] = emb_sims
            #             sim_df = sim_df.sort_values('sbert_similarity', ascending=False)
            #             st.dataframe(sim_df[['university','sbert_similarity','similarity']].head(20))
            #         else:
            #             st.info("SBERT model not available.")

            st.markdown("**Top matches (excerpt)**")
            top_n = sim_df.head(3)
            # st.dataframe(sim_df.head(3))
            
            for _, r in top_n.iterrows():
                #find corresponding value of col 'Filename' in df by searching university name from sim_df, same as filename var
                filename = df.loc[df['university'] == r['university'], 'filename'].values[0]

                # Get university name from col 'Filename' if exists, else from 'university'
                uni = r['University'] if 'University' in r else r['university']
                sel_row = sim_df.iloc[0]
                # #if logo of university exists in img/logos/filename.png display it
                # logofilename = os.path.splitext(sel_row.get('filename','').strip())[0]
                logofilename = os.path.splitext(filename.strip())[0]
                logo_path = f"img/{logofilename}.png"
                # st.write("logo:", logo_path)

                simscore = r['similarity']
                full_text = df.loc[df['university']==uni, 'policy_text'].values[0]
                # excerpt = full_text[:800].replace("\n", " ")

                with st.container(border=True):
                    col1, col2 = st.columns([3,1])
                    with col2:  
                        if os.path.exists(logo_path):
                            st.image(logo_path, width=200)  
                    with col1:
                    # st.markdown(f"**{uni}**, Similarity Score: {simscore:.3f}, out of 1.000")
                        st.markdown(f"**{uni}**")
                        #if score more then 0.85 color green, else if more than 0.5 color orange, else blue
                        if simscore >= 0.85:
                            scorecolor="green"
                            #icon up arrow
                            scoreicon=":material/arrow_upward:"
                        elif simscore >= 0.5:
                            scorecolor="orange"
                            scoreicon=":material/remove:"
                        else:
                            scorecolor="blue"
                            scoreicon=":material/arrow_downward:"
                        st.badge(f"Similarity Score: {simscore:.3f}, out of 1.000", icon=scoreicon, color=scorecolor)
                    # st.badge(f"Policies loaded: {len(display_df)}", icon=":material/check:", color="blue")
                        # st.write(excerpt + ("..." if len(full_text)>800 else ""))
                    # with st.expander("Uploaded policy text", expanded=True):
                    st.text_area(f"Policy text {uni}", value=full_text, height=250, label_visibility="collapsed")



            # Comparison table with top match
            top_match_uni = sim_df.iloc[0]['university']
            top_text = df.loc[df['university']==top_match_uni, 'policy_text'].values[0]
            top_bs = basic_stats(top_text)
            top_rd = readability_metrics(top_text)
            comp = pd.DataFrame({
                "metric": ["chars","words","sentences","avg_words_per_sentence","flesch_kincaid","flesch_reading_ease"],
                "uploaded": [up_bs['chars'], up_bs['words'], up_bs['sentences'], up_bs['avg_words_per_sentence'], up_rd['flesch_kincaid_grade'] if HAS_TEXTSTAT else None, up_rd['flesch_reading_ease'] if HAS_TEXTSTAT else None],
                "top_match": [top_bs['chars'], top_bs['words'], top_bs['sentences'], top_bs['avg_words_per_sentence'], top_rd.get('flesch_kincaid_grade'), top_rd.get('flesch_reading_ease')]
            })
            st.subheader(f"Comparison with top match: {top_match_uni}")
            st.table(comp.set_index('metric'))

            csv = sim_df.to_csv(index=False).encode('utf-8')

            # scatterPlot2col()
            uni_choice = "Uploaded Policy"
            sel_row = {
                'policy_text': uploaded_text
            }
            with st.expander("General Statistics", expanded=True):
                scatterPlot2col(df, sel_row, uni_choice, 
                           lambda t: len(re.findall(r"\w+", str(t))), 
                           "Word Count", "#EDF1FD", "{:,.0f}")
                scatterPlot2col(df, sel_row, uni_choice, 
                           lambda t: basic_stats(t)['avg_words_per_sentence'], 
                           "Words/Sentence", "#F8EDFD")
                scatterPlot2col(df, sel_row, uni_choice, 
                           lambda t: textstat.flesch_reading_ease(t), 
                           "Reading Ease", "#F2FDED")
                

            # Analyze uploaded policy with existing CorEx model and show topic pie chart
            try:
                if 'corex_model' not in globals():
                    p1 = Path("save/corex_results.pkl")
                    if p1.exists():
                        with open(p1, "rb") as f:
                            corex_model, doc_term_matrix, corex_policy_topic_means = pickle.load(f)
                        st.success("Loaded CorEx results from save/corex_results.pkl")
                    else:
                        corex_model = None
                        st.info("No CorEx pickle found in save")

                if corex_model is not None:
                    # Prepare anchors / topic count to match fitted model
                    anchors = topics_from_thematic_analysis
                    n_topics = len(anchors) + 1

                    # Chunk the uploaded policy the same way as the corpus
                    chunk_size = 60
                    chunks = chunk_policy(str(uploaded_text or ""), chunk_size)

                    # Build a CountVectorizer that uses the same vocabulary (words) the CorEx model was fitted with
                    try:
                        vocab = getattr(corex_model, "words", None)
                        if vocab is None:
                            # fallback: try attribute name used by some corex versions
                            vocab = getattr(corex_model, "vocab", None)
                        if vocab is None:
                            raise RuntimeError("Cannot find vocabulary on corex_model to vectorize uploaded text.")
                        vec = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=r"(?u)\b\w+\b")
                        doc_term_matrix_new = vec.transform(chunks) if chunks else None

                        # Transform chunks through CorEx to get topic activations per chunk
                        if doc_term_matrix_new is None or doc_term_matrix_new.shape[0] == 0:
                            topic_scores = np.zeros(n_topics)
                        else:
                            chunk_topic_dist = corex_model.transform(doc_term_matrix_new)  # shape: (n_chunks, n_topics)
                            # aggregate chunk-level topic activations to get a single vector for the uploaded policy
                            topic_scores = np.asarray(chunk_topic_dist).mean(axis=0).flatten()

                        # Ensure topic_scores length matches number of topics (pad/truncate if necessary)
                        if len(topic_scores) < n_topics:
                            topic_scores = np.pad(topic_scores, (0, n_topics - len(topic_scores)))
                        elif len(topic_scores) > n_topics:
                            topic_scores = topic_scores[:n_topics]


                        # Insert uploaded policy topic scores into df temporarily so corexResults_piechart can read them
                        # Ensure the CorEx topic columns exist
                        for i in range(n_topics):
                            col = f'CorEx_topic_{i}'
                            if col not in df.columns:
                                df[col] = 0.0

                        tmp_idx = "__uploaded_policy__"
                        # create a temporary row with the uploaded policy's topic scores
                        df.loc[tmp_idx] = { 'university': uni_choice }
                        for i in range(n_topics):
                            df.loc[tmp_idx, f'CorEx_topic_{i}'] = float(topic_scores[i])

                        # Use the existing helper to build the pie chart
                        fig = corexResults_piechart(corex_model, tmp_idx, numTopicsPerGroup=9)

                        # remove the temporary row to avoid mutating the main dataframe
                        try:
                            df.drop(index=tmp_idx, inplace=True)
                        except Exception:
                            pass
                        # Build labels from the fitted CorEx model (top 3 words per topic)
                        # labels = []
                        # if 'corex_model' in globals() and corex_model is not None:
                        #     topics_top3 = corex_model.get_topics(n_words=5)
                        #     for i, t in enumerate(topics_top3[:n_topics]):
                        #         if t:
                        #             words = [w for w, *rest in t][:3]
                        #             labels.append(', '.join(words))
                        #         else:
                        #             labels.append(f"Group{i+1}")
                        # else:
                        #     labels = [f"Group{i+1}" for i in range(n_topics)]

                        # # Plot pie chart of topic distribution for the uploaded policy
                        # if topic_scores.sum() == 0:
                        #     fig = go.Figure(data=[go.Pie(
                        #         labels=['No matching CorEx topics'],
                        #         values=[1],
                        #         marker=dict(colors=['lightgrey'])
                        #     )])
                        # else:
                        #     fig = go.Figure(data=[go.Pie(
                        #         labels=labels,
                        #         values=topic_scores,
                        #         textinfo='label+percent',
                        #         textposition='inside',
                        #         hovertemplate='<b>%{label}</b><br>Value: %{value:.3f}<br>Percent: %{percent}<extra></extra>'
                        #     )])
                        fig.update_layout(title="Topics found in uploaded policy")
                        st.plotly_chart(fig)#, width='stretch')

                    except Exception as e:
                        st.error(f"Failed to score uploaded policy with CorEx model: {e}")
                else:
                    st.info("No CorEx model available to analyze uploaded policy.")
 

                            # # Also show a small table of topic scores (percent)
                            # pct = (topic_scores / (topic_scores.sum() if topic_scores.sum() > 0 else 1)) * 100
                            # topic_df = pd.DataFrame({
                            #     "topic_index": [f"Topic {i+1}" for i in range(len(topic_scores))],
                            #     "top_words": labels,
                            #     "score": topic_scores,
                            #     "pct": np.round(pct, 2)
                            # }).sort_values("score", ascending=False).reset_index(drop=True)
                            # st.table(topic_df.head(10))
            except Exception as e:
                st.error(f"Error analyzing uploaded policy topics: {e}")

            with st.expander("Advance Options", expanded=False):
                st.download_button("Download similarity table (CSV)", data=csv, file_name="similarity_results.csv", mime="text/csv")

    else:
        st.info("Upload a policy to compare it with other universities' policies.")



#-------------------------------------------------------------------------------------------------
# ENFORCE------------------------------------------------------------------------------------------
elif mode == "Enforceablity":
    st.header("Checking Enforceablity")
    model = 'gpt-oss:120b'
    # client = Client(
    #     host="https://ollama.com",
    #     headers={'Authorization': 'Bearer ' + st.secrets["OLLAMA_API_KEY"]}
    # )

    propertyToCheck="is enforceable"
    roleSetting="You are a university's compliance officer and you are tasked with determining if the following university policy for the use of Generative AI "
    replyFormat = f" Grade the policy, providing a score from 0 to 5, where 5 is max positive value, in the following format ANSWER:<yourScore>, followed by a concise explanation in 1-2 sentences."
    rubricText= """ The marking rubric is as follows:
        Rating 5 – Fully enforceable: "The text gives specific, non-ambiguous rules (e.g. "students must not X", "students must Y"), along with who is responsible, how breaches are detected (e.g. checks, reporting routes), and what formal procedures and sanctions apply. There is a clear link to existing institutional processes (e.g. academic misconduct, ethics approval, data protection) so that every prohibited/required behaviour has a recognisable enforcement route";
        Rating 4 – Mostly enforceable: "The text gives clear "should/must" rules that could be mapped to existing processes (e.g. academic misconduct, GDPR breaches), but monitoring methods, thresholds, or decision steps are only partly spelled out. Enforcement is feasible but would rely on staff using standard procedures rather than instructions fully contained in the text";	
        Rating 3 – Partially enforceable: "The text mixes clear expectations ("students should not rely on XYZ when…") with advisory or reflective language ("students should reflect on the use of XYZ, "students should be aware that…"), and gives little or no detail on how issues would be identified or handled. Some clauses could be pursued under existing rules, but many expectations are difficult to verify or evidence, so enforcement would be uneven or highly discretionary";	
        Rating 2 – Weakly enforceable: "The text is largely aspirational or educational, focusing on awareness‑raising ("students should be aware that output can contain errors/bias", "there are better tools to surface research papers") with minimal reference to consequences or formal processes. Even where problematic behaviour is implied, there is no clear path from the wording to any specific institutional mechanism, making enforcement practically difficult except in extreme or obvious cases";		
        Rating 1 – Not enforceable in practice: "The text is purely normative or descriptive (e.g. describing a tool or process and how someone might use it) without prohibitions, requirements, or links to procedures, sanctions, or approval routes. It would be almost impossible to treat any part of the text as a formal basis for action because expectations are too vague, optional, or framed only as advice";		
        Rating 0 – Not applicable / absent: "The text is purely informational, conceptual, or not related to Gen AI use. No wording can reasonably be interpreted as creating enforceable obligations or prohibitions """
    

    SYSTEM_MESSAGE = f"""{roleSetting}{propertyToCheck}. {replyFormat}{rubricText}. The policy text is as follows: """
    accepted = st.checkbox("I have read and agree to the Terms of Use & Privacy Notice", value=False)
    with st.expander("Terms of Use & Privacy Notice)", expanded=not accepted):
        st.markdown("""
        By using this enforceability checking tool, you acknowledge and agree to the following:
        - The policy text you provide will be sent to an external service (Ollama Cloud) for analysis.
        - Ollama Cloud does not retain your data to ensure privacy and security. See https://ollama.com/blog/cloud-models for further details.
        - No personally identifiable information (PII) should be included in the policy text you submit.
        - The analysis results are generated by an AI model and should be reviewed by a qualified professional before making any compliance decisions.
        - We do not store or retain your policy text or analysis results beyond the duration of your session.
        - By proceeding, you consent to the processing of your policy text as described above.
                    
        - Enforceability checking tool uses these models 
                    gpt-oss:120b-cloud
                    gpt-oss:20b-cloud
                    deepseek-v3.1:671b-cloud          
        """)
        #allow user to select model
        # model = st.selectbox(
        #     "Select the AI model for analysis:",
        #     options=[
        #         'gpt-oss:120b',
        #         'gpt-oss:20b',
        #         'deepseek-v3.1:671b'
        #     ],
        #     index=0,
        #     help="Choose the AI model to use for enforceability analysis."
        # )
    if accepted:
        policy_text = st.text_area(
            "Enter or paste your policy text here to check enforceability:",
            height=200,
            placeholder="Paste the policy text here..."
        )
        
        if policy_text.strip():
            # Construct the full prompt with system message and user policy
            full_prompt = SYSTEM_MESSAGE + "\n\n" + policy_text
            
            st.info(f"✓ Policy text received ({len(policy_text)} characters). Ready for analysis.")
            
            # Placeholder for analysis button
            if st.button("Check Enforceability", type="primary"):
                with st.spinner("Analyzing policy enforceability..."):
                    response = client.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': full_prompt}],
                        stream=False,
                    )
                
                # Extract and parse the response
                response_text = response.message.content.strip()
                
                # Parse ANSWER:<score> format
                if "ANSWER:" in response_text:
                    parts = response_text.split("ANSWER:")
                    explanation_part = parts[1].strip()
                    
                    # Extract score (first character after ANSWER:)
                    score_char = explanation_part[0] if explanation_part else "?"
                    # Extract explanation (rest of the text)
                    explanation = explanation_part[1:].strip().lstrip(",").strip() if len(explanation_part) > 1 else ""
                    
                    # Map score to color and emoji
                    score_colors = {
                        "5": (":green", "🟢 5 – Fully Enforceable"),
                        "4": (":blue", "🔵 4 – Mostly Enforceable"),
                        "3": (":yellow", "🟡 3 – Partially Enforceable"),
                        "2": (":orange", "🟠 2 – Weakly Enforceable"),
                        "1": (":red", "🔴 1 – Not Enforceable"),
                        "0": (":gray", "⚪ 0 – Not Applicable"),
                    }
                    
                    color, label = score_colors.get(score_char, (":gray", "❓ Unknown"))
                    
                    # Display results
                    st.divider()
                    st.markdown("### Analysis Result")
                    
                    # col1, col2 = st.columns([1, 3])
                    # with col1:
                    st.markdown(f"{color}[**{label}**]")
                    # with col2:
                    if explanation:
                        st.markdown(f"_{explanation}_")
                    else:
                        st.info("No explanation available.")
                    
                    st.divider()
                else:
                    # Fallback if response doesn't contain ANSWER: format
                    st.write("**Raw Response:**")
                    st.write(response_text)
                
        else:
            st.warning("Please enter or paste a policy text to analyze.")

            # Display rubric using columns with markdown colors
        with st.expander("See Enforceability Rubric (click to expand)", expanded=False):
            rubric_items = [
                (":green[**🟢 5 – Fully Enforceable:**]", "Specific, non-ambiguous rules with clear responsibility, detection methods, and formal procedures linked to institutional processes."),
                (":blue[**🔵 4 – Mostly Enforceable:**]", "Clear 'should/must' rules mappable to existing processes, but monitoring methods and thresholds are only partly spelled out."),
                (":yellow[**🟡 3 – Partially Enforceable:**]", "Mix of clear expectations and advisory language; little detail on identification or handling; enforcement would be discretionary."),
                (":orange[**🟠 2 – Weakly Enforceable:**]", "Largely aspirational or educational focus on awareness-raising with minimal reference to consequences or institutional mechanisms."),
                (":red[**🔴 1 – Not Enforceable:**]", "Purely normative or descriptive without prohibitions, requirements, or links to procedures, sanctions, or approval routes."),
                (":gray[**⚪ 0 – Not Applicable:**]", "Purely informational, conceptual, or unrelated to Gen AI use; no enforceable obligations or prohibitions.")
            ]
            
            for rating, description in rubric_items:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(rating)
                with col2:
                    st.markdown(description)