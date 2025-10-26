"""
Streamlit app for exploring Gen-AI policies and comparing user-uploaded policy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt


# Optional libraries (graceful fallback)
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

def generate_word_cloud(texts: pd.Series):
    from wordcloud import WordCloud, STOPWORDS
    combined_text = " ".join(texts.astype(str).tolist())
    # extend stopwords with some common artifacts from scraped policies
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 'http', 'www', 'com', 'org', 'edu', 'page', 'policy', 'policytext', 'generative', 'may', 'take', 'top'])

    wc = WordCloud(width=1200, height=600, background_color='white',
                   stopwords=stopwords, collocations=False).generate(combined_text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Combined Word Cloud for All Policies', fontsize=16)
    
    # Display in streamlit
    st.pyplot(fig)
    plt.close(fig)

    # allow download as image
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    st.download_button("Download Word Cloud", buf.getvalue(), "wordcloud.png")

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
compute_button = st.sidebar.button("(Re)compute corpus metrics")


if len(df) > 0:
    with st.spinner("Computing corpus metrics..."):
        metrics_df = compute_corpus_metrics(df)
else:
    metrics_df = pd.DataFrame()

# Replace previous two-column UI with a mode-based layout
# Add a sidebar mode selector: Explore, Upload, About
mode = st.sidebar.radio("Mode", options=["About", "Explore", "Analyse", "Upload"], index=0)


# #------------------------------------------------------------------------------------------------
# # ANALYSE------------------------------------------------------------------------------------------
if mode == "Analyse":
    st.text("select university to analyze")

    # # prepare display_df for listing/searching (same logic as before)
    display_df = df.copy()
    # if 'policy_text' in display_df.columns:
    #     display_df['words'] = display_df['policy_text'].apply(lambda t: len(re.findall(r"\w+", str(t))))
    #     display_df['chars'] = display_df['policy_text'].apply(lambda t: len(str(t)))
    #     display_df['avg_word_length'] = display_df['chars'] / display_df['words'].replace(0, np.nan)
    #     display_df['flesch_kincaid'] = display_df['policy_text'].apply(lambda t: textstat.flesch_kincaid_grade(str(t)))
    #     display_df['flesch_reading_ease'] = display_df['policy_text'].apply(lambda t: textstat.flesch_reading_ease(str(t)))     


    # Sidebar: select a university (or All)
    # uni_choice = st.selectbox("Choose university", options=["All universities"] + df['university'].tolist())


    uni_options = [""] + display_df['university'].tolist()
    # uni_options = ["All universities"] + display_df['university'].tolist()
    uni_choice = st.selectbox("Choose university to view", options=uni_options, index=0)    
    sel_df = df[df['university'] == uni_choice]

    if sel_df.empty:
        st.warning(" ")
        # st.warning("Selected university not found in dataset.")
    else:
        sel_row = sel_df.iloc[0]
        st.subheader(f"{uni_choice}")
        if sel_row.get('url'):
            st.write(sel_row.get('url'))
            
        col1, col2 = st.columns(2)
            
        with col1:
            st.markdown("**Raw policy text**")
            st.text_area("Raw policy text", value=sel_row.get('policy_text',''), height=300)
            
        with col2:
            st.markdown("**Metrics**")
            bs = basic_stats(str(sel_row.get('policy_text','')))
            rd = readability_metrics(str(sel_row.get('policy_text','')))
            st.write(pd.DataFrame([ {**bs, **rd} ]).T.rename(columns={0:"value"}))


#------------------------------------------------------------------------------------------------
# EXPLORE------------------------------------------------------------------------------------------
elif mode == "Explore":
    st.text("Exploring policies")
    # prepare display_df for listing/searching (same logic as before)
    display_df = df.copy()
    if 'policy_text' in display_df.columns:
        display_df['words'] = display_df['policy_text'].apply(lambda t: len(re.findall(r"\w+", str(t))))
        display_df['chars'] = display_df['policy_text'].apply(lambda t: len(str(t)))

    #ideas: 
    # show list of universities as containers with name of university and explore button
    # show logos on UK map

    # search box and min words filter visible in main pane
    q = st.text_input("Search universities or policy text", value="")
    # min_words = st.slider("Min words", min_value=0, max_value=5000, value=0, step=50)
    min_words = 0
    if q:
        try:
            mask = display_df['university'].str.contains(q, case=False, na=False) | display_df['policy_text'].str.contains(q, case=False, na=False, regex=True)
        except Exception:
            mask = display_df['university'].str.contains(q, case=False, na=False)
        display_df = display_df[mask]
    display_df = display_df[display_df.get('words', 0) >= min_words]
    


    st.subheader("Statistics & evaluation overview")
    if len(metrics_df) == 0:
        st.info("No corpus metrics available. Add a dataset file to /mnt/data or upload one policy to compare.")
    else:
        st.subheader(f"Policies loaded: {len(display_df)}")
        st.dataframe(display_df[['university', 'words', 'chars']].head(200))
            #basic stats
        st.markdown("**Basic statistics across universities**")
            # Top longest policies
        top_long = metrics_df.sort_values("words", ascending=False).head(10)
        st.write("Top 10 longest (by words):")
        st.table(top_long[['university','words','chars','flesch_kincaid']].reset_index(drop=True))

            # Keyword aggregation
        kw_cols = [c for c in metrics_df.columns if c.startswith('kw_')]
        if kw_cols:
            kw_sum = metrics_df[kw_cols].sum().sort_values(ascending=False)
            st.subheader("Keyword mentions across universities (counts)")
            st.table(kw_sum.rename_axis('keyword').reset_index().rename(columns={0:'count'}))

            # Download metrics
            
            st.download_button("Download metrics (CSV)", data=metrics_df.to_csv(index=False).encode('utf-8'), file_name="corpus_metrics.csv", mime="text/csv")

    st.subheader("Combined word cloud")
        # Generate and display word cloud
        generate_word_cloud(display_df['policy_text'])
        # Generate and display word cloud
        wordcloud = generate_word_cloud(display_df['policy_text'])
        st.image(wordcloud, caption="Combined Word Cloud")



#-------------------------------------------------------------------------------------------------
# UPLOAD------------------------------------------------------------------------------------------
elif mode == "Upload":
    st.header("Upload & Compare")
    # upload user policy
    uploaded = st.file_uploader("Upload your policy (txt / pdf / docx)", type=['txt','pdf','docx'])
    uploaded_name, uploaded_text = read_uploaded_file(uploaded) if uploaded else ("", "")

    # The uploader is available in the sidebar by default; show upload preview and comparisons here
    if uploaded:
        st.success(f"Uploaded: {uploaded_name}")
        st.markdown("**Preview**")
        st.text_area("Uploaded policy preview", value=uploaded_text[:5000], height=250)
        up_bs = basic_stats(uploaded_text)
        up_rd = readability_metrics(uploaded_text)
        st.subheader("Uploaded policy metrics")
        st.write(pd.DataFrame([{**up_bs, **up_rd}]).T.rename(columns={0:"value"}))

        corpus_texts = df['policy_text'].astype(str).tolist()
        corpus_meta = df['university'].astype(str).tolist()
        if len(corpus_texts) == 0:
            st.warning("No corpus policies available to compare against.")
        else:
            st.markdown("**Similarity (TF-IDF cosine)**")
            vectorizer, X = build_tfidf_matrix(corpus_texts + [uploaded_text])
            qvec = X[-1]
            corpus_X = X[:-1]
            sims = cosine_similarity(corpus_X, qvec).flatten()
            sim_df = pd.DataFrame({
                "university": corpus_meta,
                "similarity": sims,
                "words": [len(re.findall(r"\w+", t)) for t in corpus_texts],
                "chars": [len(t) for t in corpus_texts]
            }).sort_values("similarity", ascending=False).reset_index(drop=True)
            st.dataframe(sim_df.head(20))

            if HAS_SBERT:
                st.markdown("**Semantic similarity (sentence-transformers)**")
                with st.spinner("Computing SBERT embeddings..."):
                    model = build_sbert_model()
                    if model is not None:
                        emb_sims = compute_embedding_sim(model, corpus_texts, uploaded_text)
                        sim_df['sbert_similarity'] = emb_sims
                        sim_df = sim_df.sort_values('sbert_similarity', ascending=False)
                        st.dataframe(sim_df[['university','sbert_similarity','similarity']].head(20))
                    else:
                        st.info("SBERT model not available.")

            st.markdown("**Top matches (excerpt)**")
            top_n = sim_df.head(3)
            for _, r in top_n.iterrows():
                uni = r['university']
                simscore = r['similarity']
                full_text = df.loc[df['university']==uni, 'policy_text'].values[0]
                excerpt = full_text[:800].replace("\n", " ")
                st.markdown(f"**{uni}** — TF-IDF similarity {simscore:.3f}")
                st.write(excerpt + ("..." if len(full_text)>800 else ""))

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
            st.download_button("Download similarity table (CSV)", data=csv, file_name="similarity_results.csv", mime="text/csv")

    else:
        st.info("Upload a policy in the sidebar to compare it with corpus policies.")



#-------------------------------------------------------------------------------------------------
# ABOUT------------------------------------------------------------------------------------------
else: #about
    st.header("About the HEI Gen-AI Policy Toolkit")
    st.markdown("""
    This app allows users to explore a corpus of UK Higher Education Institutions' policies on Generative AI, as well as upload and compare their own policies.

    **Features:**
    - Explore existing policies with search and filtering.
    - View basic statistics and readability metrics for each policy.
    - Upload your own policy document (txt, rtf, pdf, docx) and compare it against other universities using TF-IDF and semantic similarity.
    - Download computed metrics and similarity results for further analysis.

    **Usage:**
    - Use the sidebar to navigate between modes: About, Explore, Analyse, and Upload.
    - In Explore mode, search for universities or keywords within policies.
    - In Analyse mode, select a university to view detailed metrics.
    - In Upload mode, upload your policy document to see how it compares with existing policies.


    Developed as part of the the BRA(AI)N Project (Building Resilience and Accountability in Artificial Intelligence Navigation) 
    """)
    # st.expander("About BRA(AI)N Project", expanded=False, icon=None, width="stretch")
    with st.expander("About BRA(AI)N Project"):
        st.write('''
    BRA(AI)N - Building Resilience and Accountability in Artificial Intelligence Navigation

    Funder(s): UKRI EPSRC and AI Security Institute (AISI)
    Start Date: February 2025
    End Date: February 2026


    *Overview*
    This research project seeks to advance understanding of artificial intelligence (AI) and inform educational policy by examining the role of Generative Artificial Intelligence (GenAI) in UK Higher Education (HE). With a focus on technologies such as ChatGPT and DeekSeek, the project explores how these tools are being embedded into academic practice. It considers not only their potential advantages and risks but also the preparedness and resilience of users navigating this evolving digital landscape. At its core, the project is committed to producing robust, evidence-based insights that support the ethical, effective and responsible integration of AI across the HE sector. The findings will be presented in a detailed report and shared with key stakeholders, including universities and relevant government departments. To extend its reach and impact, the study will also generate peer-reviewed academic publications, contributing to ongoing scholarly discussions and future policy development.

    *Purpose and Research Focus*
    This project welcomes the participation of a wide range of individuals—university students, academic staff, administrators, policymakers, regulators and AI industry experts—to explore the use and implications of GenAI in HE. It seeks to understand the motivations behind the adoption of these tools and to examine the impact when such technologies become unreliable or inaccessible. By drawing on participants’ real-world experiences with GenAI in teaching, learning and assessment, the study aims to uncover practical insights into their value, limitations and ethical implications. These perspectives are essential to shaping responsible and evidence-informed policies for the integration of GenAI technologies in the HE landscape.

    *Get Involved*
    Ethically approved by the University of East Anglia (Ref: ETH2425-1497), this interdisciplinary research will adopt a mixed-methods approach, conducted in two distinct phases:

    *Stakeholder Engagement*
    If you want to be involved in the stakeholders' engagement, please get in touch by emailing F.Liza@uea.ac.uk
        ''')