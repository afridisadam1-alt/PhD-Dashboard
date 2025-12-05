import streamlit as st
import pandas as pd
import networkx as nx
import seaborn as sns
from collections import Counter
from pyvis.network import Network
import community as community_louvain
import tempfile
import gdown
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit.components.v1 as components

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(page_title="Disinfo Community Dashboard", layout="wide")
st.title("🕸️ Interactive Disinfo Community Dashboard")
st.write("Visualize community structures among news organizations.")

# ----------------------------------------------------
# Google Drive CSV link
# ----------------------------------------------------
file_id = "1mDEQ3y4wXX32EI24IwIMzZiGlHAaYC29"
csv_file = "EUvsDisinfoFullCleaned.csv"
url = f"https://drive.google.com/uc?id={file_id}"

@st.cache_resource
def load_dataset():
    if not os.path.exists(csv_file):
        gdown.download(url, csv_file, quiet=True)
    return pd.read_csv(csv_file, encoding='utf-8')

df = load_dataset()

# ----------------------------------------------------
# Keep only organisations spreading more than 5 titles
# ----------------------------------------------------
domain_counts = df["second_level_domain"].value_counts()
valid_domains = domain_counts[domain_counts > 5].index
df = df[df["second_level_domain"].isin(valid_domains)]

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------
st.subheader("📄 Dataset Preview")
st.markdown("""
    <style>
        [data-testid="stElementToolbar"] {
            display: none;
        }
        [data-testid="stDataFrameToolbar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.dataframe(df.head(), hide_index=True)


# ----------------------------------------------------
# Verify required columns
# ----------------------------------------------------
required_cols = ["title_english", "second_level_domain", "countries", "cleaned_keywords"]
if not all(col in df.columns for col in required_cols):
    st.error("CSV must contain 'title_english', 'second_level_domain', 'countries', and 'cleaned_keywords' columns.")
    st.stop()

# Preprocessing
df["title_english"] = df["title_english"].fillna("")
df["countries"] = df["countries"].fillna("").apply(lambda x: x.split(",") if isinstance(x, str) else [])
df["cleaned_keywords"] = df["cleaned_keywords"].fillna("")

# ----------------------------------------------------
# Caching Graph Builders
# ----------------------------------------------------
@st.cache_data
def build_graph_title(df):
    G = nx.Graph()
    for domain in df["second_level_domain"].unique():
        G.add_node(domain)
    title_groups = df[df["title_english"] != ""].groupby("title_english")["second_level_domain"].apply(list)
    for domains in title_groups:
        if len(domains) > 1:
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    G.add_edge(domains[i], domains[j])
    return G

@st.cache_data
def build_graph_country(df):
    G = nx.Graph()
    for domain in df["second_level_domain"].unique():
        G.add_node(domain)
    df_exp = df.explode("countries")
    df_exp = df_exp[df_exp["countries"] != ""]
    country_groups = df_exp.groupby("countries")["second_level_domain"].apply(list)
    for domains in country_groups:
        if len(domains) > 1:
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    if G.has_edge(domains[i], domains[j]):
                        G[domains[i]][domains[j]]["weight"] += 1
                    else:
                        G.add_edge(domains[i], domains[j], weight=1)
    return G

# ----------------------------------------------------
# Precompute Top Titles, Keywords, and Top Countries
# ----------------------------------------------------
@st.cache_data
def compute_top_info(df):
    top_titles = df.groupby("second_level_domain")["title_english"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"
    ).to_dict()
    return top_titles

@st.cache_data
def get_precomputed_keywords(df):
    keywords_dict = {}
    for domain, group in df.groupby("second_level_domain"):
        kws_series = group['cleaned_keywords'].dropna()
        if not kws_series.empty:
            keywords_list = kws_series.iloc[0].split(",")
            keywords_dict[domain] = [kw.strip() for kw in keywords_list[:5]]  # top 5 keywords
        else:
            keywords_dict[domain] = []
    return keywords_dict

@st.cache_data
def compute_top_countries(df, top_n=5):
    country_counts_dict = {}
    for domain, group in df.groupby("second_level_domain"):
        countries_list = group['countries'].explode()
        countries_list = [c for c in countries_list if c]  # remove empty
        if countries_list:
            top_countries = [c for c, _ in Counter(countries_list).most_common(top_n)]
            country_counts_dict[domain] = top_countries
        else:
            country_counts_dict[domain] = []
    return country_counts_dict

top_titles = compute_top_info(df)
top_keywords = get_precomputed_keywords(df)
country_counts = compute_top_countries(df, top_n=5)

# ----------------------------------------------------
# Sidebar Options
# ----------------------------------------------------
st.sidebar.title("⚙ Graph Options")
graph_type = st.sidebar.radio("Select Graph Type:", ("Title Graph", "Country Graph"))
TOP_COMMUNITIES = st.sidebar.slider("Top Communities", 1, 5, 3)
TOP_NODES_PER_COMM = st.sidebar.slider("Top Nodes per Community", 5, 50, 20)

# ----------------------------------------------------
# Build Graphs
# ----------------------------------------------------
G_title = build_graph_title(df)
G_country = build_graph_country(df)
G = G_title if graph_type == "Title Graph" else G_country

# ----------------------------------------------------
# Community Detection
# ----------------------------------------------------
partition = community_louvain.best_partition(G)
communities = {}
for node, comm in partition.items():
    communities.setdefault(comm, set()).add(node)
top_communities = [comm for comm, size in Counter(partition.values()).most_common(TOP_COMMUNITIES)]

# ----------------------------------------------------
# Filter Nodes (Top Communities)
# ----------------------------------------------------
nodes_to_show = []
for comm in top_communities:
    comm_nodes = [node for node, cid in partition.items() if cid == comm]
    comm_nodes_sorted = sorted(comm_nodes, key=lambda x: G.degree[x], reverse=True)[:TOP_NODES_PER_COMM]
    nodes_to_show.extend(comm_nodes_sorted)
H = G.subgraph(nodes_to_show)

# ----------------------------------------------------
# Node Selection Dropdown
# ----------------------------------------------------
st.subheader("🖱️ Select Node")
node_list = sorted(G.nodes())
selected_node = st.selectbox("Choose an Organisation:", options=[""] + node_list)
st.session_state.selected_node = selected_node

# ----------------------------------------------------
# Persistent Node Info Panel
# ----------------------------------------------------
st.subheader("ℹ️ Selected Node Details")
if selected_node:
    top_titles_node = ", ".join(df[df["second_level_domain"] == selected_node]["title_english"].head(3).tolist())
    top_countries = country_counts.get(selected_node, [])
    keywords_list = top_keywords.get(selected_node, [])
    
    st.markdown(f"### **{selected_node}**")
    st.write(f"**Top Titles:** {top_titles_node}")
    st.write(f"**Top Countries Discussed (Top 5):** {', '.join(top_countries) if top_countries else 'N/A'}")
    st.write(f"**Keywords:** {', '.join(keywords_list) if keywords_list else 'N/A'}")
    st.write(f"**Total Connections (Full Graph):** {G.degree[selected_node]}")
else:
    st.write("Click on a node in the graph or select from dropdown to see details.")

# ----------------------------------------------------
# PyVis Interactive Graph
# ----------------------------------------------------
st.subheader("🕸️ Interactive Network Graph")
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.barnes_hut()
color_palette = sns.color_palette("husl", TOP_COMMUNITIES).as_hex()
comm_to_color = {comm: color_palette[i] for i, comm in enumerate(top_communities)}

for node in H.nodes():
    if node == selected_node:
        color = "#ff3333"
        size = 35
    elif selected_node in H and node in H.neighbors(selected_node):
        color = "#33ff99"
        size = 25
    else:
        color = comm_to_color.get(partition[node], "#3399ff")
        size = 15

    keywords_html = ", ".join(top_keywords.get(node, []))
    countries_html = "<br>".join(country_counts.get(node, [])) if country_counts.get(node) else "N/A"
    top_titles_node = ", ".join(df[df["second_level_domain"] == node]["title_english"].head(3).tolist())

    tooltip_html = f"""
    <b>{node}</b><br>
    <b>Top Titles:</b> {top_titles_node}<br>
    <b>Keywords:</b> {keywords_html}<br>
    <b>Top Countries (Top 5):</b><br>{countries_html}<br>
    Connections: {len(list(H.neighbors(node)))}
    """

    net.add_node(
        node,
        label=node,
        color=color,
        size=size,
        title=tooltip_html
    )

for u, v, d in H.edges(data=True):
    net.add_edge(u, v, value=d.get("weight", 1))

html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
net.save_graph(html_file.name)
with open(html_file.name, "r", encoding="utf-8") as f:
    html_content = f.read()
components.html(html_content, height=750, scrolling=True)

# ----------------------------------------------------
# Static Matplotlib Plot
# ----------------------------------------------------
st.subheader("📷 Static Network Plot")
plt.figure(figsize=(14, 14))
pos = nx.spring_layout(H, seed=42, k=0.3, iterations=300)

node_colors = []
node_sizes = []
for node in H.nodes():
    node_comm = partition[node]
    if node == selected_node:
        node_colors.append("#ff3333")
        node_sizes.append(400)
    elif selected_node in H and node in H.neighbors(selected_node):
        node_colors.append("#33ff99")
        node_sizes.append(250)
    else:
        node_colors.append(comm_to_color.get(node_comm, "#3399ff"))
        node_sizes.append(100)

nx.draw(
    H,
    pos,
    with_labels=True,
    node_size=node_sizes,
    node_color=node_colors,
    edge_color='gray',
    width=0.5,
    font_size=10,
    font_weight='bold',
    edgecolors='w',
    alpha=0.9
)

patches = [mpatches.Patch(color=comm_to_color[comm], label=f"Community {i+1}") for i, comm in enumerate(top_communities)]
plt.legend(handles=patches, loc='best', fontsize=12)
plt.title(f"Network Graph - {graph_type}", fontsize=16)
st.pyplot(plt.gcf())
plt.clf()

# ----------------------------------------------------
# Graph Metrics and Top Communities
# ----------------------------------------------------
st.subheader("📊 Graph Metrics")
mod_score = community_louvain.modularity(partition, G)
st.write(f"**Graph Type:** {graph_type}")
st.write(f"**Modularity Score:** {mod_score:.4f}")

st.subheader("🏆 Top Communities")
for i, comm in enumerate(top_communities, 1):
    members = list(communities[comm])
    st.write(f"Community {i} (Size {len(members)}): {members}")
