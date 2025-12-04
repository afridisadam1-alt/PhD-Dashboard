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

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Disinfo Community Dashboard", layout="wide")
st.title("🕸️ Interactive Disinfo Community Dashboard")
st.write("Visualize community structures among news organizations.")

# -------------------------
# Download dataset from Google Drive
# -------------------------
file_id = "1mDEQ3y4wXX32EI24IwIMzZiGlHAaYC29"
csv_file = "EUvsDisinfoFullCleaned.csv"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(csv_file):
    gdown.download(url, csv_file, quiet=False)

df = pd.read_csv(csv_file, encoding='utf-8')
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Verify required columns
# -------------------------
required_cols = ["title_english", "second_level_domain", "countries"]
if not all(col in df.columns for col in required_cols):
    st.error("CSV must contain 'title_english', 'second_level_domain', and 'countries' columns.")
    st.stop()

# Preprocessing
df["title_english"] = df["title_english"].fillna("")
df["countries"] = df["countries"].fillna("").apply(lambda x: x.split(",") if isinstance(x, str) else [])

# -------------------------
# Caching Graph Builders
# -------------------------
@st.cache_data
def build_graph_title_fast(df):
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
def build_graph_country_fast(df):
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

# -------------------------
# Precompute top titles and countries per domain
# -------------------------
@st.cache_data
def compute_top_info(df):
    top_titles = df.groupby("second_level_domain")["title_english"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A").to_dict()
    top_countries = df.explode("countries").groupby("second_level_domain")["countries"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A").to_dict()
    return top_titles, top_countries

# -------------------------
# Sidebar Options
# -------------------------
st.sidebar.title("⚙ Graph Options")
graph_type = st.sidebar.radio("Select Graph Type:", ("Title Graph", "Country Graph"))
TOP_COMMUNITIES = st.sidebar.slider("Top Communities", 1, 5, 3)
TOP_NODES_PER_COMM = st.sidebar.slider("Top Nodes per Community", 5, 50, 20)

# -------------------------
# Build Graphs
# -------------------------
G_title = build_graph_title_fast(df)
G_country = build_graph_country_fast(df)
G = G_title if graph_type == "Title Graph" else G_country
top_titles, top_countries = compute_top_info(df)

# -------------------------
# Community Detection
# -------------------------
partition = community_louvain.best_partition(G)
communities = {}
for node, comm in partition.items():
    communities.setdefault(comm, set()).add(node)
top_communities = [comm for comm, size in Counter(partition.values()).most_common(TOP_COMMUNITIES)]

# -------------------------
# Node Selection
# -------------------------
st.subheader("🖱️ Node Selection")
if 'selected_node' not in st.session_state:
    most_connected_node = max(G.degree, key=lambda x: x[1])[0] if G.number_of_nodes() > 0 else ""
    st.session_state.selected_node = most_connected_node

selected_node = st.text_input(
    "Selected Node (type or click on graph):",
    value=st.session_state.selected_node,
    key='node_input'
)

if st.button("Reset Selection"):
    st.session_state.selected_node = ""
    selected_node = ""

# -------------------------
# Filter Nodes for Top Communities
# -------------------------
nodes_to_show = []
for comm in top_communities:
    comm_nodes = [node for node, cid in partition.items() if cid == comm]
    comm_nodes_sorted = sorted(comm_nodes, key=lambda x: G.degree[x], reverse=True)[:TOP_NODES_PER_COMM]
    nodes_to_show.extend(comm_nodes_sorted)
H = G.subgraph(nodes_to_show)

# -------------------------
# PyVis Interactive Graph
# -------------------------
st.subheader("🕸️ Interactive Network Graph")
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.barnes_hut()
color_palette = sns.color_palette("husl", TOP_COMMUNITIES).as_hex()
comm_to_color = {comm: color_palette[i] for i, comm in enumerate(top_communities)}

for node in H.nodes():
    if node == selected_node:
        color = "#ff3333"
        size = 35
    elif selected_node and node in list(H.neighbors(selected_node)):
        color = "#33ff99"
        size = 25
    else:
        color = comm_to_color.get(partition[node], "#3399ff")
        size = 15

    top_title = top_titles.get(node, "N/A")
    top_country = top_countries.get(node, "N/A")

    net.add_node(
        node,
        label=node,
        color=color,
        size=size,
        title=f"<b>{node}</b><br>Top Title: {top_title}"
              f"<br>Top Country: {top_country}"
              f"<br>Neighbors: {len(list(H.neighbors(node)))}"
    )

for u, v, d in H.edges(data=True):
    net.add_edge(u, v, value=d.get("weight", 1))

html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
net.save_graph(html_file.name)
with open(html_file.name, "r", encoding="utf-8") as f:
    html_content = f.read()
st.components.v1.html(html_content, height=750, scrolling=True)

# -------------------------
# Static Matplotlib Plot
# -------------------------
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
    elif selected_node and node in list(H.neighbors(selected_node)):
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
st.pyplot(plt)

# -------------------------
# Metrics and Top Communities
# -------------------------
st.subheader("📊 Graph Metrics")
mod_score = community_louvain.modularity(partition, G)
st.write(f"**Graph Type:** {graph_type}")
st.write(f"**Modularity Score:** {mod_score:.4f}")

st.subheader("🏆 Top Communities")
for i, comm in enumerate(top_communities, 1):
    members = list(communities[comm])
    st.write(f"Community {i} (Size {len(members)}): {members}")
