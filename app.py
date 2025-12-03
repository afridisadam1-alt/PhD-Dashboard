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
# Build Graph Functions
# -------------------------
def build_graph_title(df):
    G = nx.Graph()
    for domain in df["second_level_domain"].unique():
        G.add_node(domain)
    rows = list(zip(df["second_level_domain"], df["title_english"]))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d1, t1 = rows[i]
            d2, t2 = rows[j]
            if d1 != d2 and t1 == t2 and t1 != "":
                G.add_edge(d1, d2)
    return G

def build_graph_country(df):
    G = nx.Graph()
    for domain in df["second_level_domain"].unique():
        G.add_node(domain)
    for domain, countries in zip(df["second_level_domain"], df["countries"]):
        for other_domain, other_countries in zip(df["second_level_domain"], df["countries"]):
            if domain != other_domain:
                common = set(countries).intersection(set(other_countries))
                if common:
                    G.add_edge(domain, other_domain, weight=len(common))
    return G

G_title = build_graph_title(df)
G_country = build_graph_country(df)

# -------------------------
# Sidebar Options
# -------------------------
st.sidebar.title("⚙ Graph Options")
graph_type = st.sidebar.radio("Select Graph Type:", ("Title Graph", "Country Graph"))
G = G_title if graph_type == "Title Graph" else G_country

# -------------------------
# Community Detection
# -------------------------
partition = community_louvain.best_partition(G)
communities = {}
for node, comm in partition.items():
    communities.setdefault(comm, set()).add(node)
top_communities = [comm for comm, size in Counter(partition.values()).most_common(3)]

# -------------------------
# Node Selection and Reset
# -------------------------
st.subheader("🖱️ Click Node to Highlight / Reset Selection")

if 'selected_node' not in st.session_state:
    # Auto-select most connected node
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
# PyVis Interactive Graph
# -------------------------
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.barnes_hut()
color_palette = sns.color_palette("husl", len(top_communities)).as_hex()

for node in G.nodes():
    # Highlight selected node and neighbors
    if node == selected_node:
        color = "#ff3333"
        size = 35
    elif selected_node and node in list(G.neighbors(selected_node)):
        color = "#33ff99"
        size = 25
    else:
        color = "#3399ff"
        size = 15

    # Top title and country
    top_title = df[df["second_level_domain"] == node]["title_english"].mode().values
    top_country = df[df["second_level_domain"] == node]["countries"].explode().mode().values

    net.add_node(
        node,
        label=node,
        color=color,
        size=size,
        title=f"<b>{node}</b><br>Top Title: {top_title[0] if len(top_title) > 0 else 'N/A'}"
              f"<br>Top Country: {top_country[0] if len(top_country) > 0 else 'N/A'}"
              f"<br>Neighbors: {len(list(G.neighbors(node)))}"
    )

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d.get("weight", 1))

# Add JS click events
net.set_options("""
var nodes = network.body.nodes;
network.on("click", function(params) {
    if (params.nodes.length > 0) {
        var node_id = params.nodes[0];
        var selected_node = document.getElementById('selected_node');
        selected_node.value = node_id;
        selected_node.dispatchEvent(new Event('change'));
    }
});
""")

html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
net.save_graph(html_file.name)
with open(html_file.name, "r", encoding="utf-8") as f:
    html_content = f.read()
html_content += '<input type="text" id="selected_node" onchange="console.log(this.value)" style="display:none">'
st.components.v1.html(html_content, height=750, scrolling=True)

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
