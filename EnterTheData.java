import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from collections import Counter
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cuts import conductance
import community as community_louvain
from pyvis.network import Network
import tempfile
import os

st.set_page_config(page_title="Disinfo Community Dashboard", layout="wide")

st.title("🕸️ Disinfo Community Detection Dashboard")
st.write("Upload your dataset to visualize community structures among news organizations.")

# ----------------------------------------------------
# File Upload
# ----------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload CSV (must contain 'title_english' and 'second_level_domain')")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    if "title_english" in df.columns and "second_level_domain" in df.columns:
        df["title_english"] = df["title_english"].fillna("")

        # ----------------------------------------------------
        # Graph Construction
        # ----------------------------------------------------
        G = nx.Graph()

        for domain in df["second_level_domain"].unique():
            G.add_node(domain)

        # Add edges based on identical titles
        rows = list(zip(df["second_level_domain"], df["title_english"]))

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                d1, t1 = rows[i]
                d2, t2 = rows[j]
                if t1 == t2 and d1 != d2:
                    G.add_edge(d1, d2)

        st.success("Graph successfully constructed!")

        # ----------------------------------------------------
        # Community Detection
        # ----------------------------------------------------
        partition = community_louvain.best_partition(G)

        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, set()).add(node)
        community_list = list(communities.values())

        # ----------------------------------------------------
        # Modularity & Conductance
        # ----------------------------------------------------
        mod_score = modularity(G, community_list)

        conductance_scores = []
        for comm in community_list:
            try:
                c = conductance(G, comm)
                conductance_scores.append(c)
            except:
                continue

        avg_conductance = sum(conductance_scores) / len(conductance_scores) if conductance_scores else None

        st.subheader("📊 Metrics")
        st.write(f"**Modularity Score:** `{mod_score:.4f}`")

        if avg_conductance:
            st.write(f"**Average Conductance:** `{avg_conductance:.4f}`")
        else:
            st.write("⚠ No valid conductance scores computed")

        # ----------------------------------------------------
        # Top Communities
        # ----------------------------------------------------
        st.subheader("🏆 Top Communities")
        community_sizes = Counter(partition.values())
        top_communities = [comm for comm, size in community_sizes.most_common(3)]

        for i, comm in enumerate(top_communities, 1):
            members = list(communities[comm])
            st.write(f"### Community {i} (Size: {len(members)})")
            st.write(members)

        # ----------------------------------------------------
        # Interactive Network Graph (PyVis)
        # ----------------------------------------------------
        st.subheader("🕸️ Interactive Network Graph")

        H = G.subgraph([node for node, comm in partition.items() if comm in top_communities])

        net = Network(notebook=False, height="700px", width="100%", bgcolor="#222222", font_color="white")

        color_map = sns.color_palette("husl", len(top_communities)).as_hex()

        for node in H.nodes():
            c_id = partition[node]
            net.add_node(node, label=node, color=color_map[c_id % len(top_communities)], size=20)

        for edge in H.edges():
            net.add_edge(edge[0], edge[1])

        temp_dir = tempfile.gettempdir()
        html_path = os.path.join(temp_dir, "network.html")
        net.save_graph(html_path)

        with open(html_path, "r") as f:
            html = f.read()
            st.components.v1.html(html, height=750, scrolling=True)

        # ----------------------------------------------------
        # Matplotlib PNG Visualization
        # ----------------------------------------------------
        st.subheader("📷 Network Plot (Static Image)")

        plt.figure(figsize=(12, 12))
        colors = sns.color_palette("husl", len(top_communities))
        node_colors = [colors[partition[node] % len(top_communities)] for node in H.nodes()]
        pos = nx.spring_layout(H, seed=42, k=0.3, iterations=300)

        nx.draw(
            H, pos,
            with_labels=True,
            node_size=600,
            node_color=node_colors,
            edge_color='gray',
            width=0.5,
            font_size=9,
            font_weight='bold',
            edgecolors='white'
        )

        patches = [mpatches.Patch(color=colors[i], label=f"Community {i+1}") for i in range(len(top_communities))]
        plt.legend(handles=patches)
        st.pyplot(plt)

    else:
        st.error("CSV must contain 'title_english' and 'second_level_domain' columns.")
