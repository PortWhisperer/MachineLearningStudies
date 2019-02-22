from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pandas import read_csv
import numpy as np
import pandas as pd

# from ggplot import ggplot
import plotly
import random


# define some classes useful for K-Means and Principal component analysis
class get_PC:  # generates the principal components for a dataset
    def __init__(
        self, num_components, data, xcols
    ):  # xcols can change per dataset/x var
        self.data = data
        self.xcols = xcols
        self.pca_obj, pca = (
            PCA(n_components=num_components),
            PCA(n_components=num_components),
        )
        pca = PCA(n_components=num_components)  # instantiate PC object
        for pc in range(num_components):  # add PC values to the dataframe
            data["xprincomp__" + str(pc + 1)] = pca.fit_transform(data[xcols])[:, pc]
        data.reset_index()
        self.reshaped_data = data
        self.PC_values = data.filter(regex="xprincomp__")  # get cols of PC values
        print("Successfully calculated principal components")

    def alter_PC(self, num_components):  # overwrites self.dataPC
        data = self.data
        xcols = self.xcols
        pca = PCA(n_components=num_components)  # instantiate PC object
        for pc in range(num_components):  # add PC values to the dataframe
            data["xprincomp__" + str(pc + 1)] = pca.fit_transform(data[xcols])[:, pc]
        data.reset_index()
        self.PC_values = data
        print("Principal component altered")

    def get_PC_variances(
        self
    ):  # this method allows analyizing principal component performance
        self.variances = np.var(self.PC_values.filter(regex="xprincomp__"), axis=0)
        self.variance_ratios = self.variances / np.sum(self.variances)
        print(self.variances, self.variance_ratios)
        return (
            "PC# vs variance of y accounted for:",
            self.variances,
            "PCs ratio  of y accounted for:",
            self.variance_ratios,
        )


class cluster_Data:  # fit K-means object to data
    def __init__(self, num_clusters, data, xcols):
        self.num_clusters = num_clusters
        self.data = data
        self.xcols = xcols
        cluster = KMeans(n_clusters=num_clusters)
        # slice matrix so we only use the 0/1 indicator columns for clustering
        data["cluster"] = cluster.fit_predict(
            data[xcols]
        )  # have to run the function first
        self.predicted_clusters = data.loc[:, ["cluster"]]
        # debug

    def set_new_K(self, num_clusters):
        self.num_clusters = num_clusters
        data = self.data
        xcols = self.xcols
        cluster = KMeans(n_clusters=num_clusters)
        # slice matrix so we only use the 0/1 indicator columns for clustering
        self.predicted_clusters = cluster.fit_predict(data[xcols])
        print(
            self.predicted_clusters.value_counts()
        )  # can add plotting after this line
        print("Storing predicted clusters in self.predicted_clusters")
        return self.predicted_clusters


df_cust = read_csv(
    "https://raw.githubusercontent.com/ChicagoBoothML/MLClass/master/hw02/Wine_transactions.csv"
)
df_deals = read_csv(
    "https://raw.githubusercontent.com/ChicagoBoothML/MLClass/master/hw02/Wine_deals.csv"
)

df_cust.columns = ["cust_name", "deal_id"]
df_deals.columns = [
    "deal_id",
    "campaign",
    "Varietal",
    "MinQty",
    "Discount",
    "Origin",
    "Past_Peak",
]

df_cust["times_purchased"] = 1  # initializes times_purchased to 1 for all customers
df_combined = pd.merge(df_deals, df_cust)

# below number of purchases is examined
# instead of number of purchases, we can also look at how many purchases
# were "Past_Peak" and then cluster on that. Less interpretable?
# could just use customer name and map to quartile
# of how many past peak/pre-peak buys vs the group

# count number of purchases:
purchaseCounts = df_combined.pivot_table(
    index=["cust_name"], columns=["deal_id"], values="times_purchased"
)

purchaseCounts = purchaseCounts.fillna(0).reset_index()
xcols = purchaseCounts.columns[1:]

clust_obj = cluster_Data(7, purchaseCounts, xcols)
pc_obj = get_PC(3, purchaseCounts, xcols)  # hard-code to 3 principal components

# get each customer's k-means cluster assignment which we'll need for plotting purposes
cluster_labels = {
    "cluster" + str(int(i) + 1): i[0]
    for i in clust_obj.predicted_clusters[["cluster"]].values[
        range(len(clust_obj.predicted_clusters))
    ]
}

# dynamically create colors that we'll later map to each cluster
color = [
    "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    for i in range(len(cluster_labels))
]

# map randomly generated colors to each cluster.
# color choices are random, so some can be more difficult to see. Re-run or
# increase `size`  in `scatter` below to ease visualization

cols = clust_obj.predicted_clusters["cluster"].map(
    {clust: color for clust, color in zip(cluster_labels.values(), color)}
)

# assign hyper-parameters of 3d cluster plot
scatter = dict(
    mode="markers",
    name="y",
    type="scatter3d",
    x=pc_obj.PC_values.iloc[:, 0],
    y=pc_obj.PC_values.iloc[:, 1],
    z=pc_obj.PC_values.iloc[:, 2],
    marker=dict(size=2, color=cols),
)
clusters = dict(
    alphahull=7,
    name="y",
    opacity=0.1,
    type="mesh3d",
    x=pc_obj.PC_values.iloc[:, 0],
    y=pc_obj.PC_values.iloc[:, 1],
    z=pc_obj.PC_values.iloc[:, 2],
)
layout = dict(
    title="3d point clustering",
    scene=dict(
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
        zaxis=dict(zeroline=False),
    ),
)

# plot the data
fig = dict(data=[scatter, clusters], layout=layout)
plotly.offline.plot(fig, filename="3d point clustering")


# create a histogram with number of customers per cluster
# purchaseCounts["cluster"] = clust_obj.predicted_clusters
# ggplot(purchaseCounts, aes(x="factor(cluster)")) + geom_bar() + xlab("Cluster") + ylab(
#     "Customers\n(# in cluster)"
# )


# this code not needed for 3d cluster graph
# is used for 2d cluster graph, and
# later on for deeper analysis of the clusters' behavior
# customer_clusters = pd.concat(
#     [
#         pc_obj.data.loc[:, ["cust_name"]],
#         clust_obj.predicted_clusters.loc[:, ["cluster"]],
#         pc_obj.PC_values,
#     ],
#     axis=1,
# )
#
# df_combined = pd.merge(df_cust, customer_clusters)
# df_combined = pd.merge(df_deals, df_combined)

# ggplot(df_combined, aes(x='x', y='y', color='cluster')) + \
# geom_point(size=75) + \
# geom_point(cluster_centers, size=500) + \
# ggtitle("Customers Grouped by Cluster")

# for i in range(1, nclusters+1):
#     cluster_of_interest='is_'+str(i)
#     df_combined[cluster_of_interest] = df_combined.cluster == i
#     print(df_combined.groupby(cluster_of_interest).Varietal.value_counts())
#     print(df_combined.groupby(clus
# ter_of_interest)[['MinQty', 'Discount']].mean())
