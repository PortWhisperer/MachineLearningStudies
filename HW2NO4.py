from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pandas import DataFrame as df
from pandas import read_csv
import numpy as np
import pandas as pd  # version 0.23.4
from matplotlib import pyplot as plt
from ggplot import *
import plotly


class get_PC:  # generates the principal components for a dataset
    def __init__(
        self, num_components, data, xcols
    ):  # xcols can change per dataset/x var
        self.data = data
        self.xcols = xcols
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
        print("Principal component altered")
        self.PC_values = data

    def get_PC_variances(self):
        self.variances = np.var(self.PC_values.filter(regex="xprincomp__"), axis=0)
        self.variance_ratos = self.variances / np.sum(self.variances)
        print(self.variances, "\n", self.variance_ratos)


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
# instead of number of purchases, we can also look at how many purcheses
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

# value used for plotting
# purchaseCounts["cluster"] = clust_obj.predicted_clusters


# ggplot(purchaseCounts, aes(x="factor(cluster)")) + geom_bar() + xlab("Cluster") + ylab(
#     "Customers\n(# in cluster)"
# )


# customer_clusters = purchaseCounts[["cust_name", "cluster", "x", "y", "z"]]
pc_obj = get_PC(3, purchaseCounts, xcols)
# customer_clusters = purchaseCounts[["cust_name", "cluster", "xprincomp__1", "xprincomp__2", "xprincomp__3"]]
customer_clusters = pd.concat(
    [
        pc_obj.data.loc[:, ["cust_name"]],
        clust_obj.predicted_clusters.loc[:, ["cluster"]],
        pc_obj.PC_values,
    ]
)

df_combined = pd.merge(df_cust, customer_clusters)
df_combined = pd.merge(df_deals, df_combined)

ggplot(df_combined, aes(x='xprincomp__1', y='xprincomp__2',color='cluster')) + \
geom_point(size=75) + \
ggtitle("Customers Grouped by Cluster")

# ggplot(df_combined, aes(x='x', y='y', color='cluster')) + \
# geom_point(size=75) + \
# geom_point(cluster_centers, size=500) + \
# ggtitle("Customers Grouped by Cluster")

# for i in range(1, nclusters+1):
#     cluster_of_interest='is_'+str(i)
#     df_combined[cluster_of_interest] = df_combined.cluster == i
#     print(df_combined.groupby(cluster_of_interest).Varietal.value_counts())
#     print(df_combined.groupby(cluster_of_interest)[['MinQty', 'Discount']].mean())


# plot_ly(x=temp, y=pressure, z=dtime, type="scatter3d", mode="markers", color=temp)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x =[1,2,3,4,5,6,7,8,9,10]
# y =[5,6,2,3,13,4,1,2,4,8]
# z =[2,3,3,3,5,7,9,11,9,10]


# ax.scatter(x, y, z, c='r', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

