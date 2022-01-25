# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as pch
from matplotlib import rcParams

import scglue

# %%
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 5)

# %%
df_control = pd.concat([
    pd.read_csv(
        f"control/{dataset}+{dataset}/integration_consistency.csv"
    ).assign(Type="Same tissue", Combination=f"{dataset} + {dataset}")
    for dataset in [
        "10x-Multiome-Pbmc10k",
        "Chen-2019",
        "Ma-2020",
        "Muto-2021",
        "Yao-2021"
    ]
])

# %%
df_over_correction = pd.concat([
    pd.read_csv(
        f"over_correction/{dataset1}+{dataset2}/integration_consistency.csv"
    ).assign(Type="Different tissues", Combination=f"{dataset1} + {dataset2}")
    for dataset1, dataset2 in [
        ("10x-Multiome-Pbmc10k", "Muto-2021"),
        ("Muto-2021", "10x-Multiome-Pbmc10k"),
        ("Chen-2019", "Ma-2020"),
        ("Ma-2020", "Chen-2019"),
        ("Ma-2020", "Yao-2021"),
        ("Yao-2021", "Ma-2020"),
    ]
])

# %%
df = pd.concat([df_control, df_over_correction])
df.head()

# %%
palette = sns.color_palette()[:5] + sns.color_palette()[9:15]

# %%
fig, ax = plt.subplots()
ax = sns.scatterplot(x="n_meta", y="consistency", hue="Combination", style="Type", data=df, palette=palette, ax=ax)
ax = sns.lineplot(x="n_meta", y="consistency", hue="Combination", data=df, palette=palette, ax=ax, legend=False)
handles, labels = ax.get_legend_handles_labels()
placeholder = pch.Rectangle((0, 0), 1, 1, visible=False)
handles = [*handles[-3:], placeholder, *handles[:-3]]
labels = [*labels[-3:], "", *labels[:-3]]
leg = ax.legend(
    handles, labels,
    loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False
)
ax.axhline(y=0.05, c="darkred", ls="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Number of meta-cells")
ax.set_ylabel("Integration consistency score")
fig.savefig("integration_consistency.pdf")
