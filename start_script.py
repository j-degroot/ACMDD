#!/usr/bin/env python
# coding: utf-8

# # T001 · Compound data acquisition (ChEMBL)
# 
# **Note:** This talktorial is a part of TeachOpenCADD, a platform that aims to teach domain-specific skills and to provide pipeline templates as starting points for research projects.
# 
# Authors:
# 
# - Svetlana Leng, CADD seminar 2017, Volkamer lab, Charité/FU Berlin 
# - Paula Junge, CADD seminar 2018, Volkamer lab, Charité/FU Berlin
# - Dominique Sydow, 2019-2020, [Volkamer lab, Charité](https://volkamerlab.org/)
# - Andrea Volkamer, 2020, [Volkamer lab, Charité](https://volkamerlab.org/)
# - Yonghui Chen, 2020, [Volkamer lab, Charité](https://volkamerlab.org/)

# __Talktorial T001__: This talktorial is part of the TeachOpenCADD pipeline described in the [first TeachOpenCADD paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0351-x), comprising of talktorials T001-T010.

# ## Aim of this talktorial
# 
# In this notebook, we will learn more about the ChEMBL database and how to extract data from ChEMBL, i.e. (compound, activity data) pairs for a target of interest. These data sets can be used for many cheminformatics tasks, such as similarity search, clustering or machine learning.
# 
# Our work here will include finding compounds which were tested against a certain target and filtering available bioactivity data.

# ### Contents in *Theory*
# 
# * ChEMBL database
#     * ChEMBL web services
#     * ChEMBL webresource client
# * Compound activity measures
#     * IC50 measure
#     * pIC50 value

# ### Contents in *Practical*
#     
# **Goal: Get a list of compounds with bioactivity data for a given target**
# 
# * Connect to ChEMBL database
# * Get target data (example: EGFR kinase)
#     * Fetch and download target data
#     * Select target ChEMBL ID
# * Get bioactivity data
#     * Fetch and download bioactivity data for target
#     * Preprocess and filter bioactivity data
# * Get compound data
#     * Fetch and download compound data
#     * Preprocess and filter compound data
# * Output bioactivity-compound data
#     * Merge bioactivity and compound data, and add pIC50 values
#     * Draw molecules with highest pIC50
#     * Freeze bioactivity data to ChEMBL 27
#     * Write output file

# ### References
# 
# * ChEMBL bioactivity database: [Gaulton *et al.*, <i>Nucleic Acids Res.</i> (2017), 45(Database issue), D945–D954](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5210557/)
# * ChEMBL web services: [Davies *et al.*, <i>Nucleic Acids Res.</i> (2015), <b>43</b>, 612-620](https://academic.oup.com/nar/article/43/W1/W612/2467881) 
# * [ChEMBL web-interface](https://www.ebi.ac.uk/chembl/)
# *  GitHub [ChEMBL webrescource client](https://github.com/chembl/chembl_webresource_client)
# * The EBI RDF platform: [Jupp *et al.*, <i>Bioinformatics </i> (2014), 30(9), 1338-9](https://www.ncbi.nlm.nih.gov/pubmed/24413672)
# * Info on half maximal inhibitory concentration: [(p)IC50](https://en.wikipedia.org/wiki/IC50)
# * [UniProt website](https://www.uniprot.org/)

# ## Theory

# ### ChEMBL database
# >"ChEMBL is a manually curated database of bioactive molecules with drug-like properties. It brings together chemical, bioactivity and genomic data to aid the translation of genomic information into effective new drugs." ([ChEMBL website](https://www.ebi.ac.uk/chembl/))
# 
# * Open large-scale bioactivity database
# * **Current data content (as of 09.2020, ChEMBL 27):**
#     * \>1.9 million distinct compounds
#     * \>16 million activity values
#     * Assays are mapped to ~13,000 targets
# * **Data sources** include scientific literature, PubChem bioassays, Drugs for Neglected Diseases Initiative (DNDi), BindingDB database, ...
# * ChEMBL data can be accessed via a [web-interface](https://www.ebi.ac.uk/chembl/), the [EBI-RDF platform](https://www.ncbi.nlm.nih.gov/pubmed/24413672) and the [ChEMBL webrescource client](https://github.com/chembl/chembl_webresource_client)

# #### ChEMBL web services
# 
# * RESTful web service
# * ChEMBL web service version 2.x resource schema: 
# 
# ![ChEMBL web service schema](images/chembl_webservices_schema_diagram.jpg)
# 
# *Figure 1:* 
# "[ChEMBL web service schema diagram](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4489243/figure/F2/). The oval shapes represent ChEMBL web service resources and the line between two resources indicates that they share a common attribute. The arrow direction shows where the primary information about a resource type can be found. A dashed line indicates the relationship between two resources behaves differently. For example, the `Image` resource provides a graphical based representation of a `Molecule`."
# Figure and description taken from: [<i>Nucleic Acids Res.</i> (2015), <b>43</b>, 612-620](https://academic.oup.com/nar/article/43/W1/W612/2467881).

# #### ChEMBL webresource client
# 
# * Python client library for accessing ChEMBL data
# * Handles interaction with the HTTPS protocol
# * Lazy evaluation of results -> reduced number of network requests

# ### Compound activity measures

# #### IC50 measure
# 
# * [Half maximal inhibitory concentration](https://en.wikipedia.org/wiki/IC50)
# * Indicates how much of a particular drug or other substance is needed to inhibit a given biological process by half
# 
# ![Wiki_Example_IC50_curve_demonstrating_visually_how_IC50_is_derived](images/Wiki_Example_IC50_curve_demonstrating_visually_how_IC50_is_derived.png)
# 
# *Figure 2:* Visual demonstration of how to derive an IC50 value: 
# (i) Arrange inhibition data on y-axis and log(concentration) on x-axis. (ii) Identify maximum and minimum inhibition. (iii) The IC50 is the concentration at which the curve passes through the 50% inhibition level. Figure ["Example IC50 curve demonstrating visually how IC50 is derived"](https://en.wikipedia.org/wiki/IC50#/media/File:Example_IC50_curve_demonstrating_visually_how_IC50_is_derived.png) by JesseAlanGordon is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

# #### pIC50 value
# 
# * To facilitate the comparison of IC50 values, which have a large value range and are given in different units (M, nM, ...), often pIC50 values are used
# * The pIC50 is the negative log of the IC50 value when converted to molar units: 
#     $ pIC_{50} = -log_{10}(IC_{50}) $, where $ IC_{50}$ is specified in units of M
# * Higher pIC50 values indicate exponentially greater potency of the drug
# * Note that the conversion can be adapted to the respective IC50 unit, e.g. for nM: $pIC_{50} = -log_{10}(IC_{50}*10^{-9})= 9-log_{10}(IC_{50}) $
# 
# Other activity measures:
# 
# Besides, IC50 and pIC50, other bioactivity measures are used, such as the equilibrium constant [KI](https://en.wikipedia.org/wiki/Equilibrium_constant) and the half maximal effective concentration  [EC50](https://en.wikipedia.org/wiki/EC50).

# ## Practical
# 
# In the following, we want to download all molecules that have been tested against our target of interest, the **epidermal growth factor receptor** ([**EGFR**](https://www.uniprot.org/uniprot/P00533)) kinase.

# ### Connect to ChEMBL database

# First, the ChEMBL webresource client is installed, and Python libraries are imported.

# In[22]:


get_ipython().system('pip install chembl_webresource_client')

import math
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm


# In[23]:


HERE = Path(_dh[-1])
DATA = HERE / "data"


# Next, we create resource objects for API access.

# In[24]:


targets_api = new_client.target
compounds_api = new_client.molecule
bioactivities_api = new_client.activity


# In[25]:


type(targets_api)


# ### Get target data (EGFR kinase)
# 
# * Get UniProt ID of the target of interest (EGFR kinase:  [P00533](http://www.uniprot.org/uniprot/P00533)) from [UniProt website](https://www.uniprot.org/)
# * Use UniProt ID to get target information
# 
# Select a different UniProt ID, if you are interested in another target.

# In[26]:


uniprot_id = "P00533"


# #### Fetch target data from ChEMBL

# In[27]:


# Get target information from ChEMBL but restrict it to specified values only
targets = targets_api.get(target_components__accession=uniprot_id).only(
    "target_chembl_id", "organism", "pref_name", "target_type"
)
print(f'The type of the targets is "{type(targets)}"')


# #### Download target data from ChEMBL
# 
# The results of the query are stored in `targets`, a `QuerySet`, i.e. the results are not fetched from ChEMBL until we ask for it (here using `pandas.DataFrame.from_records`).
# 
# More information about the `QuerySet` datatype:
# 
# > QuerySets are lazy – the act of creating a QuerySet does not involve any database activity. You can stack filters together all day long, and Django will actually not run the query until the QuerySet is evaluated. 
# ([querysets-are-lazy](https://docs.djangoproject.com/en/3.0/topics/db/queries/#querysets-are-lazy))

# In[28]:


targets = pd.DataFrame.from_records(targets)
targets


# #### Select target (target ChEMBL ID)
# 
# After checking the entries, we select the first entry as our target of interest:
# 
# `CHEMBL203`: It is a single protein and represents the human Epidermal growth factor receptor (EGFR, also named erbB1) 

# In[29]:


target = targets.iloc[0]
target


# Save selected ChEMBL ID.

# In[30]:


chembl_id = target.target_chembl_id
print(f"The target ChEMBL ID is {chembl_id}")
# NBVAL_CHECK_OUTPUT


# ### Get bioactivity data
# 
# Now, we want to query bioactivity data for the target of interest.

# #### Fetch bioactivity data for the target from ChEMBL

# In this step, we fetch the bioactivity data and filter it to only consider
# 
# * human proteins, 
# * bioactivity type IC50, 
# * exact measurements (relation `'='`), and
# * binding data (assay type `'B'`).

# In[31]:


bioactivities = bioactivities_api.filter(
    target_chembl_id=chembl_id, type="IC50", relation="=", assay_type="B"
).only(
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "molecule_chembl_id",
    "type",
    "standard_units",
    "relation",
    "standard_value",
    "target_chembl_id",
    "target_organism",
)

print(f"Length and type of bioactivities object: {len(bioactivities)}, {type(bioactivities)}")


# In[37]:


get_ipython().run_line_magic('pinfo', 'bioactivities_api')


# Each entry in our bioactivity set holds the following information:

# In[32]:


print(f"Length and type of first element: {len(bioactivities[0])}, {type(bioactivities[0])}")
bioactivities[0]


# #### Download bioactivity data from ChEMBL

# Finally, we download the `QuerySet` in the form of a `pandas` `DataFrame`. 
# 
# > **Note**: This step should not take more than 2 minutes, if so try to rerun all cells starting from _"Fetch bioactivity data for the target from ChEMBL"_ or read this message below:
# 
# <details>
#     
# <summary>Load a local version of the data (in case you encounter any problems while fetching the data)</summary>
#     
# If you experience difficulties to query the ChEMBL database, we also provide the resulting dataframe you will construct in the cell below. If you want to use the saved version, use the following code instead to obtain `bioactivities_df`:
#   
# ```python
# # replace first line in cell below with this other line
# bioactivities_df = pd.read_csv(DATA / "EGFR_bioactivities_CHEMBL27.csv.zip", index_col=0)
# ```
# 
# </details>

# In[34]:


bioactivities_df = pd.read_csv(DATA / "EGFR_bioactivities_CHEMBL27.csv.zip", index_col=0)
print(f"DataFrame shape: {bioactivities_df.shape}")
bioactivities_df.head()


# Note that the first two rows describe the same bioactivity entry; we will remove such artifacts later during the deduplication step. Note also that we have columns for `standard_units`/`units` and `standard_values`/`values`; in the following, we will use the standardized columns (standardization by ChEMBL), and thus, we drop the other two columns.
# 
# If we used the `units` and `values` columns, we would need to convert all values with many different units to nM:

# In[35]:


bioactivities_df["type"].unique()


# In[36]:


bioactivities_df.drop(["units", "value"], axis=1, inplace=True)
bioactivities_df.head()


# #### Preprocess and filter bioactivity data
# 
# 1. Convert `standard_value`'s datatype from `object` to `float`
# 2. Delete entries with missing values
# 3. Keep only entries with `standard_unit == nM`
# 4. Delete duplicate molecules
# 5. Reset `DataFrame` index
# 6. Rename columns

# **1. Convert datatype of "standard_value" from "object" to "float"**
# 
# The field `standard_value` holds standardized (here IC50) values. In order to make these values usable in calculations later on, convert values to floats.

# In[51]:


bioactivities_df.dtypes


# In[52]:


bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})
bioactivities_df.dtypes


# **2. Delete entries with missing values**
# 
# Use the parameter `inplace=True` to drop values in the current `DataFrame` directly.

# In[53]:


bioactivities_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")


# In[55]:


# ?pd.DataFrame.dropna


# **3. Keep only entries with "standard_unit == nM"** 
# 
# We only want to keep bioactivity entries in `nM`, thus we remove all entries with other units.

# In[56]:


print(f"Units in downloaded data: {bioactivities_df['standard_units'].unique()}")
print(
    f"Number of non-nM entries:\
    {bioactivities_df[bioactivities_df['standard_units'] != 'nM'].shape[0]}"
)


# In[57]:


bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]
print(f"Units after filtering: {bioactivities_df['standard_units'].unique()}")


# In[58]:


print(f"DataFrame shape: {bioactivities_df.shape}")


# **4. Delete duplicate molecules**
# 
# Sometimes the same molecule (`molecule_chembl_id`) has been tested more than once, in this case, we only keep the first one.
# 
# Note other choices could be to keep the one with the best value or a mean value of all assay results for the respective compound.

# In[59]:


bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
print(f"DataFrame shape: {bioactivities_df.shape}")


# **5. Reset "DataFrame" index**
# 
# Since we deleted some rows, but we want to iterate over the index later, we reset the index to be continuous.

# In[60]:


bioactivities_df.reset_index(drop=True, inplace=True)
bioactivities_df.head()


# **6. Rename columns**

# In[61]:


bioactivities_df.rename(
    columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True
)
bioactivities_df.head()


# In[62]:


print(f"DataFrame shape: {bioactivities_df.shape}")


# We now have a set of **5575** molecule ids with respective IC50 values for our target kinase.

# ### Get compound data
# 
# We have a `DataFrame` containing all molecules tested against EGFR (with the respective measured bioactivity). 
# 
# Now, we want to get the molecular structures of the molecules that are linked to respective bioactivity ChEMBL IDs. 

# #### Fetch compound data from ChEMBL
# 
# Let's have a look at the compounds from ChEMBL which we have defined bioactivity data for: We fetch compound ChEMBL IDs and structures for the compounds linked to our filtered bioactivity data.

# In[63]:


compounds_provider = compounds_api.filter(
    molecule_chembl_id__in=list(bioactivities_df["molecule_chembl_id"])
).only("molecule_chembl_id", "molecule_structures")


# #### Download compound data from ChEMBL
# 
# Again, we want to export the `QuerySet` object into a `pandas.DataFrame`. Given the data volume, **this can take some time.** For that reason, we will first obtain the list of records through `tqdm`, so we get a nice progress bar and some ETAs. We can then pass the list of compounds to the DataFrame.

# In[64]:


compounds = list(tqdm(compounds_provider))


# In[65]:


compounds_df = pd.DataFrame.from_records(
    compounds,
)
print(f"DataFrame shape: {compounds_df.shape}")


# In[66]:


compounds_df.head()


# #### Preprocess and filter compound data
# 
# 1. Remove entries with missing entries
# 2. Delete duplicate molecules (by molecule_chembl_id)
# 3. Get molecules with canonical SMILES

# **1. Remove entries with missing molecule structure entry**

# In[67]:


compounds_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")


# **2. Delete duplicate molecules**

# In[68]:


compounds_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")


# **3. Get molecules with canonical SMILES**
# 
# So far, we have multiple different molecular structure representations. We only want to keep the canonical SMILES.

# In[69]:


compounds_df.iloc[0].molecule_structures.keys()


# In[70]:


canonical_smiles = []

for i, compounds in compounds_df.iterrows():
    try:
        canonical_smiles.append(compounds["molecule_structures"]["canonical_smiles"])
    except KeyError:
        canonical_smiles.append(None)

compounds_df["smiles"] = canonical_smiles
compounds_df.drop("molecule_structures", axis=1, inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")


# Sanity check: Remove all molecules without a canonical SMILES string.

# In[71]:


compounds_df.dropna(axis=0, how="any", inplace=True)
print(f"DataFrame shape: {compounds_df.shape}")


# ### Output (bioactivity-compound) data
# **Summary of compound and bioactivity data**

# In[72]:


print(f"Bioactivities filtered: {bioactivities_df.shape[0]}")
bioactivities_df.columns


# In[73]:


print(f"Compounds filtered: {compounds_df.shape[0]}")
compounds_df.columns


# #### Merge both datasets
# 
# Merge values of interest from `bioactivities_df` and `compounds_df` in an `output_df` based on the compounds' ChEMBL IDs (`molecule_chembl_id`), keeping the following columns:
# 
# * ChEMBL IDs: `molecule_chembl_id`
# * SMILES: `smiles`
# * units: `units`
# * IC50: `IC50`

# In[74]:


# Merge DataFrames
output_df = pd.merge(
    bioactivities_df[["molecule_chembl_id", "IC50", "units"]],
    compounds_df,
    on="molecule_chembl_id",
)

# Reset row indices
output_df.reset_index(drop=True, inplace=True)

print(f"Dataset with {output_df.shape[0]} entries.")


# In[75]:


output_df.dtypes


# In[76]:


output_df.head(10)


# #### Add pIC50 values

# As you can see the low IC50 values are difficult to read (values are distributed over multiple scales), which is why we convert the IC50 values to pIC50.

# In[77]:


def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)
    return pIC50_value


# In[78]:


# Apply conversion to each row of the compounds DataFrame
output_df["pIC50"] = output_df.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)


# In[79]:


output_df.head()


# #### Draw compound data
# 
# Let's have a look at our collected data set.
# 
# First, we plot the pIC50 value distribution

# In[45]:


output_df.hist(column="pIC50")


# In the next steps, we add a column for RDKit molecule objects to our `DataFrame` and look at the structures of the molecules with the highest pIC50 values. 

# In[46]:


# Add molecule column
PandasTools.AddMoleculeColumnToFrame(output_df, smilesCol="smiles")


# In[47]:


# Sort molecules by pIC50
output_df.sort_values(by="pIC50", ascending=False, inplace=True)

# Reset index
output_df.reset_index(drop=True, inplace=True)


# Show the three most active molecules, i.e. molecules with the highest pIC50 values.

# In[48]:


output_df.drop("smiles", axis=1).head(3)


# In[49]:


# Prepare saving the dataset: Drop the ROMol column
output_df = output_df.drop("ROMol", axis=1)
print(f"DataFrame shape: {output_df.shape}")


# #### Freeze output data to ChEMBL 27
# 
# This is a technical step: Usually, we would continue to work with the dataset that we just created (latest dataset). 
# 
# However, here on the TeachOpenCADD platform, we prefer to freeze the dataset to a certain ChEMBL releases (i.e. [ChEMBL 27](http://doi.org/10.6019/CHEMBL.database.27)), 
# so that this talktorial and other talktorials downstream in our CADD pipeline do not change in the future (helping us to maintain the talktorials).

# <div class="alert alert-block alert-info">
# 
# <b>Note:</b> If you prefer to run this notebook on the latest dataset or if you want to use it for another target, please comment the cell below.
# 
# </div>

# In[50]:


# Disable this cell to unfreeze the dataset
output_df = pd.read_csv(
    DATA / "EGFR_compounds_ea055ef.csv", index_col=0, float_precision="round_trip"
)
output_df.head()


# In[51]:


print(f"DataFrame shape: {output_df.shape}")
# NBVAL_CHECK_OUTPUT


# #### Write output data to file
# 
# We want to use this bioactivity-compound dataset in the following talktorials, thus we save the data as `csv` file. 
# Note that it is advisable to drop the molecule column (which only contains an image of the molecules) when saving the data.

# In[52]:


output_df.to_csv(DATA / "EGFR_compounds.csv")
output_df.head()


# In[53]:


print(f"DataFrame shape: {output_df.shape}")
# NBVAL_CHECK_OUTPUT


# ## Discussion

# In this tutorial, we collected bioactivity data for our target of interest from the ChEMBL database. 
# We filtered the data set in order to only contain molecules with measured IC50 bioactivity values. 
# 
# Be aware that ChEMBL data originates from various sources. Compound data has been generated in different labs by different people all over the world. Therefore, we have to be cautious with the predictions we make using this data set. It is always important to consider the source of the data and consistency of data production assays when interpreting the results and determining how much confidence we have in our predictions.
# 
# In the next tutorials, we will filter our acquired data by Lipinski's rule of five and by unwanted substructures. Another important step would be to *clean* the molecular data. As this is not shown in any of our talktorials (yet), we would like to refer to the [Standardiser library](https://github.com/flatkinson/standardiser) or [MolVS](https://molvs.readthedocs.io/en/latest/) as useful tools for this task.

# ## Quiz

# * We have downloaded in this talktorial molecules and bioactivity data from ChEMBL. What else is the ChEMBL database useful for?
# * What is the difference between IC50 and EC50?
# * What can we use the data extracted from ChEMBL for?
