import streamlit as st
import pandas as pd

from utils import ligands_to_desc, load_csv, load_model, prepare_inputs

logo_link = r'https://raw.githubusercontent.com/adosar/polipair/master/webapp/images/logo.png'

# Configuration
# =======================================
st.set_page_config(
        page_title='PoLiPaiR',
        #page_icon='🤖',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Get Help': 'https://github.com/adosar/polipair/discussions',
            'Report a bug': 'https://github.com/adosar/polipair/issues',
            }
        )
check_icon = ':material/check_circle:'
info_icon = ':material/info:'

# Sidebar
# =======================================
st.sidebar.header('About')
st.sidebar.markdown(f'PoLiPaiR demo built with **:red[Streamlit]**.', unsafe_allow_html=True)
st.sidebar.link_button(
        ':material/deployed_code_account: Created by Antonios P. Sarikas',
        'https://github.com/adosar',
        )
st.sidebar.subheader('Disclaimer')
st.sidebar.caption(f'''
This app uses AI models to make predictions. Please note that these predictions
are intended for informational purposes only. By using this app, you agree that
any decisions made based on the predictions are at your own risk. The creator of
this app is not responsible for any outcomes resulting from your use of the
predictions provided. For any high-risk or critical decisions, consider
consulting with a qualified professional or using additional resources to verify
the predictions.

Uploaded files are stored in memory and get deleted immediately as soon as
they’re not needed anymore.  For more information about data file handling,
please refer to the official [Streamlit
documentation](https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted).
Use at your own risk.
''', unsafe_allow_html=True)
st.logo(logo_link, size='large')
st.sidebar.subheader('How to Cite')
st.sidebar.write(rf"""
If you find PoLiPaiR useful, please
consider citing us.""", unsafe_allow_html=True)

bibtex = r"""To be added"""
citation_text = r"""To be added"""
cols = st.sidebar.columns(2)
with cols[0]:
    with st.popover('BibTeX', icon=':material/article:'):
        st.code(bibtex, language=None)

with cols[1]:
    with st.popover('Other', icon=':material/article:'):
        st.code(citation_text, language=None)

# Titlebar
# =======================================
st.markdown(rf"""
<h1 align="center">
  <img alt="Logo"
  src="{logo_link}" width=50%/>
</h1>
""", unsafe_allow_html=True)
st.divider()
st.title('🎉 Welcome to PoLiPaiR')
st.markdown("""
**PoLiPaiR** is a machine learning model designed to **evaluate the
fitness between a protein pocket and a candidate ligand**. By leveraging
biochemical and physicochemical features extracted from the pocket and the
ligand, respectively, PoLiPaiR predicts how well a given pair is likely to
match. Under the hood, a trained ML model transforms these features into a
quantitative compatibility score, enabling fast ranking of pocket-ligand pairs.

PoLiPaiR supports three core workflows:

1. **Score a single pocket–ligand pair**.
2. **Rank a list of candidate pockets for a given target ligand**.
3. **Rank a list of candidate ligands for a given target pocket**.

At the moment, the demo provides only the second option — you can select a
target protein pocket from a library of over 16,000 pockets, upload a CSV file
containing your candidate ligands, and receive a ranked list based on their
predicted fitness scores.
""")



# Ligands upload
# =======================================
st.header('📤 Upload your ligands', divider=True)
st.markdown("""
Please upload a `.csv` file containing your candidate ligands.

The file **must include a column named `smiles`** containing the SMILES representation of each ligand.

Example of a valid `.csv` file:

```csv
smiles
CC
CCC
CCO
```
""")

uploaded_file = st.file_uploader(
        'Upload a `.csv` file containing SMILES of ligands',
        type='csv',
        max_upload_size=1
        )
if uploaded_file:
    df_ligands = pd.read_csv(uploaded_file)
    with st.expander('Show features of uploaded ligands'):
        X_ligands = ligands_to_desc(list(df_ligands.smiles))
        st.write(X_ligands)

# Pocket selection
# =======================================
st.header('🔬 Select pocket to analyze', divider=True)
st.markdown("""
Browse the available protein pockets and select the one you would like to analyze.

Use the volume and surface sliders to filter and narrow down the list to pockets that match your criteria.
""")

col1, col2 = st.columns(2)
with col1:
    vol_slider = st.slider(
            'pocket volume',
            min_value=0,
            max_value=5_900,
            value=(0, 5_900),
            step=10,
            )
with col2:
    sur_slider = st.slider(
            'pocket surface',
            min_value=0,
            max_value=2_100,
            value=(0, 2_100),
            step=5,
            )

df_pockets = load_csv(
        filepath_or_buffer='data/Final_Receptor_dataset.csv',
        index_col='id'
        )

df_filtered = df_pockets[  # Apply filtering based on slider values
    df_pockets['pocket volume'].between(*vol_slider) &
    df_pockets['pocket surface'].between(*sur_slider)
]
event = st.dataframe(
        df_filtered.loc[:, ['pocket volume', 'pocket surface']],
        on_select='rerun',
        selection_mode='single-row'
        )

if event.selection.rows:  # Show features of selected pocket
    with st.expander('Show features of selected pocket'):
        index = event.selection.rows[0]
        X_pocket = df_filtered.iloc[[index]]
        st.write(X_pocket.iloc[0])  # Visualize as Series

# Results
# =======================================
st.header('🔮 Predictions', divider=True)

steps_completed = 0
if event.selection.rows:
    steps_completed += 1
if uploaded_file:
    steps_completed += 1

total_steps = 2
progress = steps_completed / total_steps
st.progress(progress)
st.caption(f"{steps_completed} / {total_steps} steps completed")

if steps_completed < total_steps:
    st.info('Complete all steps to generate predictions.', icon=info_icon)

#st.write(
#    f"{check_icon if event.selection.rows else '⬜'} Select a protein pocket to proceed"
#)
#st.write(
#    f"{check_icon if uploaded_file else '⬜'} Upload ligand CSV file to proceed"
#)
#if not event.selection.rows:
#    st.warning('Please select a protein pocket to proceed.', icon=':material/warning:')
#if not uploaded_file:
#    st.warning('Please upload a CSV file containing ligand SMILES to proceed.', icon=':material/warning:')
if event.selection.rows and uploaded_file:
    st.success(
            "Pocket selected and ligands uploaded successfully. Below are the predictions of PoLiPaiR.",
            icon=":material/check_circle:"
            )
    model = load_model()
    X = prepare_inputs(X_pocket, X_ligands)
    preds = model.predict_proba(X)[:, 1]  # Probability of "fit"
    X['Score'] = preds
    with st.expander('What **Score** represents?'):
        st.info("""
        **Score** represents the predicted probability that the pocket–ligand
        pair is compatible.

        It takes values between 0 and 1, with **higher values indicating better
        fitness**.

        Use it to prioritize ligands for your selected pocket.
        """, icon=info_icon)
    st.write(X.loc[:, 'Score'])

# Copyright
# =======================================
st.caption('')
st.caption(':material/copyright: Copyright 2026, Antonios P. Sarikas.')
