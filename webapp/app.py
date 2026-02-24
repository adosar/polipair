import streamlit as st
import pandas as pd


@st.cache_data
def load_csv(**kwargs):
    return pd.read_csv(**kwargs)


### Sidebar ###
#st.sidebar.link_button(
#        ':material/deployed_code_account: Created by Antonios P. Sarikas',
#        'https://github.com/adosar',
#        )
#
#st.logo('images/polipair_logo.png', size='large')

### Titlebar ###
st.image('images/logo.png')
st.divider()
st.title('🎉 Welcome to PoLiPaiR')
'''
This is some about text about the application!
'''

### Pocket selection ###
st.header('🔬 Select pocket to analyze')
'''
Here are some pockets that you can choose from.
'''
col1, col2 = st.columns(2)
with col1:
    vol_slider = st.slider(
            'pocket volume',
            min_value=0,
            max_value=5_900,
            value=(0, 5_900),
            step=10,
            help='Add help hint',
            )
with col2:
    sur_slider = st.slider(
            'pocket surface',
            min_value=0,
            max_value=2_100,
            value=(0, 2_100),
            step=5,
            help='Add help hint',
            )

df_pockets = load_csv(
        filepath_or_buffer='../data/Final_Receptor_dataset.csv', 
        index_col='id'
        )

# Apply filtering based on slider values
df_filtered = df_pockets[
    df_pockets["pocket volume"].between(*vol_slider) &
    df_pockets["pocket surface"].between(*sur_slider)
]
st.dataframe(df_filtered)

#df_show[df_show['pocket_volume'] >= vol_slider && df_show['pocket surface'] >= sur_slider]]
#out = df_pockets.loc[:, ['pocket volume', 'pocket surface']].describe()

### Ligands upload ###
#st.header('📤 Upload your ligands')
#'''
#Your `.csv` should be properly formatted
#
#```csv
#hello,tag
#1,2
#```
#'''
#st.markdown('Text')
#uploaded_file = st.file_uploader('Upload a `.csv` file', type='csv', max_upload_size=1)
#df_ligands = pd.read_csv(uploaded_file)
#df_ligands
#
#### Results ###
#st.header('📊 Ligand scores')
