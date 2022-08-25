import streamlit as st
import pandas as pd
import numpy as np

st.write('Source dataframe')

with st.form(key='condition'):
    col1, col2 = st.columns(2)
    condition_src = col1.text_input(f'enter condition on the dataframe (src)' , key=3, value='')
    condition_tgt = col2.text_input(f'enter condition on the dataframe (tgt)' , key=3, value='')
    cond_submitted = st.form_submit_button('apply condition')

pd.options.display.max_colwidth = 70000

with open('datasets/evenki/src-train.txt') as src, open('datasets/evenki/tgt-train.txt') as tgt:
    src = [x.strip() for x in src.readlines()]
    tgt = [x.strip() for x in tgt.readlines()]

evenki = pd.DataFrame({'src':src, 'tgt':tgt})
if cond_submitted:
    #st.dataframe(evenki[evenki.tgt.str.contains('.+прихо.*')], width=100000)
    st.write(f'SRC condition: "{condition_src}"')
    st.write(f'TGT condition: "{condition_tgt}"')
    if condition_src == '' and condition_tgt == '':
        st.table(evenki.head(5))
    else:
        st.table(evenki[(evenki.src.str.contains(condition_src)) & (evenki.tgt.str.contains(condition_tgt))])
        
st.write('Notes:')
st.text_input(label='')
    
st.write('Add new line to the dataset:')
with open('datasets/evenki_augmented.txt', 'a') as augmented_df:
    with st.form(key='columns_in_form'):
        src, new_src, tgt = st.columns(3)
        src = src.text_input(f'src' , key=0, value='')
        new_src = new_src.text_input(f'new src', key=1, value='')
        tgt = tgt.text_input(f'tgt', key=2, value='')
        submitted = st.form_submit_button('add new line')
        if submitted:
            if src != '' and new_src != '' and tgt != '':
                augmented_df.write(f"{src}, {new_src}, {tgt}\n")
aug = pd.read_csv('datasets/evenki_augmented.txt')
st.write('Previously generatred examples:')
st.table(aug)

