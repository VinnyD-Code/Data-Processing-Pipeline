# run the following commands in terminal:
# pip install numpy
# pip install pandas
# pip install scikit-learn

import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def import_sheets(midas_sheet_name: str, gdm_sheet_name: str, country: str) -> (pd.DataFrame, pd.DataFrame):
    df_midas = pd.read_csv(midas_sheet_name)
    df_midas = df_midas[df_midas['SINGLETON_TYPE'] == 'MIDAS']
    df_midas = df_midas[df_midas['COUNTRY_CODE'] == country]

    df_gdm = pd.read_csv(gdm_sheet_name)
    df_gdm = df_gdm[df_gdm['SINGLETON_TYPE'] == 'GDM']
    df_gdm = df_gdm[df_gdm['COUNTRY_CODE'] == country]

    return df_midas, df_gdm

def modify_df(df_midas: pd.DataFrame, df_gdm: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_midas.rename(columns={'PACKAGED_PRODUCT_DESC':'Packaged_product_description', 'MRKT_NAME_NM':'MARKET_NAME'}, inplace=True)
    df_gdm.rename(columns={'PACKAGED_PRODUCT_DESC':'Packaged_product_description', 'MRKT_NAME_NM':'MARKET_NAME'}, inplace=True)

    keep_columns = ['PKPR_SOURCE_ID', 'Packaged_product_description', 'BRAND_NAME', 'SELLER_ORG', 'SECTOR_NAME', 'COUNTRY_CODE', 'MARKET_NAME']
    all_columns = list(df_midas.columns)

    keep_set = set(keep_columns)
    all_set = set(all_columns)
    entries_not_in_both = keep_set.symmetric_difference(all_set)
    drop_columns = list(entries_not_in_both)

    df_midas = df_midas.drop(drop_columns, axis=1)
    df_gdm= df_gdm.drop(drop_columns, axis=1)
    df_midas=df_midas.drop_duplicates()
    df_gdm=df_gdm.drop_duplicates()

    df_midas.reset_index(drop=True, inplace=True) 
    df_gdm.reset_index(drop=True, inplace=True) 

    mask = df_gdm['Packaged_product_description'].str.endswith(' 1')
    df_gdm.loc[mask, 'Packaged_product_description'] = df_gdm.loc[mask, 'Packaged_product_description'].str.rstrip(' 1')
  
    return df_midas, df_gdm


def split_df_columns(df_midas: pd.DataFrame, df_gdm: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    split_columns_midas = df_midas['Packaged_product_description'].str.split(expand=True)
    split_columns_midas = split_columns_midas.replace({None: np.nan})
    for column in split_columns_midas.columns:
        for i in range(len(split_columns_midas)):
            if pd.isnull(split_columns_midas[column][i]) or not any(char.isdigit() for char in split_columns_midas[column][i]) and (split_columns_midas[column][i] != 'MG' and split_columns_midas[column][i] != 'ML' and split_columns_midas[column][i] != '/ML' and split_columns_midas[column][i] != 'STK'and split_columns_midas[column][i] != 'IU'):
                split_columns_midas.at[i, column] = np.nan
    for i, row in split_columns_midas.iterrows():
        for j in range(len(row) - 1):
            if row[j] == '1' and any(char.isdigit() for char in str(row[j + 1])):
                split_columns_midas.at[i, j] = np.nan
    split_columns_midas = split_columns_midas.dropna(axis=1, how='all')
    split_columns_midas.columns = split_columns_midas.columns.astype(str)
    df_midas = pd.concat([df_midas, split_columns_midas], axis=1)

    split_columns_gdm = df_gdm['Packaged_product_description'].str.split(expand=True)
    split_columns_gdm = split_columns_gdm.replace({None: np.nan})
    for column in split_columns_gdm.columns:
        for i in range(len(split_columns_gdm)):
            if pd.isnull(split_columns_gdm[column][i]) or not any(char.isdigit() for char in split_columns_gdm[column][i]) and (split_columns_gdm[column][i] != 'MG' and split_columns_gdm[column][i] != 'ML' and split_columns_gdm[column][i] != '/ML' and split_columns_gdm[column][i] != 'STK'and split_columns_gdm[column][i] != 'IU'):
                split_columns_gdm.at[i, column] = np.nan
    for i, row in split_columns_gdm.iterrows():
        for j in range(len(row) - 1):
            if row[j] == '1' and any(char.isdigit() for char in str(row[j + 1])):
                split_columns_gdm.at[i, j] = np.nan
    split_columns_gdm = split_columns_gdm.dropna(axis=1, how='all')
    split_columns_gdm.columns = split_columns_gdm.columns.astype(str)
    df_gdm = pd.concat([df_gdm, split_columns_gdm], axis=1)

    return df_midas, df_gdm, split_columns_midas, split_columns_gdm
def clean_dosage(dosage):
    return ''.join(filter(lambda x: x.isdigit() or x in ('.', '+', 'x'), str(dosage)))

def combined_dosage_midas(row, split_columns_midas):
    dosages = [str(row[col]) for col in split_columns_midas.columns if not pd.isna(row[col])]
    dosages = [clean_dosage(dosage) for dosage in dosages]
    return ' '.join(dosages)

def combined_dosage_gdm(row, split_columns_gdm):
    dosages = [str(row[col]) for col in split_columns_gdm.columns if not pd.isna(row[col])]
    dosages = [clean_dosage(dosage) for dosage in dosages]
    return ' '.join(dosages)

def combined_features(row):
    features = [str(row['BRAND_NAME']), str(row['BRAND_NAME']), str(row['BRAND_NAME']),
                str(row['SELLER_ORG']), str(row['SECTOR_NAME']), str(row['COUNTRY_CODE']),
                str(row['combined_dosage'])]
    features = [feature for feature in features if pd.notna(feature)]
    return ' '.join(features)

def combine_features_and_dosages(df_midas: pd.DataFrame, df_gdm: pd.DataFrame, split_columns_midas: pd.DataFrame, split_columns_gdm: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_midas["combined_dosage"] = df_midas.apply(lambda row: combined_dosage_midas(row, split_columns_midas), axis=1)
    df_midas["combined_dosage"] = df_midas["combined_dosage"].str.strip().replace('  ', ' ')
    df_gdm["combined_dosage"] = df_gdm.apply(lambda row: combined_dosage_gdm(row, split_columns_gdm), axis=1)
    df_gdm["combined_dosage"] = df_gdm["combined_dosage"].str.strip().replace('  ', ' ')
    df_midas["combined_features"] = df_midas.apply(combined_features, axis=1)
    df_midas["combined_features"] = df_midas["combined_features"].str.strip().replace('  ', ' ')
    df_gdm["combined_features"] = df_gdm.apply(combined_features, axis=1)
    df_gdm["combined_features"] = df_gdm["combined_features"].str.strip().replace('  ', ' ')
    return df_midas, df_gdm
def country_sector_dfs(df_midas: pd.DataFrame, df_gdm: pd.DataFrame) -> (List[pd.DataFrame], List[pd.DataFrame]):
    df_midas_ch_retail = df_midas[(df_midas["COUNTRY_CODE"]=="CH") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_de_retail = df_midas[(df_midas["COUNTRY_CODE"]=="DE") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_es_retail = df_midas[(df_midas["COUNTRY_CODE"]=="ES") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_at_retail = df_midas[(df_midas["COUNTRY_CODE"]=="AT") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_fr_retail = df_midas[(df_midas["COUNTRY_CODE"]=="FR") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_it_retail = df_midas[(df_midas["COUNTRY_CODE"]=="IT") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_gb_retail = df_midas[(df_midas["COUNTRY_CODE"]=="GB") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_sa_retail = df_midas[(df_midas["COUNTRY_CODE"]=="SA") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_ie_retail = df_midas[(df_midas["COUNTRY_CODE"]=="IE") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_ae_retail = df_midas[(df_midas["COUNTRY_CODE"]=="AE") & (df_midas["SECTOR_NAME"]=="RETAIL")]
    df_midas_ch_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="CH") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_de_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="DE") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_es_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="ES") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_at_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="AT") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_fr_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="FR") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_it_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="IT") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_gb_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="GB") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_sa_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="SA") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_ie_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="IE") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_midas_ae_hospital = df_midas[(df_midas["COUNTRY_CODE"]=="AE") & (df_midas["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_ch_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="CH") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_de_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="DE") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_es_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="ES") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_at_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="AT") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_fr_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="FR") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_it_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="IT") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_gb_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="GB") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_sa_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="SA") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_ie_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="IE") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_ae_retail = df_gdm[(df_gdm["COUNTRY_CODE"]=="AE") & (df_gdm["SECTOR_NAME"]=="RETAIL")]
    df_gdm_ch_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="CH") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_de_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="DE") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_es_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="ES") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_at_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="AT") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_fr_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="FR") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_it_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="IT") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_gb_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="GB") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_sa_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="SA") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_ie_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="IE") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]
    df_gdm_ae_hospital = df_gdm[(df_gdm["COUNTRY_CODE"]=="AE") & (df_gdm["SECTOR_NAME"]=="HOSPITAL")]

    midas_list= [df_midas_ch_retail, df_midas_de_retail, df_midas_es_retail, df_midas_at_retail, df_midas_fr_retail, df_midas_it_retail, df_midas_gb_retail, df_midas_sa_retail, df_midas_ie_retail, df_midas_ae_retail, df_midas_ch_hospital, df_midas_de_hospital, df_midas_es_hospital, df_midas_at_hospital, df_midas_fr_hospital, df_midas_it_hospital, df_midas_gb_hospital, df_midas_sa_hospital, df_midas_ie_hospital, df_midas_ae_hospital]
    gdm_list = [df_gdm_ch_retail, df_gdm_de_retail, df_gdm_es_retail, df_gdm_at_retail, df_gdm_fr_retail, df_gdm_it_retail, df_gdm_gb_retail, df_gdm_sa_retail, df_gdm_ie_retail, df_gdm_ae_retail, df_gdm_ch_hospital, df_gdm_de_hospital, df_gdm_es_hospital, df_gdm_at_hospital, df_gdm_fr_hospital, df_gdm_it_hospital, df_gdm_gb_hospital, df_gdm_sa_hospital, df_gdm_ie_hospital, df_gdm_ae_hospital]
    midas_search = [df for df in midas_list if not df.empty]
    gdm_search = [df for df in gdm_list if not df.empty]


    return midas_search, gdm_search
def compare_midas_and_gdm(midas_search: List[pd.DataFrame], gdm_search: List[pd.DataFrame]) -> (pd.DataFrame):
    vectorizer = CountVectorizer()
    df_midas_new = pd.DataFrame()

    for midas, gdm in zip(midas_search, gdm_search):
        features_midas = vectorizer.fit_transform(midas['combined_features'])
        features_gdm = vectorizer.transform(gdm['combined_features'])
        cosine_sim_features = cosine_similarity(features_midas, features_gdm)

        dosage_midas = vectorizer.fit_transform(midas['combined_dosage'])
        dosage_gdm = vectorizer.transform(gdm['combined_dosage'])
        cosine_sim_dosage = cosine_similarity(dosage_midas, dosage_gdm)

        seller_midas = vectorizer.fit_transform(midas['SELLER_ORG'])
        seller_gdm = vectorizer.transform(gdm['SELLER_ORG'])
        cosine_sim_seller = cosine_similarity(seller_midas, seller_gdm)

        most_similar_indices = np.argmax(cosine_sim_features, axis=1)
        midas['Package_Similarity_Score'] = np.max(cosine_sim_features, axis=1)
        midas['Dosage_Similarity_Score'] = cosine_sim_dosage[np.arange(len(midas)), most_similar_indices]
        midas['Seller_Org_Similarity_Score'] = cosine_sim_seller[np.arange(len(midas)), most_similar_indices]
        midas['Most_Similar_GDM_PKPR_SOURCE_ID'] = gdm.iloc[most_similar_indices]['PKPR_SOURCE_ID'].values
        midas['GDM_Packaged_product_description'] = gdm.iloc[most_similar_indices]['Packaged_product_description'].values
        midas['GDM_Combined_Features'] = gdm.iloc[most_similar_indices]['combined_features'].values
        midas['GDM_Combined_Dosage'] = gdm.iloc[most_similar_indices]['combined_dosage'].values
        midas['GDM_Brand'] = gdm.iloc[most_similar_indices]['BRAND_NAME'].values
        midas['GDM_Sector'] = gdm.iloc[most_similar_indices]['SECTOR_NAME'].values
        midas['GDM_Country'] = gdm.iloc[most_similar_indices]['COUNTRY_CODE'].values
        midas['GDM_Seller_Org'] = gdm.iloc[most_similar_indices]['SELLER_ORG'].values
        df_midas_new = pd.concat([df_midas_new, midas], ignore_index=True)
    df_midas = df_midas_new.copy()
    return df_midas
def clean_df_midas(df_midas: pd.DataFrame) -> (pd.DataFrame):
    keep_columns = ['PKPR_SOURCE_ID', 'Packaged_product_description', 'BRAND_NAME', 'SECTOR_NAME', 'SELLER_ORG',
                     'COUNTRY_CODE', 'MARKET_NAME', 'Package_Similarity_Score', 'Dosage_Similarity_Score','Seller_Org_Similarity_Score', 'Most_Similar_GDM_PKPR_SOURCE_ID', 'GDM_Packaged_product_description', 'GDM_Brand', 'GDM_Sector', 'GDM_Country', 'GDM_Seller_Org']
    all_columns = list(df_midas.columns)

    keep_set = set(keep_columns)
    all_set = set(all_columns)
    entries_not_in_both = keep_set.symmetric_difference(all_set)
    drop_columns = list(entries_not_in_both)
    df_midas = df_midas.drop(drop_columns, axis=1)
    df_midas=df_midas.sort_index()

    return df_midas

def main():

    # input midas_sheet_name, gdm_sheet_name, and country
    midas_sheet_name = "PKPR Singletons - IT.csv"
    gdm_sheet_name = "PKPR Singletons - IT.csv"
    country = 'IT'

    
    df_midas, df_gdm = import_sheets(midas_sheet_name, gdm_sheet_name, country)
    df_midas, df_gdm = modify_df(df_midas, df_gdm)
    df_midas, df_gdm, split_columns_midas, split_columns_gdm = split_df_columns(df_midas, df_gdm)
    df_midas, df_gdm = combine_features_and_dosages(df_midas, df_gdm, split_columns_midas, split_columns_gdm)
    midas_search, gdm_search = country_sector_dfs(df_midas, df_gdm)
    df_midas = compare_midas_and_gdm(midas_search, gdm_search)
    df_midas = clean_df_midas(df_midas)
    df_midas.to_csv(f'output {country} {midas_sheet_name}', index=False)

if __name__ == "__main__":
    main()