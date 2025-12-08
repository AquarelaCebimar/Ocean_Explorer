
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:46:23 2025

@author: anapiazzaf + Gemini

"""

# ----------------------------- PACOTES NECESSÁRIOS -----------------------------
# Seção de importação de todas as bibliotecas Python necessárias para o funcionamento do aplicativo.

import streamlit as st  # Importa a biblioteca Streamlit para criar a interface web interativa.
import pandas as pd  # Importa o Pandas para manipulação e análise de dados, especialmente DataFrames.
import numpy as np  # Importa o NumPy para operações numéricas eficientes, especialmente com arrays.
import matplotlib.pyplot as plt  # Importa o Matplotlib para a criação de gráficos estáticos.
from matplotlib.patches import Polygon  # Importa a classe Polygon para desenhar o triângulo de mistura no diagrama T-S.
import gsw  # Importa a biblioteca Gibbs SeaWater (GSW) para cálculos precisos de propriedades da água do mar (TEOS-10).
import seaborn as sns  # Importa o Seaborn para criar paletas de cores mais bonitas para os gráficos.
import io  # Importa o módulo io para manipular buffers de bytes em memória, usado para criar arquivos zip.
import zipfile  # Importa o módulo zipfile para criar arquivos .zip dinamicamente.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Importa métricas de regressão para comparar dados de CTD.
from matplotlib.ticker import ScalarFormatter
from scipy.stats import linregress # Importa a função de regressão linear para os novos gráficos.

#
# ----------------------------- ESTILO E CONFIGURAÇÕES -----------------------------
# Seção para definir a aparência visual dos gráficos e da página do Streamlit.

# Dicionário que centraliza as definições de cores para criar um tema visual consistente (modo escuro).

STYLE_CONFIG = {
    'facecolor':'#0E1117', # Cor de fundo geral da figura
    'ax_facecolor':'#2C3445', # Cor de fundo dos eixos
    'textcolor':'#FAFAFA', # Cor do texto nos eixos e legendas
    'gridcolor':'#5A5A5A', # Cor das linhas de grid
    'spinecolor':'#8A8A8A', # Cor das bordas do gráfico
    'n2_line_color':'#00FF00', # Cor da linha de N² (Brunt-Väisälä)
    'beta_line_color': '#FF8C00'  # Cor para a nova linha do perfil de beta (laranja escuro).
}
# Atualiza configurações globais do Matplotlib
plt.rcParams.update({
    'figure.facecolor': STYLE_CONFIG['facecolor'], # Fundo da figura
    'axes.facecolor': STYLE_CONFIG['ax_facecolor'], # Fundo dos eixos
    'axes.edgecolor': STYLE_CONFIG['spinecolor'], # Cor das bordas dos eixos
    'text.color': STYLE_CONFIG['textcolor'], # Cor do texto
    'axes.labelcolor': STYLE_CONFIG['textcolor'], # Cor dos rótulos dos eixos
    'xtick.color': STYLE_CONFIG['textcolor'], # Cor dos ticks do eixo x
    'ytick.color': STYLE_CONFIG['textcolor'], # Cor dos ticks do eixo y
    'grid.color': STYLE_CONFIG['gridcolor'] # Cor do grid
})

# Configura a página do Streamlit. 'layout="wide"' faz o conteúdo ocupar toda a largura da tela e da o nome ao aplicativo.
st.set_page_config(layout="wide", page_title="Ocean Optics Explorer")

#
# ----------------------------- CONSTANTES FIXAS -----------------------------
# Seção para definir valores constantes que são usados em todo o script.

PRESSURE_COL, TEMP_COL, COND_COL, WAVELENGTH_PREFIX, STATION_ID_COL = 'Depth', 'Temperature', 'Conductivity', 'X', 'station_id'
# PRESSURE_COL: coluna com profundidade
# TEMP_COL: coluna com temperatura
# COND_COL: coluna com condutividade
# WAVELENGTH_PREFIX: prefixo usado para identificar colunas de bandas espectrais (ex: X440, X560)
# STATION_ID_COL: coluna com ID da estação

# Palavras-chave para identificar automaticamente cada tipo de arquivo com base no nome do arquivo.
FILE_KEYWORDS = {'temp':'temp', 'cond':'cond', 'ed':'ed', 'lu':'lu', 'beta': 'beta'}

# Fator de transmissão da luz descendente usada para converter Lu(0-) em Lw
LW_TRANSMISSION_FACTOR = 0.54

# ----------------------------- FUNÇÕES DE PROCESSAMENTO E CÁLCULO -----------------------------

@st.cache_data
def process_uploaded_files(profiler_files, es_file=None, beta_file=None):
    file_map = {}
    for f in profiler_files:
        fname_lower = f.name.lower()
        for key, keyword in FILE_KEYWORDS.items():
            if keyword in fname_lower:
                file_map[key] = io.StringIO(f.getvalue().decode("utf-8"))
    if beta_file: file_map['beta'] = io.StringIO(beta_file.getvalue().decode("utf-8"))
    if len(file_map) < 4: st.error(f"Upload failed. Expected at least 4 in-water files (temp, cond, ed, lu), found {len(file_map)}."); return None, None
    possible_NA_values = ['#########', 'NA', 'N/A', 'Not a Number', 'missing', '-9999']
    try:
        df_temp_raw = pd.read_csv(file_map['temp'], na_values=possible_NA_values); df_cond_raw = pd.read_csv(file_map['cond'], na_values=possible_NA_values); df_ed_raw = pd.read_csv(file_map['ed'], na_values=possible_NA_values); df_lu_raw = pd.read_csv(file_map['lu'], na_values=possible_NA_values)
    except Exception as e: st.error(f"Error reading core CSV files. Please check file format. Details: {e}"); return None, None
    df_beta_raw = None
    if 'beta' in file_map: df_beta_raw = pd.read_csv(file_map['beta'], na_values=possible_NA_values)
    es_median_map = {}; es_full_data_map = {}
    if es_file:
        df_es_raw = pd.read_csv(es_file, na_values=possible_NA_values); wave_cols_es = [col for col in df_es_raw.columns if col.startswith(WAVELENGTH_PREFIX)]
        for station_id, group in df_es_raw.groupby(STATION_ID_COL): es_median_map[station_id] = group[wave_cols_es].median().values; es_full_data_map[station_id] = group[wave_cols_es]
    station_ids_from_temp = df_temp_raw[STATION_ID_COL].unique(); station_data_package = {}; successful_stations = []
    for station in station_ids_from_temp:
        df_temp = df_temp_raw[df_temp_raw[STATION_ID_COL] == station]; df_cond = df_cond_raw[df_cond_raw[STATION_ID_COL] == station]; df_ed = df_ed_raw[df_ed_raw[STATION_ID_COL] == station]; df_lu = df_lu_raw[df_lu_raw[STATION_ID_COL] == station]
        if any(df.empty for df in [df_temp, df_cond, df_ed, df_lu]): st.warning(f"Skipping station '{station}' due to missing core files."); continue
        wave_cols = [col for col in df_ed.columns if col.startswith(WAVELENGTH_PREFIX)]; wavelengths_temp = np.array([float(c.replace(WAVELENGTH_PREFIX, '')) for c in wave_cols]); idx_490 = (np.abs(wavelengths_temp - 490)).argmin(); canary_col = wave_cols[idx_490]
        df_ed_clean = df_ed.dropna(subset=[canary_col]); df_ed_clean = df_ed_clean[df_ed_clean[canary_col] > 0]
        if df_ed_clean.empty: st.warning(f"Skipping station '{station}' due to no valid Ed(490) data."); continue
        optical_min_depth = df_ed_clean[PRESSURE_COL].min(); optical_max_depth = df_ed_clean[PRESSURE_COL].max(); all_temp_depths = np.sort(df_temp[PRESSURE_COL].unique()); master_pressure = all_temp_depths[(all_temp_depths >= optical_min_depth) & (all_temp_depths <= optical_max_depth)]
        if master_pressure.size < 2: continue
        df_temp_clean = df_temp.dropna(subset=[TEMP_COL]); temp_data = np.interp(master_pressure, df_temp_clean[PRESSURE_COL], df_temp_clean[TEMP_COL])
        df_cond_clean = df_cond.dropna(subset=[COND_COL]); cond_data = np.interp(master_pressure, df_cond_clean[PRESSURE_COL], df_cond_clean[COND_COL])
        wavelengths = np.array([float(c.replace(WAVELENGTH_PREFIX, '')) for c in wave_cols])
        df_ed_grouped = df_ed[[PRESSURE_COL] + wave_cols].groupby(PRESSURE_COL).mean(); ed_df_interp = df_ed_grouped.reindex(df_ed_grouped.index.union(master_pressure)).interpolate(method='index').loc[master_pressure]; ed_data = ed_df_interp.values
        df_lu_grouped = df_lu[[PRESSURE_COL] + wave_cols].groupby(PRESSURE_COL).mean(); lu_df_interp = df_lu_grouped.reindex(df_lu_grouped.index.union(master_pressure)).interpolate(method='index').loc[master_pressure]; lu_data = lu_df_interp.values
        beta_data_interp = None; beta_cols = []; df_beta = None
        if df_beta_raw is not None: df_beta = df_beta_raw[df_beta_raw[STATION_ID_COL] == station]
        if df_beta is not None and not df_beta.empty:
            beta_cols = [col for col in df_beta.columns if col not in [PRESSURE_COL, STATION_ID_COL]]; df_beta_clean = df_beta.dropna(subset=[PRESSURE_COL] + beta_cols, how='any')
            if not df_beta_clean.empty:
                df_beta_indexed = df_beta_clean.set_index(PRESSURE_COL); combined_index = df_beta_indexed.index.union(master_pressure); df_beta_aligned = df_beta_indexed.reindex(combined_index).interpolate(method='index'); beta_data_interp = df_beta_aligned.loc[master_pressure].dropna(how='all')
                if beta_data_interp.empty: st.warning(f"For station '{station}', beta data was found but had no overlapping depth range with the main profile.")
            else: st.warning(f"For station '{station}', the uploaded beta file contained no valid numeric data rows.")
        station_data_package[station] = {'pressure': master_pressure, 'temperature': temp_data, 'conductivity': cond_data, 'Ed_data': ed_data, 'Lu_data': lu_data, 'wavelengths': wavelengths, 'Es': es_median_map.get(station, None), 'Es_all_spectra': es_full_data_map.get(station, None), 'beta_data': beta_data_interp, 'beta_cols': beta_cols}
        successful_stations.append(station)
    return station_data_package, successful_stations

def analyze_mixture(SA, CT, vertices, names):
    p1, p2, p3 = np.array(vertices[0]), np.array(vertices[1]), np.array(vertices[2])
    mixing_percentages = []
    def get_barycentric_coords(p, a, b, c):
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-9: return -1, -1, -1
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w
    for sal, temp in zip(SA, CT):
        point = np.array([sal, temp])
        w3, w1, w2 = get_barycentric_coords(point, p3, p1, p2)
        if w1 >= -1e-9 and w2 >= -1e-9 and w3 >= -1e-9:
            mixing_percentages.append((w1, w2, w3))
        else:
            mixing_percentages.append((np.nan, np.nan, np.nan))
    df_mix = pd.DataFrame(mixing_percentages, columns=names)
    avg_mix = df_mix.mean()
    avg_text = f"**Avg. Mixture ({len(df_mix.dropna())} points):**\n- {avg_mix[names[0]]*100:.1f}% {names[0]}\n- {avg_mix[names[1]]*100:.1f}% {names[1]}\n- {avg_mix[names[2]]*100:.1f}% {names[2]}"
    return avg_text, df_mix

def perform_convolution(prof_wl, lw_hyperspectral, ed_hyperspectral, srf_wl, srf_response):
    """
    Performs spectral convolution on Lw and Ed independently before division.
    
    Args:
        prof_wl (np.array): Wavelengths from the profiler.
        lw_hyperspectral (np.array): Hyperspectral water-leaving radiance (Lw).
        ed_hyperspectral (np.array): Hyperspectral downwelling irradiance (Ed or Es).
        srf_wl (np.array): Wavelengths from the SRF file.
        srf_response (np.array): The spectral response for a single satellite band.
        
    Returns:
        float: The convolved Rrs for the band.
    """
    # Step 1: Interpolate the satellite's SRF onto the profiler's wavelength grid.
    srf_interp = np.interp(prof_wl, srf_wl, srf_response, left=0, right=0)
    
    # Step 2: Calculate the integrated numerator and denominator from the formula.
    # The integral (∫) becomes a summation (np.sum) for discrete data.
    lw_integrated = np.sum(lw_hyperspectral * srf_interp)
    ed_integrated = np.sum(ed_hyperspectral * srf_interp)
    
    # Step 3: Calculate the final band-specific Rrs. Avoid division by zero.
    if ed_integrated > 0:
        return lw_integrated / ed_integrated
    else:
        return np.nan

def calculate_derived_products(station_data, station_id, analysis_layers): # <-- Added station_id here
    if not analysis_layers or not station_data: return None
    pressure, ed_data, lu_data, wavelengths = station_data['pressure'], station_data['Ed_data'], station_data['Lu_data'], station_data['wavelengths']
    es_for_rrs = st.session_state.get('new_es_median', station_data.get('Es'))
    layer_k_data, results_data, rrs_dict, k_dict = {}, [], {'wavelength_nm': wavelengths}, {'wavelength_nm': wavelengths}
    df_kw = get_smith_baker_kw(); kw_interp = np.interp(wavelengths, df_kw['wavelength'], df_kw['Kw'])
    initial_rrs = calculate_Rrs(ed_data, lu_data, pressure, es_data=es_for_rrs)
    rrs_dict['initial_rrs_sr-1'] = initial_rrs
    for layer_num, data in analysis_layers.items():
        z_min, z_max = data['range']
        kd_raw, klu_raw = calculate_k_spectra(pressure, ed_data, lu_data, wavelengths, z_min, z_max)
        if klu_raw is not None and kd_raw is not None:
            kd_corrected, klu_corrected = np.maximum(kd_raw, kw_interp), np.maximum(klu_raw, kw_interp)
            layer_k_data[layer_num] = (kd_corrected, klu_corrected)
            prop_rrs = calculate_Rrs(ed_data, lu_data, pressure, es_data=es_for_rrs, kd=kd_corrected, klu=klu_corrected, z_top=z_min)
            rrs_dict[f'propagated_rrs_L{layer_num}_sr-1'] = prop_rrs
            k_dict[f'profiler_kd_L{layer_num}_m-1'] = kd_corrected
            k_dict[f'profiler_klu_L{layer_num}_m-1'] = klu_corrected
            idx_490 = (np.abs(wavelengths - 490)).argmin()
            results_data.append({'Layer':f"Layer {layer_num}", 'Depth Range (m)':f"{z_min}-{z_max}", 'Kd(490)':kd_corrected[idx_490], 'klu(490)':klu_corrected[idx_490]})
        else:
            results_data.append({'Layer':f"Layer {layer_num}", 'Depth Range (m)':f"{z_min}-{z_max}", 'Kd(490)':np.nan, 'klu(490)':np.nan})
    results_df = pd.DataFrame(results_data)
    h, c, N_A = 6.62607015e-34, 2.99792458e8, 6.02214076e23; par_wl_range = np.arange(400, 701, 1)
    valid_indices = np.where((wavelengths >= 400) & (wavelengths <= 700))[0]
    sub_wl, sub_ed = wavelengths[valid_indices], ed_data[:, valid_indices]
    interp_ed = np.array([np.interp(par_wl_range, sub_wl, row) for row in sub_ed])
    conversion_factor = (par_wl_range * 1e-9) / (h * c) * (1e-6 * 1e4) / N_A * 1e6
    par_profile_profiler = np.sum(interp_ed * conversion_factor, axis=1)
    kd_par_results = calculate_kd_par(pressure, par_profile_profiler, analysis_layers)
    kd_par_df = pd.DataFrame([{'Layer': f"Layer {ln}", 'Kd(PAR)': val} for ln, val in kd_par_results.items()])
    
    # --- UPDATED: Collect data for the empirical model with context ---
    if not results_df.empty and not kd_par_df.empty:
        model_data_source = pd.merge(results_df, kd_par_df, on="Layer")
        
        # --- THIS IS THE FIX: Explicitly define the column order ---
        new_model_data = pd.DataFrame({
            'Station_ID': station_id,
            'Layer': model_data_source['Layer'],
            'Kd(PAR)': model_data_source['Kd(PAR)'],
            'Kd(490)': model_data_source['Kd(490)']
        }).dropna()
        # --- END OF FIX ---

        if not new_model_data.empty:
            st.session_state.empirical_model_data = pd.concat([st.session_state.empirical_model_data, new_model_data], ignore_index=True)
            st.session_state.empirical_model_data.drop_duplicates(subset=['Station_ID', 'Layer'], keep='last', inplace=True)

    return { "layer_k_data": layer_k_data, "results_df": results_df, "kd_par_df": kd_par_df, "rrs_df_export": pd.DataFrame(rrs_dict), "k_df_export": pd.DataFrame(k_dict) }

def _handle_secchi_upload():
    """Callback function to process the master secchi file upload."""
    if st.session_state.master_secchi_loader is None:
        return
    try:
        df_loaded = pd.read_csv(st.session_state.master_secchi_loader)
        expected_cols = ['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']
        if all(col in df_loaded.columns for col in expected_cols):

            # ✅ --- THIS IS THE CORRECTED MERGE LOGIC (Same as above) ---
            # 1. Get a clean copy of the data from the CURRENT session.
            df_session = st.session_state.master_kd_secchi_df.copy()

            # 2. Combine the session data FIRST, then the loaded data.
            combined_df = pd.concat([df_session, df_loaded], ignore_index=True)

            # 3. De-duplicate, keeping the FIRST instance of any Station.
            # This prioritizes any data you've added or updated in the current session.
            combined_df.drop_duplicates(subset=['Station_ID'], keep='first', inplace=True)
            # --- END OF FIX ---
            
            st.session_state.master_kd_secchi_df = combined_df
            st.session_state.notification = {'type': 'success', 'message': f"Successfully loaded and merged {len(df_loaded)} master Secchi data points."}
        else:
            st.session_state.notification = {'type': 'error', 'message': "Uploaded CSV is missing required columns."}
    except Exception as e:
        st.session_state.notification = {'type': 'error', 'message': f"Failed to read file: {e}"}

def _handle_model_upload():
    """Callback function to process the master model file upload."""
    if st.session_state.master_model_loader is None:
        return
    try:
        df_loaded = pd.read_csv(st.session_state.master_model_loader)
        expected_cols = ['Station_ID', 'Layer', 'Kd(PAR)', 'Kd(490)']
        if all(col in df_loaded.columns for col in expected_cols):

            # ✅ --- THIS IS THE CORRECTED MERGE LOGIC ---
            # 1. Get a clean copy of the data from the CURRENT session.
            df_session = st.session_state.empirical_model_data.copy()

            # 2. Combine the session data FIRST, then the loaded data.
            combined_df = pd.concat([df_session, df_loaded], ignore_index=True)

            # 3. De-duplicate, keeping the FIRST instance of any Station/Layer combo.
            # Since the session data came first, it is ALWAYS prioritized.
            combined_df.drop_duplicates(subset=['Station_ID'], keep='first', inplace=True)
            # --- END OF FIX ---

            st.session_state.empirical_model_data = combined_df
            st.session_state.notification = {'type': 'success', 'message': f"Successfully loaded and merged {len(df_loaded)} model data points."}
        else:
            st.session_state.notification = {'type': 'error', 'message': "Uploaded CSV is missing required columns."}
    except Exception as e:
        st.session_state.notification = {'type': 'error', 'message': f"Failed to read file: {e}"}
        
def generate_summary_report(station_ids, analysis_layers, results_df, kd_par_df, mixture_results=None, empirical_model=None, external_probe_filename=None, comparison_kd_df=None):
    report = io.StringIO(); report.write("=======================================\n"); report.write("      ANALYSIS SUMMARY REPORT\n"); report.write("=======================================\n\n")
    report.write("--- INSTRUMENTATION ---\n"); report.write("In-Water Profiler: Data processed from uploaded Ed, Lu, Temp, Cond files.\n")
    if external_probe_filename: report.write(f"External Probe Data Source: {external_probe_filename}\n")
    report.write("\n")
    if isinstance(station_ids, list): report.write(f"Profiles Included in Analysis: {', '.join(map(str, station_ids))}\n")
    else: report.write(f"Profile Included in Analysis: {station_ids}\n")
    report.write("\n--- ANALYSIS LAYERS & RESULTS ---\n")
    if not analysis_layers: report.write("No layers were analyzed for the selected profile(s).\n")
    elif results_df is not None and kd_par_df is not None:
        try:
            summary_df = pd.merge(results_df, kd_par_df, on="Layer")
            if comparison_kd_df is not None:
                summary_df = summary_df.merge(comparison_kd_df[['Kd(PAR) (External)', 'Estimated Kd(490) (External)']], on="Layer", how="left")
            for index, row in summary_df.iterrows():
                report.write(f"\n{row['Layer']}:\n"); report.write(f"  - Depth Range: {row['Depth Range (m)']}\n"); report.write(f"  - Kd(490) (Profiler): {row['Kd(490)']:.4f} m⁻¹\n"); report.write(f"  - klu(490) (Profiler): {row['klu(490)']:.4f} m⁻¹\n"); report.write(f"  - Kd(PAR) (Profiler): {row['Kd(PAR)']:.4f} m⁻¹\n")
                if 'Kd(PAR) (External)' in row and pd.notna(row['Kd(PAR) (External)']): report.write(f"  - Kd(PAR) (External Probe): {row['Kd(PAR) (External)']:.4f} m⁻¹\n")
                if 'Estimated Kd(490) (External)' in row and pd.notna(row['Estimated Kd(490) (External)']): report.write(f"  - Estimated Kd(490) (External Probe): {row['Estimated Kd(490) (External)']:.4f} m⁻¹\n")
        except Exception as e: report.write(f"Could not generate detailed layer results. Error: {e}\n")
    else: report.write("Layer results were not calculated.\n")
    if empirical_model:
        report.write("\n--- SITE-SPECIFIC EMPIRICAL MODEL ---\n"); report.write("The following model was generated from profiler data to estimate Kd(490) from Kd(PAR):\n"); report.write(f"  - Equation: Kd(490) = {empirical_model['slope']:.4f} * Kd(PAR) + {empirical_model['intercept']:.4f}\n"); report.write(f"  - R²: {empirical_model['r2']:.4f}\n")
    if mixture_results:
        report.write("\n--- WATER MASS MIXTURE ANALYSIS ---\n")
        if isinstance(mixture_results, dict):
            report.write("Average mixture percentages for each profile in the ensemble:\n")
            for station_id, result_text in mixture_results.items(): cleaned_text = result_text.replace("**", "").replace("\n-", "\n  -"); report.write(f"\nProfile: {station_id}\n{cleaned_text}\n")
        elif isinstance(mixture_results, str): cleaned_text = mixture_results.replace("**", "").replace("\n-", "\n  -"); report.write(f"{cleaned_text}\n")
    report.write("\n=======================================\n"); report.write("      FORMULAS & METHODS\n"); report.write("=======================================\n\n"); report.write("Apparent Optical Properties (Kd & klu):\n"); report.write("Derived from the slope of the log-linear regression of radiance/irradiance vs. depth.\n\n"); report.write("Remote Sensing Reflectance (Rrs):\n"); report.write(f"Calculated as Rrs = Lw / Ed(0+), where Lw is propagated from Lu(0-) using a factor of {LW_TRANSMISSION_FACTOR}.\n\n"); report.write("Photosynthetically Available Radiation (PAR & Kd(PAR)):\n"); report.write("PAR is the integrated quantum irradiance from 400-700nm. Kd(PAR) is derived from the log-linear regression of the PAR profile.\n\n")
    return report.getvalue()

def calculate_physical_properties(_station_data, lon, lat):
    pressure, temp, cond = _station_data['pressure'], _station_data['temperature'], _station_data['conductivity']; salinity=gsw.SP_from_C(cond, temp, pressure); SA=gsw.SA_from_SP(salinity, pressure, lon, lat); CT=gsw.CT_from_t(SA, temp, pressure); n2, p_mid=gsw.Nsquared(SA, CT, pressure); n2_padded=np.append(n2, [np.nan]*(len(pressure)-len(n2))); p_mid_padded=np.append(p_mid, [np.nan]*(len(pressure)-len(p_mid)))
    return pd.DataFrame({'Depth':pressure, 'Salinity':SA, 'Temperature':CT, 'n2':n2_padded, 'p_mid':p_mid_padded})

def calculate_k_spectra(pressure, ed_data, lu_data, wavelengths, z_min, z_max):
    indices=np.where((pressure>=z_min)&(pressure<=z_max));
    if len(indices[0]) < 2: return None, None
    depths, ed_range, lu_range=pressure[indices], ed_data[indices], lu_data[indices]; kd, klu=[], []
    for i in range(wavelengths.shape[0]):
        ed_slice, lu_slice=ed_range[:, i], lu_range[:, i]; valid_ed, valid_lu=ed_slice > 0, lu_slice > 0
        kd.append(-np.polyfit(depths[valid_ed], np.log(ed_slice[valid_ed]), 1)[0] if np.sum(valid_ed) >= 2 else np.nan)
        klu.append(-np.polyfit(depths[valid_lu], np.log(lu_slice[valid_lu]), 1)[0] if np.sum(valid_lu) >= 2 else np.nan)
    return np.array(kd), np.array(klu)

def calculate_Rrs(ed_data, lu_data, pressure, es_data=None, kd=None, klu=None, z_top=None):
    """
    Calculates Remote Sensing Reflectance (Rrs) with two key corrections:
    1. Smart propagation: If z_top is shallower than the first valid data, it propagates from the first valid point.
    2. Strict Denominator Hierarchy: ALWAYS prioritizes measured `es_data` if it is provided.
    """
    # --- STEP 1: Calculate Water-Leaving Radiance, Lw ---
    
    # Check if we have the necessary parameters for propagation mode
    use_propagation = all(param is not None for param in [kd, klu, z_top])

    if use_propagation:
        # PROPAGATION MODE:
        # Propagate Lu from the top of the layer (z_top) to just below the surface (Lu(0-))
        lu_sub_list = []
        for i in range(lu_data.shape[1]): # Iterate per wavelength
            col_data = lu_data[:, i]
            valid_mask = (col_data > 0) & (~np.isnan(col_data))
            
            if not np.any(valid_mask):
                lu_sub_list.append(np.nan)
                continue
            
            first_valid_idx = np.argmax(valid_mask)
            z_first_valid = pressure[first_valid_idx]
            lu_first_valid = col_data[first_valid_idx]

            # Use the shallowest valid data point if user selection is outside the data range
            if z_top < z_first_valid:
                z_calc, lu_calc = z_first_valid, lu_first_valid
            else:
                z_calc, lu_calc = z_top, np.interp(z_top, pressure, col_data)
            
            # Apply propagation formula: Lu(0-) = Lu(z) * exp(Klu * z)
            val = lu_calc * np.exp(klu[i] * z_calc)
            lu_sub_list.append(val)
        
        lu_sub = np.array(lu_sub_list)

    else:
        # INITIAL MODE (no K values available):
        # Use the shallowest valid measurement of Lu as the Lu(0-)
        valid_lu_indices = np.where(np.nanmedian(lu_data, axis=1) > 0)[0]
        if len(valid_lu_indices) > 0:
            lu_sub = lu_data[valid_lu_indices[0], :]
        else:
            lu_sub = np.full(lu_data.shape[1], np.nan)

    # Convert subsurface radiance Lu(0-) to water-leaving radiance Lw(0+)
    lw_spectrum = lu_sub * LW_TRANSMISSION_FACTOR

    # --- STEP 2: Determine the Correct Downwelling Irradiance, Ed(0+) ---
    
    # STRICT HIERARCHY:
    if es_data is not None:
        # PRIORITY #1: If a measured Es spectrum is available, ALWAYS use it.
        denominator = es_data
    else:
        # FALLBACK: If NO Es is available, then (and only then) use the profiler's near-surface Ed.
        valid_ed_indices = np.where(np.nanmedian(ed_data, axis=1) > 0)[0]
        if len(valid_ed_indices) > 0:
            ed_surf = ed_data[valid_ed_indices[0], :]
        else:
            ed_surf = np.full(ed_data.shape[1], np.nan)
        denominator = ed_surf

    # --- STEP 3: Final Rrs Calculation ---
    with np.errstate(divide='ignore', invalid='ignore'):
        rrs_spectrum = np.where(denominator > 0, lw_spectrum / denominator, np.nan)

    return rrs_spectrum

def calculate_kd_par(pressure, par_profile, analysis_layers):
    kd_par_results = {};
    for layer_num, data in analysis_layers.items():
        z_min, z_max = data['range']; layer_indices = np.where((pressure >= z_min) & (pressure <= z_max))[0]
        if len(layer_indices) < 2: kd_par_results[layer_num] = np.nan; continue
        depths_in_layer = pressure[layer_indices]; par_in_layer = par_profile[layer_indices]; valid_par_indices = np.where(par_in_layer > 0)[0]
        if len(valid_par_indices) < 2: kd_par_results[layer_num] = np.nan; continue
        depths_for_fit = depths_in_layer[valid_par_indices]; par_for_fit = par_in_layer[valid_par_indices]; slope = np.polyfit(depths_for_fit, np.log(par_for_fit), 1)[0]; kd_par_results[layer_num] = -slope
    return kd_par_results

def get_smith_baker_kw():
    kw_data = {'wavelength': [300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745], 'Kw': [0.154, 0.135, 0.116, 0.105, 0.0944, 0.0855, 0.0765, 0.0701, 0.0637, 0.0584, 0.053, 0.0485, 0.0439, 0.0396, 0.0353, 0.031, 0.0267, 0.025, 0.0233, 0.0221, 0.0209, 0.0203, 0.0196, 0.019, 0.0184, 0.0178, 0.0172, 0.0171, 0.017, 0.0169, 0.0168, 0.0172, 0.0176, 0.0176, 0.0175, 0.0185, 0.0194, 0.02303, 0.0212, 0.0242, 0.0271, 0.0321, 0.037, 0.043, 0.0489, 0.0504, 0.0519, 0.0544, 0.0568, 0.0608, 0.0648, 0.0683, 0.0717, 0.0762, 0.0807, 0.0949, 0.109, 0.1335, 0.158, 0.202, 0.245, 0.267, 0.29, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.34, 0.35, 0.375, 0.4, 0.415, 0.43, 0.44, 0.45, 0.475, 0.5, 0.575, 0.65, 0.742, 0.834, 1.002, 1.17, 1.485, 1.8, 2.09, 2.38, 2.425]}
    return pd.DataFrame(kw_data)

def get_comparison_stats(df_merged, col_name):
    y_true = df_merged[f'{col_name}_internal']; y_pred = df_merged[f'{col_name}_external']; r2 = r2_score(y_true, y_pred); rmse = np.sqrt(mean_squared_error(y_true, y_pred)); mae = mean_absolute_error(y_true, y_pred); bias = np.mean(y_pred - y_true)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Bias': bias}

def highlight_best(s):
    is_max = s == s.max(); is_min = s == s.min(); is_min_abs = abs(s) == abs(s).min()
    if 'R2' in s.name: return ['background-color: #006400' if v else '' for v in is_max]
    elif 'Bias' in s.name: return ['background-color: #006400' if v else '' for v in is_min_abs]
    else: return ['background-color: #006400' if v else '' for v in is_min]

def fig_to_buffer(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches='tight'); buf.seek(0); return buf

def compare_ctd_profiles(df_internal, df_external):
    min_depth = max(df_internal['Depth'].min(), df_external['Depth'].min()); max_depth_val = min(df_internal['Depth'].max(), df_external['Depth'].max())
    if min_depth >= max_depth_val: return None
    common_grid = np.arange(np.ceil(min_depth * 2) / 2, np.floor(max_depth_val * 2) / 2 + 0.5, 0.5); df_merged = pd.DataFrame({'Depth': common_grid})
    df_merged['Temperature_internal'] = np.interp(common_grid, df_internal['Depth'], df_internal['Temperature']); df_merged['Salinity_internal'] = np.interp(common_grid, df_internal['Depth'], df_internal['Salinity']); df_merged['Temperature_external'] = np.interp(common_grid, df_external['Depth'], df_external['Temperature']); df_merged['Salinity_external'] = np.interp(common_grid, df_external['Depth'], df_external['Salinity'])
    return df_merged.dropna()

def create_full_report_zip(zip_file, base_name):
    figs_to_save_from_state = {"1_Stratification_Backscatter": st.session_state.get('fig_n2'), "2_Rrs_Spectra": st.session_state.get('fig_rrs'), "3_Log_Radiance_Profiles": st.session_state.get('fig_log'), "4_Attenuation_Spectra": st.session_state.get('fig_k'), "5_TS_Diagram": st.session_state.get('fig_ts'), "6_TS_Mixture_Depth_Profile": st.session_state.get('fig_mix_depth'), "7_CTD_Comparison": st.session_state.get('fig_comp'), "8_LuEd_Spectra_Comparison": st.session_state.get('fig_spec'),}
    for fig_name, fig_obj in figs_to_save_from_state.items():
        if fig_obj: zip_file.writestr(f"{base_name}/Plots/{fig_name}.png", fig_to_buffer(fig_obj).getvalue())
    if st.session_state.get('external_figs'):
        for param_name, fig_obj in st.session_state.external_figs.items(): zip_file.writestr(f"{base_name}/Plots/9_{param_name}_vs_LuEd.png", fig_to_buffer(fig_obj).getvalue())
    if st.session_state.get('results_df') is not None: zip_file.writestr(f"{base_name}/Metrics/K_Metrics_Summary.csv", st.session_state.results_df.to_csv(index=False))
    if st.session_state.get('kd_par_df') is not None: zip_file.writestr(f"{base_name}/Metrics/Kd_PAR_Summary.csv", st.session_state.kd_par_df.to_csv(index=False))
    if st.session_state.get('comparison_table') is not None: zip_file.writestr(f"{base_name}/Metrics/CTD_Comparison_Metrics.csv", st.session_state.comparison_table.to_csv())
    if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
        sensor_name = st.session_state.get('convolved_sensor_name', 'convolved')
        df_to_save = st.session_state.convolved_rrs_df
        zip_file.writestr(f"{base_name}/L3_Data/Rrs_{sensor_name}.csv", df_to_save.to_csv(index=False))
    if 'mixture_df_profiler' in st.session_state and st.session_state.mixture_df_profiler is not None:
        df_to_save = st.session_state.mixture_df_profiler
        # Adiciona a coluna de profundidade para contexto completo
        df_with_depth = pd.concat([phys_props_df['Depth'], df_to_save], axis=1)
        zip_file.writestr(f"{base_name}/L3_Data/Mixture_Profile_Profiler.csv", df_with_depth.to_csv(index=False))

    if 'mixture_df_external' in st.session_state and st.session_state.mixture_df_external is not None:
        df_to_save = st.session_state.mixture_df_external
        zip_file.writestr(f"{base_name}/L3_Data/Mixture_Profile_External.csv", df_to_save.to_csv(index=False))
        
@st.cache_data
def load_srf_from_folder(folder_path="srf_data"):
    """
    Scans a folder for SRF csv files and loads them into a dictionary.
    This function is cached to run only once.
    """
    srf_data = {}
    try:
        import os
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                sensor_name = filename.split('.')[0]
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                srf_data[sensor_name] = df
        return srf_data
    except FileNotFoundError:
        st.error(f"SRF data folder not found at '{folder_path}'. Please create it and add your SRF csv files.")
        return None
        
def calculate_external_kd_par(df_external, depth_col, par_col, z_min, z_max):
    if df_external is None or depth_col not in df_external.columns or par_col not in df_external.columns: return np.nan
    df_external[depth_col] = pd.to_numeric(df_external[depth_col], errors='coerce'); df_external[par_col] = pd.to_numeric(df_external[par_col], errors='coerce'); df_layer = df_external.dropna(subset=[depth_col, par_col])
    df_layer = df_layer[(df_layer[depth_col] >= z_min) & (df_layer[depth_col] <= z_max)]; df_layer = df_layer[df_layer[par_col] > 0]
    if len(df_layer) < 2: return np.nan
    depths_for_fit = df_layer[depth_col]; par_for_fit = df_layer[par_col]; slope = np.polyfit(depths_for_fit, np.log(par_for_fit), 1)[0]
    return -slope

def estimate_chlorophyll_from_kdpar(kd_par):
    if kd_par <= 0.022: return 0.0
    kd_bio = kd_par - 0.022; a, b = 0.0512, 0.428; chlorophyll = (kd_bio / a)**(1 / b)
    return chlorophyll

def estimate_kd490_from_chlorophyll(chlorophyll):
    kw_490 = 0.0166; x = 0.07917; e = 0.7895; kd_490 = kw_490 + x * (chlorophyll**e)
    return kd_490

def get_regression_metrics(x_data, y_data):
    df = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
    if len(df) < 2: return {'R²': np.nan, 'RMSE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
    slope, intercept, r_value, p_value, _ = linregress(df['x'], df['y']); y_pred = slope * df['x'] + intercept; rmse = np.sqrt(mean_squared_error(df['y'], y_pred))
    return {'R²': r_value**2, 'RMSE': rmse, 'Slope': slope, 'Intercept': intercept, 'P-value': p_value}

def _handle_master_rrs_upload():
    """Callback function to process the master Rrs file upload."""
    if st.session_state.master_rrs_loader is None:
        return
    try:
        # Read the uploaded CSV into a pandas DataFrame
        df_loaded = pd.read_csv(st.session_state.master_rrs_loader)
        
        # Store the DataFrame in the session state
        st.session_state.master_rrs_df = df_loaded
        st.toast(f"Successfully loaded {len(df_loaded.columns) - 1} historical Rrs spectra!", icon="✅")
    except Exception as e:
        st.toast(f"Failed to read master Rrs file: {e}", icon="❌")

def plot_kd_secchi_relationship(df):
    if df.empty or len(df.dropna(subset=['Secchi'])) < 2: st.info("Add at least two data points with valid Secchi depths to the master sheet to generate plots and metrics."); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title("Secchi Depth vs. Kd(PAR)", weight='bold'); ax1.set_xlabel("Secchi Depth (m)", weight='bold'); ax1.set_ylabel("Kd(PAR) (m⁻¹)", weight='bold')
    df_par_prof = df[['Secchi', 'Kd(PAR)_Profiler']].dropna()
    if len(df_par_prof) >= 2:
        ax1.scatter(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'], ec='white', s=60, marker='o', label='Profiler', zorder=10); slope, intercept, r_value, _, _ = linregress(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler']); x_vals = np.array(df_par_prof['Secchi']); ax1.plot(x_vals, intercept + slope * x_vals, '--', color='cyan', label=f'Profiler R² = {r_value**2:.3f}')
    df_par_ext = df[['Secchi', 'Kd(PAR)_External']].dropna()
    if len(df_par_ext) >= 2:
        ax1.scatter(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'], ec='white', s=60, marker='^', label='External Probe', zorder=10); slope, intercept, r_value, _, _ = linregress(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External']); x_vals = np.array(df_par_ext['Secchi']); ax1.plot(x_vals, intercept + slope * x_vals, ':', color='yellow', label=f'External R² = {r_value**2:.3f}')
    ax1.grid(True, linestyle=':'); ax1.legend()
    df_490 = df[['Secchi', 'Kd(490)_Profiler']].dropna()
    if len(df_490) >= 2:
        ax2.scatter(df_490['Secchi'], df_490['Kd(490)_Profiler'], ec='white', s=50, zorder=10); slope, intercept, r_value, _, _ = linregress(df_490['Secchi'], df_490['Kd(490)_Profiler']); x_vals = np.array(ax2.get_xlim()); y_vals = intercept + slope * x_vals; ax2.plot(x_vals, y_vals, '--', color='lime', label=f'R² = {r_value**2:.3f}')
    ax2.set_xlabel("Secchi Depth (m)", weight='bold'); ax2.set_ylabel("Kd(490) (m⁻¹)", weight='bold'); ax2.set_title("Secchi Depth vs. Kd(490) (Profiler)", weight='bold'); ax2.grid(True, linestyle=':'); ax2.legend()
    plt.tight_layout(); st.pyplot(fig)
    st.markdown("**Regression Metrics**"); metrics_prof_par = get_regression_metrics(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler']); metrics_ext_par = get_regression_metrics(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External']); metrics_prof_490 = get_regression_metrics(df_490['Secchi'], df_490['Kd(490)_Profiler'])
    df_metrics = pd.DataFrame([metrics_prof_par, metrics_ext_par, metrics_prof_490], index=['Secchi vs. Kd(PAR) (Profiler)', 'Secchi vs. Kd(PAR) (External)', 'Secchi vs. Kd(490) (Profiler)']).T
    st.dataframe(df_metrics.style.format('{:.4f}'), use_container_width=True)

# === 4. SESSION STATE INITIALIZATION ===
keys_to_initialize = {
    'station_data': None, 'station_list': [], 'selected_station': None,
    'profile_specific_layers': {}, 'lon':-45.692597, 'lat':-24.101879,
    'wm_vertices': None, 'mixture_results': None, 'mixture_df': None,
    'wm_names':["AT","ACAS","AC"], 'profiler_sel_index':0, 'comparison_table': None,
    'results_df': None, 'kd_par_df': None, 'discarded_indices': [], 'new_es_median': None,
    'fig_n2': None, 'fig_rrs': None, 'fig_log': None, 'fig_k': None, 'fig_ts': None,
    'fig_comp': None, 'fig_spec': None, 'external_figs': {}, 'fig_mix_depth': None,
    'analysis_layers': {}, 'external_probe_df': None,
    'master_kd_secchi_df': pd.DataFrame(columns=[
        'Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External'
    ]),
    'empirical_model_data': pd.DataFrame(columns=['Station_ID', 'Layer', 'Kd(PAR)', 'Kd(490)']),
    'empirical_model_params': None,
    'external_probe_filename': None,
    'comparison_kd_df': None,
    'active_tab': 0,
    'notification': None
}

for key, value in keys_to_initialize.items():
    if key not in st.session_state:
        st.session_state[key] = value
    
# === 5. UI: SIDEBAR ===
with st.sidebar:
    st.title("Ocean Optics Explorer"); st.header("1. Load Data"); uploaded_files = st.file_uploader("Upload In-Water Profiler Files (Temp, Cond, Ed, Lu)", accept_multiple_files=True, type="csv"); uploaded_beta_file = st.file_uploader("Upload Backscattering 'Beta' File (Optional)", type="csv", key="beta_uploader"); uploaded_es_file = st.file_uploader("Upload Surface Es CSV File (Optional)", type="csv", key="es_uploader")
    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing data..."):
            st.session_state.station_data, st.session_state.station_list = process_uploaded_files(uploaded_files, uploaded_es_file, uploaded_beta_file)
            keys_to_reset = ['profile_specific_layers', 'wm_vertices', 'mixture_results', 'mixture_df', 'comparison_table', 'profiler_sel_index', 'results_df', 'kd_par_df', 'fig_n2', 'fig_rrs', 'fig_log', 'fig_k', 'fig_ts', 'fig_comp', 'fig_spec', 'external_figs', 'fig_mix_depth', 'discarded_indices', 'new_es_median', 'external_probe_df', 'master_kd_secchi_df', 'empirical_model_data', 'empirical_model_params', 'external_probe_filename', 'comparison_kd_df']
            for key in keys_to_reset:
                if key == 'master_kd_secchi_df': st.session_state[key] = pd.DataFrame(columns=['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External'])
                elif key == 'empirical_model_data': st.session_state[key] = pd.DataFrame(columns=['Kd(PAR)', 'Kd(490)'])
                elif key in ['profile_specific_layers', 'external_figs']: st.session_state[key] = {}
                elif key == 'discarded_indices': st.session_state[key] = []
                else: st.session_state[key] = None
            if st.session_state.station_data: st.session_state.selected_station=st.session_state.station_list[0]
    if st.session_state.station_data:
        if st.session_state.station_data[st.session_state.selected_station].get('beta_data') is not None:
            st.header("Beta Plot Options"); beta_cols_available = st.session_state.station_data[st.session_state.selected_station]['beta_cols']
            if beta_cols_available: st.session_state.selected_beta_col = st.selectbox("Select Beta Wavelength to Plot:", options=beta_cols_available, index=0)
        st.header("2. Select Profile"); st.session_state.selected_station=st.selectbox("Active Station:", options=st.session_state.station_list); st.session_state.lat=st.number_input("Latitude", value=st.session_state.lat); st.session_state.lon=st.number_input("Longitude", value=st.session_state.lon)
        st.header(f"3. Analyze Layers for '{st.session_state.selected_station}'"); current_data=st.session_state.station_data[st.session_state.selected_station]; max_depth=int(current_data['pressure'].max()) if current_data['pressure'].size > 0 else 100; selected_depth=st.slider("Select depth range (m):", 0.0, float(max_depth), (10.0, 20.0), step=0.5, key=f"slider_{st.session_state.selected_station}"); layer_num_to_add=st.selectbox("Choose layer to define:",[1,2,3], key=f"layer_select_{st.session_state.selected_station}")
        if st.button(f"Analyze Layer {layer_num_to_add}"):
            station_id = st.session_state.selected_station; z_min, z_max = selected_depth
            if station_id not in st.session_state.profile_specific_layers: st.session_state.profile_specific_layers[station_id] = {}
            st.session_state.profile_specific_layers[station_id][layer_num_to_add] = {'range': (z_min, z_max)}; st.success(f"Layer {layer_num_to_add} ({z_min}m-{z_max}m) saved for '{station_id}'.")
        station_id = st.session_state.selected_station; defined_layers = st.session_state.profile_specific_layers.get(station_id, {})
        if defined_layers:
            st.markdown("**Defined Layers for this Profile:**")
            for layer, data in sorted(defined_layers.items()): st.write(f"  - Layer {layer}: `{data['range'][0]}m` - `{data['range'][1]}m`")
            if st.button("Clear Layers for This Profile"): st.session_state.profile_specific_layers[station_id] = {}; st.rerun()

# === 6. UI: MAIN CONTENT ===
if not st.session_state.station_data:
    st.info("Welcome! Please upload your data files using the sidebar to begin.")
else:
    # --- UI STATE MANAGEMENT ---
    # This block resets certain session state variables when you switch to a new station.
    if 'previous_station' not in st.session_state or st.session_state.previous_station != st.session_state.selected_station:
        st.session_state.mixture_results = None
        st.session_state.mixture_df = None
        st.session_state.external_figs = {}
        st.session_state.derived_products = None
        st.session_state.previous_station = st.session_state.selected_station
        st.session_state.discarded_indices = []
        st.session_state.new_es_median = None

    # --- NOTIFICATION HANDLER ---
    # This is the CORRECT location for this block.
    # It displays any success/error messages left by callback functions.
    if st.session_state.notification:
        msg = st.session_state.notification['message']
        icon = "✅" if st.session_state.notification['type'] == 'success' else "❌"
        
        # Use toast instead of a static alert box
        st.toast(msg, icon=icon)
        
        # Clear the notification state
        st.session_state.notification = None # Clear the message after showing it once

    # --- DATA PREPARATION AND CENTRAL CALCULATIONS ---
    # This section prepares all the necessary data that will be used across different tabs.
    station_id = st.session_state.selected_station
    station_data = st.session_state.station_data[station_id]
    es_for_rrs = station_data.get('Es', None)
    phys_props_df = calculate_physical_properties(station_data, st.session_state.lon, st.session_state.lat)
    
    current_station_layers = st.session_state.profile_specific_layers.get(station_id, {})
    if current_station_layers:
        st.session_state.derived_products = calculate_derived_products(
            station_data, station_id, current_station_layers)
    else:
        st.session_state.derived_products = None
    
    derived_products = st.session_state.get('derived_products')
    if derived_products:
        layer_k_data = derived_products['layer_k_data']
        results_df = derived_products['results_df']
        kd_par_df = derived_products['kd_par_df']
        st.session_state.results_df = results_df
        st.session_state.kd_par_df = kd_par_df
    else:
        layer_k_data = {}
        results_df = None
        kd_par_df = None
        st.session_state.results_df = None
        st.session_state.kd_par_df = None

    if derived_products and st.session_state.get('external_probe_df') is not None:
        df_ext_raw = st.session_state.external_probe_df
        ext_depth_col = st.session_state.get('kd_ext_depth_par')
        ext_par_col = st.session_state.get('kd_ext_par')
        if ext_depth_col and ext_par_col:
            comparison_data = []
            model = st.session_state.get('empirical_model_params')
            for index, profiler_row in kd_par_df.iterrows():
                layer_num = int(profiler_row['Layer'].split(' ')[1])
                z_min, z_max = current_station_layers[layer_num]['range']
                external_kd_par = calculate_external_kd_par(df_ext_raw, ext_depth_col, ext_par_col, z_min, z_max)
                estimated_kd490_external = np.nan
                if model and not pd.isna(external_kd_par):
                    estimated_kd490_external = model['slope'] * external_kd_par + model['intercept']
                comparison_data.append({
                    "Layer": profiler_row['Layer'],
                    "Kd(PAR) (Profiler)": profiler_row['Kd(PAR)'],
                    "Kd(PAR) (External)": external_kd_par,
                    "Kd(490) (Profiler)": results_df.loc[index, 'Kd(490)'],
                    "Estimated Kd(490) (External)": estimated_kd490_external
                })
            st.session_state.comparison_kd_df = pd.DataFrame(comparison_data).set_index('Layer')
    else:
        st.session_state.comparison_kd_df = None
        
    fig_n2, fig_rrs, fig_log, fig_k, fig_ts = (plt.figure() for i in range(5))

    # --- TAB CREATION ---
    # This is the SINGLE, CORRECT place to define and create the tabs.
    def on_tab_change():
        st.session_state.active_tab = st.session_state.main_tabs
    
    tab_titles = [
        "CTD & Radiometric Comparison", "Surface Irradiance (Es)", "Core Optical Analysis", 
        "T-S Diagram", "Bio-Optical Models", "Secchi & Kd Analysis", 
        "Spectral Convolution", "Generate Report & L3 Data", "Formulas & Methods"
    ]
    
    tab_ctd, tab_es, tab_core, tab_ts, tab_models, tab_secchi, tab_convolution, tab_report, tab_formulas = st.tabs(tab_titles)

    with tab_ctd:
        st.header("External Probe Data Comparison"); st.markdown("Use this tab to upload your external probe data and compare its raw measurements against the profiler's derived products.")
        st.subheader("1. Upload External Probe Data"); uploaded_external_probe_file = st.file_uploader("Upload External Probe CSV (containing CTD, PAR, and Secchi data)", type="csv", key="external_probe_uploader"); col_sep, col_dec = st.columns(2); sep = col_sep.radio("Select CSV separator:", (',', ';'), key='probe_sep', horizontal=True); dec = col_dec.radio("Select CSV decimal:", ('.', ','), key='probe_dec', horizontal=True)
        if uploaded_external_probe_file:
            try: st.session_state.external_probe_df = pd.read_csv(uploaded_external_probe_file, sep=sep, decimal=dec); st.session_state.external_probe_filename = uploaded_external_probe_file.name; st.toast("External probe file loaded successfully!")
            except Exception as e: st.error(f"Could not parse file. Check format/separator/decimal. Error: {e}"); st.session_state.external_probe_df = None; st.session_state.external_probe_filename = None
        st.divider(); st.subheader("2. CTD Comparison")
        with st.expander("Show/Hide CTD Comparison", expanded=True):
            if st.session_state.get('external_probe_df') is None: st.info("Upload an external probe file above to compare CTD profiles.")
            else:
                df_external_raw = st.session_state.external_probe_df; st.markdown("**Map External CTD Columns:**"); c1,c2,c3 = st.columns(3); ext_cols_list = df_external_raw.columns.tolist(); depth_col_ext = c1.selectbox("Depth/Pressure Column", ext_cols_list, index=min(0, len(ext_cols_list)-1)); temp_col_ext = c2.selectbox("Temperature Column", ext_cols_list, index=min(1, len(ext_cols_list)-1)); sal_col_ext = c3.selectbox("Salinity Column", ext_cols_list, index=min(2, len(ext_cols_list)-1))
                st.markdown("**Select Profiler Casts to Compare:**"); profiler_casts_to_compare = st.multiselect("Select one or more profiler casts:", options=st.session_state.station_list, default=[st.session_state.selected_station] if st.session_state.selected_station in st.session_state.station_list else None)
                if st.button("Compare Selected CTD Profiles"):
                    if not profiler_casts_to_compare: st.warning("Please select at least one profiler cast to compare.")
                    else:
                        df_external = df_external_raw.copy(); df_external['Depth'] = pd.to_numeric(df_external[depth_col_ext], errors='coerce'); df_external['Temperature'] = pd.to_numeric(df_external[temp_col_ext], errors='coerce'); df_external['Salinity'] = pd.to_numeric(df_external[sal_col_ext], errors='coerce'); df_external = df_external.dropna(subset=['Depth', 'Temperature', 'Salinity'])
                        if not df_external.empty:
                            all_stats, all_merged_dfs = [], []
                            for cast_id in profiler_casts_to_compare:
                                df_internal_loop = calculate_physical_properties(st.session_state.station_data[cast_id], st.session_state.lon, st.session_state.lat); df_merged_loop = compare_ctd_profiles(df_internal_loop, df_external)
                                if df_merged_loop is not None and not df_merged_loop.empty:
                                    temp_stats = get_comparison_stats(df_merged_loop, 'Temperature'); sal_stats = get_comparison_stats(df_merged_loop, 'Salinity'); all_stats.append({'Cast ID': cast_id, 'Temp R2': temp_stats['R2'], 'Temp RMSE': temp_stats['RMSE'], 'Temp MAE': temp_stats['MAE'], 'Temp Bias': temp_stats['Bias'], 'Sal R2': sal_stats['R2'], 'Sal RMSE': sal_stats['RMSE'], 'Sal MAE': sal_stats['MAE'], 'Sal Bias': sal_stats['Bias']}); df_merged_loop['Cast ID'] = cast_id; all_merged_dfs.append(df_merged_loop)
                            if all_stats:
                                st.markdown("---"); st.subheader("CTD Comparison Plots"); fig_comp, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row'); ((ax_t, ax_s), (ax_td, ax_sd)) = axes; colors = sns.color_palette("viridis", len(all_merged_dfs)); ax_t.plot(df_external['Temperature'], df_external['Depth'], label='External', color='white', linestyle='--', lw=2.5, zorder=10); ax_s.plot(df_external['Salinity'], df_external['Depth'], label='External', color='white', linestyle='--', lw=2.5, zorder=10)
                                for i, df_plot in enumerate(all_merged_dfs):
                                    cast_id = df_plot['Cast ID'].iloc[0]; ax_t.plot(df_plot['Temperature_internal'], df_plot['Depth'], label=cast_id, color=colors[i], alpha=0.8); ax_s.plot(df_plot['Salinity_internal'], df_plot['Depth'], label=cast_id, color=colors[i], alpha=0.8); temp_diff = df_plot['Temperature_internal'] - df_plot['Temperature_external']; sal_diff = df_plot['Salinity_internal'] - df_plot['Salinity_external']; ax_td.plot(temp_diff, df_plot['Depth'], label=cast_id, color=colors[i], alpha=0.8); ax_sd.plot(sal_diff, df_plot['Depth'], label=cast_id, color=colors[i], alpha=0.8)
                                ax_t.set_title("Temperature Profiles"); ax_t.set_xlabel("Temperature (°C)"); ax_t.set_ylabel("Depth (m)"); ax_t.invert_yaxis(); ax_t.grid(True); ax_t.legend(); ax_s.set_title("Salinity Profiles"); ax_s.set_xlabel("Salinity (psu)"); ax_s.grid(True); ax_td.set_title("Temperature Difference"); ax_td.set_xlabel("Δ Temperature (°C)"); ax_td.axvline(0, color='white', linestyle='--'); ax_td.grid(True); ax_sd.set_title("Salinity Difference"); ax_sd.set_xlabel("Δ Salinity (psu)"); ax_sd.axvline(0, color='white', linestyle='--'); ax_sd.grid(True); plt.tight_layout(); st.pyplot(fig_comp); st.session_state.fig_comp = fig_comp
                                st.markdown("---"); st.subheader("Comparison Summary Table"); stats_summary_df = pd.DataFrame(all_stats).set_index('Cast ID'); st.session_state.comparison_table = stats_summary_df; st.dataframe(stats_summary_df.style.apply(highlight_best).format('{:.4f}'), use_container_width=True)
                            else: st.warning("No overlapping depth range found for any of the selected casts.")
                        else: st.warning("No valid numeric data found in the selected external CTD columns.")
        st.divider(); st.subheader("3. Comparative Kd Analysis (Profiler vs. External)")
        with st.expander("Show/Hide Kd Comparison", expanded=True):
            if st.session_state.get('external_probe_df') is None: st.info("Upload an external probe file above to enable Kd comparison.")
            elif not derived_products or kd_par_df is None: st.info("Analyze at least one layer in the sidebar to calculate profiler Kd values.")
            else:
                st.markdown("**Map External Probe Columns for Kd Calculation:**"); df_ext_raw = st.session_state.external_probe_df; ext_cols_list = df_ext_raw.columns.tolist(); c1, c2 = st.columns(2)
                c1.selectbox("Select Probe Depth Column:", ext_cols_list, key="kd_ext_depth_par"); c2.selectbox("Select Probe PAR Column:", ext_cols_list, key="kd_ext_par")
                if st.session_state.comparison_kd_df is not None:
                    st.markdown("**Comparison Table**"); st.dataframe(st.session_state.comparison_kd_df.style.format('{:.4f}'), use_container_width=True)
                    if not st.session_state.get('empirical_model_params'): st.info("Go to the 'Bio-Optical Models' tab to generate a site-specific empirical model. The 'Estimated Kd(490)' column will then be populated.")
                else: st.warning("Please map the correct Depth and PAR columns for the external probe.")

   
    # Replace your entire `with tab_es:` block with this restructured version.

    with tab_es:

        st.header("Surface Irradiance (Es) Visualization")
    
        all_es_spectra = station_data.get("Es_all_spectra", None)
    
        # Clean safe index
        if all_es_spectra is not None and not all_es_spectra.empty:
            all_es_spectra = all_es_spectra.reset_index(drop=True)
    
        # If no Es → stop
        if all_es_spectra is None or all_es_spectra.empty:
            st.warning("No measured Es file was uploaded for this station.")
            st.stop()
    
        # Create session state variables
        if "discarded_indices" not in st.session_state:
            st.session_state.discarded_indices = []
        if "new_es_median" not in st.session_state:
            st.session_state.new_es_median = None
    
        #  --- START OF INTELLIGENT COUNTING FIX ---
        # 1. Identify rows that have at least one valid number (are not all NA).
        #    We use dropna(how='all') which only removes rows where EVERY column is NA.
        #    This gives us a DataFrame of only the usable spectra.
        valid_spectra_df = all_es_spectra.dropna(how='all', subset=[c for c in all_es_spectra.columns if c.startswith('X')])
        
        # 2. The original count is now the number of VALID spectra.
        original_valid_count = len(valid_spectra_df)
        
        # 3. Calculate how many of these valid spectra were kept.
        # We find the intersection of the valid indices and the kept indices.
        discard_set = set(st.session_state.discarded_indices)
        kept_indices = all_es_spectra.index.difference(discard_set)
        final_kept_valid_indices = valid_spectra_df.index.intersection(kept_indices)
        final_kept_count = len(final_kept_valid_indices)
        #  --- END OF INTELLIGENT COUNTING FIX ---
    
        # ------------------------------
        #   Summary metric (now uses the new, accurate counts)
        # ------------------------------
        st.metric("Valid Spectra After Filtering", f"{final_kept_count} / {original_valid_count}")
    
        # ------------------------------
        #   Plot Es spectra (no changes needed here)
        # ------------------------------
        fig_es, ax_es = plt.subplots(figsize=(8, 6))
        wavelength_values = [
            float(col.replace(WAVELENGTH_PREFIX, "")) for col in all_es_spectra.columns
        ]
        added_disc = False
        added_keep = False
    
        for idx, spectrum in all_es_spectra.iterrows():
            if idx in discard_set:
                label = "Discarded Spectra" if not added_disc else ""
                ax_es.plot(wavelength_values, spectrum.values, lw=0.5, alpha=0.35,
                           color="#FF5733", label=label)
                added_disc = True
            else:
                # We also check if the row is valid before plotting it as kept
                if not spectrum.dropna().empty:
                    label = "Kept Spectra" if not added_keep else ""
                    ax_es.plot(wavelength_values, spectrum.values, lw=0.5, alpha=0.6,
                               color="#888888", label=label)
                    added_keep = True
    
        # Original median (calculated only on valid spectra now for accuracy)
        original_median_es = valid_spectra_df.median().values
        ax_es.plot(wavelength_values, original_median_es, lw=2.0, linestyle="--",
                   color="cyan", label=f"Original Median ({original_valid_count} valid spectra)")
    
        # New filtered median
        if st.session_state.new_es_median is not None:
            ax_es.plot(wavelength_values, st.session_state.new_es_median, lw=3.0,
                       color="#33FF57", label=f"New Median ({final_kept_count} spectra)")
    
        ax_es.set_xlabel("Wavelength (nm)", weight="bold")
        ax_es.set_ylabel(r"Es ($\mu$W cm$^{-2}$ nm$^{-1}$)", weight="bold")
        ax_es.set_title("Measured Es vs. Propagated Ed", weight="bold")
        ax_es.set_xlim(380, 750)
        ax_es.set_ylim(bottom=0)
        ax_es.grid(True, linestyle='--')
        ax_es.legend()
        st.pyplot(fig_es)
    
        # ============================================================
        #   FILTERING PANEL (No changes needed in the logic here)
        # ============================================================
        st.subheader("Es Stability Analysis & Hybrid Filtering")
        default_index = int(np.abs(np.array([float(c.replace("X", "")) for c in all_es_spectra.columns]) - 490).argmin())
        col_name_for_analysis = st.selectbox("Select reference wavelength for stability check:", options=list(all_es_spectra.columns), index=default_index)
        
        with st.form("hybrid_es_filters_form"):
            # ... (form UI elements are unchanged) ...
            st.markdown("### Active Filters")
            use_tilt_filter = st.checkbox("Enable Tilt Filter", value=True)
            threshold_pct = st.number_input("Tilt threshold (% change)", 0.1, 50.0, 10.0, 0.5)
            use_low_filter = st.checkbox("Enable Low Outlier Filter", value=True)
            low_pct = st.number_input("Discard lowest %", 0.0, 49.9, 20.0, 1.0)
            use_high_filter = st.checkbox("Enable High Outlier Filter (Hybrid)", value=True)
            high_pct = st.number_input("Discard highest %", 0.0, 49.9, 15.0, 0.5)
            submitted = st.form_submit_button("Apply Filters")
    
        if submitted:
            # ... (The entire filtering logic block is unchanged) ...
            discard_tilt = set()
            if use_tilt_filter:
                pct_change = all_es_spectra[col_name_for_analysis].pct_change().abs() * 100
                discard_tilt = set(pct_change[pct_change > threshold_pct].index)
            keep_after_tilt = all_es_spectra.index.difference(discard_tilt)
            spectra_after_tilt = all_es_spectra.loc[keep_after_tilt]
            discard_low, discard_high = set(), set()
            if not spectra_after_tilt.empty:
                hybrid_metric = (0.25 * spectra_after_tilt.max(axis=1) + 0.75 * spectra_after_tilt.sum(axis=1))
                if use_low_filter:
                    num_low = int(len(spectra_after_tilt) * (low_pct / 100))
                    discard_low = set(hybrid_metric.nsmallest(num_low).index)
                if use_high_filter:
                    num_high = int(len(spectra_after_tilt) * (high_pct / 100))
                    discard_high = set(hybrid_metric.nlargest(num_high).index)
            all_discards = sorted(list(discard_tilt | discard_low | discard_high))
            st.session_state.discarded_indices = all_discards
            keep_indices = all_es_spectra.index.difference(all_discards)
            if len(keep_indices) > 0:
                st.session_state.new_es_median = all_es_spectra.loc[keep_indices].median().values
            else:
                st.session_state.new_es_median = None
            st.rerun()
    
        if st.button("Clear All Filters & Reset"):
            st.session_state.discarded_indices = []
            st.session_state.new_es_median = None
            st.rerun()
    
    # ===================================================================
    # END OF ES BLOCK
    # ===================================================================

    with tab_core:
        st.header("Core Optical Analysis")
        top_row, bottom_row = st.columns(2), st.columns(2); plot_figsize = (5.5, 4.0)
    
        # --- Define a consistent color palette for layers ---
        layer_colors = sns.color_palette("bright", 3)
        
        # Get the analysis layers specific to the currently selected station
        current_station_id = st.session_state.selected_station
        current_station_layers = st.session_state.profile_specific_layers.get(current_station_id, {})
    
        with top_row[0]:
            fig_n2, ax_n2 = plt.subplots(figsize=plot_figsize)
            line_n2, = ax_n2.plot(phys_props_df['n2'].dropna()*1e5, phys_props_df['p_mid'].dropna(), color=STYLE_CONFIG['n2_line_color'], zorder=10, label=r'N$^2$')
            ax_n2.set_xlabel(r'N$^2$ (x 10$^{-5}$ s$^{-2}$)', weight='bold', color=STYLE_CONFIG['n2_line_color']); ax_n2.set_ylabel("Depth (m)", weight='bold')
            ax_n2.set_title("1. Stratification & Backscattering", weight='bold'); ax_n2.invert_yaxis(); ax_n2.tick_params(axis='x', labelcolor=STYLE_CONFIG['n2_line_color'])
            beta_data_df = station_data.get('beta_data')
            
            if beta_data_df is not None and not beta_data_df.empty and 'selected_beta_col' in st.session_state:
                ax_beta = ax_n2.twiny(); selected_beta_col = st.session_state.selected_beta_col
                line_beta, = ax_beta.plot(beta_data_df[selected_beta_col], beta_data_df.index, color=STYLE_CONFIG['beta_line_color'], zorder=9, label=f'Beta ({selected_beta_col})')
                ax_beta.set_xlabel(r'$\beta$ (m$^{-1}$ sr$^{-1}$)', weight='bold', color=STYLE_CONFIG['beta_line_color']); ax_beta.tick_params(axis='x', labelcolor=STYLE_CONFIG['beta_line_color'])
                formatter = ScalarFormatter(useMathText=True); formatter.set_scientific(True); formatter.set_powerlimits((-3, 3))
                ax_beta.xaxis.set_major_formatter(formatter); ax_beta.xaxis.offsetText.set_color(STYLE_CONFIG['beta_line_color'])
                lines = [line_n2, line_beta]; ax_n2.legend(lines, [l.get_label() for l in lines], loc='best')
            else: ax_n2.legend(loc='best')
            ax_n2.grid(True, linestyle='--')
            
            for i, (layer_num, data) in enumerate(current_station_layers.items()):
                ax_n2.axhspan(*data['range'], facecolor=layer_colors[i], alpha=0.3, zorder=0)
            st.pyplot(fig_n2); st.session_state.fig_n2 = fig_n2
    
        with top_row[1]:
            y_max_rrs = st.number_input("Set Y-axis Max for Rrs Plot:", min_value=0.001, value=0.010, step=0.001, format="%.3f", key="rrs_ymax_selector")
            fig_rrs, ax_rrs = plt.subplots(figsize=plot_figsize)
            
            # --- START OF PLOTTING FIX & IMPROVEMENT ---
            # Instead of re-calculating, we now directly use the results stored in the derived_products dictionary.
            # This is more efficient and less error-prone.
            if derived_products and 'rrs_df_export' in derived_products:
                rrs_df = derived_products['rrs_df_export']
                
                # Plot initial Rrs from the dataframe
                if 'initial_rrs_sr-1' in rrs_df.columns:
                    ax_rrs.plot(rrs_df['wavelength_nm'], rrs_df['initial_rrs_sr-1'], color='white', linestyle='--', label='Initial Rrs(0-)')
                
                # Plot propagated Rrs for each layer that was successfully calculated and stored in the dataframe
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    col_name = f'propagated_rrs_L{layer_num}_sr-1'
                    if col_name in rrs_df.columns:
                        ax_rrs.plot(rrs_df['wavelength_nm'], rrs_df[col_name], color=layer_colors[i], label=f'Prop. Rrs (L{layer_num})')
            else:
                # This is a fallback for the case where no layers have been analyzed yet, ensuring the initial Rrs plot always shows.
                final_es_for_rrs = st.session_state.get('new_es_median', station_data.get('Es'))
                initial_rrs = calculate_Rrs(station_data['Ed_data'], station_data['Lu_data'], pressure=station_data['pressure'], es_data=final_es_for_rrs)
                ax_rrs.plot(station_data['wavelengths'], initial_rrs, color='white', linestyle='--', label='Initial Rrs(0-)')
            # --- END OF PLOTTING FIX & IMPROVEMENT ---
    
            ax_rrs.set_xlabel("Wavelength (nm)", weight='bold'); ax_rrs.set_ylabel(r'Rrs (sr$^{-1}$)', weight='bold'); ax_rrs.set_title("2. Remote Sensing Reflectance (Rrs)", weight='bold')
            ax_rrs.set_xlim(400, 700); ax_rrs.set_ylim(bottom=0, top=y_max_rrs)
            ax_rrs.grid(True, linestyle='--'); ax_rrs.legend(fontsize=8)
            st.pyplot(fig_rrs); st.session_state.fig_rrs = fig_rrs
    
        with bottom_row[0]:
            fig_log, ax_log = plt.subplots(figsize=plot_figsize)
            wl_indices = {'490': (np.abs(station_data['wavelengths'] - 490)).argmin(), '555': (np.abs(station_data['wavelengths'] - 555)).argmin(), '600': (np.abs(station_data['wavelengths'] - 600)).argmin()}
            wl_colors = {'490': '#4da6ff', '555': '#33cc33', '600': '#ff9933'}
            for key, idx in wl_indices.items():
                wl = station_data['wavelengths'][idx]; depth_col = station_data['pressure']; ed_col = station_data['Ed_data'][:, idx]; lu_col = station_data['Lu_data'][:, idx]
                df_ed_plot = pd.DataFrame({'depth': depth_col, 'ed': ed_col}).dropna(); df_ed_plot = df_ed_plot[df_ed_plot['ed'] > 0]
                ax_log.plot(np.log(df_ed_plot['ed']), df_ed_plot['depth'], label=f'ln(Ed) @ {wl:.1f} nm', color=wl_colors[key], linestyle='-')
                df_lu_plot = pd.DataFrame({'depth': depth_col, 'lu': lu_col}).dropna(); df_lu_plot = df_lu_plot[df_lu_plot['lu'] > 0]
                ax_log.plot(np.log(df_lu_plot['lu']), df_lu_plot['depth'], label=f'ln(Lu) @ {wl:.1f} nm', color=wl_colors[key], linestyle='--')
            
            for i, (layer_num, data) in enumerate(current_station_layers.items()):
                ax_log.axhspan(*data['range'], facecolor=layer_colors[i], alpha=0.3, zorder=0)
            
            ax_log.set_xlabel("ln(Radiance / Irradiance)", weight='bold'); ax_log.set_ylabel("Depth (m)", weight='bold'); ax_log.set_title("3. Log-Linear Profiles", weight='bold')
            ax_log.invert_yaxis(); ax_log.grid(True, linestyle='--'); ax_log.legend(fontsize=8)
            st.pyplot(fig_log); st.session_state.fig_log = fig_log
    
        with bottom_row[1]:
            fig_k, ax_k = plt.subplots(figsize=plot_figsize)
            
            if current_station_layers and derived_products:
                layer_k_data = derived_products.get('layer_k_data', {})
                df_kw = get_smith_baker_kw()
                kw_interp = np.interp(station_data['wavelengths'], df_kw['wavelength'], df_kw['Kw'])
                ax_k.plot(station_data['wavelengths'], kw_interp, color='white', linestyle=':', lw=2, label='Kw (Pure Water)', zorder=5)
                
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    if layer_num in layer_k_data:
                        kd_plot, klu_plot = layer_k_data[layer_num]
                        ax_k.plot(station_data['wavelengths'], kd_plot, color=layer_colors[i], linestyle='-', label=f'Kd (L{layer_num})', zorder=10)
                        ax_k.plot(station_data['wavelengths'], klu_plot, color=layer_colors[i], linestyle='--', label=f'klu (L{layer_num})', zorder=10)
            
            ax_k.set_xlabel("Wavelength (nm)", weight='bold'); ax_k.set_ylabel(r'K (m$^{-1}$)', weight='bold'); ax_k.set_title("4. Attenuation Spectra", weight='bold')
            ax_k.set_ylim(bottom=0); ax_k.grid(True, linestyle='--')
            if current_station_layers: ax_k.legend(fontsize=8)
            st.pyplot(fig_k); st.session_state.fig_k = fig_k
    
        st.subheader("Analysis Results")
        if not current_station_layers:
            st.info("Use the sidebar to select and analyze depth layers for the current profile.")
        else:
            if derived_products:
                st.dataframe(derived_products['results_df'].set_index('Layer').style.format({'Kd(490)':'{:.4f}', 'klu(490)':'{:.4f}'}), use_container_width=True)
                st.markdown("---")
                st.subheader("Kd(PAR) Results")
                st.dataframe(derived_products['kd_par_df'].set_index('Layer').style.format({'Kd(PAR)':'{:.4f}'}), use_container_width=True)
    
        st.markdown("---")
        with st.expander("5. Final Irradiance Comparison (Surface vs. Propagated)", expanded=True):
            if not current_station_layers or not derived_products:
                st.warning("Analyze at least one layer to see the comparison plot.")
            else:
                y_max_es_comp = st.number_input("Set Y-axis Max for Irradiance Plot:", min_value=1.0, value=150.0, step=10.0, format="%.1f", key="es_comp_ymax_selector")
                fig_final_comp, ax_final_comp = plt.subplots(figsize=(8, 6))
                final_es_median = st.session_state.get('new_es_median', station_data.get('Es'))
                if final_es_median is not None:
                    label_es = "Final Es Median (Filtered)" if st.session_state.new_es_median is not None else "Original Es Median"
                    ax_final_comp.plot(station_data['wavelengths'], final_es_median, color='#33FF57', label=label_es, lw=3.0)
                
                layer_k_data = derived_products.get('layer_k_data', {})
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    if layer_num in layer_k_data:
                        kd_for_prop, _ = layer_k_data[layer_num]
                        z_top = data['range'][0]
                        ed_at_z_top = np.array([np.interp(z_top, station_data['pressure'], station_data['Ed_data'][:, j]) for j in range(station_data['Ed_data'].shape[1])])
                        propagated_ed = ed_at_z_top * np.exp(kd_for_prop * z_top)
                        ax_final_comp.plot(station_data['wavelengths'], propagated_ed, color=layer_colors[i], linestyle='--', label=f'Propagated Ed(0+) (from L{layer_num})', lw=2.0)
                
                ax_final_comp.set_xlabel("Wavelength (nm)", weight='bold'); ax_final_comp.set_ylabel(r'Irradiance ($\mu$W cm$^{-2}$ nm$^{-1}$)', weight='bold')
                ax_final_comp.set_title("Final Surface Irradiance Comparison", weight='bold'); ax_final_comp.set_xlim(380, 750); ax_final_comp.set_ylim(bottom=0, top=y_max_es_comp)
                ax_final_comp.grid(True, linestyle='--'); ax_final_comp.legend()
                st.pyplot(fig_final_comp)
                
                if final_es_median is not None:
                    df_to_save = pd.DataFrame({'wavelength_nm': station_data['wavelengths'], 'median_es_uW_cm-2_nm-1': final_es_median})
                    st.download_button(
                        label="Save Final Es Median to CSV", data=df_to_save.to_csv(index=False).encode('utf-8'),
                        file_name=f"{station_id}_final_Es_median.csv", mime='text/csv', key='download_final_es'
                    )
                st.info("**How to interpret this plot:** This plot compares the final measured surface irradiance (`Es`) with the in-water irradiance (`Ed`) propagated to the surface using the `Kd` from each analysis layer. A close match provides high confidence in the quality of both the surface and in-water measurements.")
   # ===================================================================
    # TAB 4: T-S DIAGRAM (VERSÃO FINAL COMPLETA, COM NOVOS PADRÕES)
    # ===================================================================
    with tab_ts:
        st.header("T-S Diagram and Water Mass Analysis")
        
        # --- Parte 1: Controles e Plot Principal em Colunas ---
        col_plot, col_controls = st.columns([2, 1])
        
        with col_controls:
            st.subheader("1. Data Sources")
            
            show_external = st.toggle(
                "Show External Probe Data", 
                value=True, 
                disabled=(st.session_state.get('external_probe_df') is None)
            )
            
            df_external_ts = None
            if st.session_state.get('external_probe_df') is None:
                st.info("Upload an external probe file on the 'CTD & Radiometric Comparison' tab to enable comparison.")
            
            if show_external and st.session_state.get('external_probe_df') is not None:
                df_ext = st.session_state.external_probe_df
                st.markdown("##### Map External Probe Columns")
                ext_temp_col = st.selectbox("Potential Temperature [°C] Column:", df_ext.columns, key="ts_ext_temp_en")
                ext_sal_col = st.selectbox("Absolute Salinity [PSU] Column:", df_ext.columns, key="ts_ext_sal_en")
                
                if pd.api.types.is_numeric_dtype(df_ext[ext_temp_col]) and pd.api.types.is_numeric_dtype(df_ext[ext_sal_col]):
                    df_external_ts = df_ext[[ext_temp_col, ext_sal_col]].copy()
                # No need for pd.to_numeric here anymore, as we've already confirmed the type
                    df_external_ts.rename(columns={ext_temp_col: 'Temperature', ext_sal_col: 'Salinity'}, inplace=True)
                    df_external_ts.dropna(inplace=True)
                else:
                    st.warning("Please select numeric columns for both Temperature and Salinity to display the external probe data.")
    
            st.divider()
            st.subheader("2. Define Mixture")
            with st.expander("Define Water Mass Vertices", expanded=True):
                c1,c2,c3 = st.columns(3)
                
                #  --- NOVOS VALORES PADRÃO APLICADOS ---
                st.session_state.wm_names[0] = c1.text_input("Name", value="AT", key="wm1_name")
                wm1_sal = c2.number_input("Salinity", value=37.2, format="%.2f", key="wm1_sal")
                wm1_temp = c3.number_input("Temp (°C)", value=25.29, format="%.2f", key="wm1_temp")
                
                st.session_state.wm_names[1] = c1.text_input("Name", value="ACAS", key="wm2_name")
                wm2_sal = c2.number_input("Salinity", value=35.19, format="%.2f", key="wm2_sal")
                wm2_temp = c3.number_input("Temp (°C)", value=12.67, format="%.2f", key="wm2_temp")
                
                st.session_state.wm_names[2] = c1.text_input("Name", value="AC", key="wm3_name")
                wm3_sal = c2.number_input("Salinity", value=34.36, format="%.2f", key="wm3_sal")
                wm3_temp = c3.number_input("Temp (°C)", value=27.54, format="%.2f", key="wm3_temp")
                # --- FIM DOS NOVOS VALORES ---
    
                if st.button("Analyze Mixture and Draw Triangle"):
                    st.session_state.wm_vertices = [(wm1_sal, wm1_temp), (wm2_sal, wm2_temp), (wm3_sal, wm3_temp)]
                    
                    # Análise para o Perfilador
                    results_text_profiler, df_profiler = analyze_mixture(phys_props_df['Salinity'], phys_props_df['Temperature'], st.session_state.wm_vertices, st.session_state.wm_names)
                    st.session_state.mixture_df_profiler = df_profiler
                    st.session_state.mixture_results_profiler = results_text_profiler
                    
                    # Limpa os resultados antigos do probe externo antes de recalcular
                    st.session_state.mixture_df_external = None
                    st.session_state.mixture_results_external = None
                    
                    # Análise para o Probe Externo (se existir)
                    if df_external_ts is not None and not df_external_ts.empty:
                        results_text_external, df_external = analyze_mixture(df_external_ts['Salinity'], df_external_ts['Temperature'], st.session_state.wm_vertices, st.session_state.wm_names)
                        st.session_state.mixture_df_external = df_external
                        st.session_state.mixture_results_external = results_text_external
    
        with col_plot:
            fig_ts, ax_ts = plt.subplots(figsize=(8, 8))
            
            df_internal_plot = phys_props_df[['Salinity', 'Temperature', 'Depth']].dropna()
            
            sa_min, sa_max = df_internal_plot['Salinity'].min(), df_internal_plot['Salinity'].max()
            ct_min, ct_max = df_internal_plot['Temperature'].min(), df_internal_plot['Temperature'].max()
    
            if df_external_ts is not None and not df_external_ts.empty:
                sa_min = min(sa_min, df_external_ts['Salinity'].min())
                sa_max = max(sa_max, df_external_ts['Salinity'].max())
                ct_min = min(ct_min, df_external_ts['Temperature'].min())
                ct_max = max(ct_max, df_external_ts['Temperature'].max())
                
            if st.session_state.wm_vertices:
                sal_verts = [v[0] for v in st.session_state.wm_vertices]
                temp_verts = [v[1] for v in st.session_state.wm_vertices]
                sa_min = min(sa_min, min(sal_verts))
                sa_max = max(sa_max, max(sal_verts))
                ct_min = min(ct_min, min(temp_verts))
                ct_max = max(ct_max, max(temp_verts))
            
            sal_padding = (sa_max - sa_min) * 0.1
            temp_padding = (ct_max - ct_min) * 0.1
            sal_lim = (sa_min - sal_padding, sa_max + sal_padding)
            temp_lim = (ct_min - temp_padding, ct_max + temp_padding)
                
            SA_grid, CT_grid = np.meshgrid(np.linspace(sal_lim[0], sal_lim[1], 100), np.linspace(temp_lim[0], temp_lim[1], 100))
            sigma0_grid = gsw.sigma0(SA_grid, CT_grid)
            cs = ax_ts.contour(SA_grid, CT_grid, sigma0_grid, colors=STYLE_CONFIG['gridcolor'], linestyles='--', linewidths=1.0)
            ax_ts.clabel(cs, cs.levels, inline=True, fontsize=9, fmt='%1.1f')
    
            if df_external_ts is not None and not df_external_ts.empty:
                ax_ts.scatter(df_external_ts['Salinity'], df_external_ts['Temperature'], c='#BBBBBB', s=20, alpha=0.7, label='External Probe', zorder=10)
            
            sc = ax_ts.scatter(df_internal_plot['Salinity'], df_internal_plot['Temperature'], c=df_internal_plot['Depth'], cmap='viridis_r', s=40, ec='black', lw=0.5, label='Profiler', zorder=20)
            cbar = fig_ts.colorbar(sc, ax=ax_ts, pad=0.02, label='Depth (m)')
            cbar.ax.invert_yaxis()
    
            if st.session_state.wm_vertices:
                poly = Polygon(np.array(st.session_state.wm_vertices), closed=True, fill=False, edgecolor='cyan', linestyle='--', lw=2.0, zorder=30)
                ax_ts.add_patch(poly)
                for i, (sal, temp) in enumerate(st.session_state.wm_vertices):
                    ax_ts.text(sal, temp, f" {st.session_state.wm_names[i]}", fontsize=11, weight='bold', color='cyan', zorder=31)
            
            ax_ts.set_xlim(sal_lim); ax_ts.set_ylim(temp_lim)
            ax_ts.set_xlabel("Absolute Salinity [PSU]", weight='bold')
            ax_ts.set_ylabel("Potential Temperature [°C]", weight='bold')
            ax_ts.legend(loc='lower left')
            st.pyplot(fig_ts); st.session_state.fig_ts = fig_ts
    
        st.divider()
        
        # --- Parte 2: Seção de Análise de Mistura (Abaixo do Gráfico) ---
        st.header("Mixture Analysis Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("Profiler Data")
            if 'mixture_results_profiler' in st.session_state and st.session_state.mixture_results_profiler:
                st.info(st.session_state.mixture_results_profiler)
                
                if 'mixture_df_profiler' in st.session_state and st.session_state.mixture_df_profiler is not None:
                    fig_mix_prof, ax_mix_prof = plt.subplots(figsize=(6, 5))
                    df_plot = pd.concat([phys_props_df['Depth'], st.session_state.mixture_df_profiler], axis=1).dropna()
                    for name in st.session_state.wm_names:
                        ax_mix_prof.plot(df_plot[name] * 100, df_plot['Depth'], label=name)
                    ax_mix_prof.set_xlabel("Mixing Percentage (%)")
                    ax_mix_prof.set_ylabel("Depth (m)")
                    ax_mix_prof.set_title("Profiler: Water Mass Depth Profile")
                    ax_mix_prof.invert_yaxis()
                    ax_mix_prof.set_xlim(0, 100)
                    ax_mix_prof.grid(True, linestyle='--')
                    ax_mix_prof.legend()
                    st.pyplot(fig_mix_prof)
            else:
                st.info("Click 'Analyze Mixture' to see results for the profiler.")
    
        with res_col2:
            st.subheader("External Probe Data")
            if 'mixture_results_external' in st.session_state and st.session_state.mixture_results_external:
                st.info(st.session_state.mixture_results_external)
                
                if 'mixture_df_external' in st.session_state and st.session_state.mixture_df_external is not None:
                    fig_mix_ext, ax_mix_ext = plt.subplots(figsize=(6, 5))
                    df_plot = st.session_state.mixture_df_external.dropna()
                    depth_proxy = range(len(df_plot))
                    for name in st.session_state.wm_names:
                        ax_mix_ext.plot(df_plot[name] * 100, depth_proxy, label=name)
                    ax_mix_ext.set_xlabel("Mixing Percentage (%)")
                    ax_mix_ext.set_ylabel("Measurement Index (proxy for depth)")
                    ax_mix_ext.set_title("External Probe: Water Mass Profile")
                    ax_mix_ext.invert_yaxis()
                    ax_mix_ext.set_xlim(0, 100)
                    ax_mix_ext.grid(True, linestyle='--')
                    ax_mix_ext.legend()
                    st.pyplot(fig_mix_ext)
            else:
                st.info("Enable and map the external probe to see its mixture analysis results.")
    #  ===================================================================
    # TAB 5: BIO-OPTICAL MODELS
    # ===================================================================
    with tab_models:
        st.header("Bio-Optical Model Estimations")
        
        st.subheader("1. Site-Specific Empirical Model: Kd(490) vs. Kd(PAR)")
        with st.expander("Generate and Manage Your Empirical Model", expanded=True):
            st.info("""
            Build a custom model using your profiler's data. This allows for persistent, long-term analysis.
            - **Load Data:** Start with your master model CSV from previous sessions.
            - **Analyze Layers:** Process new profiles in the sidebar to add more data points.
            - **Generate Model:** Create an updated regression model.
            - **Download Data:** Save the combined old and new data to your master CSV for future use.
            """)
    
            st.markdown("#### Manage Model Data")
            col1, col2 = st.columns(2)
            with col1:
                # --- THIS IS THE FIX: Using an on_change callback ---
                st.file_uploader(
                    "Load Master Model Data (CSV)", 
                    type="csv", 
                    key="master_model_loader",
                    on_change=_handle_model_upload
                )
            
            with col2:
                if not st.session_state.empirical_model_data.empty:
                    st.download_button(
                        label="Download Model Data (CSV)",
                        data=st.session_state.empirical_model_data.to_csv(index=False).encode('utf-8'),
                        file_name="master_empirical_model_data.csv",
                        mime='text/csv',
                        key="download_model_data"
                    )
            st.markdown("---")
    
            if st.session_state.empirical_model_data.empty:
                st.warning("No data available yet. Analyze at least two layers with your profiler, or load a master model data file.")
            else:
                st.markdown("**Current Data for Model:**")
                
                # ✅ --- THIS IS THE FIX ---
                # 1. Define the exact column order you want.
                desired_order = ['Station_ID', 'Kd(PAR)', 'Kd(490)','Layer']
                
                # 2. Create a new DataFrame with the columns in that order.
                # We also check if all desired columns exist to prevent errors.
                cols_to_display = [col for col in desired_order if col in st.session_state.empirical_model_data.columns]
                df_to_display = st.session_state.empirical_model_data[cols_to_display]
                
                # 3. Display the correctly ordered DataFrame.
                st.dataframe(df_to_display, use_container_width=True)
    
                if st.button("Generate Empirical Model from Current Data"):
                    df_model = st.session_state.empirical_model_data.dropna()
                    if len(df_model) < 2:
                        st.error("At least two valid data points are required to build a model.")
                    else:
                        slope, intercept, r_value, p_value, _ = linregress(df_model['Kd(PAR)'], df_model['Kd(490)'])
                        st.session_state.empirical_model_params = {'slope': slope, 'intercept': intercept, 'r2': r_value**2}
                        
                        st.success("Empirical model generated successfully!")
                        st.metric("Model Equation", f"Kd(490) = {slope:.4f} * Kd(PAR) + {intercept:.4f}")
                        st.metric("Model R²", f"{r_value**2:.4f}")
    
                        fig_model, ax = plt.subplots()
                        ax.scatter(df_model['Kd(PAR)'], df_model['Kd(490)'], ec='white', label='Data Points')
                        x_vals = np.array(df_model['Kd(PAR)'])
                        y_vals = slope * x_vals + intercept
                        ax.plot(x_vals, y_vals, '--', color='cyan', label='Regression Line')
                        ax.set_xlabel("Kd(PAR) (m⁻¹)", weight='bold'); ax.set_ylabel("Kd(490) (m⁻¹)", weight='bold')
                        ax.set_title("Empirical Model: Kd(PAR) vs. Kd(490)", weight='bold'); ax.grid(True, linestyle=':'); ax.legend()
                        st.pyplot(fig_model)
    
        st.subheader("2. Generic Published Model (Morel & Maritorena, 2001)")
        with st.expander("Estimate Kd(490) from Kd(PAR) using Published Model"):
            st.markdown("""This model is most accurate for open-ocean (Case-1) waters and may differ from your site-specific model.""")
            kd_par_input = st.number_input("Enter a Kd(PAR) value (m⁻¹):", min_value=0.0, value=0.1, step=0.01, format="%.4f")
            if kd_par_input > 0:
                estimated_c = estimate_chlorophyll_from_kdpar(kd_par_input)
                estimated_kd490 = estimate_kd490_from_chlorophyll(estimated_c)
                col1, col2 = st.columns(2)
                with col1: st.metric(label="Estimated Chlorophyll-a (mg/m³)", value=f"{estimated_c:.4f}")
                with col2: st.metric(label="Estimated Kd(490) (m⁻¹)", value=f"{estimated_kd490:.4f}")
  
   # ===================================================================
    # TAB 6: SECCHI & KD ANALYSIS
    # ===================================================================
    with tab_secchi:
        st.header("Secchi Depth vs. Kd Relationship Analysis")
    
        with st.expander("Load/Manage Master Secchi Data"):
            # --- THIS IS THE FIX: Using an on_change callback ---
            st.file_uploader(
                "Load Master Secchi Data (CSV)", 
                type="csv", 
                key="master_secchi_loader",
                on_change=_handle_secchi_upload
            )
    
        if st.session_state.get('external_probe_df') is None:
            st.warning("To proceed, please upload your external probe data file in the 'CTD & Radiometric Comparison' tab.")
            st.stop()
    
        st.success("External probe data file found. You can now link Secchi data to Kd values from both instruments.")
        
        st.subheader("1. Link and Save Data")
        
        df_secchi_source = st.session_state.external_probe_df
        df_cols = df_secchi_source.columns.tolist()
        
        st.markdown("**Map Columns from External Probe File:**")
        c1, c2, c3, c4 = st.columns(4)
        station_col = c1.selectbox("Station ID Column:", df_cols, index=0, key="secchi_station_col")
        secchi_col = c2.selectbox("Secchi Depth Column:", df_cols, index=min(1, len(df_cols)-1), key="secchi_depth_col")
        ext_depth_col_secchi = c3.selectbox("Probe Depth Column:", df_cols, key="secchi_ext_depth")
        ext_par_col_secchi = c4.selectbox("Probe PAR Column:", df_cols, key="secchi_ext_par")
    
        st.markdown("---")
        profiler_station_name = st.session_state.selected_station
        st.write(f"**Active Profiler Station:** `{profiler_station_name}`")
        unique_external_stations = df_secchi_source[station_col].dropna().unique().tolist()
        
        selected_external_station = st.selectbox(
            "Select the matching station name from your external file:",
            unique_external_stations
        )
    
        if selected_external_station:
            secchi_row = df_secchi_source[df_secchi_source[station_col] == selected_external_station]
            if not secchi_row.empty:
                secchi_value = pd.to_numeric(secchi_row.iloc[0][secchi_col], errors='coerce')
                
                if derived_products:
                    st.success(f"Found Secchi depth for external station **'{selected_external_station}'**: **{secchi_value:.2f} m**")
                    
                    analyzed_layers = list(derived_products['results_df']['Layer'])
                    selected_layer_for_secchi = st.selectbox("Select layer to use for all Kd calculations:", analyzed_layers, key="secchi_layer_sel")
    
                    if selected_layer_for_secchi:
                        layer_num = int(selected_layer_for_secchi.split(' ')[1])
                        z_min, z_max = current_station_layers[layer_num]['range']
                        kd_490_profiler = derived_products['results_df'].loc[derived_products['results_df']['Layer'] == selected_layer_for_secchi, 'Kd(490)'].iloc[0]
                        kd_par_profiler = derived_products['kd_par_df'].loc[derived_products['kd_par_df']['Layer'] == selected_layer_for_secchi, 'Kd(PAR)'].iloc[0]
                        kd_par_external = calculate_external_kd_par(df_secchi_source, ext_depth_col_secchi, ext_par_col_secchi, z_min, z_max)
    
                        st.info(f"""
                        You are about to save the following data for **'{profiler_station_name}'**:
                        - **Secchi Depth:** `{secchi_value:.2f} m` (from external file)
                        - **Kd(PAR) (Profiler):** `{kd_par_profiler:.4f} m⁻¹` (from {selected_layer_for_secchi})
                        - **Kd(490) (Profiler):** `{kd_490_profiler:.4f} m⁻¹` (from {selected_layer_for_secchi})
                        - **Kd(PAR) (External Probe):** `{kd_par_external:.4f} m⁻¹` (calculated from {selected_layer_for_secchi})
                        """)
                    
                    if st.button(f"Add/Update Data for '{profiler_station_name}'", use_container_width=True):
                        if not pd.isna(secchi_value):
                            new_data_row = pd.DataFrame({
                                'Station_ID': [profiler_station_name], 'Secchi': [secchi_value],
                                'Kd(PAR)_Profiler': [kd_par_profiler], 'Kd(490)_Profiler': [kd_490_profiler],
                                'Kd(PAR)_External': [kd_par_external]
                            })
                            master_df = st.session_state.master_kd_secchi_df
                            updated_df = pd.concat([master_df, new_data_row], ignore_index=True)
                            updated_df.drop_duplicates(subset=['Station_ID'], keep='last', inplace=True)
                            st.session_state.master_kd_secchi_df = updated_df
                            st.toast(f"Data for '{profiler_station_name}' saved!")
                            st.rerun() 
                        else: 
                            st.error(f"The Secchi value for '{selected_external_station}' is not a valid number.")
                else:
                    st.warning("No profiler Kd values have been calculated. Please analyze at least one layer in the sidebar.")
        
        st.divider()
        st.subheader("2. Master Data and Visualization")
        plot_kd_secchi_relationship(st.session_state.master_kd_secchi_df)
        
        with st.expander("Show/Hide Master Data Table"):
            if not st.session_state.master_kd_secchi_df.empty:
                df_display = st.session_state.master_kd_secchi_df.rename(columns={
                    'Kd(PAR)_Profiler': 'Kd(PAR) (Profiler)',
                    'Kd(490)_Profiler': 'Kd(490) (Profiler)',
                    'Kd(PAR)_External': 'Kd(PAR) (External)'
                })
                st.dataframe(df_display, use_container_width=True)
                
                st.download_button(
                    label="Download Updated Master Data (CSV)",
                    data=st.session_state.master_kd_secchi_df.to_csv(index=False).encode('utf-8'),
                    file_name="master_secchi_kd_data.csv",
                    mime='text/csv',
                    key="download_secchi_data"
                )
            else:
                st.dataframe(pd.DataFrame(columns=[
                    'Station_ID', 'Secchi', 'Kd(PAR) (Profiler)', 'Kd(490) (Profiler)', 'Kd(PAR) (External)'
                ]), use_container_width=True)
    
    # ===================================================================
    # FINAL UPGRADED TAB: SPECTRAL CONVOLUTION (WITH MASTER RRS & STEP PLOT)
    # ===================================================================
    with tab_convolution:
        st.header("Spectral Convolution for Satellite Matching")
        st.info("Use the tools below to convolve your hyperspectral Rrs and compare it against historical data or satellite bands.")
    
        #  --- PART 1: MASTER RRS UPLOADER (MOVED HERE) ---
        with st.expander("Load and Visualize Master (Historical) Rrs File"):
            st.file_uploader(
                "Upload Master Rrs Data (CSV)",
                type="csv",
                key="master_rrs_loader",
                accept_multiple_files=False, # Only one master file at a time
                on_change=_handle_master_rrs_upload
            )
        st.divider()
    
        # --- PART 2: CONVOLUTION TOOLS ---
        st.subheader("Perform Spectral Convolution")
        srf_data = load_srf_from_folder()
        derived_products = st.session_state.get('derived_products')
    
        if derived_products is None or 'rrs_df_export' not in derived_products or derived_products['rrs_df_export'].empty:
            st.warning("Please analyze at least one layer in the sidebar to generate Rrs data before you can perform a convolution.")
        
        elif srf_data:
            srf_options = list(srf_data.keys())
            
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_sensor = st.selectbox("Select Sensor:", srf_options)
            
            df_srf = srf_data[selected_sensor]
            srf_wl_col = next((col for col in df_srf.columns if 'wave' in col.lower()), None)
            if srf_wl_col is None:
                st.error(f"Could not find a 'Wavelength' column in the SRF file for '{selected_sensor}'."); st.stop()
            
            srf_bands = [col for col in df_srf.columns if col != srf_wl_col]
            
            with c2:
                rrs_options = [col for col in derived_products['rrs_df_export'].columns if 'rrs' in col]
                selected_rrs_source = st.selectbox("Select Rrs to convolve:", rrs_options)
            with c3:
                selected_bands = st.multiselect("Select satellite bands:", srf_bands, default=srf_bands[:8])
    
            if st.button("Perform Convolution", use_container_width=True):
                # ... (Calculation logic remains the same) ...
                with st.spinner("Convolving spectra..."):
                    ed_hyperspectral = st.session_state.get('new_es_median', station_data.get('Es'))
                    if ed_hyperspectral is None:
                        st.error("Cannot perform convolution: No valid Es (surface irradiance) data available."); st.stop()
                    
                    rrs_df = derived_products['rrs_df_export']
                    lw_hyperspectral = rrs_df[selected_rrs_source] * ed_hyperspectral
                    prof_wl, srf_wl = station_data['wavelengths'], df_srf[srf_wl_col].values
                    
                    convolution_results = []
                    for band in selected_bands:
                        srf_response = df_srf[band].values
                        convolved_rrs = perform_convolution(prof_wl, lw_hyperspectral, ed_hyperspectral, srf_wl, srf_response)
                        convolution_results.append({'Band': band, 'Convolved_Rrs_sr-1': convolved_rrs})
                    
                    st.session_state.convolved_rrs_df = pd.DataFrame(convolution_results)
                    st.session_state.convolved_sensor_name = selected_sensor
                st.rerun()
    
        # --- PART 3: RESULTS VISUALIZATION (UPGRADED PLOT) ---
        if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
            st.markdown("---")
            st.subheader(f"Convolution Results for {st.session_state.get('convolved_sensor_name', '')}")
            df_results = st.session_state.convolved_rrs_df
            
            res_c1, res_c2 = st.columns([1, 2])
            with res_c1:
                st.dataframe(df_results.style.format({'Convolved_Rrs_sr-1': '{:.5f}'}))
            
            with res_c2:
                fig_conv, ax = plt.subplots()

                # --- Plot Historical and Active Rrs (no changes here) ---
                if 'master_rrs_df' in st.session_state and st.session_state.master_rrs_df is not None:
                    master_df = st.session_state.master_rrs_df
                    master_wl_col = master_df.columns[0]
                    for col in master_df.columns[1:]:
                        ax.plot(master_df[master_wl_col], master_df[col], color='grey', linestyle=':', alpha=0.75, lw=1)
                    ax.plot([], [], color='grey', linestyle=':', alpha=0.95, label='Historical Rrs')
                ax.plot(station_data['wavelengths'], derived_products['rrs_df_export'][selected_rrs_source], color='white', alpha=0.8, lw=1.5, label='Active Hyperspectral Rrs')
                
                #  --- START OF NEW, ROBUST STEP PLOT LOGIC ---
                df_srf_display = srf_data[st.session_state.get('convolved_sensor_name')]
                srf_wl_col_display = next((col for col in df_srf_display.columns if 'wave' in col.lower()), None)
                
                band_centers = []
                # Calculate the central wavelength for each band to use as the x-coordinate
                for band in df_results['Band']:
                    srf_curve = df_srf_display[band]
                    center = np.sum(df_srf_display[srf_wl_col_display] * srf_curve) / np.sum(srf_curve)
                    band_centers.append(center)
                
                # Plot the convolved Rrs as a line connecting markers
                ax.plot(band_centers, df_results['Convolved_Rrs_sr-1'], 
                        marker='o',          # Add circular markers at each point
                        color='red', 
                        lw=2.0, 
                        alpha=0.37,
                        label='Convolved Rrs (Multispectral)')
                    # --- END OF NEW STEP PLOT LOGIC ---
                    
                ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Rrs (sr⁻¹)"); ax.set_title("Convolved vs. Hyperspectral Rrs"); ax.grid(True, linestyle=':'); ax.legend(fontsize=8)
                ax.set_ylim(0.0, 0.02)
                ax.set_xlim(400, 700)
                st.pyplot(fig_conv)
                
    
    # ===================================================================
    # TAB 7: GENERATE REPORT & L3 DATA
    # ===================================================================
    with tab_report:
        st.header("Generate L3 Data Product")
        st.info("Use this tab to download the final L3 data products. The downloaded .zip file will contain all relevant data, metrics, plots, and a summary report.")
        
        if st.session_state.comparison_table is not None:
            st.subheader("Download L3 Product for a Single Profile")
            try:
                options_list = st.session_state.comparison_table.index.tolist()
                default_index = options_list.index(st.session_state.selected_station)
            except (ValueError, AttributeError):
                default_index = 0
            selected_cast_for_download = st.selectbox(
                "Select a single, validated cast to download as an L3 product:",
                options=st.session_state.comparison_table.index, 
                index=default_index,
                key="single_download_select"
            )
            
            layers_for_selected_cast = st.session_state.profile_specific_layers.get(selected_cast_for_download, {})
            disable_single_download = not layers_for_selected_cast
    
            if disable_single_download:
                st.warning(f"Please analyze at least one layer for '{selected_cast_for_download}' in the sidebar to generate its L3 products.")
            
            if st.button(f"Generate L3 Package for {selected_cast_for_download}", disabled=disable_single_download):
                base_name = f"L3_{selected_cast_for_download}"
                rep_data = st.session_state.station_data[selected_cast_for_download]
                
                derived_products_for_report = calculate_derived_products(
                    rep_data, selected_cast_for_download, layers_for_selected_cast
                )
                
                zip_buffer_rep = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_rep, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    create_full_report_zip(zip_file, base_name)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        lu_ed_ratio = np.where(rep_data['Ed_data'] > 0, rep_data['Lu_data'] / rep_data['Ed_data'], np.nan)
                    wvl_cols = [f"{int(w)}nm" for w in rep_data['wavelengths']]
                    lu_ed_df = pd.DataFrame(lu_ed_ratio, columns=wvl_cols); lu_ed_df.insert(0, 'Depth', rep_data['pressure'])
                    zip_file.writestr(f"{base_name}/L3_Data/in_water_Lu_Ed_ratio.csv", lu_ed_df.to_csv(index=False))
                    
                    zip_file.writestr(f"{base_name}/L3_Data/Rrs_propagated.csv", derived_products_for_report['rrs_df_export'].to_csv(index=False))
                    zip_file.writestr(f"{base_name}/L3_Data/K_metrics.csv", derived_products_for_report['k_df_export'].to_csv(index=False))
                    
                    mixture_results_for_report = None
                    if st.session_state.get('wm_vertices'):
                        l3_phys_props = calculate_physical_properties(rep_data, st.session_state.lon, st.session_state.lat)
                        l3_mix_results_text, l3_mix_df = analyze_mixture(l3_phys_props['Salinity'], l3_phys_props['Temperature'], st.session_state.wm_vertices, st.session_state.wm_names)
                        mixture_results_for_report = l3_mix_results_text
                        mixture_profile_df = pd.concat([l3_phys_props['Depth'], l3_mix_df], axis=1).dropna()
                        zip_file.writestr(f"{base_name}/L3_Data/TS_Mixture_Profile_Data.csv", mixture_profile_df.to_csv(index=False))
    
                    report_text = generate_summary_report(
                        selected_cast_for_download, 
                        layers_for_selected_cast, 
                        derived_products_for_report['results_df'], 
                        derived_products_for_report['kd_par_df'], 
                        mixture_results_for_report,
                        empirical_model=st.session_state.get('empirical_model_params'),
                        external_probe_filename=st.session_state.get('external_probe_filename'),
                        comparison_kd_df=st.session_state.get('comparison_kd_df')
                    )
                    zip_file.writestr(f"{base_name}/summary_report.txt", report_text)
                
                st.download_button(
                    label=f"Download L3 Package: {selected_cast_for_download}",
                    data=zip_buffer_rep.getvalue(), file_name=f"{base_name}.zip", 
                    mime="application/zip", key=f"download_{selected_cast_for_download}"
                )
    
            st.divider()
            st.subheader("Generate L3 Ensemble Average Product")
            ensemble_casts = st.multiselect("Select validated profiles to include in ensemble average:", options=st.session_state.comparison_table.index, default=list(st.session_state.comparison_table.index))
            
            any_layers_defined = any(st.session_state.profile_specific_layers.get(cast) for cast in ensemble_casts)
            disable_ensemble = (len(ensemble_casts) < 2) or not any_layers_defined
            
            if len(ensemble_casts) < 2:
                st.info("Select at least two profiles above to generate an ensemble average.")
            elif not any_layers_defined:
                st.warning("Please analyze at least one layer for one of the selected profiles to generate an ensemble product.")
            
            if st.button("Generate & Download Ensemble Average Package", disabled=disable_ensemble):
                with st.spinner("Calculating ensemble average and building package..."):
                    base_name = f"L3_Ensemble_Average_{len(ensemble_casts)}_profiles"
                    zip_buffer_ens = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer_ens, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        create_full_report_zip(zip_file, base_name)
                        
                        min_depth_ens, max_depth_ens = float('inf'), float('-inf')
                        for cast_id in ensemble_casts:
                            pressure = st.session_state.station_data[cast_id]['pressure']
                            if pressure.size > 0:
                                min_depth_ens = min(min_depth_ens, pressure.min())
                                max_depth_ens = max(max_depth_ens, pressure.max())
                        
                        ensemble_depth_grid = np.arange(np.ceil(min_depth_ens*10)/10, np.floor(max_depth_ens*10)/10 + 0.1, 0.1)
                        interp_ed_list, interp_lu_list = [], []
                        wavelengths = st.session_state.station_data[ensemble_casts[0]]['wavelengths']
                        
                        for cast_id in ensemble_casts:
                            cast_data = st.session_state.station_data[cast_id]
                            interp_ed = np.array([np.interp(ensemble_depth_grid, cast_data['pressure'], cast_data['Ed_data'][:, i]) for i in range(len(wavelengths))]).T
                            interp_lu = np.array([np.interp(ensemble_depth_grid, cast_data['pressure'], cast_data['Lu_data'][:, i]) for i in range(len(wavelengths))]).T
                            interp_ed_list.append(interp_ed)
                            interp_lu_list.append(interp_lu)
                        
                        ed_stack, lu_stack = np.stack(interp_ed_list, axis=0), np.stack(interp_lu_list, axis=0)
                        ed_mean, ed_std = np.mean(ed_stack, axis=0), np.std(ed_stack, axis=0)
                        lu_mean, lu_std = np.mean(lu_stack, axis=0), np.std(lu_stack, axis=0)
                        wvl_cols = [f"{int(w)}nm" for w in wavelengths]
                        
                        df_ed = pd.DataFrame(columns=['Depth'] + [f'{c}_{s}' for c in wvl_cols for s in ['mean', 'std']]); df_ed['Depth'] = ensemble_depth_grid; df_ed[[f'{c}_mean' for c in wvl_cols]] = ed_mean; df_ed[[f'{c}_std' for c in wvl_cols]] = ed_std
                        df_lu = pd.DataFrame(columns=['Depth'] + [f'{c}_{s}' for c in wvl_cols for s in ['mean', 'std']]); df_lu['Depth'] = ensemble_depth_grid; df_lu[[f'{c}_mean' for c in wvl_cols]] = lu_mean; df_lu[[f'{c}_std' for c in wvl_cols]] = lu_std
                        with np.errstate(divide='ignore', invalid='ignore'): lu_ed_ratio_mean = np.where(ed_mean > 0, lu_mean / ed_mean, np.nan)
                        df_lu_ed = pd.DataFrame(lu_ed_ratio_mean, columns=wvl_cols); df_lu_ed.insert(0, 'Depth', ensemble_depth_grid)
                        
                        k_list, rrs_list, all_results_dfs, all_kd_par_dfs = [], [], [], []
                        processed_casts = [] 
    
                        for cast_id in ensemble_casts:
                            cast_layers = st.session_state.profile_specific_layers.get(cast_id, {})
                            if not cast_layers:
                                st.warning(f"Skipping '{cast_id}' in ensemble derived products: No analysis layers defined.")
                                continue
                            
                            derived_products_loop = calculate_derived_products(
                                st.session_state.station_data[cast_id], cast_id, cast_layers
                            )
                            
                            if derived_products_loop:
                                rrs_list.append(derived_products_loop['rrs_df_export'].set_index('wavelength_nm'))
                                k_list.append(derived_products_loop['k_df_export'].set_index('wavelength_nm'))
                                all_results_dfs.append(derived_products_loop['results_df'])
                                all_kd_par_dfs.append(derived_products_loop['kd_par_df'])
                                processed_casts.append(cast_id)
                                
                        ensemble_results_df = pd.concat(all_results_dfs).groupby('Layer').mean().reset_index()
                        ensemble_kd_par_df = pd.concat(all_kd_par_dfs).groupby('Layer').mean().reset_index()
                        df_k_ens = pd.concat(k_list).groupby(level=0).agg(['mean', 'std']); df_k_ens.columns = ['_'.join(col) for col in df_k_ens.columns]; df_k_ens.reset_index(inplace=True)
                        df_rrs_ens = pd.concat(rrs_list).groupby(level=0).agg(['mean', 'std']); df_rrs_ens.columns = ['_'.join(col) for col in df_rrs_ens.columns]; df_rrs_ens.reset_index(inplace=True)
    
                        zip_file.writestr(f"{base_name}/L3_Data_Averaged/Ed_averaged.csv", df_ed.to_csv(index=False))
                        zip_file.writestr(f"{base_name}/L3_Data_Averaged/Lu_averaged.csv", df_lu.to_csv(index=False))
                        zip_file.writestr(f"{base_name}/L3_Data_Averaged/in_water_Lu_Ed_ratio_averaged.csv", df_lu_ed.to_csv(index=False))
                        zip_file.writestr(f"{base_name}/L3_Data_Averaged/K_metrics_averaged.csv", df_k_ens.to_csv(index=False))
                        zip_file.writestr(f"{base_name}/L3_Data_Averaged/Rrs_propagated_averaged.csv", df_rrs_ens.to_csv(index=False))
                        
                        ensemble_mixture_results = {}
                        if st.session_state.get('wm_vertices'):
                            for cast_id in processed_casts:
                                ens_cast_data = st.session_state.station_data[cast_id]
                                ens_phys_props = calculate_physical_properties(ens_cast_data, st.session_state.lon, st.session_state.lat)
                                ens_mix_text, ens_mix_df = analyze_mixture(ens_phys_props['Salinity'], ens_phys_props['Temperature'], st.session_state.wm_vertices, st.session_state.wm_names)
                                ensemble_mixture_results[cast_id] = ens_mix_text
                                ens_mix_profile_df = pd.concat([ens_phys_props['Depth'], ens_mix_df], axis=1).dropna()
                                zip_file.writestr(f"{base_name}/L3_Data_Individual_Mixture_Profiles/{cast_id}_mixture_data.csv", ens_mix_profile_df.to_csv(index=False))
    
                        report_text_ens = generate_summary_report(
                            processed_casts, 
                            st.session_state.profile_specific_layers, 
                            ensemble_results_df, 
                            ensemble_kd_par_df, 
                            ensemble_mixture_results,
                            empirical_model=st.session_state.get('empirical_model_params'),
                            external_probe_filename=st.session_state.get('external_probe_filename'),
                            comparison_kd_df=st.session_state.get('comparison_kd_df')
                        )
                        zip_file.writestr(f"{base_name}/summary_report.txt", report_text_ens)
                    
                    st.download_button(label="Download Ensemble Package", data=zip_buffer_ens.getvalue(), file_name=f"{base_name}.zip", mime="application/zip", key="download_ensemble")
        
        else:
            st.info("Run a CTD Comparison in the first tab to enable report generation.")
    # ===================================================================
    # TAB 8: FORMULAS & METHODS
    # ===================================================================
    with tab_formulas:
        st.header("Formulas & Methods"); st.subheader("1. Apparent Optical Properties (Kd & klu)")
        st.markdown(r"""The diffuse attenuation coefficients for downwelling irradiance ($K_d$) and upwelling radiance ($K_u$) describe how rapidly light is attenuated with depth. They are derived from the Beer-Lambert Law, which states:$$ I(z) = I(0) \cdot e^{-K \cdot z} $$Where $I(z)$ is the intensity at depth $z$, and $I(0)$ is the intensity at the surface. By taking the natural logarithm, this becomes a linear relationship:$$ \ln(I(z)) = \ln(I(0)) - K \cdot z $$This app calculates $K_d$ and $K_u$ for each wavelength ($\lambda$) within a user-selected depth layer by performing a linear regression (using 'np.polyfit') on the natural log of the radiometric data versus depth. The coefficient $K$ is the negative of the slope of this regression. **Reference:** Mobley, C. D. (1994). *Light and Water: Radiative Transfer in Natural Waters*. Academic Press.""")
        st.subheader("2. Remote Sensing Reflectance ($R_{rs}$)"); st.markdown(r"""Remote sensing reflectance is the ratio of water-leaving radiance ($L_w$) to the downwelling irradiance ($E_d$) just above the surface:$$ R_{rs}(\lambda) = \frac{L_w(\lambda)}{E_d(\lambda, 0^+)} $$The water-leaving radiance ($L_w$) is estimated from the upwelling radiance just below the surface ($L_u(\lambda, 0^-)$) by accounting for the transmission of light across the air-sea interface. This app uses a standard approximation where the transmission factor is '{LW_TRANSMISSION_FACTOR}':$$ L_w \approx {LW_TRANSMISSION_FACTOR} \cdot L_u(0^-) $$This coefficient approximates the term $\frac{{1- \rho(\theta, n)}}{{n^2}}$, where $\rho$ is the Fresnel reflectance and $n$ is the refractive index of water. **Propagated $R_{rs}$** is calculated by using the layer-specific $K_d$ and $K_u$ to propagate the radiometric values from the top of the selected layer ($z$) back to the surface:$$ E_d(0^+) = E_d(z) \cdot e^{{K_d \cdot z}} $$$$ L_u(0^-) = L_u(z) \cdot e^{{K_u \cdot z}} $$These surface-equivalent values are then used to calculate the final propagated $R_{rs}$.""")
        st.subheader("3. Stratification (Brunt-Väisälä Frequency)"); st.markdown(r"""The stratification of the water column is represented by the Brunt-Väisälä frequency squared, $N^2$. A high $N^2$ value indicates strong stratification. This value is calculated using the Gibbs SeaWater (GSW) Oceanographic Toolbox, which implements the Thermodynamic Equation of Seawater 2010 (TEOS-10). **Reference:** McDougall, T. J., & Barker, P. M. (2011). *Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox*. SCOR/IAPSO Working Group 127, ISBN 978-0-646-55606-5.""")
        st.subheader("4. Photosynthetically Available Radiation (PAR & Kd(PAR))"); st.markdown(r"""Photosynthetically Available Radiation (PAR) is the flux of photons between 400 and 700 nm. It is calculated by converting the energy of the downwelling irradiance $E_d(\lambda)$ from energy units ($\mu W/cm^2/nm$) to quanta units ($\mu mol\ photons/m^2/s/nm$) and then integrating over the wavelength range. This app follows the protocol from the ProSoft manual:$$ PAR(z) = \int_{{400nm}}^{{700nm}} E_q(\lambda, z) d\lambda $$Where $E_q$ is the spectral irradiance in quanta units. The attenuation coefficient for PAR, $K_d(PAR)$, is then calculated for each user-defined layer by performing a log-linear regression on the PAR profile versus depth, identical to the method used for spectral $K_d(\lambda)$.""")

