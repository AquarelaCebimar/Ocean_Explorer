
# -*- coding: utf-8 -*-
"""
# Ocean_Explorer V1.beta

Author: Ana Paula Piazza Forgiarini (anapiazzaf@gmail.com)
Created on Fri Oct 10 10:46:23 2025
Modified in April 2026

# How to cite:
This software is an integral part of the ongoing research of Ana Paula Piazza Forgiarini at Aquarela Lab CEBIMar/USP.
All rights reserved. This code is shared for collaborative work within the research group. Please contact the author for formal permission and 
specific citation guidelines before using this software, its underlying logic, or generated outputs in any presentation, report, or publication.

"""

# ----------------------------- REQUIRED PACKAGES -----------------------------
# Import section for all Python libraries required for the application to function.

import streamlit as st  # Imports the Streamlit library to create the interactive web interface.
import pandas as pd  # Imports Pandas for data manipulation and analysis, especially DataFrames.
import numpy as np  # Imports NumPy for efficient numerical operations, especially with arrays.
import matplotlib.pyplot as plt  # Imports Matplotlib for creating static plots.
from matplotlib.patches import Polygon  # Imports the Polygon class to draw the mixing triangle on the T-S diagram.
import gsw  # Imports the Gibbs SeaWater (GSW) library for precise seawater property calculations (TEOS-10).
import seaborn as sns  # Imports Seaborn to create more aesthetically pleasing color palettes for the plots.
import io  # Imports the io module to handle in-memory byte buffers, used for creating zip files.
import zipfile  # Imports the zipfile module to dynamically create .zip archives.
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Imports regression metrics to compare CTD data.
from matplotlib.ticker import ScalarFormatter # Imports ScalarFormatter for scientific notation on axes.
from scipy.stats import linregress # Imports the linear regression function for the new plots.
import plotly.graph_objects as go  # Imports Plotly for interactive plots.
from scipy.integrate import simpson # Imports Simpson integral for spectral convolution


# ----------------------------- STYLE AND CONFIGURATIONS -----------------------------
# Section to define the visual appearance of the plots and the Streamlit page.

st.set_page_config(layout="wide")

# Dictionary centralizing color definitions to create a consistent visual theme (dark mode).

STYLE_CONFIG = {
    'facecolor':'#0E1117',
    'ax_facecolor':'#2C3445',
    'textcolor':'#FAFAFA',
    'gridcolor':'#5A5A5A',
    'spinecolor':'#8A8A8A',
    'n2_line_color':'#00FF00',
    'beta_line_color': '#FF8C00'
}

# Update Matplotlib's global parameters to match the dark mode theme.

plt.rcParams.update({
    'figure.facecolor': STYLE_CONFIG['facecolor'],
    'axes.facecolor': STYLE_CONFIG['ax_facecolor'],
    'axes.edgecolor': STYLE_CONFIG['spinecolor'],
    'text.color': STYLE_CONFIG['textcolor'],
    'axes.labelcolor': STYLE_CONFIG['textcolor'],
    'xtick.color': STYLE_CONFIG['textcolor'],
    'ytick.color': STYLE_CONFIG['textcolor'],
    'grid.color': STYLE_CONFIG['gridcolor']
})

# Streamlit page configuration (must be called once at the top of the script).

st.markdown("""
<style>
/* 1. Força todos os wrappers a permitirem scroll */
div[data-testid="stTabs"],
div[data-testid="stTabs"] > div,
div[data-testid="stTabs"] > div > div {
    overflow: visible !important;
}

/* 2. Container real das abas - Estilização da Fita */
div[data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: nowrap !important;
    gap: 12px !important;
    padding-bottom: 10px !important;
    scrollbar-width: auto !important;
}

/* 3. Botões das abas - Impede encolhimento e define fonte */
div[data-testid="stTabs"] button[role="tab"] {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
    border-radius: 5px 5px 0 0 !important;
}

div[data-testid="stTabs"] button[role="tab"] p {
    font-size: 16px !important; /* Tamanho médio legível */
    font-weight: bold !important;
}

/* 4. Scrollbar visível e colorida */
div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
    height: 8px;
}
div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg, #A78BFA, #FF4B4B);
    border-radius: 5px;
}

/* 5. PALETA ESPECTRAL (Target nos parágrafos dentro de cada botão) */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(1) p { color: #A78BFA !important; } /* Violet */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(2) p { color: #818CF8 !important; } /* Indigo */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(3) p { color: #60A5FA !important; } /* Blue */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(4) p { color: #22D3EE !important; } /* Cyan */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(5) p { color: #34D399 !important; } /* Green */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(6) p { color: #A3E635 !important; } /* Lime */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(7) p { color: #FACC15 !important; } /* Yellow */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(8) p { color: #FB923C !important; } /* Orange */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(9) p { color: #F87171 !important; } /* Red */
div[data-testid="stTabs"] button[role="tab"]:nth-of-type(10) p { color: #FF4B4B !important; } /* Deep Red */

/* 6. Destaque da Aba Ativa (Linha Branca) */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-bottom: 2px solid white !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------- CONSTANTS -----------------------------
# Define core column names and standard values used throughout the application.

PRESSURE_COL, TEMP_COL, COND_COL, WAVELENGTH_PREFIX, STATION_ID_COL = 'Depth', 'Temperature', 'Conductivity', 'X', 'station_id'
FILE_KEYWORDS = {'temp':'temp', 'cond':'cond', 'ed':'ed', 'lu':'lu', 'beta': 'beta'}
LW_TRANSMISSION_FACTOR = 0.54
ED_TRANSMISSION_FACTOR = 0.96

# ----------------------------- PROCESSING AND CALCULATION FUNCTIONS -----------------------------

@st.cache_data
def process_uploaded_files(profiler_files, es_file=None, beta_file=None):
    """
    Processes the uploaded in-water profiler files, Es (surface irradiance), and Beta (backscattering proxy) files.
    Matches keywords to identify each file type, cleans the data, interpolates to a common depth grid (if not already done), 
    and packages everything into a dictionary mapped by station_id.
    """
    
    file_map = {}
    for f in profiler_files:
        fname_lower = f.name.lower()
        for key, keyword in FILE_KEYWORDS.items():
            if keyword in fname_lower:
                file_map[key] = io.StringIO(f.getvalue().decode("utf-8"))
    
    if beta_file:
        file_map['beta'] = io.StringIO(beta_file.getvalue().decode("utf-8"))
            
    if len(file_map) < 4:
        st.error(f"Upload failed. Expected at least 4 in-water files (temp, cond, ed, lu), found {len(file_map)}.")
        return None, None

    # Common representations of missing data to be parsed as NaNs.

    possible_NA_values = ['#########', 'NA', 'N/A', 'Not a Number', 'missing', '-9999']
    try:
        df_temp_raw = pd.read_csv(file_map['temp'], na_values=possible_NA_values)
        df_cond_raw = pd.read_csv(file_map['cond'], na_values=possible_NA_values)
        df_ed_raw = pd.read_csv(file_map['ed'], na_values=possible_NA_values)
        df_lu_raw = pd.read_csv(file_map['lu'], na_values=possible_NA_values)
    except Exception as e:
        st.error(f"Error reading core CSV files. Please check file format. Details: {e}")
        return None, None

    df_beta_raw = None
    if 'beta' in file_map:
        df_beta_raw = pd.read_csv(file_map['beta'], na_values=possible_NA_values)

    es_median_map = {}; es_full_data_map = {}
    if es_file:
        df_es_raw = pd.read_csv(es_file, na_values=possible_NA_values)
        wave_cols_es = [col for col in df_es_raw.columns if col.startswith(WAVELENGTH_PREFIX)]
        for station_id, group in df_es_raw.groupby(STATION_ID_COL):
            es_median_map[station_id] = group[wave_cols_es].median().values
            es_full_data_map[station_id] = group[wave_cols_es]

    station_ids_from_temp = df_temp_raw[STATION_ID_COL].unique()
    station_data_package = {}; successful_stations = []

    for station in station_ids_from_temp:
        df_temp = df_temp_raw[df_temp_raw[STATION_ID_COL] == station]
        df_cond = df_cond_raw[df_cond_raw[STATION_ID_COL] == station]
        df_ed = df_ed_raw[df_ed_raw[STATION_ID_COL] == station]
        df_lu = df_lu_raw[df_lu_raw[STATION_ID_COL] == station]
        
        if any(df.empty for df in [df_temp, df_cond, df_ed, df_lu]):
            st.warning(f"Skipping station '{station}' due to missing core files."); continue

        wave_cols = [col for col in df_ed.columns if col.startswith(WAVELENGTH_PREFIX)]
        wavelengths_temp = np.array([float(c.replace(WAVELENGTH_PREFIX, '')) for c in wave_cols])
        
        # Identify the column closest to 490nm to use as a baseline for valid optical data.

        idx_490 = (np.abs(wavelengths_temp - 490)).argmin()
        canary_col = wave_cols[idx_490]
        
        # Clean optical data based on the canary wavelength (must exist and be > 0).

        df_ed_clean = df_ed.dropna(subset=[canary_col])
        df_ed_clean = df_ed_clean[df_ed_clean[canary_col] > 0]
        
        if df_ed_clean.empty: 
            st.warning(f"Skipping station '{station}' due to no valid Ed(490) data."); continue
        
        optical_min_depth = df_ed_clean[PRESSURE_COL].min()
        optical_max_depth = df_ed_clean[PRESSURE_COL].max()
        
        # Create a master depth grid based on temperature depths that fall within the optical depth range.

        all_temp_depths = np.sort(df_temp[PRESSURE_COL].unique())
        master_pressure = all_temp_depths[(all_temp_depths >= optical_min_depth) & (all_temp_depths <= optical_max_depth)]
        
        if master_pressure.size < 2: continue
    
        df_temp_clean = df_temp.dropna(subset=[TEMP_COL])
        df_cond_clean = df_cond.dropna(subset=[COND_COL])

        # Gap evaluation. Calculates the maximum difference between valid depths to detect missing data blocks
        
        if not df_temp_clean.empty:
            temp_depths_sorted = df_temp_clean[PRESSURE_COL].sort_values()
            max_temp_gap = temp_depths_sorted.diff().max()

            # If the largest gap is greater than 2 meters, emit a warning.
            if pd.notna(max_temp_gap) and max_temp_gap > 2:
                st.warning(
                    f"**Caution for station '{station}':** Missing physical data detected. "
                    f"Maximum depth gap is **{max_temp_gap:.1f} meters**. "
                    f"Data has been linearly interpolated to maintain grid consistency, but physical parameters (like N²) may be distorted in this interval."
                )

        # If no data is left (or fewer than 2 points), interpolation is impossible.
        if df_temp_clean.empty or len(df_temp_clean) < 2:
            st.warning(f"Skipping station '{station}': Not enough Temperature data (too many NaNs).")
            continue
            
        if df_cond_clean.empty or len(df_cond_clean) < 2:
            st.warning(f"Skipping station '{station}': Not enough Conductivity data.")
            continue

        # Safe interpolation mapped to the master optical depth grid
        temp_data = np.interp(master_pressure, df_temp_clean[PRESSURE_COL], df_temp_clean[TEMP_COL])
        cond_data = np.interp(master_pressure, df_cond_clean[PRESSURE_COL], df_cond_clean[COND_COL])

        wavelengths = np.array([float(c.replace(WAVELENGTH_PREFIX, '')) for c in wave_cols])
        
        # Group, reindex, and interpolate Ed data.

        df_ed_grouped = df_ed[[PRESSURE_COL] + wave_cols].groupby(PRESSURE_COL).mean()
        ed_df_interp = df_ed_grouped.reindex(df_ed_grouped.index.union(master_pressure)).interpolate(method='index').loc[master_pressure]
        ed_data = ed_df_interp.values

        # Group, reindex, and interpolate Lu data.

        df_lu_grouped = df_lu[[PRESSURE_COL] + wave_cols].groupby(PRESSURE_COL).mean()
        lu_df_interp = df_lu_grouped.reindex(df_lu_grouped.index.union(master_pressure)).interpolate(method='index').loc[master_pressure]
        lu_data = lu_df_interp.values
        
        beta_data_interp = None
        beta_cols = []
        df_beta = None
        if df_beta_raw is not None:
            df_beta = df_beta_raw[df_beta_raw[STATION_ID_COL] == station]
        
        if df_beta is not None and not df_beta.empty:
            beta_cols = [col for col in df_beta.columns if col not in [PRESSURE_COL, STATION_ID_COL]]
            df_beta_clean = df_beta.dropna(subset=[PRESSURE_COL] + beta_cols, how='any')
            
            if not df_beta_clean.empty:
                # Interpolate beta data onto the master pressure grid

                df_beta_indexed = df_beta_clean.set_index(PRESSURE_COL)
                combined_index = df_beta_indexed.index.union(master_pressure)
                df_beta_aligned = df_beta_indexed.reindex(combined_index).interpolate(method='index')
                beta_data_interp = df_beta_aligned.loc[master_pressure].dropna(how='all')
                if beta_data_interp.empty:
                    st.warning(f"For station '{station}', beta data was found but had no overlapping depth range with the main profile.")
            else:
                 st.warning(f"For station '{station}', the uploaded beta file contained no valid numeric data rows.")
                 
        # Package the successfully processed data for this station.

        station_data_package[station] = {
            'pressure': master_pressure, 'temperature': temp_data, 'conductivity': cond_data,
            'Ed_data': ed_data, 'Lu_data': lu_data, 'wavelengths': wavelengths, 
            'Es': es_median_map.get(station, None), 'Es_all_spectra': es_full_data_map.get(station, None),
            'beta_data': beta_data_interp, 'beta_cols': beta_cols
        }
        successful_stations.append(station)
        
    return station_data_package, successful_stations


def analyze_mixture(SA, CT, vertices, names):
    """
   Calculates water mass mixing fractions using barycentric coordinates 
   based on a defined T-S (Temperature-Salinity) mixing triangle.
   """
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
        
        # Only accept points that fall strictly inside (or on the edges of) the triangle

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
    Performs spectral convolution on Lw and Ed independently before division
    to simulate multi-spectral satellite bands (e.g., MODIS, Sentinel-3).
    
    Methodology follows Burggraaff (2020) to avoid convolution biases: 
    Reflectance (Rrs) is NOT convolved directly. Instead, the numerator (Lw) 
    and denominator (Ed) are convolved independently using Simpson's rule.
    """
    
    # Step 1: Interpolate the satellite's SRF onto the profiler's wavelength grid.
    srf_interp = np.interp(prof_wl, srf_wl, srf_response, left=0, right=0)
    
    # Step 2: Define the integrands
    lw_integrand = lw_hyperspectral * srf_interp
    ed_integrand = ed_hyperspectral * srf_interp
    
    # Step 3: Perform numerical integration using Simpson's Rule
    # This accounts for potentially non-uniform wavelength spacing (dx)
    lw_integrated = simpson(y=lw_integrand, x=prof_wl)
    ed_integrated = simpson(y=ed_integrand, x=prof_wl)
    
    # Step 4: Calculate the final band-specific Rrs. Avoid division by zero.
    if ed_integrated > 0:
        return lw_integrated / ed_integrated
    else:
        return np.nan

def calculate_derived_products(station_data, station_id, analysis_layers):
    """
    Computes Apparent Optical Properties (AOPs) including spectral Kd, Klu, 
    propagated Rrs(0+), and Kd(PAR) for user-defined depth layers.
    """
    if not analysis_layers or not station_data: return None
    pressure, ed_data, lu_data, wavelengths = station_data['pressure'], station_data['Ed_data'], station_data['Lu_data'], station_data['wavelengths']
    es_for_rrs = st.session_state.get('new_es_median', station_data.get('Es'))
    layer_k_data, results_data, rrs_dict, k_dict = {}, [], {'wavelength_nm': wavelengths}, {'wavelength_nm': wavelengths}
    selected_model = st.session_state.get('selected_kw_model', "Morel & Maritorena (2001)")
    df_kw = get_kw_data(selected_model); kw_interp = np.interp(wavelengths, df_kw['wavelength'], df_kw['Kw'])
    initial_rrs = calculate_Rrs(ed_data, lu_data, pressure, es_data=es_for_rrs)
    rrs_dict['initial_rrs_sr-1'] = initial_rrs
    
    for layer_num, data in analysis_layers.items():
        z_min, z_max = data['range']
        kd_raw, klu_raw = calculate_k_spectra(pressure, ed_data, lu_data, wavelengths, z_min, z_max)
        
        if klu_raw is not None and kd_raw is not None:
            # Force K values to be at least equal to pure water absorption (Kw). Here we can choose between Morel and 
            
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
    
    # Calculate PAR using quantum conversion

    h, c, N_A = 6.62607015e-34, 2.99792458e8, 6.02214076e23; par_wl_range = np.arange(400, 701, 1)
    valid_indices = np.where((wavelengths >= 400) & (wavelengths <= 700))[0]
    sub_wl, sub_ed = wavelengths[valid_indices], ed_data[:, valid_indices]
    interp_ed = np.array([np.interp(par_wl_range, sub_wl, row) for row in sub_ed])
    conversion_factor = (par_wl_range * 1e-9) / (h * c) * (1e-6 * 1e4) / N_A * 1e6
    par_profile_profiler = np.sum(interp_ed * conversion_factor, axis=1)
    kd_par_results = calculate_kd_par(pressure, par_profile_profiler, analysis_layers)
    kd_par_df = pd.DataFrame([{'Layer': f"Layer {ln}", 'Kd(PAR)': val} for ln, val in kd_par_results.items()])
    
    # Collect data for the empirical model with context
    if not results_df.empty and not kd_par_df.empty:
        model_data_source = pd.merge(results_df, kd_par_df, on="Layer")
        
        # Explicitly define the column order
        new_model_data = pd.DataFrame({
            'Station_ID': station_id,
            'Layer': model_data_source['Layer'],
            'Kd(PAR)': model_data_source['Kd(PAR)'],
            'Kd(490)': model_data_source['Kd(490)']
        }).dropna()

        if not new_model_data.empty:
            st.session_state.empirical_model_data = pd.concat([st.session_state.empirical_model_data, new_model_data], ignore_index=True)
            st.session_state.empirical_model_data.drop_duplicates(subset=['Station_ID', 'Layer'], keep='last', inplace=True)

    return { "layer_k_data": layer_k_data, "results_df": results_df, "kd_par_df": kd_par_df, "rrs_df_export": pd.DataFrame(rrs_dict), "k_df_export": pd.DataFrame(k_dict) }

def find_n2_peak(depth, temp, sal, lat):
    
    """
   Calculates N2 (Brunt-Väisälä frequency squared) using TEOS-10 
   and finds the depth of the maximum stability peak (pycnocline).
   """
   
    # Removes NaNs and sorts by depth
    
    df = pd.DataFrame({'p': depth, 't': temp, 'sp': sal}).dropna().sort_values('p')
    if len(df) < 5: return None, None, None

    SA = gsw.SA_from_SP(df['sp'], df['p'], -45, lat)
    CT = gsw.CT_from_t(SA, df['t'], df['p'])
    n2, p_mid = gsw.Nsquared(SA, CT, df['p'], lat)
    
    # Suaviza para evitar ruídos
    n2_smooth = pd.Series(n2).values#.rolling(window=3, center=True).mean().values
    
    # Ignores the first 1 meters to avoid surface noise/artifacts
    
    valid_idx = np.where(p_mid > 1.0)[0]
    if len(valid_idx) == 0: return p_mid, n2_smooth, None
    
    peak_idx = valid_idx[np.nanargmax(n2_smooth[valid_idx])]
    return p_mid, n2_smooth, p_mid[peak_idx]

def _handle_secchi_upload():
    
    """Callback function to process the master secchi file upload."""
    
    if st.session_state.master_secchi_loader is None:
        return
    try:
        df_loaded = pd.read_csv(st.session_state.master_secchi_loader)
        expected_cols = ['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']
        if all(col in df_loaded.columns for col in expected_cols):

            # Get a clean copy of the data from the CURRENT session.
            df_session = st.session_state.master_kd_secchi_df.copy()

            # Combine the session data FIRST, then the loaded data.
            combined_df = pd.concat([df_session, df_loaded], ignore_index=True)

            # De-duplicate, keeping the FIRST instance of any Station. This prioritizes any data you've added or updated in the current session.
            combined_df.drop_duplicates(subset=['Station_ID'], keep='first', inplace=True)
            
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

            # Get a clean copy of the data from the CURRENT session.
            df_session = st.session_state.empirical_model_data.copy()

            # Combine the session data FIRST, then the loaded data.
            combined_df = pd.concat([df_session, df_loaded], ignore_index=True)

            # De-duplicate, keeping the FIRST instance of any Station/Layer combo. Since the session data came first, it is ALWAYS prioritized.
            
            combined_df.drop_duplicates(subset=['Station_ID'], keep='first', inplace=True)

            st.session_state.empirical_model_data = combined_df
            st.session_state.notification = {'type': 'success', 'message': f"Successfully loaded and merged {len(df_loaded)} model data points."}
        else:
            st.session_state.notification = {'type': 'error', 'message': "Uploaded CSV is missing required columns."}
    except Exception as e:
        st.session_state.notification = {'type': 'error', 'message': f"Failed to read file: {e}"}
        
def generate_summary_report(station_id, analysis_layers, results_df, kd_par_df, wm_profiler=None, wm_external=None):
    
    """Generates a formatted text report with campaign metadata and optical results."""

    report = io.StringIO()
    report.write("============================================================\n")
    report.write("      OCEAN OPTICS EXPLORER - L3 SUMMARY REPORT\n")
    report.write("      Station ID: {station_id}\n")
    report.write("============================================================\n\n")

    report.write("--- 1. CAMPAIGN METADATA ---\n")
    report.write(f"Position: {st.session_state.lat:.5f}, {st.session_state.lon:.5f}\n")
    report.write(f"Depth Offset Applied (Profiler): {st.session_state.get('last_applied_offset_str', '0.00')} m\n\n")

    report.write("--- 2. PROFILER OPTICAL RESULTS ---\n")
    if results_df is not None:
        for _, row in results_df.iterrows():
            ln = row['Layer']
            
            # Attempts to extract the layer number (e.g., "Layer 1" -> 1) to fetch depth ranges

            try:
                layer_num = int(ln.split(' ')[1])
                if analysis_layers and layer_num in analysis_layers:
                    z_min, z_max = analysis_layers[layer_num]['range']
                    report.write(f"{ln} (Depth: {z_min:.1f}m to {z_max:.1f}m):\n")
                else:
                    report.write(f"{ln} (Depth range not found):\n")
            except:
                report.write(f"{ln}:\n")

            report.write(f"  - Kd(490):  {row['Kd(490)']:.4f} m⁻¹\n")
            
            # Adds Kd(PAR) if available for this specific layer
            
            if kd_par_df is not None:
                kpar_val = kd_par_df[kd_par_df['Layer'] == ln]['Kd(PAR)'].values
                if len(kpar_val) > 0:
                    report.write(f"  - Kd(PAR):  {kpar_val[0]:.4f} m⁻¹\n")
            
            # Adds Color (Apparent Visible Wavelength - AVW) if available
            
            idx = ln.split(" ")[1]
            c_rrs = f'propagated_rrs_L{idx}_sr-1'
            if st.session_state.derived_products and c_rrs in st.session_state.derived_products['rrs_df_export'].columns:
                rrs_v = st.session_state.derived_products['rrs_df_export'][c_rrs].values
                wls = st.session_state.derived_products['rrs_df_export']['wavelength_nm'].values
                avw_val = calculate_avw(wls, rrs_v)
                report.write(f"  - Color (AVW): {avw_val:.1f} nm\n")
            
            report.write("\n")  # Blank line between layers

    report.write("--- 3. EXTERNAL PROBE VALIDATION ---\n")
    if st.session_state.comparison_kd_df is not None:
        for idx, row in st.session_state.comparison_kd_df.iterrows():
            k_ext = row['Kd(PAR) (External)']
            if not pd.isna(k_ext): 
                report.write(f"  - {idx}: External Kd(PAR) = {k_ext:.4f} m⁻¹\n")
            else:
                report.write(f"  - {idx}: No External PAR data linked.\n")
    else:
        report.write("  No external probe data linked for optical validation.\n")

    report.write("\n--- 4. WATER MASS FRACTIONS (Average %) ---\n")
    
    report.write("  > From Profiler:\n")
    if wm_profiler is not None:
        for n in st.session_state.wm_names: 
            report.write(f"    - {n}: {wm_profiler[n].mean()*100:.1f}%\n")
    else:
        report.write("    (Not calculated)\n")
        
    report.write("\n  > From External Probe:\n")
    if wm_external is not None:
        for n in st.session_state.wm_names: 
            report.write(f"    - {n}: {wm_external[n].mean()*100:.1f}%\n")
    else:
        report.write("    (Not linked or calculated)\n")

    report.write("\n" + "="*60 + "\n")
    return report.getvalue()

def calculate_avw(wavelengths, rrs_spectrum):
    
    """Calculates the Apparent Visible Wavelength (AVW) between 400 and 700 nm."""
    
    mask = (wavelengths >= 400) & (wavelengths <= 700)
    w = wavelengths[mask]
    r = rrs_spectrum[mask]
    valid = ~np.isnan(r)
    if not np.any(valid) or np.nansum(r[valid]) <= 0:
        return np.nan
    return np.nansum(r[valid] * w[valid]) / np.nansum(r[valid])

def wavelength_to_hex(wavelength):
    
    """Maps the calculated AVW to a Hexadecimal color string for UI visualization."""

    if pd.isna(wavelength): return "#808080"
    cmap = plt.get_cmap('turbo')
    norm = plt.Normalize(vmin=400, vmax=700)
    rgba = cmap(norm(wavelength))
    return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"

def calculate_physical_properties(_station_data, lon, lat):
    
    """Computes derived physical properties (Salinity, Density, N2) using GSW (TEOS-10)."""
    
    pressure, temp, cond = _station_data['pressure'], _station_data['temperature'], _station_data['conductivity']; salinity=gsw.SP_from_C(cond, temp, pressure); SA=gsw.SA_from_SP(salinity, pressure, lon, lat); CT=gsw.CT_from_t(SA, temp, pressure); n2, p_mid=gsw.Nsquared(SA, CT, pressure); n2_padded=np.append(n2, [np.nan]*(len(pressure)-len(n2))); p_mid_padded=np.append(p_mid, [np.nan]*(len(pressure)-len(p_mid)))
    return pd.DataFrame({'Depth':pressure, 'Salinity':SA, 'Temperature':CT, 'n2':n2_padded, 'p_mid':p_mid_padded})

def calculate_k_spectra(pressure, ed_data, lu_data, wavelengths, z_min, z_max):
    
    """Calculates the diffuse attenuation coefficients (Kd and Klu) for a specific depth layer."""
    
    indices=np.where((pressure>=z_min)&(pressure<=z_max));
    if len(indices[0]) < 2: return None, None
    depths, ed_range, lu_range=pressure[indices], ed_data[indices], lu_data[indices]; kd, klu=[], []
    for i in range(wavelengths.shape[0]):
        ed_slice, lu_slice=ed_range[:, i], lu_range[:, i]; valid_ed, valid_lu=ed_slice > 0, lu_slice > 0
        kd.append(-np.polyfit(depths[valid_ed], np.log(ed_slice[valid_ed]), 1)[0] if np.sum(valid_ed) >= 2 else np.nan)
        klu.append(-np.polyfit(depths[valid_lu], np.log(lu_slice[valid_lu]), 1)[0] if np.sum(valid_lu) >= 2 else np.nan)
    return np.array(kd), np.array(klu)

def get_profile_metric_data(df, lat=-24.0): # Can change def lat
    
    """
    Returns (Depth, Value, Metric_Name) for plotting and profile alignment.
    Automatic fallback logic:
    1. Provided N2 (Best - from file)
    2. Calculated N2 (If Temp + Sal/Cond are available)
    3. Temperature Gradient (If only Temp is available) -> Thermocline
    """
    
    try:
        df = df.sort_values('Depth')
        
        # CASE 1: N2 PROVIDED IN FILE 

        if 'N2_Provided' in df.columns:
            valid = df.dropna(subset=['Depth', 'N2_Provided'])
            if not valid.empty:
                return valid['Depth'].values, valid['N2_Provided'].values, "N² (Do Arquivo)"

        # CASE 2: CALCULATE N2 (Needs Salinity or Conductivity)
        has_sal = 'Salinity' in df.columns
        has_cond = 'Conductivity' in df.columns
        
        if has_sal or has_cond:
            cols = ['Depth', 'Temperature'] + (['Salinity'] if has_sal else ['Conductivity'])
            valid = df.dropna(subset=cols)
            
            if not valid.empty:
                p = valid['Depth'].values
                t = valid['Temperature'].values
                
                if has_sal:
                    sp = valid['Salinity'].values
                else:
                    
                    # Convert conductivity units if necessary (uS/cm to mS/cm)

                    cond = valid['Conductivity'].values
                    if np.nanmean(cond) > 100: cond = cond / 1000.0
                    sp = gsw.SP_from_C(cond, t, p)
                
                SA = gsw.SA_from_SP(sp, p, -45, lat)
                CT = gsw.CT_from_t(SA, t, p)
                n2, p_mid = gsw.Nsquared(SA, CT, p, lat)
                
                # Light smoothing for visualization purposes
                
                n2_smooth = pd.Series(n2).rolling(window=3, center=True).mean().values
                return p_mid, n2_smooth, "N² (Calculado)"

        # CASE 3: TEMPERATURE ONLY (Thermal Gradient) 
        valid = df.dropna(subset=['Depth', 'Temperature'])
        if not valid.empty:
            p, t = valid['Depth'].values, valid['Temperature'].values
            
            # Calculates the vertical gradient (Peaks at the thermocline)
            
            dz = np.gradient(p)
            dt = np.gradient(t)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                dtdz = np.abs(dt / dz)
            
            dtdz_smooth = pd.Series(dtdz).rolling(window=5, center=True).mean().values
            
            return p, dtdz_smooth, "Thermal Gradient |dT/dz|"

        return None, None, "Error: No data"

    except Exception: return None, None, "Processing Error"
    
    
def extrapolate_ed0_minus(pressure, ed_data, z_min, z_max):
    
    """
    Extrapolates the downwelling irradiance just below the surface, Ed(0-), 
    from a stable, user-defined sub-surface layer using log-linear regression.
    Returns a 1D numpy array representing the Ed(0-) spectrum.
    """
    
    indices = np.where((pressure >= z_min) & (pressure <= z_max))
    if len(indices[0]) < 2:
        return np.full(ed_data.shape[1], np.nan)

    depths_in_layer = pressure[indices]
    ed_in_layer = ed_data[indices]
    
    ed0_extrapolated = []
    for i in range(ed_data.shape[1]):
        ed_slice = ed_in_layer[:, i]
        valid_mask = (ed_slice > 0) & (~np.isnan(ed_slice))
        
        if np.sum(valid_mask) >= 2:
            # Polyfit degree 1 on natural log of Ed vs depth. The intercept represents ln(Ed) at depth 0. Exponentiating it gives Ed(0-).
            
            _, intercept = np.polyfit(depths_in_layer[valid_mask], np.log(ed_slice[valid_mask]), 1)
            ed0_extrapolated.append(np.exp(intercept))
        else:
            ed0_extrapolated.append(np.nan)
            
    return np.array(ed0_extrapolated)

def load_castaway_ctd(uploaded_file):
    
    """
    Reads and parses a processed Castaway CTD file.
    - Features smart column mapping (case-insensitive).
    - Prioritizes 'calc_sal' if 'sal' is empty.
    - Correctly identifies pre-calculated 'N2' from the file.
    """
    
    try:
        # Attempts to read using comma or semicolon as separator
        
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            if df.shape[1] < 2:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        except Exception:
            return None

        # 1. Standardize column names (lowercase, stripped of whitespaces)

        df.columns = [c.strip().lower() for c in df.columns]

        # 2. Smart Mapping (Priorities). Creates a clean DataFrame to avoid duplicates and standardizes naming
        df_clean = pd.DataFrame()

        # Depth/Pressure
        
        if 'depth' in df.columns: df_clean['Depth'] = df['depth']
        elif 'press' in df.columns: df_clean['Depth'] = df['press']
        elif 'pressure' in df.columns: df_clean['Depth'] = df['pressure']
        else:
            st.error("Coluna de profundidade (depth/press) não encontrada.")
            return None

        # Temperature
        
        if 'temp' in df.columns: df_clean['Temperature'] = df['temp']
        elif 'temperature' in df.columns: df_clean['Temperature'] = df['temperature']
        else:
            st.error("Coluna de temperatura (temp) não encontrada.")
            return None

        #  N2 (From file) 
        if 'n2' in df.columns: 
            df_clean['N2_Provided'] = df['n2']
        elif 'press_n2' in df.columns: # Caso raro
            df_clean['N2_Provided'] = df['press_n2']

        # SALINITY (Prioritizes calculated salinity over raw salinity)
        
        col_sal = None
        if 'calc_sal' in df.columns and df['calc_sal'].count() > 0:
            col_sal = 'calc_sal'
        elif 'sal' in df.columns and df['sal'].count() > 0:
            col_sal = 'sal'
        elif 'salinity' in df.columns:
            col_sal = 'salinity'
            
        if col_sal:
            df_clean['Salinity'] = df[col_sal]
        
        # CONDUCTIVITY (Backup if Salinity is missing)
        if 'cond' in df.columns: df_clean['Conductivity'] = df['cond']
        elif 'conductivity' in df.columns: df_clean['Conductivity'] = df['conductivity']

        # 3. Numeric Cleaning. Corrects commas to dots for floating points and coerces to numeric
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = df_clean[col].str.replace(',', '.', regex=False)
                except:
                    pass
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Removes rows where depth is empty
        
        return df_clean.dropna(subset=['Depth'])

    except Exception as e:
        st.error(f"Erro leitura Castaway: {e}")
        return None

def get_profile_n2_data(df, lat=-24.0): #sem smoothing
    
    """
    Returns Depth and N2 vectors for plotting and alignment.
    - If Castaway (External): Uses 'N2_Provided' WITHOUT smoothing (raw file data).
    - If Profiler (Profiler): Calculates N2 using GSW library.
    """
    
    try:
        # Sorts by depth to ensure the plot renders correctly
        df = df.sort_values('Depth')
        
        # CASE 1: CASTAWAY (N2 is already provided)
        if 'N2_Provided' in df.columns:
            
            valid = df.dropna(subset=['Depth', 'N2_Provided'])
            
            if not valid.empty:
                
                # Returns RAW data (matches vendor software output exactly)

                return valid['Depth'].values, valid['N2_Provided'].values

        # CASE 2: PROFILER (Requires computation)
        
        cols_req = ['Depth', 'Temperature']
        if 'Conductivity' in df.columns: cols_req.append('Conductivity')
        elif 'Salinity' in df.columns: cols_req.append('Salinity')
        else: return None, None
        
        df_calc = df.dropna(subset=cols_req)
        if df_calc.empty: return None, None

        p = df_calc['Depth'].values
        t = df_calc['Temperature'].values
        
        if 'Salinity' in df_calc.columns:
            sp = df_calc['Salinity'].values
        else:
            cond = df_calc['Conductivity'].values
            if np.nanmean(cond) > 100: cond = cond / 1000.0 # uS -> mS
            sp = gsw.SP_from_C(cond, t, p)
            
        SA = gsw.SA_from_SP(sp, p, -45, lat)
        CT = gsw.CT_from_t(SA, t, p)
        n2, p_mid = gsw.Nsquared(SA, CT, p, lat)
        
        # Retains light smoothing for calculated data due to numerical noise. Its is commented at the moment
        n2_smooth = pd.Series(n2).values#.rolling(window=3, center=True).mean().values 
        return p_mid, n2_smooth

    except Exception as e:
        print(f"Debug N2 Error: {e}")
        return None, None
    

def calculate_Rrs(ed_data, lu_data, pressure, es_data=None, kd=None, klu=None, z_top=None):
    
    """
    Calculates the "Above-Water" Remote Sensing Reflectance (Rrs = Lw / Es), 
    with a rigorous denominator hierarchy depending on data availability.
    """
    # STEP 1: Calculate Water-Leaving Radiance, Lw(0+)
    
    use_propagation = all(param is not None for param in [kd, klu, z_top])

    if use_propagation:
        lu_sub_list = []
        for i in range(lu_data.shape[1]):
            col_data = lu_data[:, i]
            valid_mask = (col_data > 0) & (~np.isnan(col_data))
            if not np.any(valid_mask): lu_sub_list.append(np.nan); continue
            first_valid_idx = np.argmax(valid_mask)
            z_first_valid, lu_first_valid = pressure[first_valid_idx], col_data[first_valid_idx]
            if z_top < z_first_valid: z_calc, lu_calc = z_first_valid, lu_first_valid
            else: z_calc, lu_calc = z_top, np.interp(z_top, pressure, col_data)
            val = lu_calc * np.exp(klu[i] * z_calc); lu_sub_list.append(val)
        lu_sub = np.array(lu_sub_list)
    else:
        # Fallback: take the shallowest valid raw median
        valid_lu_indices = np.where(np.nanmedian(lu_data, axis=1) > 0)[0]
        lu_sub = lu_data[valid_lu_indices[0], :] if len(valid_lu_indices) > 0 else np.full(lu_data.shape[1], np.nan)

    # Transmission of Lu(0-) to Lw(0+) across the air-sea interface
    lw_spectrum = lu_sub * LW_TRANSMISSION_FACTOR

    #STEP 2: Determine the Above-Water Downwelling Irradiance, Es or Ed(0+)
        
    if es_data is not None:
        # Best scenario: Measured Surface Irradiance exists
        denominator = es_data
    elif use_propagation:
        # Second best: Extrapolate Ed(0-) from the stable layer, then transmit to Ed(0+)

        z_max = pressure[np.where(pressure >= z_top)[0][-1]]
        ed0_minus_spectrum = extrapolate_ed0_minus(pressure, ed_data, z_top, z_max)
        denominator = ed0_minus_spectrum / ED_TRANSMISSION_FACTOR
    else:
        # Fallback: take the shallowest raw reading and transmit to Ed(0+)

        valid_ed_indices = np.where(np.nanmedian(ed_data, axis=1) > 0)[0]
        ed0_minus_spectrum = ed_data[valid_ed_indices[0], :] if len(valid_ed_indices) > 0 else np.full(ed_data.shape[1], np.nan)
        denominator = ed0_minus_spectrum / ED_TRANSMISSION_FACTOR

    # STEP 3: Final Rrs Calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        rrs_spectrum = np.where(denominator > 0, lw_spectrum / denominator, np.nan)

    return rrs_spectrum

def calculate_kd_par(pressure, par_profile, analysis_layers):
    
    """
    Calculates the diffuse attenuation coefficient for PAR, Kd(PAR), 
    using a log-linear fit within defined depth layers.
    """
    
    kd_par_results = {};
    for layer_num, data in analysis_layers.items():
        z_min, z_max = data['range']; layer_indices = np.where((pressure >= z_min) & (pressure <= z_max))[0]
        if len(layer_indices) < 2: kd_par_results[layer_num] = np.nan; continue
        depths_in_layer = pressure[layer_indices]; par_in_layer = par_profile[layer_indices]; valid_par_indices = np.where(par_in_layer > 0)[0]
        if len(valid_par_indices) < 2: kd_par_results[layer_num] = np.nan; continue
        depths_for_fit = depths_in_layer[valid_par_indices]; par_for_fit = par_in_layer[valid_par_indices]; slope = np.polyfit(depths_for_fit, np.log(par_for_fit), 1)[0]; kd_par_results[layer_num] = -slope
    return kd_par_results

def get_kw_data(model_name):
    
    """Returns the pure water absorption/attenuation (Kw) dataframe based on the selected model."""
    
    if "Morel" in model_name:
        # Morel & Maritorena (2001) 

        data = {
            'wavelength': [350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700],
            'Kw': [0.0271, 0.0238, 0.0216, 0.0188, 0.0177, 0.01595, 0.0151, 0.01376, 0.01271, 0.01208, 0.01042, 0.0089, 0.00812, 0.00765, 0.00758, 0.00768, 0.0077, 0.00792, 0.00885, 0.0099, 0.01148, 0.01182, 0.01188, 0.01211, 0.01251, 0.0132, 0.01444, 0.01526, 0.0166, 0.01885, 0.02188, 0.02701, 0.03385, 0.0409, 0.04214, 0.04287, 0.04454, 0.0463, 0.04846, 0.05212, 0.05746, 0.06053, 0.0628, 0.06507, 0.07034, 0.07801, 0.09038, 0.11076, 0.13584, 0.16792, 0.2231, 0.25838, 0.26506, 0.26843, 0.27612, 0.284, 0.29218, 0.30176, 0.31134, 0.32553, 0.34052, 0.3715, 0.41048, 0.42947, 0.43946, 0.44844, 0.46543, 0.48642, 0.5164, 0.55939, 0.62438]
        }
    else:
        
        # Smith & Baker (1981)
        data = {
            'wavelength': [300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745], 
            'Kw': [0.154, 0.135, 0.116, 0.105, 0.0944, 0.0855, 0.0765, 0.0701, 0.0637, 0.0584, 0.053, 0.0485, 0.0439, 0.0396, 0.0353, 0.031, 0.0267, 0.025, 0.0233, 0.0221, 0.0209, 0.0203, 0.0196, 0.019, 0.0184, 0.0178, 0.0172, 0.0171, 0.017, 0.0169, 0.0168, 0.0172, 0.0176, 0.0176, 0.0175, 0.0185, 0.0194, 0.02303, 0.0212, 0.0242, 0.0271, 0.0321, 0.037, 0.043, 0.0489, 0.0504, 0.0519, 0.0544, 0.0568, 0.0608, 0.0648, 0.0683, 0.0717, 0.0762, 0.0807, 0.0949, 0.109, 0.1335, 0.158, 0.202, 0.245, 0.267, 0.29, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.34, 0.35, 0.375, 0.4, 0.415, 0.43, 0.44, 0.45, 0.475, 0.5, 0.575, 0.65, 0.742, 0.834, 1.002, 1.17, 1.485, 1.8, 2.09, 2.38, 2.425]
        }
    return pd.DataFrame(data)

def get_comparison_stats(df_merged, col_name):
    """Calculates R2, RMSE, MAE, and Bias between internal and external sensors."""
    y_true = df_merged[f'{col_name}_external']
    y_pred = df_merged[f'{col_name}_internal']
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    if np.sum(mask) < 2: 
        return {'R2': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'Bias': 0.0}
    
    return {
        'R2': r2_score(y_true[mask], y_pred[mask]),
        'RMSE': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
        'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
        'Bias': np.mean(y_pred[mask] - y_true[mask]) # Average deviation
    }

def get_regression_metrics(x_data, y_data):
    """Performs linear regression and returns R2, RMSE, MAE, Bias, Slope, Intercept, and P-value."""
    df = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
    
    if len(df) < 2 or df['x'].nunique() < 2:
        return {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Bias': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
    
    slope, intercept, r_value, p_value, _ = linregress(df['x'], df['y'])
    y_pred = slope * df['x'] + intercept
    
    rmse = np.sqrt(mean_squared_error(df['y'], y_pred))
    mae = mean_absolute_error(df['y'], y_pred)
    bias = np.mean(y_pred - df['y']) # Calculates bias for the regression model
    
    return {
        'R²': r_value**2,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'Slope': slope,
        'Intercept': intercept,
        'P-value': p_value
    }

def highlight_best(s):
    """Applies green background to the best statistics (Highest R2, Lowest RMSE/MAE/Bias)."""
    if s.name == 'Match Found' or s.dtype == object: 
        return [''] * len(s)
    
    is_max = s == s.max()
    is_min = s == s.min()
    is_min_abs = abs(s) == abs(s).min()

    if 'R2' in s.name:
        return ['background-color: #006400' if v else '' for v in is_max]
    elif 'Bias' in s.name:
        return ['background-color: #006400' if v else '' for v in is_min_abs]
    else: # RMSE and MAE: Lower is better
        return ['background-color: #006400' if v else '' for v in is_min]

def fig_to_buffer(fig):
    
    """Converts matplotlib figures to high-resolution PNG byte buffers for export."""
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def compare_ctd_profiles(df_i, df_e):
    
    """
    Interpolates internal and external CTD profiles to a common 0.5m grid 
    to allow direct statistical comparison. No smoothing filters are applied.
    """
    
    z_min = max(df_i['Depth'].min(), df_e['Depth'].min())
    z_max = min(df_i['Depth'].max(), df_e['Depth'].max())
    if z_min >= z_max: return None
    grid = np.arange(np.ceil(z_min*2)/2, np.floor(z_max*2)/2 + 0.5, 0.5)
    merged = pd.DataFrame({'Depth': grid})
    merged['Temperature_internal'] = np.interp(grid, df_i['Depth'], df_i['Temperature'])
    merged['Salinity_internal'] = np.interp(grid, df_i['Depth'], df_i['Salinity'])
    merged['Temperature_external'] = np.interp(grid, df_e['Depth'], df_e['Temperature'])
    merged['Salinity_external'] = np.interp(grid, df_e['Depth'], df_e['Salinity'])
    return merged.dropna()

def create_full_report_zip(zip_file, base_name):
    
    """Helper function to dump all generated plots and dataframes into a structured ZIP file."""
    
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
   Scans a local directory for Spectral Response Function (SRF) CSV files 
   and loads them into a dictionary for satellite convolution.
   This function is cached to run only once and save memory.
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
    
    """Calculates Kd(PAR) from an external probe's dataframe within a specific depth layer."""
    
    if df_external is None or depth_col not in df_external.columns or par_col not in df_external.columns: return np.nan
    df_external[depth_col] = pd.to_numeric(df_external[depth_col], errors='coerce'); df_external[par_col] = pd.to_numeric(df_external[par_col], errors='coerce'); df_layer = df_external.dropna(subset=[depth_col, par_col])
    df_layer = df_layer[(df_layer[depth_col] >= z_min) & (df_layer[depth_col] <= z_max)]; df_layer = df_layer[df_layer[par_col] > 0]
    if len(df_layer) < 2: return np.nan
    depths_for_fit = df_layer[depth_col]; par_for_fit = df_layer[par_col]; slope = np.polyfit(depths_for_fit, np.log(par_for_fit), 1)[0]
    return -slope

def estimate_chlorophyll_from_kdpar(kd_par):
    
    """
    Empirical Case-1 water model to estimate Chlorophyll-a from Kd(PAR).
    Subtracts the pure water contribution (0.022 m^-1) before calculation.
    """
    
    if kd_par <= 0.022: return 0.0
    kd_bio = kd_par - 0.022; a, b = 0.0512, 0.428; chlorophyll = (kd_bio / a)**(1 / b)
    return chlorophyll

def estimate_kd490_from_chlorophyll(chlorophyll):
    
    """
     Morel & Maritorena, 2001, to estimate 
    Kd(490) based on Chlorophyll-a concentration.
    """
    
    kw_490 = 0.0166; x = 0.07917; e = 0.7895; kd_490 = kw_490 + x * (chlorophyll**e)
    return kd_490


# ----------------------------- DATA UPLOAD HANDLERS -----------------------------

def _handle_master_rrs_upload():
    
    """Callback function to process the master Rrs file upload."""
    
    if st.session_state.master_rrs_loader is None:
        return
    try:
        # Read the uploaded CSV into a pandas DataFrame
        df_loaded = pd.read_csv(st.session_state.master_rrs_loader)
        
        # Store the DataFrame in the session state
        st.session_state.master_rrs_df = df_loaded
        st.toast(f"Successfully loaded {len(df_loaded.columns) - 1} historical Rrs spectra!")
    except Exception as e:
        st.toast(f"Failed to read master Rrs file: {e}")
        
def _handle_rad_upload():
    if st.session_state.up_rad_m:
        df = pd.read_csv(st.session_state.up_rad_m)
        st.session_state.master_vertical_radiometry = pd.concat([st.session_state.master_vertical_radiometry, df], ignore_index=True).drop_duplicates(subset=['Station_ID', 'Depth'], keep='last')
        st.toast("Radiometria mesclada!")

def _handle_anc_upload():
    if st.session_state.up_anc_m:
        df = pd.read_csv(st.session_state.up_anc_m)
        st.session_state.master_vertical_ancillary = pd.concat([st.session_state.master_vertical_ancillary, df], ignore_index=True).drop_duplicates(subset=['Station_ID', 'Depth'], keep='last')
        st.toast("Ancillary mesclada!")

def _handle_hyper_upload():
    if st.session_state.up_hyper_m:
        df = pd.read_csv(st.session_state.up_hyper_m)
        st.session_state.master_hyper_rrs_df = pd.concat([st.session_state.master_hyper_rrs_df, df], ignore_index=True).drop_duplicates(subset=['Station_ID'], keep='last')
        st.toast("Hyper Rrs mesclado!")

def _handle_multi_upload():
    if st.session_state.up_multi_m:
        df = pd.read_csv(st.session_state.up_multi_m)
        st.session_state.master_multi_rrs_df = pd.concat([st.session_state.master_multi_rrs_df, df], ignore_index=True).drop_duplicates(subset=['Station_ID', 'Sensor'], keep='last')
        st.toast("Multispectral mesclado!")        
        
def _handle_master_secchi_upload():
    """Callback to load and merge the Master Secchi/Kd database file."""
    if st.session_state.master_secchi_loader is not None:
        try:
            df_loaded = pd.read_csv(st.session_state.master_secchi_loader)
            expected_cols = ['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']
            
            if all(col in df_loaded.columns for col in expected_cols):
                # Merges with existing session data, prioritizing newly loaded data
                combined_df = pd.concat([st.session_state.master_kd_secchi_df, df_loaded], ignore_index=True)
                # Removes duplicates based on Station_ID (keeps the latest instance)
                combined_df.drop_duplicates(subset=['Station_ID'], keep='last', inplace=True)
                
                st.session_state.master_kd_secchi_df = combined_df
                st.toast("Master Secchi database successfully merged!")
            else:
                st.error("Uploaded CSV is missing the required columns.")
        except Exception as e:
            st.error(f"Failed to read the master file: {e}")
            
# ----------------------------- PLOTTING AND UTILITY FUNCTIONS -----------------------------

def plot_kd_secchi_relationship(df):
    
    """Generates comparative scatter plots and regression metrics for Secchi vs Kd."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Secchi vs Kd(PAR) (Comparative) 
    ax1.set_title("Secchi Depth vs. Kd(PAR)", weight='bold')
    ax1.set_xlabel("Secchi Depth (m)", weight='bold')
    ax1.set_ylabel("Kd(PAR) (m⁻¹)", weight='bold')

    df_par_prof = df[['Secchi', 'Kd(PAR)_Profiler']].dropna()
    df_par_ext = df[['Secchi', 'Kd(PAR)_External']].dropna()
    df_490_prof = df[['Secchi', 'Kd(490)_Profiler']].dropna()

    # Plot Profiler Data
    if not df_par_prof.empty:
        ax1.scatter(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'], ec='white', s=60, marker='o', label='Profiler', zorder=10)
        if len(df_par_prof) >= 2:
            metrics_prof = get_regression_metrics(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'])
            if not pd.isna(metrics_prof['R²']):
                x_vals = np.array(df_par_prof['Secchi'])
                ax1.plot(x_vals, metrics_prof['Intercept'] + metrics_prof['Slope'] * x_vals, '--', color='cyan', label=f'Profiler R² = {metrics_prof["R²"]:.3f}')

    # Plot External Probe Data
    if not df_par_ext.empty:
        ax1.scatter(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'], ec='white', s=60, marker='^', label='External Probe', zorder=10)
        if len(df_par_ext) >= 2:
            metrics_ext = get_regression_metrics(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'])
            if not pd.isna(metrics_ext['R²']):
                x_vals = np.array(df_par_ext['Secchi'])
                ax1.plot(x_vals, metrics_ext['Intercept'] + metrics_ext['Slope'] * x_vals, ':', color='yellow', label=f'External R² = {metrics_ext["R²"]:.3f}')
    
    ax1.grid(True, linestyle=':'); ax1.legend()

    # Plot 2: Secchi vs Kd(490) (Profiler Only)
    ax2.set_xlabel("Secchi Depth (m)", weight='bold')
    ax2.set_ylabel("Kd(490) (m⁻¹)", weight='bold')
    ax2.set_title("Secchi Depth vs. Kd(490) (Profiler)", weight='bold')
    
    if not df_490_prof.empty:
        ax2.scatter(df_490_prof['Secchi'], df_490_prof['Kd(490)_Profiler'], ec='white', s=50, zorder=10)
        if len(df_490_prof) >= 2:
            metrics_490 = get_regression_metrics(df_490_prof['Secchi'], df_490_prof['Kd(490)_Profiler'])
            if not pd.isna(metrics_490['R²']):
                x_min, x_max = df_490_prof['Secchi'].min(), df_490_prof['Secchi'].max()
                x_vals = np.array([x_min, x_max]) if x_min != x_max else np.array([x_min - 1, x_max + 1])  # Handles identical x-values
                ax2.plot(x_vals, metrics_490['Intercept'] + metrics_490['Slope'] * x_vals, '--', color='lime', label=f'R² = {metrics_490["R²"]:.3f}')
    
    ax2.grid(True, linestyle=':'); ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

    # Display All Regression Metrics 
    st.markdown("**Regression Metrics**")
    
    # Only calculate metrics if there is enough data 
    if len(df_par_prof) >= 2:
        metrics_prof_par = get_regression_metrics(df_par_prof['Secchi'], df_par_prof['Kd(PAR)_Profiler'])
    else:
        metrics_prof_par = {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}

    if len(df_par_ext) >= 2:
        metrics_ext_par = get_regression_metrics(df_par_ext['Secchi'], df_par_ext['Kd(PAR)_External'])
    else:
        metrics_ext_par = {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
        
    if len(df_490_prof) >= 2:
        metrics_prof_490 = get_regression_metrics(df_490_prof['Secchi'], df_490_prof['Kd(490)_Profiler'])
    else:
        metrics_prof_490 = {'R²': np.nan, 'RMSE': np.nan, 'Slope': np.nan, 'Intercept': np.nan, 'P-value': np.nan}
    
    df_metrics = pd.DataFrame([metrics_prof_par, metrics_ext_par, metrics_prof_490], 
                              index=['Secchi vs. Kd(PAR) (Profiler)', 
                                     'Secchi vs. Kd(PAR) (External)', 
                                     'Secchi vs. Kd(490) (Profiler)']).T
    st.dataframe(df_metrics.style.format('{:.4f}'), use_container_width=True)
    
def find_best_match(target_id, candidates):
    
    """
    Ultra-permissive fuzzy matching logic to link profiler station IDs 
    to external probe station IDs (e.g., links 'p1' to 'perfil_p1' or 'p1a').
    """
    
    if not target_id or candidates is None or len(candidates) == 0: return None
    t = str(target_id).lower().strip()
    c_list = [str(c) for c in candidates if str(c) != "nan"]
    
    # 1. Tries exact match first
    for c in c_list:
        if c.lower().strip() == t: return c
    
    # 2. Tries substring match
    for c in c_list:
        clean_c = c.lower().strip()
        if t in clean_c or clean_c in t: return c
        
    # 3. Fallback: If there's only one station in the external file, assume it's the right one
    if len(c_list) == 1: return c_list[0]
    
    return None


def plot_interactive_n2_offset(depth_ref, n2_ref, depth_target, n2_target, applied_offset, xlabel="Métrica", normalize=False):
    
    """
    Generates an interactive Plotly graph for visual alignment of stratification peaks.
    Layering order: Simulated (Background) -> Original (Middle) -> Reference (Top).
    """
    
    fig = go.Figure()

    # NORMALIZATION LOGIC
    if normalize:
        # Ref
        rmin, rmax = np.nanmin(n2_ref), np.nanmax(n2_ref)
        if rmax > rmin: n2_ref = (n2_ref - rmin) / (rmax - rmin)
        
        # Target
        tmin, tmax = np.nanmin(n2_target), np.nanmax(n2_target)
        if tmax > tmin: n2_target = (n2_target - tmin) / (tmax - tmin)
        
        xlabel = "Escala Normalizada (0 a 1)"

    #LAYER 1 (BACKGROUND): Simulated Profiler (Thick Transparent Blue)
    depth_corrected = depth_target + applied_offset
    fig.add_trace(go.Scatter(
        x=n2_target, y=depth_corrected, mode='lines', name='Simulated Offset (Blue)',
        line=dict(color='#00B4D8', width=6), # Bem grosso
        opacity=0.4, # Bem transparente
        hovertemplate='<b>Simulado</b><br>Prof: %{y:.2f}m<br>Val: %{x:.2f}<extra></extra>'
    ))

   # LAYER 2 (MIDDLE): Original Profiler (Solid Thin Red)
    fig.add_trace(go.Scatter(
        x=n2_target, y=depth_target, mode='lines', name='Original (Red)',
        line=dict(color='#FF4B4B', width=2, dash='solid'), # Sólido para ver melhor
        opacity=1.0, 
        hovertemplate='<b>Orig</b><br>Prof: %{y:.2f}m<br>Val: %{x:.2f}<extra></extra>'
    ))

    # LAYER 3 (TOP): External Reference (Green)
    fig.add_trace(go.Scatter(
        x=n2_ref, y=depth_ref, mode='lines', name='External Ref (Green)',
        line=dict(color='#00FF00', width=2),
        hovertemplate='<b>Ref</b><br>Prof: %{y:.2f}m<br>Val: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"<b>Ajuste Fino Interativo</b> (Current Offset: {applied_offset:+.2f}m)",
        xaxis_title=xlabel,
        yaxis_title="Profundidade (m)",
        # Autorange reversed 0 at the top, allowing negative depths if existent
        yaxis=dict(autorange="reversed"), 
        height=600,
        hovermode="y unified",
        template="plotly_dark",
        legend=dict(x=0.05, y=0.05, bgcolor='rgba(0,0,0,0.5)')
    )
    return fig

def get_col_index(columns, session_key):
    """Returns the index of a saved column in session_state to persist widget selections."""
    if session_key in st.session_state:
        val = st.session_state[session_key]
        if val in columns:
            return list(columns).index(val)
    return 0


def aggregate_to_vertical_masters(station_id, station_data, derived_products):
    """
    Aggregates processed data for the global campaign database, including 
    Radiometry, PAR, Chlorophyll, CDOM, and Water Mass fractions.
    """
    depths, wavelengths = station_data['pressure'], station_data['wavelengths']
    
    # 1. MASTER VERTICAL RADIOMETRY (Ed, Lu, and apparent Lu/Ed ratio)
    
    rad_df = pd.DataFrame({'Station_ID': station_id, 'Depth': depths})
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(station_data['Ed_data'] > 0, station_data['Lu_data'] / station_data['Ed_data'], np.nan)
    for i, wl in enumerate(wavelengths):
        w = int(wl)
        rad_df[f'Ed_{w}'] = station_data['Ed_data'][:, i]
        rad_df[f'Lu_{w}'] = station_data['Lu_data'][:, i]
        rad_df[f'LuEd_{w}'] = ratio[:, i]
    
    st.session_state.master_vertical_radiometry = pd.concat([
        st.session_state.master_vertical_radiometry[st.session_state.master_vertical_radiometry['Station_ID'] != station_id], 
        rad_df], ignore_index=True, sort=False)

    # 2. MASTER ANCILLARY DATA (PAR, Chl, CDOM, WM)
    anc_df = pd.DataFrame({'Station_ID': station_id, 'Depth': depths})
    
    # A. Profiler PAR (Calculated Quantum Flux)
    h_c, c_c, Na_c = 6.626e-34, 3e8, 6.022e23
    p_idx = np.where((wavelengths >= 400) & (wavelengths <= 700))[0]
    ed_q = station_data['Ed_data'][:, p_idx] * ((wavelengths[p_idx]*1e-9)/(h_c*c_c)*(1e-6*1e4)/Na_c*1e6)
    anc_df['PAR_Profiler'] = np.sum(ed_q, axis=1)

    # Initializes external columns with NaN
    anc_df['PAR_External'] = np.nan
    anc_df['Chl_External'] = np.nan
    anc_df['CDOM_External'] = np.nan

    # B. External Probe 1 Data (PAR and Chlorophyll)
    if st.session_state.external_probe_df is not None:
        df_p1 = st.session_state.external_probe_df
        st1_col = st.session_state.get('p1_st')
        z1_col = st.session_state.get('p1_z')
        par_col = st.session_state.get('p1_par')
        chl_col = st.session_state.get('p1_chl')

        if st1_col != "None" and z1_col != "None":
            # Finds corresponding station
            match1 = find_best_match(station_id, df_p1[st1_col].dropna().unique())
            if match1:
                df_s1 = df_p1[df_p1[st1_col].astype(str) == str(match1)]
                z_ext1 = pd.to_numeric(df_s1[z1_col], errors='coerce')
                
                # Interpolates PAR
                if par_col != "None" and par_col in df_s1:
                    anc_df['PAR_External'] = np.interp(depths, z_ext1, pd.to_numeric(df_s1[par_col], errors='coerce'), left=np.nan, right=np.nan)
                
                # Interpolates Chl 
                if chl_col != "None" and chl_col in df_s1:
                    anc_df['Chl_External'] = np.interp(depths, z_ext1, pd.to_numeric(df_s1[chl_col], errors='coerce'), left=np.nan, right=np.nan)

     # C. External Probe 2 Data (CDOM / Fluorimeter)
    if st.session_state.external_bio_df is not None:
        df_p2 = st.session_state.external_bio_df
        st2_col = st.session_state.get('p2_st')
        z2_col = st.session_state.get('p2_z')
        cdom_col = st.session_state.get('p2_cdom')

        if st2_col != "None" and z2_col != "None" and cdom_col != "None":
            match2 = find_best_match(station_id, df_p2[st2_col].dropna().unique())
            if match2:
                df_s2 = df_p2[df_p2[st2_col].astype(str) == str(match2)]
                z_ext2 = pd.to_numeric(df_s2[z2_col], errors='coerce')
                
                # Interpolates CDOM
                anc_df['CDOM_External'] = np.interp(depths, z_ext2, pd.to_numeric(df_s2[cdom_col], errors='coerce'), left=np.nan, right=np.nan)

    # D. Water Mass Fractions (WM)
    for n in st.session_state.wm_names:
        # From Profiler
        anc_df[f'WM_{n}_Profiler'] = st.session_state.mixture_df_profiler[n].values if st.session_state.mixture_df_profiler is not None else np.nan
        
        # From External Probe
        anc_df[f'WM_{n}_External'] = np.nan
        if st.session_state.mixture_df_external is not None and st.session_state.external_probe_df is not None:
            
            if st1_col != "None" and z1_col != "None":
                 match_mix = find_best_match(station_id, st.session_state.external_probe_df[st1_col].dropna().unique())
                 if match_mix:
                     df_mix_src = st.session_state.external_probe_df[st.session_state.external_probe_df[st1_col].astype(str) == str(match_mix)]
                     # Requires Temperature and Salinity to exist (same as Tab 4 logic)
                     df_mix_src = df_mix_src.dropna(subset=[st.session_state.p1_t, st.session_state.p1_s])
                     z_mix = pd.to_numeric(df_mix_src[z1_col], errors='coerce')
                     
                     if len(z_mix) == len(st.session_state.mixture_df_external):
                         anc_df[f'WM_{n}_External'] = np.interp(depths, z_mix, st.session_state.mixture_df_external[n], left=np.nan, right=np.nan)

    st.session_state.master_vertical_ancillary = pd.concat([
        st.session_state.master_vertical_ancillary[st.session_state.master_vertical_ancillary['Station_ID'] != station_id], 
        anc_df], ignore_index=True, sort=False)
    
    # 3. MASTER HYPERSPECTRAL RRS SNAPSHOT (L1 Propagated)
    if derived_products:
        rrs_df = derived_products['rrs_df_export']
        # Identifies the highest priority Rrs column (Propagated L1 > Propagated > Initial)

        target_col = next((c for c in rrs_df.columns if 'propagated_rrs_L1' in c), 
                          next((c for c in rrs_df.columns if 'propagated' in c), 'initial_rrs_sr-1'))
        rrs_snap = pd.DataFrame([rrs_df[target_col].values], columns=[f"Rrs_{int(w)}" for w in rrs_df['wavelength_nm']])
        rrs_snap.insert(0, 'Station_ID', station_id)
        st.session_state.master_hyper_rrs_df = pd.concat([st.session_state.master_hyper_rrs_df[st.session_state.master_hyper_rrs_df['Station_ID'] != station_id], rrs_snap], ignore_index=True, sort=False)

    st.toast(f"Snapshot of '{station_id}' successfully saved to master database!")

# ---------------------------- 1. SESSION STATE INITIALIZATION ----------------------------
keys_to_init = {
    'station_data': None, 
    'station_list': [], 
    'selected_station': None, 
    'lat': -24.101879, # Can be changed so a new default lat is set
    'lon': -45.692597, # Can be changed so a new default lon is set
    'profile_specific_layers': {},
    'derived_products': None,
    'notification': None,
    'discarded_indices': [],
    'new_es_median': None,
    'wm_names': ["AT", "ACAS", "AC"], # Can be changed so new default water masses are set
    'wm_vertices': None,
    
    # EXTERNAL PROBES
    'external_probe_df': None,
    'external_bio_df': None, 
    
    # MIXTURE 
    'mixture_df_profiler': None,
    'mixture_df_external': None,
    'mixture_results_profiler': None,
    'mixture_results_external': None,
    
    'uploader_id': 0,
    
    # MASTER TABLES
    'master_vertical_radiometry': pd.DataFrame(columns=['Station_ID', 'Depth']),
    'master_vertical_ancillary': pd.DataFrame(columns=['Station_ID', 'Depth']),
    'master_hyper_rrs_df': pd.DataFrame(columns=['Station_ID']),
    'master_multi_rrs_df': pd.DataFrame(columns=['Station_ID', 'Sensor']),
    'master_kd_secchi_df': pd.DataFrame(columns=['Station_ID', 'Secchi', 'Kd(PAR)_Profiler', 'Kd(490)_Profiler', 'Kd(PAR)_External']),
    'empirical_model_data': pd.DataFrame(columns=['Station_ID', 'Layer', 'Kd(PAR)', 'Kd(490)']), # Importante para a Tab 5
    
    # FIGURES 
    'fig_vp': None, 'fig_ve': None, 'fig_ts': None, 'fig_n2': None, 
    'fig_rrs': None, 'fig_log': None, 'fig_k': None, 'fig_comp': None,
    
    # SELECTED COLUMNS (Prevent key errors in Tab 1)
    'p1_st': "None", 'p1_z': "None", 'p1_t': "None", 'p1_s': "None", 
    'p1_chl': "None", 'p1_sec': "None", 'p1_par': "None",
    'p2_st': "None", 'p2_z': "None", 'p2_cdom': "None"
}

for key, value in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Ghost column cleanup to keep Master Ancillary organized
cols_limpeza = ['PAR_Profiler_Profiler', 'PAR_External_Probe', 'PAR_External_Probe_umol_m2_s', 'PAR_External_umol_m2_s']
for col in cols_limpeza:
    if col in st.session_state.master_vertical_ancillary.columns:
        st.session_state.master_vertical_ancillary = st.session_state.master_vertical_ancillary.drop(columns=[col])
        
        
        
# 5. UI: SIDEBAR
with st.sidebar:
    st.title("Ocean Optics Explorer"); st.header("1. Load Data"); uploaded_files = st.file_uploader("Upload In-Water Profiler Files (Temp, Cond, Ed, Lu)", accept_multiple_files=True, type="csv"); uploaded_beta_file = st.file_uploader("Upload Backscattering 'Beta' File (Optional)", type="csv", key="beta_uploader"); uploaded_es_file = st.file_uploader("Upload Es CSV File (Optional)", type="csv", key="es_uploader")
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
        st.header(f"3. Analyze Layers for '{st.session_state.selected_station}'"); current_data=st.session_state.station_data[st.session_state.selected_station]; max_depth=int(current_data['pressure'].max()) if current_data['pressure'].size > 0 else 100; selected_depth=st.slider("Select depth range (m):", 0.0, float(max_depth), (10.0, 20.0), step=0.5, key=f"slider_{st.session_state.selected_station}"); 
        layer_num_to_add = st.selectbox(
            "Choose layer to define:", 
            [1, 2, 3], 
            key=f"layer_select_{st.session_state.selected_station}"
        )
        

        col_add, col_rem = st.columns(2)
        
        station_id = st.session_state.selected_station
        z_min, z_max = selected_depth

        # Button 1: Save (Analyze)
        if col_add.button(f"Analyze L{layer_num_to_add}", use_container_width=True):
            if station_id not in st.session_state.profile_specific_layers: 
                st.session_state.profile_specific_layers[station_id] = {}
            
            st.session_state.profile_specific_layers[station_id][layer_num_to_add] = {'range': (z_min, z_max)}
            st.toast(f"Layer {layer_num_to_add} saved!")
            st.rerun()

        # Button 2: Clear (Remove)
        if col_rem.button(f"Clear L{layer_num_to_add}", use_container_width=True):
            # Check if layer exists before trying to delete
            if (station_id in st.session_state.profile_specific_layers and 
                layer_num_to_add in st.session_state.profile_specific_layers[station_id]):
                
                # 1. Remove from Analysis Definition
                del st.session_state.profile_specific_layers[station_id][layer_num_to_add]
                
                # 2. Remove from Bio-Optical Model Data (Tab 5)
                if not st.session_state.empirical_model_data.empty:
                    df_model = st.session_state.empirical_model_data
                    target_layer_name = f"Layer {layer_num_to_add}"
                    
                    # Filter: Keep rows that are NOT (Current Station AND Current Layer)
                    condition = (df_model['Station_ID'] == station_id) & (df_model['Layer'] == target_layer_name)
                    st.session_state.empirical_model_data = df_model[~condition]

                st.toast(f"Layer {layer_num_to_add} cleared.")
                st.rerun()
            else:
                st.toast("Layer not defined yet.")
        
        station_id = st.session_state.selected_station; defined_layers = st.session_state.profile_specific_layers.get(station_id, {})
        
        if defined_layers:
            st.markdown("**Defined Layers for this Profile:**")
            for layer, data in sorted(defined_layers.items()): st.write(f"  - Layer {layer}: `{data['range'][0]}m` - `{data['range'][1]}m`")
            if st.button("Clear Layers for This Profile"): st.session_state.profile_specific_layers[station_id] = {}; st.rerun()
            st.markdown("---")
            
        if st.button("Reset All & Restart App", type="primary", use_container_width=True):
            # 1. Increment the uploader ID so new widgets are created on rerun
            new_id = st.session_state.uploader_id + 1
            # 2. Clear all data
            st.session_state.clear()
            # 3. Restore the incremented ID
            st.session_state.uploader_id = new_id
            # 4. Rerun the app (Uploaders will now be empty)
            st.rerun()

# 6. UI: MAIN CONTENT
if not st.session_state.station_data:
    st.info("Welcome! Please upload your data files using the sidebar to begin.")
else:
    if 'previous_station' not in st.session_state or st.session_state.previous_station != st.session_state.selected_station:
       # Clear optical results on station change
       st.session_state.mixture_results = None
       st.session_state.mixture_df = None
       st.session_state.derived_products = None
       st.session_state.discarded_indices = []
       st.session_state.new_es_median = None
       
       # Critical cleanup to avoid dimension mismatch
       st.session_state.mixture_df_profiler = None
       st.session_state.mixture_df_external = None
       st.session_state.mixture_results_profiler = None
       st.session_state.mixture_results_external = None
       
       #Updates stations
       st.session_state.previous_station = st.session_state.selected_station

    # NOTIFICATION HANDLER
    if st.session_state.notification:
        msg = st.session_state.notification['message']
        icon = "✅" if st.session_state.notification['type'] == 'success' else "❌"
        
        st.toast(msg, icon=icon)
        
        # Clear the notification state
        st.session_state.notification = None # Clear the message after showing it once

    # DATA PREPARATION AND CENTRAL CALCULATIONS
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
        layer_k_data = {}; results_df = None; kd_par_df = None
        st.session_state.results_df = None; st.session_state.kd_par_df = None

    # EXTERNAL KD(PAR) CALCULATION
    if derived_products and st.session_state.get('external_probe_df') is not None:
        df_ext_raw = st.session_state.external_probe_df
        # Use as chaves estáveis da Tab 1
        ext_depth_col = st.session_state.get('p1_z')
        ext_par_col = st.session_state.get('p1_par')
        
        if ext_depth_col != "None" and ext_par_col != "None" and ext_depth_col in df_ext_raw.columns:
            comparison_data = []
            model = st.session_state.get('empirical_model_params')
            for index, profiler_row in kd_par_df.iterrows():
                # Correctly parse layer number
                layer_str = profiler_row['Layer']
                layer_num = int(layer_str.split(' ')[1])
                z_min, z_max = current_station_layers[layer_num]['range']
                
                external_kd_par = calculate_external_kd_par(df_ext_raw, ext_depth_col, ext_par_col, z_min, z_max)
                
                estimated_kd490_external = np.nan
                if model and not pd.isna(external_kd_par):
                    estimated_kd490_external = model['slope'] * external_kd_par + model['intercept']
                
                comparison_data.append({
                    "Layer": layer_str,
                    "Kd(PAR) (Profiler)": profiler_row['Kd(PAR)'],
                    "Kd(PAR) (External)": external_kd_par,
                    "Kd(490) (Profiler)": results_df.loc[index, 'Kd(490)'],
                    "Estimated Kd(490) (External)": estimated_kd490_external
                })
            st.session_state.comparison_kd_df = pd.DataFrame(comparison_data).set_index('Layer')
    else:
        st.session_state.comparison_kd_df = None
        
    fig_n2, fig_rrs, fig_log, fig_k, fig_ts = (plt.figure() for i in range(5))
    
    

    # TAB CREATION
    def on_tab_change():
        st.session_state.active_tab = st.session_state.main_tabs
    
    # 1. Clean Tab List
    tab_titles = [
        "Depth Correction", "CTD Comparison", "Es Tilt", 
        "Core Optical Analysis", "T-S Diagram", "Bio-Optical Models", 
        "Secchi & Kd Analysis", "Spectral Convolution", "Master Data & Reports", "Methods"
    ]
    
    # 2. Assign Tabs
    (tab_tools, tab_ctd, tab_es, tab_core, tab_ts, tab_models, 
     tab_secchi, tab_convolution, tab_report, tab_formulas) = st.tabs(tab_titles)


       
    # ------------------ TAB 1: DEPTH CORRECTION (TOOLS) ------------------
    
    with tab_tools:
        st.header("Depth Correction Tool")
        st.info("Visually align the profiles. The BLUE line should overlap the GREEN line.")
        
        if not st.session_state.station_data:
            st.warning("Please upload profiler files first.")
        else:
            c1, c2 = st.columns([1, 1])
            target_station = c1.selectbox("1. Profiler:", st.session_state.station_list, key="fix_tgt")
            uploaded_ref = c2.file_uploader("2. Reference (e.g. Castaway/External .csv):", type="csv", key="ref_upl")

            if uploaded_ref:
                if 'last_uploaded_ref' not in st.session_state or st.session_state.last_uploaded_ref != uploaded_ref.name:
                    st.session_state.df_ref_tool = load_castaway_ctd(uploaded_ref)
                    st.session_state.last_uploaded_ref = uploaded_ref.name
                
                df_ref = st.session_state.df_ref_tool
                
                if df_ref is not None:
                    if 'current_offset_val' not in st.session_state:
                        st.session_state.current_offset_val = 0.0
                    
                    if 'tool_d_ref' not in st.session_state or st.button("Reload Original Data"):
                        
                        # Process Reference
                        d_ref, val_ref, label_ref = get_profile_metric_data(df_ref, st.session_state.lat)
                        
                        if d_ref is None:
                            st.error("Error: Invalid or empty reference file.")
                            st.stop()

                        # Process Target
                        t_data = st.session_state.station_data[target_station]
                        
                        df_target = pd.DataFrame({'Depth': t_data['pressure'], 'Temperature': t_data['temperature']})
                        
                        # Se a referência usou Gradiente, NÃO damos condutividade para o Profiler,
                        # forçando ele a usar Gradiente também.
                        if "Gradiente" not in label_ref:
                            if len(t_data['conductivity']) == len(t_data['pressure']):
                                df_target['Conductivity'] = t_data['conductivity']
                        
                        # Process Target
                        d_tgt, val_tgt, label_tgt = get_profile_metric_data(df_target, st.session_state.lat)
                        
                        st.success(f"Comparing using: **{label_ref}**")

                        # Salva vetores
                        st.session_state.tool_d_ref = d_ref
                        st.session_state.tool_n2_ref = val_ref
                        st.session_state.tool_d_tgt = d_tgt
                        st.session_state.tool_n2_tgt = val_tgt
                        st.session_state.metric_label = label_ref
                        
                        # Tenta sugestão automática inicial
                        try:
                            # Ref (>2m)
                            mask_r = (d_ref > 2.0) & (~np.isnan(val_ref))
                            pk_r = d_ref[np.where(mask_r)[0][np.nanargmax(val_ref[mask_r])]] if np.any(mask_r) else 0.0
                            # Tgt (Todo)
                            mask_t = ~np.isnan(val_tgt)
                            pk_t = d_tgt[np.where(mask_t)[0][np.nanargmax(val_tgt[mask_t])]] if np.any(mask_t) else 0.0
                            
                            st.session_state.ruler_ref = float(pk_r)
                            st.session_state.ruler_tgt = float(pk_t)
                            st.session_state.current_offset_val = float(pk_r - pk_t)
                        except:
                            st.session_state.ruler_ref = 0.0
                            st.session_state.ruler_tgt = 0.0
                            st.session_state.current_offset_val = 0.0

                    # INTERFACE
                    st.markdown("---")
                    col_graph, col_ctrl = st.columns([3, 1])
                    
                    with col_ctrl:
                        st.subheader("Ruler & Adjustment")
                        st.markdown("Use the graph to read exact peak values.")
                        
                        val_ref = st.number_input("GREEN Peak (Reference):", value=st.session_state.get('ruler_ref', 0.0), step=0.1, format="%.2f")
                        val_tgt = st.number_input("RED Peak (Profiler):", value=st.session_state.get('ruler_tgt', 0.0), step=0.1, format="%.2f")
                        
                        if st.button("Calculate and Test Difference"):
                            diff = val_ref - val_tgt
                            st.session_state.current_offset_val = diff
                            st.rerun()
                        
                        st.markdown("---")
                        
                        final_offset = st.number_input(
                            "**Final Offset (m):**",
                            value=st.session_state.current_offset_val, 
                            step=0.1, format="%.2f",
                            help="This is the value that will be permanently applied."
                        )
                        
                        # Sincroniza
                        if final_offset != st.session_state.current_offset_val:
                            st.session_state.current_offset_val = final_offset
                            st.rerun()

                        st.markdown("---")
                        if st.button("APPLY FINAL OFFSET", type="primary"):
                            # 1. Aplica nos dados principais (Pressure, Temp, Cond, Ed, Lu)
                            t_data = st.session_state.station_data[target_station]
                            
                            new_p = t_data['pressure'] + st.session_state.current_offset_val
                            st.session_state.station_data[target_station]['pressure'] = new_p
                            
                            if t_data['beta_data'] is not None and not t_data['beta_data'].empty:
                                t_data['beta_data'].index = t_data['beta_data'].index + st.session_state.current_offset_val

                            if target_station in st.session_state.profile_specific_layers:
                                del st.session_state.profile_specific_layers[target_station]
                            
                            st.session_state.last_applied_offset_str = f"{st.session_state.current_offset_val:+.2f}"

                            st.toast(f"Corrected: {st.session_state.current_offset_val:+.2f}m em todos os sensores!")
                            
                            keys_to_clear = [
                                'tool_d_ref', 'tool_n2_ref', 
                                'tool_d_tgt', 'tool_n2_tgt', 
                                'current_offset_val', 
                                'ruler_ref', 'ruler_tgt'
                            ]
                            for k in keys_to_clear:
                                if k in st.session_state:
                                    del st.session_state[k]
                            
                            st.rerun()

                    with col_graph:
                        if 'tool_d_ref' in st.session_state:
                            
                            
                            norm_view = st.checkbox("Normalize Curves (0 to 1)", value=True, 
                                                  help="Forces curves to have the same visual width. Essential when comparing Gradients with N2.")
                            
                            d_r, n_r = st.session_state.tool_d_ref, st.session_state.tool_n2_ref
                            sort_r = np.argsort(d_r)
                            
                            d_t, n_t = st.session_state.tool_d_tgt, st.session_state.tool_n2_tgt
                            sort_t = np.argsort(d_t)
                            
                            xlabel_txt = st.session_state.get('metric_label', "Metric")
                            
                            fig = plot_interactive_n2_offset(
                                d_r[sort_r], n_r[sort_r],
                                d_t[sort_t], n_t[sort_t],
                                st.session_state.current_offset_val,
                                xlabel=xlabel_txt,
                                normalize=norm_view # <--- AQUI A MÁGICA
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Data error.")
            
            st.markdown("---")
            st.subheader("3. Export Corrected Data")
            
            if target_station in st.session_state.station_data:
                offset_suffix = st.session_state.get('last_applied_offset_str', 'corrected')
                zip_filename = f"{target_station}_Depth_Offset_{offset_suffix}m.zip"
                zip_buffer_fix = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer_fix, 'w', zipfile.ZIP_DEFLATED) as zf:
                    t_data = st.session_state.station_data[target_station]
                    
                    df_main = pd.DataFrame({'Depth_Corrected': t_data['pressure']})
                    if len(t_data['temperature']) == len(df_main): df_main['Temperature'] = t_data['temperature']
                    if len(t_data['conductivity']) == len(df_main): df_main['Conductivity'] = t_data['conductivity']
                    wavelengths = t_data['wavelengths']
                    if t_data['Ed_data'].shape[0] == len(df_main):
                        for i, wl in enumerate(wavelengths): df_main[f'Ed_{int(wl)}'] = t_data['Ed_data'][:, i]
                    if t_data['Lu_data'].shape[0] == len(df_main):
                        for i, wl in enumerate(wavelengths): df_main[f'Lu_{int(wl)}'] = t_data['Lu_data'][:, i]
                    
                    zf.writestr(f"{target_station}_Profiler_Offset_{offset_suffix}.csv", df_main.to_csv(index=False).encode('utf-8'))
                    
                    # 2. Beta
                    if t_data['beta_data'] is not None:
                        df_beta_save = t_data['beta_data'].reset_index()
                        col_prof = df_beta_save.columns[0]
                        df_beta_save.rename(columns={col_prof: 'Depth_Corrected'}, inplace=True)
                        zf.writestr(f"{target_station}_Beta_Offset_{offset_suffix}.csv", df_beta_save.to_csv(index=False).encode('utf-8'))
                
                st.download_button(
                    label=" Baixar Dados Corrigidos (.zip)",
                    data=zip_buffer_fix.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip",
                    key="dl_fixed_btn"
                )        
   
    
    # ------------------TAB 2: CTD & RADIOMETRIC COMPARISON ------------------
    with tab_ctd:

        st.header("External Probes Setup")
        
        st.subheader("Probe 1: CTD, Chl and Secchi")
        c1, c2 = st.columns(2)
        sep1 = c1.radio("Separator (P1)", [',', ';'], horizontal=True, key="s_p1_final")
        dec1 = c2.radio("Decimal (P1)", ['.', ','], horizontal=True, key="d_p1_final")
        
        up_p1 = st.file_uploader("Upload Probe 1 (Castaway/RBR/JFE)", type="csv", key=f"up_p1_{st.session_state.uploader_id}")
        if up_p1:
            st.session_state.external_probe_df = pd.read_csv(up_p1, sep=sep1, decimal=dec1)
        
        if st.session_state.external_probe_df is not None:
            df1 = st.session_state.external_probe_df
            cols1 = ["None"] + list(df1.columns)
            
            st.markdown("#### Configure Columns for Probe 1")
            m = st.columns(5)
            # Usamos on_change para garantir que o valor fique preso no session_state
            st.session_state.p1_st = m[0].selectbox("Station ID", cols1, index=get_col_index(cols1, 'p1_st'), key="sel_st1_f")
            st.session_state.p1_z = m[1].selectbox("Depth", cols1, index=get_col_index(cols1, 'p1_z'), key="sel_z1_f")
            st.session_state.p1_t = m[2].selectbox("Temp", cols1, index=get_col_index(cols1, 'p1_t'), key="sel_t1_f")
            st.session_state.p1_s = m[3].selectbox("Salinity", cols1, index=get_col_index(cols1, 'p1_s'), key="sel_s1_f")
            st.session_state.p1_chl = m[4].selectbox("Chl-a", cols1, index=get_col_index(cols1, 'p1_chl'), key="sel_chl1_f")
            
            ce = st.columns(2)
            st.session_state.p1_sec = ce[0].selectbox("Secchi Col", cols1, index=get_col_index(cols1, 'p1_sec'), key="sel_sec1_f")
            st.session_state.p1_par = ce[1].selectbox("PAR Col", cols1, index=get_col_index(cols1, 'p1_par'), key="sel_par1_f")

        st.divider()

        st.subheader("Probe 2: CDOM")
        c3, c4 = st.columns(2)
        sep2 = c3.radio("Separator (P2)", [',', ';'], horizontal=True, key="s_p2_final")
        dec2 = c4.radio("Decimal (P2)", ['.', ','], horizontal=True, key="d_p2_final")
        
        up_p2 = st.file_uploader("Upload Probe 2 (C3/Fluorimeter", type="csv", key=f"up_p2_{st.session_state.uploader_id}")
        if up_p2:
            st.session_state.external_bio_df = pd.read_csv(up_p2, sep=sep2, decimal=dec2)
            
        if st.session_state.external_bio_df is not None:
            df2 = st.session_state.external_bio_df
            cols2 = ["None"] + list(df2.columns)
            st.markdown("#### Configure Columns for Probe 2")
            b = st.columns(3)
            st.session_state.p2_st = b[0].selectbox("Station ID (P2)", cols2, index=get_col_index(cols2, 'p2_st'), key="sel_st2_f")
            st.session_state.p2_z = b[1].selectbox("Depth (P2)", cols2, index=get_col_index(cols2, 'p2_z'), key="sel_z2_f")
            st.session_state.p2_cdom = b[2].selectbox("CDOM Col", cols2, index=get_col_index(cols2, 'p2_cdom'), key="sel_cdom2_f")

        st.divider()
        
        st.subheader("Comparison Analysis (Profiler vs. Probe 1)")
        if st.session_state.external_probe_df is not None:
            casts = st.multiselect("Select casts to compare:", options=st.session_state.station_list, default=[st.session_state.selected_station], key="multi_comp")
            
            if st.button("Run Comparison Analysis", key="btn_run_comp_final", use_container_width=True):
                with st.spinner("Processing..."):
                    df_ext_raw = st.session_state.external_probe_df
                    st_col = st.session_state.p1_st
                    all_stats, all_merged_dfs = [], []
                    candidates = df_ext_raw[st_col].dropna().unique()
                    
                    for cid in casts:
                        match = find_best_match(cid, candidates)
                        if match:
                            df_e = df_ext_raw[df_ext_raw[st_col].astype(str) == str(match)].copy()
                            df_e['Depth'] = pd.to_numeric(df_e[st.session_state.p1_z], errors='coerce')
                            df_e['Temperature'] = pd.to_numeric(df_e[st.session_state.p1_t], errors='coerce')
                            df_e['Salinity'] = pd.to_numeric(df_e[st.session_state.p1_s], errors='coerce')
                            df_e.dropna(subset=['Depth', 'Temperature', 'Salinity'], inplace=True)
                            
                            df_i = calculate_physical_properties(st.session_state.station_data[cid], st.session_state.lon, st.session_state.lat)
                            merged = compare_ctd_profiles(df_i, df_e)
                            
                            if merged is not None and not merged.empty:
                                t_res = get_comparison_stats(merged, 'Temperature')
                                s_res = get_comparison_stats(merged, 'Salinity')
                                
                                all_stats.append({
                                    'Cast ID': cid, 'Match Found': str(match), 
                                    'Temp R2': t_res['R2'], 'Temp RMSE': t_res['RMSE'], 'Temp MAE': t_res['MAE'], 'Temp Bias': t_res['Bias'],
                                    'Sal R2': s_res['R2'], 'Sal RMSE': s_res['RMSE'], 'Sal MAE': s_res['MAE'], 'Sal Bias': s_res['Bias']
                                })
                                merged['Cast ID'] = cid; all_merged_dfs.append(merged)
                    
                    if all_merged_dfs:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
                        ((ax_t, ax_s), (ax_td, ax_sd)) = axes
                        colors = sns.color_palette("viridis", len(all_merged_dfs))
                        for i, df in enumerate(all_merged_dfs):
                            ax_t.plot(df['Temperature_internal'], df['Depth'], color=colors[i], label=f"Profiler {df['Cast ID'].iloc[0]}")
                            ax_s.plot(df['Salinity_internal'], df['Depth'], color=colors[i])
                            ax_t.plot(df['Temperature_external'], df['Depth'], 'w--', alpha=0.7, lw=1.5)
                            ax_s.plot(df['Salinity_external'], df['Depth'], 'w--', alpha=0.7, lw=1.5)
                            ax_td.plot(df['Temperature_internal']-df['Temperature_external'], df['Depth'], color=colors[i])
                            ax_sd.plot(df['Salinity_internal']-df['Salinity_external'], df['Depth'], color=colors[i])
                        
                        ax_t.set_title("Temperature"); ax_s.set_title("Salinity"); ax_t.invert_yaxis(); ax_td.invert_yaxis()
                        for ax in axes.flat: ax.grid(True, linestyle=':', alpha=0.5)
                        ax_t.legend(fontsize=8); ax_td.axvline(0, color='w', alpha=0.3); ax_sd.axvline(0, color='w', alpha=0.3)
                        
                        st.session_state.fig_comp = fig
                        st.session_state.comparison_table = pd.DataFrame(all_stats).set_index('Cast ID')
                        plt.close(fig)

        if st.session_state.get('fig_comp') is not None:
            st.pyplot(st.session_state.fig_comp)
            df_disp = st.session_state.comparison_table
            num_cols = [c for c in df_disp.columns if c != 'Match Found']
            st.dataframe(df_disp.style.apply(highlight_best).format({c: '{:.4f}' for c in num_cols}), use_container_width=True)

    # ------------------TAB 3: ES ------------------

    with tab_es:

        st.header("Surface Irradiance (Es) Visualization")
    
        all_es_spectra = station_data.get("Es_all_spectra", None)
    
        # Clean safe index
        if all_es_spectra is not None and not all_es_spectra.empty:
            all_es_spectra = all_es_spectra.reset_index(drop=True)
            y_max_es = st.number_input("Set Y-axis Max for Es Plot:", min_value=1.0, value=200.0, step=10.0, format="%.1f", key="es_ymax_selector")
            fig_es, ax_es = plt.subplots(figsize=(8, 6))
    
        # If no Es → stop
        if all_es_spectra is None or all_es_spectra.empty:
            st.warning("No measured Es file was uploaded for this station.")
            st.stop()
    
        # Create session state variables
        if "discarded_indices" not in st.session_state:
            st.session_state.discarded_indices = []
        if "new_es_median" not in st.session_state:
            st.session_state.new_es_median = None
    
        # 1. Identify rows that have at least one valid number (are not all NA).
        #    We use dropna(how='all') which only removes rows where EVERY column is NA.
        #    This gives us a DataFrame of only the usable spectra.
        valid_spectra_df = all_es_spectra.dropna(how='all', subset=[c for c in all_es_spectra.columns if c.startswith('X')])
        
        # 2. The original count is  the number of VALID spectra.
        original_valid_count = len(valid_spectra_df)
        
        # 3. Calculate how many of these valid spectra were kept.
        # We find the intersection of the valid indices and the kept indices.
        discard_set = set(st.session_state.discarded_indices)
        kept_indices = all_es_spectra.index.difference(discard_set)
        final_kept_valid_indices = valid_spectra_df.index.intersection(kept_indices)
        final_kept_count = len(final_kept_valid_indices)
    
        #   Summary metric 
        st.metric("Valid Spectra After Filtering", f"{final_kept_count} / {original_valid_count}")
    
    
        #   Plot Es spectra 
     
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
    
        # Original median (calculated only on valid spectra for accuracy)
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
        ax_es.set_xlim(380, 700)
        ax_es.set_ylim(bottom=0)
        ax_es.grid(True, linestyle='--')
        ax_es.legend()
        st.pyplot(fig_es)
    
        
        #   FILTERING PANEL 
        st.subheader("Es Stability Analysis & Hybrid Filtering")
        default_index = int(np.abs(np.array([float(c.replace("X", "")) for c in all_es_spectra.columns]) - 490).argmin())
        col_name_for_analysis = st.selectbox("Select reference wavelength for stability check:", options=list(all_es_spectra.columns), index=default_index)
        
        with st.form("hybrid_es_filters_form"):
            st.markdown("### Active Filters")
            use_tilt_filter = st.checkbox("Enable Tilt Filter", value=True)
            threshold_pct = st.number_input("Tilt threshold (% change)", 0.1, 50.0, 2.0, 0.5)
            use_low_filter = st.checkbox("Enable Low Outlier Filter", value=True)
            low_pct = st.number_input("Discard lowest %", 0.0, 49.9, 5.0, 1.0)
            use_high_filter = st.checkbox("Enable High Outlier Filter (Hybrid)", value=True)
            high_pct = st.number_input("Discard highest %", 0.0, 49.9, 5.0, 0.5)
            submitted = st.form_submit_button("Apply Filters")
    
        if submitted:
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
    
    


    
    # ------------------ TAB 4: CORE OPTICAL ANALYSIS ------------------
    with tab_core:
        st.header("Core Optical Analysis")
        top_row = st.columns(2)
        bottom_row = st.columns(2)
        plot_figsize = (5.5, 4.0)
        
        layer_colors = sns.color_palette("bright", 3)
        current_station_id = st.session_state.selected_station
        current_station_layers = st.session_state.profile_specific_layers.get(current_station_id, {})
    
        with top_row[0]:
            fig_n2, ax_n2 = plt.subplots(figsize=plot_figsize)
            
            # --- PLOT 1: N2 (Stratification) ---
            
            n2_clean = phys_props_df[['n2', 'p_mid']].dropna()
            
            line_n2, = ax_n2.plot(n2_clean['n2']*1e5, n2_clean['p_mid'], 
                                  color=STYLE_CONFIG['n2_line_color'], 
                                  zorder=10, 
                                  label=r'N$^2$')
            
            ax_n2.set_xlabel(r'N$^2$ (x 10$^{-5}$ s$^{-2}$)', weight='bold', color=STYLE_CONFIG['n2_line_color'])
            ax_n2.set_ylabel("Depth (m)", weight='bold')
            ax_n2.set_title("1. Stratification & Backscattering", weight='bold')
            ax_n2.invert_yaxis()
            ax_n2.tick_params(axis='x', labelcolor=STYLE_CONFIG['n2_line_color'])
    
            # --- PLOT 2: BETA (Backscattering) ---
            beta_data_df = station_data.get('beta_data')
            
            if beta_data_df is not None and not beta_data_df.empty and 'selected_beta_col' in st.session_state:
                ax_beta = ax_n2.twiny() # Create the secondary axis
                selected_beta_col = st.session_state.selected_beta_col
                
                
                beta_series = beta_data_df[selected_beta_col]
                
                line_beta, = ax_beta.plot(beta_series.values, beta_series.index, 
                                          color=STYLE_CONFIG['beta_line_color'], 
                                          zorder=9, 
                                          label=f'Beta ({selected_beta_col})')
                
                ax_beta.set_xlabel(r'$\beta$ (m$^{-1}$ sr$^{-1}$)', weight='bold', color=STYLE_CONFIG['beta_line_color'])
                ax_beta.tick_params(axis='x', labelcolor=STYLE_CONFIG['beta_line_color'])
                
                # Format Scientific Notation (e.g. 3.0 x 10^-3)
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3))
                ax_beta.xaxis.set_major_formatter(formatter)
                ax_beta.xaxis.offsetText.set_color(STYLE_CONFIG['beta_line_color'])
                
                # Combine Legends from both axes
                lines = [line_n2, line_beta]
                ax_n2.legend(lines, [l.get_label() for l in lines], loc='best')
            else:
                ax_n2.legend(loc='best')
                
            ax_n2.grid(True, linestyle='--')
            
            # Add background shading for layers (Using Seaborn colors)
            for i, (layer_num, data) in enumerate(current_station_layers.items()):
                ax_n2.axhspan(*data['range'], facecolor=layer_colors[i], alpha=0.3, zorder=0)
                
            st.pyplot(fig_n2)
            st.session_state.fig_n2 = fig_n2
        
        with top_row[1]:
            y_max_rrs = st.number_input("Set Y-axis Max for Rrs Plot:", min_value=0.001, value=0.010, step=0.001, format="%.3f", key="rrs_ymax_selector")
            fig_rrs, ax_rrs = plt.subplots(figsize=plot_figsize)
            if derived_products and 'rrs_df_export' in derived_products:
                rrs_df = derived_products['rrs_df_export']
                if 'initial_rrs_sr-1' in rrs_df.columns:
                    ax_rrs.plot(rrs_df['wavelength_nm'], rrs_df['initial_rrs_sr-1'], color='white', linestyle='--', label='Initial Rrs(0-)')
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    col_name = f'propagated_rrs_L{layer_num}_sr-1'
                    if col_name in rrs_df.columns:
                        ax_rrs.plot(rrs_df['wavelength_nm'], rrs_df[col_name], color=layer_colors[i], label=f'Prop. Rrs (L{layer_num})')
            else:
                final_es_for_rrs = st.session_state.get('new_es_median', station_data.get('Es'))
                initial_rrs = calculate_Rrs(station_data['Ed_data'], station_data['Lu_data'], pressure=station_data['pressure'], es_data=final_es_for_rrs)
                ax_rrs.plot(station_data['wavelengths'], initial_rrs, color='white', linestyle='--', label='Initial Rrs(0-)')
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
            # Aw Selection Box
            st.selectbox(
                "Select Pure Water Model (Kw):", 
                ["Morel & Maritorena (2001)", "Smith & Baker (1981)"],
                index=0,
                key="selected_kw_model"  # This key updates the session state automatically
            )

            fig_k, ax_k = plt.subplots(figsize=plot_figsize)
            
            # Recalculate which one to plot based on the state
            current_model = st.session_state.get('selected_kw_model', "Morel & Maritorena (2001)")
            df_kw_plot = get_kw_data(current_model)
            kw_interp_plot = np.interp(station_data['wavelengths'], df_kw_plot['wavelength'], df_kw_plot['Kw'])
            
            # Plot the dotted white line for the selected Pure Water Model
            label_name = "Kw (Morel 01)" if "Morel" in current_model else "Kw (Smith 81)"
            ax_k.plot(station_data['wavelengths'], kw_interp_plot, color='white', linestyle=':', lw=2, label=label_name, zorder=5)

            if current_station_layers and derived_products:
                layer_k_data = derived_products.get('layer_k_data', {})
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    if layer_num in layer_k_data:
                        # These values (kd_plot, klu_plot) were calculated using 
                        # the model selected in the box above because Streamlit reruns the script
                        kd_plot, klu_plot = layer_k_data[layer_num]
                        ax_k.plot(station_data['wavelengths'], kd_plot, color=layer_colors[i], linestyle='-', label=f'Kd (L{layer_num})', zorder=10)
                        ax_k.plot(station_data['wavelengths'], klu_plot, color=layer_colors[i], linestyle='--', label=f'klu (L{layer_num})', zorder=10)
            
            ax_k.set_xlabel("Wavelength (nm)", weight='bold')
            ax_k.set_ylabel(r'K (m$^{-1}$)', weight='bold')
            ax_k.set_title("4. Attenuation Spectra", weight='bold')
            ax_k.set_xlim(380, 700)
            ax_k.set_ylim(0,1)
            ax_k.grid(True, linestyle='--')
            if current_station_layers: ax_k.legend(fontsize=8)
            st.pyplot(fig_k)
            st.session_state.fig_k = fig_k
            
        # Lu/Ed RATIOS (SIDE-BY-SIDE). I think this may need adjustments
        st.markdown("---")
        st.subheader("5. In-Water Reflectance (Lu / Ed) Analysis")
        
        with st.expander("Show Reflectance Plots (Top vs. Average)", expanded=True):
            if not current_station_layers:
                st.info("Define layers in the sidebar to visualize water mass properties.")
            else:
                rat_col1, rat_col2 = st.columns(2)
                
                #  PLOT LEFT: TOP OF LAYER (Water Mass Interface)
                with rat_col1:
                    fig_rat_top, ax_rat_top = plt.subplots(figsize=(5.5, 4.0))
                    
                    for i, (layer_num, data) in enumerate(current_station_layers.items()):
                        z_min, z_max = data['range']
                        
                        # Logic: Find the SHALLOWEST VALID index within the layer
                        # We don't just take z_min, because data might be NaN there.
                        indices = np.where((station_data['pressure'] >= z_min) & 
                                           (station_data['pressure'] <= z_max))[0]
                        
                        if len(indices) > 0:
                            # Find the first index in this range where Ed has valid data
                            # (We check the reference wavelength, usually near 490nm, or just sum)
                            valid_idx = None
                            for idx in indices:
                                if np.sum(station_data['Ed_data'][idx, :] > 0) > 5: # Threshold to ensure spectrum exists
                                    valid_idx = idx
                                    break
                            
                            if valid_idx is not None:
                                # Calculate Ratio for this single depth
                                ed_spec = station_data['Ed_data'][valid_idx, :]
                                lu_spec = station_data['Lu_data'][valid_idx, :]
                                
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    ratio_top = lu_spec / ed_spec
                                
                                actual_depth = station_data['pressure'][valid_idx]
                                
                                ax_rat_top.plot(station_data['wavelengths'], ratio_top, 
                                                color=layer_colors[i], lw=2,
                                                label=f'L{layer_num} Top (@ {actual_depth:.1f}m)')
    
                    ax_rat_top.set_xlabel("Wavelength (nm)", weight='bold')
                    ax_rat_top.set_ylabel(r"$L_u / E_d$ (Unitless)", weight='bold')
                    ax_rat_top.set_title("Top-of-Layer Spectra (Interface)", weight='bold')
                    ax_rat_top.set_xlim(380, 700)
                    ax_rat_top.set_ylim(0,0.05)
                    ax_rat_top.grid(True, linestyle='--')
                    ax_rat_top.legend(fontsize=8)
                    st.pyplot(fig_rat_top)
                    st.session_state.fig_rat_top = fig_rat_top
    
                # PLOT RIGHT: AVERAGE OF LAYER (Bulk Property)
                with rat_col2:
                    fig_rat_avg, ax_rat_avg = plt.subplots(figsize=(5.5, 4.0))
                    
                    for i, (layer_num, data) in enumerate(current_station_layers.items()):
                        z_min, z_max = data['range']
                        
                        # Logic: Average all valid data in the range
                        indices = np.where((station_data['pressure'] >= z_min) & 
                                           (station_data['pressure'] <= z_max))[0]
                        
                        if len(indices) > 0:
                            ed_layer = station_data['Ed_data'][indices, :]
                            lu_layer = station_data['Lu_data'][indices, :]
                            
                            # Mean of spectra
                            mean_ed = np.nanmean(ed_layer, axis=0)
                            mean_lu = np.nanmean(lu_layer, axis=0)
                            
                            with np.errstate(divide='ignore', invalid='ignore'):
                                ratio_avg = mean_lu / mean_ed
                            
                            ax_rat_avg.plot(station_data['wavelengths'], ratio_avg, 
                                            color=layer_colors[i], lw=2, linestyle='--',
                                            label=f'L{layer_num} Mean ({z_min}-{z_max}m)')
    
                    ax_rat_avg.set_xlabel("Wavelength (nm)", weight='bold')
                    # Y label removed to save space, redundant with left plot
                    ax_rat_avg.set_title("Layer-Average Spectra (Bulk)", weight='bold')
                    ax_rat_avg.set_xlim(380, 700)
                    ax_rat_avg.set_ylim(0,0.05)
                    ax_rat_avg.grid(True, linestyle='--')
                    ax_rat_avg.legend(fontsize=8)
                    st.pyplot(fig_rat_avg)
                    st.session_state.fig_rat_avg = fig_rat_avg    
    
        st.subheader("6. Analysis Results")
        if not current_station_layers:
            st.info("Use the sidebar to select and analyze depth layers for the current profile.")
        else:
            if derived_products:
                st.dataframe(derived_products['results_df'].set_index('Layer').style.format({'Kd(490)':'{:.4f}', 'klu(490)':'{:.4f}'}), use_container_width=True)
                st.markdown("---")
                st.subheader("Kd(PAR) Results")
                st.dataframe(derived_products['kd_par_df'].set_index('Layer').style.format({'Kd(PAR)':'{:.4f}'}), use_container_width=True)
    
        st.markdown("---")
        with st.expander("7. Ed(0) Comparison: Surface vs. Extrapolated/Propagated", expanded=True):
            if not current_station_layers or not derived_products:
                st.warning("Analyze at least one layer to see the comparison plot.")
            else:
                y_max_ed_comp = st.number_input("Set Y-axis Max for Irradiance Plot:", min_value=1.0, value=150.0, step=10.0, format="%.1f", key="ed_comp_ymax_selector")
                
                fig_ed_comp, ax_ed_comp = plt.subplots(figsize=(8, 6))
                
                # Plot 1: Measured Es (if available) - This is Ed(0+)
                final_es_median = st.session_state.get('new_es_median', station_data.get('Es'))
                if final_es_median is not None:
                    label_es = "Measured Es (Filtered)" if st.session_state.new_es_median is not None else "Measured Es (Original)"
                    ax_ed_comp.plot(station_data['wavelengths'], final_es_median, color='#33FF57', label=label_es, lw=3.0, zorder=20)
                
                # Plot 2: Propagated/Extrapolated Ed(0+) for each layer
                layer_k_data = derived_products.get('layer_k_data', {})
                for i, (layer_num, data) in enumerate(current_station_layers.items()):
                    if layer_num in layer_k_data:
                        z_min, z_max = data['range']
                        # Calculate the in-water extrapolated value, Ed(0-)
                        ed0_minus = extrapolate_ed0_minus(station_data['pressure'], station_data['Ed_data'], z_min, z_max)
                        
                        # Plot both Ed(0-) and Ed(0+) 
                        if ed0_minus is not None:
                            # Plot the in-water value Ed(0-)
                            ax_ed_comp.plot(station_data['wavelengths'], ed0_minus, color='magenta', linestyle=':', label=f'Extrapolated Ed(0-) (from L{layer_num})', lw=2.0, zorder=18)
                            
                            # Plot the above-water equivalent, Ed(0+)
                            ed0_plus = ed0_minus / 0.96
                            ax_ed_comp.plot(station_data['wavelengths'], ed0_plus, color=layer_colors[i], linestyle='--', label=f'Extrapolated Ed(0+) (from L{layer_num})', lw=2.0, zorder=10)
    
                # Plot 3: Near-surface Ed from profiler
                valid_ed_indices = np.where(np.nanmedian(station_data['Ed_data'], axis=1) > 0)[0]
                if len(valid_ed_indices) > 0:
                    shallowest_ed = station_data['Ed_data'][valid_ed_indices[0], :]
                    ax_ed_comp.plot(station_data['wavelengths'], shallowest_ed, color='white', linestyle=':', label='Near-Surface Ed (Profiler)', lw=2.0, zorder=15)
                
                ax_ed_comp.set_xlabel("Wavelength (nm)", weight='bold')
                ax_ed_comp.set_ylabel(r'Irradiance ($\mu$W cm$^{-2}$ nm$^{-1}$)', weight='bold')
                ax_ed_comp.set_title("Comparison of Surface Irradiance Estimates", weight='bold')
                ax_ed_comp.set_xlim(400, 700)
                ax_ed_comp.set_ylim(bottom=0, top=y_max_ed_comp)
                ax_ed_comp.grid(True, linestyle='--'); ax_ed_comp.legend()
                st.pyplot(fig_ed_comp)
    
                st.info("""
                **How to interpret this plot:**
                - **Measured Es (Green):** Ground truth `Ed(0+)` from the surface sensor.
                - **Extrapolated Ed(0+) (Dashed Colors):** The `Ed` from the layer, extrapolated to the surface and converted to an above-water equivalent. This should ideally match the green line.
                - **Extrapolated Ed(0-) (Dotted Magenta):** The pure in-water surface value, before conversion. This line shows you exactly what the model predicts for just below the surface. It will always be slightly lower than `Ed(0+)`.
                - **Near-Surface Ed (Dotted White):** The shallowest raw measurement from the profiler, for reference.
                """)

    # ------------------ TAB 5: T-S DIAGRAM AND WATER MASS ANALYSIS ------------------
    with tab_ts:
        st.header("T-S Diagram and Water Mass Analysis")
        
        # Retrieves selected columns from session state for external probe data

        t_col = st.session_state.get('p1_t', "None")
        s_col = st.session_state.get('p1_s', "None")
        z_col = st.session_state.get('p1_z', "None")
        st_col = st.session_state.get('p1_st', "None")
        
        df_ext_ts = None
        ext_station_label = "None"

        col_plot, col_controls = st.columns([2, 1])

        with col_controls:
            st.subheader("Configuration & Linkage")
            
            # Logic to link external CTD data to the current profiler station

            if st.session_state.external_probe_df is not None:
                if t_col != "None" and s_col != "None" and st_col != "None":
                    st.markdown("### Link External Probe Data")
                    opcoes_probe = list(st.session_state.external_probe_df[st_col].dropna().unique().astype(str))
                    
                    # Uses fuzzy matching to suggest the best external station match

                    sugestao = find_best_match(st.session_state.selected_station, opcoes_probe)
                    
                    try: idx_inicial = opcoes_probe.index(str(sugestao))
                    except: idx_inicial = 0

                    estacao_probe_manual = st.selectbox(
                        "Select corresponding External Station:",
                        options=opcoes_probe,
                        index=idx_inicial,
                        key="ts_manual_probe_select"
                    )

                    df_target_ext = st.session_state.external_probe_df[
                        st.session_state.external_probe_df[st_col].astype(str) == estacao_probe_manual
                    ].copy()
                    
                    # Prepares the external T-S dataframe

                    df_ext_ts = pd.DataFrame({
                        'Temperature': pd.to_numeric(df_target_ext[t_col], errors='coerce'),
                        'Salinity': pd.to_numeric(df_target_ext[s_col], errors='coerce'),
                        'Depth': pd.to_numeric(df_target_ext[z_col], errors='coerce') if z_col != "None" else 0
                    }).dropna(subset=['Temperature', 'Salinity'])
                    
                    ext_station_label = estacao_probe_manual
                    st.success(f"Linked to: {ext_station_label}")

            st.divider()

            # WATER MASS VERTEX DEFINITION
            with st.expander("Water Mass End-Member Definition", expanded=True):
                st.info("Define the Temperature and Salinity of your regional water masses to calculate mixing fractions.")
                c1, c2, c3 = st.columns(3)
                for i in range(3):
                    st.session_state.wm_names[i] = c1.text_input(f"Name M{i+1}", value=st.session_state.wm_names[i], key=f"n{i}")
                
                # Standard vertices (e.g., Tropical Water, South Atlantic Central Water, Coastal Water)

                wm1_s = c2.number_input("Sal 1", value=37.20, format="%.2f", key="s1")
                wm1_t = c3.number_input("Temp 1", value=25.29, format="%.2f", key="t1")
                wm2_s = c2.number_input("Sal 2", value=35.19, format="%.2f", key="s2")
                wm2_t = c3.number_input("Temp 2", value=12.67, format="%.2f", key="t2")
                wm3_s = c2.number_input("Sal 3", value=34.36, format="%.2f", key="s3")
                wm3_t = c3.number_input("Temp 3", value=27.54, format="%.2f", key="t3")
                
                if st.button("Run Mixing Analysis", type="primary", use_container_width=True):
                    
                    # Triggers the barycentric coordinate calculation for both sensors
                    st.session_state.wm_vertices = [(wm1_s, wm1_t), (wm2_s, wm2_t), (wm3_s, wm3_t)]
                    st.session_state.mixture_results_profiler, st.session_state.mixture_df_profiler = analyze_mixture(
                        phys_props_df['Salinity'], phys_props_df['Temperature'], 
                        st.session_state.wm_vertices, st.session_state.wm_names
                    )
                    if df_ext_ts is not None:
                        st.session_state.mixture_results_external, st.session_state.mixture_df_external = analyze_mixture(
                            df_ext_ts['Salinity'], df_ext_ts['Temperature'], 
                            st.session_state.wm_vertices, st.session_state.wm_names
                        )
                    st.rerun()

            if st.session_state.mixture_df_profiler is not None:
                st.markdown("### Average Fractions (%)")
                df_avg = (st.session_state.mixture_df_profiler.mean() * 100).to_frame(name="Profiler")
                if st.session_state.mixture_df_external is not None:
                    df_avg["External Probe"] = st.session_state.mixture_df_external.mean() * 100
                st.dataframe(df_avg.style.format("{:.1f}%"), use_container_width=True)

        with col_plot:
            fig_ts, ax_ts = plt.subplots(figsize=(8, 8))
            df_int = phys_props_df[['Salinity', 'Temperature', 'Depth']].dropna()
            
            # DYNAMIC AXIS LIMITS (Data + Vertices)
            s_list = [df_int['Salinity'].min(), df_int['Salinity'].max()]
            t_list = [df_int['Temperature'].min(), df_int['Temperature'].max()]
            if df_ext_ts is not None and not df_ext_ts.empty:
                s_list.extend([df_ext_ts['Salinity'].min(), df_ext_ts['Salinity'].max()])
                t_list.extend([df_ext_ts['Temperature'].min(), df_ext_ts['Temperature'].max()])
            if st.session_state.wm_vertices:
                v_s = [v[0] for v in st.session_state.wm_vertices]
                v_t = [v[1] for v in st.session_state.wm_vertices]
                s_list.extend([min(v_s), max(v_s)])
                t_list.extend([min(v_t), max(v_t)])

            s_min, s_max = min(s_list), max(s_list)
            t_min, t_max = min(t_list), max(t_list)

            # ISOPYCNAL GRID (Density Lines)
            s_grid_ax = np.linspace(s_min - 1.0, s_max + 1.0, 100)
            t_grid_ax = np.linspace(t_min - 1.5, t_max + 1.5, 100)
            S_mesh, T_mesh = np.meshgrid(s_grid_ax, t_grid_ax)
            sigma_theta = gsw.sigma0(S_mesh, T_mesh)
            cnt = ax_ts.contour(S_mesh, T_mesh, sigma_theta, colors=STYLE_CONFIG['gridcolor'], alpha=0.3, linestyles='dashed', zorder=1)
            ax_ts.clabel(cnt, inline=True, fontsize=8, fmt='%.1f')
            
            # DATA PLOTTING
            if df_ext_ts is not None and not df_ext_ts.empty:
                ax_ts.scatter(df_ext_ts['Salinity'], df_ext_ts['Temperature'], c='grey', s=12, alpha=0.4, label=f'Probe: {ext_station_label}', zorder=2)
            
            # Profiler data colored by depth to show vertical structure

            sc = ax_ts.scatter(df_int['Salinity'], df_int['Temperature'], c=df_int['Depth'], cmap='viridis_r', s=45, ec='black', lw=0.5, label='Perfilador (Profiler)', zorder=3)
            cbar = plt.colorbar(sc, ax=ax_ts, pad=0.02)
            cbar.set_label('Profundidade (m)', weight='bold'); cbar.ax.invert_yaxis()
            
            # MIXING TRIANGLE OVERLAY
            if st.session_state.wm_vertices:
               poly = Polygon(np.array(st.session_state.wm_vertices), closed=True, fill=False, edgecolor='cyan', ls='--', lw=2, zorder=4)
               ax_ts.add_patch(poly)
               for i, (sal, temp) in enumerate(st.session_state.wm_vertices):
                   ax_ts.text(sal, temp, f" {st.session_state.wm_names[i]}", color='cyan', fontsize=12, fontweight='bold', zorder=5)
            
            ax_ts.set_xlim(s_min - 0.3, s_max + 0.3)
            ax_ts.set_ylim(t_min - 0.5, t_max + 0.5)
            ax_ts.set_xlabel("Absolute Salinity (g/kg)", weight='bold')
            ax_ts.set_ylabel("Conservative Temperature (°C)", weight='bold')
            ax_ts.set_title(f"T-S Diagram: {st.session_state.selected_station}", weight='bold', pad=15)
            ax_ts.grid(True, linestyle=':', alpha=0.2)
            ax_ts.legend(loc='lower right', fontsize=9, framealpha=0.8)
            
            st.pyplot(fig_ts)
            st.session_state.fig_ts = fig_ts

        # VERTICAL MIXING PROFILES
        st.divider()
        st.subheader("Vertical Mixing Profiles (%)")
        m1, m2 = st.columns(2)
        
        with m1:
            st.markdown(f"**Profiler ({st.session_state.selected_station})**")
            if st.session_state.mixture_df_profiler is not None:
                fig_vp, ax_vp = plt.subplots(figsize=(5, 7))
                for name in st.session_state.wm_names: 
                    ax_vp.plot(st.session_state.mixture_df_profiler[name]*100, phys_props_df['Depth'], label=name, lw=2.5)
                ax_vp.invert_yaxis()
                ax_vp.set_xlabel("Proportion (%)")
                ax_vp.set_ylabel("Depth (m)")
                ax_vp.set_xlim(0, 105)
                ax_vp.legend()
                ax_vp.grid(True, ls=':')
                st.pyplot(fig_vp)
                st.session_state.fig_vp = fig_vp
            else:
                st.info("Run mixture analysis to visualize.")

        with m2:
            
            # Fraction vs Depth for the External Probe (e.g., C3/Castaway)
            st.markdown(f"**External Probe ({ext_station_label})**")
            if st.session_state.mixture_df_external is not None and df_ext_ts is not None:
                fig_ve, ax_ve = plt.subplots(figsize=(5, 7))
                z_ext = df_ext_ts['Depth']
                for name in st.session_state.wm_names: 
                    ax_ve.plot(st.session_state.mixture_df_external[name]*100, z_ext, label=name, lw=2.5)
                ax_ve.invert_yaxis()
                ax_ve.set_xlabel("Proportion (%)")
                ax_ve.set_xlim(0, 105)
                ax_ve.legend()
                ax_ve.grid(True, ls=':')
                st.pyplot(fig_ve)
                st.session_state.fig_ve = fig_ve
            else:
                st.info("External probe data unavailable or not calculated.")
                
                
    
    # ------------------ TAB 6: BIO-OPTICAL MODELS ------------------
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
                # Callback to handle historical model data loading

                st.file_uploader(
                    "Load Master Model Data (CSV)", 
                    type="csv", 
                    key="master_model_loader",
                    on_change=_handle_model_upload
                )
            
            with col2:
                # Export current training data for future sessions

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
                
                
                # 1. Define column order
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
                        # Performs linear regression to find site-specific relationship

                        slope, intercept, r_value, p_value, _ = linregress(df_model['Kd(PAR)'], df_model['Kd(490)'])
                        st.session_state.empirical_model_params = {'slope': slope, 'intercept': intercept, 'r2': r_value**2}
                        
                        st.success("Empirical model generated successfully!")
                        st.metric("Model Equation", f"Kd(490) = {slope:.4f} * Kd(PAR) + {intercept:.4f}")
                        st.metric("Model R²", f"{r_value**2:.4f}")
                        
                        # Plot the empirical regression line
                        fig_model, ax = plt.subplots()
                        ax.scatter(df_model['Kd(PAR)'], df_model['Kd(490)'], ec='white', label='Data Points')
                        x_vals = np.array(df_model['Kd(PAR)'])
                        y_vals = slope * x_vals + intercept
                        ax.plot(x_vals, y_vals, '--', color='cyan', label='Regression Line')
                        ax.set_xlabel("Kd(PAR) (m⁻¹)", weight='bold'); ax.set_ylabel("Kd(490) (m⁻¹)", weight='bold')
                        ax.set_title("Empirical Model: Kd(PAR) vs. Kd(490)", weight='bold'); ax.grid(True, linestyle=':'); ax.legend()
                        st.pyplot(fig_model)
    
        st.subheader("2. Literature Model (Morel & Maritorena, 2001)")
        with st.expander("Estimate Kd(490) from Kd(PAR) using Published Model"):
            st.markdown("""This model is most accurate for open-ocean (Case-1) waters and may differ from your site-specific model.""")
            kd_par_input = st.number_input("Enter a Kd(PAR) value (m⁻¹):", min_value=0.0, value=0.1, step=0.01, format="%.4f")
            if kd_par_input > 0:
                # Estimate Chl-a and then Kd(490) using literature coefficients

                estimated_c = estimate_chlorophyll_from_kdpar(kd_par_input)
                estimated_kd490 = estimate_kd490_from_chlorophyll(estimated_c)
                col1, col2 = st.columns(2)
                with col1: st.metric(label="Estimated Chlorophyll-a (mg/m³)", value=f"{estimated_c:.4f}")
                with col2: st.metric(label="Estimated Kd(490) (m⁻¹)", value=f"{estimated_kd490:.4f}")
  
   
    # ------------------ TAB 7: SECCHI & KD ANALYSIS ------------------

    with tab_secchi:
        st.header("Secchi & Kd Analysis")

        # --- GERENCIAMENTO DE DADOS HISTÓRICOS (MASTER) ---
        with st.expander("Manage Campaign Master Database (Historical Data)", expanded=False):
            st.info("Merge data from different days or download the current progress.")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.file_uploader(
                    "Upload Master Secchi/Kd (CSV)", 
                    type="csv", 
                    key="master_secchi_loader",
                    on_change=_handle_master_secchi_upload,
                    help="Upload a previously saved .csv to continue historical analysis."
                )
            
            with col_m2:
                if not st.session_state.master_kd_secchi_df.empty:
                    csv_data = st.session_state.master_kd_secchi_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Master Campaign CSV",
                        data=csv_data,
                        file_name="Master_Campaign_Secchi_Kd.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                else:
                    st.button("Download base (empty) master csv", disabled=True, use_container_width=True)

        st.divider()

        # LIVE DATA PREVIEW
        st.subheader("1. Active Station Preview")
        if st.session_state.get('comparison_kd_df') is not None:
            current_layer = st.session_state.results_df.iloc[0]['Layer'] if st.session_state.results_df is not None else "N/A"
            p_col1, p_col2, p_col3 = st.columns(3)
            try:
                val_prof = st.session_state.comparison_kd_df.loc[current_layer, "Kd(PAR) (Profiler)"]
                val_ext = st.session_state.comparison_kd_df.loc[current_layer, "Kd(PAR) (External)"]
                val_490 = st.session_state.comparison_kd_df.loc[current_layer, "Kd(490) (Profiler)"]
                
                p_col1.metric(f"Kd(PAR) Profiler ({current_layer})", f"{val_prof:.4f} m⁻¹")
                p_col2.metric(f"Kd(PAR) External ({current_layer})", f"{val_ext:.4f} m⁻¹" if not pd.isna(val_ext) else "No Data")
                p_col3.metric(f"Kd(490) Profiler ({current_layer})", f"{val_490:.4f} m⁻¹")
            except:
                st.info("Define layers in Tab 4 to see metrics.")
        else:
            st.info("Link an external probe in Tab 2 to enable cross-validation.")
    
        st.divider()

        # LINK & SAVE
        st.subheader("2. Link Secchi & Save to Campaign")
        
        secchi_input = np.nan
        if st.session_state.external_probe_df is not None:
            
            # Logic to automatically pull Secchi values from the external probe file based on station ID

            df_s = st.session_state.external_probe_df
            st_col = st.session_state.get('p1_st')
            val_col = st.session_state.get('p1_sec')
            
            if st_col != "None" and val_col != "None" and st_col in df_s.columns:
                opcoes_probe = list(df_s[st_col].dropna().unique().astype(str))
                sugestao = find_best_match(st.session_state.selected_station, opcoes_probe)
                try: idx_sug = opcoes_probe.index(str(sugestao))
                except: idx_sug = 0

                c_vinc, c_met = st.columns([2, 1])
                with c_vinc:
                    vinc_estacao = st.selectbox("Link Secchi from Station (External Probe):", options=opcoes_probe, index=idx_sug, key="secchi_station_link_select")
                with c_met:
                    entry_mode = st.radio("Method:", ["Automated", "Manual"], horizontal=True)

                df_row = df_s[df_s[st_col].astype(str) == vinc_estacao]
                auto_val = pd.to_numeric(df_row.iloc[0][val_col], errors='coerce') if not df_row.empty else np.nan

                if entry_mode == "Automated/Link":
                    if not pd.isna(auto_val):
                        st.success(f"Value found: **{auto_val:.2f} m**")
                        secchi_input = auto_val
                    else:
                        st.error("No Secchi value found for this station ID.")
                else:
                    secchi_input = st.number_input("Manually Enter Secchi Depth (m):", min_value=0.0, step=0.1, format="%.2f")
            else:
                secchi_input = st.number_input("Manually Enter Secchi Depth (m):", min_value=0.0, step=0.1, format="%.2f")
        else:
            secchi_input = st.number_input("Manually Enter Secchi Depth (m):", min_value=0.0, step=0.1, format="%.2f")

        
        if st.button("Commit Current Station to Master Database", type="primary", use_container_width=True):
            
            # Saves a single station's optical and Secchi data to the cumulative campaign dataframe
            if not pd.isna(secchi_input) and st.session_state.results_df is not None:
                layer_key = st.session_state.results_df.iloc[0]['Layer']
                new_row = pd.DataFrame({
                    'Station_ID': [st.session_state.selected_station], 
                    'Secchi': [secchi_input],
                    'Kd(PAR)_Profiler': [st.session_state.kd_par_df.iloc[0]['Kd(PAR)']], 
                    'Kd(490)_Profiler': [st.session_state.results_df.iloc[0]['Kd(490)']],
                    'Kd(PAR)_External': [st.session_state.comparison_kd_df.loc[layer_key, "Kd(PAR) (External)"] if st.session_state.comparison_kd_df is not None else np.nan]
                })
                
                # Mescla e evita duplicatas na sessão
                m_sec = st.session_state.master_kd_secchi_df
                st.session_state.master_kd_secchi_df = pd.concat([m_sec[m_sec['Station_ID'] != st.session_state.selected_station], new_row], ignore_index=True)
                st.toast(f"Station {st.session_state.selected_station} saved!")
            else:
                st.error("Double-check for Kd and Secchi data validity ")

        st.divider()

        st.subheader("3. Campaign Master Database")
        df_master = st.session_state.master_kd_secchi_df
        if not df_master.empty:
            st.dataframe(df_master.style.format({'Secchi': '{:.2f}', 'Kd(PAR)_Profiler': '{:.4f}', 'Kd(490)_Profiler': '{:.4f}', 'Kd(PAR)_External': '{:.4f}'}), use_container_width=True)
            
            st.subheader("4. Regression Analysis: Secchi vs. Kd")
            
            fig_secchi, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            d_Profiler = df_master[['Secchi', 'Kd(PAR)_Profiler']].dropna()
            if not d_Profiler.empty:
                ax1.scatter(d_Profiler['Secchi'], d_Profiler['Kd(PAR)_Profiler'], 
                            color='#00E5FF', ec='white', s=100, marker='o', label='Profiler (Profiler)', zorder=5)
                
                if len(d_Profiler) >= 2 and d_Profiler['Secchi'].nunique() > 1:
                    m_j = get_regression_metrics(d_Profiler['Secchi'], d_Profiler['Kd(PAR)_Profiler'])
                    x_fit = np.array([d_Profiler['Secchi'].min(), d_Profiler['Secchi'].max()])
                    ax1.plot(x_fit, m_j['Intercept'] + m_j['Slope'] * x_fit, 
                             '--', color='#00E5FF', alpha=0.8, label=f'Fit Profiler (R²={m_j["R²"]:.2f})')

            d_ext = df_master[['Secchi', 'Kd(PAR)_External']].dropna()
            if not d_ext.empty:
                ax1.scatter(d_ext['Secchi'], d_ext['Kd(PAR)_External'], 
                            color='#FF8C00', ec='white', s=100, marker='^', label='External Probe', zorder=5)
                
                if len(d_ext) >= 2 and d_ext['Secchi'].nunique() > 1:
                    m_e = get_regression_metrics(d_ext['Secchi'], d_ext['Kd(PAR)_External'])
                    x_fit_e = np.array([d_ext['Secchi'].min(), d_ext['Secchi'].max()])
                    ax1.plot(x_fit_e, m_e['Intercept'] + m_e['Slope'] * x_fit_e, 
                             ':', color='#FF8C00', alpha=0.8, label=f'Fit Probe (R²={m_e["R²"]:.2f})')
            
            ax1.set_xlabel("Secchi Depth (m)", weight='bold')
            ax1.set_ylabel(r"$K_d(PAR)$ ($m^{-1}$)", weight='bold')
            ax1.set_title("Secchi Depth vs. Kd(PAR)", weight='bold')
            ax1.legend(loc='upper right', fontsize=9, framealpha=0.6)
            ax1.grid(True, ls=':', alpha=0.4)

            d_490 = df_master[['Secchi', 'Kd(490)_Profiler']].dropna()
            if not d_490.empty:
                ax2.scatter(d_490['Secchi'], d_490['Kd(490)_Profiler'], 
                            color='#39FF14', ec='white', s=100, marker='s', label='Kd(490) Profiler', zorder=5)
                
                if len(d_490) >= 2 and d_490['Secchi'].nunique() > 1:
                    m4 = get_regression_metrics(d_490['Secchi'], d_490['Kd(490)_Profiler'])
                    x_fit4 = np.array([d_490['Secchi'].min(), d_490['Secchi'].max()])
                    ax2.plot(x_fit4, m4['Intercept'] + m4['Slope'] * x_fit4, 
                             '--', color='#39FF14', alpha=0.8, label=f'R²={m4["R²"]:.2f}')
            
            ax2.set_xlabel("Secchi Depth (m)", weight='bold')
            ax2.set_ylabel(r"$K_d(490)$ ($m^{-1}$)", weight='bold')
            ax2.set_title("Secchi Depth vs. Kd(490)", weight='bold')
            ax2.legend(loc='upper right', fontsize=9, framealpha=0.6)
            ax2.grid(True, ls=':', alpha=0.4)
            
            plt.tight_layout()
            st.pyplot(fig_secchi)
            
            if df_master['Secchi'].nunique() <= 1:
                st.warning("Regression line will appear once you have at least 2 stations with different Secchi depths.")
                
    # st.subheader("Generate L3 Ensemble Average Product")
    # ensemble_casts = st.multiselect("Select validated profiles to include in ensemble average:", options=st.session_state.comparison_table.index, default=list(st.session_state.comparison_table.index))
    
    # any_layers_defined = any(st.session_state.profile_specific_layers.get(cast) for cast in ensemble_casts)
    # disable_ensemble = (len(ensemble_casts) < 2) or not any_layers_defined
    
    # if len(ensemble_casts) < 2:
    #     st.info("Select at least two profiles above to generate an ensemble average.")
    # elif not any_layers_defined:
    #     st.warning("Please analyze at least one layer for one of the selected profiles to generate an ensemble product.")
    
    # if st.button("Generate & Download Ensemble Average Package", disabled=disable_ensemble):
    #     with st.spinner("Calculating ensemble average and building package..."):
    #         base_name = f"L3_Ensemble_Average_{len(ensemble_casts)}_profiles"
    #         zip_buffer_ens = io.BytesIO()
    #         with zipfile.ZipFile(zip_buffer_ens, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    #             create_full_report_zip(zip_file, base_name)
                
    #             min_depth_ens, max_depth_ens = float('inf'), float('-inf')
    #             for cast_id in ensemble_casts:
    #                 pressure = st.session_state.station_data[cast_id]['pressure']
    #                 if pressure.size > 0:
    #                     min_depth_ens = min(min_depth_ens, pressure.min())
    #                     max_depth_ens = max(max_depth_ens, pressure.max())
                
    #             ensemble_depth_grid = np.arange(np.ceil(min_depth_ens*10)/10, np.floor(max_depth_ens*10)/10 + 0.1, 0.1)
    #             interp_ed_list, interp_lu_list = [], []
    #             wavelengths = st.session_state.station_data[ensemble_casts[0]]['wavelengths']
                
    #             for cast_id in ensemble_casts:
    #                 cast_data = st.session_state.station_data[cast_id]
    #                 interp_ed = np.array([np.interp(ensemble_depth_grid, cast_data['pressure'], cast_data['Ed_data'][:, i]) for i in range(len(wavelengths))]).T
    #                 interp_lu = np.array([np.interp(ensemble_depth_grid, cast_data['pressure'], cast_data['Lu_data'][:, i]) for i in range(len(wavelengths))]).T
    #                 interp_ed_list.append(interp_ed)
    #                 interp_lu_list.append(interp_lu)
                
    #             ed_stack, lu_stack = np.stack(interp_ed_list, axis=0), np.stack(interp_lu_list, axis=0)
    #             ed_mean, ed_std = np.mean(ed_stack, axis=0), np.std(ed_stack, axis=0)
    #             lu_mean, lu_std = np.mean(lu_stack, axis=0), np.std(lu_stack, axis=0)
    #             wvl_cols = [f"{int(w)}nm" for w in wavelengths]
                
    #             df_ed = pd.DataFrame(columns=['Depth'] + [f'{c}_{s}' for c in wvl_cols for s in ['mean', 'std']]); df_ed['Depth'] = ensemble_depth_grid; df_ed[[f'{c}_mean' for c in wvl_cols]] = ed_mean; df_ed[[f'{c}_std' for c in wvl_cols]] = ed_std
    #             df_lu = pd.DataFrame(columns=['Depth'] + [f'{c}_{s}' for c in wvl_cols for s in ['mean', 'std']]); df_lu['Depth'] = ensemble_depth_grid; df_lu[[f'{c}_mean' for c in wvl_cols]] = lu_mean; df_lu[[f'{c}_std' for c in wvl_cols]] = lu_std
    #             with np.errstate(divide='ignore', invalid='ignore'): lu_ed_ratio_mean = np.where(ed_mean > 0, lu_mean / ed_mean, np.nan)
    #             df_lu_ed = pd.DataFrame(lu_ed_ratio_mean, columns=wvl_cols); df_lu_ed.insert(0, 'Depth', ensemble_depth_grid)
                
    #             k_list, rrs_list, all_results_dfs, all_kd_par_dfs = [], [], [], []
    #             processed_casts = [] 

    #             for cast_id in ensemble_casts:
    #                 cast_layers = st.session_state.profile_specific_layers.get(cast_id, {})
    #                 if not cast_layers:
    #                     st.warning(f"Skipping '{cast_id}' in ensemble derived products: No analysis layers defined.")
    #                     continue
                    
    #                 derived_products_loop = calculate_derived_products(
    #                     st.session_state.station_data[cast_id], cast_id, cast_layers
    #                 )
                    
    #                 if derived_products_loop:
    #                     rrs_list.append(derived_products_loop['rrs_df_export'].set_index('wavelength_nm'))
    #                     k_list.append(derived_products_loop['k_df_export'].set_index('wavelength_nm'))
    #                     all_results_dfs.append(derived_products_loop['results_df'])
    #                     all_kd_par_dfs.append(derived_products_loop['kd_par_df'])
    #                     processed_casts.append(cast_id)
                        
    #             ensemble_results_df = pd.concat(all_results_dfs).groupby('Layer').mean().reset_index()
    #             ensemble_kd_par_df = pd.concat(all_kd_par_dfs).groupby('Layer').mean().reset_index()
    #             df_k_ens = pd.concat(k_list).groupby(level=0).agg(['mean', 'std']); df_k_ens.columns = ['_'.join(col) for col in df_k_ens.columns]; df_k_ens.reset_index(inplace=True)
    #             df_rrs_ens = pd.concat(rrs_list).groupby(level=0).agg(['mean', 'std']); df_rrs_ens.columns = ['_'.join(col) for col in df_rrs_ens.columns]; df_rrs_ens.reset_index(inplace=True)

    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/Ed_averaged.csv", df_ed.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/Lu_averaged.csv", df_lu.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/in_water_Lu_Ed_ratio_averaged.csv", df_lu_ed.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/K_metrics_averaged.csv", df_k_ens.to_csv(index=False))
    #             zip_file.writestr(f"{base_name}/L3_Data_Averaged/Rrs_propagated_averaged.csv", df_rrs_ens.to_csv(index=False))
                
    #             ensemble_mixture_results = {}
    #             if st.session_state.get('wm_vertices'):
    #                 for cast_id in processed_casts:
    #                     ens_cast_data = st.session_state.station_data[cast_id]
    #                     ens_phys_props = calculate_physical_properties(ens_cast_data, st.session_state.lon, st.session_state.lat)
    #                     ens_mix_text, ens_mix_df = analyze_mixture(ens_phys_props['Salinity'], ens_phys_props['Temperature'], st.session_state.wm_vertices, st.session_state.wm_names)
    #                     ensemble_mixture_results[cast_id] = ens_mix_text
    #                     ens_mix_profile_df = pd.concat([ens_phys_props['Depth'], ens_mix_df], axis=1).dropna()
    #                     zip_file.writestr(f"{base_name}/L3_Data_Individual_Mixture_Profiles/{cast_id}_mixture_data.csv", ens_mix_profile_df.to_csv(index=False))

    #             report_text_ens = generate_summary_report(
    #                 processed_casts, 
    #                 st.session_state.profile_specific_layers, 
    #                 ensemble_results_df, 
    #                 ensemble_kd_par_df, 
    #                 ensemble_mixture_results,
    #                 empirical_model=st.session_state.get('empirical_model_params'),
    #                 external_probe_filename=st.session_state.get('external_probe_filename'),
    #                 comparison_kd_df=st.session_state.get('comparison_kd_df')
    #             )
    #             zip_file.writestr(f"{base_name}/summary_report.txt", report_text_ens)
            
    #         st.download_button(label="Download Ensemble Package", data=zip_buffer_ens.getvalue(), file_name=f"{base_name}.zip", mime="application/zip", key="download_ensemble")


    
    # ------------------ TAB 8: SPECTRAL CONVOLUTION ------------------
    with tab_convolution:
        st.header("Spectral Convolution for Satellite Matching")
        
        with st.expander("Load Historical Master Rrs Data"):
            st.file_uploader("Upload Master Rrs Data (CSV)", type="csv", key="master_rrs_loader", on_change=_handle_master_rrs_upload)
        st.divider()
    
        srf_data = load_srf_from_folder()
        derived_products = st.session_state.get('derived_products')
    
        if derived_products is None or 'rrs_df_export' not in derived_products:
            st.warning("Analyze at least one layer in 'Core Optical Analysis' to generate Rrs data.")
        elif srf_data:
            c1, c2, c3 = st.columns(3)
            selected_sensor = c1.selectbox("Select Sensor:", list(srf_data.keys()))
            df_srf = srf_data[selected_sensor]
            srf_wl_col = next((col for col in df_srf.columns if 'wave' in col.lower()), None)
            
            rrs_options = [col for col in derived_products['rrs_df_export'].columns if 'rrs' in col]
            selected_rrs_source = c2.selectbox("Select Rrs source:", rrs_options)
            
            srf_bands = [col for col in df_srf.columns if col != srf_wl_col]
            selected_bands = c3.multiselect("Select bands:", srf_bands, default=srf_bands[:16])
    
            if st.button("Perform Convolution", use_container_width=True):
                with st.spinner("Processing..."):
                    ed_hyper = st.session_state.get('new_es_median', station_data.get('Es'))
                    if ed_hyper is not None:
                        lw_hyper = derived_products['rrs_df_export'][selected_rrs_source] * ed_hyper
                        prof_wl = station_data['wavelengths']
                        srf_wl = df_srf[srf_wl_col].values
                        
                        results = []
                        for band in selected_bands:
                            conv_val = perform_convolution(prof_wl, lw_hyper, ed_hyper, srf_wl, df_srf[band].values)
                            results.append({'Band': band, 'Convolved_Rrs_sr-1': conv_val})
                        
                        st.session_state.convolved_rrs_df = pd.DataFrame(results)
                        st.session_state.convolved_sensor_name = selected_sensor
                        st.rerun()

            # RESULTS 
            if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
                st.markdown("---")
                df_res = st.session_state.convolved_rrs_df
                res_c1, res_c2 = st.columns([1, 2])
                
                with res_c1:
                    st.write(f"**Sensor:** {st.session_state.convolved_sensor_name}")
                    st.dataframe(df_res.style.format({'Convolved_Rrs_sr-1': '{:.5f}'}))
                    if st.button("Save to Multispectral Master"):
                        new_row = df_res.set_index('Band').T
                        new_row.insert(0, 'Station_ID', st.session_state.selected_station)
                        new_row.insert(1, 'Sensor', st.session_state.convolved_sensor_name)
                        st.session_state.master_multi_rrs_df = pd.concat([st.session_state.master_multi_rrs_df, new_row], ignore_index=True)
                        st.toast("Saved!")

                with res_c2:
                    fig_conv, ax = plt.subplots(figsize=(6,4))
                    # Plot Hyper
                    ax.plot(station_data['wavelengths'], derived_products['rrs_df_export'][selected_rrs_source], color='white', alpha=0.5, label='Hyperspectral')
                    # Plot convolved points (Step plot style approximation)
                    df_s_active = srf_data[st.session_state.convolved_sensor_name]
                    centers = []
                    for b in df_res['Band']:
                        c = np.sum(df_s_active[srf_wl_col] * df_s_active[b]) / np.sum(df_s_active[b])
                        centers.append(c)
                    ax.plot(centers, df_res['Convolved_Rrs_sr-1'], color='red', marker='o', alpha=0.37, lw=2, label='Convolved (Bands)')
                    ax.set_ylim(0, 0.02); ax.set_xlim(380, 700); ax.legend(); ax.grid(True, ls=':')
                    st.pyplot(fig_conv)
                    plt.close(fig_conv)
                
                
                
                
                
    
    # ------------------ TAB 9: MASTER DATA & REPORTS ------------------
    with tab_report:
        st.header("Campaign Management & L3 Data Products")
        
        with st.expander("Step 0: Load Historical Master Tables", expanded=False):
            st.info("Suba seus arquivos Master (.csv) para mesclar dados de diferentes dias ou campanhas.")
            
            c_l1, c_l2 = st.columns(2)
            with c_l1:
                st.file_uploader("Load Vertical Radiometry", type="csv", key="up_rad_m", on_change=_handle_rad_upload)
                st.file_uploader("Load Summary Hyper Rrs", type="csv", key="up_hyper_m", on_change=_handle_hyper_upload)
                st.file_uploader("Load Summary Secchi & Kd", type="csv", key="up_secchi_central", on_change=_handle_master_secchi_upload)
            
            with c_l2:
                st.file_uploader("Load Vertical Ancillary", type="csv", key="up_anc_m", on_change=_handle_anc_upload)
                st.file_uploader("Load Summary Convolved Rrs", type="csv", key="up_multi_m", on_change=_handle_multi_upload)
                st.file_uploader("Load Master Bio-Optical Model", type="csv", key="up_model_central", on_change=_handle_model_upload)

        st.divider()

        st.subheader("1. Global Campaign Database Status")
        
        # Summary metrics of the cumulative database

        n_rad = st.session_state.master_vertical_radiometry['Station_ID'].nunique() if not st.session_state.master_vertical_radiometry.empty else 0
        n_sec = st.session_state.master_kd_secchi_df['Station_ID'].nunique() if not st.session_state.master_kd_secchi_df.empty else 0
        n_hyp = st.session_state.master_hyper_rrs_df['Station_ID'].nunique() if not st.session_state.master_hyper_rrs_df.empty else 0
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Stations (Radiometry)", n_rad)
        m_col2.metric("Stations (Secchi/Kd)", n_sec)
        m_col3.metric("Stations (Rrs Spectra)", n_hyp)

        col_agg, col_dl = st.columns(2)
        
        if col_agg.button("Snapshot Current Station to Masters", use_container_width=True, type="primary"):
            
            # Triggers the aggregation of all current results into the master tables
            if derived_products: 
                aggregate_to_vertical_masters(st.session_state.selected_station, station_data, derived_products)
                st.success(f"'{st.session_state.selected_station}' Snapshot saved in memory")
                st.rerun() 
            else:
                st.error("Analyze layers in 'Core Optical Analysis' first.")
        
        if col_dl.button("Export Full Campaign ZIP", use_container_width=True):
            
            # Packages all master tables into one ZIP for archival
            if st.session_state.master_vertical_radiometry.empty:
                st.error("Database is empty.")
            else:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("1_Master_Vertical_Radiometry.csv", st.session_state.master_vertical_radiometry.to_csv(index=False))
                    zf.writestr("2_Master_Vertical_Ancillary.csv", st.session_state.master_vertical_ancillary.to_csv(index=False))
                    zf.writestr("3_Summary_Hyper_Rrs.csv", st.session_state.master_hyper_rrs_df.to_csv(index=False))
                    zf.writestr("4_Summary_Convolved_Rrs.csv", st.session_state.master_multi_rrs_df.to_csv(index=False))
                    zf.writestr("5_Summary_Secchi_Kd.csv", st.session_state.master_kd_secchi_df.to_csv(index=False))
                    if not st.session_state.empirical_model_data.empty:
                        zf.writestr("6_Master_BioOptical_Model.csv", st.session_state.empirical_model_data.to_csv(index=False))
                
                st.download_button("Download ZIP Package", buf.getvalue(), "Campaign_Full_Database.zip", use_container_width=True)

        st.divider()

        # L3 STATION PACKAGE ---
        st.subheader(f"2. Individual L3 Package: {st.session_state.selected_station}")
        st.info("Build Detailed L3 Package (ZIP)")
        
        # Generates a comprehensive per-station folder with all physics, optics, and plots
        if st.button("Build L3 Station Package", use_container_width=True, key="btn_l3_pro"):
            station_id = st.session_state.selected_station
            base = f"C_{station_id}_L3"
            l3_buf = io.BytesIO()
            
            s_data = st.session_state.station_data[station_id]
            w_wavelengths = s_data['wavelengths']
            wvl_cols = [f"{int(w)}nm" for w in w_wavelengths]
            
            with zipfile.ZipFile(l3_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                
                # Export Physics
                # Profiler
                df_phys_prof = calculate_physical_properties(s_data, st.session_state.lon, st.session_state.lat)
                if st.session_state.mixture_df_profiler is not None:
                    df_phys_prof = pd.concat([df_phys_prof.reset_index(drop=True), st.session_state.mixture_df_profiler.reset_index(drop=True)], axis=1)
                zf.writestr(f"{base}/1_Profiler_Profiler/PHYS_Data_Profiler.csv", df_phys_prof.to_csv(index=False))

                df_opt_prof = pd.DataFrame({'Depth_m': s_data['pressure']})
                with np.errstate(divide='ignore', invalid='ignore'):
                    lued = np.where(s_data['Ed_data'] > 0, s_data['Lu_data'] / s_data['Ed_data'], np.nan)
                for i, wl in enumerate(wvl_cols):
                    df_opt_prof[f'Ed_{wl}'] = s_data['Ed_data'][:, i]
                    df_opt_prof[f'Lu_{wl}'] = s_data['Lu_data'][:, i]
                    df_opt_prof[f'LuEd_{wl}'] = lued[:, i]
                zf.writestr(f"{base}/1_Profiler_Profiler/OPTICAL_Profiles_Profiler.csv", df_opt_prof.to_csv(index=False))

                # External Probe
                if st.session_state.external_probe_df is not None:
                    st_col = st.session_state.p1_st
                    match = find_best_match(station_id, st.session_state.external_probe_df[st_col].dropna().unique())
                    if match:
                        df_ext_st = st.session_state.external_probe_df[st.session_state.external_probe_df[st_col].astype(str) == str(match)].copy()
                        zf.writestr(f"{base}/2_External_Probe/PHYS_Data_External_Probe_Original.csv", df_ext_st.to_csv(index=False))

                # Export Raw and derived Optics
                if derived_products:
                    zf.writestr(f"{base}/3_Derived_Optical_Products/AOP_Summary_Metrics.csv", derived_products['results_df'].to_csv(index=False))
                    zf.writestr(f"{base}/3_Derived_Optical_Products/Rrs_PROFILER_Propagated.csv", derived_products['rrs_df_export'].to_csv(index=False))
                    # Adiciona a tabela de comparação de Kd (Tab 6) se existir
                    if st.session_state.get('comparison_kd_df') is not None:
                        zf.writestr(f"{base}/3_Derived_Optical_Products/Kd_Validation_Table.csv", st.session_state.comparison_kd_df.to_csv())
                        
                    if 'convolved_rrs_df' in st.session_state and st.session_state.convolved_rrs_df is not None:
                        zf.writestr(f"{base}/3_Derived_Optical_Products/Rrs_SATELLITE_{st.session_state.convolved_sensor_name}_Convolved.csv", st.session_state.convolved_rrs_df.to_csv(index=False))

                # VALIDATION METRICS 
                if st.session_state.get('comparison_table') is not None:                    
                    zf.writestr(f"{base}/4_Validation_and_Plots/CTD_Comparison_Metrics_ALL_CASTS.csv", st.session_state.comparison_table.to_csv())

                # Save all current session plots as high-res PNGs
                plots_to_save = {
                    "1_Stratification": st.session_state.get('fig_n2'),
                    "2_Rrs_Spectra": st.session_state.get('fig_rrs'),
                    "3_Log_Profiles": st.session_state.get('fig_log'),
                    "4_K_Spectra": st.session_state.get('fig_k'),
                    "5_TS_Diagram": st.session_state.get('fig_ts'),
                    "6_Water_Mass_Profiler": st.session_state.get('fig_vp'),
                    "7_Water_Mass_External": st.session_state.get('fig_ve'),
                    "8_CTD_Validation": st.session_state.get('fig_comp')
                }
                for name, fig in plots_to_save.items():
                    if fig: zf.writestr(f"{base}/4_Validation_and_Plots/Plots_PNG/{name}.png", fig_to_buffer(fig).getvalue())

                # Generate the final Summary Report text
                report_txt = generate_summary_report(
                    station_id, current_station_layers, 
                    st.session_state.results_df, st.session_state.kd_par_df, 
                    wm_profiler=st.session_state.mixture_df_profiler, 
                    wm_external=st.session_state.mixture_df_external
                )
                zf.writestr(f"{base}/Summary_Report.txt", report_txt)
                
            st.download_button(f"Download L3 Package for: {station_id}", l3_buf.getvalue(), f"{base}.zip", mime="application/zip", use_container_width=True)
            
    
    # ------------------ TAB 10: FORMULAS & METHODS ------------------ Needs work
    with tab_formulas:
        st.header("Formulas & Methods"); st.subheader("1. Apparent Optical Properties (Kd & klu)")
        st.markdown(r"""The diffuse attenuation coefficients for downwelling irradiance ($K_d$) and upwelling radiance ($K_u$) describe how rapidly light is attenuated with depth. They are derived from the Beer-Lambert Law, which states:$$ I(z) = I(0) \cdot e^{-K \cdot z} $$Where $I(z)$ is the intensity at depth $z$, and $I(0)$ is the intensity at the surface. By taking the natural logarithm, this becomes a linear relationship:$$ \ln(I(z)) = \ln(I(0)) - K \cdot z $$This app calculates $K_d$ and $K_u$ for each wavelength ($\lambda$) within a user-selected depth layer by performing a linear regression (using 'np.polyfit') on the natural log of the radiometric data versus depth. The coefficient $K$ is the negative of the slope of this regression. **Reference:** Mobley, C. D. (1994). *Light and Water: Radiative Transfer in Natural Waters*. Academic Press.""")
        st.subheader("2. Remote Sensing Reflectance ($R_{rs}$)"); st.markdown(f"""Remote sensing reflectance is the ratio of water-leaving radiance ($L_w$) to the downwelling irradiance ($E_d$) just above the surface: $$ R_{{rs}}(\lambda) = \\frac{{L_w(\lambda)}}{{E_d(\lambda, 0^+)}} $$        The water-leaving radiance ($L_w$) is estimated from the upwelling radiance just below the surface ($L_u(\lambda, 0^-)$) by accounting for the transmission of light across the air-sea interface. This app uses a standard approximation where the transmission factor is **{LW_TRANSMISSION_FACTOR}**:        $$ L_w \\approx {LW_TRANSMISSION_FACTOR} \\cdot L_u(0^-) $$        This coefficient approximates the term $\\frac{{1- \\rho(\\theta, n)}}{{n^2}}$, where $\\rho$ is the Fresnel reflectance and $n$ is the refractive index of water.         **Propagated $R_{{rs}}$** is calculated by using the layer-specific $K_d$ and $K_u$ to propagate the radiometric values from the top of the selected layer ($z$) back to the surface: $$ E_d(0^+) = E_d(z) \\cdot e^{{K_d \\cdot z}} $$        $$ L_u(0^-) = L_u(z) \\cdot e^{{K_u \\cdot z}} $$        These surface-equivalent values are then used to calculate the final propagated $R_{{rs}}$.
        """)
        st.subheader("3. Stratification (Brunt-Väisälä Frequency)"); st.markdown(r"""The stratification of the water column is represented by the Brunt-Väisälä frequency squared, $N^2$. A high $N^2$ value indicates strong stratification. This value is calculated using the Gibbs SeaWater (GSW) Oceanographic Toolbox, which implements the Thermodynamic Equation of Seawater 2010 (TEOS-10). **Reference:** McDougall, T. J., & Barker, P. M. (2011). *Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox*. SCOR/IAPSO Working Group 127, ISBN 978-0-646-55606-5.""")
        st.subheader("4. Photosynthetically Available Radiation (PAR & Kd(PAR))"); st.markdown(r"""Photosynthetically Available Radiation (PAR) is the flux of photons between 400 and 700 nm. It is calculated by converting the energy of the downwelling irradiance $E_d(\lambda)$ from energy units ($\mu W/cm^2/nm$) to quanta units ($\mu mol\ photons/m^2/s/nm$) and then integrating over the wavelength range. This app follows the protocol from the ProSoft manual:$$ PAR(z) = \int_{{400nm}}^{{700nm}} E_q(\lambda, z) d\lambda $$Where $E_q$ is the spectral irradiance in quanta units. The attenuation coefficient for PAR, $K_d(PAR)$, is then calculated for each user-defined layer by performing a log-linear regression on the PAR profile versus depth, identical to the method used for spectral $K_d(\lambda)$.""")

