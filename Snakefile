from pathlib import Path

PROJECT_NAME = "swiss-uhi-lcd"
HF_USERNAME = "martibosch"
CODE_DIR = "swiss_uhi_lcd"
PYTHON_VERSION = "3.13"

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_OUTPUT_DIR = NOTEBOOKS_DIR / "output"

DATA_DIR = Path("data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path("models")

FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")

# ACHTUNG: we are kind of adding this unnecessary AGGLOM_SLUGS because we are using
# decentlab/barani as figure labels whereas the rules and other file names here use
# awel/ugz. TODO: fix it (maybe)?
AGGLOM_SLUGS = ["bern", "lausanne", "neuchatel", "zurich-decentlab", "zurich-barani"]


rule results:
    input:
        # TABLES_DIR / "agreement-metrics.csv",
        FIGURES_DIR / "bland-altman-plot.pdf",
        FIGURES_DIR / "heat-warnings-barplot-separate.pdf",
        FIGURES_DIR / "tn-barplot-separate.pdf",
        FIGURES_DIR / "tn-station-maps-single-scale.png",
        FIGURES_DIR / "t-diurnal-cycle-separate.pdf",


# 1. get urban extents -----------------------------------------------------------------
NOMINATIM_QUERY_DICT = {
    "bern": "Kanton Bern",
    "lausanne": "Canton de Vaud",
    "neuchatel": "Canton de Neuchatel",
    "zurich": "Kanton Zürich",
}


rule spatial_extent:
    input:
        notebook=NOTEBOOKS_DIR / "get-spatial-extent.ipynb",
    output:
        spatial_extent=DATA_PROCESSED_DIR / "{slug}-extent.gpkg",
        notebook=NOTEBOOKS_OUTPUT_DIR / "get-spatial-extent-{slug}.ipynb",
    params:
        nominatim_query=lambda wc: NOMINATIM_QUERY_DICT[wc.slug],
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p nominatim_query '{params.nominatim_query}'"
        " -p dst_filepath {output.spatial_extent}"


# rule spatial_extents:
#     input:
#         expand(
#             DATA_PROCESSED_DIR / "{slug}-extent.gpkg",
#             slug=NOMINATIM_QUERY_DICT.keys(),
#         ),


# 2. get meteo data --------------------------------------------------------------------
# reference data to get study period
TS_DF_FILEPATH = DATA_RAW_DIR / "parallel-2025-int.csv"


# 2.1 automated weather stations (AWS)
rule aws_meteo_data:
    input:
        spatial_extent=rules.spatial_extent.output.spatial_extent,
        notebook=NOTEBOOKS_DIR / "get-aws-data.ipynb",
    output:
        ts_cube=DATA_INTERIM_DIR / "{slug}-aws-ts-cube.nc",
        notebook=NOTEBOOKS_OUTPUT_DIR / "get-aws-data-{slug}.ipynb",
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p spatial_extent_filepath {input.spatial_extent}"
        " -p dst_ts_cube_filepath {output.ts_cube}"


# 2.2 low-cost devices (LCD)
# expand slugs to consider the two networks in zurich
SLUG_CITY_DICT = {
    "bern": "bern",
    "lausanne": "lausanne",
    "neuchatel": "neuchatel",
    "zurich-awel": "zurich",
    "zurich-ugz": "zurich",
}
LCD_SLUGS = tuple(SLUG_CITY_DICT.keys())


rule lcd_meteo_data:
    input:
        spatial_extent=lambda wc: DATA_PROCESSED_DIR
        / f"{SLUG_CITY_DICT[wc.slug]}-extent.gpkg",
        ts_df=DATA_RAW_DIR / "{slug}-summer-2025-pcd.csv",
        stations_gdf=DATA_RAW_DIR / "{slug}-metadata-2025.csv",
        notebook=NOTEBOOKS_DIR / "get-lcd-data.ipynb",
    output:
        ts_df=DATA_INTERIM_DIR / "{slug}-lcd-ts-df.csv",
        stations_gdf=DATA_INTERIM_DIR / "{slug}-lcd-stations.gpkg",
        notebook=NOTEBOOKS_OUTPUT_DIR / "get-lcd-data-{slug}.ipynb",
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p spatial_extent_filepath {input.spatial_extent}"
        " -p ts_df_filepath {input.ts_df}"
        " -p stations_gdf_filepath '{input.stations_gdf}'"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


rule zurich_awel_lcd_meteo_data:
    input:
        spatial_extent=DATA_PROCESSED_DIR / "zurich-extent.gpkg",
        notebook=NOTEBOOKS_DIR / "get-zurich-awel-data.ipynb",
    output:
        ts_df=DATA_INTERIM_DIR / "zurich-awel-lcd-ts-df.csv",
        stations_gdf=DATA_INTERIM_DIR / "zurich-awel-lcd-stations.gpkg",
        notebook=NOTEBOOKS_OUTPUT_DIR / "get-lcd-data-zurich-awel.ipynb",
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p spatial_extent_filepath {input.spatial_extent}"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


# when several rules match, prefer the specific ones
ruleorder: zurich_awel_lcd_meteo_data > lcd_meteo_data


# rule lcd_meteo_data:
#     input:
#         ts_df=expand(
#             DATA_INTERIM_DIR / "{slug}-lcd-ts-df.csv",
#             slug=NOMINATIM_QUERY_DICT.keys(),
#         ),
#         stations_gdf=expand(
#             DATA_INTERIM_DIR / "{slug}-lcd-stations.gpkg",
#             slug=NOMINATIM_QUERY_DICT.keys(),
#         ),


# 3. bias correction -------------------------------------------------------------------
PARALLEL_TS_DF_FILEPATH = DATA_RAW_DIR / "parallel-2025-int.csv"


rule agreement_metrics:
    input:
        ts_df=DATA_RAW_DIR / "parallel-2025-int.csv",
        notebook=NOTEBOOKS_DIR / "agreement-metrics.ipynb",
    output:
        # agreement_table=TABLES_DIR / "agreement-metrics.csv",
        bland_altman_plot=FIGURES_DIR / "bland-altman-plot.pdf",
        notebook=NOTEBOOKS_OUTPUT_DIR / "agreement-metrics.ipynb",
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p ts_df_filepath {input.ts_df}"

        " -p dst_fig_bland_altman_plot_filepath {output.bland_altman_plot}"
        # " -p dst_agreement_table_filepath {output.agreement_table}"


rule train_bias_correction:
    input:
        ts_df=PARALLEL_TS_DF_FILEPATH,
        notebook=NOTEBOOKS_DIR / "train-bias-correction.ipynb",
    output:
        station_model_repo_dict=DATA_PROCESSED_DIR / "station-model-repo-dict.json",
        notebook=NOTEBOOKS_OUTPUT_DIR / "train-bias-correction.ipynb",
    params:
        hf_username=HF_USERNAME,
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p ts_df_filepath {input.ts_df}"
        " -p hf_username {params.hf_username}"
        " -p dst_station_model_repo_dict_filepath {output.station_model_repo_dict}"


STATION_MODEL_DICT = {
    "bern": "Abilium",
    "lausanne": "Koalasense",
    "neuchatel": "Onset_big",
    "zurich-awel": "Decentlab",
    "zurich-ugz": "Barani",
}


rule apply_bias_correction:
    input:
        aws_ts_cube=lambda wc: DATA_INTERIM_DIR
        / f"{SLUG_CITY_DICT[wc.slug]}-aws-ts-cube.nc",
        lcd_ts_df=DATA_INTERIM_DIR / "{slug}-lcd-ts-df.csv",
        lcd_stations_gdf=DATA_INTERIM_DIR / "{slug}-lcd-stations.gpkg",
        station_model_repo_dict=(
            rules.train_bias_correction.output.station_model_repo_dict
        ),
        parallel_ts_df=PARALLEL_TS_DF_FILEPATH,
        notebook=NOTEBOOKS_DIR / "apply-bias-correction.ipynb",
    output:
        ts_df=DATA_INTERIM_DIR / "{slug}-cor-ts-df.csv",
        notebook=NOTEBOOKS_OUTPUT_DIR / "apply-bias-correction-{slug}.ipynb",
    params:
        station_model=lambda wc: STATION_MODEL_DICT[wc.slug],
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p aws_ts_cube_filepath {input.aws_ts_cube}"
        " -p lcd_ts_df_filepath {input.lcd_ts_df}"
        " -p lcd_stations_gdf_filepath {input.lcd_stations_gdf}"
        " -p station_model '{params.station_model}'"
        " -p station_model_repo_dict_filepath {input.station_model_repo_dict}"
        " -p parallel_ts_df_filepath {input.parallel_ts_df}"
        " -p dst_ts_df_filepath {output.ts_df}"


rule apply_bias_corrections:
    input:
        expand(
            DATA_INTERIM_DIR / "{slug}-cor-ts-df.csv",
            slug=LCD_SLUGS,
        ),


# 4. heat indices ----------------------------------------------------------------------
CITY_SLUGS = list(dict.fromkeys(SLUG_CITY_DICT.values()))


rule heat_indices:
    input:
        aws_ts_cubes=expand(
            DATA_INTERIM_DIR / "{slug}-aws-ts-cube.nc",
            slug=CITY_SLUGS,
        ),
        lcd_ts_dfs=expand(
            DATA_INTERIM_DIR / "{slug}-lcd-ts-df.csv",
            slug=LCD_SLUGS,
        ),
        lcd_stations_gdfs=expand(
            DATA_INTERIM_DIR / "{slug}-lcd-stations.gpkg",
            slug=LCD_SLUGS,
        ),
        cor_ts_dfs=expand(
            DATA_INTERIM_DIR / "{slug}-cor-ts-df.csv",
            slug=LCD_SLUGS,
        ),
        notebook=NOTEBOOKS_DIR / "heat-indices.ipynb",
    output:
        heat_warnings_barplot=FIGURES_DIR / "heat-warnings-barplot-separate.pdf",
        tn_barplot=FIGURES_DIR / "tn-barplot-separate.pdf",
        tn_station_maps=expand(
            FIGURES_DIR / "tn-station-maps-separate-{slug}.png",
            slug=AGGLOM_SLUGS,
        ),
        tn_station_maps_single_scale=FIGURES_DIR / "tn-station-maps-single-scale.png",
        t_diurnal_cycle=FIGURES_DIR / "t-diurnal-cycle-separate.pdf",
        notebook=NOTEBOOKS_OUTPUT_DIR / "heat-indices.ipynb",
    shell:
        "papermill {input.notebook} {output.notebook}"
