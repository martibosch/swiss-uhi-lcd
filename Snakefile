from os import path

PROJECT_NAME = "swiss-uhi-lcd"
CODE_DIR = "swiss_uhi_lcd"
PYTHON_VERSION = "3.13"

NOTEBOOKS_DIR = "notebooks"
NOTEBOOKS_OUTPUT_DIR = path.join(NOTEBOOKS_DIR, "output")

DATA_DIR = "data"
DATA_RAW_DIR = path.join(DATA_DIR, "raw")
DATA_INTERIM_DIR = path.join(DATA_DIR, "interim")
DATA_PROCESSED_DIR = path.join(DATA_DIR, "processed")

MODELS_DIR = "models"

FIGURES_DIR = "reports/figures"


# 1. get urban extents -----------------------------------------------------------------
NOMINATIM_QUERY_DICT = {
    "bern": "Kanton Bern",
    "lausanne": "Canton de Vaud",
    "neuchatel": "Canton de Neuchatel",
    "zurich": "Kanton Zürich",
}


rule spatial_extent:
    input:
        notebook=path.join(NOTEBOOKS_DIR, "get-spatial-extent.ipynb"),
    params:
        nominatim_query=lambda wc: NOMINATIM_QUERY_DICT[wc.slug],
    output:
        spatial_extent=path.join(DATA_PROCESSED_DIR, "{slug}-extent.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "get-spatial-extent-{slug}.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p nominatim_query '{params.nominatim_query}'"
        " -p dst_filepath {output.spatial_extent}"


# rule spatial_extents:
#     input:
#         expand(
#             path.join(DATA_PROCESSED_DIR, "{slug}-extent.gpkg"),
#             slug=NOMINATIM_QUERY_DICT.keys(),
#         ),


# 2. get meteo data --------------------------------------------------------------------
# reference data to get study period
TS_DF_FILEPATH = path.join(DATA_RAW_DIR, "parallel-2025-int.csv")


# 2.1 automated weather stations (AWS)
rule aws_meteo_data:
    input:
        spatial_extent=rules.spatial_extent.output.spatial_extent,
        notebook=path.join(NOTEBOOKS_DIR, "get-aws-data.ipynb"),
    output:
        ts_cube=path.join(DATA_INTERIM_DIR, "{slug}-aws-ts-cube.nc"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "get-aws-data-{slug}.ipynb"),
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
        spatial_extent=lambda wc: path.join(
            DATA_PROCESSED_DIR, f"{SLUG_CITY_DICT[wc.slug]}-extent.gpkg"
        ),
        ts_df=path.join(DATA_RAW_DIR, "{slug}-summer-2025-pcd.csv"),
        stations_gdf=path.join(DATA_RAW_DIR, "{slug}-metadata-2025.csv"),
        notebook=path.join(NOTEBOOKS_DIR, "get-lcd-data.ipynb"),
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "{slug}-lcd-ts-df.csv"),
        stations_gdf=path.join(DATA_INTERIM_DIR, "{slug}-lcd-stations.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "get-lcd-data-{slug}.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p spatial_extent_filepath {input.spatial_extent}"
        " -p ts_df_filepath {input.ts_df}"
        " -p stations_gdf_filepath '{input.stations_gdf}'"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


rule zurich_awel_lcd_meteo_data:
    input:
        spatial_extent=path.join(DATA_PROCESSED_DIR, "zurich-extent.gpkg"),
        notebook=path.join(NOTEBOOKS_DIR, "get-zurich-awel-data.ipynb"),
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "zurich-awel-lcd-ts-df.csv"),
        stations_gdf=path.join(DATA_INTERIM_DIR, "zurich-awel-lcd-stations.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "get-lcd-data-zurich-awel.ipynb"),
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
#             path.join(DATA_INTERIM_DIR, "{slug}-lcd-ts-df.csv"),
#             slug=NOMINATIM_QUERY_DICT.keys(),
#         ),
#         stations_gdf=expand(
#             path.join(DATA_INTERIM_DIR, "{slug}-lcd-stations.gpkg"),
#             slug=NOMINATIM_QUERY_DICT.keys(),
#         ),


# 3. bias correction -------------------------------------------------------------------
rule train_bias_correction:
    input:
        ts_df=path.join(DATA_RAW_DIR, "parallel-2025-int.csv"),
        notebook=path.join(NOTEBOOKS_DIR, "train-bias-correction.ipynb"),
    params:
        models_dir=MODELS_DIR,
    output:
        station_model_dict=path.join(DATA_PROCESSED_DIR, "station-model-dict.json"),
        station_scale_dict=path.join(DATA_PROCESSED_DIR, "station-scale-dict.json"),
        bland_altman_plot=path.join(FIGURES_DIR, "bland-altman-plot.pdf"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "train-bias-correction.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p ts_df_filepath {input.ts_df}"
        " -p models_dir {params.models_dir}"
        " -p dst_station_model_dict_filepath {output.station_model_dict}"
        " -p dst_station_scale_dict_filepath {output.station_scale_dict}"
        " -p dst_fig_bland_altman_plot_filepath {output.bland_altman_plot}"


STATION_MODEL_DICT = {
    "bern": "Abilium",
    "lausanne": "Koalasense",
    "neuchatel": "Onset_big",
    "zurich-awel": "Decentlab",
    "zurich-ugz": "Barani",
}


rule apply_bias_correction:
    input:
        aws_ts_cube=lambda wc: path.join(
            DATA_INTERIM_DIR, f"{SLUG_CITY_DICT[wc.slug]}-aws-ts-cube.nc"
        ),
        lcd_ts_df=path.join(DATA_INTERIM_DIR, "{slug}-lcd-ts-df.csv"),
        lcd_stations_gdf=path.join(DATA_INTERIM_DIR, "{slug}-lcd-stations.gpkg"),
        station_model_dict=rules.train_bias_correction.output.station_model_dict,
        station_scale_dict=rules.train_bias_correction.output.station_scale_dict,
        notebook=path.join(NOTEBOOKS_DIR, "apply-bias-correction.ipynb"),
    params:
        models_dir=MODELS_DIR,
        station_model=lambda wc: STATION_MODEL_DICT[wc.slug],
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "{slug}-cor-ts-df.csv"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "apply-bias-correction-{slug}.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p aws_ts_cube_filepath {input.aws_ts_cube}"
        " -p lcd_ts_df_filepath {input.lcd_ts_df}"
        " -p lcd_stations_gdf_filepath {input.lcd_stations_gdf}"
        " -p models_dir {params.models_dir}"
        " -p station_model '{params.station_model}'"
        " -p station_model_dict_filepath {input.station_model_dict}"
        " -p station_scale_dict_filepath {input.station_scale_dict}"
        " -p dst_ts_df_filepath {output.ts_df}"


rule apply_bias_corrections:
    input:
        expand(
            path.join(DATA_INTERIM_DIR, "{slug}-cor-ts-df.csv"),
            slug=LCD_SLUGS,
        ),
