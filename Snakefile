from os import path

PROJECT_NAME = "swiss-lcd-heatwaves"
CODE_DIR = "swiss_lcd_heatwaves"
PYTHON_VERSION = "3.13"

NOTEBOOKS_DIR = "notebooks"
NOTEBOOKS_OUTPUT_DIR = path.join(NOTEBOOKS_DIR, "output")

DATA_DIR = "data"
DATA_RAW_DIR = path.join(DATA_DIR, "raw")
DATA_INTERIM_DIR = path.join(DATA_DIR, "interim")
DATA_PROCESSED_DIR = path.join(DATA_DIR, "processed")

MODELS_DIR = "models"


# 0. conda/mamba environment -----------------------------------------------------------
rule create_environment:
    shell:
        "mamba env create -f environment.yml"


rule register_ipykernel:
    shell:
        "python -m ipykernel install --user --name {PROJECT_NAME} --display-name"
        " 'Python ({PROJECT_NAME})'"


# 1. get urban extents -----------------------------------------------------------------
NOMINATIM_QUERY_DICT = {
    "bern": "Kanton Bern",
    # "lausanne": "Canton de Vaud",
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
UCB_DIR = path.join(DATA_RAW_DIR, "ucb")
LCD_INPUT_DICT = {
    "bern": {
        "ts_df": path.join(UCB_DIR, "ts-df-jja-2023-2024.csv"),
        "stations_gdf": path.join(UCB_DIR, "metadata_network_2023_2024.csv"),
    },
    "zurich": {"ts_df": [], "stations_gdf": []},
}


rule lcd_meteo_data:
    input:
        spatial_extent=rules.spatial_extent.output.spatial_extent,
        ts_df=lambda wc: LCD_INPUT_DICT[wc.slug]["ts_df"],
        stations_gdf=lambda wc: LCD_INPUT_DICT[wc.slug]["stations_gdf"],
        notebook=path.join(NOTEBOOKS_DIR, "get-{slug}-lcd-data.ipynb"),
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "{slug}-lcd-ts-df.csv"),
        stations_gdf=path.join(DATA_INTERIM_DIR, "{slug}-lcd-stations.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "get-lcd-data-{slug}.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p spatial_extent_filepath {input.spatial_extent}"
        " -p ts_df_filepath '{input.ts_df}'"
        " -p stations_gdf_filepath '{input.stations_gdf}'"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


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


# 3. radiative bias correction ---------------------------------------------------------
rule radiative_bias:
    input:
        ts_df=path.join(DATA_RAW_DIR, "parallel-2025-int.csv"),
        notebook=path.join(NOTEBOOKS_DIR, "radiative-bias.ipynb"),
    params:
        models_dir=MODELS_DIR,
    output:
        station_model_dict=path.join(DATA_PROCESSED_DIR, "station-model-dict.json"),
        station_scale_dict=path.join(DATA_PROCESSED_DIR, "station-scale-dict.json"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "radiative-bias.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p ts_df_filepath {input.ts_df}"
        " -p models_dir {params.models_dir}"
        " -p dst_station_model_dict_filepath {output.station_model_dict}"
        " -p dst_station_scale_dict_filepath {output.station_scale_dict}"


STATION_MODEL_DICT = {
    "bern": "Abilum_2",
    # "lausanne": "Koalasense",
    "zurich": "Decentlab",
}


rule bias_correction:
    input:
        aws_ts_cube=rules.aws_meteo_data.output.ts_cube,
        lcd_ts_df=rules.lcd_meteo_data.output.ts_df,
        lcd_stations_gdf=rules.lcd_meteo_data.output.stations_gdf,
        station_model_dict=rules.radiative_bias.output.station_model_dict,
        station_scale_dict=rules.radiative_bias.output.station_scale_dict,
        notebook=path.join(NOTEBOOKS_DIR, "bias-correction.ipynb"),
    params:
        models_dir=MODELS_DIR,
        station_model=lambda wc: STATION_MODEL_DICT[wc.slug],
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "{slug}-cor-ts-df.csv"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "bias-correction-{slug}.ipynb"),
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


rule bias_correctoins:
    input:
        expand(
            path.join(DATA_INTERIM_DIR, "{slug}-cor-ts-df.csv"),
            slug=NOMINATIM_QUERY_DICT.keys(),
        ),
