# Sentinel-2 Regional Dataset Downloader

To reproduce results in Appendix D of our paper or to download regional Sentinel-2 image datasets in S2-100k-style formatting, we provide scripts to query and use regional Sentinel-2 image datasets. These scripts download multi-band (13 channels) Sentinel-2 imagery patches (256×256 pixels at 10m resolution) for any of 50+ predefined regions including continents, countries, and US states.


## Installation

Install and authenticate Earth Engine
```
pip install earthengine-api
earthengine authenticate
```

```
pip install earthengine-api rasterio geopandas shapely pandas numpy tqdm
```

## Usage 

You can list the available regions with `python sentinel2_download.py --list
`
To download Sentinel-2 imagery in a predefined region, run
`
python sentinel2_download.py colorado \
    --samples 5000 \
    --workers 5 \
    --cloud 20
`

Full command
```
python sentinel2_download.py united_states \
    --output /data/sentinel2_usa \
    --samples 50000 \
    --workers 5 \
    --batch 100 \
    --cloud 25 \
    --strategy stratified \
    --start-date 2023-01-01 \
    --end-date 2024-12-31
```

Parameters:
```
region: Region name (use --list to see options)

--samples: Number of patches to download (default: region-specific)

--workers: Parallel workers (1-8, limited by Earth Engine)

--cloud: Maximum cloud coverage % (default: region-specific)

--strategy: Sampling strategy (random or stratified)

--start-date/--end-date: Date range for imagery
```

## Validation

Run `python setup_and_utils.py validate --dataset-dir /data/sentinel2_colorado`

This checks directory structure, integrity of the generated `index.csv`, image dimensions, band count, and missing/corrupted files. 

You can visualize the spatial coverage of the generated dataset with `python plot_spatial_coverage.py /data/sentinel2_colorado colorado \
    --style hexbin \
    --output colorado_coverage.png
    `

To generate band-statistics for normalization, run 
`python setup_and_utils.py statistics --dataset-dir /data/sentinel2_colorado
`
