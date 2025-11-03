# AlphaEarth Embeddings Extraction & Visualization

We provide scripts for extracting and analyzing Google DeepMind's AlphaEarth embeddings - 64-dimensional feature vectors derived from Sentinel-2 satellite imagery at 10-meter spatial resolution. 

1. Extraction Script (`alphaearth_extractor.py`): Retrieves AlphaEarth embeddings from Google Earth Engine for specified geographic coordinates with configurable spatial aggregation.
2. Visualization Script (`alphaearth_pca_viz.py`): Applies PCA dimensionality reduction and creates RGB-mapped visualizations of the embeddings.

This script uses the Google Cloud SDK and the Google Earth Engine API. Below, we detail a short setup guide.

## Google Earth Engine Setup

Create a Google Earth Engine account:

1. Visit https://earthengine.google.com/signup/ and Sign up with your Google account
2. Install Google Cloud SDK (required for authentication):

```
# macOS
brew install google-cloud-sdk

# Ubuntu/Debian
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli

# Or download from: https://cloud.google.com/sdk/docs/install   
```

3. Authenticate with `  earthengine authenticate`

## Installation
Install the EarthEngine API. The only new dependency needed outside of those listed in `requirements.txt` can be installed as follows.
```
pip install earthengine-api
```

AlphaEarth embeddings can be queried with geographic coordinates. To reproduce a dataset setup we use in Table 1, download the `index.csv` from SatCLIP's S2-100K dataset. This can be found here: https://huggingface.co/datasets/davanstrien/satclip/blob/main/index.csv

## Running the embedding extractor

```
python alphaearth_extractor.py \
    --csv /path/to/coordinates.csv \
    --output /path/to/output/dir \
    --year 2024 \
    --buffer 10 \
    --workers 10 \
    --limit 0 
```

Increasing workers may cause issues with the rate-limits. Always monitor your Google cloud console at https://console.cloud.google.com/ to monitor for errors and latency. We recommend setting the number of workers to 8 or below. 

The parameters are:

```
--csv: Path to input CSV with coordinates

--output: Base output directory (buffer size will be appended automatically)

--year: Year of AlphaEarth composite (2017-2024)

--buffer: Buffer radius in meters for spatial aggregation
10m = single pixel (no aggregation)
20m = ~12 pixels averaged
50m = ~78 pixels averaged
100m = ~314 pixels averaged

--workers: Number of parallel extraction threads (1-50, default: 10)

--limit: Number of points to process (0 for all, useful for testing)

--no-auto-dir: Disable automatic directory naming by buffer size
```

## Visualizing downloaded embeddings

You can plot the extracted embeddings with 
```
python alphaearth_pca_viz.py \
    /path/to/alphaearth_embeddings_buffer20m \
    --year 2024 \
    --output ./visualizations
```