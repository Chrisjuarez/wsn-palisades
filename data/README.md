# data/

This directory is **gitignored** and gets populated by `scripts/fetch_data.py`.

Expected layout after fetch:

```
data/
├── palisadesoutput.dtm.tif       # bare-earth DTM (~1.8 GB)
├── palisadesoutput.dsm.tif       # digital surface model (~1.2 GB)
└── palisadesCHM.tif              # canopy height model (~1.5 GB)
```

These rasters are derived from public LiDAR returns and are hosted on S3. Set `WSN_DATA_BUCKET` in `.env` and run `python scripts/fetch_data.py` to download.

If you want to bring your own data, drop matching `.tif` files here with the names above.
