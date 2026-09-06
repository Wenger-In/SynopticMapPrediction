# Data layout

Place data under the configured `data_root` using the layout below, or override
individual paths in `config/local.json`.

```text
data/
├── GONG/{fits,harmonics,mrzqs/fits}/
├── OMNI/omni_27_av.dat
├── Sunspot/{sn.mat,sn_sm.mat,sn_interp.mat,SN_*_tot_V2.0*.csv}
├── WSO/{download/dat,download/txt,field,harmonics}/
├── WSO/gather_harmonic_coefficient.mat
└── transformer/{2002_2004_5min_data.csv,2005_data_5min_4day.csv}
```

## Upstream sources

- WSO synoptic maps: <http://wso.stanford.edu/#Synoptic>
- SILSO International Sunspot Number: <https://www.sidc.be/SILSO/datafiles>
- GONG magnetic-field products: <https://gong.nso.edu/data/magmap/>
- NASA OMNI data: <https://omniweb.gsfc.nasa.gov/>

The preprocessed `.mat`, `.fits`, and `.xlsx` files already committed here are
retained as compact workflow examples or derived products. A derived file does
not automatically inherit permission for unrestricted reuse from its source.

Configuration precedence is `config/default.json`, untracked
`config/local.json`, then the `SMP_*` environment variables.
