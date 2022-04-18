# multimodal_sentiment_analysis

## IEMOCAP

Download dataset.
Extract the folder named `iemocap`.
Set the environment variable `DATADIR` to the folder containing `iemocap`.

```bash
export DATADIR="<data_dir>"
python3 src/data_loaders/iemocap.py
```

## CMU_MOSI

`Language = English`
`Emotions positive or negative over [-3,3] (regression)`
Download dataset :

```bash
cd data 
bash get_CMU_MOSI.sh
```

You can then use the `src.data_loaders.cmu_mosi.py` file to import the dataset 👍. Please refer to the readme `src.data_loaders.README.md` to learn more.

## AESDD

Acted Emotional Speech Dynamic Database, for more information please refer to the [main site](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/) for this dataset.
`Language = Greek`
`5 emotions : ['anger', 'disgust', 'fear', 'happiness', 'sadness']`
Donwload dataset :

```bash
cd data
bash get_aesdd.sh
```
