# deep-karaoke-maker
A program to separate a given track into vocal and instrumental stems, based on PyTorch.
Based on [1], who have their own implementation (in MATLAB) [here](https://github.com/jaidevd/deep_kareoke_source_separation).

## Running the Code
1. Get the [MedleyDB](http://medleydb.weebly.com/downloads.html) dataset, the dataset used to train the model is extracted from it. You need to ask them for access to download, they are very responsive by email. Note that the DB is heavy (41GB compressed).
1. Update the file `medleydb_deepkaraoke.json` with the path to where you extracted MedleyDB. If you use the MedleyDB Sample, make sure to erase all entries from the json file and keep only the ones you have in the sample.
1. Run `python spectrum_helper/__init__.py` to generate the dataset.
1. Run `python deep_karaoke.py train` to train the network. (run with `--help` to see options)

## References
[1] 
    Simpson A.J.R., Roma G., Plumbley M.D. (2015) Deep Karaoke: Extracting Vocals from Musical Mixtures Using a Convolutional Deep Neural Network. In: Vincent E., Yeredor A., Koldovský Z., Tichavský P. (eds) Latent Variable Analysis and Signal Separation. LVA/ICA 2015. Lecture Notes in Computer Science, vol 9237. Springer, Cham.
