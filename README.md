# Dataset Requirements

This project uses the [TFRecord format](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) to consume data in the training and evaluation process. Creating a TFRecord from raw image files is pretty straight forward and will be covered here.

*Note:* **This project includes a script for creating a TFRecord for WeedMap dataset**, but not other datasets. To create your own TFRecord script, use the one in this project as are reference. Please note that the fields in the record must be the same as defined in the script. Read below for details.

## Creating TFRecords for WeedMap

In order to download the WeedMap dataset, you must go to [website](https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap). After this, make sure to download `(Tile) RedEdge 5 channel multispectral images`. You should end up with a folder that will be in the structure

```
+ raw
 + 000
  + groundtruth
  + mask
  + tile
   + B
   + CIR
   + G
   + NDVI
   + NIR
   + R
   + RE
   + RGB
 + 001
 + 002
 + 003
 + 004
```

this default structure must be in data/RedEdge/raw folder. Finally, you can now run the conversion script provided in this project with

```
python src/main.py \
	--protocol ORIGINAL_PROTOCOL  # the folder 000,001,002,003 for train and 004 for test
	--pattern 1 # the input images contain CIR, G, NDVI, NIR, R, RE, RGB spectral bands
```
