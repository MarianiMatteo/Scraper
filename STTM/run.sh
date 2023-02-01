STTM_PATH=$PWD

DATA_PATH=$PWD/dataset


java -jar jar/STTM.jar -model GPU_PDMM -corpus corpus.txt -vectors "glove.twitter.27B/glove.twitter.27B.50d.txt" -ntopics 10
