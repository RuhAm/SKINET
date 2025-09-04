# SKINET
<b>S</b>upport vector <b>K</b>ernel for <b>I</b>nferring <b>N</b>atural selection and <b>E</b>volutionary <b>T</b>arget detection

<i>SKINET</i> is a framework to distinguish signs of selective sweep from genomic data. We use a custom kernel using Support Vector Machine (SVM) framework to facilitate classification using machine learning. This software package can be used for applying <i>SKINET</i> to classify any genetic regions into sweep or neutral (i.e, regions showing signs of selective sweep and region without them). We demonstrate how to work with simulated and empirical data in the .vcf format in a step by step fashion in this implementation. 





Downloads and requirements
==========================

First clone the repo using the following command

        git clone https://github.com/RuhAm/SKINET

We will need `python3` installed for different parts of the
project.

For `python3` dependencies, please install the following packages found in the `requirements.txt` file inside the `./SKINET` folder using the following commands. 

`cd ./SKINET`


    pip install -r requirements.txt

In this software we have utilized `argparse` for command line interactions including data preprocessing, model training and testing, and empirical application.


Data Preprocessing
===========================================
Users can preprocess the sweep and neutral .ms files using global sorting and alignment processing using our Data Preprocessing pipeline. First, we need to change our working directory to the folder

        cd ./SKINET

We will need to convert the .ms files for sweep and neutral observations that are present in the `./SKINET/Data/MS` folder. The folder contains two subdirectories `./SKINET/Data/MS/Neutral` and `./SKINET/Data/MS/Sweep` that contains the .ms files. The directories are preloaded with 50 observations per class. Please feel free to replace the files with appropriate number of observations as per your experimental needs.

To convert the .ms files into .csv format, please use the following command:

    python3 ms_to_csv.py <number of files> <class>

The first argument is the number of files that the user wants to convert to csv. The second argument takes on 0 or 1 as value where 1 is used for sweep observations and 0 is used for neutral observations.

For example, we use the following command to preprocess the 50 sweep observations that are in .ms format.

    python3 ms_to_csv.py 50 1

we use the following command to preprocess the 50 neutral observations that are in .ms format.

    python3 ms_to_csv.py 50 0

Now that we have our .csv files, we go on to globally sort them and process the haplotypes using our unique haplotype alignment strategy using the following commands:

    python3 parse_csv.py <number of files> <class>

The first argument is the number of csv files that the user wants to process. The second argument takes on 0 or 1 as value where 1 is used for processing sweep observations and 0 is used for neutral observations.

For example, we use the following command to preprocess the 50 sweep observations.

    python3 parse_csv.py 50 1

We use the following command to preprocess the 50 neutral observations.

    python3 parse_csv.py 50 0

Model training and testing
===========================================

These scripts allow users to train the <i>SKINET</i> models for detecting positive natural selection. We have the four models: <i>SKINET</i>[D1] (uses first order difference operator), <i>SKINET</i>[D2] (uses second order difference operator), <i>SKINET</i>[D1-RBF](uses mixture of first order difference operator and RBF kernel), and <i>SKINET</i>[D2-RBF] (uses mixture of second order difference operator and RBF kernel).

## Training and testing with <i>SKINET</i>[D1]

This script allows users to train <i>SKINET</i>[D1] model and allows them to specify the number of training and test observations. 

    python3 d1_train.py <--train_obs TRAIN_OBS> <--test_obs TEST_OBS>

For example, we use the following command to train the <i>SKINET</i>[D1] model with 40 observations per class for training and 10 observations per class for testing.

    python3 d1_train.py --train_obs 40 --test_obs 10

The resulting confusion matrix and probabilities will be saved in the `./SKINET/Results` folder. 

## Training and testing with <i>SKINET</i>[D2]

This script allows users to train <i>SKINET</i>[D2] model and allows them to specify the number of training and test observations. 

    python3 d2_train.py <--train_obs TRAIN_OBS> <--test_obs TEST_OBS>

For example, we use the following command to train the <i>SKINET</i>[D2] model with 40 observations per class for training and 10 observations per class for testing.

    python3 d2_train.py --train_obs 40 --test_obs 10

The resulting confusion matrix and probabilities will be saved in the `./SKINET/Results` folder. 

## Training and testing with <i>SKINET</i>[D1-RBF]

This script allows users to train <i>SKINET</i>[D1-RBF] model and allows them to specify the number of training and test observations. 

    python3 d1_mix_train.py <--train_obs TRAIN_OBS> <--test_obs TEST_OBS>

For example, we use the following command to train the <i>SKINET</i>[D1] model with 40 observations per class for training and 10 observations per class for testing.

    python3 d1_mix_train.py --train_obs 40 --test_obs 10

The resulting confusion matrix and probabilities will be saved in the `./SKINET/Results` folder. 

## Training and testing with <i>SKINET</i>[D2-RBF]

This script allows users to train <i>SKINET</i>[D2-RBF] model and allows them to specify the number of training and test observations. 

    python3 d2_mix_train.py <--train_obs TRAIN_OBS> <--test_obs TEST_OBS>

For example, we use the following command to train the <i>SKINET</i>[D1] model with 40 observations per class for training and 10 observations per class for testing.

    python3 d2_mix_train.py --train_obs 40 --test_obs 10

The resulting confusion matrix and probabilities will be saved in the `./SKINET/Results` folder. 


Empirical application
===========================================
We use the tool <i>SISSSCO</i> to convert the .ms and .vcf files into summary statistic arrays (for training data) and use the multitaper trasformation for image generation.

## Transforming input .ms files to summary statistic arrays

This script allows users to generate summary statistic arrays from the .ms files of both classes (sweep and neutral).

    python3 sum_stat_ms.py <.ms file prefix> <Class name | 1 = sweep, 0 = neutral> <number>

For example, we use the following command to generate summary statistic arrays from the 50 sweep files

    python3 sum_stat_ms.py CEU_neut 0 50

For example, we use the following command to generate summary statistic arrays from the 50 sweep files

    python3 sum_stat_ms.py CEU_sweep 1 50


## Transforming input summary statistic arrays into multitaper images

This script allows users to convert the summary statistic arrays to multitaper images of both classes (sweep and neutral).

    python3 Mult.py <sweep_filename> <neutral_filename> <train number> <test number> <validation number>

For example, we use the following command to generate multitaper images for 40 observations per class for training and 10 observations per class for testing.

    python3 Mult.py training_h12Sweep.csv training_h12Neutral.csv 40 10 0


## Parsing a VCF file

This script allows users to parse the VCF file that is in the `./SKINET/VCF` folder. We have provided one example vcf file `CEU22.vcf` but users can drop in their own vcf files to parse and. After running the script the parsed file and positions will be saved in the `./SKINET/Parsed_VCF` folder.
   
    python3 VCF_parser.py <VCF file name>

For example, we use the following command to parse the `CEU22.vcf` file.

        python3 VCF_parser.py CEU22.vcf

## Generating summary statistics from parsed VCF file

This script allows users to use the parsed VCF file and generate summary statistics. The generated summary statistics file will be saved in the `./SKINET/Summary_statistics` folder in .npy format.

    python3 sum_stat_vcf.py <Parsed VCF file name>

For example, we use the following command to use the parsed `parsed_CEU22.npy` file in order to generate summary statistics.

        python3 sum_stat_vcf.py parsed_CEU22.npy

## Multitaper image generation from summary statistics

This script allows users to use the summary statistics .npy file and transform it into multitaper images. The transformed image will be stored in the  `./SKINET/TFA` folder.

  python3 multitaper_analysis_vcf.py <Empirical summary statistics file name>

For example, we use the following command to use the summary statistics file `empirical_d_CEU22.csv` and generate multitaper images.

        python3 multitaper_analysis_vcf.py empirical_d_CEU22.csv


## Training the <i>SKINET</i>[D1] and testing with empirical data


This script allows users to train <i>SKINET</i>[D1] model using multitaper transformed images and test using the multitaper transformed empirical image.

    python3 EMP_d1.py <training filename> <empirical test filename>>

For example, we use the following command to train the <i>SKINET</i>[D1] model with 40 observations per class for training and 10 observations per class for testing.

    python3 d1_train.py train_Mul_training_h12.csv empirical_CEU22_multitaper.npy

The resulting probabilities will be saved in the `./SKINET/Results` folder. 













