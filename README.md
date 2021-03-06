# Spoken Language Idenfication

Simple implementation of neural networks for spoken language identification

## Dataset
* TIMIT
* 한국어 전화망 DB [NAVER Corp]
* number of train : 3696 file * 2 language
* number of validation : 400 file * 2 language
* number of test : 192 file * 2 language

## Requirement

* python 3.6.8
* torch==1.2.0
* numpy
* matplotlib
* librosa

## Result
#### 1. DNN
|   |ENG|KOR|Precision|
|:---:|:---:|:---:|:---:|
|ENG|191|2|98.96|
|KOR|1|190|99.48|
|Recall|99.48|98.96|99.22|

#### 2. CRNN
|   |ENG|KOR|Precision|
|:---:|:---:|:---:|:---:|
|ENG|192|0|100.00|
|KOR|0|192|100.00|
|Recall|100.00|100.00|100.00|

#### 3. Total
|   |Accuracy|Operation|
|:---:|:---:|:---:|
|DNN|99.22|   |
|CRNN|100.00|   |

 ---
 
# 발화된 언어 인식을 위한 딥러닝

발화된 언어를 인식하기 위한 뉴럴 네트워크를 구현했습니다.

## Dataset
* TIMIT
* 한국어 전화망 DB [NAVER Corp]
* train data : 3696 파일 * 2 언어
* validation data : 400 파일 * 2 언어
* test data : 192 파일 * 2 언

## Requirement

* python 3.6.8
* torch==1.2.0
* numpy
* matplotlib
* librosa

## Result
#### 1. DNN
|   |ENG|KOR|Precision|
|:---:|:---:|:---:|:---:|
|ENG|191|2|98.96|
|KOR|1|190|99.48|
|Recall|99.48|98.96|99.22|

#### 2. CRNN
|   |ENG|KOR|Precision|
|:---:|:---:|:---:|:---:|
|ENG|192|0|100.00|
|KOR|0|192|100.00|
|Recall|100.00|100.00|100.00|

#### 3. Total
|   |Accuracy|Operation|
|:---:|:---:|:---:|
|DNN|99.22|   |
|CRNN|100.00|   |