# Spoken Language Idenfication

Simple implementation of neural networks for spoken language identification

## Dataset
* TIMIT
* 한국어 전화망 DB [NAVER Corp]
* number of train : 3696
* number of validation : 400
* number of test : 192

## Requirement

* python 3.6.8
* tensorflow==1.13.1
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
|ENG|   |   |   |
|KOR|   |   |   |
|Recall|    |   |   |

#### 3. Total
|   |Accuracy|Operation|
|:---:|:---:|:---:|
|DNN|   |   |
|CNN|   |   |
|CRNN|    |   |

 ---
 
# 발화된 언어 인식을 위한 딥러닝

발화된 언어를 인식하기 위한 뉴럴 네트워크를 구현했습니다.

## Dataset
* TIMIT
* 한국어 전화망 DB [NAVER Corp]
* train data : 3696개
* validation data : 400개
* test data : 192개

## Requirement

* python 3.6.8
* tensorflow==1.13.1
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
|ENG|   |   |   |
|KOR|   |   |   |
|Recall|    |   |   |

#### 3. Total
|   |Accuracy|Operation|
|:---:|:---:|:---:|
|DNN|   |   |
|CNN|   |   |
|CRNN|    |   |