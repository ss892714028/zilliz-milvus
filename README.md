# zilliz-milvus
This repository hosts a simple use case of Milvus, an open source similarity search engine for embeddings. [Link to Milvus](https://github.com/milvus-io/milvus)

## Background
[MNIST](http://yann.lecun.com/exdb/mnist/) is a database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples.

## End-to-end illustration: A Toy Example
A Convolution Neural Network is trained using 60,000 training images to create embedding. Then each of the 10,000 test image is transformed into vector with shape (50,). The vectors are then inserted into Milvus to achieve real time similarity search. This example demostrates how Milvus can be useful in a image recommender system scenario. [Link to Jupyter notebook Demostration](https://github.com/ss892714028/zilliz-milvus/blob/master/test.ipynb). The flowchart can be seen in the appendix. 

## Potential Use Cases
* Protein Structure Search: [ProteinNet](https://github.com/aqlaboratory/proteinnet)  
The ever growing protein sequence/structure database awaits methods that are capable of fast similarity search of protein structures. 
* Electronic Health Records (EHR): Milvus can achieve similarity search on retrospective data in real time; assist doctors to make decisions on medical procedure approaches and dosage control.

## My thoughts..
* About Ease of use: The installation/setup cannot be more straightforward. For this assignment, I used Ubuntu 20.04 with a virtual machine running on windows, the whole process took less than 20 mins with most of the time waiting for things to download (slow internet T.T).
* Milvus can be really useful in the field of healthcare/bioinformatics. I work very close to healthcare domain both in academia and in the industry. There is a huge disconnect between the tech and the healthcare community. 


## Appendix
![flowchart](https://github.com/ss892714028/zilliz-milvus/blob/master/flowchart.png)
