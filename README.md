# Specialize Word Embedding

------
## Word embedding：
In short, word embedding is to represent words in a vector space. Natural language processing traditionally treats words as many discrete atomic symbols, which is not like image processing. It seems to meet our intuition that the pixel matrix of image is more reasonable to be encoded as a rich and high-dimension vector and words can not be. In 2013, [Mikolov et al][1] propose a novel method to train and encode words to be vectors effectively with unsupervised learning, which is called word2vec, a popular tool now.

## Shortcoming of most current word embedding models
Briefly most current embedding models depend on the correlation between words appearing nearly. The hypothesis of this idea is that words occuring in similar contexts have similar meanings. But actually the words having similar contexts only means similar grammar, not similar semanteme. So the vector spaces being trained from these models are suitable for analyzing word semantic relatedness, not word semantic similarity.
## Embedding based on corpus and thesaurus
sometimes we are eager for word embedding that is specialized for specific task. For example specialized word similarity for translation and specialized word relatedness for document classification. Some paper retrofit and specialize word embedding with both corpus context and specific thesaurus. I reproduct the idea of [this paper][2] and extra two thesauruses: the word similarity thessaurus from the electrionic dictionary [MyThes][3]. and the word relatedness thesaurus from the data set [USF Free Association Norms][4]. 

## Usage
for training on corpus text8(a text dataset in the folder named corpus, unzip it first) with thesaurus jointly, command  line：
```shell
python specializing_word_embedding.py ./corpus/text8 ./models/relatednesss_embedding.model ./thesaurus/USF_words_relatedness.js
python specializing_word_embedding.py ./corpus/text8 ./models/similarity_embedding.model ./thesaurus/th_en_US_new.js
```
for training without thesaurus, command  line：
```shell
python specializing_word_embedding.py ./corpus/text8 ./models/raw_embedding.model
```
for evaulating the result of relatedness and similarity specializing, command line:
```shell
python evaluation_with_pearson_coeff.py ./models/raw_embedding.model ./models/relatedness_embedding.model ./models/similarity_embedding.model
```
## Result
I only train this model briefly with corpus text8 for about 30 mins, the evalution result show as follow:
> relatedness evaluation: raw embedding(0.514) vs. relatedness embedding(0.605)
> similarity evaluation: raw embedding(0.297) vs. similarity embedding(0.521)

environment: Ubuntu(16.04.2), CUDA(9.0), python(3.5.4), pytorch(0.1.12)


[1]:https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
[2]:http://www.aclweb.org/anthology/D15-1242
[3]:http://www.openoffice.org/lingucomponent/thesaurus.html
[4]:http://w3.usf.edu/FreeAssociation/
