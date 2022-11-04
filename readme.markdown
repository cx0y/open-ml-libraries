⚠️ Developing

<h1 align="center">Open Machine Learning</h1>
<p align="center"> <a href="#ml-frem">ML Frameworks</a> | <a href="#npl-frem">NLP Frameworks</a> | <a href="#cvl">Computer Vision Libraries</a> | <a href="#mlt">ML Tools</a> | <a href="#mlh">ML Hosting</a> | <a href="#res">Resources</a> </p>
<p align="center"> | <a href="#contributing">Contributing</a> | <a href="license">License</a> | <a href=""></a> </p>

<h2 align="center" id="ml-frem">ML Frameworks</h2>

<p align="center">
<a href="#acme">Acme</a>  |  
<a href="#AdaNet">AdaNet</a> | 
<a href="#Analytics-Zoo">Analytics Zoo</a> |
<a href="#Apache-MXNet">Apache MXNet</a> |
<a href="#Apache-Spark">Apache Spark</a> |
<a href="#auto-ml">auto_ml</a> |
<a href="#BigDL">BigDL</a> |
<a href="#Blocks">Blocks</a> |
<a href="#Caffe">Caffe</a> |
<a href="#ConvNetJS">ConvNetJS</a> |
<a href="#DatumBox">DatumBox</a> 

</p>

---

> > <h2 id="acme">1. Acme</h2>

Acme is a library of reinforcement learning (RL) building blocks that strives to expose simple, efficient, and readable agents. These agents first and foremost serve both as reference implementations as well as providing strong baselines for algorithm performance. However, the baseline agents exposed by Acme should also provide enough flexibility and simplicity that they can be used as a starting block for novel research. Finally, the building blocks of Acme are designed in such a way that the agents can be run at multiple scales (e.g. single-stream vs. distributed agents).

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/deepmind/acme">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
  <table>
  
  
---
  
  
>> <h2 id="AdaNet">2. AdaNet</h2>

**AdaNet** is a lightweight TensorFlow-based framework for automatically learning high-quality models with minimal expert intervention. AdaNet builds on recent AutoML efforts to be fast and flexible while providing learning guarantees. Importantly, AdaNet provides a general framework for not only learning a neural network architecture, but also for learning to ensemble to obtain even better models.

This project is based on the _AdaNet algorithm_, presented in “[AdaNet: Adaptive Structural Learning of Artificial Neural Networks](http://proceedings.mlr.press/v70/cortes17a.html)” at [ICML 2017](https://icml.cc/Conferences/2017), for learning the structure of a neural network as an ensemble of subnetworks.

AdaNet has the following goals:

- _Ease of use_: Provide familiar APIs (e.g. Keras, Estimator) for training, evaluating, and serving models.
- _Speed_: Scale with available compute and quickly produce high quality models.
- _Flexibility_: Allow researchers and practitioners to extend AdaNet to novel subnetwork architectures, search spaces, and tasks.
- _Learning guarantees_: Optimize an objective that offers theoretical learning guarantees.

For more information, you may [read the docs](https://adanet.readthedocs.io)

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/tensorflow/adanet">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
 <table>

---

> > <h2 id="Analytics-Zoo">3. Analytics Zoo</h2>

Analytics Zoo is an open source _**Big Data AI**_ platform, and includes the following features for scaling end-to-end AI to distributed Big Data:

- [Orca](#getting-started-with-orca): seamlessly scale out TensorFlow and PyTorch for Big Data (using Spark & Ray)

- [RayOnSpark](#getting-started-with-rayonspark): run Ray programs directly on Big Data clusters

- [BigDL Extensions](#getting-started-with-bigdl-extensions): high-level Spark ML pipeline and Keras-like APIs for BigDL

- [Chronos](#getting-started-with-chronos): scalable time series analysis using AutoML

- [PPML](#ppml-privacy-preserving-machine-learning): privacy preserving big data analysis and machine learning (_experimental_)

For more information, you may [read the docs](https://analytics-zoo.readthedocs.io/).

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/intel-analytics/analytics-zoo">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
 <table>

---

> > <h2 id="Apache-MXNet">4. Apache MXNet</h2>

Apache MXNet is a deep learning framework designed for both _efficiency_ and _flexibility_.
It allows you to **_mix_** [symbolic and imperative programming](https://mxnet.apache.org/api/architecture/program_model)
to **_maximize_** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scalable to many GPUs and machines.

Apache MXNet is more than a deep learning project. It is a [community](https://mxnet.apache.org/versions/master/community)
on a mission of democratizing AI. It is a collection of [blue prints and guidelines](https://mxnet.apache.org/api/architecture/overview)
for building deep learning systems, and interesting insights of DL systems for hackers.

For more information, visit [mxnet.apache.org](https://mxnet.apache.org)

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>cpp, python</th>
    <th><a href="https://github.com/apache/incubator-mxnet">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
 <table>

---

> > <h2 id="Apache-Spark">5. Apache Spark</h2>

Spark is a unified analytics engine for large-scale data processing. It provides
high-level APIs in Scala, Java, Python, and R, and an optimized engine that
supports general computation graphs for data analysis. It also supports a
rich set of higher-level tools including Spark SQL for SQL and DataFrames,
pandas API on Spark for pandas workloads, MLlib for machine learning, GraphX for graph processing,
and Structured Streaming for stream processing.

<https://spark.apache.org/>

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>scala, python, java, etc</th>
    <th><a href="https://github.com/apache/spark">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
 <table>

---

> > <h2 id="auto-ml">6. auto_ml</h2>

Automated machine learning for analytics & production. [UNMAINTAINED]

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/ClimbsRocks/auto_ml">Github</a></th>
    <th> MIT License</th>
  </tr>
 <table>
 
 ---
 
 >> <h2 id="BigDL">7. BigDL</h2> 
 
BigDL seamlessly scales your data analytics & AI applications from laptop to cloud, with the following libraries:

- [Orca](#orca): Distributed Big Data & AI (TF & PyTorch) Pipeline on Spark and Ray

- [Nano](#nano): Transparent Acceleration of Tensorflow & PyTorch Programs

- [DLlib](#dllib): “Equivalent of Spark MLlib” for Deep Learning

- [Chronos](#chronos): Scalable Time Series Analysis using AutoML

- [Friesian](#friesian): End-to-End Recommendation Systems

- [PPML](#ppml) (experimental): Secure Big Data and AI (with SGX Hardware Security)

For more information, you may [read the docs](https://bigdl.readthedocs.io/).

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>scala, python, java, etc</th>
    <th><a href="https://github.com/intel-analytics/BigDL">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
 <table>

---

> > <h2 id="Blocks">8. Blocks</h2>

Blocks is a framework that helps you build neural network models on top of
Theano. Currently it supports and provides:

- Constructing parametrized Theano operations, called "bricks"
- Pattern matching to select variables and bricks in large models
- Algorithms to optimize your model
- Saving and resuming of training
- Monitoring and analyzing values during training progress (on the training set
  as well as on test sets)
- Application of graph transformations, such as dropout

In the future we also hope to support:

- Dimension, type and axes-checking

Please see the [documentation](http://blocks.readthedocs.org) for more information.

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/mila-iqia/blocks">Github</a></th>
    <th><a href="https://github.com/mila-iqia/blocks/blob/master/LICENSE">LICENSE</a></th>
  </tr>
 <table>
 
---
 
 >> <h2 id="Caffe">9. Caffe</h2> 
 
Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>cpp, python</th>
    <th><a href="https://github.com/BVLC/caffe">Github</a></th>
    <th><a href="https://github.com/BVLC/caffe/blob/master/LICENSE">LICENSE</a></th>
  </tr>
 <table>
 
 
---

> > <h2 id="ConvNetJS">10. ConvNetJS</h2>

ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An experimental **Reinforcement Learning** module, based on Deep Q Learning

For much more information, see the main page at [convnetjs.com](http://convnetjs.com)

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>javascript</th>
    <th><a href="https://github.com/karpathy/convnetjs">Github</a></th>
    <th>MIT license</th>
  </tr>
 <table>
 
 
---

> > <h2 id="DatumBox">11. DatumBox</h2>

The Datumbox Machine Learning Framework is an open-source framework written in Java which allows the rapid development Machine Learning and Statistical applications. The main focus of the framework is to include a large number of machine learning algorithms & statistical methods and to be able to handle large sized datasets.

Datumbox comes with a large number of pre-trained models which allow you to perform Sentiment Analysis (Document & Twitter), Subjectivity Analysis, Topic Classification, Spam Detection, Adult Content Detection, Language Detection, Commercial Detection, Educational Detection and Gender Detection. To get the binary models check out the [Datumbox Zoo](https://github.com/datumbox/datumbox-framework-zoo/).

The Framework currently supports performing multiple Parametric & non-parametric Statistical tests, calculating descriptive statistics on censored & uncensored data, performing ANOVA, Cluster Analysis, Dimension Reduction, Regression Analysis, Timeseries Analysis, Sampling and calculation of probabilities from the most common discrete and continues Distributions. In addition it provides several implemented algorithms including Max Entropy, Naive Bayes, SVM, Bootstrap Aggregating, Adaboost, Kmeans, Hierarchical Clustering, Dirichlet Process Mixture Models, Softmax Regression, Ordinal Regression, Linear Regression, Stepwise Regression, PCA and several other techniques that can be used for feature selection, ensemble learning, linear programming solving and recommender systems.

<https://www.datumbox.com>

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>javascript</th>
    <th><a href="https://github.com/karpathy/convnetjs">Github</a></th>
    <th>MIT license</th>
  </tr>
 <table>
 
 
---

<h2 align="center" id="npl-frem">NLP Frameworks</h2>
<p align="center"> 
<a href="#AllenNLP">AllenNLP</a>  |  
<a href="#Apache-OpenNLP">Apache OpenNLP</a> | 
<a href="#ERNIE">ERNIE</a> | 
<a href="#flair">flair</a> | 
<a href="#gensim">gensim</a> |
<a href="#icecaps">icecaps</a> 

</p>

---

> > <h2 id="AllenNLP">1. AllenNLP</h2>

An Apache 2.0 NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.

<https://allennlp.org>

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/allenai/allennlp">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
  <table>
  
  
  -----------------------------------------------------

> > <h2 id="Apache-OpenNLP">2. Apache OpenNLP</h2>

The Apache OpenNLP library is a machine learning based toolkit for the processing of natural language text.

This toolkit is written completely in Java and provides support for common NLP tasks, such as tokenization, sentence segmentation, part-of-speech tagging, named entity extraction, chunking, parsing, coreference resolution, language detection and more!

These tasks are usually required to build more advanced text processing services.

The goal of the OpenNLP project is to be a mature toolkit for the above mentioned tasks.

An additional goal is to provide a large number of pre-built models for a variety of languages, as well as the annotated text resources that those models are derived from.

Presently, OpenNLP includes common classifiers such as Maximum Entropy, Perceptron and Naive Bayes.

OpenNLP can be used both programmatically through its Java API or from a terminal through its CLI. OpenNLP API can be easily plugged into distributed streaming data pipelines like Apache Flink, Apache NiFi, Apache Spark.

<https://opennlp.apache.org>

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>java</th>
    <th><a href="https://github.com/apache/opennlp">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
  <table>
  
  -----------------------------------------------------

> > <h2 id="ERNIE">3. ERNIE</h2>

Official implementations for various pre-training models of ERNIE-family, covering topics of Language Understanding & Generation, Multimodal Understanding & Generation, and beyond.

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/PaddlePaddle/ERNIE">Github</a></th>
    <th> Apache-2.0 license</th>
  </tr>
  <table>
  
  -----------------------------------------------------

> > <h2 id="flair">4. flair</h2>

- **A powerful NLP library.** Flair allows you to apply our state-of-the-art natural language processing (NLP)
  models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS),
  special support for [biomedical data](https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR.md),
  sense disambiguation and classification, with support for a rapidly growing number of languages.

- **A text embedding library.** Flair has simple interfaces that allow you to use and combine different word and
  document embeddings, including our proposed **[Flair embeddings](https://www.aclweb.org/anthology/C18-1139/)**, BERT embeddings and ELMo embeddings.

- **A PyTorch NLP framework.** Our framework builds directly on [PyTorch](https://pytorch.org/), making it easy to
  train your own models and experiment with new approaches using Flair embeddings and classes.

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/flairNLP/flair">Github</a></th>
    <th>MIT License</th>
  </tr>
  <table>
  
  -----------------------------------------------------

> > <h2 id="gensim">5. gensim</h2>

Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/RaRe-Technologies/gensim">Github</a></th>
    <th>MIT License</th>
  </tr>
  <table>

---

> > <h2 id="icecaps">6. icecaps</h2>

Microsoft Icecaps is an open-source toolkit for building neural conversational systems. Icecaps provides an array of tools from recent conversation modeling and general NLP literature within a flexible paradigm that enables complex multi-task learning setups. 

Icecaps is currently on version 0.2.0. In this version we introduced several functionalities:
* Personalization embeddings for transformer models
* Early stopping variant for performing validation across all saved checkpoints
* Implementations for both SpaceFusion and StyleFusion
* New text data processing features, including sorting and trait grounding
* Tree data processing features from JSON files using the new JSONDataProcessor

<table>
  <tr>
    <th>language</th>
    <th>source</th>
    <th>license</th>
  </tr>
   <tr>
    <th>python</th>
    <th><a href="https://github.com/microsoft/icecaps">Github</a></th>
    <th>MIT License</th>
  </tr>
  <table>

---





---

<h2 align="center" id="contributing">Contributing</h2>

pull requests are welcome! for major changes, please open an issue first to discuss what you would like to change.
please make sure to update tests as appropriate.

> > [Code of Conduct](./code_of_conduct.md)

> > [Contributing](./contributing.md)

<h2 align="center" id="license">License</h2>

```py
MIT License

Copyright (c) 2022 wiz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


```

[Link](./license)

---
