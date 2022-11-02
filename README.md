<h1 align="center">open-machine-learning</h1>
<p align="center">| <a href="#ml-frem">ML Frameworks</a> | <a href="#npl-frem">NLP Frameworks</a> | <a href="#cvl">Computer Vision Libraries</a> | <a href="#mlt">ML Tools</a> | <a href="#mlh">ML Hosting</a> | <a href="#res">Resources</a> |</p>
<p align="center"> | <a href="#contributing">contributing</a> | <a href="license">license</a> | <a href=""></a> </p> 

<h2 align="center" id="ml-frem">ML Frameworks</h2>
<a href="#acme">Acme</a>  |  <a href="#AdaNet">AdaNet</a>

-----------------------------------------------------------

>> <h2 id="acme">1. Acme</h2>

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
  
  
  -----------------------------------------------------
  
  
>> <h2 id="AdaNet">2. AdaNet</h2> 

**AdaNet** is a lightweight TensorFlow-based framework for automatically learning high-quality models with minimal expert intervention. AdaNet builds on recent AutoML efforts to be fast and flexible while providing learning guarantees. Importantly, AdaNet provides a general framework for not only learning a neural network architecture, but also for learning to ensemble to obtain even better models.

This project is based on the _AdaNet algorithm_, presented in “[AdaNet: Adaptive Structural Learning of Artificial Neural Networks](http://proceedings.mlr.press/v70/cortes17a.html)” at [ICML 2017](https://icml.cc/Conferences/2017), for learning the structure of a neural network as an ensemble of subnetworks.

AdaNet has the following goals:

* _Ease of use_: Provide familiar APIs (e.g. Keras, Estimator) for training, evaluating, and serving models.
* _Speed_: Scale with available compute and quickly produce high quality models.
* _Flexibility_: Allow researchers and practitioners to extend AdaNet to novel subnetwork architectures, search spaces, and tasks.
* _Learning guarantees_: Optimize an objective that offers theoretical learning guarantees.
[documentation](https://adanet.readthedocs.io)

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

---------------------------------------------------

<h2 align="center" id="contributing">Contributing</h2>

pull requests are welcome! for major changes, please open an issue first to discuss what you would like to change.
please make sure to update tests as appropriate.

>> [Code of Conduct](./code_of_conduct.md)

>> [Contributing](./contributing.md)

<h2 align="center" id="license">LICENSE</h2>

>> [MIT license](./LICENSE)

--------------------------------
