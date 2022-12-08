# Neural FCA

A big homework to merge [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
and [Formal Concept Analysis](https://en.wikipedia.org/wiki/Formal_concept_analysis).
Created for the course Ordered Sets in Data Analysis (GitHub [repo](https://github.com/EgorDudyrev/OSDA_course))
taught in Data Science Master programme in HSE University, Moscow. 

![Example of a network build upon Fruits dataset](https://github.com/EgorDudyrev/OSDA_course/blob/Autumn_2022/neural_fca/fitted_network.png)

# Task description

## To-do list
* Bare minimum (5-6 pts.)
  * [ ] Find a dataset for binary classification:\
  _Ideal dataset would be: openly available, with various datatypes (numbers, categories, graphs, etc), with hundreds of rows_;
  * [ ] Describe scaling (binarization) strategy for the dataset features,
  * [ ] Describe prediction quality measure best suited for the dataset\
  _(e.g. accuracy, f1 score, or any measure that you find reasonable)_,
  * [ ] Fit and test the network on your task
  (consider NeuralFCA.ipynb to create and fit your network).
* Ways to improve the basic pipeline (additional 1-4 pts):
  * [ ] Write better scaling algorithm to binarize the original data
  * [ ] Test and document various techniques to select best concepts from the concept lattice
  _(it would be great if your network will only contain no more than 30 concepts)_
  * [ ] Test and document the efficiency of various nonlinearities to put in the network
  * [ ] Compare the prediction quality of the proposed model with State-of-the-Art approaches
  * [ ] Create a nice and readable visualization of the network


## How to submit

Students are expected to provide the working code and pdf report for their homework by the end of the semester. 

The code should be available on the students GitHub accounts.
Please, create a new notebook and a new pipeline similar to that of
[LazyFCA](https://github.com/EgorDudyrev/OSDA_course/blob/Autumn_2022/lazy_fca) homework.

You can consult [neural_lib.py](https://github.com/EgorDudyrev/OSDA_course/blob/Autumn_2022/neural_fca/neural_lib.py)
and [NeuralFCA.ipynb](https://github.com/EgorDudyrev/OSDA_course/blob/Autumn_2022/neural_fca/NeuralFCA.ipynb) 
for examples how to create, fit and visualize your network.

A pdf report should describe the reasoning behind every step of your homework. For example,
it should contain a description of your dataset, what quality measure you have chosen (and why), 
how did you optimize the network, etc.
