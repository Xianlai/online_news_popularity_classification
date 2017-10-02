# Online News Popularity Classification

## Abstract:
Facilitated by the fast spreading and developing of internet and smart devices, how can we understand online news browsing data and find the pattern inside it become more and more important.

In this project I am using online news popularity data set containing 39644 news articles and 57 features about each article including statistical features like number of words in title, rate of non-stop words in the content, article publish weekdays etc. and NLP features like positive word rate, title subjectivity level etc. The goal is to classify whether these articles are popular or not quantified by article shares. 

## Motivation:
The main motivation of this project is not aiming at developing a new method or fine tunning a state of art model to achieve extreme high accuracy. The focus here is to set up a systematic procedure and framework to understand the dataset and compare different models on this dataset.

0_Preprocessing.ipynb:
- By investigating mean, std, range, unique value counts and outlier counts, we are able to merge related features, remove outliers and standardize the dataset for future model fitting.
- By decompositing data set using differnt method like PCA, sparse PCA, factor analysis and NMF, we are able to look at the dataset from more perspectives and possibly improve the model performance.

1_Model fitting and selection.ipynb:
- By setting up performance evaluation, expected loss, as sum of square bias and variance, We are able to find the balance between model's flexibility and steadiness.
- By visulizing the results of parameter tunning, we can understand how each of the parameters changed the model's behavior on this dataset.
- By comparing the decision boundaries of model before and after parameter tunning, we are able to tell how does model adapt to this dataset and identify possible problems and improvements.
- By plotting the histogram of prediction confidence, we are able to understand better how predictions are made by different models, and discard problematic ones which looks fine if we just judge by expected loss.

## API References:
- scikit-learn: Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay. Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830 (2011)

- Matplotlib: John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55

- Pandas: Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)

- NumPy & SciPy: Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

- xgBoost: DMLC eXtra Gradient Boost Python Package, https://github.com/dmlc/xgboost

## License:
MIT License

Copyright (c) [2017] [Xian Lai]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
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

## Contact:
Xian Lai
Data Analytics, CIS @ Fordham University
XianLaaai@gmail.com


