# Generative Learning algorithms
----------------------------------
[lecture notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf)

The generative model is the model that first try to learn what each object might look like. Then, based on input, it gives a probability of the input being this class.



### Gaussian discriminant analysis
--------------------------------
In this model, we’ll assume that p(x|y) is distributed according to a multivariate normal distribution. Let’s talk briefly about the properties of multivariate normal distributions before moving on to the GDA model itself.



### Naive Bayes
--------------
![image](https://miro.medium.com/max/510/1*tjcmj9cDQ-rHXAtxCu5bRQ.png)

Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.[source blog](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)


### Setup

```
pip install -r requirements.txt
```


### usage
---------------
    
refer to usage notebook


