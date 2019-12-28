import numpy as np

class NaiveBayes():
    def __init__(self,train_data,label_data):
        '''
        given train/label data initializes the NaiveBayes  class
        
        Parameters
        ----------
        train_data : numpy-array 
        label_data : numpy-array
        '''
        self.train_data = train_data
        self.label_data = label_data
        
        self.feature_cout = len(self.train_data[0])
        self.classes = np.unique(self.label_data)
        
        self.priors = {i:self.get_prior(i) for i in np.unique(self.label_data)}
        self.means = {i:self.get_mean(i) for i in np.unique(self.label_data)}
        self.variances = {i:self.get_variance(i) for i in np.unique(self.label_data)}
        
        
    def get_prior(self,i):
        return np.sum(self.label_data==i)/len(self.label_data)
    
    def get_mean(self,i):
        return np.nansum(self.train_data*(self.label_data==i),axis=0)/(np.nansum(self.label_data==i))
    
    def get_variance(self,i):
        inner = self.train_data*(self.label_data==i)-self.means[i]
        inner[np.isnan(inner)] = 0
        return np.dot(inner.T,inner)/np.nansum(self.label_data==i)
    
    def get_p_xc(self,x,i):
        mean = self.means[i]
        variance = self.variances[i]
        inner = x-mean
        cov_sqrt = np.sqrt(np.abs(np.linalg.det(variance)))
        
        if cov_sqrt < 0.000001:
                cov_sqrt=1
        
        first = 1/(((2*np.pi)**(len(self.classes)/2))*cov_sqrt)
        second = np.exp(-0.5*np.dot(np.dot(inner.T,np.linalg.pinv(variance)),inner))
        
        return np.prod(first*second)
        
    
    def get_predictions(self,x):
        p_x_cc = 0
        for i in self.classes:
            p_x_cc += np.sum(self.get_p_xc(x,i))*self.priors[i]
        
        prob_list = []
        for i in self.classes:
            p_x_c = self.get_p_xc(x,i)
            prior = self.priors[i]
            
            prob = (p_x_c*prior)/p_x_cc
            prob_list.append(prob)
            
        return np.argmax(prob_list)
    
    def evaluate(self,data,label):
        pred = []
        for i in data:
            p = self.get_predictions(i)
            pred.append(p)
        
        return np.mean(pred==label.flatten())
        