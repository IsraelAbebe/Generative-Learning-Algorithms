import numpy as np 

class GaussianDiscriminantAnalysis():
    def __init__(self,train_data,label_data):
        self.train_data = train_data
        self.label_data = label_data
        
        self.phis = {i:self.get_phi(int(i)) for i in np.unique(self.label_data)}
        self.muis = {i:self.get_mui(int(i)) for i in np.unique(self.label_data)}
        self.sigma = self.get_sigma()
         
    def get_phi(self,i):
        return np.sum(self.label_data==i)/len(self.label_data)
    
    def get_mui(self,i):
        return  (np.nansum((self.label_data==i)*self.train_data,axis=0)+np.nansum(self.label_data==i))/(np.nansum(self.label_data==i)+len(self.label_data))
    
    def get_sigma(self):
        features = len(self.train_data[0])
        
        sigma = np.zeros((features,features)) 
        for i in np.unique(self.label_data):
            train = self.train_data*(self.label_data==i)
            train = train[~(train==0).all(1)]
            values = train-self.muis[i] 
            values[np.isnan(values)] = 0
            cov_value = np.dot(values.T,values)
            sigma += cov_value
                
            sigma = sigma + cov_value
        return sigma/len(self.train_data)
    
    def get_predictions(self,data):
        cov = self.sigma
        classes = np.unique(self.label_data)
        
        pred = []
        for i in classes:
            phi = self.phis[i]
            det = np.abs(np.linalg.det(cov))
            
            if np.abs(det) < 0.000001:
                det=1
            
            first = 1/(((2*np.pi)**(len(classes)/2))*np.sqrt(np.abs(det)))
            sigma_inv = np.linalg.pinv(cov)
            
            mui = self.muis[i] 
            value = (data-mui)
            
            result = first*np.exp((-1/2)*np.dot(np.dot(value,sigma_inv),value.T))
            pred.append(np.sum(result))
          
        return np.argmax(pred)
    
    def evaluate(self,data,label):
        pred = []
        for i in data:
            p = self.get_predictions(i)
            pred.append(p)
            
        return np.mean(pred==label.flatten())
        