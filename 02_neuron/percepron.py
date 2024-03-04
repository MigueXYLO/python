import numpy as np

class Perceptron:
    def __init__(self):
        self.w=None
        self.b=None
    
    def fit(self, xx:np.ndarray, yy:np.ndarray, learning_rate:float=0.1, seed: int=1) -> None:
        # Create random weights
        num_features=xx.shape[1]
        rng=np.random.default_rng(seed)
        self.w=rng.random(num_features)
        self.b=rng.random()
        num_samples = xx.shape[1]
        
        # Training algorithm
        change = True
        while change:
            change=False
            for i,x in enumerate(xx):
                o=self.predict(x)
                print(str(x)+ ' ' +str(o))
                print("-------------------")
                if o != yy[i]:
                    change=True
                    update=learning_rate*(yy[i]-o)
                    self.w+=update*x
                    self.b+=update
            print("--------------------")        

    def predict(self, x: np.ndarray) -> int:
        weighted_sum=np.dot(x,self.w)+self.b
        return 1 if weighted_sum > 0 else -1
