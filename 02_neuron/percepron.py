import numpy as np

class Perceptron:
    def __init__(self):
        self.w=None
        self.b=None
    
    def fit(self, xx:np.ndarray, yy:np.ndarray, learning_rate:float=0.1, seed: int=1) -> None:
        #criar os pesos random
        num_features=xx.shape[1]
        rng=np.random.default_rng(seed)
        self.w=rng.random(num_features)
        self.b=rng.random()
        num_samples = xx.shape[1]
        #algoritmo

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



        #copilot
        #for i in range(num_samples):
        #    # Retrieve the input sample and corresponding label
        #    x = xx[i]
        #    y = yy[i]
        #    
        #    # Make a prediction using the perceptron's predict function
        #    y_pred = self.predict(x)
        #    
        #    # Calculate the error by subtracting the predicted label from the actual label
        #    error = y - y_pred
        #    
        #    # Update the weights and bias of the perceptron based on the error and learning rate
        #    self.w += learning_rate * error * x
        #    self.b += learning_rate * error
            


    def predict(self, x: np.ndarray) -> int:
        weighted_sum=np.dot(x,self.w)+self.b
        return 1 if weighted_sum > 0 else -1
    
    