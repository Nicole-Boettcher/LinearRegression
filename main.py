import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# one feature = x

def generate_training_data():
    
    target_training = []
    x = 0
    x_training = []

    for i in range(12):
        noise = random.normal(0, 0.25)   #gaussian distribution 
        target = np.sin(4*np.pi*x + (np.pi/2)) + noise
        target_training.append(target)
        print("\ni = ", i+1, ", x = ", x, ", target = ", target)
        x_training.append(x)
        x += 1/11


    return x_training, target_training

def generate_validation_data():
    target_validation = []

    x = 0
    x_validation = []

    for i in range(120):
        noise = random.normal(0, 0.25)   #gaussian distribution 
        target = np.sin(4*np.pi*x + (np.pi/2)) + noise
        target_validation.append(target)
        #print("\ni = ", i+1, ", x = ", x, ", target = ", target)
        x_validation.append(x)
        x += 1/119

    return x_validation, target_validation


def train_model(M, x_training, target_training, x_validation, target_validation, t):
    print("\nM = ", M)

    #create X
    X = np.empty((12, M+1)) #width of matrix is M+1 because of the dummy feature

    for i in range(12):
        row = []
        for exp in range(M+1):
            if exp == 0:
                row.append(1)
            else:
                row.append(np.power(x_training[i], exp))
        #print("row ", i, ": ", row)
        X[i] = row

    #print("X = \n", X)

    #compute the parameters 
    X_transpose = X.transpose()
    X_product_inv = np.linalg.inv((X_transpose.dot(X)))

    w = (X_product_inv.dot(X_transpose)).dot(t)

    #print("w = \n", w)

    # now have the parameters, are able to compute the training and validation error

    #f_pred(x) = input matrix * w
    # training error C(w) = 1/N * (Xw - t)_trans * (Xw - t)
    # the X above in the same X used to train the data and so is the t

    # plotting 
    x_range = np.arange(0,1.05,0.025) # 20 examples
    # *MAYBE CHANGE THIS TO VECTOR WAY*
    #print("x range = ", x_range)
    pred = []
    true = []
     
    # assuming M = 7
    for i in range(len(x_range)):
        print("i = ", i)
        print("x = ", x_range[i])
        pred_calc = 0
        for j in range(M+1):
            pred_calc += w[j,0]*(x_range[i]**j)

        pred.append(pred_calc)       
        true.append(np.sin(4*np.pi*x_range[i] + (np.pi/2)))
        #pred.append(w[0,0] + w[1,0]*x_range[i] + w[2,0]*(x_range[i]**2) + w[3,0]*(x_range[i]**3) + w[4,0]*(x_range[i]**4) + w[5,0]*(x_range[i]**5) + w[6,0]*(x_range[i]**6) + w[7,0]*(x_range[i]**7))
        print("\npredicited value of x = ", x_range[i], " is ", pred[i])

    plt.subplot(4,3,M+1)
    title = "M = " + str(M)
    plt.plot(x_range, pred, 'o:b', ms=3)
    plt.plot(x_range, true, 'r', ms=3)
    plt.plot(x_training, target_training, 'o:y', ms=5)
    plt.plot(x_validation, target_validation, 'o:g', ms=2)
    plt.title(title)
    #plt.show()




def main():
    print("Start")

    x_training, target_training = generate_training_data()
    #print(x_training)
    #print(target_training)

    #plt.plot(x_training, target_training, 'o:b')
    #plt.title("Training data")
    #plt.show()

    x_validation, target_validation = generate_validation_data()
    #print(x_validation)
    #print(target_validation)

    #plt.plot(x_validation, target_validation, 'o:r')
    #plt.title("Validation data")
    #plt.show()

    #train twelve models 
    # for each model steps are:
    # 1. Declare X (include dummy 1) and t as matricies -- DONE
    # 2. Create X transpose
    # 3. w = (Xtrans*X)-1 * Xtrans*t  this is the solution to setting the gradient of the cost function to 0
    # 4. Compute the training error and validation error
    # 5. Plot the prediction function and the true curve
    # 
    # Can have a function that takes in M and the scales the function

    # 1
    t = np.empty((12,1))

    for i in range(12):
        t[i] = target_training[i]
                 
    print("target matrix: \n", t)

    for i in range(12):
      train_model(i, x_training, target_training, x_validation, target_validation, t)
    
    plt.show()
      
    #train_model(7, x_training, t) 



main()