import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# one feature = x

def generate_training_data():
    
    target_training = []
    x = 0
    x_training = []

    random_seed = random.default_rng(2267)
    for i in range(12):
        noise = random_seed.normal(0, 0.25)   #gaussian distribution
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

    random_seed = random.default_rng(2267)
     
    for i in range(120):
        noise = random_seed.normal(0, 0.25)   #gaussian distribution
        target = np.sin(4*np.pi*x + (np.pi/2)) + noise
        target_validation.append(target)
        #print("\ni = ", i+1, ", x = ", x, ", target = ", target)
        x_validation.append(x)
        x += 1/119

    return x_validation, target_validation


def train_model(M, x_training, target_training, x_validation, target_validation, t_training, t_validation):
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

    X_validation = np.empty((120, M+1)) #width of matrix is M+1 because of the dummy feature

    for i in range(120):
        row = []
        for exp in range(M+1):
            if exp == 0:
                row.append(1)
            else:
                row.append(np.power(x_validation[i], exp))
        #print("row ", i, ": ", row)
        X_validation[i] = row

    #print("X = \n", X)

    #compute the parameters 
    X_transpose = X.transpose()
    X_product_inv = np.linalg.inv((X_transpose.dot(X)))

    w = (X_product_inv.dot(X_transpose)).dot(t_training)

    #print("w = \n", w)

    # now have the parameters, are able to compute the training and validation error


    # training error C(w) = 1/N * (Xw - t)_trans * (Xw - t)
    t_error_matrix = X.dot(w) - t_training
    training_error = 1/12*(t_error_matrix.transpose().dot(t_error_matrix))


    # validation error C(w) = 1/N * (Xw - t)_trans * (Xw - t)
    v_error_matrix = X_validation.dot(w) - t_validation
    validation_error = 1/120*(v_error_matrix.transpose().dot(v_error_matrix))


    # plotting 
    x_range = np.arange(0,1.025,0.025) # 20 examples
    # *MAYBE CHANGE THIS TO VECTOR WAY*
    #print("x range = ", x_range)
    pred = []
    true = []
     
    # assuming M = 7
    for i in range(len(x_range)):
        #print("i = ", i)
        #print("x = ", x_range[i])
        pred_calc = 0
        for j in range(M+1):
            pred_calc += w[j,0]*(x_range[i]**j)

        pred.append(pred_calc)       
        true.append(np.sin(4*np.pi*x_range[i] + (np.pi/2)))
        #pred.append(w[0,0] + w[1,0]*x_range[i] + w[2,0]*(x_range[i]**2) + w[3,0]*(x_range[i]**3) + w[4,0]*(x_range[i]**4) + w[5,0]*(x_range[i]**5) + w[6,0]*(x_range[i]**6) + w[7,0]*(x_range[i]**7))
        #print("\npredicited value of x = ", x_range[i], " is ", pred[i])

    #plt.subplot(4,3,M+1)
    title = "M = " + str(M)
    plt.plot(x_range, pred, 'o:b', ms=3)
    plt.plot(x_range, true, 'r', ms=3)
    plt.plot(x_training, target_training, 'o:y', ms=5)
    plt.plot(x_validation, target_validation, 'o:g', ms=2)
    plt.title(title)
    plt.show()

    return training_error, validation_error




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
    t_training = np.empty((12,1))
    t_validation = np.empty((120,1))

    for i in range(12): t_training[i] = target_training[i]
    
    for i in range(120): t_validation[i] = target_validation[i]
                 
    #print("target matrix: \n", t)

    training_error = np.empty(12)
    validation_error = np.empty(12)

    for i in range(12):
      training_error[i], validation_error[i] = train_model(i, x_training, target_training, x_validation, target_validation, t_training, t_validation)
    
    plt.show()

    #print("############################################\n", training_error)
      
    plt.plot(training_error, 'b:x')
    plt.plot(validation_error, 'g:x')
    plt.title("Training (blue) and Validation (green) Error vs. Model Capacity")
    plt.show()



main()