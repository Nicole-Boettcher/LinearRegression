import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# one feature = x

def generate_training_data():
    
    target_training = []
    x = 0
    x_training = []

    random_seed = random.default_rng(2276)
    for i in range(12):
        noise = random_seed.normal(0, 0.25)   #gaussian distribution
        #noise = 0
        target = np.sin(4*np.pi*x + (np.pi/2)) + noise
        target_training.append(target)
        #print("\ni = ", i+1, ", x = ", x, ", target = ", target)
        x_training.append(x)
        x += 1/11


    return x_training, target_training

def generate_validation_data():
    target_validation = []

    x = 0
    x_validation = []
    true_function = []

    random_seed = random.default_rng(2267)
     
    for i in range(120):
        noise = random_seed.normal(0, 0.25)   #gaussian distribution
        true = np.sin(4*np.pi*x + (np.pi/2))
        target = true + noise
        true_function.append(true)
        target_validation.append(target)
        #print("\ni = ", i+1, ", x = ", x, ", target = ", target)
        x_validation.append(x)
        x += 1/119

    return x_validation, target_validation, true_function


def train_model(M, x_training, target_training, x_validation, target_validation, t_training, t_validation):
    #print("\n##################################################\nM = ", M)

    #print("x_training = ", x_training)
    #print("target_training = ", target_training)
    col_target_training = np.empty((12,1))
    for i in range(12):
        col_target_training[i] = target_training[i]

    #print("target training matrix? = ", col_target_training)
    #print("size = ", (np.array(target_training).transpose()).size)

    ################################
    # create X for the trainging set
    ################################

    X = np.empty((12, M+1)) #width of matrix is M+1 because of the dummy feature

    for i in range(12):
        row = []
        for exp in range(M+1):
            if exp == 0:
                row.append(1)
            else:
                row.append(np.power(x_training[i], exp))
        print("row ", i, ": ", row)
        X[i] = row

    #print("X matrix = ", X)
    #print("X size = ", X.shape)

    #################################
    # create X for the validation set
    #################################

    X_validation = np.empty((120, M+1)) #width of matrix is M+1 because of the dummy feature

    for i in range(120):
        row = []
        for exp in range(M+1):
            if exp == 0:
                row.append(1)
            else:
                row.append(np.power(x_validation[i], exp))
        ##print("row ", i, ": ", row)
        X_validation[i] = row

    ####################################################
    # compute the parameters for the prediction function 
    ####################################################

    print("X = ", X)
    X_transpose = X.transpose()
    print("transpose = ", X_transpose)
    X_product_inv = np.linalg.inv((X_transpose.dot(X)))
    print("X_prod_inv = ", X_product_inv)


    w = (X_product_inv.dot(X_transpose)).dot(col_target_training)

    #print("w = \n", w)

    ################################################################################
    # now have the parameters, are able to compute the training and validation error
    ################################################################################

    # training error C(w) = 1/N * (Xw - t)_trans * (Xw - t)
    t_error_matrix = X.dot(w) - t_training
    training_error = 1/12*(t_error_matrix.transpose().dot(t_error_matrix))


    # validation error C(w) = 1/N * (Xw - t)_trans * (Xw - t)
    v_error_matrix = X_validation.dot(w) - t_validation
    validation_error = 1/120*(v_error_matrix.transpose().dot(v_error_matrix))


    #########################################################
    # populate the predicted and true function with x = [0,1]
    #########################################################

    x_range = np.arange(0,1.025,0.025) # 20 samples
    # *MAYBE CHANGE THIS TO VECTOR WAY*
    ##print("x range = ", x_range)
    pred = []
    true = []
     
    for i in range(len(x_range)):
        #print("i = ", i)
        #print("x = ", x_range[i])
        pred_calc = 0
        for j in range(M+1):    #for M = 11
            pred_calc += w[j,0]*(x_range[i]**j) #(w[0,0] + w[1,0]*x_range[i] + w[2,0]*(x_range[i]**2) + w[3,0]*(x_range[i]**3) + w[4,0]*(x_range[i]**4) + w[5,0]*(x_range[i]**5) + w[6,0]*(x_range[i]**6) + w[7,0]*(x_range[i]**7))

        pred.append(pred_calc)       
        true.append(np.sin(4*np.pi*x_range[i] + (np.pi/2)))
        
        #print("\npredicited value of x = ", x_range[i], " is ", pred[i])

    ##########
    # plotting
    ##########

    plt.subplot(4,3,M+1)
    title = "M = " + str(M)
    plt.plot(x_range, pred, 'o:b', ms=3)
    plt.plot(x_range, true, 'r', ms=2)
    plt.plot(x_training, target_training, 'o:y', ms=6)
    plt.plot(x_validation, target_validation, 'og', ms=1)
    if M == 0:  
        plt.legend(['Prediction', 'True', 'Training', 'Validation'])
    plt.title(title)

    return training_error, validation_error


def standardize_features(x_training, x_validation):
    x_t = np.array(x_training)
    x_v = np.array(x_validation)
    print("reshaped training: ", x_t.reshape(-1,1))
    x_training_norm = sc.fit_transform(x_t.reshape(-1,1))
    x_validation_norm = sc.transform(x_v.reshape(-1,1))
    return x_training_norm, x_validation_norm

def train_M11(x_training_norm, x_validation_norm, target_training, t_training, t_validation, t_true, lam):

    col_target_training = np.empty((12,1))
    for i in range(12):
        col_target_training[i] = target_training[i]


    X = np.empty((12, 12)) #width of matrix is M+1 because of the dummy feature

    for i in range(12):
        row = []
        for exp in range(12):
            if exp == 0:
                row.append(1)
            else:
                row.append(np.power(x_training_norm[i,0], exp))
        #print("row ", i, ": ", row)
        X[i] = row


    X_validation = np.empty((120, 12)) #width of matrix is M+1 because of the dummy feature

    for i in range(120):
        row = []
        for exp in range(12):
            if exp == 0:
                row.append(1)
            else:
                row.append(np.power(x_validation_norm[i,0], exp))
        ##print("row ", i, ": ", row)
        X_validation[i] = row


    #print("X matrix = ", X)
    #print("X size = ", X.shape)
    num_of_lam = 9
    training_error = np.empty(num_of_lam)
    validation_error = np.empty(num_of_lam)
    identity = np.identity(12)
    lambdas = np.empty(num_of_lam)

    for i in range(num_of_lam):
        # for each lambda we will make a new B
        lambdas[i] = lam
        B = lam*identity
        #print("B = ", B)

        X_tranpose = X.transpose()
        product = X_tranpose.dot(X) + 6*B
        product_inverse = np.linalg.inv(product)
        w = product_inverse.dot(X_tranpose).dot(col_target_training)
        #print("lambda = ", lam)
        #print("w = ", w)
        
        x_range = sc.fit_transform(np.arange(-1,1,0.05).reshape(-1,1)) # 20 samples STANDIZE
        #print("x range = ", x_range)

        pred = []
        #true = []
        
        for k in range(len(x_range)):
            #print("i = ", i)
            #print("x = ", x_range[i])
            pred_calc = 0
            for j in range(12):    #for M = 11
                pred_calc += w[j,0]*(x_range[k]**j) 


            pred.append(pred_calc)       
            #true.append(np.sin(np.pi*x_range[k] + (np.pi/2)))
            
        ##########
        # plotting
        ##########

        # need to calculate validation e
        t_error_matrix = X.dot(w) - t_training
        training_error[i] = 1/12*(t_error_matrix.transpose().dot(t_error_matrix))


        # validation error C(w) = 1/N * (Xw - t)_trans * (Xw - t)
        v_error_matrix = X_validation.dot(w) - t_validation
        validation_error[i] = 1/120*(v_error_matrix.transpose().dot(v_error_matrix))


        #standard_pred = sc.transform(np.array(pred).reshape(-1,1))
        print("lambda = ", lam)
        print("predictions: ", pred)
        #print("standward pred: ", standard_pred)

        title = "Lambda = " + str(lam)
        plt.subplot(3,3,i+1)
        plt.plot(x_range, pred, 'o:b', ms=3)
        plt.plot(x_validation_norm, t_true, 'r', ms=2)
        plt.plot(x_training_norm, target_training, 'o:y', ms=6)
        plt.plot(x_validation_norm, t_validation, 'og', ms=1)
        if i == 0: plt.legend(['Prediction', 'True', 'Training'])
        plt.title(title)
        plt.ylim(-2,2)
        #plt.plot(x_validation, target_validation, 'o:g', ms=2)
        lam += 0.005

    return training_error, validation_error, lambdas

def plot_error(t_error, v_error, avg_sq_error, x_axis, x_title):

    plt.plot(x_axis, t_error, 'b:x')
    plt.plot(x_axis, v_error, 'g:x')
    plt.plot(x_axis, avg_sq_error, 'r:x')
    plt.title("Training, Validation and Average Squared Error vs. " + x_title)
    plt.xlabel(x_title)
    plt.ylabel("Error")
    plt.legend(['Training Error', 'Validation Error', 'Average Squared Error'])
    plt.show()


def average_squared_error(validation_targets, true_targets, x_range):
    # want to calulate the difference bewteen the validation set targets and the true targets and average them
    sq_error = 0

    for i in range(len(x_range)):
        sq_error += math.pow(validation_targets[i,0] - true_targets[i], 2)

    #print("sq error: ", sq_error/len(x_range))

    return sq_error/len(x_range)


def main():

    #####################################
    # create training and validation data
    #####################################

    x_training, target_training = generate_training_data()
    x_validation, target_validation, true_values = generate_validation_data()

    t_training = np.empty((12,1))
    t_validation = np.empty((120,1))
    t_true = np.empty((120,1))

    for i in range(12): t_training[i] = target_training[i]
    
    for i in range(120):
        t_validation[i] = target_validation[i]
        t_true[i] = true_values[i]
                 
    ##############################
    # create true function targets
    ##############################

    training_error = np.empty(12)
    validation_error = np.empty(12)
    model_num = np.arange(0,12)
    
    for i in range(12):
      training_error[i], validation_error[i] = train_model(i, x_training, target_training, x_validation, target_validation, t_training, t_validation)

    avg_sqr_error_value = average_squared_error(t_validation, t_true, x_validation)
    avg_sqr_error = avg_sqr_error_value*np.ones(12)

    plt.show()

    plot_error(training_error, validation_error, avg_sqr_error, model_num, "Model Capacity")

    ######################################################
    # now train with new x features which are standardized
    ######################################################

    x_training_norm, x_validation_norm = standardize_features(x_training, x_validation)
    starting_lambda = 0
    training_error_reg, validation_error_reg, lambdas = train_M11(x_training_norm, x_validation_norm, target_training, t_training, t_validation, t_true, starting_lambda)
    plt.show()

    avg_sqr_error_value_reg = average_squared_error(t_validation, t_true, x_validation_norm)    #x_validation_norm is a matrix and x_validation is just an array
    avg_sqr_error_reg = avg_sqr_error_value_reg*np.ones(9)

    plot_error(training_error_reg, validation_error_reg, avg_sqr_error_reg, lambdas, "Lambda")

    # regularization with underfitting 
    starting_lambda = 0.1
    training_error_reg, validation_error_reg, lambdas = train_M11(x_training_norm, x_validation_norm, target_training, t_training, t_validation, t_true, starting_lambda)
    plt.show()

    plot_error(training_error_reg, validation_error_reg, avg_sqr_error_reg, lambdas, "Lambda")


main()