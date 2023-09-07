import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn import tree
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import classification_report
from trustee import RegressionTrustee
from sklearn.metrics import r2_score
import graphviz
FLAGS=None

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual[i][1] == 1 and y_pred[i][1] >= 0.5:
            TP += 1
        if y_pred[i][1] >= 0.5 and y_actual[i][1] == 0:
            FP += 1
        if y_actual[i][1] == 0 and y_pred[i][1] < 0.5:
            TN += 1
        if y_pred[i][1] <= 0.5 and y_actual[i][1] == 1:
            FN += 1

    return(TP, FP, TN, FN)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--n',
        type=int,
        required=True,
        help='ID of training.'
    )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='ID of model. The models to load should be placed in the model directory, named \"classifier-N-K-[0,1,2,3].h5\"'
    )
    FLAGS, unparsed =parser.parse_known_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

    n_epoch = 10
    n_batch = 50

    ss=FLAGS.k//90

    X_train=np.load(os.path.join("data","X_train-%.3d.npy"%ss),allow_pickle=True)
    oldshape = X_train.shape
    print("X_train Orignal:"+str(X_train.shape))
    X_train = X_train.reshape(X_train.shape[0], -1) 
    print("X_train reshape:"+str(X_train.shape))
    newshape = X_train.shape
    X_test=np.load(os.path.join("data","X_test-%.3d.npy"%ss),allow_pickle=True)
    print("X_test Orignal:"+str(X_test.shape))
    X_test = X_test.reshape(X_test.shape[0], -1) 
    print("X_test reshape:"+str(X_test.shape))
    Y_train=np.load(os.path.join("data","Y_train-%.3d.npy"%ss),allow_pickle=True)
    print("Y_train Orignal:"+str(Y_train.shape))
    Y_train = Y_train.reshape(Y_train.shape[0], -1) 
    print("Y_train reshape:"+str(Y_train.shape))
    Y_test=np.load(os.path.join("data","Y_test-%.3d.npy"%ss),allow_pickle=True)
    print("Y_test Orignal:"+str(Y_test.shape))
    Y_test = Y_test.reshape(Y_test.shape[0], -1) 
    print("Y_test reshape:"+str(Y_test.shape))
    '''
    example:
    (193072, 4, 32, 8)
    (193072, 1024)

    (48270, 4, 32, 8)
    (48270, 1024)

    (193072, 2)
    (193072, 2)
    
    (48270, 2)
    (48270, 2)
    '''
    predictions_train=[]
    predictions_test=[]
    for i in range(1):
        model=tf.keras.models.load_model(os.path.join("model","classifier-"+str(FLAGS.n)+"-"+str(FLAGS.k)+"-"+str(i)+".h5"),custom_objects={"metric":F1Score(num_classes=2)})
        '''Added'''
        
        
        
        new_input = tf.keras.layers.Input(shape=(newshape[1],))
        
        reshaped_input = tf.keras.layers.Reshape((oldshape[1], oldshape[2], oldshape[3]))(new_input)
        
        # Get the output of your trained model
        output = model(reshaped_input)

        # Create a new model
        new_model = tf.keras.Model(inputs=new_input, outputs=output)
                
        
        
        
        '''
        '''
        predictions_train.append(new_model.predict(X_train))
        predictions_test.append(new_model.predict(X_test))
    new_model.summary()

    
    # performing predictions on the test dataset
    y_pred = new_model.predict(X_test)

    # Evaluate model accuracy
    print("Model R2-score:")
    print(r2_score(Y_test, y_pred))

    # Initialize Trustee and fit for classification models
    trustee = RegressionTrustee(expert=new_model)
    trustee.fit(X_train, Y_train, num_iter=50, num_stability_iter=10, samples_size=0.3, verbose=True)

    # Get the best explanation from Trustee
    dt, pruned_dt, agreement, reward = trustee.explain()
    print(f"Model explanation training (agreement, fidelity): ({agreement}, {reward})")
    print(f"Model Explanation size: {dt.tree_.node_count}")
    print(f"Top-k Prunned Model explanation size: {pruned_dt.tree_.node_count}")

    # Use explanations to make predictions
    dt_y_pred = dt.predict(X_test)
    pruned_dt_y_pred = pruned_dt.predict(X_test)

    # Evaluate accuracy and fidelity of explanations
    print("Model explanation global fidelity:")
    print(r2_score(y_pred, dt_y_pred))
    print("Top-k Model explanation global fidelity:")
    print(r2_score(y_pred, pruned_dt_y_pred))

    print("Model explanation R2-score:")
    print(r2_score(Y_test, dt_y_pred))
    print("Top-k Model explanation R2-score:")
    print(r2_score(Y_test, pruned_dt_y_pred))

    # Output decision tree to pdf
    dot_data = tree.export_graphviz(
        dt,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("dt_explanation")

    # Output pruned decision tree to pdf
    dot_data = tree.export_graphviz(
        pruned_dt,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("pruned_dt_explation")
