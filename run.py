from helper import *
from optuna_hybrid import *
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":

################################# Set parameters #############################

    # set countermeasure out of 'gauss', 'rdi' and 'desync'
    countermeasure = 'rdi'

    # set the hyperparameters for the CAE model
    activation = 'selu'
    dense_nodes = 512
    n_layers = 6
    optimizer = 'Adam'
    total_pool = 35

    possible_pool_lists = find_pool_lists(n_layers, total_pool)
    possible_n_filter_0_list = possibilities(total_pool)

    n_filter_0 = possible_n_filter_0_list[3]
    width_reshape = int(700 / total_pool)
    s_latent = width_reshape * n_filter_0

    dilation = [1, 2, 2, 2, 2, 3, 2, 2, 4, 2, 2, 3]
    kernel_size = [1, 14, 13, 13, 16, 6, 12, 2, 16, 7, 3, 13]
    n_filters = [26, 242, 228, 107, 96, n_filter_0, n_filter_0, 96, 107, 228, 242, 26]
    pool = possible_pool_lists[16]

    reverse_pool = pool.copy()
    reverse_pool.reverse()
    pool += reverse_pool
    CAE_epochs = 161
    CAE_batch_size = 128

    # use command line parameters if provided
    if len(sys.argv) > 1:
        if sys.argv[1] in ['gauss', 'rdi', 'desync']:
            countermeasure = sys.argv[1]
        else:
            raise ValueError(f"Invalid input: For the countermeasure choos one of the following: 'gauss', 'rdi', 'desync'")

################################# Make data folders #############################

    model_folder = "model_data"
    
    if os.path.exists(model_folder) == False:
        os.mkdir(model_folder)
        os.mkdir(model_folder + "/CAE")
        os.mkdir(model_folder + "/CNN")
        os.mkdir(model_folder + "/GE")
        os.mkdir(model_folder + "/MLP")
    else:
        i = 1
        while os.path.exists(model_folder + "(" + str(i) + ")") == True:
            i += 1
        model_folder = model_folder + "(" + str(i) + ")"
        os.mkdir(model_folder)
        os.mkdir(model_folder + "/CAE")
        os.mkdir(model_folder + "/CNN")
        os.mkdir(model_folder + "/GE")
        os.mkdir(model_folder + "/MLP")

################################# Add Noise #################################

    clean_trace_dir = "data/clean.h5"
    noisy_traces_dir = "data/noisy.h5"
    denoised_data_dir = "data/denoised.h5"

    # load traces
    print('load traces add_noise...')
    (x_train_desync0, x_test_desync0) = load_ascad_add_noise(clean_trace_dir)
    print(np.shape(x_train_desync0))
    print(np.shape(x_test_desync0))

    print('generate noisy traces')
    generate_traces(clean_trace_dir, noisy_traces_dir,
                    x_train_desync0, x_test_desync0, countermeasure, 8)

################################# Denoise #################################

    # load traces
    print('load traces denoise...')
    X_noisy_o = load_ascad_denoise(noisy_traces_dir)
    Y_clean_o = load_ascad_denoise(clean_trace_dir)
    X_noisy = scale(X_noisy_o)
    Y_clean = scale(Y_clean_o)

    print('train CAE...')

    # create the CAE model
    autoencoder = optuna_model(0.0001, 
                    len(X_noisy[0]),
                    activation,
                    dense_nodes,
                    dilation,
                    kernel_size,
                    optimizer,
                    n_layers,
                    n_filters,
                    pool,
                    width_reshape,
                    s_latent)

    # set checkpoints to save the CAE model and stats
    checkpoint_path = model_folder + "/" "CAE/CAE_weights_best_val.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    log_csv = CSVLogger(model_folder + "/" + "CAE" + "/" + 'model_data.csv', append=False)
    
    callbacks_list = [checkpoint, log_csv]

    # train the CAE model
    autoencoder.fit(X_noisy[:40000], Y_clean[:40000], 
                    validation_data=(X_noisy[40000:], Y_clean[40000:]),
                    epochs=CAE_epochs,
                    batch_size=CAE_batch_size,
                    verbose=2,
                    callbacks=callbacks_list)

    # save the final model
    autoencoder.save(model_folder + "/" "CAE/CAE_weights_last.hdf5")

    # denoise the data
    print('generate denoised traces...')
    generate_traces_denoise(autoencoder, noisy_traces_dir, denoised_data_dir, X_noisy, Y_clean_o)


################################# Attack #################################

    # set parameters for both the models and the attack
    batch_size = 128
    nb_attacks = 100
    input_size = 700
    learning_rate = 0.0001
    nb_traces_attacks = 10000

    # key for ascad fixed key
    real_key = 224

    ################ CNN ###############

    # Load the profiling traces
    (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(denoised_data_dir, load_metadata=True)

    (X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

    # make cnn model
    model = cnn_best(len(X_profiling[0]))

    # record the metrics
    history = train_model(X_profiling[10000:45000], 
                        Y_profiling[10000:45000], 
                        X_profiling[45000:], 
                        Y_profiling[45000:], 
                        'CNN', 
                        model, 
                        model_folder,
                        epochs=1000,
                        batch_size=batch_size, 
                        max_lr=learning_rate)

    # attack on the test traces
    predictions = model.predict(X_attack)
    avg_rank = np.array(perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=2), int)

    # generate plot and save model    
    print("\n t_GE = ")
    print(np.where(avg_rank<=0))
    np.save(model_folder + '/GE/' + '_GE_CNN', avg_rank)
    plt.plot(avg_rank)
    plt.savefig(model_folder + '/GE/' + '_GE_CNN.png')
    print("CNN Attacking finished...")

    ############## MLP ################

    # make mlp model
    model = mlp_best(len(X_profiling[0]))

    # record the metrics
    history = train_model(X_profiling[10000:45000], 
                        Y_profiling[10000:45000], 
                        X_profiling[45000:], 
                        Y_profiling[45000:], 
                        'MLP', 
                        model, 
                        model_folder,
                        epochs=100, 
                        batch_size=batch_size, 
                        max_lr=learning_rate)

    # attack on the test traces
    predictions = model.predict(X_attack)
    avg_rank = np.array(perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=2), int)

    # generate plot and save model
    print("\n t_GE = ")
    print(np.where(avg_rank<=0))
    np.save(model_folder + '/GE/' + '_GE_MLP', avg_rank)
    plt.plot(avg_rank)
    plt.savefig(model_folder + '/GE/' + '_GE_MLP.png')
    print("Attacking finished...")
