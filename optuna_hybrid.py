from helper import *
import optuna
from optuna.integration import KerasPruningCallback
import numpy as np
import sys

def find_pool_lists(n, m):
  """
  Find all possible lists of length n with the product of all elements equal to m
  - n: length of lists
  - m: product of elements
  """
  
  # list of possible lists
  lists = []

  # if the length reached, return an empty list
  if n == 0:
    return [[]]

  # loop through possible numbers
  for i in range(1, m+1):

    if m % i == 0:
      new_list = [i]

      # recursively generate the rest of the list
      rest_of_list = find_pool_lists(n-1, m//i)

      # add the generated lists to the current list
      for l in rest_of_list:
        lists.append(new_list + l)

  # filter out lists with incorrect product
  lists = [l for l in lists if np.product(l) == m]

  return lists

def possibilities(total_pool):
    """
    make a list of possible filter amounts in the first layer of decoder
    - total_pool: product of all pool layers
    """
    pos_list = []
    
    # bruteforce possible dense node amounts
    for i in range(1, 512):
        latent = (700 / total_pool) * i

        # check if larger than latent space and smaller than upper boundary
        if latent > 700 and latent < 1500:
            pos_list.append(i)
    
    return pos_list

def optuna_model(lr, input_length, activation, dense_nodes, dilation, kernel_size, optimizer, n_layers, filters, pool, width_reshape, s_latent):
    """
    create the CAE off which the parameters have been picked by Optuna
    """

    img_input = Input(shape=(input_length, 1))

    # create encoder layers
    x = conv(img_input, filters[0], kernel_size[0], activation, pool[0], dilation[0])

    for i in range(1, n_layers):
        x = conv(x, filters[i], kernel_size[i], activation, pool[i], dilation[i])
    x = Flatten(name='flatten')(x)

    # create latent space layers
    x = Dense(dense_nodes, activation=activation)(x)
    x = Dense(s_latent, activation=activation)(x)

    # create decoder layers
    x = Reshape((width_reshape, filters[n_layers]))(x)
    for i in range(n_layers, n_layers + n_layers):
        x = deconv(x, filters[i], kernel_size[i], activation, pool[i], dilation[i])
    
    x = deconv(x, 1, 2, 'sigmoid', 0)

    model = Model(img_input, x)

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def objective(trial):
    """
    objective function which optuna tries to optimize
    """

    # get data
    X_noisy_o = load_ascad_denoise(noisy_traces_dir)
    Y_clean_o = load_ascad_denoise(clean_trace_dir)
    
    X_noisy = scale(X_noisy_o)
    Y_clean = scale(Y_clean_o)

    # suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 6)
    epoch = trial.suggest_int('epoch', 10, 200)
    batch = 128
    activation = trial.suggest_categorical('activation', ['selu', 'relu'])
    optimizer = trial.suggest_categorical('opt_func', ['RMSprop', 'SGD', 'Adam'])
    dense_nodes = trial.suggest_int('dense_nodes', 1, 700)

    # suggest hyperparameters per layer
    dilation = []
    kernel_size = []
    n_filters = []

    for i in range(n_layers * 2):
        kernel_size.append(trial.suggest_int('kernel_size_' + str(i), 1, 16))
        dilation.append(trial.suggest_int('dilation_' + str(i), 1, 4))
    
    for i in range(n_layers-1):
        n_filters.append(trial.suggest_int('filter_encoder_' + str(i), 1, 256))

    # pick the product of all max pool sizes
    # it has to be one of these numbers to make sure that the
    # reshape width before the decoder is a positive integer
    total_pool = trial.suggest_categorical('total_pool', [1, 2, 4, 5, 7, 10, 14, 20, 25, 28, 35, 50, 70, 100, 140, 175, 350])

    # make a list of all possible max pools for each layer
    possible_pool_lists = find_pool_lists(n_layers, total_pool)

    if len(possible_pool_lists) > 1:
        # pick one of these possible lists
        picked_list = trial.suggest_int('pool_list_number', 1, len(possible_pool_lists)-1)
        pool = possible_pool_lists[picked_list]
    else:
        pool = possible_pool_lists[0]

    # get the filter amount for the first layer in decoder
    possible_n_filter_0_list = possibilities(total_pool)

    if len(possible_n_filter_0_list) > 1:
        picked_filter_amount = trial.suggest_int('filter_decoder_0', 1, len(possible_n_filter_0_list)-1)
        n_filters.append(possible_n_filter_0_list[picked_filter_amount])
    else:
        n_filters.append(possible_n_filter_0_list[0])
        picked_filter_amount = 0

    width_reshape = int(700 / total_pool)
    s_latent = width_reshape * possible_n_filter_0_list[picked_filter_amount]

    # add deconv parameters to lists
    reverse_filters = n_filters.copy()
    reverse_filters.reverse()
    n_filters += reverse_filters

    reverse_pool = pool.copy()
    reverse_pool.reverse()
    pool += reverse_pool

    # print chosen hyperparameters
    print('Pool layers:')
    print(pool)
    print('Filters: ')
    print(n_filters)
    print("Pool total = " + str(total_pool))
    print("s latent = " + str(s_latent))
    print("n_filter_0 = " + str(possible_n_filter_0_list[picked_filter_amount]))
    print("dilation: ")
    print(dilation)
    print("Dense Nodes: " + str(dense_nodes))
    print("Kernel Size: ")
    print(kernel_size)
    print("Number of layers: " + str(n_layers))

    # create model
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

    # fit model
    autoencoder.fit(X_noisy[:40000], Y_clean[:40000], 
                    validation_data=(X_noisy[40000:], Y_clean[40000:]),
                    epochs=epoch,
                    callbacks=[KerasPruningCallback(trial, "val_loss")],
                    batch_size=batch,
                    verbose=2)

    # calculate validation score
    score = autoencoder.evaluate(X_noisy[40000:], Y_clean[40000:], verbose=0)
    return score[0]

if __name__ == "__main__":

    ################### set parameters ###################

    # data paths
    clean_trace_dir = "data/clean.h5"
    noisy_traces_dir = "data/noisy.h5"

    # optuna settings
    optuna_study_name='random_delay_interrupt'
    optuna_storage = 'mysql://username:password@ip_adress:port/database_name'

    # set countermeasure type out of 'gauss', 'rdi', 'desync'
    countermeasure = 'rdi'

    ################### cmd line parameters ###################

    if len(sys.argv) > 1:
        if sys.argv[1] in ['gauss', 'rdi', 'desync']:
            countermeasure = sys.argv[1]
        else:
            raise ValueError(f"Invalid input: For the countermeasure choos one of the following: 'gauss', 'rdi', 'desync'")
    if len(sys.argv) > 2:
        if sys.argv[2] == 'None':
            optuna_storage = None
        else:
            optuna_storage = sys.argv[2]
    if len(sys.argv) > 3:
        optuna_study_name = sys.argv[3]
    
    ################### start of code ###################

    # load traces
    print('load traces add_noise...')
    (x_train, x_test) = load_ascad_add_noise(clean_trace_dir)

    # add noise
    print('generate noisy traces')
    generate_traces(clean_trace_dir, noisy_traces_dir,
                    x_train, x_test, countermeasure, 8)
    
    # start optuna study
    study = optuna.create_study(direction="minimize", study_name=optuna_study_name, storage=optuna_storage, load_if_exists=True)
    study.optimize(objective, n_trials=1000, gc_after_trial=True)
    print("Number of finished trials: {}".format(len(study.trials)))

    # print the best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
