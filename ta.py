import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys
from random import randrange
import h5py
import os
from multiprocessing import Pool
import random

"""
Pass a 1 or 0 when running the file to make it pooled or not pooled.
If nothing is passed a normal template attack is used.
"""
pooled = len(sys.argv) > 1 and sys.argv[1] == "1"
print("running pooled template attack" if pooled else "running normal template attack")
hamming = [bin(n).count("1") for n in range(256)]


sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16) 

def cov(x, y):
    """
    np.cov(x, y) returns 2x2 matrix with:
        X         Y
    X   cov(x, x) cov(x, y)
    Y   cov(y, x) cov(y, y)

    So that's why [0][1] is taken.
    """
    return np.cov(x, y)[0][1]

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

# Loads the profiling and attack datasets from the ASCAD database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['plaintext'])

denoised_data_dir = "data/denoised.h5"
(X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(denoised_data_dir, load_metadata=True)

real_key = 224

print(Y_attack)
print(Y_attack.shape)

# select key byte to attack
key_byte = 1

# Sbox output for every plaintext. Only for 1st byte?
outputSbox = [sbox[plt_profiling[i][key_byte] ^ Y_profiling[key_byte]] for i in range(len(plt_profiling))]

# Hamming weight of every Sbox output
outputSboxHW   = [hamming[s] for s in outputSbox]

# Hamming weight has 9 values: 0-8
TracesHW = [[] for _ in range(9)]

# Put every trace into the correct class according to hamming weight
for i in range(len(X_profiling)):
    HW = outputSboxHW[i]
    TracesHW[HW].append(X_profiling[i])

TracesHW = [np.array(TracesHW[HW]) for HW in range(9)]

# Calculate means. Column-wise for every different hamming weight.
Means = np.zeros((9, len(X_profiling[0])))
for i in range(9):
    Means[i] = np.average(TracesHW[i], 0)

# Performs difference of mean. For every feature in trace, there is now a number which indicates the 
# difference in mean across hamming weights.
SumDiff = np.zeros(len(X_profiling[0]))
for i in range(9):
    for j in range(i):
        SumDiff += np.abs(Means[i] - Means[j])

features = []

numFeatures = 20

featureSpacing = 5

for i in range(numFeatures):
    nextFeature = SumDiff.argmax()
    features.append(nextFeature)
    
    featureMin = max(0, nextFeature - featureSpacing)
    featureMax = min(nextFeature + featureSpacing, len(SumDiff)-1)
    for j in range(featureMin, featureMax):
        SumDiff[j] = 0

# Set up means and covariances of the features
meanMatrix = np.zeros((9, numFeatures))
covMatrix  = np.zeros((9, numFeatures, numFeatures))
for HW in range(9):
    for i in range(numFeatures):
        meanMatrix[HW][i] = Means[HW][features[i]]
        for j in range(numFeatures):
            x = TracesHW[HW][:,features[i]]
            y = TracesHW[HW][:,features[j]]
            covMatrix[HW,i,j] = cov(x, y)

if pooled:
    pooled_covMatrix = np.mean(covMatrix, axis=(0))

# ATTACK PHASE - NORMAL
# end of traces array (avoid index out of bounds)
max_range = 10000

# set up guessing entropy
ge = np.zeros(16)
    
# sample number for guessing entropy
ge_sample = 100

# The number of traces used for attacking
attack_samples = 10000

print("Using %d traces to attack.."%attack_samples)

def worker_func(test):
    attack_samples, X_attack, features, hamming, sbox, plt_attack, key_byte, meanMatrix, covMatrix, real_key = test
    ge_data = np.zeros(attack_samples)
    P_k = np.zeros(256)

    print(X_attack.shape)
    print(plt_attack.shape)
    print(features)

    l = list(zip(X_attack,plt_attack))
    random.shuffle(l)
    X_attack, plt_attack = list(zip(*l))
    X_attack = np.array(X_attack)
    plt_attack = np.array(plt_attack)
    print(X_attack.shape)
    print(plt_attack.shape)

    for j in range(0, 10000, 1):
        # Take selected features of the trace
        a = [X_attack[j][features[i]] for i in range(len(features))]
        
        
        for kguess in range(0, 256):
            # Hamming weight of key guess
            HW = hamming[sbox[plt_attack[j][key_byte] ^ kguess]]
        
            # Calculate the multivariate normal distribution...
            rv = multivariate_normal(meanMatrix[HW], covMatrix[HW])

            # What's the probability of these measurements, given the multivariate normal distribution of
            # the hamming weight of the hypothesis key 
            p_kj = rv.pdf(a)

            # Sum up probabilities in logarithmic scale
            # summing in log <-> multiplying in probabilities
            P_k[kguess] += np.log(p_kj)
        tarefs = np.argsort(P_k)[::-1]
        ge_data[j] += list(tarefs).index(real_key)

    return ge_data

if __name__ == '__main__':

    with Pool(100) as p:
        test = p.map(worker_func, [(attack_samples, X_attack, features, hamming, sbox, plt_attack, key_byte, meanMatrix, covMatrix, real_key) for i in range(100)])
        print(test)
        ge_data = np.mean(test, axis=0)

    print(ge_data[:10])
    np.save('model_data/ta_ge_data.npy', np.array(ge_data))
