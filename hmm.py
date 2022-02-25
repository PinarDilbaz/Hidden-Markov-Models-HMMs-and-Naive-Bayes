import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    """
    alphas = []
    size_of_A = len(A)
    size_of_B = len(B)
    size_of_O = len(O)

    for i in range(size_of_A):
        alphas.append([])
        
        for j in range(size_of_O):
            alphas[i].append(0.0)     
    alphas = np.asarray(alphas)
    
    for i in range(size_of_O):
        arr = []
        arr2 = []
        index = O[i]
        if (i == 0):
            for j in range(size_of_B):
                arr.append(B[j, index])
            alphas[:, 0] = arr * pi
        elif (i != 0):
            for k in range(len(alphas)):
                arr2.append(alphas[k, i - 1])
            
            for j in range(size_of_A):
                summ = sum(arr2 * (A[:, j]) * (B[j, index]))
                alphas[j, i] = summ
    summ = 0
    for i in range(len(alphas)):
        summ = summ + alphas[i, size_of_O - 1]
    probability = summ

    return probability, alphas           

def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """
    deltas = []
    deltas2 = []
    viterbi_result = []
    size_of_A = len(A)
    size_of_B = len(B)
    size_of_O = len(O)

    for i in range(size_of_A):
        deltas.append([])
        deltas2.append([])
        for j in range(size_of_O):
            deltas[i].append(0.0)
            deltas2[i].append(0.0)
            if i == 0:
                viterbi_result.append(0.0)
    deltas = np.asarray(deltas)
    deltas2 = np.array(deltas2)

    for i in range(size_of_O):
        arr = []
        arr2 = []
        index = O[i]
        if (i == 0):
            for j in range(size_of_B):
                arr.append(B[j, index])
            deltas[:, 0] = arr * pi
        elif (i != 0):
            for k in range(len(deltas)):
                arr2.append(deltas[k, i - 1])
            
            for j in range(size_of_A):
                maxi = max(arr2 * (A[:, j]) * (B[j, index]))
                argi = np.argmax(arr2 * (A[:, j]) * (B[j, index]))
                deltas[j, i] = maxi
                deltas2[j, i] = argi
                
    for i in range(size_of_O):
        deltas_arr = []
        maxi = 10
        if (i == 0):
            for j in range(len(deltas)):
                for k in deltas[j]:
                    deltas_arr.append(k)
            for j in range(len(deltas)):
                if(deltas[j,size_of_O - 1]<maxi):
                    maxi = deltas[j,size_of_O - 1]        
            viterbi_result[i]=int(maxi)
            
        if (i !=0):
            x = int(viterbi_result[i - 1])
            y = size_of_O - i
            viterbi_result[i] = deltas2[x,y]     
    viterbi_result = viterbi_result[::-1]

    viterbi_result = np.array(viterbi_result)
    return viterbi_result, deltas
