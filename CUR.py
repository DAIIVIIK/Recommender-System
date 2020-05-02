import numpy as np
import random
import math
from cur import cur_decomposition

if __name__ == '__main__':

    f=open("ratings.dat", "r")
    ratings = []
    if f.mode == 'r':
        contents = f.read()
        ratings = contents.strip().split('\n')
    f.close()

    ratings_list =[]
    for r in ratings:
        r = r.split('::')
        ratings_list.append([int(r[0]), int(r[1]), int(r[2]), int(r[3])])

    print("ratings done")

    random.shuffle(ratings_list)
    test_data = ratings_list[int(0.7*len(ratings_list)):]
    train_data =  ratings_list[:int(0.7*len(ratings_list))]

    NO_USERS = 6041 #Max of userID
    NO_MOVIES = 3953 #max of movieID
    matrix = np.zeros((NO_MOVIES , NO_USERS))
    matrix2 = np.zeros((NO_MOVIES , NO_USERS))

    for r in ratings_list:
        matrix[r[1]][r[0]] = r[2]

    C, U, R = cur_decomposition(matrix,2)

    print(C.shape, U.shape, R.shape)

    temp = np.dot(U,R)
    matrix_new_cur = np.dot(C,temp)

    error_cur = (matrix - matrix_new_cur)

    sum2 = 0
    for i in range(len(error_cur)):
        for j in range(len(error_cur[i])):
            sum2+=(error_cur[i][j]*error_cur[i][j])

    sum2/=(NO_USERS*NO_MOVIES)

    rmse_cur = math.sqrt(sum2)

    print(rmse_cur)
