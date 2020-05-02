import numpy as np
from sklearn.model_selection import train_test_split

def getUserDetails():
    file = open("users.dat")
    users=[]
    for line in file:
        line = line.rstrip("\n\r")
        users.append(line.split('::'))
    ne=[]
    for i in range(len(users)):
        ne.append(int(users[i]))
    users=np.array(ne)
    users=np.int64(users)
    return users

#Function to get movie details
def getMovieDetails():
    file = open("movies.dat")
    movies=[]
    for line in file:
        line = line.rstrip("\n\r")
        movies.append(line.split('::'))
    ne=[]
    for i in range(len(movies)):
        ne.append(int(movies[i]))
    movies=np.array(ne)
    movies=np.int64(movies)
    return movies

#Function to get rating details of the users
def getRatingDetails():
    file = open("ratings.dat")
    ratings=[]
    for line in file:
        line = line.rstrip("\n\r")
        ratings.append(line.split('::'))
    ne=[]
    for i in range(len(ratings)):
        x=[]
        x.append(int(ratings[i][0]))
        x.append(int(ratings[i][1]))
        x.append(int(ratings[i][2]))
        x=np.int64(x)
        ne.append(x)

    ratings=np.array(ne)
    return ratings

def make_matrix():
    ratings=getRatingDetails()
    print(ratings[0])
    print(type(ratings[0][0]))
    USER_MAX=6040+1
    MOVIE_MAX=3952+1
    rate_test=np.zeros((USER_MAX,MOVIE_MAX))
    rate_train=np.zeros((USER_MAX,MOVIE_MAX))
    ratings_train,ratings_test=train_test_split(ratings, test_size=0.3)
    for i in ratings_train:
        rate_train[i[0]][i[1]]=i[2]
    for i in ratings_test:
        rate_test[i[0]][i[1]]=i[2]
    return rate_test,rate_train,USER_MAX,MOVIE_MAX

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

if __name__=='__main__':
    rate_test,rate_train,us,mo=make_matrix()
    mf=MF(rate_train,7,0.0001,0.00001,20)
    mf.train()
    rate_pred=np.zeros((us,mo))
    for i in range(rate_test.shape[0]):
        for j in range(rate_test.shape[1]):
            if rate_test[i][j]!=0:
                rate_pred[i][j]=mf.get_rating(i,j)
    print(rate_pred)
