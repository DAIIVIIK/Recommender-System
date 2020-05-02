import numpy as np
from numpy import linalg as la
from numpy.linalg import svd
import math
import operator
import time
import os
from numpy.linalg import norm
from numpy import dot as dt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
def SVD(A,B,bias,c):
	print('inside SVD:','\n')
	U,sigmas,Vt=svd(A,full_matrices=False)
	s=np.zeros((len(A.T),len(A.T)))
	for i in range(len(sigmas)):
		s[i][i]=sigmas[i]
	
	rmse,mae,precision=pred(A,B,Vt,bias,c)
	print('rmse :',rmse)
	print('mae :',mae)
	print('precision :',precision)
	eigens=[]
	sigma_sum=0
	for i in range(len(sigmas)):
		sigma_sum+=sigmas[i]**2
	threshold=0.9*sigma_sum
	sum=0
	last=0
	for i in range(len(sigmas)):
		sum+=sigmas[i]**2
		if(sum>threshold):
			last=i-1
			break
	
	Vt1=np.zeros((last+1,len(Vt[0])))
	print('after 90%  energy retain :')
	rmse1,mae1,precision1=pred(A,B,Vt1,bias,c)
	print('rmse1 :',rmse1)
	print('mae1 :',mae1)
	print('precision1 :',precision1)

def pred(A,B,Vt,bias,c):
	predictions=0
	square_err=0
	err=0
	start_time=time.time()
	for i in range(len(A)):
		AV=A[i].dot(Vt.T)
		rating_for_movie=AV.dot(Vt)
		rating_for_movie=rating_for_movie+bias[i]

		for j in range(len(A[i])):
			
			if(B[i][j]!=0 and A[i][j]+bias[i]!=B[i][j]):
				predictions+=1
				err+=abs(rating_for_movie[j]-B[i][j])
				square_err+=(rating_for_movie[j]-B[i][j])**2

				c[i][j]=rating_for_movie[j]
	
	rmse=float(math.sqrt(square_err)/predictions)
	mae=float((err)/predictions)

	count=0
	k_movies_c=k_top_movies(c,k)
	
	for movie in k_movies_B:
		if movie in k_movies_c:
			count+=1

	precision=float(count)/k
	print('time taken for prediction: ',time.time()-start_time)
	return rmse,mae,precision



def k_top_movies(m,k):
	movie_rating=[]
	avg_rating=np.zeros(len(m[0]))
	k_movies_m=[]
	for j in range(len(m[0])):
		sum=0
		raters=0
		for i in range (len(m)):
			if(m[i][j]!=0):
				sum+=m[i][j]
				raters+=1
		if raters>=1:
			avg_rating[j]=float(sum)/raters
			movie_rating.append([j,avg_rating[j]])
	sorted_movies=sorted(movie_rating,key=operator.itemgetter(1),reverse=True)
	for j,ind in zip(range(k),range(len(sorted_movies))):
		k_movies_m.append(sorted_movies[j][0])
	return k_movies_m
def CF(At,Bt,c):
	print('inside CF','\n')
	print(Bt)
	start_time=time.time()
	norm=np.zeros(len(At))
	for i in range(len(At)):
		sum=0
		raters=0

		for j in range(len(At[i])):
			if(At[i][j]!=0):
				sum+=At[i][j]
				raters+=1
		if raters>=1:
			norm[i]=float(sum)/raters
		for j in range(len(At[i])):
			if At[i][j]!=0:
				At[i][j]-=norm[i]
	similarity=[]
	top_k=[]
	predictions=0
	square_err=0
	err=0
	print('checkpoint 1')
	for i in range(len(Bt)):
		similarity.append([])
		similarity[i]=[]
		top_k.append([])
		if(dt(Bt[i],Bt[i])==0):
			continue
		for j in range(len(Bt)):
			if(dt(Bt[j],Bt[j])==0 or i==j):
				continue
			den=math.sqrt((dt(Bt[i],Bt[i]))*(dt(Bt[j],Bt[j])))
			print(den)
			if den==0:
				print('new movie')

				return
			sim=dt(Bt[i],Bt[j])/den
						
			similarity[i].append((j,sim))
		dec_i=sorted(similarity[i],key=operator.itemgetter(1),reverse=True)
		# print(dec_i)
		# return

		top_k[i]=[]
		
		for j,value in zip(range(k),range(len(dec_i))):
			top_k[i].append(dec_i[j][0])
	print(top_k)
	for m in range(len(similarity)):
		similarity[m].append((m,-2323))	
	print('chechpoint 2')
	for i in range(len(A)):
		for j in range(len(A[i])):
			if(B[i][j]!=0 and B[i][j]!=A[i][j]+bias[i]):
				rating=0
				n=0
				for l in range(len(top_k[j])):
					
					rating+=similarity[j][top_k[j][l]][1]*B[i][top_k[j][l]]
					n+=similarity[j][top_k[j][l]][1]
				rate=float(rating)/n
				predictions+=1
				err+=abs(rate-B[i][j])
				square_err+=(rate-B[i][j])**2
	print('time taken for prediction: ',time.time()-start_time)
	rmse=float(math.sqrt(square_err)/predictions)
	mae=float(err/predictions)
	print('rmse :', rmse)
	print('mae: ',mae)
# =========================================================================================
# 											CUR
# =========================================================================================
# def cur()


























# =========================================================================================
# 											MAIN
# =========================================================================================
max_user=0
max_movie=0
count=0
f=open("C:\\Users\\home\\Desktop\\ir3\\rate.txt",'r')
for line in f:
	count+=1
	val=line.split("::")
	a=int(val[0])-1
	b=int(val[1])-1
	if(a>max_user):
		max_user=a
	if(b>max_movie):
		max_movie=b
test=int(0.80*count)
A=np.zeros((max_user+1,max_movie+1))
B=np.zeros((max_user+1,max_movie+1))
c=0
f=open("C:\\Users\\home\\Desktop\\ir3\\rate.txt",'r')
for line in f:
	
	val=line.split("::")
	a=int(val[0])
	b=int(val[1])
	B[a-1][b-1]=int(val[2])
	if(c<test):
		A[a-1][b-1]=int(val[2])
		c+=1
c=A.copy()
bias=np.zeros(max_user+1)
for i in range(max_user+1):
	sum=0
	movies_rated=0
	for j in range(max_movie+1):
		if A[i][j]!=0:
			sum+=A[i][j]
			movies_rated+=1
	if(movies_rated>=1):
		bias[i]=float(sum/movies_rated)
	for j in range(max_movie+1):
		if A[i][j]!=0:
			A[i][j]=A[i][j]-bias[i]


k=3
# print(B)
# print(c)
k_movies_B=k_top_movies(B,k)
# SVD(A,B,bias,c)
CF(A.T,B.T,c)
