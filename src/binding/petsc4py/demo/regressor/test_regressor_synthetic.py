# Linear regression with synthetic data generation
# ================================================
#
# Generate a synthetic data set using sklearn.datasets.make_regression() and
# then solve it using the petsc4py interface to PetscRegressor.

import numpy as np
# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt

# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt

# Needed for evaluating the quality of the regression for the test data set
from sklearn.metrics import mean_squared_error

import argparse
parser = argparse.ArgumentParser('Test MLG using synthetic data')
parser.add_argument('--nsample', type=int, default=10000)
parser.add_argument('--nfeature', type=int, default=10)
parser.add_argument('--add_noise', action='store_true')
args, unknown = parser.parse_known_args()

import sys
import petsc4py
sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Define the seed so that results can be reproduced
seed = 11
rand_state = 11

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

def petsc_regression_test(nsample, nfeature, noise=0, rand_state=11):
  ntr = round(nsample*9/10)
  nte = nsample - ntr
  x,y = dt.make_regression(n_samples=nsample,
                           n_features=nfeature,
                           noise=noise,
                           random_state=rand_state)
  x_train, y_train = x[:ntr,], y[:ntr]
  xte, yte = x[ntr:,], y[ntr:]
  comm = PETSc.COMM_WORLD
  rank = comm.getRank()
  regressor = PETSc.Regressor().create(comm=comm)
  regressor.setType(PETSc.Regressor.Type.LINEAR)
  rows_ix = np.arange(ntr,dtype=np.int32)
  cols_ix = np.arange(nfeature,dtype=np.int32)
  X = PETSc.Mat().create(comm=comm)
  X.setSizes((ntr,nfeature))
  X.setFromOptions()
  X.setUp()
  y = PETSc.Vec().create(comm=comm)
  y.setSizes(ntr)
  y.setFromOptions()
  if not rank :
    X.setValues(rows_ix,cols_ix,x_train,addv=True)
    y.setValues(rows_ix,y_train,addv=False)
  X.assemblyBegin(X.AssemblyType.FINAL)
  X.assemblyEnd(X.AssemblyType.FINAL)
  y.assemblyBegin()
  y.assemblyEnd()
  regressor.fit(X,y)
  rows_ix = np.arange(nte,dtype=np.int32)
  X = PETSc.Mat().create(comm=comm)
  X.setSizes((nte,nfeature))
  X.setFromOptions()
  X.setUp()
  y = PETSc.Vec().create(comm=comm)
  y.setSizes(nte)
  y.setFromOptions()
  if not rank :
    X.setValues(rows_ix,cols_ix,xte,addv=True)
    y.zeroEntries()
  X.assemblyBegin(X.AssemblyType.FINAL)
  X.assemblyEnd(X.AssemblyType.FINAL)
  y.assemblyBegin()
  y.assemblyEnd()
  regressor.predict(X,y)
  ypr = y.getArray()
  error = mean_squared_error(ypr,yte)
  print(f"Test MSE: {error:f}")
  return xte,ypr

xte,ypr = petsc_regression_test(args.nsample, args.nfeature, 0)

if args.add_noise:
  fig,ax = plt.subplots(nrows=2, ncols=3,figsize=(16,7))
  plt_ind_list = np.arange(6)+231

  for noise,plt_ind in zip([0,0.1,1,10,100,1000],plt_ind_list):
    xte,ypr = petsc_regression_test(args.nsample, args.nfeature, noise, rand_state)

    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(xte[:,0],
                                  xte[:,1],
                                  c=ypr,
                                  vmin=min(ypr),
                                  vmax=max(ypr),
                                  s=35,
                                  cmap=color_map)

    plt.title('noise: '+str(noise))
    plt.colorbar(my_scatter_plot)

  fig.subplots_adjust(hspace=0.3,wspace=.3)
  plt.suptitle('PETSc regression tests with different noise levels',fontsize=20)
  plt.show(block=True)
