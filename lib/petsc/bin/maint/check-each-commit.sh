#!/bin/bash -ex

dest=`lib/petsc/bin/maint/check-merge-branch.sh`
for commit in $(git log --reverse --format=format:%H $dest..HEAD)
do
  git checkout $commit
  git clean -f -d -x -q
  ./configure --with-clanguage=cxx --with-coverage-exec=0 --with-syclc=0 --with-hipc=0 --with-cudac=0 --with-x=0 --with-bison=0 --with-cmake=0 --with-pthread=0 --with-regex=0 --with-mkl_sparse_optimize=0 --with-mkl_sparse=0 --with-debugging=0
  make vermin
  make checkclangformat
  make checkbadSource
  make checkbadFileChange
  make -f gmakefile check_output
  make check_petsc4py_rst
  make CFLAGS=-Werror CXXFLAGS=-Werror FFLAGS=-Werror all
  make CFLAGS=-Werror CXXFLAGS=-Werror FFLAGS=-Werror check
  make CFLAGS=-Werror CXXFLAGS=-Werror FFLAGS=-Werror allgtests-tap gmakesearch=snes_tutorials-ex48%
done
