#!/usr/bin/env python

configure_options = [
  'CC=pgcc',
  'CXX=pgc++',
  'FC=pgf90',
  #'COPTFLAGS=-g -O', #-O gives compile errors with fblaslapack? so disabling for now
  #'FOPTFLAGS=-g -O',
  #'CXXOPTFLAGS=-g -O',
  '--with-hwloc=0', # ubuntu -lhwloc requires -lnuma - which conflicts with -lnuma from pgf90
  '--download-mpich=1',
  '--download-fblaslapack=1',
  '--download-codipack=1',
  '--download-adblaslapack=1',
  'CXXPPFLAGS=-std=c++11',
  'CXXFLAGS=-std=c++11',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
