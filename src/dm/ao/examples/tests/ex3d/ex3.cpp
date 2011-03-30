 
static char help[] = "AO test contributed by Sebastian Steiger <steiger@purdue.edu>, March 2011\n\n";

/*
  Example of usage:
    mpiexec -n 12 ./ex3
    mpiexec -n 30 ./ex3 -ao_type basic
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <mpi.h>
#include "petsc.h"

using namespace std;

int main(int argc, char** argv)
{
  PetscErrorCode ierr;
  AO ao;
  IS isapp;

  PetscInitialize(&argc, &argv, (char*)0, help);
  int size=-1;   MPI_Comm_size(PETSC_COMM_WORLD, &size);
  int myrank=-1; MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);

  // read in application indices
  char infile[1000]; sprintf(infile,"./AO%dCPUs/ao_p%d_appindices.txt",size,myrank);
  //cout << infile << endl;
  ifstream fin(infile);
  assert(fin);
  vector<int>  myapp;
  int tmp=-1;
  while (!fin.eof()) {
    tmp=-1;
    fin >> tmp;
    if (tmp==-1) break;
    myapp.push_back(tmp);
  }
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] has %D indices.\n",
          myrank,myapp.size());CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, myapp.size(), &(myapp[0]), PETSC_USE_POINTER, &isapp);CHKERRQ(ierr);
  //ierr = ISView(isapp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = AOCreate(PETSC_COMM_WORLD, &ao);CHKERRQ(ierr);
  ierr = AOSetIS(ao, isapp, PETSC_NULL);CHKERRQ(ierr); 
  ierr = AOSetType(ao, AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr = AOSetFromOptions(ao);CHKERRQ(ierr);
  
  if (myrank==0) cout << "AO has been set up." << endl;
                              
  ierr = AODestroy(ao);CHKERRQ(ierr);     
  ierr = ISDestroy(isapp);CHKERRQ(ierr);

  if (myrank==0) cout << "AO is done." << endl;
  
  PetscFinalize();
  return 0;
}
