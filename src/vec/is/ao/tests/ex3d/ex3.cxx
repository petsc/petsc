
static char help[] = "AO test contributed by Sebastian Steiger <steiger@purdue.edu>, March 2011\n\n";

/*
  Example of usage:
    mpiexec -n 12 ./ex3
    mpiexec -n 30 ./ex3 -ao_type basic
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <petscvec.h>
#include <petscao.h>

using namespace std;

int main(int argc, char** argv)
{
  PetscErrorCode ierr;
  AO             ao;
  IS             isapp;
  char           infile[PETSC_MAX_PATH_LEN],datafiles[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscMPIInt    size,rank;

  ierr = PetscInitialize(&argc, &argv, (char*)0, help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-datafiles",datafiles,sizeof(datafiles),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must specify -datafiles ${DATAFILESPATH}/ao");

  // read in application indices
  ierr = PetscSNPrintf(infile,sizeof(infile),"%s/AO%dCPUs/ao_p%d_appindices.txt",datafiles,size,rank);CHKERRQ(ierr);
  ifstream fin(infile);
  if (!fin) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"File not found: %s",infile);
  vector<PetscInt>  myapp;
  int tmp=-1;
  while (!fin.eof()) {
    tmp=-1;
    fin >> tmp;
    if (tmp==-1) break;
    myapp.push_back(tmp);
  }
#if __cplusplus >= 201103L // c++11
  static_assert(is_same<decltype(myapp.size()),size_t>::value,"");
#endif
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] has %zu indices.\n",rank,myapp.size());CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, myapp.size(), &(myapp[0]), PETSC_USE_POINTER, &isapp);CHKERRQ(ierr);

  ierr = AOCreate(PETSC_COMM_WORLD, &ao);CHKERRQ(ierr);
  ierr = AOSetIS(ao, isapp, NULL);CHKERRQ(ierr);
  ierr = AOSetType(ao, AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr = AOSetFromOptions(ao);CHKERRQ(ierr);

  if (rank==0) cout << "AO has been set up." << endl;

  ierr = AODestroy(&ao);CHKERRQ(ierr);
  ierr = ISDestroy(&isapp);CHKERRQ(ierr);

  if (rank==0) cout << "AO is done." << endl;

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !defined(PETSC_USE_64BIT_INDICES)

   test:
      nsize: 12
      requires: double !complex datafilespath
      args: -datafiles ${DATAFILESPATH}/ao
      output_file: output/ex3_1.out

   test:
      suffix: 2
      nsize: 12
      requires: double !complex datafilespath
      args: -ao_type basic -datafiles ${DATAFILESPATH}/ao
      output_file: output/ex3_1.out

   test:
      suffix: 3
      nsize: 30
      requires: double !complex datafilespath
      args: -datafiles ${DATAFILESPATH}/ao

   test:
      suffix: 4
      nsize: 30
      requires: double !complex datafilespath
      args: -ao_type basic -datafiles ${DATAFILESPATH}/ao
      output_file: output/ex3_3.out

TEST*/
