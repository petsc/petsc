
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

int main(int argc, char **argv)
{
  AO          ao;
  IS          isapp;
  char        infile[PETSC_MAX_PATH_LEN], datafiles[PETSC_MAX_PATH_LEN];
  PetscBool   flg;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-datafiles", datafiles, sizeof(datafiles), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify -datafiles ${DATAFILESPATH}/ao");

  // read in application indices
  PetscCall(PetscSNPrintf(infile, sizeof(infile), "%s/AO%dCPUs/ao_p%d_appindices.txt", datafiles, size, rank));
  ifstream fin(infile);
  PetscCheck(fin, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "File not found: %s", infile);
  vector<PetscInt> myapp;
  int              tmp = -1;
  while (!fin.eof()) {
    tmp = -1;
    fin >> tmp;
    if (tmp == -1) break;
    myapp.push_back(tmp);
  }
#if __cplusplus >= 201103L // c++11
  static_assert(is_same<decltype(myapp.size()), size_t>::value, "");
#endif
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] has %zu indices.\n", rank, myapp.size()));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, myapp.size(), &(myapp[0]), PETSC_USE_POINTER, &isapp));

  PetscCall(AOCreate(PETSC_COMM_WORLD, &ao));
  PetscCall(AOSetIS(ao, isapp, NULL));
  PetscCall(AOSetType(ao, AOMEMORYSCALABLE));
  PetscCall(AOSetFromOptions(ao));

  if (rank == 0) cout << "AO has been set up." << endl;

  PetscCall(AODestroy(&ao));
  PetscCall(ISDestroy(&isapp));

  if (rank == 0) cout << "AO is done." << endl;

  PetscCall(PetscFinalize());
  return 0;
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
