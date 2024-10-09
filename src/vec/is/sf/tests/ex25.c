static const char help[] = "Test PetscSF with derived data types created with MPI large count\n\n";

#include <petscsys.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscSF        sf;
  PetscInt       i, nroots, nleaves;
  const PetscInt m = 4, n = 64;
  PetscSFNode   *iremote = NULL;
  PetscMPIInt    rank, size;
  int           *rootdata = NULL, *leafdata = NULL;
  MPI_Datatype   newtype;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "The test can only run with two MPI ranks");

  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
  PetscCall(PetscSFSetFromOptions(sf));

  if (rank == 0) {
    nroots  = n;
    nleaves = 0;
  } else {
    nroots  = 0;
    nleaves = n;
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (i = 0; i < nleaves; i++) {
      iremote[i].rank  = 0;
      iremote[i].index = i;
    }
  }
  PetscCall(PetscSFSetGraph(sf, nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));

  PetscCall(PetscCalloc2(nroots * m, &rootdata, nleaves * m, &leafdata)); // allocate fat nodes to apply a derived data type of m MPI_INTs

  if (rank == 0) rootdata[nroots * m - 1] = 123; // set the last integer in rootdata and then check on leaves

#if defined(PETSC_HAVE_MPI_LARGE_COUNT)
  PetscCallMPI(MPI_Type_contiguous_c(m, MPI_INT, &newtype));
#else
  PetscCallMPI(MPI_Type_contiguous(m, MPI_INT, &newtype));
#endif

  PetscCallMPI(MPI_Type_commit(&newtype));

  PetscCall(PetscSFBcastBegin(sf, newtype, rootdata, leafdata, MPI_REPLACE)); //  bcast rootdata to leafdata
  PetscCall(PetscSFBcastEnd(sf, newtype, rootdata, leafdata, MPI_REPLACE));

  if (rank == 1) PetscCheck(leafdata[nleaves * m - 1] == 123, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SF: wrong results");

  PetscCallMPI(MPI_Type_free(&newtype));
  PetscCall(PetscFree2(rootdata, leafdata));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFinalize());
  return 0;
}

/**TEST
   test:
     nsize: 2
     output_file: output/empty.out
     args: -sf_type {{basic neighbor}}

TEST**/
