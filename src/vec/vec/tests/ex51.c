static char help[] = "Test integrity of subvector data, use \n\
use -hdf5 to specify HDF5 viewer format for subvector I/O \n\n";

/*
   Tests for transfer of data from subvectors to parent vectors after
   loading data into subvector. This routine does the following : creates
   a vector of size 50, sets it to 2 and saves it to disk. Creates a
   vector of size 100, set it to 1 and extracts the last 50 elements
   as a subvector. Loads the saved vector from disk into the subvector
   and restores the subvector. To verify that the data has been loaded
   into the parent vector, the sum of its elements is calculated.
   The arithmetic mean is also calculated in order to test VecMean().
*/

#include <petscvec.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv)
{
  Vec         testvec;         /* parent vector of size 100 */
  Vec         loadvec;         /* subvector extracted from the parent vector */
  Vec         writevec;        /* vector used to save data to be loaded by loadvec */
  IS          loadis;          /* index set to extract last 50 elements of testvec */
  PetscInt    low, high;       /* used to store vecownership output */
  PetscInt    issize, isstart; /* index set params */
  PetscInt    skipuntil = 50;  /* parameter to slice the last N elements of parent vec */
  PetscViewer viewer;          /* viewer for I/O */
  PetscScalar sum;             /* used to test sum of parent vector elements */
  PetscScalar mean;            /* used to test mean of parent vector elements */
  PetscBool   usehdf5 = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* parse input options to determine I/O format */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-hdf5", &usehdf5, NULL));

  /* Create parent vector with 100 elements, set it to 1 */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &testvec));
  PetscCall(VecSetSizes(testvec, PETSC_DECIDE, 100));
  PetscCall(VecSetUp(testvec));
  PetscCall(VecSet(testvec, (PetscScalar)1));

  /* Create a vector with 50 elements, set it to 2. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &writevec));
  PetscCall(VecSetSizes(writevec, PETSC_DECIDE, 50));
  PetscCall(VecSetUp(writevec));
  PetscCall(VecSet(writevec, (PetscScalar)2));
  PetscCall(PetscObjectSetName((PetscObject)writevec, "temp"));

  /* Save to disk in specified format, destroy vector & viewer */
  if (usehdf5) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "writing vector in hdf5 to vector.dat ...\n"));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_WRITE, &viewer));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "writing vector in binary to vector.dat ...\n"));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_WRITE, &viewer));
  }
  PetscCall(VecView(writevec, viewer));
  PetscCall(VecDestroy(&writevec));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Create index sets on each mpi rank to select the last 50 elements of parent vec */
  PetscCall(VecGetOwnershipRange(testvec, &low, &high));
  if (low >= skipuntil) {
    isstart = low;
    issize  = high - low;
  } else if (low <= skipuntil && high >= skipuntil) {
    isstart = skipuntil;
    issize  = high - skipuntil;
  } else {
    isstart = low;
    issize  = 0;
  }
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, issize, isstart, 1, &loadis));

  /* Create subvector using the index set created above */
  PetscCall(VecGetSubVector(testvec, loadis, &loadvec));
  PetscCall(PetscObjectSetName((PetscObject)loadvec, "temp"));

  /* Load the previously saved vector into the subvector, destroy viewer */
  if (usehdf5) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "reading vector in hdf5 from vector.dat ...\n"));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_READ, &viewer));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "reading vector in binary from vector.dat ...\n"));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_READ, &viewer));
  }
  PetscCall(VecLoad(loadvec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Restore subvector to transfer loaded data into parent vector */
  PetscCall(VecRestoreSubVector(testvec, loadis, &loadvec));

  /* Compute sum of parent vector elements */
  PetscCall(VecSum(testvec, &sum));
  PetscCall(VecMean(testvec, &mean));

  /* to verify that the loaded data has been transferred */
  PetscCheck(sum == 150, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Data has not been transferred from subvector to parent vector");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecSum on parent vec is : %e\n", (double)PetscAbsScalar(sum)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecMean on parent vec is : %e\n", (double)PetscAbsScalar(mean)));

  /* destroy parent vector, index set and exit */
  PetscCall(VecDestroy(&testvec));
  PetscCall(ISDestroy(&loadis));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hdf5

  test:
    nsize: 4

  test:
    suffix: 2
    nsize: 4
    args: -hdf5

TEST*/
