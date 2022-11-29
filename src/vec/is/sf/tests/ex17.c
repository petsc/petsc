static const char help[] = "Test PetscSF with MPI large count (more than 2 billion elements in messages)\n\n";

#include <petscsys.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscSF            sf;
  PetscInt           i, nroots, nleaves;
  PetscInt           n       = (1ULL << 31) + 1024; /* a little over 2G elements */
  PetscSFNode       *iremote = NULL;
  PetscMPIInt        rank, size;
  char              *rootdata = NULL, *leafdata = NULL;
  Vec                x, y;
  VecScatter         vscat;
  PetscInt           rstart, rend;
  IS                 ix;
  const PetscScalar *xv;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "The test can only run with two MPI ranks");

  /* Test PetscSF */
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
  PetscCall(PetscSFSetGraph(sf, nroots, nleaves, NULL, PETSC_COPY_VALUES, iremote, PETSC_COPY_VALUES));
  PetscCall(PetscMalloc2(nroots, &rootdata, nleaves, &leafdata));
  if (rank == 0) {
    memset(rootdata, 11, nroots);
    rootdata[nroots - 1] = 12; /* Use a different value at the end */
  }

  PetscCall(PetscSFBcastBegin(sf, MPI_SIGNED_CHAR, rootdata, leafdata, MPI_REPLACE)); /* rank 0->1, bcast rootdata to leafdata */
  PetscCall(PetscSFBcastEnd(sf, MPI_SIGNED_CHAR, rootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(sf, MPI_SIGNED_CHAR, leafdata, rootdata, MPI_SUM)); /* rank 1->0, add leafdata to rootdata */
  PetscCall(PetscSFReduceEnd(sf, MPI_SIGNED_CHAR, leafdata, rootdata, MPI_SUM));
  PetscCheck(rank != 0 || (rootdata[0] == 22 && rootdata[nroots - 1] == 24), PETSC_COMM_SELF, PETSC_ERR_PLIB, "SF: wrong results");

  PetscCall(PetscFree2(rootdata, leafdata));
  PetscCall(PetscFree(iremote));
  PetscCall(PetscSFDestroy(&sf));

  /* Test VecScatter */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(x, rank == 0 ? n : 64, PETSC_DECIDE));
  PetscCall(VecSetSizes(y, rank == 0 ? 64 : n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetFromOptions(y));

  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, rend - rstart, rstart, 1, &ix));
  PetscCall(VecScatterCreate(x, ix, y, ix, &vscat));

  PetscCall(VecSet(x, 3.0));
  PetscCall(VecScatterBegin(vscat, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat, x, y, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecScatterBegin(vscat, y, x, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(vscat, y, x, ADD_VALUES, SCATTER_REVERSE));

  PetscCall(VecGetArrayRead(x, &xv));
  PetscCheck(xv[0] == 6.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecScatter: wrong results");
  PetscCall(VecRestoreArrayRead(x, &xv));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecScatterDestroy(&vscat));
  PetscCall(ISDestroy(&ix));

  PetscCall(PetscFinalize());
  return 0;
}

/**TEST
   test:
     requires: defined(PETSC_HAVE_MPI_LARGE_COUNT) defined(PETSC_USE_64BIT_INDICES)
     TODO: need a machine with big memory (~150GB) to run the test
     nsize: 2
     args: -sf_type {{basic neighbor}}

TEST**/
