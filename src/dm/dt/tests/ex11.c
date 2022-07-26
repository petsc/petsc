#include <petscfv.h>

static char help[] = "Test memory allocation of PetscFV arrays used in PetscFVComputeGradient";

int main(int argc, char **argv)
{
  PetscFV        fvm;
  PetscInt       dim, numFaces;
  PetscScalar    *dx, *grad;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, PETSC_NULL, help));

  /*
   Working with a 2D mesh, made of triangles, and using the 2nd neighborhood
   to reconstruct the cell gradient with a least square method, we use numFaces = 9
   The array dx is not initialised, but it doesn't matter here
  */
  dim = 2;
  numFaces = 9;
  PetscCall(PetscMalloc2(dim * numFaces, &dx, dim * numFaces, &grad));
  PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
  PetscCall(PetscFVSetType(fvm, PETSCFVLEASTSQUARES));
  PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, numFaces));

  /* Issue here */
  PetscCall(PetscFVComputeGradient(fvm, numFaces, dx, grad));

  PetscCall(PetscFVDestroy(&fvm));
  PetscCall(PetscFree2(dx, grad));
  PetscCall(PetscFinalize());
  return(0);
}

/*TEST

  test:
    suffix: 1

TEST*/
