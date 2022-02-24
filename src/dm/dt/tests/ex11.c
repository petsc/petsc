#include <petscfv.h>

static char help[] = "Test memory allocation of PetscFV arrays used in PetscFVComputeGradient";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    PetscFV        fvm;
    PetscInt       dim, numFaces;
    PetscScalar    *dx, *grad;

    PetscFunctionBeginUser;
    ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); if (ierr) return ierr;

    /*
      Working with a 2D mesh, made of triangles, and using the 2nd neighborhood
      to reconstruct the cell gradient with a least square method, we use numFaces = 9
      The array dx is not initialised, but it doesn't matter here
      */
    dim = 2;
    numFaces = 9;
    CHKERRQ(PetscMalloc2(dim * numFaces, &dx, dim * numFaces, &grad));
    CHKERRQ(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
    CHKERRQ(PetscFVSetType(fvm, PETSCFVLEASTSQUARES));
    CHKERRQ(PetscFVLeastSquaresSetMaxFaces(fvm, numFaces));

    /* Issue here */
    CHKERRQ(PetscFVComputeGradient(fvm, numFaces, dx, grad));

    CHKERRQ(PetscFVDestroy(&fvm));
    CHKERRQ(PetscFree2(dx, grad));
    CHKERRQ(PetscFinalize());
    PetscFunctionReturn(0);
}

/*TEST
  test:
    suffix: 1
TEST*/
