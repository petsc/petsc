
static char help[] = "Solves a tridiagonal linear system.  Designed to compare SOR for different Mat impls.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  KSP            ksp;      /* linear solver context */
  Mat            A;        /* linear system matrix */
  Vec            x,b;      /* approx solution, RHS */
  PetscInt       Ii,Istart,Iend;
  PetscErrorCode ierr;
  PetscScalar    v[3] = {-1./2., 1., -1./2.};
  PetscInt       j[3];
  PetscInt       k=15;
  PetscInt       M,m=420;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPGetOperators(ksp,&A,NULL));

  CHKERRQ(MatSetSizes(A,m,m,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  CHKERRQ(MatGetSize(A,&M,NULL));
  for (Ii=Istart; Ii<Iend; Ii++) {
    j[0] = Ii - k;
    j[1] = Ii;
    j[2] = (Ii + k) < M ? (Ii + k) : -1;
    CHKERRQ(MatSetValues(A,1,&Ii,3,j,v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&x,&b));

  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecSet(b,1.0));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSet(x,2.0));

  CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}
