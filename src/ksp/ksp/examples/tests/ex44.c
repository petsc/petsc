
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);

  ierr = MatSetSizes(A,m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    j[0] = Ii - k;
    j[1] = Ii;
    j[2] = (Ii + k) < M ? (Ii + k) : -1;
    ierr = MatSetValues(A,1,&Ii,3,j,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);

  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,2.0);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
