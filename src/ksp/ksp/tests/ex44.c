
static char help[] = "Solves a tridiagonal linear system.  Designed to compare SOR for different Mat impls.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  KSP            ksp;      /* linear solver context */
  Mat            A;        /* linear system matrix */
  Vec            x,b;      /* approx solution, RHS */
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v[3] = {-1./2., 1., -1./2.};
  PetscInt       j[3];
  PetscInt       k=15;
  PetscInt       M,m=420;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetOperators(ksp,&A,NULL));

  PetscCall(MatSetSizes(A,m,m,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  PetscCall(MatGetSize(A,&M,NULL));
  for (Ii=Istart; Ii<Iend; Ii++) {
    j[0] = Ii - k;
    j[1] = Ii;
    j[2] = (Ii + k) < M ? (Ii + k) : -1;
    PetscCall(MatSetValues(A,1,&Ii,3,j,v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A,&x,&b));

  PetscCall(VecSetFromOptions(b));
  PetscCall(VecSet(b,1.0));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSet(x,2.0));

  PetscCall(KSPSolve(ksp,b,x));

  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}
