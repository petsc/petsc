
static char help[] = "Tests MatILUFactorSymbolic() on matrix with missing diagonal.\n\n";

#include <petscmat.h>
#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j;
  PetscScalar    v;
  PC             pc;
  Vec            xtmp;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,3,3));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,3,&xtmp));
  i    = 0; j = 0; v = 4;
  PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 0; j = 2; v = 1;
  PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 1; j = 0; v = 1;
  PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 1; j = 1; v = 4;
  PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 2; j = 1; v = 1;
  PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PCCreate(PETSC_COMM_WORLD,&pc));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetOperators(pc,C,C));
  PetscCall(PCSetUp(pc));
  PetscCall(PCFactorGetMatrix(pc,&A));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&xtmp));
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}
