
static char help[] = "Demonstrates the use of fast Richardson for SOR. And tests\n\
the MatSOR() routines.\n\n";

#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            mat;
  Vec            b,u;
  PC             pc;
  PetscInt       n = 5,i,col[3];
  PetscScalar    value[3];

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /* Create vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&u));

  /* Create and assemble matrix */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&mat));
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  PetscCall(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  PetscCall(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* Create PC context and set up data structures */
  PetscCall(PCCreate(PETSC_COMM_WORLD,&pc));
  PetscCall(PCSetType(pc,PCSOR));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetOperators(pc,mat,mat));
  PetscCall(PCSetUp(pc));

  value[0] = 1.0;
  for (i=0; i<n; i++) {
    PetscCall(VecSet(u,0.0));
    PetscCall(VecSetValues(u,1,&i,value,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyEnd(u));
    PetscCall(PCApply(pc,u,b));
    PetscCall(VecView(b,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Free data structures */
  PetscCall(MatDestroy(&mat));
  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
