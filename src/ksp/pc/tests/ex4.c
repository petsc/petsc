
static char help[] = "Demonstrates the use of fast Richardson for SOR. And tests\n\
the MatSOR() routines.\n\n";

#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            mat;
  Vec            b,u;
  PC             pc;
  PetscErrorCode ierr;
  PetscInt       n = 5,i,col[3];
  PetscScalar    value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Create vectors */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&b));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&u));

  /* Create and assemble matrix */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&mat));
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(mat,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  CHKERRQ(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  CHKERRQ(MatSetValues(mat,1,&i,2,col,value,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* Create PC context and set up data structures */
  CHKERRQ(PCCreate(PETSC_COMM_WORLD,&pc));
  CHKERRQ(PCSetType(pc,PCSOR));
  CHKERRQ(PCSetFromOptions(pc));
  CHKERRQ(PCSetOperators(pc,mat,mat));
  CHKERRQ(PCSetUp(pc));

  value[0] = 1.0;
  for (i=0; i<n; i++) {
    CHKERRQ(VecSet(u,0.0));
    CHKERRQ(VecSetValues(u,1,&i,value,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(u));
    CHKERRQ(VecAssemblyEnd(u));
    CHKERRQ(PCApply(pc,u,b));
    CHKERRQ(VecView(b,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Free data structures */
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PCDestroy(&pc));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
