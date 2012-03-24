static char help[] = "Test MatMatMult dual dispatch.\n\n";

#include <petscmat.h>
#include <petsc-private/matimpl.h>                  /* Need MatOpRegister for this test. */
#include <../src/mat/impls/dense/seq/dense.h> /* Need MatMatMult_SeqAIJ_SeqDense for this test. */
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat            A /*sparse*/, B /* dense */, C;
  PetscInt       i,j,m,n;
  PetscErrorCode ierr;
  MatScalar      val;


  PetscInitialize(&argc,&argv,(char *)0,help);

  /* Create A (sparse) */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  i=0; j=0; val=1.0; ierr = MatSetValues(A,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  i=1; j=3; val=2.0; ierr = MatSetValues(A,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  i=2; j=2; val=3.0; ierr = MatSetValues(A,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  i=3; j=0; val=4.0; ierr = MatSetValues(A,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create B (dense) */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(B,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(B,PETSC_NULL);CHKERRQ(ierr);
  i=0; j=0; val=1.0; ierr = MatSetValues(B,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  i=1; j=3; val=2.0; ierr = MatSetValues(B,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  i=2; j=2; val=3.0; ierr = MatSetValues(B,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  i=3; j=0; val=4.0; ierr = MatSetValues(B,1,&i,1,&j,&val,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "B: %d,%d\n",m,n);

  /* Register the multiplication routine. */
  ierr = MatRegisterOp(((PetscObject)A)->comm, PETSC_NULL,(PetscVoidFunction)MatMatMult_SeqAIJ_SeqDense,"MatMatMult",2,MATSEQAIJ,MATSEQDENSE); CHKERRQ(ierr);

  /* Remove this routine from B's FList to prevent double dispatch from that FList: do this by passing in a null url and function pointer together with the type- (and language-) specific op name. */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMatMult_seqaij_seqdense_C", PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);

  /* C = A*B */
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,2.0,&C);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
 
  /* Free space */
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}
