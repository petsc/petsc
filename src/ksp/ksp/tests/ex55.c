static const char help[]="Example demonstrating PCCOMPOSITE where one of the inner PCs uses a different operator\n\
\n";

/*T
   Concepts: KSP^using nested solves
   Concepts: PC^using composite PCs
   Processors: n
T*/
#include <petscksp.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n=10,i,col[3];
  Vec            x,b;
  Mat            A,B;
  KSP            ksp;
  PC             pc,subpc;
  PetscScalar    value[3];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Create a diagonal matrix with a given distribution of diagonal elements */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
   /*
     Assemble matrix
  */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);

  /* Create a KSP object */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /* Set up a composite preconditioner */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCOMPOSITE);CHKERRQ(ierr); /* default composite with single Identity PC */
  ierr = PCCompositeSetType(pc,PC_COMPOSITE_ADDITIVE);CHKERRQ(ierr);
  ierr = PCCompositeAddPC(pc,PCLU);CHKERRQ(ierr);
  ierr = PCCompositeGetPC(pc,0,&subpc);CHKERRQ(ierr);
  /*  B is set to the diagonal of A; this demonstrates that setting the operator for a subpc changes the preconditioning */
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatGetDiagonal(A,b);CHKERRQ(ierr);
  ierr = MatDiagonalSet(B,b,ADD_VALUES);CHKERRQ(ierr);
  ierr = PCSetOperators(subpc,B,B);CHKERRQ(ierr);
  ierr = PCCompositeAddPC(pc,PCNONE);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -ksp_monitor -pc_composite_type multiplicative

TEST*/
