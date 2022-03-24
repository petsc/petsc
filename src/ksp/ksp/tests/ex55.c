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
  PetscInt       n=10,i,col[3];
  Vec            x,b;
  Mat            A,B;
  KSP            ksp;
  PC             pc,subpc;
  PetscScalar    value[3];

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Create a diagonal matrix with a given distribution of diagonal elements */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
   /*
     Assemble matrix
  */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateVecs(A,&x,&b));

  /* Create a KSP object */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));

  /* Set up a composite preconditioner */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCOMPOSITE)); /* default composite with single Identity PC */
  CHKERRQ(PCCompositeSetType(pc,PC_COMPOSITE_ADDITIVE));
  CHKERRQ(PCCompositeAddPCType(pc,PCLU));
  CHKERRQ(PCCompositeGetPC(pc,0,&subpc));
  /*  B is set to the diagonal of A; this demonstrates that setting the operator for a subpc changes the preconditioning */
  CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B));
  CHKERRQ(MatGetDiagonal(A,b));
  CHKERRQ(MatDiagonalSet(B,b,ADD_VALUES));
  CHKERRQ(PCSetOperators(subpc,B,B));
  CHKERRQ(PCCompositeAddPCType(pc,PCNONE));

  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -ksp_monitor -pc_composite_type multiplicative

TEST*/
