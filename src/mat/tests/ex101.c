static char help[] = "Testing PtAP for SeqMAIJ matrix, P, with SeqAIJ matrix, A.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            pA,P,aijP;
  PetscScalar    pa[]={1.,-1.,0.,0.,1.,-1.,0.,0.,1.};
  PetscInt       i,pij[]={0,1,2};
  PetscInt       aij[3][3]={{0,1,2},{3,4,5},{6,7,8}};
  Mat            A,mC,C;
  PetscScalar    one=1.;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Create MAIJ matrix, P */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&pA));
  CHKERRQ(MatSetSizes(pA,3,3,3,3));
  CHKERRQ(MatSetType(pA,MATSEQAIJ));
  CHKERRQ(MatSetUp(pA));
  CHKERRQ(MatSetOption(pA,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatSetValues(pA,3,pij,3,pij,pa,ADD_VALUES));
  CHKERRQ(MatAssemblyBegin(pA,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(pA,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateMAIJ(pA,3,&P));
  CHKERRQ(MatDestroy(&pA));

  /* Create AIJ equivalent matrix, aijP, for comparison testing */
  CHKERRQ(MatConvert(P,MATSEQAIJ,MAT_INITIAL_MATRIX,&aijP));

  /* Create AIJ matrix A */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,9,9,9,9));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatSetValues(A,3,aij[0],3,aij[0],pa,ADD_VALUES));
  CHKERRQ(MatSetValues(A,3,aij[1],3,aij[1],pa,ADD_VALUES));
  CHKERRQ(MatSetValues(A,3,aij[2],3,aij[2],pa,ADD_VALUES));
  for (i=0; i<9; i++) {
    CHKERRQ(MatSetValue(A,i,i,one,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Perform PtAP_SeqAIJ_SeqAIJ for comparison testing */
  CHKERRQ(MatPtAP(A,aijP,MAT_INITIAL_MATRIX,1.,&C));

  /* Perform PtAP_SeqAIJ_SeqMAIJ */
  /* Developer API */
  CHKERRQ(MatProductCreate(A,P,NULL,&mC));
  CHKERRQ(MatProductSetType(mC,MATPRODUCT_PtAP));
  CHKERRQ(MatProductSetAlgorithm(mC,"default"));
  CHKERRQ(MatProductSetFill(mC,PETSC_DEFAULT));
  CHKERRQ(MatProductSetFromOptions(mC));
  CHKERRQ(MatProductSymbolic(mC));
  CHKERRQ(MatProductNumeric(mC));
  CHKERRQ(MatProductNumeric(mC));

  /* Check mC = C */
  CHKERRQ(MatEqual(C,mC,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"MatProduct C != mC");
  CHKERRQ(MatDestroy(&mC));

  /* User API */
  CHKERRQ(MatPtAP(A,P,MAT_INITIAL_MATRIX,1.,&mC));
  CHKERRQ(MatPtAP(A,P,MAT_REUSE_MATRIX,1.,&mC));

  /* Check mC = C */
  CHKERRQ(MatEqual(C,mC,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"MatPtAP C != mC");
  CHKERRQ(MatDestroy(&mC));

  /* Cleanup */
  CHKERRQ(MatDestroy(&P));
  CHKERRQ(MatDestroy(&aijP));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      output_file: output/ex101.out

TEST*/
