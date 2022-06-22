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
  PetscMPIInt    size;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Create MAIJ matrix, P */
  PetscCall(MatCreate(PETSC_COMM_SELF,&pA));
  PetscCall(MatSetSizes(pA,3,3,3,3));
  PetscCall(MatSetType(pA,MATSEQAIJ));
  PetscCall(MatSetUp(pA));
  PetscCall(MatSetOption(pA,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  PetscCall(MatSetValues(pA,3,pij,3,pij,pa,ADD_VALUES));
  PetscCall(MatAssemblyBegin(pA,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(pA,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateMAIJ(pA,3,&P));
  PetscCall(MatDestroy(&pA));

  /* Create AIJ equivalent matrix, aijP, for comparison testing */
  PetscCall(MatConvert(P,MATSEQAIJ,MAT_INITIAL_MATRIX,&aijP));

  /* Create AIJ matrix A */
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,9,9,9,9));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  PetscCall(MatSetValues(A,3,aij[0],3,aij[0],pa,ADD_VALUES));
  PetscCall(MatSetValues(A,3,aij[1],3,aij[1],pa,ADD_VALUES));
  PetscCall(MatSetValues(A,3,aij[2],3,aij[2],pa,ADD_VALUES));
  for (i=0; i<9; i++) {
    PetscCall(MatSetValue(A,i,i,one,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Perform PtAP_SeqAIJ_SeqAIJ for comparison testing */
  PetscCall(MatPtAP(A,aijP,MAT_INITIAL_MATRIX,1.,&C));

  /* Perform PtAP_SeqAIJ_SeqMAIJ */
  /* Developer API */
  PetscCall(MatProductCreate(A,P,NULL,&mC));
  PetscCall(MatProductSetType(mC,MATPRODUCT_PtAP));
  PetscCall(MatProductSetAlgorithm(mC,"default"));
  PetscCall(MatProductSetFill(mC,PETSC_DEFAULT));
  PetscCall(MatProductSetFromOptions(mC));
  PetscCall(MatProductSymbolic(mC));
  PetscCall(MatProductNumeric(mC));
  PetscCall(MatProductNumeric(mC));

  /* Check mC = C */
  PetscCall(MatEqual(C,mC,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"MatProduct C != mC");
  PetscCall(MatDestroy(&mC));

  /* User API */
  PetscCall(MatPtAP(A,P,MAT_INITIAL_MATRIX,1.,&mC));
  PetscCall(MatPtAP(A,P,MAT_REUSE_MATRIX,1.,&mC));

  /* Check mC = C */
  PetscCall(MatEqual(C,mC,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"MatPtAP C != mC");
  PetscCall(MatDestroy(&mC));

  /* Cleanup */
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&aijP));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex101.out

TEST*/
