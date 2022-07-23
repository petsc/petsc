
static char help[] = "Tests MatGetRowIJ for SeqAIJ, SeqBAIJ and SeqSBAIJ\n\n";

#include <petscmat.h>

PetscErrorCode DumpCSR(Mat A,PetscInt shift,PetscBool symmetric,PetscBool compressed)
{
  MatType        type;
  PetscInt       i,j,nr,bs = 1;
  const PetscInt *ia,*ja;
  PetscBool      done,isseqbaij,isseqsbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQBAIJ,&isseqbaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQSBAIJ,&isseqsbaij));
  if (isseqbaij || isseqsbaij) {
    PetscCall(MatGetBlockSize(A,&bs));
  }
  PetscCall(MatGetType(A,&type));
  PetscCall(MatGetRowIJ(A,shift,symmetric,compressed,&nr,&ia,&ja,&done));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"===========================================================\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"CSR for %s: shift %" PetscInt_FMT " symmetric %" PetscInt_FMT " compressed %" PetscInt_FMT "\n",type,shift,(PetscInt)symmetric,(PetscInt)compressed));
  for (i=0;i<nr;i++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT ":",i+shift));
    for (j=ia[i];j<ia[i+1];j++) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF," %" PetscInt_FMT,ja[j-shift]));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n"));
  }
  PetscCall(MatRestoreRowIJ(A,shift,symmetric,compressed,&nr,&ia,&ja,&done));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B,C;
  PetscInt       i,j,k,m = 3,n = 3,bs = 1;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  /* adjust sizes by block size */
  if (m%bs) m += bs-m%bs;
  if (n%bs) n += bs-n%bs;

  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetBlockSize(A,bs));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatCreate(PETSC_COMM_SELF,&B));
  PetscCall(MatSetSizes(B,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetBlockSize(B,bs));
  PetscCall(MatSetType(B,MATSEQBAIJ));
  PetscCall(MatSetUp(B));
  PetscCall(MatCreate(PETSC_COMM_SELF,&C));
  PetscCall(MatSetSizes(C,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetBlockSize(C,bs));
  PetscCall(MatSetType(C,MATSEQSBAIJ));
  PetscCall(MatSetUp(C));
  PetscCall(MatSetOption(C,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {

      PetscScalar v = -1.0;
      PetscInt    Ii = j + n*i,J;
      J = Ii - n;
      if (J>=0)  {
        PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      J = Ii + n;
      if (J<m*n) {
        PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      J = Ii - 1;
      if (J>=0)  {
        PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      J = Ii + 1;
      if (J<m*n) {
        PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      v = 4.0;
      PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
      PetscCall(MatSetValues(B,1,&Ii,1,&Ii,&v,INSERT_VALUES));
      PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* test MatGetRowIJ for the three Mat types */
  PetscCall(MatView(A,NULL));
  PetscCall(MatView(B,NULL));
  PetscCall(MatView(C,NULL));
  for (i=0;i<2;i++) {
    PetscInt shift = i;
    for (j=0;j<2;j++) {
      PetscBool symmetric = ((j>0) ? PETSC_FALSE : PETSC_TRUE);
      for (k=0;k<2;k++) {
        PetscBool compressed = ((k>0) ? PETSC_FALSE : PETSC_TRUE);
        PetscCall(DumpCSR(A,shift,symmetric,compressed));
        PetscCall(DumpCSR(B,shift,symmetric,compressed));
        PetscCall(DumpCSR(C,shift,symmetric,compressed));
      }
    }
  }
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -bs 2

TEST*/
