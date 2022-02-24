
static char help[] = "Tests MatGetRowIJ for SeqAIJ, SeqBAIJ and SeqSBAIJ\n\n";

#include <petscmat.h>

PetscErrorCode DumpCSR(Mat A,PetscInt shift,PetscBool symmetric,PetscBool compressed)
{
  MatType        type;
  PetscInt       i,j,nr,bs = 1;
  const PetscInt *ia,*ja;
  PetscBool      done,isseqbaij,isseqsbaij;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQBAIJ,&isseqbaij));
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQSBAIJ,&isseqsbaij));
  if (isseqbaij || isseqsbaij) {
    CHKERRQ(MatGetBlockSize(A,&bs));
  }
  CHKERRQ(MatGetType(A,&type));
  CHKERRQ(MatGetRowIJ(A,shift,symmetric,compressed,&nr,&ia,&ja,&done));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"===========================================================\n"));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"CSR for %s: shift %" PetscInt_FMT " symmetric %" PetscInt_FMT " compressed %" PetscInt_FMT "\n",type,shift,(PetscInt)symmetric,(PetscInt)compressed));
  for (i=0;i<nr;i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT ":",i+shift));
    for (j=ia[i];j<ia[i+1];j++) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %" PetscInt_FMT,ja[j-shift]));
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));
  }
  CHKERRQ(MatRestoreRowIJ(A,shift,symmetric,compressed,&nr,&ia,&ja,&done));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B,C;
  PetscInt       i,j,k,m = 3,n = 3,bs = 1;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  /* adjust sizes by block size */
  if (m%bs) m += bs-m%bs;
  if (n%bs) n += bs-n%bs;

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetBlockSize(A,bs));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&B));
  CHKERRQ(MatSetSizes(B,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetBlockSize(B,bs));
  CHKERRQ(MatSetType(B,MATSEQBAIJ));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetBlockSize(C,bs));
  CHKERRQ(MatSetType(C,MATSEQSBAIJ));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatSetOption(C,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {

      PetscScalar v = -1.0;
      PetscInt    Ii = j + n*i,J;
      J = Ii - n;
      if (J>=0)  {
        CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      J = Ii + n;
      if (J<m*n) {
        CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      J = Ii - 1;
      if (J>=0)  {
        CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      J = Ii + 1;
      if (J<m*n) {
        CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES));
        CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      }
      v = 4.0;
      CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
      CHKERRQ(MatSetValues(B,1,&Ii,1,&Ii,&v,INSERT_VALUES));
      CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* test MatGetRowIJ for the three Mat types */
  CHKERRQ(MatView(A,NULL));
  CHKERRQ(MatView(B,NULL));
  CHKERRQ(MatView(C,NULL));
  for (i=0;i<2;i++) {
    PetscInt shift = i;
    for (j=0;j<2;j++) {
      PetscBool symmetric = ((j>0) ? PETSC_FALSE : PETSC_TRUE);
      for (k=0;k<2;k++) {
        PetscBool compressed = ((k>0) ? PETSC_FALSE : PETSC_TRUE);
        CHKERRQ(DumpCSR(A,shift,symmetric,compressed));
        CHKERRQ(DumpCSR(B,shift,symmetric,compressed));
        CHKERRQ(DumpCSR(C,shift,symmetric,compressed));
      }
    }
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -bs 2

TEST*/
