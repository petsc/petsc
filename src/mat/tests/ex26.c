
static char help[] = "Tests MatGetRowIJ for SeqAIJ, SeqBAIJ and SeqSBAIJ\n\n";

#include <petscmat.h>

PetscErrorCode DumpCSR(Mat A,PetscInt shift,PetscBool symmetric,PetscBool compressed)
{
  MatType        type;
  PetscInt       i,j,nr,bs = 1;
  const PetscInt *ia,*ja;
  PetscBool      done,isseqbaij,isseqsbaij;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQBAIJ,&isseqbaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQSBAIJ,&isseqsbaij);CHKERRQ(ierr);
  if (isseqbaij || isseqsbaij) {
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  }
  ierr = MatGetType(A,&type);CHKERRQ(ierr);
  ierr = MatGetRowIJ(A,shift,symmetric,compressed,&nr,&ia,&ja,&done);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"===========================================================\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"CSR for %s: shift %D symmetric %D compressed %D\n",type,shift,(PetscInt)symmetric,(PetscInt)compressed);CHKERRQ(ierr);
  for (i=0;i<nr;i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%D:",i+shift);CHKERRQ(ierr);
    for (j=ia[i];j<ia[i+1];j++) {
      ierr = PetscPrintf(PETSC_COMM_SELF," %D",ja[j-shift]);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
  }
  ierr = MatRestoreRowIJ(A,shift,symmetric,compressed,&nr,&ia,&ja,&done);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A,B,C;
  PetscInt       i,j,k,m = 3,n = 3,bs = 1;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  /* adjust sizes by block size */
  if (m%bs) m += bs-m%bs;
  if (n%bs) n += bs-n%bs;

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(B,bs);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m*n,m*n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(C,bs);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      
      PetscScalar v = -1.0;
      PetscInt    Ii = j + n*i,J;
      J = Ii - n;
      if (J>=0)  {
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      J = Ii + n;
      if (J<m*n) {
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      J = Ii - 1;
      if (J>=0)  {
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      J = Ii + 1;
      if (J<m*n) {
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(B,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      v = 4.0;
      ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* test MatGetRowIJ for the three Mat types */
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatView(B,NULL);CHKERRQ(ierr);
  ierr = MatView(C,NULL);CHKERRQ(ierr);
  for (i=0;i<2;i++) {
    PetscInt shift = i;
    for (j=0;j<2;j++) {
      PetscBool symmetric = ((j>0) ? PETSC_FALSE : PETSC_TRUE);
      for (k=0;k<2;k++) {
        PetscBool compressed = ((k>0) ? PETSC_FALSE : PETSC_TRUE);
        ierr = DumpCSR(A,shift,symmetric,compressed);CHKERRQ(ierr);
        ierr = DumpCSR(B,shift,symmetric,compressed);CHKERRQ(ierr);
        ierr = DumpCSR(C,shift,symmetric,compressed);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

   test:
      suffix: 2
      args: -bs 2

TEST*/
