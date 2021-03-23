
static char help[] = "Creates MatSeqBAIJ matrix of given BS for timing tests of MatMult().\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  Vec            x,y;
  PetscErrorCode ierr;
  PetscInt       m=50000,bs=12,i,j,k,l,row,col,M;
  PetscScalar    rval,*vals;
  PetscRandom    rdm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL);CHKERRQ(ierr);
  M    = m*bs;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,27,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,M,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* For each block row insert at most 27 blocks */
  ierr = PetscMalloc1(bs*bs,&vals);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    row = i;
    for (j=0; j<27; j++) {
      ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
      col  = (PetscInt)(PetscRealPart(rval)*m);
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
          vals[k*bs + l] = rval;
        }
      }
      ierr = MatSetValuesBlocked(A,1,&row,1,&col,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);

  /* Time MatMult(), MatMultAdd() */
  for (i=0; i<25; i++) {
    ierr  = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr  = MatMult(A,x,y);CHKERRQ(ierr);
    ierr  = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr  = VecSetRandom(y,rdm);CHKERRQ(ierr);
    ierr  = MatMultAdd(A,x,y,y);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      requires: define(PETSC_USING_64BIT_PTR)
      args: -mat_block_size {{1 2 4 5 6 8 12 15}}

TEST*/
