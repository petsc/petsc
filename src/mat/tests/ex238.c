static char help[] = "Creates MatSeqBAIJ matrix of given BS for timing tests of MatMult().\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  Vec            x,y;
  PetscErrorCode ierr;
  PetscInt       m=50000,bs=12,i,j,k,l,row,col,M, its = 25;
  PetscScalar    rval,*vals;
  PetscRandom    rdm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-its",&its,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  M    = m*bs;
  CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,27,NULL,&A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,M,&x));
  CHKERRQ(VecDuplicate(x,&y));

  /* For each block row insert at most 27 blocks */
  CHKERRQ(PetscMalloc1(bs*bs,&vals));
  for (i=0; i<m; i++) {
    row = i;
    for (j=0; j<27; j++) {
      CHKERRQ(PetscRandomGetValue(rdm,&rval));
      col  = (PetscInt)(PetscRealPart(rval)*m);
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          CHKERRQ(PetscRandomGetValue(rdm,&rval));
          vals[k*bs + l] = rval;
        }
      }
      CHKERRQ(MatSetValuesBlocked(A,1,&row,1,&col,vals,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree(vals));

  /* Time MatMult(), MatMultAdd() */
  for (i=0; i<its; i++) {
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(MatMult(A,x,y));
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(VecSetRandom(y,rdm));
    CHKERRQ(MatMultAdd(A,x,y,y));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscRandomDestroy(&rdm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
     requires: defined(PETSC_USING_64BIT_PTR)
     output_file: output/ex238_1.out
     test:
       suffix: 1
       args: -mat_block_size 1 -mat_size 1000 -its 2
     test:
       suffix: 2
       args: -mat_block_size 2 -mat_size 1000 -its 2
     test:
       suffix: 4
       args: -mat_block_size 4 -mat_size 1000 -its 2
     test:
       suffix: 5
       args: -mat_block_size 5 -mat_size 1000 -its 2
     test:
       suffix: 6
       args: -mat_block_size 6 -mat_size 1000 -its 2
     test:
       suffix: 8
       args: -mat_block_size 8 -mat_size 1000 -its 2
     test:
       suffix: 12
       args: -mat_block_size 12 -mat_size 1000 -its 2
     test:
       suffix: 15
       args: -mat_block_size 15 -mat_size 1000 -its 2

TEST*/
