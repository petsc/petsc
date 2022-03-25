static char help[] = "Creates MatSeqBAIJ matrix of given BS for timing tests of MatMult().\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  Vec            x,y;
  PetscInt       m=50000,bs=12,i,j,k,l,row,col,M, its = 25;
  PetscScalar    rval,*vals;
  PetscRandom    rdm;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-its",&its,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  M    = m*bs;
  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,27,NULL,&A));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,M,&x));
  PetscCall(VecDuplicate(x,&y));

  /* For each block row insert at most 27 blocks */
  PetscCall(PetscMalloc1(bs*bs,&vals));
  for (i=0; i<m; i++) {
    row = i;
    for (j=0; j<27; j++) {
      PetscCall(PetscRandomGetValue(rdm,&rval));
      col  = (PetscInt)(PetscRealPart(rval)*m);
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          PetscCall(PetscRandomGetValue(rdm,&rval));
          vals[k*bs + l] = rval;
        }
      }
      PetscCall(MatSetValuesBlocked(A,1,&row,1,&col,vals,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(vals));

  /* Time MatMult(), MatMultAdd() */
  for (i=0; i<its; i++) {
    PetscCall(VecSetRandom(x,rdm));
    PetscCall(MatMult(A,x,y));
    PetscCall(VecSetRandom(x,rdm));
    PetscCall(VecSetRandom(y,rdm));
    PetscCall(MatMultAdd(A,x,y,y));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
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
