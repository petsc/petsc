
static char help[] = "Tests various routines in MatMPISBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,y,u,s1,s2;
  Mat            A,sA,sB;
  PetscRandom    rctx;
  PetscReal      r1,r2,rnorm,tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscScalar    one=1.0, neg_one=-1.0, value[3], four=4.0,alpha=0.1;
  PetscInt       n,col[3],n1,block,row,i,j,i2,j2,Ii,J,rstart,rend,bs=1,mbs=16,d_nz=3,o_nz=3,prob=1;
  PetscMPIInt    size,rank;
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-prob",&prob,NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a BAIJ matrix A */
  n = mbs*bs;
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetType(A,MATBAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIBAIJSetPreallocation(A,bs,d_nz,NULL,o_nz,NULL));
  PetscCall(MatSeqBAIJSetPreallocation(A,bs,d_nz,NULL));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  if (bs == 1) {
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 0.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      }
      i       = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1.0; value[2]=0.0;
      PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));

      i        = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 0.0; value[1] = -1.0; value[2]=0.1;
      PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    } else if (prob ==2) { /* matrix for the five point stencil */
      n1 = (int) PetscSqrtReal((PetscReal)n);
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0)    {J = Ii - n1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          if (i<n1-1) {J = Ii + n1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          if (j>0)    {J = Ii - 1;  PetscCall(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          if (j<n1-1) {J = Ii + 1;  PetscCall(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));}
          PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES));
        }
      }
    }
    /* end of if (bs == 1) */
  } else {  /* bs > 1 */
    for (block=0; block<n/bs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      }
      i       = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;
      PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));

      i       = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;
      value[0]=4.0; value[1] = -1.0;
      PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;
      PetscCall(MatSetValues(A,1,&i,1,col,value,INSERT_VALUES));
      col[0]=i; row=i+bs;
      PetscCall(MatSetValues(A,1,&row,1,col,value,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /* Get SBAIJ matrix sA from A */
  PetscCall(MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA));

  /* Test MatGetSize(), MatGetLocalSize() */
  PetscCall(MatGetSize(sA, &i,&j));
  PetscCall(MatGetSize(A, &i2,&j2));
  i   -= i2; j -= j2;
  if (i || j) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetSize()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  PetscCall(MatGetLocalSize(sA, &i,&j));
  PetscCall(MatGetLocalSize(A, &i2,&j2));
  i2  -= i; j2 -= j;
  if (i2 || j2) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetLocalSize()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  /* vectors */
  /*--------------------*/
  /* i is obtained from MatGetLocalSize() */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,i,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&y));
  PetscCall(VecDuplicate(x,&u));
  PetscCall(VecDuplicate(x,&s1));
  PetscCall(VecDuplicate(x,&s2));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(VecSetRandom(x,rctx));
  PetscCall(VecSet(u,one));

  /* Test MatNorm() */
  PetscCall(MatNorm(A,NORM_FROBENIUS,&r1));
  PetscCall(MatNorm(sA,NORM_FROBENIUS,&r2));
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS(), Anorm=%16.14e, sAnorm=%16.14e bs=%" PetscInt_FMT "\n",(double)r1,(double)r2,bs));
  }
  PetscCall(MatNorm(A,NORM_INFINITY,&r1));
  PetscCall(MatNorm(sA,NORM_INFINITY,&r2));
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_INFINITY(), Anorm=%16.14e, sAnorm=%16.14e bs=%" PetscInt_FMT "\n",(double)r1,(double)r2,bs));
  }
  PetscCall(MatNorm(A,NORM_1,&r1));
  PetscCall(MatNorm(sA,NORM_1,&r2));
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_1(), Anorm=%16.14e, sAnorm=%16.14e bs=%" PetscInt_FMT "\n",(double)r1,(double)r2,bs));
  }

  /* Test MatGetOwnershipRange() */
  PetscCall(MatGetOwnershipRange(sA,&rstart,&rend));
  PetscCall(MatGetOwnershipRange(A,&i2,&j2));
  i2  -= rstart; j2 -= rend;
  if (i2 || j2) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MaGetOwnershipRange()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  /* Test MatDiagonalScale() */
  PetscCall(MatDiagonalScale(A,x,x));
  PetscCall(MatDiagonalScale(sA,x,x));
  PetscCall(MatMultEqual(A,sA,10,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDiagonalScale");

  /* Test MatGetDiagonal(), MatScale() */
  PetscCall(MatGetDiagonal(A,s1));
  PetscCall(MatGetDiagonal(sA,s2));
  PetscCall(VecNorm(s1,NORM_1,&r1));
  PetscCall(VecNorm(s2,NORM_1,&r2));
  r1  -= r2;
  if (r1<-tol || r1>tol) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDiagonalScale() or MatGetDiagonal(), r1=%g \n",rank,(double)r1));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  PetscCall(MatScale(A,alpha));
  PetscCall(MatScale(sA,alpha));

  /* Test MatGetRowMaxAbs() */
  PetscCall(MatGetRowMaxAbs(A,s1,NULL));
  PetscCall(MatGetRowMaxAbs(sA,s2,NULL));

  PetscCall(VecNorm(s1,NORM_1,&r1));
  PetscCall(VecNorm(s2,NORM_1,&r2));
  r1  -= r2;
  if (r1<-tol || r1>tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRowMaxAbs() \n"));
  }

  /* Test MatMult(), MatMultAdd() */
  PetscCall(MatMultEqual(A,sA,10,&flg));
  if (!flg) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  PetscCall(MatMultAddEqual(A,sA,10,&flg));
  if (!flg) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }

  /* Test MatMultTranspose(), MatMultTransposeAdd() */
  for (i=0; i<10; i++) {
    PetscCall(VecSetRandom(x,rctx));
    PetscCall(MatMultTranspose(A,x,s1));
    PetscCall(MatMultTranspose(sA,x,s2));
    PetscCall(VecNorm(s1,NORM_1,&r1));
    PetscCall(VecNorm(s2,NORM_1,&r2));
    r1  -= r2;
    if (r1<-tol || r1>tol) {
      PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale(), err=%g\n",rank,(double)r1));
      PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    }
  }
  for (i=0; i<10; i++) {
    PetscCall(VecSetRandom(x,rctx));
    PetscCall(VecSetRandom(y,rctx));
    PetscCall(MatMultTransposeAdd(A,x,y,s1));
    PetscCall(MatMultTransposeAdd(sA,x,y,s2));
    PetscCall(VecNorm(s1,NORM_1,&r1));
    PetscCall(VecNorm(s2,NORM_1,&r2));
    r1  -= r2;
    if (r1<-tol || r1>tol) {
      PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd(), err=%g \n",rank,(double)r1));
      PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    }
  }

  /* Test MatDuplicate() */
  PetscCall(MatDuplicate(sA,MAT_COPY_VALUES,&sB));
  PetscCall(MatEqual(sA,sB,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Error in MatDuplicate(), sA != sB \n"));
  }
  PetscCall(MatMultEqual(sA,sB,5,&flg));
  if (!flg) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMult()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }
  PetscCall(MatMultAddEqual(sA,sB,5,&flg));
  if (!flg) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMultAdd(()\n",rank));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  }
  PetscCall(MatDestroy(&sB));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscCall(MatDestroy(&sA));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 3}}
      args: -bs {{1 2 3  5  7 8}} -mat_ignore_lower_triangular -prob {{1 2}}

TEST*/
