
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
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-prob",&prob,NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Create a BAIJ matrix A */
  n = mbs*bs;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,bs,d_nz,NULL,o_nz,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A,bs,d_nz,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  if (bs == 1) {
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 0.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i       = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1.0; value[2]=0.0;
      ierr    = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i        = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 0.0; value[1] = -1.0; value[2]=0.1;
      ierr     = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    } else if (prob ==2) { /* matrix for the five point stencil */
      n1 = (int) PetscSqrtReal((PetscReal)n);
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0)    {J = Ii - n1; ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (i<n1-1) {J = Ii + n1; ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (j>0)    {J = Ii - 1;  ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (j<n1-1) {J = Ii + 1;  ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          ierr = MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES);CHKERRQ(ierr);
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
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i       = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;
      ierr    = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i       = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;
      value[0]=4.0; value[1] = -1.0;
      ierr    = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;
      ierr  = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr  = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  /* Get SBAIJ matrix sA from A */
  ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);

  /* Test MatGetSize(), MatGetLocalSize() */
  ierr = MatGetSize(sA, &i,&j);CHKERRQ(ierr);
  ierr = MatGetSize(A, &i2,&j2);CHKERRQ(ierr);
  i   -= i2; j -= j2;
  if (i || j) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetSize()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  ierr = MatGetLocalSize(sA, &i,&j);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &i2,&j2);CHKERRQ(ierr);
  i2  -= i; j2 -= j;
  if (i2 || j2) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetLocalSize()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  /* vectors */
  /*--------------------*/
  /* i is obtained from MatGetLocalSize() */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,i,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);

  /* Test MatNorm() */
  ierr  = MatNorm(A,NORM_FROBENIUS,&r1);CHKERRQ(ierr);
  ierr  = MatNorm(sA,NORM_FROBENIUS,&r2);CHKERRQ(ierr);
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS(), Anorm=%16.14e, sAnorm=%16.14e bs=%D\n",r1,r2,bs);CHKERRQ(ierr);
  }
  ierr  = MatNorm(A,NORM_INFINITY,&r1);CHKERRQ(ierr);
  ierr  = MatNorm(sA,NORM_INFINITY,&r2);CHKERRQ(ierr);
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_INFINITY(), Anorm=%16.14e, sAnorm=%16.14e bs=%D\n",r1,r2,bs);CHKERRQ(ierr);
  }
  ierr  = MatNorm(A,NORM_1,&r1);CHKERRQ(ierr);
  ierr  = MatNorm(sA,NORM_1,&r2);CHKERRQ(ierr);
  rnorm = PetscAbsReal(r1-r2)/r2;
  if (rnorm > tol && rank == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_1(), Anorm=%16.14e, sAnorm=%16.14e bs=%D\n",r1,r2,bs);CHKERRQ(ierr);
  }

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(sA,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&i2,&j2);CHKERRQ(ierr);
  i2  -= rstart; j2 -= rend;
  if (i2 || j2) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MaGetOwnershipRange()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  /* Test MatDiagonalScale() */
  ierr = MatDiagonalScale(A,x,x);CHKERRQ(ierr);
  ierr = MatDiagonalScale(sA,x,x);CHKERRQ(ierr);
  ierr = MatMultEqual(A,sA,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDiagonalScale");

  /* Test MatGetDiagonal(), MatScale() */
  ierr = MatGetDiagonal(A,s1);CHKERRQ(ierr);
  ierr = MatGetDiagonal(sA,s2);CHKERRQ(ierr);
  ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
  r1  -= r2;
  if (r1<-tol || r1>tol) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDiagonalScale() or MatGetDiagonal(), r1=%g \n",rank,(double)r1);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  ierr = MatScale(A,alpha);CHKERRQ(ierr);
  ierr = MatScale(sA,alpha);CHKERRQ(ierr);

  /* Test MatGetRowMaxAbs() */
  ierr = MatGetRowMaxAbs(A,s1,NULL);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(sA,s2,NULL);CHKERRQ(ierr);

  ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
  r1  -= r2;
  if (r1<-tol || r1>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRowMaxAbs() \n");CHKERRQ(ierr);
  }

  /* Test MatMult(), MatMultAdd() */
  ierr = MatMultEqual(A,sA,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  ierr = MatMultAddEqual(A,sA,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  /* Test MatMultTranspose(), MatMultTransposeAdd() */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,x,s1);CHKERRQ(ierr);
    ierr = MatMultTranspose(sA,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
    r1  -= r2;
    if (r1<-tol || r1>tol) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale(), err=%g\n",rank,(double)r1);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    }
  }
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(sA,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
    r1  -= r2;
    if (r1<-tol || r1>tol) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd(), err=%g \n",rank,(double)r1);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    }
  }

  /* Test MatDuplicate() */
  ierr = MatDuplicate(sA,MAT_COPY_VALUES,&sB);CHKERRQ(ierr);
  ierr = MatEqual(sA,sB,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," Error in MatDuplicate(), sA != sB \n");CHKERRQ(ierr);
  }
  ierr = MatMultEqual(sA,sB,5,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMult()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }
  ierr = MatMultAddEqual(sA,sB,5,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMultAdd(()\n",rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&sB);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = MatDestroy(&sA);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{1 3}}
      args: -bs {{1 2 3  5  7 8}} -mat_ignore_lower_triangular -prob {{1 2}}

TEST*/
