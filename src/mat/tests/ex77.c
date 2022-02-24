
static char help[] = "Tests the various sequential routines in MatSBAIJ format. Same as ex74.c except diagonal entries of the matrices are zeros.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,y,b,s1,s2;
  Mat            A;           /* linear system matrix */
  Mat            sA;         /* symmetric part of the matrices */
  PetscInt       n,mbs=16,bs=1,nz=3,prob=2,i,j,col[3],row,Ii,J,n1;
  const PetscInt *ip_ptr;
  PetscScalar    neg_one = -1.0,value[3],alpha=0.1;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  IS             ip, isrow, iscol;
  PetscRandom    rdm;
  PetscBool      reorder=PETSC_FALSE;
  MatInfo        minfo1,minfo2;
  PetscReal      norm1,norm2,tol=10*PETSC_SMALL;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL));

  n   = mbs*bs;
  CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,NULL, &A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatCreateSeqSBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,NULL, &sA));
  CHKERRQ(MatSetOption(sA,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  /* Test MatGetOwnershipRange() */
  CHKERRQ(MatGetOwnershipRange(A,&Ii,&J));
  CHKERRQ(MatGetOwnershipRange(sA,&i,&j));
  if (i-Ii || j-J) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n"));
  }

  /* Assemble matrix */
  if (bs == 1) {
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-test_problem",&prob,NULL));
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
        CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;

      value[0]= 0.1; value[1]=-1; value[2]=2;
      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;

      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));
    } else if (prob ==2) { /* matrix for the five point stencil */
      n1 = (PetscInt) (PetscSqrtReal((PetscReal)n) + 0.001);
      PetscCheckFalse(n1*n1 - n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive integer!");
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0) {
            J    = Ii - n1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          if (i<n1-1) {
            J    = Ii + n1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          if (j>0) {
            J    = Ii - 1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
          if (j<n1-1) {
            J    = Ii + 1;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
            CHKERRQ(MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES));
          }
        }
      }
    }
  } else { /* bs > 1 */
#if defined(DIAGB)
    for (block=0; block<n/bs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
        CHKERRQ(MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES));
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;

      value[0]=-1.0; value[1]=4.0;
      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES));

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;

      value[0]=4.0; value[1] = -1.0;
      CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES));
    }
#endif
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;
      CHKERRQ(MatSetValues(A,1,&i,1,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES));
      col[0]=i; row=i+bs;
      CHKERRQ(MatSetValues(A,1,&row,1,col,value,INSERT_VALUES));
      CHKERRQ(MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY));

  /* Test MatNorm() */
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&norm1));
  CHKERRQ(MatNorm(sA,NORM_FROBENIUS,&norm2));
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), fnorm1-fnorm2=%16.14e\n",(double)norm1));
  }
  CHKERRQ(MatNorm(A,NORM_INFINITY,&norm1));
  CHKERRQ(MatNorm(sA,NORM_INFINITY,&norm2));
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), inf_norm1-inf_norm2=%16.14e\n",(double)norm1));
  }

  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  CHKERRQ(MatGetInfo(A,MAT_LOCAL,&minfo1));
  CHKERRQ(MatGetInfo(sA,MAT_LOCAL,&minfo2));
  i = (int) (minfo1.nz_used - minfo2.nz_used);
  j = (int) (minfo1.nz_allocated - minfo2.nz_allocated);
  if (i<0 || j<0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetInfo()\n"));
  }

  CHKERRQ(MatGetSize(A,&Ii,&J));
  CHKERRQ(MatGetSize(sA,&i,&j));
  if (i-Ii || j-J) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetSize()\n"));
  }

  CHKERRQ(MatGetBlockSize(A, &Ii));
  CHKERRQ(MatGetBlockSize(sA, &i));
  if (i-Ii) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetBlockSize()\n"));
  }

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(VecDuplicate(x,&s1));
  CHKERRQ(VecDuplicate(x,&s2));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecDuplicate(x,&b));

  CHKERRQ(VecSetRandom(x,rdm));

  CHKERRQ(MatDiagonalScale(A,x,x));
  CHKERRQ(MatDiagonalScale(sA,x,x));

  CHKERRQ(MatGetDiagonal(A,s1));
  CHKERRQ(MatGetDiagonal(sA,s2));
  CHKERRQ(VecNorm(s1,NORM_1,&norm1));
  CHKERRQ(VecNorm(s2,NORM_1,&norm2));
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatGetDiagonal() \n"));
  }

  CHKERRQ(MatScale(A,alpha));
  CHKERRQ(MatScale(sA,alpha));

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) {
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(MatMult(A,x,s1));
    CHKERRQ(MatMult(sA,x,s2));
    CHKERRQ(VecNorm(s1,NORM_1,&norm1));
    CHKERRQ(VecNorm(s2,NORM_1,&norm2));
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMult(), MatDiagonalScale() or MatScale()\n"));
    }
  }

  for (i=0; i<40; i++) {
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(VecSetRandom(y,rdm));
    CHKERRQ(MatMultAdd(A,x,y,s1));
    CHKERRQ(MatMultAdd(sA,x,y,s2));
    CHKERRQ(VecNorm(s1,NORM_1,&norm1));
    CHKERRQ(VecNorm(s2,NORM_1,&norm2));
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd(), MatDiagonalScale() or MatScale() \n"));
    }
  }

  /* Test MatReordering() */
  CHKERRQ(MatGetOrdering(A,MATORDERINGNATURAL,&isrow,&iscol));
  ip   = isrow;

  if (reorder) {
    IS       nip;
    PetscInt *nip_ptr;
    CHKERRQ(PetscMalloc1(mbs,&nip_ptr));
    CHKERRQ(ISGetIndices(ip,&ip_ptr));
    CHKERRQ(PetscArraycpy(nip_ptr,ip_ptr,mbs));
    i    = nip_ptr[1]; nip_ptr[1] = nip_ptr[mbs-2]; nip_ptr[mbs-2] = i;
    i    = nip_ptr[0]; nip_ptr[0] = nip_ptr[mbs-1]; nip_ptr[mbs-1] = i;
    CHKERRQ(ISRestoreIndices(ip,&ip_ptr));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,mbs,nip_ptr,PETSC_COPY_VALUES,&nip));
    CHKERRQ(PetscFree(nip_ptr));

    CHKERRQ(MatReorderingSeqSBAIJ(sA, ip));
    CHKERRQ(ISDestroy(&nip));
  }

  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&sA));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscRandomDestroy(&rdm));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -bs {{1 2 3 4 5 6 7 8}}

TEST*/
