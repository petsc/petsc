
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL);CHKERRQ(ierr);

  n   = mbs*bs;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,NULL, &A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatCreateSeqSBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,NULL, &sA);CHKERRQ(ierr);
  ierr = MatSetOption(sA,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(A,&Ii,&J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRQ(ierr);
  if (i-Ii || j-J) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n");CHKERRQ(ierr);
  }

  /* Assemble matrix */
  if (bs == 1) {
    ierr = PetscOptionsGetInt(NULL,NULL,"-test_problem",&prob,NULL);CHKERRQ(ierr);
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        ierr   = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;

      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr    = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr    = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;

      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr     = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr     = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    } else if (prob ==2) { /* matrix for the five point stencil */
      n1 = (PetscInt) (PetscSqrtReal((PetscReal)n) + 0.001);
      PetscAssertFalse(n1*n1 - n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive integer!");
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0) {
            J    = Ii - n1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (i<n1-1) {
            J    = Ii + n1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j>0) {
            J    = Ii - 1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j<n1-1) {
            J    = Ii + 1;
            ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
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
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        ierr   = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;

      value[0]=-1.0; value[1]=4.0;
      ierr    = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr    = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;

      value[0]=4.0; value[1] = -1.0;
      ierr    = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr    = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
#endif
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++) {
      col[0]=i+bs;
      ierr  = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr  = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr  = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr  = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatNorm() */
  ierr   = MatNorm(A,NORM_FROBENIUS,&norm1);CHKERRQ(ierr);
  ierr   = MatNorm(sA,NORM_FROBENIUS,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), fnorm1-fnorm2=%16.14e\n",(double)norm1);CHKERRQ(ierr);
  }
  ierr   = MatNorm(A,NORM_INFINITY,&norm1);CHKERRQ(ierr);
  ierr   = MatNorm(sA,NORM_INFINITY,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), inf_norm1-inf_norm2=%16.14e\n",(double)norm1);CHKERRQ(ierr);
  }

  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  ierr = MatGetInfo(A,MAT_LOCAL,&minfo1);CHKERRQ(ierr);
  ierr = MatGetInfo(sA,MAT_LOCAL,&minfo2);CHKERRQ(ierr);
  i = (int) (minfo1.nz_used - minfo2.nz_used);
  j = (int) (minfo1.nz_allocated - minfo2.nz_allocated);
  if (i<0 || j<0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetInfo()\n");CHKERRQ(ierr);
  }

  ierr = MatGetSize(A,&Ii,&J);CHKERRQ(ierr);
  ierr = MatGetSize(sA,&i,&j);CHKERRQ(ierr);
  if (i-Ii || j-J) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetSize()\n");CHKERRQ(ierr);
  }

  ierr = MatGetBlockSize(A, &Ii);CHKERRQ(ierr);
  ierr = MatGetBlockSize(sA, &i);CHKERRQ(ierr);
  if (i-Ii) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetBlockSize()\n");CHKERRQ(ierr);
  }

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);

  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);

  ierr = MatDiagonalScale(A,x,x);CHKERRQ(ierr);
  ierr = MatDiagonalScale(sA,x,x);CHKERRQ(ierr);

  ierr   = MatGetDiagonal(A,s1);CHKERRQ(ierr);
  ierr   = MatGetDiagonal(sA,s2);CHKERRQ(ierr);
  ierr   = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
  ierr   = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatGetDiagonal() \n");CHKERRQ(ierr);
  }

  ierr = MatScale(A,alpha);CHKERRQ(ierr);
  ierr = MatScale(sA,alpha);CHKERRQ(ierr);

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) {
    ierr   = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr   = MatMult(A,x,s1);CHKERRQ(ierr);
    ierr   = MatMult(sA,x,s2);CHKERRQ(ierr);
    ierr   = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
    ierr   = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMult(), MatDiagonalScale() or MatScale()\n");CHKERRQ(ierr);
    }
  }

  for (i=0; i<40; i++) {
    ierr   = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr   = VecSetRandom(y,rdm);CHKERRQ(ierr);
    ierr   = MatMultAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr   = MatMultAdd(sA,x,y,s2);CHKERRQ(ierr);
    ierr   = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
    ierr   = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd(), MatDiagonalScale() or MatScale() \n");CHKERRQ(ierr);
    }
  }

  /* Test MatReordering() */
  ierr = MatGetOrdering(A,MATORDERINGNATURAL,&isrow,&iscol);CHKERRQ(ierr);
  ip   = isrow;

  if (reorder) {
    IS       nip;
    PetscInt *nip_ptr;
    ierr = PetscMalloc1(mbs,&nip_ptr);CHKERRQ(ierr);
    ierr = ISGetIndices(ip,&ip_ptr);CHKERRQ(ierr);
    ierr = PetscArraycpy(nip_ptr,ip_ptr,mbs);CHKERRQ(ierr);
    i    = nip_ptr[1]; nip_ptr[1] = nip_ptr[mbs-2]; nip_ptr[mbs-2] = i;
    i    = nip_ptr[0]; nip_ptr[0] = nip_ptr[mbs-1]; nip_ptr[mbs-1] = i;
    ierr = ISRestoreIndices(ip,&ip_ptr);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,mbs,nip_ptr,PETSC_COPY_VALUES,&nip);CHKERRQ(ierr);
    ierr = PetscFree(nip_ptr);CHKERRQ(ierr);

    ierr = MatReorderingSeqSBAIJ(sA, ip);CHKERRQ(ierr);
    ierr = ISDestroy(&nip);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&sA);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -bs {{1 2 3 4 5 6 7 8}}

TEST*/
