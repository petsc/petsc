
static char help[] = "Tests the use of interface functions for MATIS matrices.\n\
This example tests: MatZeroRows(), MatZeroRowsLocal(), MatView(), MatDuplicate(),\n\
MatCopy(), MatGetSubMatrix(), MatGetLocalSubMatrix(), MatAXPY(), MatShift()\n\
MatDiagonalSet(), MatTranspose() and MatISGetMPIXAIJ()\n";

#include <petscmat.h>

PetscErrorCode TestMatZeroRows(Mat,Mat,IS,PetscScalar);
PetscErrorCode CheckMat(Mat,Mat,const char*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                    A,B,A2,B2;
  Mat                    Aee,Aeo,Aoe,Aoo;
  Vec                    x;
  ISLocalToGlobalMapping cmap,rmap;
  IS                     is,is2,reven,rodd,ceven,codd;
  PetscScalar            diag = 2.;
  PetscInt               n,m,i;
  PetscMPIInt            rank,size;
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  m = n = 2*size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  if ((size > 1 && m < 4) || m < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be more than two");
  if (n < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of cols should be more than two");

  /* create a MATIS matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATIS);CHKERRQ(ierr);
  /* Each process has the same l2gmap for rows and cols
     This is not the proper setting for MATIS for finite elements, it is just used to test the routines */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&cmap);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&rmap);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,rmap,cmap);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatISSetPreallocation(A,3,NULL,0,NULL);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscScalar v[3] = { -1.*(i+1),2.*(i+1),-1.*(i+1)};
    PetscInt    cols[3] = {(i-1+n)%n,i%n,(i+1)%n};
 
    ierr = MatSetValuesLocal(A,1,&i,3,cols,v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* test MatView */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatView\n");CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);

  /* Create a MPIAIJ matrix, same as A */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B,rmap,cmap);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,3,NULL,3,NULL);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscScalar v[3] = { -1.*(i+1),2.*(i+1),-1.*(i+1)};
    PetscInt    cols[3] = {(i-1+n)%n,i%n,(i+1)%n};
 
    ierr = MatSetValuesLocal(B,1,&i,3,cols,v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);

  /* test MatISGetMPIXAIJ */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatISGetMPIXAIJ\n");CHKERRQ(ierr);
  ierr = CheckMat(A,B,"MatISGetMPIXAIJ");CHKERRQ(ierr);

  /* test MatGetLocalSubMatrix */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatGetLocalSubMatrix\n");CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,m/2+m%2,0,2,&reven);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,m/2,1,2,&rodd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n/2+n%2,0,2,&ceven);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n/2,1,2,&codd);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,reven,ceven,&Aee);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,reven,codd,&Aeo);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,rodd,ceven,&Aoe);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,rodd,codd,&Aoo);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscScalar v[3] = { -1.*(i+1),2.*(i+1),-1.*(i+1)};
    PetscInt    cols[3] = {(i-1+n)%n,i%n,(i+1)%n};
    PetscInt    j,je,jo,colse[3], colso[3];
    PetscScalar ve[3], vo[3];

    for (j=0,je=0,jo=0;j<3;j++) {
      if (cols[j]%2) {
        vo[jo] = v[j];
        colso[jo++] = cols[j]/2;
      } else {
        ve[je] = v[j];
        colse[je++] = cols[j]/2;
      }
    }
    if (i%2) {
      PetscInt row = i/2;
      ierr = MatSetValuesLocal(Aoe,1,&row,je,colse,ve,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(Aoo,1,&row,jo,colso,vo,ADD_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt row = i/2;
      ierr = MatSetValuesLocal(Aee,1,&row,je,colse,ve,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(Aeo,1,&row,jo,colso,vo,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatRestoreLocalSubMatrix(A2,reven,ceven,&Aee);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(A2,reven,codd,&Aeo);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(A2,rodd,ceven,&Aoe);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(A2,rodd,codd,&Aoo);CHKERRQ(ierr);
  ierr = ISDestroy(&reven);CHKERRQ(ierr);
  ierr = ISDestroy(&ceven);CHKERRQ(ierr);
  ierr = ISDestroy(&rodd);CHKERRQ(ierr);
  ierr = ISDestroy(&codd);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAXPY(A2,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = CheckMat(A2,NULL,"MatGetLocalSubMatrix");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  /* test MatGetSubMatrix */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatGetSubMatrix\n");CHKERRQ(ierr);
  if (!rank) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,1,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,2,0,1,&is2);CHKERRQ(ierr);
  } else if (rank == 1) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,3,1,&is2);CHKERRQ(ierr);
  } else if (rank == 2 && n > 4) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,n-4,4,1,&is2);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2);CHKERRQ(ierr);
  }
  ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(B,is,is,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"first MatGetSubMatrix");CHKERRQ(ierr);

  ierr = MatGetSubMatrix(A,is,is,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"reuse MatGetSubMatrix");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);

  ierr = MatGetSubMatrix(A,is,is2,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(B,is,is2,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"second MatGetSubMatrix");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  /* Create an IS required by MatZeroRows(): just rank zero provides the rows to be eliminated */
  if (size > 1) {
    if (!rank) {
      PetscInt st = (m+1)/2;
      PetscInt len = PetscMin(m/2,PetscMax(m-(m+1)/2-1,0));
      ierr = ISCreateStride(PETSC_COMM_WORLD,len,st,1,&is);CHKERRQ(ierr);
    } else {
      ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is);CHKERRQ(ierr);
    }
  } else {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is);CHKERRQ(ierr);
  }

  if (m == n) { /* tests for square matrices only */
    /* test MatDiagonalSet */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalSet\n");CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,NULL,&x);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = MatDiagonalSet(A2,x,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatDiagonalSet(B2,x,INSERT_VALUES);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,"MatDiagonalSet");CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);

    /* test MatShift (MatShift_IS internally uses MatDiagonalSet_IS with ADD_VALUES) */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatShift\n");CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = MatShift(A2,1.0);CHKERRQ(ierr);
    ierr = MatShift(B2,1.0);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,"MatShift");CHKERRQ(ierr);
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);

    /* nonzero diag value is supported for square matrices only */
    ierr = TestMatZeroRows(A,B,is,diag);CHKERRQ(ierr);
  }
  ierr = TestMatZeroRows(A,B,is,0.0);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* test MatTranspose */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatTranspose\n");CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = MatTranspose(B,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"initial matrix MatTranspose");CHKERRQ(ierr);

  ierr = MatTranspose(A,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"reuse matrix (not in place) MatTranspose");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = MatTranspose(A2,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"reuse matrix (in place) MatTranspose");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  ierr = MatDuplicate(B,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,"reuse matrix (different type) MatTranspose");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  ierr = MatDestroy(&B2);CHKERRQ(ierr);

  /* free testing matrices */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "CheckMat"
PetscErrorCode CheckMat(Mat A, Mat B, const char* func)
{
  Mat            Bcheck;
  PetscReal      error;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatISGetMPIXAIJ(A,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
  if (B) { /* if B is present, subtract it */
    ierr = MatAXPY(Bcheck,-1.,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatNorm(Bcheck,NORM_INFINITY,&error);CHKERRQ(ierr);
  if (error > PETSC_SQRT_MACHINE_EPSILON) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR ON %s: %g",func,error);CHKERRQ(ierr);
  ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestMatZeroRows"
PetscErrorCode TestMatZeroRows(Mat A, Mat Afull, IS is, PetscScalar diag)
{
  Mat                    B,Bcheck,B2 = NULL;
  Vec                    x = NULL, b = NULL, b2 = NULL;
  ISLocalToGlobalMapping l2gr,l2gc;
  PetscReal              error;
  char                   diagstr[16];
  const PetscInt         *idxs;
  PetscInt               rst,ren,i,n,M,N,d;
  PetscMPIInt            rank;
  PetscBool              miss,square;
  PetscErrorCode         ierr;

  PetscFunctionBeginUser;
  if (diag == 0.) {
    ierr = PetscStrcpy(diagstr,"zero");CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(diagstr,"nonzero");CHKERRQ(ierr);
  }
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(A,&l2gr,&l2gc);CHKERRQ(ierr);
  /* tests MatDuplicate and MatCopy */
  if (diag == 0.) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  } else {
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  square = M == N ? PETSC_TRUE : PETSC_FALSE;
  if (square) {
    ierr = MatCreateVecs(B,&x,&b);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(b,l2gr);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(x,l2gc);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
    /* mimic b[is] = x[is] */
    ierr = VecDuplicate(b,&b2);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(b2,l2gr);CHKERRQ(ierr);
    ierr = VecCopy(b,b2);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      if (0 <= idxs[i] && idxs[i] < N) {
        ierr = VecSetValue(b2,idxs[i],diag,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecSetValue(x,idxs[i],1.,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(b2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b2);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
    /*  test ZeroRows on MATIS */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(B,is,diag,x,b);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRowsColumns (diag %s)\n",diagstr);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(B2,is,diag,NULL,NULL);CHKERRQ(ierr);
  } else {
    /*  test ZeroRows on MATIS */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(B,is,diag,NULL,NULL);CHKERRQ(ierr);
    b = b2 = x = NULL;
  }
  if (square) {
    ierr = VecAXPY(b2,-1.,b);CHKERRQ(ierr);
    ierr = VecNorm(b2,NORM_INFINITY,&error);CHKERRQ(ierr);
    if (error > PETSC_SQRT_MACHINE_EPSILON) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR IN ZEROROWS ON B %g (diag %s)",error,diagstr);CHKERRQ(ierr);
  }
  /* test MatMissingDiagonal */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatMissingDiagonal\n");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MatMissingDiagonal(B,&miss,&d);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&rst,&ren);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD, "[%d] [%D,%D) Missing %d, row %D (diag %s)\n",rank,rst,ren,(int)miss,d,diagstr);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&b2);CHKERRQ(ierr);

  /* check the result of ZeroRows with that from MPIAIJ routines
     assuming that MatISGetMPIXAIJ and MatZeroRows_MPIAIJ work fine */
  ierr = MatDuplicate(Afull,MAT_COPY_VALUES,&Bcheck);CHKERRQ(ierr);
  ierr = MatSetOption(Bcheck,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(Bcheck,is,diag,NULL,NULL);CHKERRQ(ierr);
  ierr = CheckMat(B,Bcheck,"Zerorows");CHKERRQ(ierr);
  ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  if (B2) { /* test MatZeroRowsColumns */
    ierr = MatDuplicate(Afull,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(B,is,diag,NULL,NULL);CHKERRQ(ierr);
    ierr = CheckMat(B2,B,"MatZeroRowsColumns");CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
