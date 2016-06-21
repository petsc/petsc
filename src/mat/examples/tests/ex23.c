
static char help[] = "Tests the use of MatZeroRows() for parallel MATIS matrices.\n\
This example also tests the use of MatView(), MatDuplicate() and MatISGetMPIXAIJ() for MATIS";

#include <petscmat.h>

extern PetscErrorCode TestMatZeroRows_Basic(Mat,Mat,IS,PetscScalar);
extern PetscErrorCode TestMatZeroRows_with_no_allocation(Mat,Mat,IS,PetscScalar);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat                    A,B,Bcheck;
  ISLocalToGlobalMapping map;
  IS                     is;
  PetscScalar            diag = 2.;
  PetscReal              error;
  PetscInt               n,i;
  PetscMPIInt            rank,size;
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n    = 2*size;

  /* create a square MATIS matrix in MATIS */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATIS);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&map);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(A,map,map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatISSetPreallocation(A,3,NULL,0,NULL);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    PetscScalar v[3] = { -1.,2.,-1.};
    PetscInt    cols[3] = {(i-1+n)%n,i,(i+1)%n};
 
    ierr = MatSetValuesLocal(A,1,&i,3,cols,v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* test MatView */
  ierr = MatView(A,NULL);CHKERRQ(ierr);

  /* Create a MPIAIJ matrix, same as A */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&map);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B,map,map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,3,NULL,3,NULL);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    PetscScalar v[3] = { -1.,2.,-1.};
    PetscInt    cols[3] = {(i-1+n)%n,i,(i+1)%n};
 
    ierr = MatSetValuesLocal(B,1,&i,3,cols,v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* test MatISGetMPIXAIJ */
  ierr = MatISGetMPIXAIJ(A,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
  ierr = MatAXPY(Bcheck,-1.,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(Bcheck,NORM_INFINITY,&error);CHKERRQ(ierr);
  if (error > PETSC_SQRT_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR ON ASSEMBLY %g\n",error);CHKERRQ(ierr); 
  }
  ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);

  /* Create an IS required by MatZeroRows(): just rank zero provides the rows to be eliminated */
  if (!rank) {
    ierr = ISCreateStride(PETSC_COMM_SELF,(size+1)/2,size/2,1,&is);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is);CHKERRQ(ierr);
  }
  ierr = TestMatZeroRows_Basic(A,B,is,0.0);CHKERRQ(ierr);
  ierr = TestMatZeroRows_Basic(A,B,is,diag);CHKERRQ(ierr);

  ierr = TestMatZeroRows_with_no_allocation(A,B,is,0.0);CHKERRQ(ierr);
  ierr = TestMatZeroRows_with_no_allocation(A,B,is,diag);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "TestMatZeroRows_Basic"
PetscErrorCode TestMatZeroRows_Basic(Mat A,Mat Afull, IS is,PetscScalar diag)
{
  Mat            B,Bcheck;
  PetscErrorCode ierr;
  PetscBool      keepnonzeropattern;
  PetscReal      error;

  /* Now copy A into B, and test it with MatZeroRows() */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-keep_nonzero_pattern",&keepnonzeropattern);CHKERRQ(ierr);
  if (keepnonzeropattern) {
    ierr = MatSetOption(B,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = MatZeroRowsIS(B,is,diag,0,0);CHKERRQ(ierr);
  /* check */
  ierr = MatISGetMPIXAIJ(B,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDuplicate(Afull,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(B,is,diag,0,0);CHKERRQ(ierr);
  ierr = MatAXPY(Bcheck,-1.,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(Bcheck,NORM_INFINITY,&error);CHKERRQ(ierr);
  if (error > PETSC_SQRT_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR ON ASSEMBLY IN ZEROROWS DUPLICATE %g\n",error);CHKERRQ(ierr); 
  }
  ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TestMatZeroRows_with_no_allocation"
PetscErrorCode TestMatZeroRows_with_no_allocation(Mat A,Mat Afull,IS is,PetscScalar diag)
{
  Mat            B,Bcheck;
  PetscReal      error;
  PetscErrorCode ierr;

  /* Now copy A into B, and test it with MatZeroRows() */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(B,is,diag,0,0);CHKERRQ(ierr);
  /* check */
  ierr = MatISGetMPIXAIJ(B,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDuplicate(Afull,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(B,is,diag,0,0);CHKERRQ(ierr);
  ierr = MatAXPY(Bcheck,-1.,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(Bcheck,NORM_INFINITY,&error);CHKERRQ(ierr);
  if (error > PETSC_SQRT_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR ON ASSEMBLY IN ZEROROWS NOALLOC %g\n",error);CHKERRQ(ierr); 
  }
  ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  return 0;
}
