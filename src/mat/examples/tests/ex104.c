static char help[] = "Test MatMatMult(), MatTranspose(), MatTransposeMatMult() for Dense and Elemental matrices.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex104 -mat_type elemental
*/

#include <petscmat.h>

/* Test C*x = A^T*B*x for n random vector x */
PetscErrorCode MatTransposeMatMultEqual(Mat A,Mat B,Mat C,PetscInt n,PetscBool  *flg)
{
  PetscErrorCode ierr;
  Vec            x,s1,s2,s3;
  PetscRandom    rctx;
  PetscReal      r1,r2,tol=100.*PETSC_MACHINE_EPSILON;
  PetscInt       am,an,bm,bn,cm,cn,k;
  PetscScalar    none = -1.0;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&cm,&cn);CHKERRQ(ierr);
  if (am != bm || an != cm || bn != cn) SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A, B, C local dim %D %D %D %D",am,an,bm,bn,cm, cn);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)C),&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,&s1);CHKERRQ(ierr);
  ierr = MatCreateVecs(C,NULL,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(s2,&s3);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMult(B,x,s1);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,s1,s2);CHKERRQ(ierr);
    ierr = MatMult(C,x,s3);CHKERRQ(ierr);

    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol) {
      ierr = VecNorm(s3,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s3);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1  /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      ierr = PetscInfo2(A,"Error: %D-th MatTransposeMatMult() %g\n",k,(double)r1);CHKERRQ(ierr);
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = VecDestroy(&s3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscInt       i,M=10,N=5,j,nrows,ncols;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      equal,iselemental;
  PetscReal      fill = 1.0;
  IS             isrows,iscols;
  const PetscInt *rows,*cols;
  PetscScalar    *v,rval;
  PetscBool      Test_MatMatMult=PETSC_TRUE;
  PetscMPIInt    size;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* Set local matrix entries */
  ierr = MatGetOwnershipIS(A,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr         = PetscRandomGetValue(r,&rval);CHKERRQ(ierr);
      v[i*ncols+j] = rval; 
    }
  }
  ierr = MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);

  /* Test MatMatMult() */
  if (Test_MatMatMult && size == 1) { 
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr); /* B = A^T */
    ierr = MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); /* C = B*A = A^T*A */

    ierr = MatMatMultSymbolic(B,A,fill,&D);CHKERRQ(ierr); /* D = B*A = A^T*A */
    for (i=0; i<2; i++) {
      /* Repeat the numeric product to test reuse of the previous symbolic product */
      ierr = MatMatMultNumeric(B,A,D);CHKERRQ(ierr);
    }
    ierr = MatMultEqual(C,D,10,&equal);CHKERRQ(ierr);
    if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
  }

  /* Test MatTransposeMatMult() */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATELEMENTAL,&iselemental);CHKERRQ(ierr);
  if (!iselemental) {
    ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr); /* D = A^T*A */
    ierr = MatTransposeMatMult(A,A,MAT_REUSE_MATRIX,fill,&D);CHKERRQ(ierr);
    /* ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = MatTransposeMatMultEqual(A,A,D,10,&equal);CHKERRQ(ierr);
    if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"D*x != A^T*A*x");
    ierr = MatDestroy(&D);CHKERRQ(ierr);

    /* Test D = A^T* C * A, where C is in AIJ format */
    PetscInt    am,an,rstart,rend;
    PetscScalar v = 1.0;
    ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,am,am);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(C,am,am,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = MatSetFromOptions(C);CHKERRQ(ierr);
    ierr = MatSetUp(C);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(C,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* B = C * A; D = A^T * B */
    ierr = MatMatMult(C,A,MAT_INITIAL_MATRIX,1.0,&B);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(A,B,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr);
    ierr = MatTransposeMatMultEqual(A,B,D,10,&equal);CHKERRQ(ierr);
    if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"D*x != A^T*A");

    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}
