
static char help[] = "Tests Elemental interface.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            Cdense,B,C,Ct;
  Vec            d;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscErrorCode ierr;
  PetscScalar    *v;
  PetscMPIInt    rank,size;
  PetscReal      Cnorm;
  PetscBool      flg,mats_view=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  n    = m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(C,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipIS(C,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  PetscRandom rand;
  PetscScalar rval;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr         = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      v[i*ncols+j] = rval;
    }
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
#else
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(10000*rank+100*rows[i]+cols[j]);
    }
  }
#endif
  ierr = MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);

  /* Test MatView(), MatDuplicate() and out-of-place MatConvert() */
  ierr = MatDuplicate(C,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Duplicated C:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdense);CHKERRQ(ierr);
  ierr = MatMultEqual(C,Cdense,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Cdense != C. MatConvert() fails");

  /* Test MatNorm() */
  ierr = MatNorm(C,NORM_1,&Cnorm);CHKERRQ(ierr);

  /* Test MatTranspose(), MatZeroEntries() and MatGetDiagonal() */
  ierr = MatTranspose(C,MAT_INITIAL_MATRIX,&Ct);CHKERRQ(ierr);
  ierr = MatConjugate(Ct);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"C's Transpose Conjugate:\n");CHKERRQ(ierr);
    ierr = MatView(Ct,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(Ct);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&d);CHKERRQ(ierr);
  ierr = VecSetSizes(d,m>n ? n : m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(d);CHKERRQ(ierr);
  ierr = MatGetDiagonal(C,d);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Diagonal of C:\n");CHKERRQ(ierr);
    ierr = VecView(d,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (m>n) {
    ierr = MatDiagonalScale(C,NULL,d);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalScale(C,d,NULL);CHKERRQ(ierr);
  }
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Diagonal Scaled C:\n");CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatAXPY(), MatAYPX() and in-place MatConvert() */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipIS(B,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
    }
  }
  ierr = MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAXPY(B,2.5,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAYPX(B,3.75,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"B after MatAXPY and MatAYPX:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Test MatMatTransposeMult(): B = C*C^T */
  ierr = MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"C MatMatTransposeMult C:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&Cdense);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = VecDestroy(&d);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}




/*TEST

   test:
      nsize: 2
      args: -m 3 -n 2
      requires: elemental

   test:
      suffix: 2
      nsize: 6
      args: -m 2 -n 3
      requires: elemental

TEST*/
