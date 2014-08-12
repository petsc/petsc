static char help[] = "Test MatSetValues() by converting MATDENSE to MATELEMENTAL. \n\
Modified from the code contributed by Yaning Liu @lbl.gov \n\n";
/*
 Example:
   mpiexec -n <np> ./ex103 -mat_view
   mpiexec -n <np> ./ex103 -mat_type elemental -mat_view
*/
    
#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv)
{
  Mat            mat_dense,mat_elemental;
  PetscInt       i,j,M=10,N=5,nrows,ncols;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  IS             isrows,iscols;
  const PetscInt *rows,*cols;
  PetscScalar    *v;
  MatType        type;
  PetscBool      isDense;

  PetscInitialize(&argc, &argv, (char*)0, help);
#if !defined(PETSC_HAVE_ELEMENTAL)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires ELEMENTAL");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Creat a matrix */
  ierr = PetscOptionsGetInt(NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &mat_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_dense,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(mat_dense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat_dense);CHKERRQ(ierr);
  ierr = MatSetUp(mat_dense);CHKERRQ(ierr);

  /* Set local matrix entries */
  ierr = MatGetOwnershipIS(mat_dense,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);

  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscScalar)rank; 
    }
  }
  ierr = MatSetValues(mat_dense,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat_dense, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_dense, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%D] local nrows %D, ncols %D\n",rank,nrows,ncols);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  /* Test MatSetValues() by converting mat_dense to mat_elemental */
  ierr = MatGetType(mat_dense,&type);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectTypeCompare((PetscObject)mat_dense,MATSEQDENSE,&isDense);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)mat_dense,MATMPIDENSE,&isDense);CHKERRQ(ierr);
  }
  if (isDense) {
    //ierr = MatConvert(mat_dense, MATELEMENTAL, MAT_INITIAL_MATRIX, &mat_elemental);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &mat_elemental);CHKERRQ(ierr);
    ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(mat_elemental,MATELEMENTAL);CHKERRQ(ierr);
    ierr = MatSetFromOptions(mat_elemental);CHKERRQ(ierr);
    ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);
    for (i=0; i<nrows; i++) {
      for (j=0; j<ncols; j++) {
        /* PETSc-Elemental interaface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
        ierr = MatSetValues(mat_elemental,1,&rows[i],1,&cols[j],v+i*ncols+j,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDestroy(&mat_elemental);CHKERRQ(ierr); 
  }

  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_dense);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
