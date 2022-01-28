
static char help[] = "Show MatShift BUG happening after copying a matrix with no rows on a process";
/*
   Contributed by: Eric Chamberland
*/
#include <petscmat.h>

/* DEFINE this to turn on/off the bug: */
#define SET_2nd_PROC_TO_HAVE_NO_LOCAL_LINES

int main(int argc,char **args)
{
  Mat               C;
  PetscInt          i,m = 3;
  PetscMPIInt       rank,size;
  PetscErrorCode    ierr;
  PetscScalar       v;
  Mat               lMatA;
  PetscInt          locallines;
  PetscInt          d_nnz[3] = {0,0,0};
  PetscInt          o_nnz[3] = {0,0,0};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  PetscAssertFalse(2 != size,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Relevant with 2 processes only");
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);

#ifdef SET_2nd_PROC_TO_HAVE_NO_LOCAL_LINES
  if (0 == rank) {
    locallines = m;
    d_nnz[0] = 1;
    d_nnz[1] = 1;
    d_nnz[2] = 1;
  } else {
   locallines = 0;
  }
#else
  if (0 == rank) {
    locallines = m-1;
    d_nnz[0] = 1;
    d_nnz[1] = 1;
  } else {
    locallines = 1;
    d_nnz[0] = 1;
  }
#endif

  ierr = MatSetSizes(C,locallines,locallines,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(C,1,d_nnz,o_nnz,NULL,NULL);CHKERRQ(ierr);

  v = 2;
  /* Assembly on the diagonal: */
  for (i=0; i<m; i++) {
     ierr = MatSetValues(C,1,&i,1,&i,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatConvert(C,MATSAME, MAT_INITIAL_MATRIX, &lMatA);CHKERRQ(ierr);
  ierr = MatView(lMatA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatShift(lMatA,-1.0);CHKERRQ(ierr);

  ierr = MatDestroy(&lMatA);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
