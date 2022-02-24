
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheckFalse(2 != size,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"Relevant with 2 processes only");
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));

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

  CHKERRQ(MatSetSizes(C,locallines,locallines,m,m));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatXAIJSetPreallocation(C,1,d_nnz,o_nnz,NULL,NULL));

  v = 2;
  /* Assembly on the diagonal: */
  for (i=0; i<m; i++) {
     CHKERRQ(MatSetValues(C,1,&i,1,&i,&v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  CHKERRQ(MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatConvert(C,MATSAME, MAT_INITIAL_MATRIX, &lMatA));
  CHKERRQ(MatView(lMatA,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatShift(lMatA,-1.0));

  CHKERRQ(MatDestroy(&lMatA));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
