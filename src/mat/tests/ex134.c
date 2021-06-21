static const char help[] = "Test parallel assembly of SBAIJ matrices\n\n";

#include <petscmat.h>

PetscErrorCode Assemble(MPI_Comm comm,PetscInt bs,MatType mtype)
{
  const PetscInt    rc[]   = {0,1,2,3};
  const PetscScalar vals[] = {100, 2, 3, 4, 5, 600, 7, 8,
                              9,100,11,1200,13,14,15,1600,
                              17,18,19,20,21,22,23,24,
                              25,26,27,2800,29,30,31,32,
                              33,34,35,36,37,38,39,40,
                              41,42,43,44,45,46,47,48,
                              49,50,51,52,53,54,55,56,
                              57,58,49,60,61,62,63,64};
  Mat               A;
#if defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_MKL_CPARDISO)
  Mat               F;
  MatSolverType     stype;
  PetscRandom       rdm;
  Vec               b,x,y;
  PetscInt          i,j;
  PetscReal         norm2,tol=100*PETSC_SMALL;
  PetscBool         issbaij;
#endif
  PetscViewer       viewer;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4*bs,4*bs);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A,bs,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,bs,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* All processes contribute a global matrix */
  ierr = MatSetValuesBlocked(A,4,rc,4,rc,vals,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Matrix %s(%D)\n",mtype,bs);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_MKL_CPARDISO)
  ierr = PetscStrcmp(mtype,MATMPISBAIJ,&issbaij);CHKERRQ(ierr);
  if (!issbaij) {
    ierr = MatShift(A,10);CHKERRQ(ierr);
  }
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  for (j=0; j<2; j++) {
#if defined(PETSC_HAVE_MUMPS)
    if (j==0) stype = MATSOLVERMUMPS;
#else
    if (j==0) continue;
#endif
#if defined(PETSC_HAVE_MKL_CPARDISO)
    if (j==1) stype = MATSOLVERMKL_CPARDISO;
#else
    if (j==1) continue;
#endif
    if (issbaij) {
      ierr = MatGetFactor(A,stype,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
      ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatGetFactor(A,stype,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    }
    for (i=0; i<10; i++) {
      ierr = VecSetRandom(b,rdm);CHKERRQ(ierr);
      ierr = MatSolve(F,b,y);CHKERRQ(ierr);
      /* Check the error */
      ierr = MatMult(A,y,x);CHKERRQ(ierr);
      ierr = VecAXPY(x,-1.0,b);CHKERRQ(ierr);
      ierr = VecNorm(x,NORM_2,&norm2);CHKERRQ(ierr);
      if (norm2>tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error:MatSolve(), norm2: %g\n",(double)norm2);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&F);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
#endif
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size != 2) SETERRQ(comm,PETSC_ERR_USER,"This example must be run with exactly two processes");
  ierr = Assemble(comm,2,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = Assemble(comm,2,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = Assemble(comm,1,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = Assemble(comm,1,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -mat_ignore_lower_triangular
      filter: sed -e "s~mem [0-9]*~mem~g"

TEST*/
