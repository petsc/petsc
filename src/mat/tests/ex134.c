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
  MatSolverType     stype = MATSOLVERPETSC;
  PetscRandom       rdm;
  Vec               b,x,y;
  PetscInt          i,j;
  PetscReal         norm2,tol=100*PETSC_SMALL;
  PetscBool         issbaij;
#endif
  PetscViewer       viewer;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(comm,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4*bs,4*bs));
  CHKERRQ(MatSetType(A,mtype));
  CHKERRQ(MatMPIBAIJSetPreallocation(A,bs,2,NULL,2,NULL));
  CHKERRQ(MatMPISBAIJSetPreallocation(A,bs,2,NULL,2,NULL));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  /* All processes contribute a global matrix */
  CHKERRQ(MatSetValuesBlocked(A,4,rc,4,rc,vals,ADD_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscPrintf(comm,"Matrix %s(%" PetscInt_FMT ")\n",mtype,bs));
  CHKERRQ(PetscViewerASCIIGetStdout(comm,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(MatView(A,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(MatView(A,viewer));
#if defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_MKL_CPARDISO)
  CHKERRQ(PetscStrcmp(mtype,MATMPISBAIJ,&issbaij));
  if (!issbaij) {
    CHKERRQ(MatShift(A,10));
  }
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(MatCreateVecs(A,&x,&y));
  CHKERRQ(VecDuplicate(x,&b));
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
      CHKERRQ(MatGetFactor(A,stype,MAT_FACTOR_CHOLESKY,&F));
      CHKERRQ(MatCholeskyFactorSymbolic(F,A,NULL,NULL));
      CHKERRQ(MatCholeskyFactorNumeric(F,A,NULL));
    } else {
      CHKERRQ(MatGetFactor(A,stype,MAT_FACTOR_LU,&F));
      CHKERRQ(MatLUFactorSymbolic(F,A,NULL,NULL,NULL));
      CHKERRQ(MatLUFactorNumeric(F,A,NULL));
    }
    for (i=0; i<10; i++) {
      CHKERRQ(VecSetRandom(b,rdm));
      CHKERRQ(MatSolve(F,b,y));
      /* Check the error */
      CHKERRQ(MatMult(A,y,x));
      CHKERRQ(VecAXPY(x,-1.0,b));
      CHKERRQ(VecNorm(x,NORM_2,&norm2));
      if (norm2>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error:MatSolve(), norm2: %g\n",(double)norm2));
      }
    }
    CHKERRQ(MatDestroy(&F));
  }
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscRandomDestroy(&rdm));
#endif
  CHKERRQ(MatDestroy(&A));
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  MPI_Comm       comm;
  PetscMPIInt    size;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  PetscCheckFalse(size != 2,comm,PETSC_ERR_USER,"This example must be run with exactly two processes");
  CHKERRQ(Assemble(comm,2,MATMPIBAIJ));
  CHKERRQ(Assemble(comm,2,MATMPISBAIJ));
  CHKERRQ(Assemble(comm,1,MATMPIBAIJ));
  CHKERRQ(Assemble(comm,1,MATMPISBAIJ));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args: -mat_ignore_lower_triangular
      filter: sed -e "s~mem [0-9]*~mem~g"

TEST*/
