
static char help[] = "Test repeated LU factorizations. Used for checking memory leak\n\
  -m <size> : problem size\n\
  -mat_nonsym : use nonsymmetric matrix (default is symmetric)\n\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            C,F;                /* matrix */
  Vec            x,u,b;          /* approx solution, RHS, exact solution */
  PetscReal      norm;             /* norm of solution error */
  PetscScalar    v,none = -1.0;
  PetscInt       I,J,ldim,low,high,iglobal,Istart,Iend;
  PetscErrorCode ierr;
  PetscInt       i,j,m = 3,n = 2,its;
  PetscMPIInt    size,rank;
  PetscBool      mat_nonsymmetric;
  PetscInt       its_max;
  MatFactorInfo  factinfo;
  IS             perm,iperm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /*
     Set flag if we are doing a nonsymmetric problem; the default is symmetric.
  */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mat_nonsym",&mat_nonsymmetric));

  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));

  /*
     Set matrix entries matrix in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global row and columns of matrix entries.
  */
  for (I=Istart; I<Iend; I++) {
    v = -1.0; i = I/n; j = I - i*n;
    if (i>0)   {J = I - n; CHKERRQ(MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = I + n; CHKERRQ(MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = I - 1; CHKERRQ(MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = I + 1; CHKERRQ(MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES));
  }

  /*
     Make the matrix nonsymmetric if desired
  */
  if (mat_nonsymmetric) {
    for (I=Istart; I<Iend; I++) {
      v = -1.5; i = I/n;
      if (i>1)   {J = I-n-1; CHKERRQ(MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES));}
    }
  } else {
    CHKERRQ(MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE));
    CHKERRQ(MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  its_max=1000;
  /*
     Create parallel vectors.
      - When using VecSetSizes(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,m*n));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(b,&x));

  /*
     Currently, all parallel PETSc vectors are partitioned by
     contiguous chunks across the processors.  Determine which
     range of entries are locally owned.
  */
  CHKERRQ(VecGetOwnershipRange(x,&low,&high));

  /*
    Set elements within the exact solution vector in parallel.
     - Each processor needs to insert only elements that it owns
       locally (but any non-local entries will be sent to the
       appropriate processor during vector assembly).
     - Always specify global locations of vector entries.
  */
  CHKERRQ(VecGetLocalSize(x,&ldim));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100*rank);
    CHKERRQ(VecSetValues(u,1,&iglobal,&v,INSERT_VALUES));
  }

  /*
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition,
     by placing code between these two statements.
  */
  CHKERRQ(VecAssemblyBegin(u));
  CHKERRQ(VecAssemblyEnd(u));

  /* Compute right-hand-side vector */
  CHKERRQ(MatMult(C,u,b));

  CHKERRQ(MatGetOrdering(C,MATORDERINGNATURAL,&perm,&iperm));
  its_max = 2000;
  for (i=0; i<its_max; i++) {
    CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&F));
    CHKERRQ(MatLUFactorSymbolic(F,C,perm,iperm,&factinfo));
    for (j=0; j<1; j++) {
      CHKERRQ(MatLUFactorNumeric(F,C,&factinfo));
    }
    CHKERRQ(MatSolve(F,b,x));
    CHKERRQ(MatDestroy(&F));
  }
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));

  /* Check the error */
  CHKERRQ(VecAXPY(x,none,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %t\n",(double)norm));

  /* Free work space. */
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}
