
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = 2*size;

  /*
     Set flag if we are doing a nonsymmetric problem; the default is symmetric.
  */
  ierr = PetscOptionsHasName(NULL,NULL,"-mat_nonsym",&mat_nonsymmetric);CHKERRQ(ierr);

  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);

  /*
     Set matrix entries matrix in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global row and columns of matrix entries.
  */
  for (I=Istart; I<Iend; I++) {
    v = -1.0; i = I/n; j = I - i*n;
    if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  /*
     Make the matrix nonsymmetric if desired
  */
  if (mat_nonsymmetric) {
    for (I=Istart; I<Iend; I++) {
      v = -1.5; i = I/n;
      if (i>1)   {J = I-n-1; ierr = MatSetValues(C,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    }
  } else {
    ierr = MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  its_max=1000;
  /*
     Create parallel vectors.
      - When using VecSetSizes(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /*
     Currently, all parallel PETSc vectors are partitioned by
     contiguous chunks across the processors.  Determine which
     range of entries are locally owned.
  */
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);

  /*
    Set elements within the exact solution vector in parallel.
     - Each processor needs to insert only elements that it owns
       locally (but any non-local entries will be sent to the
       appropriate processor during vector assembly).
     - Always specify global locations of vector entries.
  */
  ierr = VecGetLocalSize(x,&ldim);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100*rank);
    ierr    = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition,
     by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  /* Compute right-hand-side vector */
  ierr = MatMult(C,u,b);CHKERRQ(ierr);

  ierr    = MatGetOrdering(C,MATORDERINGNATURAL,&perm,&iperm);CHKERRQ(ierr);
  its_max = 2000;
  for (i=0; i<its_max; i++) {
    ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(F,C,perm,iperm,&factinfo);CHKERRQ(ierr);
    for (j=0; j<1; j++) {
      ierr = MatLUFactorNumeric(F,C,&factinfo);CHKERRQ(ierr);
    }
    ierr = MatSolve(F,b,x);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);

  /* Check the error */
  ierr = VecAXPY(x,none,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %t\n",(double)norm);CHKERRQ(ierr);

  /* Free work space. */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

