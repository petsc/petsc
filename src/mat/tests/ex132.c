
static char help[] = "Test MatAXPY(), and illustrate how to reduce number of mallocs used during MatSetValues() calls \n\
                      Matrix C is copied from ~petsc/src/ksp/ksp/tutorials/ex5.c\n\n";
/*
  Example: ./ex132 -mat_view ascii::ascii_info
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,C1,C2;           /* matrix */
  PetscScalar    v;
  PetscInt       Ii,J,Istart,Iend;
  PetscErrorCode ierr;
  PetscInt       i,j,m = 3,n = 2;
  PetscMPIInt    size,rank;
  PetscBool      mat_nonsymmetric = PETSC_FALSE;
  MatInfo        info;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = 2*size;

  /* Set flag if we are doing a nonsymmetric problem; the default is symmetric. */
  ierr = PetscOptionsGetBool(NULL,NULL,"-mat_nonsym",&mat_nonsymmetric,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(C,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  /* Make the matrix nonsymmetric if desired */
  if (mat_nonsymmetric) {
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.5; i = Ii/n;
      if (i>1)   {J = Ii-n-1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    }
  } else {
    ierr = MatSetOption(C,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(C,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* First, create C1 = 2.0*C1 + C, C1 has less non-zeros than C */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\ncreate C1 = 2.0*C1 + C, C1 has less non-zeros than C \n");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C1);CHKERRQ(ierr);
  ierr = MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C1);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C1,1,NULL);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    ierr = MatSetValues(C1,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C1,2.0,C,DIFFERENT_NONZERO_PATTERN)...\n");CHKERRQ(ierr);
  ierr = MatAXPY(C1,2.0,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatGetInfo(C1,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," C1: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded);CHKERRQ(ierr);

  /* Secondly, create C2 = 2.0*C2 + C, C2 has non-zero pattern of C2 + C */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\ncreate C2 = 2.0*C2 + C, C2 has non-zero pattern of C2 + C \n");CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C2);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) {
    v    = 1.0;
    ierr = MatSetValues(C2,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," MatAXPY(C2,2.0,C,SUBSET_NONZERO_PATTERN)...\n");CHKERRQ(ierr);
  ierr = MatAXPY(C2,2.0,C,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatGetInfo(C2,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," C2: nz_allocated = %g; nz_used = %g; nz_unneeded = %g\n",info.nz_allocated,info.nz_used, info.nz_unneeded);CHKERRQ(ierr);

  ierr = MatDestroy(&C1);CHKERRQ(ierr);
  ierr = MatDestroy(&C2);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
