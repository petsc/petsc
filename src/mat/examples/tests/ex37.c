/*$Id: ex37.c,v 1.10 1999/05/04 20:33:03 balay Exp bsmith $*/

static char help[] = "Tests MatCopy() and MatStore/RetrieveValues().\n\n"; 

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C,A; 
  int         i,  n = 10, midx[3], ierr,flg;
  Scalar      v[3];
  PetscTruth  flag;

  PetscInitialize(&argc,&args,(char *)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&C);CHKERRA(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A);CHKERRA(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for ( i=1; i<n-1; i++ ){
    midx[2] = i-1; midx[1] = i; midx[0] = i+1;
    ierr = MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES);CHKERRA(ierr);
  }
  i = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRA(ierr);
  i = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRA(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* test matrices with different nonzero patterns */
  ierr = MatCopy(C,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* Now C and A have the same nonzero pattern */
  ierr = MatSetOption(C,MAT_NO_NEW_NONZERO_LOCATIONS);CHKERRA(ierr);
  ierr = MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);CHKERRA(ierr);
  ierr = MatCopy(C,A,SAME_NONZERO_PATTERN);CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = MatEqual(A,C,&flag);CHKERRA(ierr);
  if (flag) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrices are equal\n");CHKERRA(ierr);
  } else {
    SETERRA(1,1,"Matrices are NOT equal");
  }

  ierr = MatStoreValues(A);CHKERRA(ierr);
  ierr = MatZeroEntries(A);CHKERRA(ierr);
  ierr = MatRetrieveValues(A);CHKERRA(ierr);
  ierr = MatEqual(A,C,&flag);CHKERRA(ierr);
  if (flag) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrices are equal\n");CHKERRA(ierr);
  } else {
    SETERRA(1,1,"Matrices are NOT equal");
  }

  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

 
