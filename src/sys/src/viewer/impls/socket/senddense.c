#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: senddense.c,v 1.26 1998/04/04 00:04:06 balay Exp curfman $";
#endif

#include "src/viewer/impls/matlab/matlab.h"

#undef __FUNC__  
#define __FUNC__ "ViewerMatlabPutScalar_Private"
/*
   ViewerMatlabPutScalar_Private - Passes an Scalar array to a Matlab viewer.

  Input Parameters:
.  viewer - obtained from ViewerMatlabOpen()
.  m, n - number of rows and columns of array
.  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ
   either
.vb
     MatView(Mat matrix,Viewer viewer)
              or
     VecView(Vec vector,Viewer viewer)
.ve

.keywords: Viewer, Matlab, put, dense, array, vector

.seealso: ViewerMatlabOpen(), MatView(), VecView()
*/
int ViewerMatlabPutScalar_Private(Viewer viewer,int m,int n,Scalar *array)
{
  int ierr,t = viewer->port,type = DENSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0); CHKERRQ(ierr); 
#if !defined(USE_PETSC_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,PETSC_SCALAR,0);
  PetscFunctionReturn(0);
}

/*
   ViewerMatlabPutDouble_Private - Passes an integer array to a Matlab viewer.

  Input Parameters:
.  viewer - obtained from ViewerMatlabOpen()
.  m, n - number of rows and columns of array
.  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ
   either
$     MatView(Mat matrix,Viewer viewer)
$
$              or
$
$     VecView(Vec vector,Viewer viewer)

.keywords: Viewer, Matlab, put, dense, array, vector

.seealso: ViewerMatlabOpen(), MatView(), VecView()
*/
int ViewerMatlabPutDouble_Private(Viewer viewer,int m,int n,double *array)
{
  int ierr,t = viewer->port,type = DENSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0); CHKERRQ(ierr); 
  value = 0;
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,PETSC_DOUBLE,0);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "ViewerMatlabPutInt_Private"
/*
   ViewerMatlabPutInt_Private - Passes an integer array to a Matlab viewer.

   Input Parameters:
+  viewer - obtained from ViewerMatlabOpen()
.  m - number of rows of array
-  array - the array stored in Fortran 77 style (matrix or vector data) 

   Notes:
   Most users should not call this routine, but instead should employ either
.vb
     MatView(Mat matrix,Viewer viewer)
              or
     VecView(Vec vector,Viewer viewer)
.ve

.keywords: Viewer, Matlab, put, dense, array, vector

.seealso: ViewerMatlabOpen(), MatView(), VecView()
*/
int ViewerMatlabPutInt_Private(Viewer viewer,int m,int *array)
{
  int ierr,t = viewer->port,type = DENSEINT;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0); CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m,PETSC_INT,0);
  PetscFunctionReturn(0);
}

