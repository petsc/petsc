
#include "src/sys/src/viewer/impls/socket/socket.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSocketPutScalar" 
/*@C
   PetscViewerSocketPutScalar - Passes a Scalar array to a Socket PetscViewer.

  Input Parameters:
+  viewer - obtained from PetscViewerSocketOpen()
.  m - number of rows of array
.  m - number of columns of array
-  array - the array stored in column ordering (matrix or vector data) 

    Level: advanced

   Notes:
   Most users should not call this routine, but instead should employ
   either
.vb
     MatView(Mat matrix,PetscViewer viewer)
              or
     VecView(Vec vector,PetscViewer viewer)
.ve

   Concepts: Matlab^sending data
   Concepts: sockets^sending data

.seealso: PetscViewerSocketOpen(), MatView(), VecView(), PetscViewerSocketPutReal(), PetscViewerSocketPutScalar(),
      PETSC_VIEWER_SOCKET_, PETSC_VIEWER_SOCKET_WORLD, PETSC_VIEWER_SOCKET_SELF
@*/
int PetscViewerSocketPutScalar(PetscViewer viewer,int m,int n,PetscScalar *array)
{
  PetscViewer_Socket *vmatlab = (PetscViewer_Socket*)viewer->data;
  int                ierr,t = vmatlab->port,type = DENSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0);CHKERRQ(ierr); 
#if !defined(PETSC_USE_COMPLEX)
  value = 0;
#else
  value = 1;
#endif
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,PETSC_SCALAR,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSocketPutReal" 
/*@C
   PetscViewerSocketPutReal - Passes a double (or single) precision array to 
   a Matlab PetscViewer.

  Input Parameters:
+  viewer - obtained from PetscViewerSocketOpen()
.  m - number of rows of array
.  m - number of columns of array
-  array - the array stored in column ordering (matrix or vector data) 

    Level: advanced

   Notes:
   Most users should not call this routine, but instead should employ
   either
.vb
     MatView(Mat matrix,PetscViewer viewer)
              or
     VecView(Vec vector,PetscViewer viewer)
.ve

   Concepts: Matlab^sending data
   Concepts: sockets^sending data

.seealso: PetscViewerSocketOpen(), MatView(), VecView(), PetscViewerSocketPutInt(), PetscViewerSocketPutReal(),
          PETSC_VIEWER_SOCKET_, PETSC_VIEWER_SOCKET_WORLD, PETSC_VIEWER_SOCKET_SELF
@*/
int PetscViewerSocketPutReal(PetscViewer viewer,int m,int n,PetscReal *array)
{
  PetscViewer_Socket *vmatlab = (PetscViewer_Socket*)viewer->data;
  int                ierr,t = vmatlab->port,type = DENSEREAL,value;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&n,1,PETSC_INT,0);CHKERRQ(ierr); 
  value = 0;
  ierr = PetscBinaryWrite(t,&value,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m*n,PETSC_REAL,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSocketPutInt" 
/*@C
   PetscViewerSocketPutInt - Passes an integer array to a Socket PetscViewer.

   Input Parameters:
+  viewer - obtained from PetscViewerSocketOpen()
.  m - number of rows of array
-  array - the array stored in column ordering (matrix or vector data) 

    Level: advanced

   Notes:
   Most users should not call this routine, but instead should employ either
.vb
     MatView(Mat matrix,PetscViewer viewer)
              or
     VecView(Vec vector,PetscViewer viewer)
.ve

   Concepts: Matlab^sending data
   Concepts: sockets^sending data

.seealso: PetscViewerSocketOpen(), MatView(), VecView(), PetscViewerSocketPutScalar(), PetscViewerSocketPutReal(),
       PETSC_VIEWER_SOCKET_, PETSC_VIEWER_SOCKET_WORLD, PETSC_VIEWER_SOCKET_SELF
@*/
int PetscViewerSocketPutInt(PetscViewer viewer,int m,int *array)
{
  PetscViewer_Socket *vmatlab = (PetscViewer_Socket*)viewer->data;
  int                ierr,t = vmatlab->port,type = DENSEINT;

  PetscFunctionBegin;
  ierr = PetscBinaryWrite(t,&type,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,&m,1,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscBinaryWrite(t,array,m,PETSC_INT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

