#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vcreatea.c,v 1.4 1999/01/31 16:04:38 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I     "petsc.h"   I*/
#include "pinclude/petscfix.h"
#include <stdarg.h>

Viewer VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF, VIEWER_STDOUT_WORLD, VIEWER_STDERR_WORLD;

/*
      This is called by PETScInitialize() to create the 
   default PETSc viewers.
*/
#undef __FUNC__  
#define __FUNC__ "ViewerInitializeASCII_Private"
int ViewerInitializeASCII_Private(void)
{
  int ierr;
  PetscFunctionBegin;
  ierr = ViewerASCIIOpen(PETSC_COMM_SELF,"stderr",&VIEWER_STDERR_SELF);CHKERRQ(ierr);
  ierr = ViewerASCIIOpen(PETSC_COMM_SELF,"stdout",&VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ViewerASCIIOpen(PETSC_COMM_WORLD,"stdout",&VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIIOpen(PETSC_COMM_WORLD,"stderr",&VIEWER_STDERR_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      This is called in PetscFinalize() to destroy all
   traces of the default viewers.
*/
#undef __FUNC__  
#define __FUNC__ "ViewerDestroyASCII_Private"
int ViewerDestroyASCII_Private(void)
{
  int ierr;

  PetscFunctionBegin;
  ierr = ViewerDestroy(VIEWER_STDERR_SELF);CHKERRQ(ierr);
  ierr = ViewerDestroy(VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ViewerDestroy(VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerDestroy(VIEWER_STDERR_WORLD);CHKERRQ(ierr);
  ierr = VIEWER_STDOUT_Destroy(PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = VIEWER_STDERR_Destroy(PETSC_COMM_SELF);CHKERRQ(ierr);
  ierr = VIEWER_STDOUT_Destroy(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = VIEWER_STDERR_Destroy(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stdout_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_STDOUT_" 
/*@C
   VIEWER_STDOUT_ - Creates a window viewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the window viewer

   Level: beginner

   Notes: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_STDOUT_(comm));

.seealso: VIEWER_DRAW_(), ViewerASCIIOpen()

@*/
Viewer VIEWER_STDOUT_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stdout_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stdout_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerASCIIOpen(comm,"stdout",&viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Stdout_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_STDOUT_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stdout_keyval, (void **)&viewer, &flag ); CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Stdout_keyval); CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stderr_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_STDERR_" 
/*@C
   VIEWER_STDERR_ - Creates a window viewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the window viewer

   Level: beginner

   Note: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_STDERR_(comm));

.seealso: VIEWER_DRAW_, ViewerASCIIOpen(), 
@*/
Viewer VIEWER_STDERR_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stderr_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stderr_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerASCIIOpen(comm,"stderr",&viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Stderr_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_STDERR_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stderr_keyval, (void **)&viewer, &flag );CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Stderr_keyval);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIOpen"
/*@C
   ViewerASCIIOpen - Opens an ASCII file as a viewer.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator
-  name - the file name

   Output Parameter:
.  lab - the viewer to use with the specified file

   Level: beginner

   Notes:
   This viewer can be destroyed with ViewerDestroy().

   If a multiprocessor communicator is used (such as PETSC_COMM_WORLD), 
   then only the first processor in the group opens the file.  All other 
   processors send their data to the first processor to print. 

   Each processor can instead write its own independent output by
   specifying the communicator PETSC_COMM_SELF.

   As shown below, ViewerASCIIOpen() is useful in conjunction with 
   MatView() and VecView()
.vb
     ViewerASCIIOpen(PETSC_COMM_WORLD,"mat.output",&viewer);
     MatView(matrix,viewer);
.ve

.keywords: Viewer, file, open

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerBinaryOpen(),
          ViewerASCIIGetPointer(), ViewerSetFormat()
@*/
int ViewerASCIIOpen(MPI_Comm comm,const char name[],Viewer *lab)
{
  int ierr;

  PetscFunctionBegin;
  ierr = ViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = ViewerSetType(*lab,ASCII_VIEWER);CHKERRQ(ierr);
  if (name) {
    ierr = ViewerSetFilename(*lab,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


