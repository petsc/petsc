/*$Id: vcreatea.c,v 1.16 2000/05/04 16:24:23 bsmith Exp bsmith $*/

#include "petsc.h"  /*I     "petsc.h"   I*/

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stdout_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ /*<a name="VIEWER_STDOUT_"></a>*/"VIEWER_STDOUT_"  
/*@C
   VIEWER_STDOUT_ - Creates a ASCII viewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the viewer

   Level: beginner

   Notes: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_STDOUT_(comm));

.seealso: VIEWER_DRAW_(), ViewerASCIIOpen(), VIEWER_STDERR_, VIEWER_STDOUT_WORLD,
          VIEWER_STDOUT_SELF

@*/
Viewer VIEWER_STDOUT_(MPI_Comm comm)
{
  int        ierr;
  PetscTruth flg;
  Viewer     viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stdout_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Stdout_keyval,(void **)&viewer,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flg) { /* viewer not yet created */
    ierr = ViewerASCIIOpen(comm,"stdout",&viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Stdout_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stderr_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ /*<a name="VIEWER_STDERR_"></a>*/"VIEWER_STDERR_" 
/*@C
   VIEWER_STDERR_ - Creates a ASCII viewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the viewer

   Level: beginner

   Note: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_STDERR_(comm));

.seealso: VIEWER_DRAW_, ViewerASCIIOpen(), VIEWER_STDOUT_, VIEWER_STDOUT_WORLD,
          VIEWER_STDOUT_SELF, VIEWER_STDERR_WORLD, VIEWER_STDERR_SELF
@*/
Viewer VIEWER_STDERR_(MPI_Comm comm)
{
  int        ierr;
  PetscTruth flg;
  Viewer     viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stderr_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Stderr_keyval,(void **)&viewer,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flg) { /* viewer not yet created */
    ierr = ViewerASCIIOpen(comm,"stderr",&viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Stderr_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="ViewerASCIIOpen"></a>*/"ViewerASCIIOpen" 
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

  Concepts: ViewerASCII^creating
  Concepts: printf
  Concepts: printing
  Concepts: accessing remote file
  Concepts: remote file

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerBinaryOpen(),
          ViewerASCIIGetPointer(), ViewerSetFormat(), VIEWER_STDOUT_, VIEWER_STDERR_,
          VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF, 
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


