#define PETSC_DLL

#include "../src/sys/viewer/impls/ascii/asciiimpl.h"  /*I     "petscsys.h"   I*/

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stdout_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static PetscMPIInt Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerASCIIGetStdout"
/*@C
   PetscViewerASCIIGetStdout - Creates a ASCII PetscViewer shared by all processors 
                    in a communicator. Error returning version of PETSC_VIEWER_STDOUT_()

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the PetscViewer

   Level: beginner

   Notes: 
     This should be used in all PETSc source code instead of PETSC_VIEWER_STDOUT_()

.seealso: PETSC_VIEWER_DRAW_(), PetscViewerASCIIOpen(), PETSC_VIEWER_STDERR_, PETSC_VIEWER_STDOUT_WORLD,
          PETSC_VIEWER_STDOUT_SELF

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIGetStdout(MPI_Comm comm,PetscViewer *viewer)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stdout_keyval,0);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Stdout_keyval,(void **)viewer,(PetscMPIInt*)&flg);CHKERRQ(ierr);
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscViewerASCIIOpen(comm,"stdout",viewer);CHKERRQ(ierr);
    ierr = PetscObjectRegisterDestroy((PetscObject)*viewer);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Stdout_keyval,(void*)*viewer);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_STDOUT_"  
/*@C
   PETSC_VIEWER_STDOUT_ - Creates a ASCII PetscViewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the PetscViewer

   Level: beginner

   Notes: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,PETSC_VIEWER_STDOUT_(comm));

.seealso: PETSC_VIEWER_DRAW_(), PetscViewerASCIIOpen(), PETSC_VIEWER_STDERR_, PETSC_VIEWER_STDOUT_WORLD,
          PETSC_VIEWER_STDOUT_SELF

@*/
PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_STDOUT_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStdout(comm,&viewer);
  if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_STDOUT_",__FILE__,__SDIR__,1,1," "); PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stderr_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static PetscMPIInt Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerASCIIGetStderr"
/*@C
   PetscViewerASCIIGetStderr - Creates a ASCII PetscViewer shared by all processors 
                    in a communicator. Error returning version of PETSC_VIEWER_STDERR_()

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the PetscViewer

   Level: beginner

   Notes: 
     This should be used in all PETSc source code instead of PETSC_VIEWER_STDERR_()

.seealso: PETSC_VIEWER_DRAW_(), PetscViewerASCIIOpen(), PETSC_VIEWER_STDERR_, PETSC_VIEWER_STDERR_WORLD,
          PETSC_VIEWER_STDERR_SELF

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIGetStderr(MPI_Comm comm,PetscViewer *viewer)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stderr_keyval,0);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Stderr_keyval,(void **)viewer,(PetscMPIInt*)&flg);CHKERRQ(ierr);
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscViewerASCIIOpen(comm,"stderr",viewer);CHKERRQ(ierr);
    ierr = PetscObjectRegisterDestroy((PetscObject)*viewer);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Stderr_keyval,(void*)*viewer);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_STDERR_" 
/*@C
   PETSC_VIEWER_STDERR_ - Creates a ASCII PetscViewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the PetscViewer

   Level: beginner

   Note: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,PETSC_VIEWER_STDERR_(comm));

.seealso: PETSC_VIEWER_DRAW_, PetscViewerASCIIOpen(), PETSC_VIEWER_STDOUT_, PETSC_VIEWER_STDOUT_WORLD,
          PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_STDERR_WORLD, PETSC_VIEWER_STDERR_SELF
@*/
PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_STDERR_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStderr(comm,&viewer);
  if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_STDERR_",__FILE__,__SDIR__,1,1," "); PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}


PetscMPIInt Petsc_Viewer_keyval = MPI_KEYVAL_INVALID;
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "Petsc_DelViewer" 
/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.

*/
PetscMPIInt PETSC_DLLEXPORT MPIAPI Petsc_DelViewer(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting viewer data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerASCIIOpen" 
/*@C
   PetscViewerASCIIOpen - Opens an ASCII file as a PetscViewer.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator
-  name - the file name

   Output Parameter:
.  lab - the PetscViewer to use with the specified file

   Level: beginner

   Notes:
   This PetscViewer can be destroyed with PetscViewerDestroy().

   If a multiprocessor communicator is used (such as PETSC_COMM_WORLD), 
   then only the first processor in the group opens the file.  All other 
   processors send their data to the first processor to print. 

   Each processor can instead write its own independent output by
   specifying the communicator PETSC_COMM_SELF.

   As shown below, PetscViewerASCIIOpen() is useful in conjunction with 
   MatView() and VecView()
.vb
     PetscViewerASCIIOpen(PETSC_COMM_WORLD,"mat.output",&viewer);
     MatView(matrix,viewer);
.ve

  Concepts: PetscViewerASCII^creating
  Concepts: printf
  Concepts: printing
  Concepts: accessing remote file
  Concepts: remote file

.seealso: MatView(), VecView(), PetscViewerDestroy(), PetscViewerBinaryOpen(),
          PetscViewerASCIIGetPointer(), PetscViewerSetFormat(), PETSC_VIEWER_STDOUT_, PETSC_VIEWER_STDERR_,
          PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF, 
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerASCIIOpen(MPI_Comm comm,const char name[],PetscViewer *lab)
{
  PetscErrorCode    ierr;
  PetscViewerLink   *vlink,*nv;
  PetscTruth        flg,eq;
  size_t            len;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) {
    ierr = PetscViewerASCIIGetStdout(comm,lab);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*lab);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (Petsc_Viewer_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelViewer,&Petsc_Viewer_keyval,(void*)0);CHKERRQ(ierr);
  }
  /* make sure communicator is a PETSc communicator */
  ierr = PetscCommDuplicate(comm,&comm,PETSC_NULL);CHKERRQ(ierr);
  /* has file already been opened into a viewer */
  ierr = MPI_Attr_get(comm,Petsc_Viewer_keyval,(void**)&vlink,(PetscMPIInt*)&flg);CHKERRQ(ierr);
  if (flg) {
    while (vlink) {
      ierr = PetscStrcmp(name,((PetscViewer_ASCII*)(vlink->viewer->data))->filename,&eq);CHKERRQ(ierr);
      if (eq) {
        ierr = PetscObjectReference((PetscObject)vlink->viewer);CHKERRQ(ierr);
        *lab = vlink->viewer;
        ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }            
      vlink = vlink->next;
    }
  }
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  if (name) {
    ierr = PetscViewerFileSetName(*lab,name);CHKERRQ(ierr);
  }
  /* save viewer into communicator if needed later */
  ierr       = PetscNew(PetscViewerLink,&nv);CHKERRQ(ierr);
  nv->viewer = *lab;
  if (!flg) {
    ierr = MPI_Attr_put(comm,Petsc_Viewer_keyval,nv);CHKERRQ(ierr);
  } else {
    ierr = MPI_Attr_get(comm,Petsc_Viewer_keyval,(void**)&vlink,(PetscMPIInt*)&flg);CHKERRQ(ierr);
    if (vlink) {
      while (vlink->next) vlink = vlink->next;
      vlink->next = nv;
    } else {
      ierr = MPI_Attr_put(comm,Petsc_Viewer_keyval,nv);CHKERRQ(ierr);
    }
  }
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


