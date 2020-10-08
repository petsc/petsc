
#include <../src/sys/classes/viewer/impls/ascii/asciiimpl.h>  /*I     "petscviewer.h"   I*/

/* ---------------------------------------------------------------------*/

/*
    The variable Petsc_Viewer_Stdout_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;

/*@
   PetscViewerASCIIGetStdout - Creates a ASCII PetscViewer shared by all processors
                    in a communicator. Error returning version of PETSC_VIEWER_STDOUT_()

   Collective

   Input Parameter:
.  comm - the MPI communicator to share the PetscViewer

   Level: beginner

   Notes:
     This should be used in all PETSc source code instead of PETSC_VIEWER_STDOUT_()

.seealso: PETSC_VIEWER_DRAW_(), PetscViewerASCIIOpen(), PETSC_VIEWER_STDERR_, PETSC_VIEWER_STDOUT_WORLD,
          PETSC_VIEWER_STDOUT_SELF

@*/
PetscErrorCode  PetscViewerASCIIGetStdout(MPI_Comm comm,PetscViewer *viewer)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscSpinlockLock(&PetscViewerASCIISpinLockStdout);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&ncomm,NULL);CHKERRQ(ierr);
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Viewer_Stdout_keyval,NULL);CHKERRMPI(ierr);
  }
  ierr = MPI_Comm_get_attr(ncomm,Petsc_Viewer_Stdout_keyval,(void**)viewer,(PetscMPIInt*)&flg);CHKERRMPI(ierr);
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscViewerASCIIOpen(ncomm,"stdout",viewer);CHKERRQ(ierr);
    ierr = PetscObjectRegisterDestroy((PetscObject)*viewer);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(ncomm,Petsc_Viewer_Stdout_keyval,(void*)*viewer);CHKERRMPI(ierr);
  }
  ierr = PetscCommDestroy(&ncomm);CHKERRQ(ierr);
  ierr = PetscSpinlockUnlock(&PetscViewerASCIISpinLockStdout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PETSC_VIEWER_STDOUT_ - Creates a ASCII PetscViewer shared by all processors
                    in a communicator.

   Collective

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
PetscViewer  PETSC_VIEWER_STDOUT_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStdout(comm,&viewer);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_STDOUT_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(NULL);}
  PetscFunctionReturn(viewer);
}

/* ---------------------------------------------------------------------*/

/*
    The variable Petsc_Viewer_Stderr_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;

/*@
   PetscViewerASCIIGetStderr - Creates a ASCII PetscViewer shared by all processors
                    in a communicator. Error returning version of PETSC_VIEWER_STDERR_()

   Collective

   Input Parameter:
.  comm - the MPI communicator to share the PetscViewer

   Level: beginner

   Notes:
     This should be used in all PETSc source code instead of PETSC_VIEWER_STDERR_()

.seealso: PETSC_VIEWER_DRAW_(), PetscViewerASCIIOpen(), PETSC_VIEWER_STDERR_, PETSC_VIEWER_STDERR_WORLD,
          PETSC_VIEWER_STDERR_SELF

@*/
PetscErrorCode  PetscViewerASCIIGetStderr(MPI_Comm comm,PetscViewer *viewer)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscSpinlockLock(&PetscViewerASCIISpinLockStderr);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&ncomm,NULL);CHKERRQ(ierr);
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Viewer_Stderr_keyval,NULL);CHKERRMPI(ierr);
  }
  ierr = MPI_Comm_get_attr(ncomm,Petsc_Viewer_Stderr_keyval,(void**)viewer,(PetscMPIInt*)&flg);CHKERRMPI(ierr);
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscViewerASCIIOpen(ncomm,"stderr",viewer);CHKERRQ(ierr);
    ierr = PetscObjectRegisterDestroy((PetscObject)*viewer);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(ncomm,Petsc_Viewer_Stderr_keyval,(void*)*viewer);CHKERRMPI(ierr);
  }
  ierr = PetscCommDestroy(&ncomm);CHKERRQ(ierr);
  ierr = PetscSpinlockUnlock(&PetscViewerASCIISpinLockStderr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PETSC_VIEWER_STDERR_ - Creates a ASCII PetscViewer shared by all processors
                    in a communicator.

   Collective

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
PetscViewer  PETSC_VIEWER_STDERR_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStderr(comm,&viewer);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_STDERR_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," "); PetscFunctionReturn(NULL);}
  PetscFunctionReturn(viewer);
}


PetscMPIInt Petsc_Viewer_keyval = MPI_KEYVAL_INVALID;
/*
   Called with MPI_Comm_free() is called on a communicator that has a viewer as an attribute. The viewer is not actually destroyed because that is managed by
   PetscObjectDestroyRegisterAll(). PetscViewerASCIIGetStdout() registers the viewer with PetscObjectDestroyRegister() to be destroyed when PetscFinalize() is called.

  This is called by MPI, not by users.

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelViewer(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Removing viewer data attribute in an MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*@C
   PetscViewerASCIIOpen - Opens an ASCII file for writing as a PetscViewer.

   Collective

   Input Parameters:
+  comm - the communicator
-  name - the file name

   Output Parameter:
.  lab - the PetscViewer to use with the specified file

   Level: beginner

   Notes:
   To open a ASCII file as a viewer for reading one must use the sequence
$     PetscViewerCreate(comm,&lab);
$     PetscViewerSetType(lab,PETSCVIEWERASCII);
$     PetscViewerFileSetMode(lab,FILE_MODE_READ);
$     PetscViewerFileSetName(lab,name);

   This PetscViewer can be destroyed with PetscViewerDestroy().

   The MPI communicator used here must match that used by the object one is viewing. For example if the
   Mat was created with a PETSC_COMM_WORLD, then the Viewer must be created with PETSC_COMM_WORLD

   As shown below, PetscViewerASCIIOpen() is useful in conjunction with
   MatView() and VecView()
.vb
     PetscViewerASCIIOpen(PETSC_COMM_WORLD,"mat.output",&viewer);
     MatView(matrix,viewer);
.ve

.seealso: MatView(), VecView(), PetscViewerDestroy(), PetscViewerBinaryOpen(), PetscViewerASCIIRead()
          PetscViewerASCIIGetPointer(), PetscViewerPushFormat(), PETSC_VIEWER_STDOUT_, PETSC_VIEWER_STDERR_,
          PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF,
@*/
PetscErrorCode  PetscViewerASCIIOpen(MPI_Comm comm,const char name[],PetscViewer *lab)
{
  PetscErrorCode  ierr;
  PetscViewerLink *vlink,*nv;
  PetscBool       flg,eq;
  size_t          len;

  PetscFunctionBegin;
  ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
  if (!len) {
    ierr = PetscViewerASCIIGetStdout(comm,lab);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*lab);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscSpinlockLock(&PetscViewerASCIISpinLockOpen);CHKERRQ(ierr);
  if (Petsc_Viewer_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelViewer,&Petsc_Viewer_keyval,(void*)0);CHKERRMPI(ierr);
  }
  /*
       It would be better to move this code to PetscFileSetName() but since it must return a preexiting communicator
     we cannot do that, since PetscFileSetName() takes a communicator that already exists.

      Plus if the original communicator that created the file has since been close this will not detect the old
      communictor and hence will overwrite the old data. It may be better to simply remove all this code
  */
  /* make sure communicator is a PETSc communicator */
  ierr = PetscCommDuplicate(comm,&comm,NULL);CHKERRQ(ierr);
  /* has file already been opened into a viewer */
  ierr = MPI_Comm_get_attr(comm,Petsc_Viewer_keyval,(void**)&vlink,(PetscMPIInt*)&flg);CHKERRMPI(ierr);
  if (flg) {
    while (vlink) {
      ierr = PetscStrcmp(name,((PetscViewer_ASCII*)(vlink->viewer->data))->filename,&eq);CHKERRQ(ierr);
      if (eq) {
        ierr = PetscObjectReference((PetscObject)vlink->viewer);CHKERRQ(ierr);
        *lab = vlink->viewer;
        ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
        ierr = PetscSpinlockUnlock(&PetscViewerASCIISpinLockOpen);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      vlink = vlink->next;
    }
  }
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERASCII);CHKERRQ(ierr);
  if (name) {
    ierr = PetscViewerFileSetName(*lab,name);CHKERRQ(ierr);
  }
  /* save viewer into communicator if needed later */
  ierr       = PetscNew(&nv);CHKERRQ(ierr);
  nv->viewer = *lab;
  if (!flg) {
    ierr = MPI_Comm_set_attr(comm,Petsc_Viewer_keyval,nv);CHKERRMPI(ierr);
  } else {
    ierr = MPI_Comm_get_attr(comm,Petsc_Viewer_keyval,(void**)&vlink,(PetscMPIInt*)&flg);CHKERRMPI(ierr);
    if (vlink) {
      while (vlink->next) vlink = vlink->next;
      vlink->next = nv;
    } else {
      ierr = MPI_Comm_set_attr(comm,Petsc_Viewer_keyval,nv);CHKERRMPI(ierr);
    }
  }
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  ierr = PetscSpinlockUnlock(&PetscViewerASCIISpinLockOpen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerASCIIOpenWithFILE - Given an open file creates an ASCII viewer that prints to it.

   Collective

   Input Parameters:
+  comm - the communicator
-  fd - the FILE pointer

   Output Parameter:
.  lab - the PetscViewer to use with the specified file

   Level: beginner

   Notes:
   This PetscViewer can be destroyed with PetscViewerDestroy(), but the fd will NOT be closed.

   If a multiprocessor communicator is used (such as PETSC_COMM_WORLD),
   then only the first processor in the group uses the file.  All other
   processors send their data to the first processor to print.

.seealso: MatView(), VecView(), PetscViewerDestroy(), PetscViewerBinaryOpen(),
          PetscViewerASCIIGetPointer(), PetscViewerPushFormat(), PETSC_VIEWER_STDOUT_, PETSC_VIEWER_STDERR_,
          PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF, PetscViewerASCIIOpen()
@*/
PetscErrorCode  PetscViewerASCIIOpenWithFILE(MPI_Comm comm,FILE *fd,PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerASCIISetFILE(*lab,fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerASCIISetFILE(PetscViewer viewer,FILE *fd)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  vascii->fd        = fd;
  vascii->closefile = PETSC_FALSE;
  PetscFunctionReturn(0);
}
