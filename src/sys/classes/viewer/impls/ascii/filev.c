
#include <../src/sys/classes/viewer/impls/ascii/asciiimpl.h>  /*I "petscviewer.h" I*/

#define QUEUESTRINGSIZE 8192

static PetscErrorCode PetscViewerFileClose_ASCII(PetscViewer viewer)
{
  PetscMPIInt       rank;
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;
  int               err;

  PetscFunctionBegin;
  PetscCheck(!vascii->sviewer,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"Cannot call with outstanding call to PetscViewerRestoreSubViewer()");
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
  if (rank == 0 && vascii->fd != stderr && vascii->fd != PETSC_STDOUT) {
    if (vascii->fd && vascii->closefile) {
      err = fclose(vascii->fd);
      PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
    }
    if (vascii->storecompressed) {
      char par[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
      FILE *fp;
      CHKERRQ(PetscStrncpy(par,"gzip ",sizeof(par)));
      CHKERRQ(PetscStrlcat(par,vascii->filename,sizeof(par)));
#if defined(PETSC_HAVE_POPEN)
      CHKERRQ(PetscPOpen(PETSC_COMM_SELF,NULL,par,"r",&fp));
      PetscCheckFalse(fgets(buf,1024,fp),PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from compression command %s\n%s",par,buf);
      CHKERRQ(PetscPClose(PETSC_COMM_SELF,fp));
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
    }
  }
  CHKERRQ(PetscFree(vascii->filename));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------*/
PetscErrorCode PetscViewerDestroy_ASCII(PetscViewer viewer)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;
  PetscViewerLink   *vlink;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCheck(!vascii->sviewer,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"Cannot call with outstanding call to PetscViewerRestoreSubViewer()");
  CHKERRQ(PetscViewerFileClose_ASCII(viewer));
  CHKERRQ(PetscFree(vascii));

  /* remove the viewer from the list in the MPI Communicator */
  if (Petsc_Viewer_keyval == MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelViewer,&Petsc_Viewer_keyval,(void*)0));
  }

  CHKERRMPI(MPI_Comm_get_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_keyval,(void**)&vlink,(PetscMPIInt*)&flg));
  if (flg) {
    if (vlink && vlink->viewer == viewer) {
      if (vlink->next) {
        CHKERRMPI(MPI_Comm_set_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_keyval,vlink->next));
      } else {
        CHKERRMPI(MPI_Comm_delete_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_keyval));
      }
      CHKERRQ(PetscFree(vlink));
    } else {
      while (vlink && vlink->next) {
        if (vlink->next->viewer == viewer) {
          PetscViewerLink *nv = vlink->next;
          vlink->next = vlink->next->next;
          CHKERRQ(PetscFree(nv));
        }
        vlink = vlink->next;
      }
    }
  }

  if (Petsc_Viewer_Stdout_keyval != MPI_KEYVAL_INVALID) {
    PetscViewer aviewer;
    CHKERRMPI(MPI_Comm_get_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_Stdout_keyval,(void**)&aviewer,(PetscMPIInt*)&flg));
    if (flg && aviewer == viewer) {
      CHKERRMPI(MPI_Comm_delete_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_Stdout_keyval));
    }
  }
  if (Petsc_Viewer_Stderr_keyval != MPI_KEYVAL_INVALID) {
    PetscViewer aviewer;
    CHKERRMPI(MPI_Comm_get_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_Stderr_keyval,(void**)&aviewer,(PetscMPIInt*)&flg));
    if (flg && aviewer == viewer) {
      CHKERRMPI(MPI_Comm_delete_attr(PetscObjectComm((PetscObject)viewer),Petsc_Viewer_Stderr_keyval));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerDestroy_ASCII_SubViewer(PetscViewer viewer)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerRestoreSubViewer(vascii->bviewer,0,&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlush_ASCII(PetscViewer viewer)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;
  int               err;
  MPI_Comm          comm;
  PetscMPIInt       rank,size;
  FILE              *fd = vascii->fd;

  PetscFunctionBegin;
  PetscCheck(!vascii->sviewer,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"Cannot call with outstanding call to PetscViewerRestoreSubViewer()");
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  if (!vascii->bviewer && rank == 0 && (vascii->mode != FILE_MODE_READ)) {
    err = fflush(vascii->fd);
    PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() call failed");
  }

  if (vascii->allowsynchronized) {
    PetscMPIInt   tag,i,j,n = 0,dummy = 0;
    char          *message;
    MPI_Status    status;

    CHKERRQ(PetscCommDuplicate(comm,&comm,&tag));

    /* First processor waits for messages from all other processors */
    if (rank == 0) {
      /* flush my own messages that I may have queued up */
      PrintfQueue next = vascii->petsc_printfqueuebase,previous;
      for (i=0; i<vascii->petsc_printfqueuelength; i++) {
        if (!vascii->bviewer) {
          CHKERRQ(PetscFPrintf(comm,fd,"%s",next->string));
        } else {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(vascii->bviewer,"%s",next->string));
        }
        previous = next;
        next     = next->next;
        CHKERRQ(PetscFree(previous->string));
        CHKERRQ(PetscFree(previous));
      }
      vascii->petsc_printfqueue       = NULL;
      vascii->petsc_printfqueuelength = 0;
      for (i=1; i<size; i++) {
        /* to prevent a flood of messages to process zero, request each message separately */
        CHKERRMPI(MPI_Send(&dummy,1,MPI_INT,i,tag,comm));
        CHKERRMPI(MPI_Recv(&n,1,MPI_INT,i,tag,comm,&status));
        for (j=0; j<n; j++) {
          PetscMPIInt size = 0;

          CHKERRMPI(MPI_Recv(&size,1,MPI_INT,i,tag,comm,&status));
          CHKERRQ(PetscMalloc1(size, &message));
          CHKERRMPI(MPI_Recv(message,size,MPI_CHAR,i,tag,comm,&status));
          if (!vascii->bviewer) {
            CHKERRQ(PetscFPrintf(comm,fd,"%s",message));
          } else {
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(vascii->bviewer,"%s",message));
          }
          CHKERRQ(PetscFree(message));
        }
      }
    } else { /* other processors send queue to processor 0 */
      PrintfQueue next = vascii->petsc_printfqueuebase,previous;

      CHKERRMPI(MPI_Recv(&dummy,1,MPI_INT,0,tag,comm,&status));
      CHKERRMPI(MPI_Send(&vascii->petsc_printfqueuelength,1,MPI_INT,0,tag,comm));
      for (i=0; i<vascii->petsc_printfqueuelength; i++) {
        CHKERRMPI(MPI_Send(&next->size,1,MPI_INT,0,tag,comm));
        CHKERRMPI(MPI_Send(next->string,next->size,MPI_CHAR,0,tag,comm));
        previous = next;
        next     = next->next;
        CHKERRQ(PetscFree(previous->string));
        CHKERRQ(PetscFree(previous));
      }
      vascii->petsc_printfqueue       = NULL;
      vascii->petsc_printfqueuelength = 0;
    }
    CHKERRQ(PetscCommDestroy(&comm));
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerASCIIGetPointer - Extracts the file pointer from an ASCII PetscViewer.

    Not Collective, depending on the viewer the value may be meaningless except for process 0 of the viewer

    Input Parameter:
.    viewer - PetscViewer context, obtained from PetscViewerASCIIOpen()

    Output Parameter:
.    fd - file pointer

    Notes: for the standard PETSCVIEWERASCII the value is valid only on process 0 of the viewer

    Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscViewerASCIIOpen(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerCreate(), PetscViewerASCIIPrintf(),
          PetscViewerASCIISynchronizedPrintf(), PetscViewerFlush()
@*/
PetscErrorCode  PetscViewerASCIIGetPointer(PetscViewer viewer,FILE **fd)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  *fd = vascii->fd;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileGetMode_ASCII(PetscViewer viewer, PetscFileMode *mode)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  *mode = vascii->mode;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetMode_ASCII(PetscViewer viewer, PetscFileMode mode)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  vascii->mode = mode;
  PetscFunctionReturn(0);
}

/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
PETSC_INTERN FILE *petsc_history;

/*@
    PetscViewerASCIISetTab - Causes PetscViewer to tab in a number of times

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - obtained with PetscViewerASCIIOpen()
-    tabs - number of tabs

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIIGetTab(),
          PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer(), PetscViewerASCIIPushTab()
@*/
PetscErrorCode  PetscViewerASCIISetTab(PetscViewer viewer,PetscInt tabs)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) ascii->tab = tabs;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerASCIIGetTab - Return the number of tabs used by PetscViewer.

    Not Collective, meaningful on first processor only.

    Input Parameters:
.    viewer - obtained with PetscViewerASCIIOpen()

    Output Parameters:
.    tabs - number of tabs

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISetTab(),
          PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer(), PetscViewerASCIIPushTab()
@*/
PetscErrorCode  PetscViewerASCIIGetTab(PetscViewer viewer,PetscInt *tabs)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii && tabs) *tabs = ascii->tab;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerASCIIAddTab - Add to the number of times an ASCII viewer tabs before printing

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - obtained with PetscViewerASCIIOpen()
-    tabs - number of tabs

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer(), PetscViewerASCIIPushTab()
@*/
PetscErrorCode  PetscViewerASCIIAddTab(PetscViewer viewer,PetscInt tabs)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) ascii->tab += tabs;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerASCIISubtractTab - Subtracts from the number of times an ASCII viewer tabs before printing

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - obtained with PetscViewerASCIIOpen()
-    tabs - number of tabs

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer(), PetscViewerASCIIPushTab()
@*/
PetscErrorCode  PetscViewerASCIISubtractTab(PetscViewer viewer,PetscInt tabs)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) ascii->tab -= tabs;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerASCIIPushSynchronized - Allows calls to PetscViewerASCIISynchronizedPrintf() for this viewer

    Collective on PetscViewer

    Input Parameters:
.    viewer - obtained with PetscViewerASCIIOpen()

    Level: intermediate

    Notes:
    See documentation of PetscViewerASCIISynchronizedPrintf() for more details how the synchronized output should be done properly.

.seealso: PetscViewerASCIISynchronizedPrintf(), PetscViewerFlush(), PetscViewerASCIIPopSynchronized(),
          PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType()
@*/
PetscErrorCode  PetscViewerASCIIPushSynchronized(PetscViewer viewer)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscCheck(!ascii->sviewer,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"Cannot call with outstanding call to PetscViewerRestoreSubViewer()");
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) ascii->allowsynchronized++;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerASCIIPopSynchronized - Undoes most recent PetscViewerASCIIPushSynchronized() for this viewer

    Collective on PetscViewer

    Input Parameters:
.    viewer - obtained with PetscViewerASCIIOpen()

    Level: intermediate

    Notes:
    See documentation of PetscViewerASCIISynchronizedPrintf() for more details how the synchronized output should be done properly.

.seealso: PetscViewerASCIIPushSynchronized(), PetscViewerASCIISynchronizedPrintf(), PetscViewerFlush(),
          PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType()
@*/
PetscErrorCode  PetscViewerASCIIPopSynchronized(PetscViewer viewer)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscCheck(!ascii->sviewer,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"Cannot call with outstanding call to PetscViewerRestoreSubViewer()");
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    ascii->allowsynchronized--;
    PetscCheckFalse(ascii->allowsynchronized < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Called more times than PetscViewerASCIIPushSynchronized()");
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerASCIIPushTab - Adds one more tab to the amount that PetscViewerASCIIPrintf()
     lines are tabbed.

    Not Collective, but only first processor in set has any effect

    Input Parameters:
.    viewer - obtained with PetscViewerASCIIOpen()

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer()
@*/
PetscErrorCode  PetscViewerASCIIPushTab(PetscViewer viewer)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) ascii->tab++;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerASCIIPopTab - Removes one tab from the amount that PetscViewerASCIIPrintf()
     lines are tabbed.

    Not Collective, but only first processor in set has any effect

    Input Parameters:
.    viewer - obtained with PetscViewerASCIIOpen()

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIIPushTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer()
@*/
PetscErrorCode  PetscViewerASCIIPopTab(PetscViewer viewer)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCheckFalse(ascii->tab <= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"More tabs popped than pushed");
    ascii->tab--;
  }
  PetscFunctionReturn(0);
}

/*@
    PetscViewerASCIIUseTabs - Turns on or off the use of tabs with the ASCII PetscViewer

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - obtained with PetscViewerASCIIOpen()
-    flg - PETSC_TRUE or PETSC_FALSE

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIPushTab(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer()
@*/
PetscErrorCode  PetscViewerASCIIUseTabs(PetscViewer viewer,PetscBool flg)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    if (flg) ascii->tab = ascii->tab_store;
    else {
      ascii->tab_store = ascii->tab;
      ascii->tab       = 0;
    }
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------- */

/*@C
    PetscViewerASCIIPrintf - Prints to a file, only from the first
    processor in the PetscViewer

    Not Collective, but only first processor in set has any effect

    Input Parameters:
+    viewer - obtained with PetscViewerASCIIOpen()
-    format - the usual printf() format string

    Level: developer

    Fortran Note:
    The call sequence is PetscViewerASCIIPrintf(PetscViewer, character(*), int ierr) from Fortran.
    That is, you can only pass a single character string from Fortran.

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIOpen(),
          PetscViewerASCIIPushTab(), PetscViewerASCIIPopTab(), PetscViewerASCIISynchronizedPrintf(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType(), PetscViewerASCIIGetPointer(), PetscViewerASCIIPushSynchronized()
@*/
PetscErrorCode  PetscViewerASCIIPrintf(PetscViewer viewer,const char format[],...)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  PetscMPIInt       rank;
  PetscInt          tab,intab = ascii->tab;
  FILE              *fd = ascii->fd;
  PetscBool         iascii;
  int               err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscCheck(!ascii->sviewer,PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_WRONGSTATE,"Cannot call with outstanding call to PetscViewerRestoreSubViewer()");
  PetscValidCharPointer(format,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCheck(iascii,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not ASCII PetscViewer");
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
  if (rank) PetscFunctionReturn(0);

  if (ascii->bviewer) { /* pass string up to parent viewer */
    char        *string;
    va_list     Argp;
    size_t      fullLength;

    CHKERRQ(PetscCalloc1(QUEUESTRINGSIZE, &string));
    va_start(Argp,format);
    CHKERRQ(PetscVSNPrintf(string,QUEUESTRINGSIZE,format,&fullLength,Argp));
    va_end(Argp);
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"%s",string));
    CHKERRQ(PetscFree(string));
  } else { /* write directly to file */
    va_list Argp;
    /* flush my own messages that I may have queued up */
    PrintfQueue next = ascii->petsc_printfqueuebase,previous;
    PetscInt    i;
    for (i=0; i<ascii->petsc_printfqueuelength; i++) {
      CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,fd,"%s",next->string));
      previous = next;
      next     = next->next;
      CHKERRQ(PetscFree(previous->string));
      CHKERRQ(PetscFree(previous));
    }
    ascii->petsc_printfqueue       = NULL;
    ascii->petsc_printfqueuelength = 0;
    tab = intab;
    while (tab--) {
      CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,fd,"  "));
    }

    va_start(Argp,format);
    CHKERRQ((*PetscVFPrintf)(fd,format,Argp));
    err  = fflush(fd);
    PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
    if (petsc_history) {
      va_start(Argp,format);
      tab = intab;
      while (tab--) {
        CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,petsc_history,"  "));
      }
      CHKERRQ((*PetscVFPrintf)(petsc_history,format,Argp));
      err  = fflush(petsc_history);
      PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
    }
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscViewerFileSetName - Sets the name of the file the PetscViewer uses.

    Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; either ASCII or binary
-  name - the name of the file it should use

    Level: advanced

.seealso: PetscViewerCreate(), PetscViewerSetType(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), PetscViewerDestroy(),
          PetscViewerASCIIGetPointer(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf()

@*/
PetscErrorCode  PetscViewerFileSetName(PetscViewer viewer,const char name[])
{
  char           filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(name,2);
  CHKERRQ(PetscStrreplace(PetscObjectComm((PetscObject)viewer),name,filename,sizeof(filename)));
  CHKERRQ(PetscTryMethod(viewer,"PetscViewerFileSetName_C",(PetscViewer,const char[]),(viewer,filename)));
  PetscFunctionReturn(0);
}

/*@C
     PetscViewerFileGetName - Gets the name of the file the PetscViewer uses.

    Not Collective

  Input Parameter:
.  viewer - the PetscViewer; either ASCII or binary

  Output Parameter:
.  name - the name of the file it is using

    Level: advanced

.seealso: PetscViewerCreate(), PetscViewerSetType(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), PetscViewerFileSetName()

@*/
PetscErrorCode  PetscViewerFileGetName(PetscViewer viewer,const char **name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(name,2);
  CHKERRQ(PetscUseMethod(viewer,"PetscViewerFileGetName_C",(PetscViewer,const char**),(viewer,name)));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileGetName_ASCII(PetscViewer viewer,const char **name)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  *name = vascii->filename;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetName_ASCII(PetscViewer viewer,const char name[])
{
  size_t            len;
  char              fname[PETSC_MAX_PATH_LEN],*gz;
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;
  PetscBool         isstderr,isstdout;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerFileClose_ASCII(viewer));
  if (!name) PetscFunctionReturn(0);
  CHKERRQ(PetscStrallocpy(name,&vascii->filename));

  /* Is this file to be compressed */
  vascii->storecompressed = PETSC_FALSE;

  CHKERRQ(PetscStrstr(vascii->filename,".gz",&gz));
  if (gz) {
    CHKERRQ(PetscStrlen(gz,&len));
    if (len == 3) {
      PetscCheckFalse(vascii->mode != FILE_MODE_WRITE,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Cannot open ASCII PetscViewer file that is compressed; uncompress it manually first");
      *gz = 0;
      vascii->storecompressed = PETSC_TRUE;
    }
  }
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
  if (rank == 0) {
    CHKERRQ(PetscStrcmp(name,"stderr",&isstderr));
    CHKERRQ(PetscStrcmp(name,"stdout",&isstdout));
    /* empty filename means stdout */
    if (name[0] == 0)  isstdout = PETSC_TRUE;
    if (isstderr)      vascii->fd = PETSC_STDERR;
    else if (isstdout) vascii->fd = PETSC_STDOUT;
    else {

      CHKERRQ(PetscFixFilename(name,fname));
      switch (vascii->mode) {
      case FILE_MODE_READ:
        vascii->fd = fopen(fname,"r");
        break;
      case FILE_MODE_WRITE:
        vascii->fd = fopen(fname,"w");
        break;
      case FILE_MODE_APPEND:
        vascii->fd = fopen(fname,"a");
        break;
      case FILE_MODE_UPDATE:
        vascii->fd = fopen(fname,"r+");
        if (!vascii->fd) vascii->fd = fopen(fname,"w+");
        break;
      case FILE_MODE_APPEND_UPDATE:
        /* I really want a file which is opened at the end for updating,
           not a+, which opens at the beginning, but makes writes at the end.
        */
        vascii->fd = fopen(fname,"r+");
        if (!vascii->fd) vascii->fd = fopen(fname,"w+");
        else {
          CHKERRQ(fseek(vascii->fd, 0, SEEK_END));
        }
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[vascii->mode]);
      }
      PetscCheck(vascii->fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open PetscViewer file: %s",fname);
    }
  }
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)viewer,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerGetSubViewer_ASCII(PetscViewer viewer,MPI_Comm subcomm,PetscViewer *outviewer)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data,*ovascii;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
  PetscCheck(!vascii->sviewer,PETSC_COMM_SELF,PETSC_ERR_ORDER,"SubViewer already obtained from PetscViewer and not restored");
  /*
     The following line is a bug; it does another PetscViewerASCIIPushSynchronized() on viewer, but if it is removed the code won't work
     because it relies on this behavior in other places. In particular this line causes the synchronized flush to occur when the viewer is destroyed
     (since the count never gets to zero) in some examples this displays information that otherwise would be lost

     This code also means another call to PetscViewerASCIIPopSynchronized() must be made after the PetscViewerRestoreSubViewer(), see, for example,
     PCView_GASM().
  */
  CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
  CHKERRQ(PetscViewerCreate(subcomm,outviewer));
  CHKERRQ(PetscViewerSetType(*outviewer,PETSCVIEWERASCII));
  CHKERRQ(PetscViewerASCIIPushSynchronized(*outviewer));
  ovascii      = (PetscViewer_ASCII*)(*outviewer)->data;
  ovascii->fd  = vascii->fd;
  ovascii->tab = vascii->tab;
  ovascii->closefile = PETSC_FALSE;

  vascii->sviewer = *outviewer;
  (*outviewer)->format  = viewer->format;
  ((PetscViewer_ASCII*)((*outviewer)->data))->bviewer = viewer;
  (*outviewer)->ops->destroy = PetscViewerDestroy_ASCII_SubViewer;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerRestoreSubViewer_ASCII(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  PetscViewer_ASCII *ascii  = (PetscViewer_ASCII*)viewer->data;

  PetscFunctionBegin;
  PetscCheck(ascii->sviewer,PETSC_COMM_SELF,PETSC_ERR_ORDER,"SubViewer never obtained from PetscViewer");
  PetscCheckFalse(ascii->sviewer != *outviewer,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"This PetscViewer did not generate this SubViewer");

  CHKERRQ(PetscViewerASCIIPopSynchronized(*outviewer));
  ascii->sviewer             = NULL;
  (*outviewer)->ops->destroy = PetscViewerDestroy_ASCII;
  CHKERRQ(PetscViewerDestroy(outviewer));
  CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerView_ASCII(PetscViewer v,PetscViewer viewer)
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)v->data;

  PetscFunctionBegin;
  if (ascii->filename) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Filename: %s\n",ascii->filename));
  }
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERASCII - A viewer that prints to stdout or an ASCII file

.seealso:  PETSC_VIEWER_STDOUT_(),PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_STDOUT_WORLD,PetscViewerCreate(), PetscViewerASCIIOpen(),
           PetscViewerMatlabOpen(), VecView(), DMView(), PetscViewerMatlabPutArray(), PETSCVIEWERBINARY, PETSCVIEWERMATLAB,
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner

M*/
PETSC_EXTERN PetscErrorCode PetscViewerCreate_ASCII(PetscViewer viewer)
{
  PetscViewer_ASCII *vascii;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(viewer,&vascii));
  viewer->data = (void*)vascii;

  viewer->ops->destroy          = PetscViewerDestroy_ASCII;
  viewer->ops->flush            = PetscViewerFlush_ASCII;
  viewer->ops->getsubviewer     = PetscViewerGetSubViewer_ASCII;
  viewer->ops->restoresubviewer = PetscViewerRestoreSubViewer_ASCII;
  viewer->ops->view             = PetscViewerView_ASCII;
  viewer->ops->read             = PetscViewerASCIIRead;

  /* defaults to stdout unless set with PetscViewerFileSetName() */
  vascii->fd        = PETSC_STDOUT;
  vascii->mode      = FILE_MODE_WRITE;
  vascii->bviewer   = NULL;
  vascii->subviewer = NULL;
  vascii->sviewer   = NULL;
  vascii->tab       = 0;
  vascii->tab_store = 0;
  vascii->filename  = NULL;
  vascii->closefile = PETSC_TRUE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",PetscViewerFileSetName_ASCII));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",PetscViewerFileGetName_ASCII));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_ASCII));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_ASCII));
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerASCIISynchronizedPrintf - Prints synchronized output to the specified file from
    several processors.  Output of the first processor is followed by that of the
    second, etc.

    Not Collective, must call collective PetscViewerFlush() to get the results out

    Input Parameters:
+   viewer - the ASCII PetscViewer
-   format - the usual printf() format string

    Level: intermediate

    Notes:
    You must have previously called PetscViewerASCIIPushSynchronized() to allow this routine to be called.
    Then you can do multiple independent calls to this routine.
    The actual synchronized print is then done using PetscViewerFlush().
    PetscViewerASCIIPopSynchronized() should be then called if we are already done with the synchronized output
    to conclude the "synchronized session".
    So the typical calling sequence looks like
$ PetscViewerASCIIPushSynchronized(viewer);
$ PetscViewerASCIISynchronizedPrintf(viewer, ...);
$ PetscViewerASCIISynchronizedPrintf(viewer, ...);
$ ...
$ PetscViewerFlush(viewer);
$ PetscViewerASCIISynchronizedPrintf(viewer, ...);
$ PetscViewerASCIISynchronizedPrintf(viewer, ...);
$ ...
$ PetscViewerFlush(viewer);
$ PetscViewerASCIIPopSynchronized(viewer);

    Fortran Note:
      Can only print a single character* string

.seealso: PetscViewerASCIIPushSynchronized(), PetscViewerFlush(), PetscViewerASCIIPopSynchronized(),
          PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIIOpen(),
          PetscViewerCreate(), PetscViewerDestroy(), PetscViewerSetType()
@*/
PetscErrorCode  PetscViewerASCIISynchronizedPrintf(PetscViewer viewer,const char format[],...)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;
  PetscMPIInt       rank;
  PetscInt          tab = vascii->tab;
  MPI_Comm          comm;
  FILE              *fp;
  PetscBool         iascii,hasbviewer = PETSC_FALSE;
  int               err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(format,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCheck(iascii,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not ASCII PetscViewer");
  PetscCheck(vascii->allowsynchronized,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"First call PetscViewerASCIIPushSynchronized() to allow this call");

  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  if (vascii->bviewer) {
    hasbviewer = PETSC_TRUE;
    if (rank == 0) {
      vascii = (PetscViewer_ASCII*)vascii->bviewer->data;
      CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
      CHKERRMPI(MPI_Comm_rank(comm,&rank));
    }
  }

  fp   = vascii->fd;

  if (rank == 0 && !hasbviewer) {   /* First processor prints immediately to fp */
    va_list Argp;
    /* flush my own messages that I may have queued up */
    PrintfQueue next = vascii->petsc_printfqueuebase,previous;
    PetscInt    i;
    for (i=0; i<vascii->petsc_printfqueuelength; i++) {
      CHKERRQ(PetscFPrintf(comm,fp,"%s",next->string));
      previous = next;
      next     = next->next;
      CHKERRQ(PetscFree(previous->string));
      CHKERRQ(PetscFree(previous));
    }
    vascii->petsc_printfqueue       = NULL;
    vascii->petsc_printfqueuelength = 0;

    while (tab--) {
      CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,fp,"  "));
    }

    va_start(Argp,format);
    CHKERRQ((*PetscVFPrintf)(fp,format,Argp));
    err  = fflush(fp);
    PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
    if (petsc_history) {
      va_start(Argp,format);
      CHKERRQ((*PetscVFPrintf)(petsc_history,format,Argp));
      err  = fflush(petsc_history);
      PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
    }
    va_end(Argp);
  } else { /* other processors add to queue */
    char        *string;
    va_list     Argp;
    size_t      fullLength;
    PrintfQueue next;

    CHKERRQ(PetscNew(&next));
    if (vascii->petsc_printfqueue) {
      vascii->petsc_printfqueue->next = next;
      vascii->petsc_printfqueue       = next;
    } else {
      vascii->petsc_printfqueuebase = vascii->petsc_printfqueue = next;
    }
    vascii->petsc_printfqueuelength++;
    next->size = QUEUESTRINGSIZE;
    CHKERRQ(PetscCalloc1(next->size, &next->string));
    string     = next->string;
    tab       *= 2;
    while (tab--) {
      *string++ = ' ';
    }
    va_start(Argp,format);
    CHKERRQ(PetscVSNPrintf(string,next->size-2*vascii->tab,format,&fullLength,Argp));
    va_end(Argp);
    if (fullLength > (size_t) (next->size-2*vascii->tab)) {
      CHKERRQ(PetscFree(next->string));
      next->size = fullLength + 2*vascii->tab;
      CHKERRQ(PetscCalloc1(next->size, &next->string));
      string     = next->string;
      tab        = 2*vascii->tab;
      while (tab--) {
        *string++ = ' ';
      }
      va_start(Argp,format);
      CHKERRQ(PetscVSNPrintf(string,next->size-2*vascii->tab,format,NULL,Argp));
      va_end(Argp);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerASCIIRead - Reads from a ASCII file

   Only process 0 in the PetscViewer may call this

   Input Parameters:
+  viewer - the ascii viewer
.  data - location to write the data
.  num - number of items of data to read
-  datatype - type of data to read

   Output Parameters:
.  count - number of items of data actually read, or NULL

   Level: beginner

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(), PetscViewerCreate(), PetscViewerFileSetMode(), PetscViewerFileSetName()
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead()
@*/
PetscErrorCode PetscViewerASCIIRead(PetscViewer viewer,void *data,PetscInt num,PetscInt *count,PetscDataType dtype)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII*)viewer->data;
  FILE              *fd = vascii->fd;
  PetscInt           i;
  int                ret = 0;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank));
  PetscCheck(!rank,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,"Can only be called from process 0 in the PetscViewer");
  for (i=0; i<num; i++) {
    if (dtype == PETSC_CHAR)         ret = fscanf(fd, "%c",  &(((char*)data)[i]));
    else if (dtype == PETSC_STRING)  ret = fscanf(fd, "%s",  &(((char*)data)[i]));
    else if (dtype == PETSC_INT)     ret = fscanf(fd, "%" PetscInt_FMT,  &(((PetscInt*)data)[i]));
    else if (dtype == PETSC_ENUM)    ret = fscanf(fd, "%d",  &(((int*)data)[i]));
    else if (dtype == PETSC_INT64)   ret = fscanf(fd, "%" PetscInt64_FMT,  &(((PetscInt64*)data)[i]));
    else if (dtype == PETSC_LONG)    ret = fscanf(fd, "%ld", &(((long*)data)[i]));
    else if (dtype == PETSC_FLOAT)   ret = fscanf(fd, "%f",  &(((float*)data)[i]));
    else if (dtype == PETSC_DOUBLE)  ret = fscanf(fd, "%lg", &(((double*)data)[i]));
#if defined(PETSC_USE_REAL___FLOAT128)
    else if (dtype == PETSC___FLOAT128) {
      double tmp;
      ret = fscanf(fd, "%lg", &tmp);
      ((__float128*)data)[i] = tmp;
    }
#endif
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Data type %d not supported", (int) dtype);
    PetscCheck(ret,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Conversion error for data type %d", (int) dtype);
    else if (ret < 0) break; /* Proxy for EOF, need to check for it in configure */
  }
  if (count) *count = i;
  else PetscCheckFalse(ret < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Insufficient data, read only %" PetscInt_FMT " < %" PetscInt_FMT " items", i, num);
  PetscFunctionReturn(0);
}
