/*$Id: errtrace.c,v 1.11 2000/01/20 03:58:24 bsmith Exp bsmith $*/

#include "petsc.h"           /*I "petsc.h" I*/


#undef __FUNC__  
#define __FUNC__ "PetscTraceBackErrorHandler"
/*@C
   PetscTraceBackErrorHandler - Default error handler routine that generates
   a traceback on error detection.

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
-  ctx - error handler context

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,p,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscStopErrorHandler()

.keywords: default, error, handler, traceback

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler()
 @*/
int PetscTraceBackErrorHandler(int line,char *fun,char* file,char *dir,int n,int p,char *mess,void *ctx)
{
  PLogDouble        mem,rss;
  int               rank,ierr;
  PetscTruth        flg1,flg2;

  PetscFunctionBegin;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
  switch(n)
  {
  case PETSC_ERR_MEM:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Out of memory. This could be due to allocating\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   too large an object or bleeding by not properly\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   destroying unneeded objects.\n",rank);
    ierr = PetscTrSpace(&mem,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscGetResidentSetSize(&rss);CHKERRQ(ierr);
    OptionsHasName(PETSC_NULL,"-trdump",&flg1);
    OptionsHasName(PETSC_NULL,"-trmalloc_log",&flg2);
    if (flg2) {
      ierr = PetscTrLogDump(stderr);CHKERRQ(ierr);
    } else if (flg1) {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
      ierr = PetscTrDump(stderr);CHKERRQ(ierr);
    } else {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Try running with -trdump or -trmalloc_log for info.\n",rank);
    }
    n = 1;
    break;
  case PETSC_ERR_SUP:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   No support for this operation for this object type!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_SIG:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Signal received!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_FP:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Floating point exception!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_COR:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Corrupted Petsc object!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_LIB:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Error in external library!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_PLIB:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Petsc has generated inconsistent data!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_MEMC:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory corruption!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_SIZ:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Nonconforming object sizes!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_IDN:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Argument aliasing not permitted!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_WRONG:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Invalid argument!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_CORRUPT:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Null or corrupt argument!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_OUTOFRANGE:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Argument out of range!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_BADPTR:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Invalid pointer!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_NOTSAMETYPE:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Arguments must have same type!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_WRONGSTATE:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Object is in wrong state!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_ARG_INCOMP:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Arguments are incompatible!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_FILE_OPEN:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Unable to open file!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_FILE_READ:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Read from file failed!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_FILE_WRITE:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Write to file failed!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_FILE_UNEXPECTED:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Unexpected data in file!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_KSP_BRKDWN:
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Detected breakdown in Krylov method!\n",rank);
    n = 1;
    break;
  case PETSC_ERR_MAT_LU_ZRPVT:
    /* Also PETSC_ERR_MAT_CH_ZRPVT */
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Detected zero pivot in factor!\n",rank);
    n = 1;
    break;
  default:;
  }
  if (mess) {
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   %s\n",rank,mess);
  }
  PetscFunctionReturn(n);
}

