/*$Id: errtrace.c,v 1.25 2001/06/21 21:15:22 bsmith Exp $*/

#include "petsc.h"           /*I "petsc.h" I*/

static char *PetscErrorStrings[] = {
  /*55 */ "Out of memory",
          "No support for this operation for this object type",
          "",
  /*58 */ "",
  /*59 */ "Signal received",
  /*60 */  "Nonconforming object sizes",
    "Argument aliasing not permitted",
    "Invalid argument",
  /*63 */    "Argument out of range",
    "Null or corrupt argument",
    "Unable to open file",
    "Read from file failed",
    "Write to file failed",
    "Invalid pointer",
  /*69 */      "Arguments must have same type",
    "Detected breakdown in Krylov method",
  /*71 */    "Detected zero pivot in LU factorization",
  /*72 */    "Floating point exception",
  /*73 */    "Object is in wrong state",
    "Corrupted Petsc object",
    "Arguments are incompatible",
    "Error in external library",
  /*77 */    "Petsc has generated inconsistent data",
    "Memory corruption",
    "Unexpected data in file",
  /*80 */ "Arguments must have same communicators",
  /*81 */ "Detected zero pivot in Cholesky factorization"};

extern char PetscErrorBaseMessage[1024];

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorMessage" 
/*@C
   PetscErrorMessage - returns the text string associated with a PETSc error code.

   Not Collective

   Input Parameter:
.   errnum - the error code

   Output Parameter: 
+  text - the error message (PETSC_NULL if not desired) 
-  specific - the specific error message that was set with SETERRxxx() or PetscError().  (PETSC_NULL if not desired) 

   Level: developer

   Concepts: error handler^messages

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler(), PetscTraceBackErrorHandler()
 @*/
int PetscErrorMessage(int errnum,char **text,char **specific)
{
  PetscFunctionBegin;
  if (text && errnum >= PETSC_ERR_MEM && errnum <= PETSC_ERR_MEM_MALLOC_0) {
    *text = PetscErrorStrings[errnum-PETSC_ERR_MEM];
  } else if (text) *text = 0;

  if (specific) {
    *specific = PetscErrorBaseMessage;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscIgnoreErrorHandler" 
/*@C
   PetscIgnoreErrorHandler - Ignores the error, allows program to continue as if error did not occure

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNCT__)
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

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler(), PetscTraceBackErrorHandler()
 @*/
int PetscIgnoreErrorHandler(int line,const char *fun,const char* file,const char *dir,int n,int p,const char *mess,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(n);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscTraceBackErrorHandler" 
/*@C

   PetscTraceBackErrorHandler - Default error handler routine that generates
   a traceback on error detection.

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNCT__)
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

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler()
 @*/
int PetscTraceBackErrorHandler(int line,const char *fun,const char* file,const char *dir,int n,int p,const char *mess,void *ctx)
{
  PetscLogDouble    mem,rss;
  int               rank;
  PetscTruth        flg1,flg2;

  PetscFunctionBegin;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
  if (p == 1) {
    if (n == PETSC_ERR_MEM) {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Out of memory. This could be due to allocating\n",rank);
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   too large an object or bleeding by not properly\n",rank);
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   destroying unneeded objects.\n",rank);
      PetscTrSpace(&mem,PETSC_NULL,PETSC_NULL);
      PetscGetResidentSetSize(&rss);
      PetscOptionsHasName(PETSC_NULL,"-trdump",&flg1);
      PetscOptionsHasName(PETSC_NULL,"-trmalloc_log",&flg2);
      if (flg2) {
        PetscTrLogDump(stdout);
      } else if (flg1) {
        (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
        PetscTrDump(stdout);
      } else {
        (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
        (*PetscErrorPrintf)("[%d]PETSC ERROR:   Try running with -trdump or -trmalloc_log for info.\n",rank);
      }
    } else {
        char *text;
        PetscErrorMessage(n,&text,PETSC_NULL);
        if (text) (*PetscErrorPrintf)("[%d]PETSC ERROR:   %s!\n",rank,text);
    }
    if (mess) {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   %s!\n",rank,mess);
    }
  }
  PetscFunctionReturn(n);
}

