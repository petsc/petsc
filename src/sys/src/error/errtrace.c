#define PETSC_DLL

#include "petsc.h"           /*I "petsc.h" I*/


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
PetscErrorCode PETSC_DLLEXPORT PetscIgnoreErrorHandler(int line,const char *fun,const char* file,const char *dir,int n,int p,const char *mess,void *ctx)
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
PetscErrorCode PETSC_DLLEXPORT PetscTraceBackErrorHandler(int line,const char *fun,const char* file,const char *dir,int n,int p,const char *mess,void *ctx)
{
  PetscLogDouble    mem,rss;
  PetscTruth        flg1,flg2;

  PetscFunctionBegin;

  (*PetscErrorPrintf)("%s() line %d in %s%s\n",fun,line,dir,file);
  if (p == 1) {
    if (n == PETSC_ERR_MEM) {
      (*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
      (*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
      (*PetscErrorPrintf)("destroying unneeded objects.\n");
      PetscMallocGetCurrentUsage(&mem);
      PetscMemoryGetCurrentUsage(&rss);
      PetscOptionsHasName(PETSC_NULL,"-malloc_dump",&flg1);
      PetscOptionsHasName(PETSC_NULL,"-malloc_log",&flg2);
      if (flg2) {
        PetscMallocDumpLog(stdout);
      } else {
        (*PetscErrorPrintf)("Memory allocated %D Memory used by process %D\n",(PetscInt)mem,(PetscInt)rss);
        if (flg1) {
          PetscMallocDump(stdout);
        } else {
          (*PetscErrorPrintf)("Try running with -malloc_dump or -malloc_log for info.\n");
        }
      }
    } else {
        const char *text;
        PetscErrorMessage(n,&text,PETSC_NULL);
        if (text) (*PetscErrorPrintf)("%s!\n",text);
    }
    if (mess) {
      (*PetscErrorPrintf)("%s!\n",mess);
    }
  }
  PetscFunctionReturn(n);
}

