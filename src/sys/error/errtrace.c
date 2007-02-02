#define PETSC_DLL

#include "petscsys.h"        /*I "petscsys.h" I*/
#include "petscconfiginfo.h"

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
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscMPIAbortErrorHandler()

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler(), PetscTraceBackErrorHandler()
 @*/
PetscErrorCode PETSC_DLLEXPORT PetscIgnoreErrorHandler(int line,const char *fun,const char* file,const char *dir,PetscErrorCode n,int p,const char *mess,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(n);
}

/* ---------------------------------------------------------------------------------------*/

static char  arch[10],hostname[64],username[16],pname[PETSC_MAX_PATH_LEN],date[64];
static PetscTruth PetscErrorPrintfInitializeCalled = PETSC_FALSE;
static char version[256];
static FILE *PetscErrorPrintfFILE = stdout;

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfInitialize"
/*
   Initializes arch, hostname, username,date so that system calls do NOT need
   to be made during the error handler.
*/
PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfInitialize()
{
  PetscErrorCode ierr;
  PetscTruth     use_stderr;

  PetscFunctionBegin;
  ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,64);CHKERRQ(ierr);
  ierr = PetscGetUserName(username,16);CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetDate(date,64);CHKERRQ(ierr);
  ierr = PetscGetVersion(&version,256);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-error_output_stderr",&use_stderr);CHKERRQ(ierr);
  if (use_stderr) {
      PetscErrorPrintfFILE = stderr;
    } else {
      PetscErrorPrintfFILE = PETSC_STDOUT;
    }
  PetscErrorPrintfInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfNone" 
PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfNone(const char format[],...)
{
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfDefault" 
PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfDefault(const char format[],...)
{
  va_list            Argp;
  static  PetscTruth PetscErrorPrintfCalled    = PETSC_FALSE;

  /*
      This function does not call PetscFunctionBegin and PetscFunctionReturn() because
    it may be called by PetscStackView().

      This function does not do error checking because it is called by the error handlers.
  */

  if (!PetscErrorPrintfCalled) {
    PetscErrorPrintfCalled    = PETSC_TRUE;

    /*
        On the SGI machines and Cray T3E, if errors are generated  "simultaneously" by
      different processors, the messages are printed all jumbled up; to try to 
      prevent this we have each processor wait based on their rank
    */
#if defined(PETSC_CAN_SLEEP_AFTER_ERROR)
    {
      PetscMPIInt rank;
      if (PetscGlobalRank > 8) rank = 8; else rank = PetscGlobalRank;
      PetscSleep(rank);
    }
#endif
  }
    
  PetscFPrintf(PETSC_COMM_SELF,PetscErrorPrintfFILE,"[%d]PETSC ERROR: ",PetscGlobalRank);
  va_start(Argp,format);
  PetscVFPrintf(PetscErrorPrintfFILE,format,Argp);
  va_end(Argp);

  return 0;
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
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscMPIAbortErrorHandler()

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler()
 @*/
PetscErrorCode PETSC_DLLEXPORT PetscTraceBackErrorHandler(int line,const char *fun,const char* file,const char *dir,PetscErrorCode n,int p,const char *mess,void *ctx)
{
  PetscLogDouble    mem,rss;
  PetscTruth        flg1,flg2;

  PetscFunctionBegin;

  if (p == 1) {
    (*PetscErrorPrintf)("--------------------- Error Message ------------------------------------\n");
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
    (*PetscErrorPrintf)("------------------------------------------------------------------------\n");
    (*PetscErrorPrintf)("%s\n",version);
    (*PetscErrorPrintf)("See docs/changes/index.html for recent updates.\n");
    (*PetscErrorPrintf)("See docs/faq.html for hints about trouble shooting.\n");
    (*PetscErrorPrintf)("See docs/index.html for manual pages.\n");
    (*PetscErrorPrintf)("------------------------------------------------------------------------\n");
    if (PetscErrorPrintfInitializeCalled) {
      (*PetscErrorPrintf)("%s on a %s named %s by %s %s\n",pname,arch,hostname,username,date);
    }
    (*PetscErrorPrintf)("Libraries linked from %s\n",PETSC_LIB_DIR);
    (*PetscErrorPrintf)("Configure run at %s\n",petscconfigureruntime);
    (*PetscErrorPrintf)("Configure options %s\n",petscconfigureoptions);
    (*PetscErrorPrintf)("------------------------------------------------------------------------\n");
  }


  /* first line in stack trace? */
  (*PetscErrorPrintf)("%s() line %d in %s%s\n",fun,line,dir,file);


  PetscFunctionReturn(n);
}

