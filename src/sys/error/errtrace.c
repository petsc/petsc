
#include <petscsys.h>        /*I "petscsys.h" I*/
#include <petscconfiginfo.h>

#undef __FUNCT__  
#define __FUNCT__ "PetscIgnoreErrorHandler" 
/*@C
   PetscIgnoreErrorHandler - Ignores the error, allows program to continue as if error did not occure

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
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
$     SETERRQ(comm,number,p,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscMPIAbortErrorHandler()

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler(), PetscTraceBackErrorHandler()
 @*/
PetscErrorCode  PetscIgnoreErrorHandler(MPI_Comm comm,int line,const char *fun,const char* file,const char *dir,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(n);
}

/* ---------------------------------------------------------------------------------------*/

static char  arch[10],hostname[64],username[16],pname[PETSC_MAX_PATH_LEN],date[64];
static PetscBool  PetscErrorPrintfInitializeCalled = PETSC_FALSE;
static char version[256];

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfInitialize"
/*
   Initializes arch, hostname, username,date so that system calls do NOT need
   to be made during the error handler.
*/
PetscErrorCode  PetscErrorPrintfInitialize()
{
  PetscErrorCode ierr;
  PetscBool      use_stdout = PETSC_FALSE,use_none = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,64);CHKERRQ(ierr);
  ierr = PetscGetUserName(username,16);CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetDate(date,64);CHKERRQ(ierr);
  ierr = PetscGetVersion(version,256);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-error_output_stdout",&use_stdout,PETSC_NULL);CHKERRQ(ierr);
  if (use_stdout) {
    PETSC_STDERR = PETSC_STDOUT;
  }
  ierr = PetscOptionsGetBool(PETSC_NULL,"-error_output_none",&use_none,PETSC_NULL);CHKERRQ(ierr);
  if (use_none) {
    PetscErrorPrintf = PetscErrorPrintfNone;
  }
  PetscErrorPrintfInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfNone" 
PetscErrorCode  PetscErrorPrintfNone(const char format[],...)
{
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfDefault" 
PetscErrorCode  PetscErrorPrintfDefault(const char format[],...)
{
  va_list           Argp;
  static PetscBool  PetscErrorPrintfCalled = PETSC_FALSE;

  /*
      This function does not call PetscFunctionBegin and PetscFunctionReturn() because
    it may be called by PetscStackView().

      This function does not do error checking because it is called by the error handlers.
  */

  if (!PetscErrorPrintfCalled) {
    PetscErrorPrintfCalled = PETSC_TRUE;

    /*
        On the SGI machines and Cray T3E, if errors are generated  "simultaneously" by
      different processors, the messages are printed all jumbled up; to try to 
      prevent this we have each processor wait based on their rank
    */
#if defined(PETSC_CAN_SLEEP_AFTER_ERROR)
    {
      PetscMPIInt rank;
      if (PetscGlobalRank > 8) rank = 8; else rank = PetscGlobalRank;
      PetscSleep((PetscReal)rank);
    }
#endif
  }
    
  PetscFPrintf(PETSC_COMM_SELF,PETSC_STDERR,"[%d]PETSC ERROR: ",PetscGlobalRank);
  va_start(Argp,format);
  (*PetscVFPrintf)(PETSC_STDERR,format,Argp);
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
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNCT__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - PETSC_ERROR_INITIAL if this is the first call the the error handler, otherwise PETSC_ERROR_REPEAT
-  ctx - error handler context

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(comm,number,n,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscMPIAbortErrorHandler()

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler()
 @*/
PetscErrorCode  PetscTraceBackErrorHandler(MPI_Comm comm,int line,const char *fun,const char* file,const char *dir,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscLogDouble    mem,rss;
  PetscBool         flg1 = PETSC_FALSE,flg2 = PETSC_FALSE;
  PetscMPIInt       rank = 0;

  PetscFunctionBegin;
  if (comm != PETSC_COMM_SELF) {
    MPI_Comm_rank(comm,&rank);
  }
  if (!rank) {
    if (p == PETSC_ERROR_INITIAL) {
      (*PetscErrorPrintf)("--------------------- Error Message ------------------------------------\n");
      if (n == PETSC_ERR_MEM) {
	(*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
	(*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
	(*PetscErrorPrintf)("destroying unneeded objects.\n");
	PetscMallocGetCurrentUsage(&mem);
	PetscMemoryGetCurrentUsage(&rss);
	PetscOptionsGetBool(PETSC_NULL,"-malloc_dump",&flg1,PETSC_NULL);
	PetscOptionsGetBool(PETSC_NULL,"-malloc_log",&flg2,PETSC_NULL);
	if (flg2) {
	  PetscMallocDumpLog(stdout);
	} else {
	  (*PetscErrorPrintf)("Memory allocated %.0f Memory used by process %.0f\n",mem,rss);
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
    /* print line of stack trace */
    (*PetscErrorPrintf)("%s() line %d in %s%s\n",fun,line,dir,file);
  } else {
    /* do not print error messages since process 0 will print them, sleep before aborting so will not accidently kill process 0*/
    PetscSleep(10.0);
    abort();
  }
  PetscFunctionReturn(n);
}

