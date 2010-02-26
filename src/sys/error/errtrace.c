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

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfInitialize"
/*
   Initializes arch, hostname, username,date so that system calls do NOT need
   to be made during the error handler.
*/
PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfInitialize()
{
  PetscErrorCode ierr;
  PetscTruth     use_stdout = PETSC_FALSE,use_none = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,64);CHKERRQ(ierr);
  ierr = PetscGetUserName(username,16);CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetDate(date,64);CHKERRQ(ierr);
  ierr = PetscGetVersion(version,256);CHKERRQ(ierr);

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-error_output_stdout",&use_stdout,PETSC_NULL);CHKERRQ(ierr);
  if (use_stdout) {
    PETSC_STDERR = PETSC_STDOUT;
  }
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-error_output_none",&use_none,PETSC_NULL);CHKERRQ(ierr);
  if (use_none) {
    PetscErrorPrintf = PetscErrorPrintfNone;
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
  va_list           Argp;
  static PetscTruth PetscErrorPrintfCalled = PETSC_FALSE;

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
  PetscTruth        flg1 = PETSC_FALSE,flg2 = PETSC_FALSE;

  PetscFunctionBegin;

  if (p == 1) {
    (*PetscErrorPrintf)("--------------------- Error Message ------------------------------------\n");
    if (n == PETSC_ERR_MEM) {
      (*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
      (*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
      (*PetscErrorPrintf)("destroying unneeded objects.\n");
      PetscMallocGetCurrentUsage(&mem);
      PetscMemoryGetCurrentUsage(&rss);
      PetscOptionsGetTruth(PETSC_NULL,"-malloc_dump",&flg1,PETSC_NULL);
      PetscOptionsGetTruth(PETSC_NULL,"-malloc_log",&flg2,PETSC_NULL);
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

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
#undef __FUNCT__  
#define __FUNCT__ "PetscTraceBackErrorHandlerCxx"
/*@C

   PetscTraceBackErrorHandlerCxx - Default error handler routine that generates
   a traceback on error detection.

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNCT__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  n - the generic error number
.  p - specific error number
-  msg - The message stream

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERROR, which has 
   the calling sequence
$     SETERROR(number,p,mess)

   Concepts: error handler^traceback
   Concepts: traceback^generating

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler()
 @*/
void PETSC_DLLEXPORT PetscTraceBackErrorHandlerCxx(int line,const char *fun,const char* file,const char *dir,PetscErrorCode n,int p, std::ostringstream& msg)
{
  if (p == 1) {
    PetscLogDouble mem, rss;
    PetscTruth     flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;

    msg << "--------------------- Error Message ------------------------------------" << std::endl;
    if (n == PETSC_ERR_MEM) {
      msg << "Out of memory. This could be due to allocating" << std::endl;
      msg << "too large an object or bleeding by not properly" << std::endl;
      msg << "destroying unneeded objects." << std::endl;
      PetscMallocGetCurrentUsage(&mem);
      PetscMemoryGetCurrentUsage(&rss);
      PetscOptionsGetTruth(PETSC_NULL,"-malloc_dump",&flg1,PETSC_NULL);
      PetscOptionsGetTruth(PETSC_NULL,"-malloc_log",&flg2,PETSC_NULL);
      if (flg2) {
        //PetscMallocDumpLog(stdout);
        msg << "Option -malloc_log does not work in C++." << std::endl;
      } else {
        msg << "Memory allocated " << mem << " Memory used by process " << rss << std::endl;
        if (flg1) {
          //PetscMallocDump(stdout);
          msg << "Option -malloc_dump does not work in C++." << std::endl;
        } else {
          msg << "Try running with -malloc_dump or -malloc_log for info." << std::endl;
        }
      }
    } else {
      const char *text;

      PetscErrorMessage(n,&text,PETSC_NULL);
      if (text) {msg << text << "!" << std::endl;}
    }
    msg << "------------------------------------------------------------------------" << std::endl;
    msg << version << std::endl;
    msg << "See docs/changes/index.html for recent updates." << std::endl;
    msg << "See docs/faq.html for hints about trouble shooting." << std::endl;
    msg << "See docs/index.html for manual pages." << std::endl;
    msg << "------------------------------------------------------------------------" << std::endl;
    if (PetscErrorPrintfInitializeCalled) {
      msg << pname << " on a " << arch << " named " << hostname << " by " << username << " " << date << std::endl;
    }
    msg << "Libraries linked from " << PETSC_LIB_DIR << std::endl;
    msg << "Configure run at " << petscconfigureruntime << std::endl;
    msg << "Configure options " << petscconfigureoptions << std::endl;
    msg << "------------------------------------------------------------------------" << std::endl;
  } else {
    msg << fun<<"() line " << line << " in " << dir << file << std::endl;
  }
}
#endif
