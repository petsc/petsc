#define PETSC_DLL
/*
      Utilites routines to add simple ASCII IO capability.
*/
#include "src/sys/src/fileio/mprint.h"
#include "petscconfiginfo.h"
/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
extern FILE *petsc_history;
/*
     Allows one to overwrite where standard out is sent. For example
     PETSC_STDOUTPUT = fopen("/dev/ttyXX","w") will cause all standard out
     writes to go to terminal XX; assuming you have write permission there
*/
FILE *PETSC_STDOUT = stdout;

#undef __FUNCT__  
#define __FUNCT__ "PetscFormatConvert"
PetscErrorCode PETSC_DLLEXPORT PetscFormatConvert(const char *format,char *newformat)
{
  PetscInt i = 0,j = 0;

  while (format[i] && i < 8*1024-1) {
    if (format[i] == '%' && format[i+1] == 'D') {
      newformat[j++] = '%';
#if defined(PETSC_USE_32BIT_INT)
      newformat[j++] = 'd';
#else
      newformat[j++] = 'l';
      newformat[j++] = 'l';
      newformat[j++] = 'd';
#endif
      i += 2;
    } else if (format[i] == '%' && format[i+1] >= '1' && format[i+1] <= '9' && format[i+2] == 'D') {
      newformat[j++] = '%';
      newformat[j++] = format[i+1];
#if defined(PETSC_USE_32BIT_INT)
      newformat[j++] = 'd';
#else
      newformat[j++] = 'l';
      newformat[j++] = 'l';
      newformat[j++] = 'd';
#endif
      i += 3;
    }else {
      newformat[j++] = format[i++];
    }
  }
  newformat[j] = 0;
  return 0;
}
 
#undef __FUNCT__  
#define __FUNCT__ "PetscVSNPrintf"
/* 
   No error handling because may be called by error handler
*/
PetscErrorCode PETSC_DLLEXPORT PetscVSNPrintf(char *str,size_t len,const char *format,va_list Argp)
{
  /* no malloc since may be called by error handler */
  char     newformat[8*1024];
 
  PetscFormatConvert(format,newformat); 
#if defined(PETSC_HAVE_VPRINTF_CHAR)
  vsprintf(str,newformat,(char *)Argp);
#else
  vsprintf(str,newformat,Argp);
#endif
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscVFPrintf"
/* 
   All PETSc standard out and error messages are sent through this function; so, in theory, this can
   can be replaced with something that does not simply write to a file. 

   Note: For error messages this may be called by a process, for regular standard out it is
   called only by process 0 of a given communicator

   No error handling because may be called by error handler
*/
PetscErrorCode PETSC_DLLEXPORT PetscVFPrintf(FILE *fd,const char *format,va_list Argp)
{
  /* no malloc since may be called by error handler */
  char     newformat[8*1024];
 
  PetscFormatConvert(format,newformat); 
#if defined(PETSC_HAVE_VPRINTF_CHAR)
  vfprintf(fd,newformat,(char *)Argp);
#else
  vfprintf(fd,newformat,Argp);
  fflush(fd);
#endif
  return 0;
}

/* ----------------------------------------------------------------------- */

PrintfQueue queue       = 0,queuebase = 0;
int         queuelength = 0;
FILE        *queuefile  = PETSC_NULL;

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedPrintf" 
/*@C
    PetscSynchronizedPrintf - Prints synchronized output from several processors.
    Output of the first processor is followed by that of the second, etc.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

   Level: intermediate

    Notes:
    REQUIRES a intervening call to PetscSynchronizedFlush() for the information 
    from all the processors to be printed.

    Fortran Note:
    The call sequence is PetscSynchronizedPrintf(PetscViewer, character(*), PetscErrorCode ierr) from Fortran. 
    That is, you can only pass a single character string from Fortran.

    The length of the formatted message cannot exceed QUEUESTRINGSIZE characters.

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), 
          PetscPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedPrintf(MPI_Comm comm,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* First processor prints immediately to stdout */
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
    ierr = PetscVFPrintf(PETSC_STDOUT,format,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      ierr = PetscVFPrintf(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    va_list     Argp;
    PrintfQueue next;

    ierr = PetscNew(struct _PrintfQueue,&next);CHKERRQ(ierr);
    if (queue) {queue->next = next; queue = next; queue->next = 0;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start(Argp,format);
    ierr = PetscMemzero(next->string,QUEUESTRINGSIZE);CHKERRQ(ierr);
    ierr = PetscVSNPrintf(next->string,QUEUESTRINGSIZE,format,Argp);CHKERRQ(ierr);
    va_end(Argp);
  }
    
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedFPrintf" 
/*@C
    PetscSynchronizedFPrintf - Prints synchronized output to the specified file from
    several processors.  Output of the first processor is followed by that of the 
    second, etc.

    Not Collective

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   format - the usual printf() format string 

    Level: intermediate

    Notes:
    REQUIRES a intervening call to PetscSynchronizedFlush() for the information 
    from all the processors to be printed.

    The length of the formatted message cannot exceed QUEUESTRINGSIZE characters.

    Contributed by: Matthew Knepley

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(), PetscFPrintf(),
          PetscFOpen(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIPrintf()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedFPrintf(MPI_Comm comm,FILE* fp,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* First processor prints immediately to fp */
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
    ierr = PetscVFPrintf(fp,format,Argp);CHKERRQ(ierr);
    queuefile = fp;
    if (petsc_history) {
      ierr = PetscVFPrintf(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    va_list     Argp;
    PrintfQueue next;
    ierr = PetscNew(struct _PrintfQueue,&next);CHKERRQ(ierr);
    if (queue) {queue->next = next; queue = next; queue->next = 0;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start(Argp,format);
    ierr = PetscMemzero(next->string,QUEUESTRINGSIZE);CHKERRQ(ierr);
    ierr = PetscVSNPrintf(next->string,QUEUESTRINGSIZE,format,Argp);CHKERRQ(ierr);
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedFlush" 
/*@C
    PetscSynchronizedFlush - Flushes to the screen output from all processors 
    involved in previous PetscSynchronizedPrintf() calls.

    Collective on MPI_Comm

    Input Parameters:
.   comm - the communicator

    Level: intermediate

    Notes:
    Usage of PetscSynchronizedPrintf() and PetscSynchronizedFPrintf() with
    different MPI communicators REQUIRES an intervening call to PetscSynchronizedFlush().

.seealso: PetscSynchronizedPrintf(), PetscFPrintf(), PetscPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIISynchronizedPrintf()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedFlush(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag,i,j,n;
  char           message[QUEUESTRINGSIZE];
  MPI_Status     status;
  FILE           *fd;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* First processor waits for messages from all other processors */
  if (!rank) {
    if (queuefile) {
      fd = queuefile;
    } else {
      fd = PETSC_STDOUT;
    }
    for (i=1; i<size; i++) {
      ierr = MPI_Recv(&n,1,MPI_INT,i,tag,comm,&status);CHKERRQ(ierr);
      for (j=0; j<n; j++) {
        ierr = MPI_Recv(message,QUEUESTRINGSIZE,MPI_CHAR,i,tag,comm,&status);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm,fd,"%s",message);
      }
    }
    queuefile = PETSC_NULL;
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = queuebase,previous;

    ierr = MPI_Send(&queuelength,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
    for (i=0; i<queuelength; i++) {
      ierr     = MPI_Send(next->string,QUEUESTRINGSIZE,MPI_CHAR,0,tag,comm);CHKERRQ(ierr);
      previous = next; 
      next     = next->next;
      ierr     = PetscFree(previous);CHKERRQ(ierr);
    }
    queue       = 0;
    queuelength = 0;
  }
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PetscFPrintf" 
/*@C
    PetscFPrintf - Prints to a file, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   format - the usual printf() format string 

    Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

   Concepts: printing^in parallel
   Concepts: printf^in parallel

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIISynchronizedPrintf(), PetscSynchronizedFlush()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscFPrintf(MPI_Comm comm,FILE* fd,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
    ierr = PetscVFPrintf(fd,format,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      ierr = PetscVFPrintf(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPrintf" 
/*@C
    PetscPrintf - Prints to standard out, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

   Level: intermediate

    Fortran Note:
    The call sequence is PetscPrintf(PetscViewer, character(*), PetscErrorCode ierr) from Fortran. 
    That is, you can only pass a single character string from Fortran.

   Notes: %A is replace with %g unless the value is < 1.e-12 when it is 
          replaced with < 1.e-12

   Concepts: printing^in parallel
   Concepts: printf^in parallel

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscPrintf(MPI_Comm comm,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  size_t         len;
  char           *nformat,*sub1,*sub2;
  PetscReal      value;

  PetscFunctionBegin;
  if (!comm) comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);

    ierr = PetscStrstr(format,"%A",&sub1);CHKERRQ(ierr);
    if (sub1) {
      ierr = PetscStrstr(format,"%",&sub2);CHKERRQ(ierr);
      if (sub1 != sub2) SETERRQ(PETSC_ERR_ARG_WRONG,"%%A format must be first in format string");
      ierr    = PetscStrlen(format,&len);CHKERRQ(ierr);
      ierr    = PetscMalloc((len+16)*sizeof(char),&nformat);CHKERRQ(ierr);
      ierr    = PetscStrcpy(nformat,format);CHKERRQ(ierr);
      ierr    = PetscStrstr(nformat,"%",&sub2);CHKERRQ(ierr);
      sub2[0] = 0;
      value   = (double)va_arg(Argp,double);
      if (PetscAbsReal(value) < 1.e-12) {
        ierr    = PetscStrcat(nformat,"< 1.e-12");CHKERRQ(ierr);
      } else {
        ierr    = PetscStrcat(nformat,"%g");CHKERRQ(ierr);
        va_end(Argp);
        va_start(Argp,format);
      }
      ierr    = PetscStrcat(nformat,sub1+2);CHKERRQ(ierr);
    } else {
      nformat = (char*)format;
    }
    ierr = PetscVFPrintf(PETSC_STDOUT,nformat,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      ierr = PetscVFPrintf(petsc_history,nformat,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
    if (sub1) {ierr = PetscFree(nformat);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscHelpPrintfDefault" 
PetscErrorCode PETSC_DLLEXPORT PetscHelpPrintfDefault(MPI_Comm comm,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (!comm) comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
    ierr = PetscVFPrintf(PETSC_STDOUT,format,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      ierr = PetscVFPrintf(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/

static char  arch[10],hostname[64],username[16],pname[PETSC_MAX_PATH_LEN],date[64];
static PetscTruth PetscErrorPrintfInitializeCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfInitialize"
/*
   Initializes arch, hostname, username,date so that system calls do NOT need
   to be made during the error handler.
*/
PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfInitialize()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,64);CHKERRQ(ierr);
  ierr = PetscGetUserName(username,16);CHKERRQ(ierr);
  ierr = PetscGetProgramName(pname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscGetDate(date,64);CHKERRQ(ierr);
  PetscErrorPrintfInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscErrorPrintfDefault" 
PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfDefault(const char format[],...)
{
  va_list            Argp;
  static  PetscTruth PetscErrorPrintfCalled    = PETSC_FALSE;
  static  PetscTruth InPetscErrorPrintfDefault = PETSC_FALSE;
  static  FILE       *fd;
  char               version[256];
  PetscErrorCode     ierr;

  /*
      InPetscErrorPrintfDefault is used to prevent the error handler called (potentially)
     from PetscSleep(), PetscGetArchName(), ... below from printing its own error message.
  */

  /*
      This function does not call PetscFunctionBegin and PetscFunctionReturn() because
    it may be called by PetscStackView().

      This function does not do error checking because it is called by the error handlers.
  */

  if (!PetscErrorPrintfCalled) {
    PetscTruth use_stderr;

    PetscErrorPrintfCalled    = PETSC_TRUE;
    InPetscErrorPrintfDefault = PETSC_TRUE;

    PetscOptionsHasName(PETSC_NULL,"-error_output_stderr",&use_stderr);
    if (use_stderr) {
      fd = stderr;
    } else {
      fd = PETSC_STDOUT;
    }

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
    
    ierr = PetscGetVersion(&version);CHKERRQ(ierr);

    PetscFPrintf(PETSC_COMM_SELF,fd,"------------------------------------------------------------------------\n");
    PetscFPrintf(PETSC_COMM_SELF,fd,"%s\n",version);
    PetscFPrintf(PETSC_COMM_SELF,fd,"See docs/changes/index.html for recent updates.\n");
    PetscFPrintf(PETSC_COMM_SELF,fd,"See docs/faq.html for hints about trouble shooting.\n");
    PetscFPrintf(PETSC_COMM_SELF,fd,"See docs/index.html for manual pages.\n");
    PetscFPrintf(PETSC_COMM_SELF,fd,"------------------------------------------------------------------------\n");
    if (PetscErrorPrintfInitializeCalled) {
      PetscFPrintf(PETSC_COMM_SELF,fd,"%s on a %s named %s by %s %s\n",pname,arch,hostname,username,date);
    }
    PetscFPrintf(PETSC_COMM_SELF,fd,"Libraries linked from %s\n",PETSC_LIB_DIR);
    PetscFPrintf(PETSC_COMM_SELF,fd,"Configure run at %s\n",petscconfigureruntime);
    PetscFPrintf(PETSC_COMM_SELF,fd,"Configure options %s\n",petscconfigureoptions);
    PetscFPrintf(PETSC_COMM_SELF,fd,"------------------------------------------------------------------------\n");
    InPetscErrorPrintfDefault = PETSC_FALSE;
  }

  if (!InPetscErrorPrintfDefault) {
    PetscFPrintf(PETSC_COMM_SELF,fd,"[%d]PETSC ERROR: ",PetscGlobalRank);
    va_start(Argp,format);
    PetscVFPrintf(fd,format,Argp);
    va_end(Argp);
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedFGets" 
/*@C
    PetscSynchronizedFGets - Several processors all get the same line from a file.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   len - the length of the output buffer

    Output Parameter:
.   string - the line read from the file

    Level: intermediate

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(), 
          PetscFOpen(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIPrintf()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedFGets(MPI_Comm comm,FILE* fp,size_t len,char string[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  if (!rank) {
    fgets(string,len,fp);
  }
  ierr = MPI_Bcast(string,len,MPI_BYTE,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
