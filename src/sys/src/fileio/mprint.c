/*$Id: mprint.c,v 1.48 2000/05/04 16:24:43 bsmith Exp bsmith $*/
/*
      Utilites routines to add simple ASCII IO capability.
*/
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "petscfix.h"

/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
extern FILE *petsc_history;

/* ----------------------------------------------------------------------- */

typedef struct _PrintfQueue *PrintfQueue;
struct _PrintfQueue {
  char        string[1024];
  PrintfQueue next;
};
PrintfQueue queue       = 0,queuebase = 0;
int         queuelength = 0;
FILE        *queuefile  = PETSC_NULL;

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSynchronizedPrintf" 
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

    The length of the formatted message cannot exceed 1024 charactors.

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), 
          PetscPrintf(), ViewerASCIIPrintf(), ViewerASCIISynchronizedPrintf()
@*/
int PetscSynchronizedPrintf(MPI_Comm comm,const char format[],...)
{
  int ierr,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* First processor prints immediately to stdout */
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VFPRINTF_CHAR)
    vfprintf(stdout,format,(char*)Argp);
#else
    vfprintf(stdout,format,Argp);
#endif
    fflush(stdout);
    if (petsc_history) {
#if defined(PETSC_HAVE_VFPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    int         len;
    va_list     Argp;
    PrintfQueue next = PetscNew(struct _PrintfQueue);CHKPTRQ(next);
    if (queue) {queue->next = next; queue = next;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VFPRINTF_CHAR)
    vsprintf(next->string,format,(char *)Argp);
#else
    vsprintf(next->string,format,Argp);
#endif
    va_end(Argp);
    ierr = PetscStrlen(next->string,&len);CHKERRQ(ierr);
    if (len > 1024) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Formatted string longer than 1024 bytes");
  }
    
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSynchronizedFPrintf" 
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

    The length of the formatted message cannot exceed 1024 charactors.

    Contributed by: Matthew Knepley

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(), PetscFPrintf(),
          PetscFOpen(), ViewerASCIISynchronizedPrintf(), ViewerASCIIPrintf()

@*/
int PetscSynchronizedFPrintf(MPI_Comm comm,FILE* fp,const char format[],...)
{
  int ierr,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* First processor prints immediately to fp */
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(fp,format,(char*)Argp);
#else
    vfprintf(fp,format,Argp);
#endif
    fflush(fp);
    queuefile = fp;
    if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    int         len;
    va_list     Argp;
    PrintfQueue next = PetscNew(struct _PrintfQueue);CHKPTRQ(next);
    if (queue) {queue->next = next; queue = next;}
    else       {queuebase   = queue = next;}
    queuelength++;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vsprintf(next->string,format,(char *)Argp);
#else
    vsprintf(next->string,format,Argp);
#endif
    va_end(Argp);
    ierr = PetscStrlen(next->string,&len);CHKERRQ(ierr);
    if (len > 1024) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Formatted string longer then 1024 bytes");
  }
    
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSynchronizedFlush" 
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

.seealso: PetscSynchronizedPrintf(), PetscFPrintf(), PetscPrintf(), ViewerASCIIPrintf(),
          ViewerASCIISynchronizedPrintf()
@*/
int PetscSynchronizedFlush(MPI_Comm comm)
{
  int        rank,size,i,j,n,tag,ierr;
  char       message[1024];
  MPI_Status status;
  FILE       *fd;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  /* First processor waits for messages from all other processors */
  if (!rank) {
    if (queuefile) {
      fd = queuefile;
    } else {
      fd = stdout;
    }
    for (i=1; i<size; i++) {
      ierr = MPI_Recv(&n,1,MPI_INT,i,tag,comm,&status);CHKERRQ(ierr);
      for (j=0; j<n; j++) {
        ierr = MPI_Recv(message,1024,MPI_CHAR,i,tag,comm,&status);CHKERRQ(ierr);
        fprintf(fd,"%s",message);
        if (petsc_history) {
          fprintf(petsc_history,"%s",message);
        }
      }
    }
    fflush(fd);
    if (petsc_history) fflush(petsc_history);
    queuefile = PETSC_NULL;
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = queuebase,previous;

    ierr = MPI_Send(&queuelength,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
    for (i=0; i<queuelength; i++) {
      ierr     = MPI_Send(next->string,1024,MPI_CHAR,0,tag,comm);CHKERRQ(ierr);
      previous = next; 
      next     = next->next;
      ierr     = PetscFree(previous);CHKERRQ(ierr);
    }
    queue       = 0;
    queuelength = 0;
  }
  ierr = PetscCommRestoreNewTag(comm,&tag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscFPrintf" 
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

.keywords: parallel, fprintf

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), ViewerASCIIPrintf(),
          ViewerASCIISynchronizedPrintf(), PetscSynchronizedFlush()
@*/
int PetscFPrintf(MPI_Comm comm,FILE* fd,const char format[],...)
{
  int rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(fd,format,(char*)Argp);
#else
    vfprintf(fd,format,Argp);
#endif
    fflush(fd);
    if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscPrintf"></a>*/"PetscPrintf" 
/*@C
    PetscPrintf - Prints to standard out, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

   Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

   Notes: %A is replace with %g unless the value is < 1.e-12 when it is 
          replaced with < 1.e-12

.keywords: parallel, printf

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
int PetscPrintf(MPI_Comm comm,const char format[],...)
{
  int       rank,ierr,len;
  char      *nformat,*sub1,*sub2;
  PetscReal value;

  PetscFunctionBegin;
  if (!comm) comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);

    ierr = PetscStrstr(format,"%A",&sub1);CHKERRQ(ierr);
    if (sub1) {
      ierr = PetscStrstr(format,"%",&sub2);CHKERRQ(ierr);
      if (sub1 != sub2) SETERRQ(1,1,"%A format must be first in format string");
      ierr    = PetscStrlen(format,&len);CHKERRQ(ierr);
      nformat = (char*)PetscMalloc((len+16)*sizeof(char));CHKPTRQ(nformat);
      ierr    = PetscStrcpy(nformat,format);CHKERRQ(ierr);
      ierr    = PetscStrstr(nformat,"%",&sub2);CHKERRQ(ierr);
      sub2[0] = 0;
      value   = (double)va_arg(Argp,double);
      if (PetscAbsDouble(value) < 1.e-12) {
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
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(stdout,nformat,(char *)Argp);
#else
    vfprintf(stdout,nformat,Argp);
#endif
    fflush(stdout);
    if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,nformat,(char *)Argp);
#else
      vfprintf(petsc_history,nformat,Argp);
#endif
      fflush(petsc_history);
    }
    va_end(Argp);
    if (sub1) {ierr = PetscFree(nformat);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscHelpPrintfDefault" 
/*@C
    PetscHelpPrintfDefault - Prints to standard out, only from the first
    processor in the communicator.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, printf

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
int PetscHelpPrintfDefault(MPI_Comm comm,const char format[],...)
{
  int rank,ierr;

  PetscFunctionBegin;
  if (!comm) comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    va_list Argp;
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(stdout,format,(char *)Argp);
#else
    vfprintf(stdout,format,Argp);
#endif
    fflush(stdout);
    if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
      vfprintf(petsc_history,format,(char *)Argp);
#else
      vfprintf(petsc_history,format,Argp);
#endif
      fflush(petsc_history);
    }
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscErrorPrintfDefault" 
/*@C
    PetscErrorPrintfDefault - Prints error messages.

    Not Collective

    Input Parameters:
.   format - the usual printf() format string 

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: parallel, printf

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
int PetscErrorPrintfDefault(const char format[],...)
{
  va_list     Argp;
  static  int PetscErrorPrintfCalled = 0;
  static  int InPetscErrorPrintfDefault = 0;

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
    int  rank;
    char arch[10],hostname[64],username[16],pname[256],date[64];

    PetscErrorPrintfCalled    = 1;
    InPetscErrorPrintfDefault = 1;

    /*
        On the SGI machines and Cray T3E, if errors are generated  "simultaneously" by
      different processors, the messages are printed all jumbled up; to try to 
      prevent this we have each processor wait based on their rank
    */
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (rank > 8) rank = 8;
#if defined(PETSC_CAN_SLEEP_AFTER_ERROR)
    PetscSleep(rank);
#endif

    /* Cannot do error checking on these calls because we are called by error handler */
    PetscGetArchType(arch,10);
    PetscGetHostName(hostname,64);
    PetscGetUserName(username,16);
    PetscGetProgramName(pname,256);
    PetscGetInitialDate(date,64);
    fprintf(stderr,"--------------------------------------------\
------------------------------\n");
    fprintf(stderr,"%s\n",PETSC_VERSION_NUMBER);
    fprintf(stderr,"Satish Balay, Bill Gropp, Lois Curfman McInnes, Barry Smith.\n");
    fprintf(stderr,"Bug reports, questions: petsc-maint@mcs.anl.gov\n");
    fprintf(stderr,"Web page: http://www.mcs.anl.gov/petsc/\n");
    fprintf(stderr,"See docs/copyright.html for copyright information.\n");
    fprintf(stderr,"See docs/changes.html for recent updates.\n");
    fprintf(stderr,"See docs/troubleshooting.html for hints about trouble shooting.\n");
    fprintf(stderr,"See docs/manualpages/index.html for manual pages.\n");
    fprintf(stderr,"--------------------------------------------\
---------------------------\n");
    fprintf(stderr,"%s on a %s named %s by %s %s\n",pname,arch,hostname,username,date);
#if !defined (PARCH_win32)
    fprintf(stderr,"Libraries linked from %s\n",PETSC_LDIR);
#endif
    fprintf(stderr,"--------------------------------------------\
---------------------------\n");
    fflush(stderr);
    InPetscErrorPrintfDefault = 0;
  }

  if (!InPetscErrorPrintfDefault) {
    va_start(Argp,format);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(stderr,format,(char *)Argp);
#else
    vfprintf(stderr,format,Argp);
#endif
    fflush(stderr);
    va_end(Argp);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscSynchronizedFGets" 
/*@C
    PetscSynchronizedFGets - Several processors all get the same line from a file.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   len - the lenght of the output buffer

    Output Parameter:
.   string - the line read from the file

    Level: intermediate

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(), 
          PetscFOpen(), ViewerASCIISynchronizedPrintf(), ViewerASCIIPrintf()

@*/
int PetscSynchronizedFGets(MPI_Comm comm,FILE* fp,int len,char string[])
{
  int ierr,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* First processor prints immediately to fp */
  if (!rank) {
    fgets(string,len,fp);
  }
  ierr = MPI_Bcast(string,len,MPI_BYTE,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
