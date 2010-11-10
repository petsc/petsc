#define PETSC_DLL
/*
      Utilites routines to add simple ASCII IO capability.
*/
#include "../src/sys/fileio/mprint.h"
/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
extern FILE *petsc_history;
/*
     Allows one to overwrite where standard out is sent. For example
     PETSC_STDOUT = fopen("/dev/ttyXX","w") will cause all standard out
     writes to go to terminal XX; assuming you have write permission there
*/
FILE *PETSC_STDOUT = 0;
/*
     Allows one to overwrite where standard error is sent. For example
     PETSC_STDERR = fopen("/dev/ttyXX","w") will cause all standard error
     writes to go to terminal XX; assuming you have write permission there
*/
FILE *PETSC_STDERR = 0;
/*
     Used to output to Zope
*/
FILE *PETSC_ZOPEFD = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscFormatConvert"
PetscErrorCode PETSC_DLLEXPORT PetscFormatConvert(const char *format,char *newformat,PetscInt size)
{
  PetscInt i = 0,j = 0;

  while (format[i] && i < size-1) {
    if (format[i] == '%' && format[i+1] == 'D') {
      newformat[j++] = '%';
#if !defined(PETSC_USE_64BIT_INDICES)
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
#if !defined(PETSC_USE_64BIT_INDICES)
      newformat[j++] = 'd';
#else
      newformat[j++] = 'l';
      newformat[j++] = 'l';
      newformat[j++] = 'd';
#endif
      i += 3;
    } else if (format[i] == '%' && format[i+1] == 'G') {
      newformat[j++] = '%';
#if defined(PETSC_USE_SCALAR_INT)
      newformat[j++] = 'd';
#elif !defined(PETSC_USE_SCALAR_LONG_DOUBLE)
      newformat[j++] = 'g';
#else
      newformat[j++] = 'L';
      newformat[j++] = 'g';
#endif
      i += 2;
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
PetscErrorCode PETSC_DLLEXPORT PetscVSNPrintf(char *str,size_t len,const char *format,int *fullLength,va_list Argp)
{
  /* no malloc since may be called by error handler */
  char          *newformat;
  char           formatbuf[8*1024];
  size_t         oldLength,length;
  PetscErrorCode ierr;
 
  ierr = PetscStrlen(format, &oldLength);CHKERRQ(ierr);
  if (oldLength < 8*1024) {
    newformat = formatbuf;
  } else {
    ierr = PetscMalloc((oldLength+1) * sizeof(char), &newformat);CHKERRQ(ierr);
  }
  PetscFormatConvert(format,newformat,oldLength+1);
  ierr = PetscStrlen(newformat, &length);CHKERRQ(ierr);
#if 0
  if (length > len) {
    newformat[len] = '\0';
  }
#endif
#if defined(PETSC_HAVE_VSNPRINTF_CHAR)
  *fullLength = vsnprintf(str,len,newformat,(char *)Argp);
#elif defined(PETSC_HAVE_VSNPRINTF)
  *fullLength = vsnprintf(str,len,newformat,Argp);
#elif defined(PETSC_HAVE__VSNPRINTF)
  *fullLength = _vsnprintf(str,len,newformat,Argp);
#else
#error "vsnprintf not found"
#endif
  if (oldLength >= 8*1024) {
    ierr = PetscFree(newformat);CHKERRQ(ierr);
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscZopeLog"

PetscErrorCode PETSC_DLLEXPORT PetscZopeLog(const char *format,va_list Argp){
  /* no malloc since may be called by error handler */
  char     newformat[8*1024];
  char     log[8*1024];
  
  extern FILE * PETSC_ZOPEFD;
  char logstart[] = " <<<log>>>";
  size_t len;
  size_t formatlen;
  PetscFormatConvert(format,newformat,8*1024);
  PetscStrlen(logstart, &len);
  PetscMemcpy(log, logstart, len);
  PetscStrlen(newformat, &formatlen);
  PetscMemcpy(&(log[len]), newformat, formatlen);
  if(PETSC_ZOPEFD != NULL){
#if defined(PETSC_HAVE_VFPRINTF_CHAR)
  vfprintf(PETSC_ZOPEFD,log,(char *)Argp);
#else
  vfprintf(PETSC_ZOPEFD,log,Argp);
#endif
  fflush(PETSC_ZOPEFD);
}
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
PetscErrorCode PETSC_DLLEXPORT PetscVFPrintfDefault(FILE *fd,const char *format,va_list Argp)
{
  /* no malloc since may be called by error handler (assume no long messages in errors) */
  char        *newformat;
  char         formatbuf[8*1024];
  size_t       oldLength;
  extern FILE *PETSC_ZOPEFD;

  PetscStrlen(format, &oldLength);
  if (oldLength < 8*1024) {
    newformat = formatbuf;
  } else {
    (void)PetscMalloc((oldLength+1) * sizeof(char), &newformat);
  }
  PetscFormatConvert(format,newformat,oldLength+1);
  if(PETSC_ZOPEFD != NULL && PETSC_ZOPEFD != PETSC_STDOUT){
    va_list s;
#if defined(PETSC_HAVE_VA_COPY)
    va_copy(s, Argp);
#elif defined(PETSC_HAVE___VA_COPY)
    __va_copy(s, Argp);
#else
    SETERRQ(PETSC_ERR_SUP_SYS,"Zope not supported due to missing va_copy()");
#endif

#if defined(PETSC_HAVE_VA_COPY) || defined(PETSC_HAVE___VA_COPY)
#if defined(PETSC_HAVE_VFPRINTF_CHAR)
    vfprintf(PETSC_ZOPEFD,newformat,(char *)s);
#else
    vfprintf(PETSC_ZOPEFD,newformat,s);
#endif
    fflush(PETSC_ZOPEFD);
#endif
  }

#if defined(PETSC_HAVE_VFPRINTF_CHAR)
  vfprintf(fd,newformat,(char *)Argp);
#else
  vfprintf(fd,newformat,Argp);
#endif
  fflush(fd);
  if (oldLength >= 8*1024) {
    if (PetscFree(newformat)) {};
  }
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSNPrintf" 
/*@C
    PetscSNPrintf - Prints to a string of given length

    Not Collective

    Input Parameters:
+   str - the string to print to
.   len - the length of str
.   format - the usual printf() format string 
-   any arguments

   Level: intermediate

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), PetscVSNPrintf(),
          PetscPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscSNPrintf(char *str,size_t len,const char format[],...)
{
  PetscErrorCode ierr;
  int            fullLength;
  va_list        Argp;

  PetscFunctionBegin;
  va_start(Argp,format);
  ierr = PetscVSNPrintf(str,len,format,&fullLength,Argp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
    The call sequence is PetscSynchronizedPrintf(MPI_Comm, character(*), PetscErrorCode ierr) from Fortran. 
    That is, you can only pass a single character string from Fortran.

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
    ierr = (*PetscVFPrintf)(PETSC_STDOUT,format,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    va_list     Argp;
    PrintfQueue next;
    int         fullLength = 8191;

    ierr = PetscNew(struct _PrintfQueue,&next);CHKERRQ(ierr);
    if (queue) {queue->next = next; queue = next; queue->next = 0;}
    else       {queuebase   = queue = next;}
    queuelength++;
    next->size = -1;
    while(fullLength >= next->size) {
      next->size = fullLength+1;
      ierr = PetscMalloc(next->size * sizeof(char), &next->string);CHKERRQ(ierr);
      va_start(Argp,format);
      ierr = PetscMemzero(next->string,next->size);CHKERRQ(ierr);
      ierr = PetscVSNPrintf(next->string,next->size,format, &fullLength,Argp);CHKERRQ(ierr);
      va_end(Argp);
    }
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
    ierr = (*PetscVFPrintf)(fp,format,Argp);CHKERRQ(ierr);
    queuefile = fp;
    if (petsc_history && (fp !=petsc_history)) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    va_list     Argp;
    PrintfQueue next;
    int         fullLength = 8191;
    ierr = PetscNew(struct _PrintfQueue,&next);CHKERRQ(ierr);
    if (queue) {queue->next = next; queue = next; queue->next = 0;}
    else       {queuebase   = queue = next;}
    queuelength++;
    next->size = -1;
    while(fullLength >= next->size) {
      next->size = fullLength+1;
      ierr = PetscMalloc(next->size * sizeof(char), &next->string);CHKERRQ(ierr);
      va_start(Argp,format);
      ierr = PetscMemzero(next->string,next->size);CHKERRQ(ierr);
      ierr = PetscVSNPrintf(next->string,next->size,format,&fullLength,Argp);CHKERRQ(ierr);
      va_end(Argp);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSynchronizedFlush" 
/*@
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
  char          *message;
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
        int size;

        ierr = MPI_Recv(&size,1,MPI_INT,i,tag,comm,&status);CHKERRQ(ierr);
        ierr = PetscMalloc(size * sizeof(char), &message);CHKERRQ(ierr);
        ierr = MPI_Recv(message,size,MPI_CHAR,i,tag,comm,&status);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm,fd,"%s",message);
        ierr = PetscFree(message);CHKERRQ(ierr);
      }
    }
    queuefile = PETSC_NULL;
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = queuebase,previous;

    ierr = MPI_Send(&queuelength,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
    for (i=0; i<queuelength; i++) {
      ierr     = MPI_Send(&next->size,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
      ierr     = MPI_Send(next->string,next->size,MPI_CHAR,0,tag,comm);CHKERRQ(ierr);
      previous = next; 
      next     = next->next;
      ierr     = PetscFree(previous->string);CHKERRQ(ierr);
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
    ierr = (*PetscVFPrintf)(fd,format,Argp);CHKERRQ(ierr);
    if (petsc_history && (fd !=petsc_history)) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,format,Argp);CHKERRQ(ierr);
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
    The call sequence is PetscPrintf(MPI_Comm, character(*), PetscErrorCode ierr) from Fortran. 
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
    ierr = (*PetscVFPrintf)(PETSC_STDOUT,nformat,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,nformat,Argp);CHKERRQ(ierr);
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
    ierr = (*PetscVFPrintf)(PETSC_STDOUT,format,Argp);CHKERRQ(ierr);
    if (petsc_history) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/


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
