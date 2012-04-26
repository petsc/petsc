/*
      Utilites routines to add simple ASCII IO capability.
*/
#include <../src/sys/fileio/mprint.h>
#include <errno.h>
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

/*
     Return the maximum expected new size of the format
*/
#define PETSC_MAX_LENGTH_FORMAT(l) (l+l/8)

#undef __FUNCT__  
#define __FUNCT__ "PetscFormatConvert"
/*@C 
     PetscFormatConvert - Takes a PETSc format string and converts it to a reqular C format string

   Input Parameters:
+   format - the PETSc format string
.   newformat - the location to put the standard C format string values
-   size - the length of newformat

    Note: this exists so we can have the same code when PetscInt is either int or long long and PetscScalar is either __float128, double, or float

 Level: developer

@*/
PetscErrorCode  PetscFormatConvert(const char *format,char *newformat,size_t size)
{
  PetscInt i = 0,j = 0;

  PetscFunctionBegin;
  while (format[i] && j < (PetscInt)size-1) {
    if (format[i] == '%' && format[i+1] != '%') {
      /* Find the letter */
      for ( ; format[i] && format[i] <= '9'; i++) newformat[j++] = format[i];
      switch (format[i]) {
      case 'D':
#if !defined(PETSC_USE_64BIT_INDICES)
        newformat[j++] = 'd';
#else
        newformat[j++] = 'l';
        newformat[j++] = 'l';
        newformat[j++] = 'd';
#endif
        break;
      case 'G':
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL_SINGLE)
        newformat[j++] = 'g';
#elif defined(PETSC_USE_REAL___FLOAT128)
        newformat[j++] = 'Q';
        newformat[j++] = 'g';
#endif
        break;
      case 'F':
#if defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_REAL_SINGLE)
        newformat[j++] = 'f';
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
        newformat[j++] = 'L';
        newformat[j++] = 'f';
#elif defined(PETSC_USE_REAL___FLOAT128)
        newformat[j++] = 'Q';
        newformat[j++] = 'f';
#endif
        break;
      default:
        newformat[j++] = format[i];
        break;
      }
      i++;
    } else {
      newformat[j++] = format[i++];
    }
  }
  newformat[j] = 0;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "PetscVSNPrintf"
/*@C 
     PetscVSNPrintf - The PETSc version of vsnprintf(). Converts a PETSc format string into a standard C format string and then puts all the 
       function arguments into a string using the format statement.

   Input Parameters:
+   str - location to put result
.   len - the amount of space in str
+   format - the PETSc format string
-   fullLength - the amount of space in str actually used.

    Developer Notes: this function may be called from an error handler, if an error occurs when it is called by the error handler than likely
      a recursion will occur and possible crash.

 Level: developer

@*/
PetscErrorCode  PetscVSNPrintf(char *str,size_t len,const char *format,size_t *fullLength,va_list Argp)
{
  char          *newformat;
  char           formatbuf[8*1024];
  size_t         oldLength,length;
  int            fullLengthInt;
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  ierr = PetscStrlen(format, &oldLength);CHKERRQ(ierr);
  if (oldLength < 8*1024) {
    newformat = formatbuf;
    oldLength = 8*1024-1;
  } else {
    oldLength = PETSC_MAX_LENGTH_FORMAT(oldLength);
    ierr = PetscMalloc(oldLength * sizeof(char), &newformat);CHKERRQ(ierr);
  }
  PetscFormatConvert(format,newformat,oldLength);
  ierr = PetscStrlen(newformat, &length);CHKERRQ(ierr);
#if 0
  if (length > len) {
    newformat[len] = '\0';
  }
#endif
#if defined(PETSC_HAVE_VSNPRINTF_CHAR)
  fullLengthInt = vsnprintf(str,len,newformat,(char *)Argp);
#elif defined(PETSC_HAVE_VSNPRINTF)
  fullLengthInt = vsnprintf(str,len,newformat,Argp);
#elif defined(PETSC_HAVE__VSNPRINTF)
  fullLengthInt = _vsnprintf(str,len,newformat,Argp);
#else
#error "vsnprintf not found"
#endif
  if (fullLengthInt < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"vsnprintf() failed");
  if (fullLength) *fullLength = (size_t)fullLengthInt;
  if (oldLength >= 8*1024) {
    ierr = PetscFree(newformat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscVFPrintfDefault"
/*@C 
     PetscVFPrintf -  All PETSc standard out and error messages are sent through this function; so, in theory, this can
        can be replaced with something that does not simply write to a file. 

      To use, write your own function for example,
$PetscErrorCode mypetscvfprintf(FILE *fd,const char format[],va_list Argp)
${
$  PetscErrorCode ierr;
$
$  PetscFunctionBegin;
$   if (fd != stdout && fd != stderr) {  handle regular files 
$      ierr = PetscVFPrintfDefault(fd,format,Argp); CHKERR(ierr);
$  } else {
$     char   buff[BIG];
$     size_t length;
$     ierr = PetscVSNPrintf(buff,BIG,format,&length,Argp);CHKERRQ(ierr);
$     now send buff to whatever stream or whatever you want 
$ }
$ PetscFunctionReturn(0);
$}
then before the call to PetscInitialize() do the assignment
$    PetscVFPrintf = mypetscvfprintf;

      Notes: For error messages this may be called by any process, for regular standard out it is
          called only by process 0 of a given communicator

      Developer Notes: this could be called by an error handler, if that happens then a recursion of the error handler may occur 
                       and a crash

  Level:  developer

.seealso: PetscVSNPrintf(), PetscErrorPrintf()

@*/
PetscErrorCode  PetscVFPrintfDefault(FILE *fd,const char *format,va_list Argp)
{
  char           *newformat;
  char           formatbuf[8*1024];
  size_t         oldLength;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(format, &oldLength);CHKERRQ(ierr);
  if (oldLength < 8*1024) {
    newformat = formatbuf;
    oldLength = 8*1024-1;
  } else {
    oldLength = PETSC_MAX_LENGTH_FORMAT(oldLength);
    ierr = PetscMalloc(oldLength * sizeof(char), &newformat);CHKERRQ(ierr);
  }
  ierr = PetscFormatConvert(format,newformat,oldLength);CHKERRQ(ierr);

#if defined(PETSC_HAVE_VFPRINTF_CHAR)
  vfprintf(fd,newformat,(char *)Argp);
#else
  vfprintf(fd,newformat,Argp);
#endif
  fflush(fd);
  if (oldLength >= 8*1024) {
    ierr = PetscFree(newformat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
PetscErrorCode  PetscSNPrintf(char *str,size_t len,const char format[],...)
{
  PetscErrorCode ierr;
  size_t         fullLength;
  va_list        Argp;

  PetscFunctionBegin;
  va_start(Argp,format);
  ierr = PetscVSNPrintf(str,len,format,&fullLength,Argp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSNPrintfCount"
/*@C
    PetscSNPrintfCount - Prints to a string of given length, returns count

    Not Collective

    Input Parameters:
+   str - the string to print to
.   len - the length of str
.   format - the usual printf() format string
.   countused - number of characters used
-   any arguments

   Level: intermediate

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), PetscVSNPrintf(),
          PetscPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf(), PetscSNPrintf()
@*/
PetscErrorCode  PetscSNPrintfCount(char *str,size_t len,const char format[],size_t *countused,...)
{
  PetscErrorCode ierr;
  va_list        Argp;

  PetscFunctionBegin;
  va_start(Argp,countused);
  ierr = PetscVSNPrintf(str,len,format,countused,Argp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------- */

PrintfQueue petsc_printfqueue = 0,petsc_printfqueuebase = 0;
int         petsc_printfqueuelength = 0;
FILE        *petsc_printfqueuefile  = PETSC_NULL;

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
PetscErrorCode  PetscSynchronizedPrintf(MPI_Comm comm,const char format[],...)
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
    size_t      fullLength = 8191;

    ierr = PetscNew(struct _PrintfQueue,&next);CHKERRQ(ierr);
    if (petsc_printfqueue) {petsc_printfqueue->next = next; petsc_printfqueue = next; petsc_printfqueue->next = 0;}
    else                   {petsc_printfqueuebase   = petsc_printfqueue = next;}
    petsc_printfqueuelength++;
    next->size = -1;
    while((PetscInt)fullLength >= next->size) {
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
PetscErrorCode  PetscSynchronizedFPrintf(MPI_Comm comm,FILE* fp,const char format[],...)
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
    petsc_printfqueuefile = fp;
    if (petsc_history && (fp !=petsc_history)) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    va_list     Argp;
    PrintfQueue next;
    size_t      fullLength = 8191;
    ierr = PetscNew(struct _PrintfQueue,&next);CHKERRQ(ierr);
    if (petsc_printfqueue) {petsc_printfqueue->next = next; petsc_printfqueue = next; petsc_printfqueue->next = 0;}
    else                   {petsc_printfqueuebase   = petsc_printfqueue = next;}
    petsc_printfqueuelength++;
    next->size = -1;
    while((PetscInt)fullLength >= next->size) {
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
PetscErrorCode  PetscSynchronizedFlush(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag,i,j,n,dummy = 0;
  char          *message;
  MPI_Status     status;
  FILE           *fd;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* First processor waits for messages from all other processors */
  if (!rank) {
    if (petsc_printfqueuefile) {
      fd = petsc_printfqueuefile;
    } else {
      fd = PETSC_STDOUT;
    }
    for (i=1; i<size; i++) {
      /* to prevent a flood of messages to process zero, request each message separately */
      ierr = MPI_Send(&dummy,1,MPI_INT,i,tag,comm);CHKERRQ(ierr);
      ierr = MPI_Recv(&n,1,MPI_INT,i,tag,comm,&status);CHKERRQ(ierr);
      for (j=0; j<n; j++) {
        PetscMPIInt size;

        ierr = MPI_Recv(&size,1,MPI_INT,i,tag,comm,&status);CHKERRQ(ierr);
        ierr = PetscMalloc(size * sizeof(char), &message);CHKERRQ(ierr);
        ierr = MPI_Recv(message,size,MPI_CHAR,i,tag,comm,&status);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm,fd,"%s",message);
        ierr = PetscFree(message);CHKERRQ(ierr);
      }
    }
    petsc_printfqueuefile = PETSC_NULL;
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = petsc_printfqueuebase,previous;

    ierr = MPI_Recv(&dummy,1,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Send(&petsc_printfqueuelength,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
    for (i=0; i<petsc_printfqueuelength; i++) {
      ierr     = MPI_Send(&next->size,1,MPI_INT,0,tag,comm);CHKERRQ(ierr);
      ierr     = MPI_Send(next->string,next->size,MPI_CHAR,0,tag,comm);CHKERRQ(ierr);
      previous = next; 
      next     = next->next;
      ierr     = PetscFree(previous->string);CHKERRQ(ierr);
      ierr     = PetscFree(previous);CHKERRQ(ierr);
    }
    petsc_printfqueue       = 0;
    petsc_printfqueuelength = 0;
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
PetscErrorCode  PetscFPrintf(MPI_Comm comm,FILE* fd,const char format[],...)
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
    processor in the communicator. Calls from other processes are ignored.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string 

   Level: intermediate

    Fortran Note:
    The call sequence is PetscPrintf(MPI_Comm, character(*), PetscErrorCode ierr) from Fortran. 
    That is, you can only pass a single character string from Fortran.

   Concepts: printing^in parallel
   Concepts: printf^in parallel

.seealso: PetscFPrintf(), PetscSynchronizedPrintf()
@*/
PetscErrorCode  PetscPrintf(MPI_Comm comm,const char format[],...)
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
#define __FUNCT__ "PetscHelpPrintfDefault" 
/*@C 
     PetscHelpPrintf -  All PETSc help messages are passing through this function. You can change how help messages are printed by 
        replacinng it  with something that does not simply write to a stdout. 

      To use, write your own function for example,
$PetscErrorCode mypetschelpprintf(MPI_Comm comm,const char format[],....)
${
$ PetscFunctionReturn(0);
$}
then before the call to PetscInitialize() do the assignment
$    PetscHelpPrintf = mypetschelpprintf;

  Note: the default routine used is called PetscHelpPrintfDefault().

  Level:  developer

.seealso: PetscVSNPrintf(), PetscVFPrintf(), PetscErrorPrintf()
@*/
PetscErrorCode  PetscHelpPrintfDefault(MPI_Comm comm,const char format[],...)
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
PetscErrorCode  PetscSynchronizedFGets(MPI_Comm comm,FILE* fp,size_t len,char string[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (!rank) {
    char *ptr = fgets(string, len, fp);

    if (!ptr) {
      if (feof(fp)) {
        len = 0;
      } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from file: %d", errno);
    }
  }
  ierr = MPI_Bcast(string,len,MPI_BYTE,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <mex.h>
#undef __FUNCT__
#define __FUNCT__ "PetscVFPrintf_Matlab"
PetscErrorCode  PetscVFPrintf_Matlab(FILE *fd,const char format[],va_list Argp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fd != stdout && fd != stderr) { /* handle regular files */ 
    ierr = PetscVFPrintfDefault(fd,format,Argp); CHKERRQ(ierr);
  } else {
    size_t len=8*1024,length;
    char   buf[len];

    ierr = PetscVSNPrintf(buf,len,format,&length,Argp);CHKERRQ(ierr);
    mexPrintf("%s",buf);
 }
 PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscFormatStrip"
/*@C 
     PetscFormatStrip - Takes a PETSc format string and removes all numerical modifiers to % operations

   Input Parameters:
.   format - the PETSc format string

 Level: developer

@*/
PetscErrorCode  PetscFormatStrip(char *format)
{
  size_t   loc1 = 0, loc2 = 0;

  PetscFunctionBegin;
  while (format[loc2]){
    if (format[loc2] == '%') {
      format[loc1++] = format[loc2++];
      while (format[loc2] && ((format[loc2] >= '0' && format[loc2] <= '9') || format[loc2] == '.')) loc2++;
    } 
    format[loc1++] = format[loc2++];
  }
  PetscFunctionReturn(0);
}

