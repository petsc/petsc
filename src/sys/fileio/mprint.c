/*
      Utilites routines to add simple ASCII IO capability.
*/
#include <../src/sys/fileio/mprint.h>
#include <errno.h>
/*
   If petsc_history is on, then all Petsc*Printf() results are saved
   if the appropriate (usually .petschistory) file.
*/
PETSC_INTERN FILE *petsc_history;
/*
     Allows one to overwrite where standard out is sent. For example
     PETSC_STDOUT = fopen("/dev/ttyXX","w") will cause all standard out
     writes to go to terminal XX; assuming you have write permission there
*/
FILE *PETSC_STDOUT = NULL;
/*
     Allows one to overwrite where standard error is sent. For example
     PETSC_STDERR = fopen("/dev/ttyXX","w") will cause all standard error
     writes to go to terminal XX; assuming you have write permission there
*/
FILE *PETSC_STDERR = NULL;

/*@C
     PetscFormatConvertGetSize - Gets the length of a string needed to hold format converted with PetscFormatConvert()

   Input Parameter:
.   format - the PETSc format string

   Output Parameter:
.   size - the needed length of the new format

 Level: developer

.seealso: PetscFormatConvert(), PetscVSNPrintf(), PetscVFPrintf()

@*/
PetscErrorCode PetscFormatConvertGetSize(const char *format,size_t *size)
{
  size_t   sz = 0;
  PetscInt i  = 0;

  PetscFunctionBegin;
  PetscValidCharPointer(format,1);
  PetscValidPointer(size,2);
  while (format[i]) {
    if (format[i] == '%') {
      if (format[i+1] == '%') {
        i  += 2;
        sz += 2;
        continue;
      }
      /* Find the letter */
      while (format[i] && (format[i] <= '9')) {++i;++sz;}
      switch (format[i]) {
#if PetscDefined(USE_64BIT_INDICES)
      case 'D':
        sz += 2;
        break;
#endif
      case 'g':
        sz += 4;
      default:
        break;
      }
    }
    ++i;
    ++sz;
  }
  *size = sz+1; /* space for NULL character */
  PetscFunctionReturn(0);
}

/*@C
     PetscFormatConvert - Takes a PETSc format string and converts the %D to %d for 32 bit PETSc indices and %lld for 64 bit PETSc indices. Also
                        converts %g to [|%g|] so that PetscVSNPrintf() can easily insure all %g formatted numbers have a decimal point when printed.

   Input Parameters:
+   format - the PETSc format string
.   newformat - the location to put the new format
-   size - the length of newformat, you can use PetscFormatConvertGetSize() to compute the needed size

    Note: this exists so we can have the same code when PetscInt is either int or long long int

 Level: developer

.seealso: PetscFormatConvertGetSize(), PetscVSNPrintf(), PetscVFPrintf()

@*/
PetscErrorCode PetscFormatConvert(const char *format,char *newformat)
{
  PetscInt i = 0, j = 0;

  PetscFunctionBegin;
  while (format[i]) {
    if (format[i] == '%' && format[i+1] == '%') {
      newformat[j++] = format[i++];
      newformat[j++] = format[i++];
    } else if (format[i] == '%') {
      if (format[i+1] == 'g') {
        newformat[j++] = '[';
        newformat[j++] = '|';
      }
      /* Find the letter */
      for (; format[i] && format[i] <= '9'; i++) newformat[j++] = format[i];
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
      case 'g':
        newformat[j++] = format[i];
        if (format[i-1] == '%') {
          newformat[j++] = '|';
          newformat[j++] = ']';
        }
        break;
      case 'G':
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%%G format is no longer supported, use %%g and cast the argument to double");
      case 'F':
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%%F format is no longer supported, use %%f and cast the argument to double");
      default:
        newformat[j++] = format[i];
        break;
      }
      i++;
    } else newformat[j++] = format[i++];
  }
  newformat[j] = 0;
  PetscFunctionReturn(0);
}

#define PETSCDEFAULTBUFFERSIZE 8*1024

/*@C
     PetscVSNPrintf - The PETSc version of vsnprintf(). Converts a PETSc format string into a standard C format string and then puts all the
       function arguments into a string using the format statement.

   Input Parameters:
+   str - location to put result
.   len - the amount of space in str
+   format - the PETSc format string
-   fullLength - the amount of space in str actually used.

    Developer Notes:
    this function may be called from an error handler, if an error occurs when it is called by the error handler than likely
      a recursion will occur and possible crash.

 Level: developer

.seealso: PetscVSNPrintf(), PetscErrorPrintf(), PetscVPrintf()

@*/
PetscErrorCode PetscVSNPrintf(char *str,size_t len,const char *format,size_t *fullLength,va_list Argp)
{
  char           *newformat = NULL;
  char           formatbuf[PETSCDEFAULTBUFFERSIZE];
  size_t         newLength;
  PetscErrorCode ierr;
  int            flen;

  PetscFunctionBegin;
  ierr = PetscFormatConvertGetSize(format,&newLength);CHKERRQ(ierr);
  if (newLength < PETSCDEFAULTBUFFERSIZE) {
    newformat = formatbuf;
    newLength = PETSCDEFAULTBUFFERSIZE-1;
  } else {
    ierr      = PetscMalloc1(newLength, &newformat);CHKERRQ(ierr);
  }
  ierr = PetscFormatConvert(format,newformat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VSNPRINTF)
  flen = vsnprintf(str,len,newformat,Argp);
#else
#error "vsnprintf not found"
#endif
  if (newLength > PETSCDEFAULTBUFFERSIZE-1) {
    ierr = PetscFree(newformat);CHKERRQ(ierr);
  }
  {
    PetscBool foundedot;
    size_t cnt = 0,ncnt = 0,leng;
    ierr = PetscStrlen(str,&leng);CHKERRQ(ierr);
    if (leng > 4) {
      for (cnt=0; cnt<leng-4; cnt++) {
        if (str[cnt] == '[' && str[cnt+1] == '|') {
          flen -= 4;
          cnt++; cnt++;
          foundedot = PETSC_FALSE;
          for (; cnt<leng-1; cnt++) {
            if (str[cnt] == '|' && str[cnt+1] == ']') {
              cnt++;
              if (!foundedot) str[ncnt++] = '.';
              ncnt--;
              break;
            } else {
              if (str[cnt] == 'e' || str[cnt] == '.') foundedot = PETSC_TRUE;
              str[ncnt++] = str[cnt];
            }
          }
        } else {
          str[ncnt] = str[cnt];
        }
        ncnt++;
      }
      while (cnt < leng) {
        str[ncnt] = str[cnt]; ncnt++; cnt++;
      }
      str[ncnt] = 0;
    }
  }
#if defined(PETSC_HAVE_WINDOWS_H) && !defined(PETSC_HAVE__SET_OUTPUT_FORMAT)
  /* older Windows OS always produces e-+0np for floating point output; remove the extra 0 */
  {
    size_t cnt = 0,ncnt = 0,leng;
    ierr = PetscStrlen(str,&leng);CHKERRQ(ierr);
    if (leng > 5) {
      for (cnt=0; cnt<leng-4; cnt++) {
        if (str[cnt] == 'e' && (str[cnt+1] == '-' || str[cnt+1] == '+') && str[cnt+2] == '0'  && str[cnt+3] >= '0' && str[cnt+3] <= '9' && str[cnt+4] >= '0' && str[cnt+4] <= '9') {
          str[ncnt] = str[cnt]; ncnt++; cnt++;
          str[ncnt] = str[cnt]; ncnt++; cnt++; cnt++;
          str[ncnt] = str[cnt];
        } else {
          str[ncnt] = str[cnt];
        }
        ncnt++;
      }
      while (cnt < leng) {
        str[ncnt] = str[cnt]; ncnt++; cnt++;
      }
      str[ncnt] = 0;
    }
  }
#endif
  if (fullLength) *fullLength = 1 + (size_t) flen;
  PetscFunctionReturn(0);
}

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
$      ierr = PetscVFPrintfDefault(fd,format,Argp);CHKERR(ierr);
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

      Notes:
    For error messages this may be called by any process, for regular standard out it is
          called only by process 0 of a given communicator

      Developer Notes:
    this could be called by an error handler, if that happens then a recursion of the error handler may occur
                       and a crash

  Level:  developer

.seealso: PetscVSNPrintf(), PetscErrorPrintf()

@*/
PetscErrorCode PetscVFPrintfDefault(FILE *fd,const char *format,va_list Argp)
{
  char           str[PETSCDEFAULTBUFFERSIZE];
  char           *buff = str;
  size_t         fullLength;
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_VA_COPY)
  va_list        Argpcopy;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VA_COPY)
  va_copy(Argpcopy,Argp);
#endif
  ierr = PetscVSNPrintf(str,sizeof(str),format,&fullLength,Argp);CHKERRQ(ierr);
  if (fullLength > sizeof(str)) {
    ierr = PetscMalloc1(fullLength,&buff);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VA_COPY)
    ierr = PetscVSNPrintf(buff,fullLength,format,NULL,Argpcopy);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"C89 does not support va_copy() hence cannot print long strings with PETSc printing routines");
#endif
  }
  fprintf(fd,"%s",buff);
  fflush(fd);
  if (buff != str) {
    ierr = PetscFree(buff);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscSNPrintf - Prints to a string of given length

    Not Collective

    Input Parameters:
+   str - the string to print to
.   len - the length of str
.   format - the usual printf() format string
-   ... - any arguments that are to be printed, each much have an appropriate symbol in the format argument

   Level: intermediate

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), PetscVSNPrintf(),
          PetscPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf(), PetscVFPrintf()
@*/
PetscErrorCode PetscSNPrintf(char *str,size_t len,const char format[],...)
{
  PetscErrorCode ierr;
  size_t         fullLength;
  va_list        Argp;

  PetscFunctionBegin;
  va_start(Argp,format);
  ierr = PetscVSNPrintf(str,len,format,&fullLength,Argp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscSNPrintfCount - Prints to a string of given length, returns count

    Not Collective

    Input Parameters:
+   str - the string to print to
.   len - the length of str
.   format - the usual printf() format string
-   ... - any arguments that are to be printed, each much have an appropriate symbol in the format argument

    Output Parameter:
.   countused - number of characters used

   Level: intermediate

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(), PetscVSNPrintf(),
          PetscPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf(), PetscSNPrintf(), PetscVFPrintf()
@*/
PetscErrorCode PetscSNPrintfCount(char *str,size_t len,const char format[],size_t *countused,...)
{
  PetscErrorCode ierr;
  va_list        Argp;

  PetscFunctionBegin;
  va_start(Argp,countused);
  ierr = PetscVSNPrintf(str,len,format,countused,Argp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------- */

PrintfQueue petsc_printfqueue       = NULL,petsc_printfqueuebase = NULL;
int         petsc_printfqueuelength = 0;

/*@C
    PetscSynchronizedPrintf - Prints synchronized output from several processors.
    Output of the first processor is followed by that of the second, etc.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string

   Level: intermediate

    Notes:
    REQUIRES a call to PetscSynchronizedFlush() by all the processes after the completion of the calls to PetscSynchronizedPrintf() for the information
    from all the processors to be printed.

    Fortran Note:
    The call sequence is PetscSynchronizedPrintf(MPI_Comm, character(*), PetscErrorCode ierr) from Fortran.
    That is, you can only pass a single character string from Fortran.

.seealso: PetscSynchronizedFlush(), PetscSynchronizedFPrintf(), PetscFPrintf(),
          PetscPrintf(), PetscViewerASCIIPrintf(), PetscViewerASCIISynchronizedPrintf()
@*/
PetscErrorCode PetscSynchronizedPrintf(MPI_Comm comm,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* First processor prints immediately to stdout */
  if (rank == 0) {
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
    size_t      fullLength = PETSCDEFAULTBUFFERSIZE;

    ierr = PetscNew(&next);CHKERRQ(ierr);
    if (petsc_printfqueue) {
      petsc_printfqueue->next = next;
      petsc_printfqueue       = next;
      petsc_printfqueue->next = NULL;
    } else petsc_printfqueuebase = petsc_printfqueue = next;
    petsc_printfqueuelength++;
    next->size   = -1;
    next->string = NULL;
    while ((PetscInt)fullLength >= next->size) {
      next->size = fullLength+1;
      ierr = PetscFree(next->string);CHKERRQ(ierr);
      ierr = PetscMalloc1(next->size, &next->string);CHKERRQ(ierr);
      va_start(Argp,format);
      ierr = PetscArrayzero(next->string,next->size);CHKERRQ(ierr);
      ierr = PetscVSNPrintf(next->string,next->size,format, &fullLength,Argp);CHKERRQ(ierr);
      va_end(Argp);
    }
  }
  PetscFunctionReturn(0);
}

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
PetscErrorCode PetscSynchronizedFPrintf(MPI_Comm comm,FILE *fp,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* First processor prints immediately to fp */
  if (rank == 0) {
    va_list Argp;
    va_start(Argp,format);
    ierr = (*PetscVFPrintf)(fp,format,Argp);CHKERRQ(ierr);
    if (petsc_history && (fp !=petsc_history)) {
      va_start(Argp,format);
      ierr = (*PetscVFPrintf)(petsc_history,format,Argp);CHKERRQ(ierr);
    }
    va_end(Argp);
  } else { /* other processors add to local queue */
    va_list     Argp;
    PrintfQueue next;
    size_t      fullLength = PETSCDEFAULTBUFFERSIZE;

    ierr = PetscNew(&next);CHKERRQ(ierr);
    if (petsc_printfqueue) {
      petsc_printfqueue->next = next;
      petsc_printfqueue       = next;
      petsc_printfqueue->next = NULL;
    } else petsc_printfqueuebase = petsc_printfqueue = next;
    petsc_printfqueuelength++;
    next->size   = -1;
    next->string = NULL;
    while ((PetscInt)fullLength >= next->size) {
      next->size = fullLength+1;
      ierr = PetscFree(next->string);CHKERRQ(ierr);
      ierr = PetscMalloc1(next->size, &next->string);CHKERRQ(ierr);
      va_start(Argp,format);
      ierr = PetscArrayzero(next->string,next->size);CHKERRQ(ierr);
      ierr = PetscVSNPrintf(next->string,next->size,format,&fullLength,Argp);CHKERRQ(ierr);
      va_end(Argp);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscSynchronizedFlush - Flushes to the screen output from all processors
    involved in previous PetscSynchronizedPrintf()/PetscSynchronizedFPrintf() calls.

    Collective

    Input Parameters:
+   comm - the communicator
-   fd - the file pointer (valid on process 0 of the communicator)

    Level: intermediate

    Notes:
    If PetscSynchronizedPrintf() and/or PetscSynchronizedFPrintf() are called with
    different MPI communicators there must be an intervening call to PetscSynchronizedFlush() between the calls with different MPI communicators.

    From Fortran pass PETSC_STDOUT if the flush is for standard out; otherwise pass a value obtained from PetscFOpen()

.seealso: PetscSynchronizedPrintf(), PetscFPrintf(), PetscPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIISynchronizedPrintf()
@*/
PetscErrorCode PetscSynchronizedFlush(MPI_Comm comm,FILE *fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,tag,i,j,n = 0,dummy = 0;
  char          *message;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /* First processor waits for messages from all other processors */
  if (rank == 0) {
    if (!fd) fd = PETSC_STDOUT;
    for (i=1; i<size; i++) {
      /* to prevent a flood of messages to process zero, request each message separately */
      ierr = MPI_Send(&dummy,1,MPI_INT,i,tag,comm);CHKERRMPI(ierr);
      ierr = MPI_Recv(&n,1,MPI_INT,i,tag,comm,&status);CHKERRMPI(ierr);
      for (j=0; j<n; j++) {
        PetscMPIInt size = 0;

        ierr = MPI_Recv(&size,1,MPI_INT,i,tag,comm,&status);CHKERRMPI(ierr);
        ierr = PetscMalloc1(size, &message);CHKERRQ(ierr);
        ierr = MPI_Recv(message,size,MPI_CHAR,i,tag,comm,&status);CHKERRMPI(ierr);
        ierr = PetscFPrintf(comm,fd,"%s",message);CHKERRQ(ierr);
        ierr = PetscFree(message);CHKERRQ(ierr);
      }
    }
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = petsc_printfqueuebase,previous;

    ierr = MPI_Recv(&dummy,1,MPI_INT,0,tag,comm,&status);CHKERRMPI(ierr);
    ierr = MPI_Send(&petsc_printfqueuelength,1,MPI_INT,0,tag,comm);CHKERRMPI(ierr);
    for (i=0; i<petsc_printfqueuelength; i++) {
      ierr     = MPI_Send(&next->size,1,MPI_INT,0,tag,comm);CHKERRMPI(ierr);
      ierr     = MPI_Send(next->string,next->size,MPI_CHAR,0,tag,comm);CHKERRMPI(ierr);
      previous = next;
      next     = next->next;
      ierr     = PetscFree(previous->string);CHKERRQ(ierr);
      ierr     = PetscFree(previous);CHKERRQ(ierr);
    }
    petsc_printfqueue       = NULL;
    petsc_printfqueuelength = 0;
  }
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------*/

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

.seealso: PetscPrintf(), PetscSynchronizedPrintf(), PetscViewerASCIIPrintf(),
          PetscViewerASCIISynchronizedPrintf(), PetscSynchronizedFlush()
@*/
PetscErrorCode PetscFPrintf(MPI_Comm comm,FILE* fd,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
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

/*@C
    PetscPrintf - Prints to standard out, only from the first
    processor in the communicator. Calls from other processes are ignored.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual printf() format string

    Level: intermediate

    Notes:
    PetscPrintf() supports some format specifiers that are unique to PETSc.
    See the manual page for PetscFormatConvert() for details.

    Fortran Note:
    The call sequence is PetscPrintf(MPI_Comm, character(*), PetscErrorCode ierr) from Fortran.
    That is, you can only pass a single character string from Fortran.

.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscFormatConvert()
@*/
PetscErrorCode PetscPrintf(MPI_Comm comm,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
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

PetscErrorCode PetscHelpPrintfDefault(MPI_Comm comm,const char format[],...)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
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

/*@C
    PetscSynchronizedFGets - Several processors all get the same line from a file.

    Collective

    Input Parameters:
+   comm - the communicator
.   fd - the file pointer
-   len - the length of the output buffer

    Output Parameter:
.   string - the line read from the file, at end of file string[0] == 0

    Level: intermediate

.seealso: PetscSynchronizedPrintf(), PetscSynchronizedFlush(),
          PetscFOpen(), PetscViewerASCIISynchronizedPrintf(), PetscViewerASCIIPrintf()

@*/
PetscErrorCode PetscSynchronizedFGets(MPI_Comm comm,FILE *fp,size_t len,char string[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  if (rank == 0) {
    char *ptr = fgets(string, len, fp);

    if (!ptr) {
      string[0] = 0;
      if (!feof(fp)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from file: %d", errno);
    }
  }
  ierr = MPI_Bcast(string,len,MPI_BYTE,0,comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CLOSURE)
int (^SwiftClosure)(const char*) = 0;

PetscErrorCode PetscVFPrintfToString(FILE *fd,const char format[],va_list Argp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fd != stdout && fd != stderr) { /* handle regular files */
    ierr = PetscVFPrintfDefault(fd,format,Argp);CHKERRQ(ierr);
  } else {
    size_t length;
    char   buff[PETSCDEFAULTBUFFERSIZE];

    ierr = PetscVSNPrintf(buff,sizeof(buff),format,&length,Argp);CHKERRQ(ierr);
    ierr = SwiftClosure(buff);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Provide a Swift function that processes all the PETSc calls to PetscVFPrintf()
*/
PetscErrorCode PetscVFPrintfSetClosure(int (^closure)(const char*))
{
  PetscVFPrintf = PetscVFPrintfToString;
  SwiftClosure  = closure;
  return 0;
}
#endif

/*@C
     PetscFormatStrip - Takes a PETSc format string and removes all numerical modifiers to % operations

   Input Parameters:
.   format - the PETSc format string

 Level: developer

@*/
PetscErrorCode PetscFormatStrip(char *format)
{
  size_t loc1 = 0, loc2 = 0;

  PetscFunctionBegin;
  while (format[loc2]) {
    if (format[loc2] == '%') {
      format[loc1++] = format[loc2++];
      while (format[loc2] && ((format[loc2] >= '0' && format[loc2] <= '9') || format[loc2] == '.')) loc2++;
    }
    format[loc1++] = format[loc2++];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFormatRealArray(char buf[],size_t len,const char *fmt,PetscInt n,const PetscReal x[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         left,count;
  char           *p;

  PetscFunctionBegin;
  for (i=0,p=buf,left=len; i<n; i++) {
    ierr = PetscSNPrintfCount(p,left,fmt,&count,(double)x[i]);CHKERRQ(ierr);
    if (count >= left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Insufficient space in buffer");
    left -= count;
    p    += count-1;
    *p++  = ' ';
  }
  p[i ? 0 : -1] = 0;
  PetscFunctionReturn(0);
}
