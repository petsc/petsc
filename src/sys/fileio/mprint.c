/*
      Utilities routines to add simple ASCII IO capability.
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
     PetscFormatConvertGetSize - Gets the length of a string needed to hold data converted with `PetscFormatConvert()` based on format

   No Fortran Support

   Input Parameter:
.   format - the PETSc format string

   Output Parameter:
.   size - the needed length of the new format

   Level: developer

.seealso: `PetscFormatConvert()`, `PetscVSNPrintf()`, `PetscVFPrintf()`
@*/
PetscErrorCode PetscFormatConvertGetSize(const char *format, size_t *size)
{
  size_t   sz = 0;
  PetscInt i  = 0;

  PetscFunctionBegin;
  PetscValidCharPointer(format, 1);
  PetscValidPointer(size, 2);
  while (format[i]) {
    if (format[i] == '%') {
      if (format[i + 1] == '%') {
        i += 2;
        sz += 2;
        continue;
      }
      /* Find the letter */
      while (format[i] && (format[i] <= '9')) {
        ++i;
        ++sz;
      }
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
  *size = sz + 1; /* space for NULL character */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PetscFormatConvert - converts %g to [|%g|] so that `PetscVSNPrintf()` can ensure all %g formatted numbers have a decimal point when printed. The
     decimal point is then used by the `petscdiff` script so that differences in floating point number output is ignored in the test harness.

   No Fortran Support

   Input Parameters:
+   format - the PETSc format string
.   newformat - the location to put the new format
-   size - the length of newformat, you can use `PetscFormatConvertGetSize()` to compute the needed size

   Level: developer

    Note:
    Deprecated usage also converts the %D to %d for 32 bit PETSc indices and %lld for 64 bit PETSc indices. This feature is no
    longer used in PETSc code instead use %" PetscInt_FMT " in the format string

.seealso: `PetscFormatConvertGetSize()`, `PetscVSNPrintf()`, `PetscVFPrintf()`
@*/
PetscErrorCode PetscFormatConvert(const char *format, char *newformat)
{
  PetscInt i = 0, j = 0;

  PetscFunctionBegin;
  while (format[i]) {
    if (format[i] == '%' && format[i + 1] == '%') {
      newformat[j++] = format[i++];
      newformat[j++] = format[i++];
    } else if (format[i] == '%') {
      if (format[i + 1] == 'g') {
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
        if (format[i - 1] == '%') {
          newformat[j++] = '|';
          newformat[j++] = ']';
        }
        break;
      case 'G':
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "%%G format is no longer supported, use %%g and cast the argument to double");
      case 'F':
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "%%F format is no longer supported, use %%f and cast the argument to double");
      default:
        newformat[j++] = format[i];
        break;
      }
      i++;
    } else newformat[j++] = format[i++];
  }
  newformat[j] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSCDEFAULTBUFFERSIZE 8 * 1024

/*@C
     PetscVSNPrintf - The PETSc version of `vsnprintf()`. Ensures that all `%g` formatted arguments' output contains the decimal point (which
     is used by the test harness)

   Input Parameters:
+   str - location to put result
.   len - the amount of space in str
+   format - the PETSc format string
-   fullLength - the amount of space in str actually used.

   Level: developer

   Developer Note:
   This function may be called from an error handler, if an error occurs when it is called by the error handler than likely
   a recursion will occur resulting in a crash of the program.

   If the length of the format string `format` is on the order of `PETSCDEFAULTBUFFERSIZE` (8 * 1024 bytes), this function will call `PetscMalloc()`

.seealso: `PetscFormatConvert()`, `PetscFormatConvertGetSize()`, `PetscVSNPrintf()`, `PetscErrorPrintf()`, `PetscVPrintf()`
@*/
PetscErrorCode PetscVSNPrintf(char *str, size_t len, const char *format, size_t *fullLength, va_list Argp)
{
  char  *newformat = NULL;
  char   formatbuf[PETSCDEFAULTBUFFERSIZE];
  size_t newLength;
  int    flen;

  PetscFunctionBegin;
  PetscCall(PetscFormatConvertGetSize(format, &newLength));
  if (newLength < sizeof(formatbuf)) {
    newformat = formatbuf;
    newLength = sizeof(formatbuf) - 1;
  } else {
    PetscCall(PetscMalloc1(newLength, &newformat));
  }
  PetscCall(PetscFormatConvert(format, newformat));
#if defined(PETSC_HAVE_VSNPRINTF)
  flen = vsnprintf(str, len, newformat, Argp);
#else
  #error "vsnprintf not found"
#endif
  if (newLength > sizeof(formatbuf) - 1) PetscCall(PetscFree(newformat));
  {
    PetscBool foundedot;
    size_t    cnt = 0, ncnt = 0, leng;
    PetscCall(PetscStrlen(str, &leng));
    if (leng > 4) {
      for (cnt = 0; cnt < leng - 4; cnt++) {
        if (str[cnt] == '[' && str[cnt + 1] == '|') {
          flen -= 4;
          cnt++;
          cnt++;
          foundedot = PETSC_FALSE;
          for (; cnt < leng - 1; cnt++) {
            if (str[cnt] == '|' && str[cnt + 1] == ']') {
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
        str[ncnt] = str[cnt];
        ncnt++;
        cnt++;
      }
      str[ncnt] = 0;
    }
  }
#if defined(PETSC_HAVE_WINDOWS_H) && !defined(PETSC_HAVE__SET_OUTPUT_FORMAT)
  /* older Windows OS always produces e-+0np for floating point output; remove the extra 0 */
  {
    size_t cnt = 0, ncnt = 0, leng;
    PetscCall(PetscStrlen(str, &leng));
    if (leng > 5) {
      for (cnt = 0; cnt < leng - 4; cnt++) {
        if (str[cnt] == 'e' && (str[cnt + 1] == '-' || str[cnt + 1] == '+') && str[cnt + 2] == '0' && str[cnt + 3] >= '0' && str[cnt + 3] <= '9' && str[cnt + 4] >= '0' && str[cnt + 4] <= '9') {
          str[ncnt] = str[cnt];
          ncnt++;
          cnt++;
          str[ncnt] = str[cnt];
          ncnt++;
          cnt++;
          cnt++;
          str[ncnt] = str[cnt];
        } else {
          str[ncnt] = str[cnt];
        }
        ncnt++;
      }
      while (cnt < leng) {
        str[ncnt] = str[cnt];
        ncnt++;
        cnt++;
      }
      str[ncnt] = 0;
    }
  }
#endif
  if (fullLength) *fullLength = 1 + (size_t)flen;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFFlush - Flush a file stream

  Input Parameter:
. fd - The file stream handle

  Level: intermediate

  Notes:
  For output streams (and for update streams on which the last operation was output), writes
  any unwritten data from the stream's buffer to the associated output device.

  For input streams (and for update streams on which the last operation was input), the
  behavior is undefined.

  If `fd` is `NULL`, all open output streams are flushed, including ones not directly
  accessible to the program.

.seealso: `PetscPrintf()`, `PetscFPrintf()`, `PetscVFPrintf()`, `PetscVSNPrintf()`
@*/
PetscErrorCode PetscFFlush(FILE *fd)
{
  int ret;

  PetscFunctionBegin;
  if (fd) PetscValidPointer(fd, 1);
  ret = fflush(fd);
  // could also use PetscCallExternal() here, but since we can get additional error explanation
  // from strerror() we opted for a manual check
  PetscCheck(ret == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Error in fflush(): error code %d (%s)", ret, strerror(errno));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PetscVFPrintf -  All PETSc standard out and error messages are sent through this function; so, in theory, this can
        can be replaced with something that does not simply write to a file.

      To use, write your own function for example,
.vb
   PetscErrorCode mypetscvfprintf(FILE *fd, const char format[], va_list Argp)
   {
     PetscErrorCode ierr;

     PetscFunctionBegin;
      if (fd != stdout && fd != stderr) {  handle regular files
         CHKERR(PetscVFPrintfDefault(fd,format,Argp));
     } else {
        char   buff[BIG];
        size_t length;
        PetscCall(PetscVSNPrintf(buff,BIG,format,&length,Argp));
        now send buff to whatever stream or whatever you want
    }
    PetscFunctionReturn(PETSC_SUCCESS);
   }
.ve
   then before the call to `PetscInitialize()` do the assignment `PetscVFPrintf = mypetscvfprintf`;

  Level:  developer

   Note:
   For error messages this may be called by any process, for regular standard out it is
   called only by process 0 of a given communicator

   Developer Note:
   This could be called by an error handler, if that happens then a recursion of the error handler may occur
   and a resulting crash

.seealso: `PetscVSNPrintf()`, `PetscErrorPrintf()`, `PetscFFlush()`
@*/
PetscErrorCode PetscVFPrintfDefault(FILE *fd, const char *format, va_list Argp)
{
  char   str[PETSCDEFAULTBUFFERSIZE];
  char  *buff = str;
  size_t fullLength;
#if defined(PETSC_HAVE_VA_COPY)
  va_list Argpcopy;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VA_COPY)
  va_copy(Argpcopy, Argp);
#endif
  PetscCall(PetscVSNPrintf(str, sizeof(str), format, &fullLength, Argp));
  if (fullLength > sizeof(str)) {
    PetscCall(PetscMalloc1(fullLength, &buff));
#if defined(PETSC_HAVE_VA_COPY)
    PetscCall(PetscVSNPrintf(buff, fullLength, format, NULL, Argpcopy));
#else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "C89 does not support va_copy() hence cannot print long strings with PETSc printing routines");
#endif
  }
#if defined(PETSC_HAVE_VA_COPY)
  va_end(Argpcopy);
#endif
  {
    const int err = fprintf(fd, "%s", buff);
    // cannot use PetscCallExternal() for fprintf since the return value is "number of
    // characters transmitted to the output stream" on success
    PetscCheck(err >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "fprintf() returned error code %d", err);
  }
  PetscCall(PetscFFlush(fd));
  if (buff != str) PetscCall(PetscFree(buff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscSNPrintf - Prints to a string of given length

    Not Collective

    Input Parameters:
+   str - the string to print to
.   len - the length of `str`
.   format - the usual `printf()` format string
-   ... - any arguments that are to be printed, each much have an appropriate symbol in the format argument

   Level: intermediate

.seealso: `PetscSynchronizedFlush()`, `PetscSynchronizedFPrintf()`, `PetscFPrintf()`, `PetscVSNPrintf()`,
          `PetscPrintf()`, `PetscViewerASCIIPrintf()`, `PetscViewerASCIISynchronizedPrintf()`,
          `PetscVFPrintf()`, `PetscFFlush()`
@*/
PetscErrorCode PetscSNPrintf(char *str, size_t len, const char format[], ...)
{
  size_t  fullLength;
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscVSNPrintf(str, len, format, &fullLength, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscSNPrintfCount - Prints to a string of given length, returns count of characters printed

    Not Collective

    Input Parameters:
+   str - the string to print to
.   len - the length of `str`
.   format - the usual `printf()` format string
-   ... - any arguments that are to be printed, each much have an appropriate symbol in the format argument

    Output Parameter:
.   countused - number of characters printed

   Level: intermediate

.seealso: `PetscSynchronizedFlush()`, `PetscSynchronizedFPrintf()`, `PetscFPrintf()`, `PetscVSNPrintf()`,
          `PetscPrintf()`, `PetscViewerASCIIPrintf()`, `PetscViewerASCIISynchronizedPrintf()`, `PetscSNPrintf()`, `PetscVFPrintf()`
@*/
PetscErrorCode PetscSNPrintfCount(char *str, size_t len, const char format[], size_t *countused, ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, countused);
  PetscCall(PetscVSNPrintf(str, len, format, countused, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PrintfQueue petsc_printfqueue = NULL, petsc_printfqueuebase = NULL;
int         petsc_printfqueuelength = 0;

static inline PetscErrorCode PetscVFPrintf_Private(MPI_Comm comm, FILE *fd, const char format[], va_list Argp)
{
  const PetscBool tee = (PetscBool)(petsc_history && (fd != petsc_history));
  PetscMPIInt     rank;
  va_list         cpy;

  PetscFunctionBegin;
  PetscCheck(comm != MPI_COMM_NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (PetscLikely(rank != 0)) PetscFunctionReturn(PETSC_SUCCESS);
  // must do this before we possibly consume Argp
  if (tee) va_copy(cpy, Argp);
  PetscCall((*PetscVFPrintf)(fd, format, Argp));
  if (tee) {
    PetscCall((*PetscVFPrintf)(petsc_history, format, cpy));
    va_end(cpy);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscSynchronizedFPrintf_Private(MPI_Comm comm, FILE *fp, const char format[], va_list Argp)
{
  PetscMPIInt rank;
  va_list     cpy;

  PetscFunctionBegin;
  PetscCheck(comm != MPI_COMM_NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Called with MPI_COMM_NULL, likely PetscObjectComm() failed");
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* First processor prints immediately to fp */
  if (rank == 0) {
    va_copy(cpy, Argp);
    PetscCall(PetscVFPrintf_Private(comm, fp, format, cpy));
    va_end(cpy);
  } else { /* other processors add to local queue */
    PrintfQueue next;
    size_t      fullLength = PETSCDEFAULTBUFFERSIZE;

    PetscCall(PetscNew(&next));
    if (petsc_printfqueue) {
      petsc_printfqueue->next = next;
      petsc_printfqueue       = next;
      petsc_printfqueue->next = NULL;
    } else petsc_printfqueuebase = petsc_printfqueue = next;
    petsc_printfqueuelength++;
    next->size   = 0;
    next->string = NULL;
    while (fullLength >= next->size) {
      next->size = fullLength + 1;
      PetscCall(PetscFree(next->string));
      PetscCall(PetscMalloc1(next->size, &next->string));
      PetscCall(PetscArrayzero(next->string, next->size));
      va_copy(cpy, Argp);
      PetscCall(PetscVSNPrintf(next->string, next->size, format, &fullLength, cpy));
      va_end(cpy);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscSynchronizedPrintf - Prints synchronized output from several processors.
    Output of the first processor is followed by that of the second, etc.

    Not Collective

    Input Parameters:
+   comm - the MPI communicator
-   format - the usual `printf()` format string

   Level: intermediate

    Note:
    REQUIRES a call to `PetscSynchronizedFlush()` by all the processes after the completion of the calls to `PetscSynchronizedPrintf()` for the information
    from all the processors to be printed.

    Fortran Note:
    The call sequence is `PetscSynchronizedPrintf`(`MPI_Comm`, `character`(*), `PetscErrorCode` ierr).
    That is, you can only pass a single character string from Fortran.

.seealso: `PetscSynchronizedFlush()`, `PetscSynchronizedFPrintf()`, `PetscFPrintf()`,
          `PetscPrintf()`, `PetscViewerASCIIPrintf()`, `PetscViewerASCIISynchronizedPrintf()`,
          `PetscFFlush()`
@*/
PetscErrorCode PetscSynchronizedPrintf(MPI_Comm comm, const char format[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscSynchronizedFPrintf_Private(comm, PETSC_STDOUT, format, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscSynchronizedFPrintf - Prints synchronized output to the specified file from
    several processors.  Output of the first processor is followed by that of the
    second, etc.

    Not Collective

    Input Parameters:
+   comm - the MPI communicator
.   fd - the file pointer
-   format - the usual `printf()` format string

    Level: intermediate

    Note:
    REQUIRES a intervening call to `PetscSynchronizedFlush()` for the information
    from all the processors to be printed.

.seealso: `PetscSynchronizedPrintf()`, `PetscSynchronizedFlush()`, `PetscFPrintf()`,
          `PetscFOpen()`, `PetscViewerASCIISynchronizedPrintf()`, `PetscViewerASCIIPrintf()`,
          `PetscFFlush()`
@*/
PetscErrorCode PetscSynchronizedFPrintf(MPI_Comm comm, FILE *fp, const char format[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscSynchronizedFPrintf_Private(comm, fp, format, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscSynchronizedFlush - Flushes to the screen output from all processors
    involved in previous `PetscSynchronizedPrintf()`/`PetscSynchronizedFPrintf()` calls.

    Collective

    Input Parameters:
+   comm - the MPI communicator
-   fd - the file pointer (valid on process 0 of the communicator)

    Level: intermediate

    Note:
    If `PetscSynchronizedPrintf()` and/or `PetscSynchronizedFPrintf()` are called with
    different MPI communicators there must be an intervening call to `PetscSynchronizedFlush()` between the calls with different MPI communicators.

    Fortran Note:
    Pass `PETSC_STDOUT` if the flush is for standard out; otherwise pass a value obtained from `PetscFOpen()`

.seealso: `PetscSynchronizedPrintf()`, `PetscFPrintf()`, `PetscPrintf()`, `PetscViewerASCIIPrintf()`,
          `PetscViewerASCIISynchronizedPrintf()`
@*/
PetscErrorCode PetscSynchronizedFlush(MPI_Comm comm, FILE *fd)
{
  PetscMPIInt rank, size, tag, i, j, n = 0, dummy = 0;
  char       *message;
  MPI_Status  status;

  PetscFunctionBegin;
  PetscCall(PetscCommDuplicate(comm, &comm, &tag));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* First processor waits for messages from all other processors */
  if (rank == 0) {
    if (!fd) fd = PETSC_STDOUT;
    for (i = 1; i < size; i++) {
      /* to prevent a flood of messages to process zero, request each message separately */
      PetscCallMPI(MPI_Send(&dummy, 1, MPI_INT, i, tag, comm));
      PetscCallMPI(MPI_Recv(&n, 1, MPI_INT, i, tag, comm, &status));
      for (j = 0; j < n; j++) {
        PetscMPIInt size = 0;

        PetscCallMPI(MPI_Recv(&size, 1, MPI_INT, i, tag, comm, &status));
        PetscCall(PetscMalloc1(size, &message));
        PetscCallMPI(MPI_Recv(message, size, MPI_CHAR, i, tag, comm, &status));
        PetscCall(PetscFPrintf(comm, fd, "%s", message));
        PetscCall(PetscFree(message));
      }
    }
  } else { /* other processors send queue to processor 0 */
    PrintfQueue next = petsc_printfqueuebase, previous;

    PetscCallMPI(MPI_Recv(&dummy, 1, MPI_INT, 0, tag, comm, &status));
    PetscCallMPI(MPI_Send(&petsc_printfqueuelength, 1, MPI_INT, 0, tag, comm));
    for (i = 0; i < petsc_printfqueuelength; i++) {
      PetscCallMPI(MPI_Send(&next->size, 1, MPI_INT, 0, tag, comm));
      PetscCallMPI(MPI_Send(next->string, next->size, MPI_CHAR, 0, tag, comm));
      previous = next;
      next     = next->next;
      PetscCall(PetscFree(previous->string));
      PetscCall(PetscFree(previous));
    }
    petsc_printfqueue       = NULL;
    petsc_printfqueuelength = 0;
  }
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscFPrintf - Prints to a file, only from the first
    processor in the communicator.

    Not Collective; No Fortran Support

    Input Parameters:
+   comm - the MPI communicator
.   fd - the file pointer
-   format - the usual `printf()` format string

    Level: intermediate

    Developer Note:
    This maybe, and is, called from PETSc error handlers and `PetscMallocValidate()` hence it does not use `PetscCallMPI()` which
    could recursively restart the malloc validation.

.seealso: `PetscPrintf()`, `PetscSynchronizedPrintf()`, `PetscViewerASCIIPrintf()`,
          `PetscViewerASCIISynchronizedPrintf()`, `PetscSynchronizedFlush()`, `PetscFFlush()`
@*/
PetscErrorCode PetscFPrintf(MPI_Comm comm, FILE *fd, const char format[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscVFPrintf_Private(comm, fd, format, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscPrintf - Prints to standard out, only from the first
    processor in the communicator. Calls from other processes are ignored.

    Not Collective

    Input Parameters:
+   comm - the communicator
-   format - the usual `printf()` format string

    Level: intermediate

    Note:
    Deprecated information: `PetscPrintf()` supports some format specifiers that are unique to PETSc.
    See the manual page for `PetscFormatConvert()` for details.

    Fortran Note:
    The call sequence is `PetscPrintf`(MPI_Comm, character(*), `PetscErrorCode` ierr) from Fortran.
    That is, you can only pass a single character string from Fortran.

.seealso: `PetscFPrintf()`, `PetscSynchronizedPrintf()`, `PetscFormatConvert()`, `PetscFFlush()`
@*/
PetscErrorCode PetscPrintf(MPI_Comm comm, const char format[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscVFPrintf_Private(comm, PETSC_STDOUT, format, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscHelpPrintfDefault(MPI_Comm comm, const char format[], ...)
{
  va_list Argp;

  PetscFunctionBegin;
  va_start(Argp, format);
  PetscCall(PetscVFPrintf_Private(comm, PETSC_STDOUT, format, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

.seealso: `PetscSynchronizedPrintf()`, `PetscSynchronizedFlush()`,
          `PetscFOpen()`, `PetscViewerASCIISynchronizedPrintf()`, `PetscViewerASCIIPrintf()`
@*/
PetscErrorCode PetscSynchronizedFGets(MPI_Comm comm, FILE *fp, size_t len, char string[])
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (!fgets(string, len, fp)) {
    string[0] = 0;
    PetscCheck(feof(fp), PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from file: %d", errno);
  }
  PetscCallMPI(MPI_Bcast(string, len, MPI_BYTE, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PetscFormatStrip - Takes a PETSc format string and removes all numerical modifiers to % operations

   Input Parameter:
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFormatRealArray(char buf[], size_t len, const char *fmt, PetscInt n, const PetscReal x[])
{
  PetscInt i;
  size_t   left, count;
  char    *p;

  PetscFunctionBegin;
  for (i = 0, p = buf, left = len; i < n; i++) {
    PetscCall(PetscSNPrintfCount(p, left, fmt, &count, (double)x[i]));
    PetscCheck(count < left, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Insufficient space in buffer");
    left -= count;
    p += count - 1;
    *p++ = ' ';
  }
  p[i ? 0 : -1] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
