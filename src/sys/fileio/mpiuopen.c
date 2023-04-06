#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for popen() */
/*
      Some PETSc utility routines to add simple parallel IO capabilities
*/
#include <petscsys.h>
#include <petsc/private/logimpl.h>
#include <errno.h>

/*@C
    PetscFOpen - Has the first process in the communicator open a file;
    all others do nothing.

    Logically Collective; No Fortran Support

    Input Parameters:
+   comm - the communicator
.   name - the filename
-   mode - the mode for `fopen()`, usually "w"

    Output Parameter:
.   fp - the file pointer

    Level: developer

    Note:
       `NULL`, "stderr" or "stdout" may be passed in as the filename

.seealso: `PetscFClose()`, `PetscSynchronizedFGets()`, `PetscSynchronizedPrintf()`, `PetscSynchronizedFlush()`,
          `PetscFPrintf()`
@*/
PetscErrorCode PetscFOpen(MPI_Comm comm, const char name[], const char mode[], FILE **fp)
{
  PetscMPIInt rank;
  FILE       *fd;
  char        fname[PETSC_MAX_PATH_LEN], tname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscBool isstdout, isstderr;
    PetscCall(PetscStrcmp(name, "stdout", &isstdout));
    PetscCall(PetscStrcmp(name, "stderr", &isstderr));
    if (isstdout || !name) fd = PETSC_STDOUT;
    else if (isstderr) fd = PETSC_STDERR;
    else {
      PetscBool devnull = PETSC_FALSE;
      PetscCall(PetscStrreplace(PETSC_COMM_SELF, name, tname, PETSC_MAX_PATH_LEN));
      PetscCall(PetscFixFilename(tname, fname));
      PetscCall(PetscStrbeginswith(fname, "/dev/null", &devnull));
      if (devnull) PetscCall(PetscStrncpy(fname, "/dev/null", sizeof(fname)));
      PetscCall(PetscInfo(0, "Opening file %s\n", fname));
      fd = fopen(fname, mode);
      PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to open file %s", fname);
    }
  } else fd = NULL;
  *fp = fd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscFClose - Has the first processor in the communicator close a
    file; all others do nothing.

    Logically Collective; No Fortran Support

    Input Parameters:
+   comm - the communicator
-   fd - the file, opened with PetscFOpen()

   Level: developer

.seealso: `PetscFOpen()`
@*/
PetscErrorCode PetscFClose(MPI_Comm comm, FILE *fd)
{
  PetscMPIInt rank;
  int         err;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0 && fd != PETSC_STDOUT && fd != PETSC_STDERR) {
    err = fclose(fd);
    PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_POPEN)
static char PetscPOpenMachine[128] = "";

/*@C
      PetscPClose - Closes (ends) a program on processor zero run with `PetscPOpen()`

     Collective, but only MPI rank 0 runs the command

   Input Parameters:
+   comm - MPI communicator, only rank 0 runs the program
-   fp - the file pointer where program input or output may be read or `NULL` if don't care

   Level: intermediate

   Note:
   Does not work under Microsoft Windows

.seealso: `PetscFOpen()`, `PetscFClose()`, `PetscPOpen()`
@*/
PetscErrorCode PetscPClose(MPI_Comm comm, FILE *fd)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    char buf[1024];
    while (fgets(buf, 1024, fd))
      ; /* wait till it prints everything */
    (void)pclose(fd);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
      PetscPOpen - Runs a program on processor zero and sends either its input or output to
          a file.

     Logically Collective, but only MPI rank 0 runs the command

   Input Parameters:
+   comm - MPI communicator, only processor zero runs the program
.   machine - machine to run command on or `NULL`, or string with 0 in first location
.   program - name of program to run
-   mode - either "r" or "w"

   Output Parameter:
.   fp - the file pointer where program input or output may be read or `NULL` if don't care

   Level: intermediate

   Notes:
       Use `PetscPClose()` to close the file pointer when you are finished with it

       Does not work under Microsoft Windows

       If machine is not provided will use the value set with `PetsPOpenSetMachine()` if that was provided, otherwise
       will use the machine running node zero of the communicator

       The program string may contain ${DISPLAY}, ${HOMEDIRECTORY} or ${WORKINGDIRECTORY}; these
    will be replaced with relevant values.

.seealso: `PetscFOpen()`, `PetscFClose()`, `PetscPClose()`, `PetscPOpenSetMachine()`
@*/
PetscErrorCode PetscPOpen(MPI_Comm comm, const char machine[], const char program[], const char mode[], FILE **fp)
{
  PetscMPIInt rank;
  size_t      i, len, cnt;
  char        commandt[PETSC_MAX_PATH_LEN], command[PETSC_MAX_PATH_LEN];
  FILE       *fd;

  PetscFunctionBegin;
  /* all processors have to do the string manipulation because PetscStrreplace() is a collective operation */
  if (PetscPOpenMachine[0] || (machine && machine[0])) {
    PetscCall(PetscStrncpy(command, "ssh ", sizeof(command)));
    if (PetscPOpenMachine[0]) {
      PetscCall(PetscStrlcat(command, PetscPOpenMachine, sizeof(command)));
    } else {
      PetscCall(PetscStrlcat(command, machine, sizeof(command)));
    }
    PetscCall(PetscStrlcat(command, " \" export DISPLAY=${DISPLAY}; ", sizeof(command)));
    /*
        Copy program into command but protect the " with a \ in front of it
    */
    PetscCall(PetscStrlen(command, &cnt));
    PetscCall(PetscStrlen(program, &len));
    for (i = 0; i < len; i++) {
      if (program[i] == '\"') command[cnt++] = '\\';
      command[cnt++] = program[i];
    }
    command[cnt] = 0;

    PetscCall(PetscStrlcat(command, "\"", sizeof(command)));
  } else {
    PetscCall(PetscStrncpy(command, program, sizeof(command)));
  }

  PetscCall(PetscStrreplace(comm, command, commandt, 1024));

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscCall(PetscInfo(NULL, "Running command :%s\n", commandt));
    PetscCheck((fd = popen(commandt, mode)), PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot run command %s", commandt);
    if (fp) *fp = fd;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
      PetscPOpenSetMachine - Sets the name of the default machine to run `PetscPOpen()` calls on

     Logically Collective, but only MPI rank 0 runs the command

   Input Parameter:
.   machine - machine to run command on or `NULL` for the current machine

   Options Database Key:
.   -popen_machine <machine> - run the process on this machine

   Level: intermediate

.seealso: `PetscFOpen()`, `PetscFClose()`, `PetscPClose()`, `PetscPOpen()`
@*/
PetscErrorCode PetscPOpenSetMachine(const char machine[])
{
  PetscFunctionBegin;
  if (machine) {
    PetscCall(PetscStrncpy(PetscPOpenMachine, machine, sizeof(PetscPOpenMachine)));
  } else {
    PetscPOpenMachine[0] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
