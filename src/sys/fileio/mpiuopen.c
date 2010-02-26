#define PETSC_DLL
/*
      Some PETSc utilites routines to add simple parallel IO capability
*/
#include "petscsys.h"
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscFOpen"
/*@C
    PetscFOpen - Has the first process in the communicator open a file;
    all others do nothing.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
.   name - the filename
-   mode - the mode for fopen(), usually "w"

    Output Parameter:
.   fp - the file pointer

    Level: developer

    Notes:
       PETSC_NULL (0), "stderr" or "stdout" may be passed in as the filename
  
    Fortran Note:
    This routine is not supported in Fortran.

    Concepts: opening ASCII file
    Concepts: files^opening ASCII

.seealso: PetscFClose(), PetscSynchronizedFGets(), PetscSynchronizedPrintf(), PetscSynchronizedFlush(),
          PetscFPrintf()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscFOpen(MPI_Comm comm,const char name[],const char mode[],FILE **fp)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  FILE           *fd;
  char           fname[PETSC_MAX_PATH_LEN],tname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscTruth isstdout,isstderr;
    ierr = PetscStrcmp(name,"stdout",&isstdout);CHKERRQ(ierr);
    ierr = PetscStrcmp(name,"stderr",&isstderr);CHKERRQ(ierr);
    if (isstdout || !name) {
      fd = PETSC_STDOUT;
    } else if (isstderr) {
      fd = PETSC_STDERR;
    } else {
      ierr = PetscStrreplace(PETSC_COMM_SELF,name,tname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      ierr = PetscFixFilename(tname,fname);CHKERRQ(ierr);
      ierr = PetscInfo1(0,"Opening file %s\n",fname);CHKERRQ(ierr);
      fd   = fopen(fname,mode);
      if (!fd) SETERRQ1(PETSC_ERR_FILE_OPEN,"Unable to open file %s\n",fname);
    }
  } else fd = 0;
  *fp = fd;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscFClose"
/*@
    PetscFClose - Has the first processor in the communicator close a 
    file; all others do nothing.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
-   fd - the file, opened with PetscFOpen()

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

    Concepts: files^closing ASCII
    Concepts: closing file

.seealso: PetscFOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscFClose(MPI_Comm comm,FILE *fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  int            err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank && fd != PETSC_STDOUT && fd != PETSC_STDERR) {
    err = fclose(fd);
    if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_POPEN)

#undef __FUNCT__  
#define __FUNCT__ "PetscPClose"
/*@C
      PetscPClose - Closes (ends) a program on processor zero run with PetscPOpen()

     Collective on MPI_Comm, but only process 0 runs the command

   Input Parameters:
+   comm - MPI communicator, only processor zero runs the program
-   fp - the file pointer where program input or output may be read or PETSC_NULL if don't care

   Level: intermediate

   Notes:
       Does not work under Windows

.seealso: PetscFOpen(), PetscFClose(), PetscPOpen()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscPClose(MPI_Comm comm,FILE *fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  int            err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    char buf[1024];
    while (fgets(buf,1024,fd)) {;} /* wait till it prints everything */
    err = pclose(fd);
    if (err) SETERRQ1(PETSC_ERR_SYS,"pclose() failed on process %D",err);    
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscPOpen"
/*@C
      PetscPOpen - Runs a program on processor zero and sends either its input or output to 
          a file.

     Collective on MPI_Comm, but only process 0 runs the command

   Input Parameters:
+   comm - MPI communicator, only processor zero runs the program
.   machine - machine to run command on or PETSC_NULL, or string with 0 in first location
.   program - name of program to run
-   mode - either r or w

   Output Parameter:
.   fp - the file pointer where program input or output may be read or PETSC_NULL if don't care

   Level: intermediate

   Notes:
       Use PetscPClose() to close the file pointer when you are finished with it
       Does not work under Windows

       The program string may contain ${DISPLAY}, ${HOMEDIRECTORY} or ${WORKINGDIRECTORY}; these
    will be replaced with relevent values.

.seealso: PetscFOpen(), PetscFClose(), PetscPClose()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscPOpen(MPI_Comm comm,const char machine[],const char program[],const char mode[],FILE **fp)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  size_t         i,len,cnt;
  char           commandt[PETSC_MAX_PATH_LEN],command[PETSC_MAX_PATH_LEN];
  FILE           *fd;

  PetscFunctionBegin;

  /* all processors have to do the string manipulation because PetscStrreplace() is a collective operation */
  if (machine && machine[0]) {
    ierr = PetscStrcpy(command,"ssh ");CHKERRQ(ierr);
    ierr = PetscStrcat(command,machine);CHKERRQ(ierr);
    ierr = PetscStrcat(command," \" setenv DISPLAY ${DISPLAY}; ");CHKERRQ(ierr);
    /*
        Copy program into command but protect the " with a \ in front of it 
    */
    ierr = PetscStrlen(command,&cnt);CHKERRQ(ierr);
    ierr = PetscStrlen(program,&len);CHKERRQ(ierr);
    for (i=0; i<len; i++) {
      if (program[i] == '\"') {
        command[cnt++] = '\\';
      }
      command[cnt++] = program[i];
    }
    command[cnt] = 0; 
    ierr = PetscStrcat(command,"\"");CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(command,program);CHKERRQ(ierr);
  }

  ierr = PetscStrreplace(comm,command,commandt,1024);CHKERRQ(ierr);
    
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscInfo1(0,"Running command :%s\n",commandt);CHKERRQ(ierr);
    if (!(fd = popen(commandt,mode))) {
       SETERRQ1(PETSC_ERR_LIB,"Cannot run command %s",commandt);
    }
    if (fp) *fp = fd;
  }
  PetscFunctionReturn(0);
}

#endif
