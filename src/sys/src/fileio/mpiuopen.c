/*$Id: mpiuopen.c,v 1.26 2000/03/23 18:41:12 balay Exp bsmith $*/
/*
      Some PETSc utilites routines to add simple parallel IO capability
*/
#include "petsc.h"
#include "sys.h"
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "petscfix.h"

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscFOpen"
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

.keywords: file, open

.seealso: PetscFClose()
@*/
int PetscFOpen(MPI_Comm comm,const char name[],const char mode[],FILE **fp)
{
  int  rank,ierr;
  FILE *fd;
  char fname[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscTruth isstdout,isstderr;
    ierr = PetscStrcmp(name,"stdout",&isstdout);CHKERRQ(ierr);
    ierr = PetscStrcmp(name,"stderr",&isstderr);CHKERRQ(ierr);
    if (isstdout || !name) {
      fd = stdout;
    } else if (isstderr) {
      fd = stderr;
    } else {
      ierr = PetscFixFilename(name,fname);CHKERRQ(ierr);
      fd = fopen(fname,mode);
    }
  } else fd = 0;
  *fp = fd;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscFClose"
/*@C
    PetscFClose - Has the first processor in the communicator close a 
    file; all others do nothing.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
-   fd - the file, opened with PetscFOpen()

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: file, close

.seealso: PetscFOpen()
@*/
int PetscFClose(MPI_Comm comm,FILE *fd)
{
  int  rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank && fd != stdout && fd != stderr) fclose(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscPClose"
int PetscPClose(MPI_Comm comm,FILE *fd)
{
  int  rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    char buf[1024];
    while (fgets(buf,1024,fd)) {;} /* wait till it prints everything */
#if defined (PARCH_win32)
    SETERRQ(1,1,"Cannot run programs on NT");
#else
    pclose(fd);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"PetscPOpen"
/*@C
      PetscPOpen - Runs a program on processor zero and sends either its input or output to 
          a file.

     Not Collective

   Input Parameters:
+   comm - MPI communicator, only processor zero runs the program
.   machine - machine to run command on or PETSC_NULL, or string with 0 in first location
.   program - name of program to run
-   mode - either r or w

   Output Parameter:
.   fp - the file pointer where program input or output may be read or PETSC_NULL if don't care

   Level: intermediate

   Notes:
       Does not work under Windows

       The program string may contain ${DISPLAY}, ${HOMEDIRECTORY} or ${WORKINGDIRECTORY}; these
    will be replaced with relevent values.

.seealso: PetscFOpen(), PetscFClose(), PetscPClose()

@*/
int PetscPOpen(MPI_Comm comm,char *machine,char *program,const char mode[],FILE **fp)
{
  int  ierr,rank;
  FILE *fd;
  char commandt[1024],command[1024];

  PetscFunctionBegin;

  /* all processors have to do the string manipulation because PetscStrreplace() is a collective operation */
  if (machine && machine[0]) {
    ierr = PetscStrcpy(command,"rsh ");CHKERRQ(ierr);
    ierr = PetscStrcat(command,machine);CHKERRQ(ierr);
    ierr = PetscStrcat(command," ");CHKERRQ(ierr);
    ierr = PetscStrcat(command,program);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(command,program);CHKERRQ(ierr);
  }

  ierr = PetscStrreplace(comm,command,commandt,1024);CHKERRQ(ierr);
    
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PLogInfo(0,"Running command :%s\n",commandt);

#if defined (PARCH_win32)
    SETERRQ(1,1,"Cannot run programs on NT");
#else 
    if (!(fd = popen(commandt,mode))) {
       SETERRQ1(1,1,"Cannot run command %s",commandt);
    }
#endif
    if (fp) *fp = fd;
  }
  PetscFunctionReturn(0);
}
