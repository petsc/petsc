/*$Id: mpiuopen.c,v 1.21 1999/10/24 14:01:25 bsmith Exp bsmith $*/
/*
      Some PETSc utilites routines to add simple parallel IO capability
*/
#include "petsc.h"
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscFOpen"
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
#define __FUNC__ "PetscFClose"
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
#define __FUNC__ "PetscPClose"
int PetscPClose(MPI_Comm comm,FILE *fd)
{
  int  rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) fclose(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPOpen"
int PetscPOpen(MPI_Comm comm,char *program,const char mode[],FILE **fp)
{
  int  ierr,rank;
  FILE *fd;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PLogInfo(0,"Running command :%s\n",program);

#if defined (PARCH_win32)
    SETERRQ(1,1,"Cannot run programs on NT");
#else 
    if (!(fd = popen(program,mode))) {
       SETERRQ1(1,1,"Cannot run command %s",program);
    }
#endif
    if (fp) *fp = fd;
  }
  PetscFunctionReturn(0);
}
