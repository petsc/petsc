/*$Id: mpiuopen.c,v 1.20 1999/05/12 03:27:04 bsmith Exp bsmith $*/
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

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: file, open

.seealso: PetscFClose()
@*/
FILE *PetscFOpen(MPI_Comm comm,const char name[],const char mode[])
{
  int  rank,ierr;
  FILE *fd;
  char fname[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);if (ierr) PetscFunctionReturn(0);
  if (!rank) {
    ierr = PetscFixFilename(name,fname);
    if (ierr) PetscFunctionReturn(0);
    fd = fopen(fname,mode);
  } else fd = 0;
  PetscFunctionReturn(fd);
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
  if (!rank) fclose(fd);
  PetscFunctionReturn(0);
}

