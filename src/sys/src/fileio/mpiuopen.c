#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mpiuopen.c,v 1.12 1997/10/19 03:23:45 bsmith Exp bsmith $";
#endif
/*
      Some PETSc utilites routines to add simple IO capability to MPI.
*/
#include "petsc.h"
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscFOpen"
/*@C
    PetscFOpen - Has the first process in the communicator open a file;
    all others do nothing.

    Input Parameters:
.   comm - the communicator
.   name - the filename
.   mode - the mode for fopen(), usually "w"

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: file, open

.seealso: PetscFClose()
@*/
FILE *PetscFOpen(MPI_Comm comm,char *name,char *mode)
{
  int  rank,ierr;
  FILE *fd;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    ierr = PetscFixFilename(name);
    if (ierr) PetscFunctionReturn(0);
    fd = fopen(name,mode);
  } else fd = 0;
  PetscFunctionReturn(fd);
}

#undef __FUNC__  
#define __FUNC__ "PetscFClose"
/*@C
    PetscFClose - Has the first processor in the communicator close a 
    file; all others do nothing.

    Input Parameters:
.   comm - the communicator
.   fd - the file, opened with PetscFOpen()

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: file, close

.seealso: PetscFOpen()
@*/
int PetscFClose(MPI_Comm comm,FILE *fd)
{
  int  rank;

  PetscFunctionBegin;
  MPI_Comm_rank(comm,&rank);
  if (!rank) fclose(fd);
 PetscFunctionReturn(0);
}

