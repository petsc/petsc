
#ifndef lint
static char vcid[] = "$Id: mpiuopen.c,v 1.6 1996/09/28 17:33:48 curfman Exp balay $";
#endif
/*
      Some PETSc utilites routines to add simple IO capability to MPI.
*/
#include "petsc.h"
#include <stdio.h>
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNCTION__  
#define __FUNCTION__ "PetscFOpen"
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
  int  rank;
  FILE *fd;
  MPI_Comm_rank(comm,&rank);
  if (!rank) fd = fopen(name,mode);
  else fd = 0;
  return fd;
}

#undef __FUNCTION__  
#define __FUNCTION__ "PetscFClose"
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
  MPI_Comm_rank(comm,&rank);
  if (!rank) return fclose(fd);
  else return 0;
}

