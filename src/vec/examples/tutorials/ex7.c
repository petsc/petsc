#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex7.c,v 1.12 1997/07/07 17:01:29 balay Exp balay $";
#endif

static char help[] = "Demonstrates calling a Fortran computational routine from C.\n\
Also demonstrates passing  PETSc objects, MPI Communicators from C to Fortran\n\
and from Fortran to C\n\n";

#include <stdio.h>
#include "vec.h"

/*
  Ugly stuff to insure the function names match between Fortran 
  and C. Sorry, but this is out of our PETSc hands to cleanup.
*/
#if defined(HAVE_FORTRAN_CAPS)
#define ex7f_ EX7F
#define ex7c_ EX7C
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define ex7f_ ex7f
#define ex7c_ ex7c
#endif
#if defined(__cplusplus)
extern "C" {
#endif
extern void ex7f_(int*,int*);
#if defined(__cplusplus)
}
#endif

int main(int argc,char **args)
{
  int     ierr, m = 10,fvec,fcomm;
  Vec     vec;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* This function should be called to be able to use PETSc routines
     from the FORTRAN subroutines needed by this program */

  PetscInitializeFortran();  

  ierr = VecCreate(MPI_COMM_WORLD,m,&vec); CHKERRA(ierr);

  /* 
     Call Fortran routine - the use of PetscCObjectToFortranObject()
     insures that the PETSc vector is properly interpreted on the 
     Fortran side. Note that Fortran treats all PETSc objects as integers.
     Similarly the MPI Communicator is passed through MPICCommToFortranComm()
     so that it can be properly interpreted from Fortran.
  */
  ierr = PetscCObjectToFortranObject(vec,&fvec); CHKERRA(ierr);
  ierr = MPICCommToFortranComm(MPI_COMM_WORLD,&fcomm); CHKERRA(ierr);

  ex7f_(&fvec,&fcomm);

  ierr = VecView(vec,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = VecDestroy(vec); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

#if defined(__cplusplus)
extern "C" {
#endif

int ex7c_(int *fvec, int *fcomm)
{
  Vec vec;
  MPI_Comm comm;
  int ierr,size;

  /*
    Translate Fortran integer pointer back to C and
    Fortran Communicator back to C communicator
  */
  ierr = PetscFortranObjectToCObject(*fvec,&vec);
  ierr = MPIFortranCommToCComm(*fcomm,&comm);
  
  /*
    Some PETSc/MPI operations on Vec/Communicator objects 
  */
  ierr = VecGetSize(vec,&size); CHKERRA(ierr);
  MPI_Barrier(comm);
  
  return 0;
}
 
#if defined(__cplusplus)
}
#endif
