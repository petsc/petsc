#ifndef lint
static char vcid[] = "$Id: ex24.c,v 1.4 1996/05/28 22:46:47 balay Exp bsmith $";
#endif

static char help[] = "Demonstrates calling a Fortran computational routine from C.\n\n";

#include <stdio.h>
#include "vec.h"

int MAIN__()
{
  return 0;
}

/*
  Ugly stuff to insure the function names match between Fortran 
  and C. Sorry, but this is out of our PETSc hands to cleanup.
*/
#if defined(HAVE_FORTRAN_CAPS)
#define ex24f_ EX24F
#define ex24c_ EX24C
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define ex24f_ ex24f
#define ex24c_ ex24c
#endif
#if defined(__cplusplus)
extern "C" {
#endif
extern void ex24f_(int*);
#if defined(__cplusplus)
}
#endif

int main(int argc,char **args)
{
  int     ierr, m = 10,fvec;
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
  */
  ierr = PetscCObjectToFortranObject(vec,&fvec);
  ex24f_(&fvec);

  ierr = VecView(vec,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = VecDestroy(vec); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

#if defined(__cplusplus)
extern "C" {
#endif

int ex24c_(int *fvec)
{
  Vec vec;
  int ierr,size;

  /*
      Translate Fortran integer pointer back to C
  */
  ierr = PetscFortranObjectToCObject(*fvec,&vec);
  ierr = VecGetSize(vec,&size); CHKERRA(ierr);
  
  return 0;
}
 
#if defined(__cplusplus)
}
#endif
