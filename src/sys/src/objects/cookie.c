#ifndef lint
static char vcid[] = "$Id: cookie.c,v 1.7 1996/12/16 21:15:28 balay Exp balay $";
#endif

#include "petsc.h"  /*I "petsc.h" I*/
int LARGEST_PETSC_COOKIE = LARGEST_PETSC_COOKIE_PREDEFINED;

#undef __FUNCTION__  
#define __FUNCTION__ "PetscRegisterCookie"
/*@
    PetscRegisterCookie - Registers a new cookie for use with a
    newly created PETSc object class.  The user should pass in
    a variable initialized to zero; then it will be assigned a cookie.
    Repeated calls to this routine with the same variable will 
    not change the cookie. 

    Output Parameter:
.   cookie - the cookie you have been assigned

    Note:
    The initial cookie variable MUST be set to zero on the
    first call to this routine.

.keywords:  register, cookie
@*/
int PetscRegisterCookie(int *cookie)
{
  if (LARGEST_PETSC_COOKIE >= LARGEST_PETSC_COOKIE_ALLOWED) { 
    SETERRQ(1,"You have used too many PETSc cookies");
  }
  if (!*cookie) *cookie = LARGEST_PETSC_COOKIE++;
  return 0;
}
