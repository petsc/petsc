#ifndef lint
static char vcid[] = "$Id: cookie.c,v 1.3 1996/02/09 14:57:35 bsmith Exp curfman $";
#endif

#include "petsc.h"
int LARGEST_PETSC_COOKIE = LARGEST_PETSC_COOKIE_STATIC;

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
  if (!*cookie) *cookie = LARGEST_PETSC_COOKIE++;
  return 0;
}
