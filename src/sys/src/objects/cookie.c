/*$Id: cookie.c,v 1.24 2000/10/24 20:24:36 bsmith Exp bsmith $*/

#include "petsc.h"  /*I "petsc.h" I*/
int PETSC_LARGEST_COOKIE = PETSC_LARGEST_COOKIE_PREDEFINED;

#undef __FUNC__  
#define __FUNC__ "PetscRegisterCookie"
/*@
    PetscRegisterCookie - Registers a new cookie for use with a
    newly created PETSc object class.  The user should pass in
    a variable initialized to zero; then it will be assigned a cookie.
    Repeated calls to this routine with the same variable will 
    not change the cookie. 

    Not Collective

    Output Parameter:
.   cookie - the cookie you have been assigned

    Level: developer

    Note:
    The initial cookie variable MUST be set to zero on the
    first call to this routine.

    Concepts: cookie^getting new one

@*/
int PetscRegisterCookie(int *cookie)
{
  PetscFunctionBegin;
  if (PETSC_LARGEST_COOKIE >= PETSC_LARGEST_COOKIE_ALLOWED) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"You have used too many PETSc cookies");
  }
  if (!*cookie) *cookie = PETSC_LARGEST_COOKIE++;
  PetscFunctionReturn(0);
}
