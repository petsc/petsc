/*$Id: cookie.c,v 1.26 2001/03/23 23:20:38 balay Exp $*/

#include "petscconfig.h"
#include "petsc.h"  /*I "petsc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscRegisterCookie"
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
  SETERRQ(PETSC_ERR_SUP, "This function is now obsolete. Please use PetscLogClassRegister().");
}
