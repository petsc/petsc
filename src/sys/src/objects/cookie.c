#ifndef lint
static char vcid[] = "$Id: gcookie.c,v 1.2 1996/02/08 18:26:06 bsmith Exp $";
#endif

int LARGEST_PETSC_COOKIE = 30;

/*@
     PetscRegisterCookie - Registers a new cookie for use 
        with a newly created PETSc object class. You should pass
        in a variable preset to zero and it will be assigned a cookie,
        repeated calls to this routine with the same variable will 
        not change the cookie. 

  Output Parameter:
.   cookie - the cookie you have been assigned

@*/
int PetscRegisterCookie(int *cookie)
{
  if (!*cookie) *cookie = LARGEST_PETSC_COOKIE++;
  return 0;
}
