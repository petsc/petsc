#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: itcl.c,v 1.108 1999/01/13 22:29:35 curfman Exp curfman $";
#endif
/*
    Code for setting KSP options from the options database.
*/

#include "src/ksp/kspimpl.h"  /*I "ksp.h" I*/
#include "sys.h"

/*
       We retain a list of functions that also take KSP command 
    line options. These are called at the end KSPSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
int numberofsetfromoptions;
int (*othersetfromoptions[MAXSETFROMOPTIONS])(KSP);

#undef __FUNC__  
#define __FUNC__ "KSPAddOptionsChecker"
/*@
    KSPAddOptionsChecker - Adds an additional function to check for KSP options.

    Not Collective

    Input Parameter:
.   kspcheck - function that checks for options

.keywords: KSP, add, options, checker

.seealso: KSPSetFromOptions()

    Level: developer
@*/
int KSPAddOptionsChecker(int (*kspcheck)(KSP) )
{
  PetscFunctionBegin;
  if (numberofsetfromoptions >= MAXSETFROMOPTIONS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many options checkers, only 5 allowed");
  }

  othersetfromoptions[numberofsetfromoptions++] = kspcheck;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPSetOptionsPrefix"
/*@C
   KSPSetOptionsPrefix - Sets the prefix used for searching for all 
   KSP options in the database.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different KSP contexts, one could call
.vb
      KSPSetOptionsPrefix(ksp1,"sys1_")
      KSPSetOptionsPrefix(ksp2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_ksp_type gmres -sys1_ksp_rtol 1.e-3
      -sys2_ksp_type bcgs  -sys2_ksp_rtol 1.e-4
.ve

.keywords: KSP, set, options, prefix, database

.seealso: KSPAppendOptionsPrefix(), KSPGetOptionsPrefix()

   Level: advanced
@*/
int KSPSetOptionsPrefix(KSP ksp,char *prefix)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ksp, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
 
#undef __FUNC__  
#define __FUNC__ "KSPAppendOptionsPrefix"
/*@C
   KSPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   KSP options in the database.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

.keywords: KSP, append, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPGetOptionsPrefix()

   Level: advanced
@*/
int KSPAppendOptionsPrefix(KSP ksp,char *prefix)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ksp, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPGetOptionsPrefix"
/*@
   KSPGetOptionsPrefix - Gets the prefix used for searching for all 
   KSP options in the database.

   Not Collective

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix()

   Level: advanced
@*/
int KSPGetOptionsPrefix(KSP ksp,char **prefix)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ksp, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

 



