#ifndef lint
static char vcid[] = "$Id: prefix.c,v 1.6 1997/01/06 20:22:55 balay Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetOptionsPrefix" /* ADIC Ignore */
/*
   PetscObjectSetOptionsPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database. You must NOT include the - at the beginning of 
   the prefix name.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

.keywords: object, set, options, prefix, database
*/
int PetscObjectSetOptionsPrefix(PetscObject obj, char *prefix)
{
  if (obj->prefix) PetscFree(obj->prefix);
  if (prefix == PETSC_NULL) {obj->prefix = PETSC_NULL; return 0;}
  obj->prefix = (char*) PetscMalloc((1+PetscStrlen(prefix))* 
                sizeof(char)); CHKPTRQ(obj->prefix);
  PetscStrcpy(obj->prefix,prefix);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectAppendOptionsPrefix" /* ADIC Ignore */
/*
   PetscObjectAppendOptionsPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database. You must NOT include the - at the beginning of 
   the prefix name.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

.keywords: object, append, options, prefix, database
*/
int PetscObjectAppendOptionsPrefix(PetscObject obj, char *prefix)
{
  char *buf = obj->prefix ;
  if (!prefix) {return 0;}
  if (!buf) return PetscObjectSetOptionsPrefix(obj, prefix);
  obj->prefix = (char*)PetscMalloc((1 + PetscStrlen(prefix) + PetscStrlen(buf))*
                sizeof(char));  CHKPTRQ(obj->prefix);
  PetscStrcpy(obj->prefix,buf);
  PetscStrcat(obj->prefix,prefix);
  PetscFree(buf);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetOptionsPrefix" /* ADIC Ignore */
/*
   PetscObjectGetOptionsPrefix - Gets the prefix of the PetscObject.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.keywords: object, get, options, prefix, database
*/
int PetscObjectGetOptionsPrefix(PetscObject obj ,char** prefix)
{
  *prefix = obj->prefix;
  return 0;
}


