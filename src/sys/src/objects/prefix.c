#ifndef lint
static char vcid[] = "$Id: prefix.c,v 1.2 1996/02/08 18:26:06 bsmith Exp balay $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

/*
   PetscObjectSetPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

.keywords: object, set, options, prefix, database
*/
int PetscObjectSetPrefix(PetscObject obj, char *prefix)
{
  if (obj->prefix) PetscFree(obj->prefix);
  if (prefix == PETSC_NULL) {obj->prefix = PETSC_NULL; return 0;}
  obj->prefix = (char*) PetscMalloc((1+PetscStrlen(prefix))* 
                sizeof(char)); CHKPTRQ(obj->prefix);
  PetscStrcpy(obj->prefix,prefix);
  return 0;
}

/*
   PetscObjectAppendPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

.keywords: object, append, options, prefix, database
*/
int PetscObjectAppendPrefix(PetscObject obj, char *prefix)
{
  char *buf = obj->prefix ;
  if (!prefix) {return 0;}
  if (!buf) return PetscObjectSetPrefix(obj, prefix);
  obj->prefix = (char*)PetscMalloc((1 + PetscStrlen(prefix) + PetscStrlen(buf))*
                sizeof(char));  CHKPTRQ(obj->prefix);
  PetscStrcpy(obj->prefix,buf);
  PetscStrcat(obj->prefix,prefix);
  PetscFree(buf);
  return 0;
}

/*
   PetscObjectGetPrefix - Gets the prefix of the PetscObject.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.keywords: object, get, options, prefix, database
*/
int PetscObjectGetPrefix(PetscObject obj ,char** prefix)
{
  *prefix = obj->prefix;
  return 0;
}


