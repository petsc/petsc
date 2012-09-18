
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/

#undef __FUNCT__
#define __FUNCT__ "PetscObjectSetOptionsPrefix"
/*
   PetscObjectSetOptionsPrefix - Sets the prefix used for searching for all
   options of PetscObjectType in the database.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Concepts: prefix^setting

*/
PetscErrorCode  PetscObjectSetOptionsPrefix(PetscObject obj,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (!prefix) {
    ierr = PetscFree(obj->prefix);CHKERRQ(ierr);
  } else {
    if (prefix[0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hypen");
    ierr = PetscFree(obj->prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&obj->prefix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAppendOptionsPrefix"
/*
   PetscObjectAppendOptionsPrefix - Sets the prefix used for searching for all
   options of PetscObjectType in the database.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Concepts: prefix^setting

*/
PetscErrorCode  PetscObjectAppendOptionsPrefix(PetscObject obj,const char prefix[])
{
  char   *buf = obj->prefix;
  PetscErrorCode ierr;
  size_t len1,len2;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (!prefix) {PetscFunctionReturn(0);}
  if (!buf) {
    ierr = PetscObjectSetOptionsPrefix(obj,prefix);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (prefix[0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hypen");

  ierr  = PetscStrlen(prefix,&len1);CHKERRQ(ierr);
  ierr  = PetscStrlen(buf,&len2);CHKERRQ(ierr);
  ierr  = PetscMalloc((1+len1+len2)*sizeof(char),&obj->prefix);CHKERRQ(ierr);
  ierr  = PetscStrcpy(obj->prefix,buf);CHKERRQ(ierr);
  ierr  = PetscStrcat(obj->prefix,prefix);CHKERRQ(ierr);
  ierr  = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectGetOptionsPrefix"
/*
   PetscObjectGetOptionsPrefix - Gets the prefix of the PetscObject.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Concepts: prefix^getting

*/
PetscErrorCode  PetscObjectGetOptionsPrefix(PetscObject obj,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidPointer(prefix,2);
  *prefix = obj->prefix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectPrependOptionsPrefix"
/*
   PetscObjectPrependOptionsPrefix - Sets the prefix used for searching for all
   options of PetscObjectType in the database.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Concepts: prefix^setting

*/
PetscErrorCode  PetscObjectPrependOptionsPrefix(PetscObject obj,const char prefix[])
{
  char           *buf;
  size_t         len1,len2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  buf = obj->prefix;
  if (!prefix) {PetscFunctionReturn(0);}
  if (!buf) {
    ierr = PetscObjectSetOptionsPrefix(obj,prefix);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (prefix[0] == '-') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Options prefix should not begin with a hypen");

  ierr = PetscStrlen(prefix,&len1);CHKERRQ(ierr);
  ierr = PetscStrlen(buf,&len2);CHKERRQ(ierr);
  ierr = PetscMalloc((1+len1+len2)*sizeof(char),&obj->prefix);CHKERRQ(ierr);
  ierr = PetscStrcpy(obj->prefix,prefix);CHKERRQ(ierr);
  ierr = PetscStrcat(obj->prefix,buf);CHKERRQ(ierr);
  ierr = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

