/*$Id: prefix.c,v 1.23 1999/11/24 21:53:05 bsmith Exp bsmith $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetOptionsPrefix"
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

.keywords: object, set, options, prefix, database
*/
int PetscObjectSetOptionsPrefix(PetscObject obj,const char prefix[])
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(obj->prefix);CHKERRQ(ierr);
  if (!prefix) {
    obj->prefix = PETSC_NULL;
  } else {
     if (prefix[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,1,"Options prefix should not begin with a hypen");
    ierr  = PetscStrallocpy(prefix,&obj->prefix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectAppendOptionsPrefix"
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

.keywords: object, append, options, prefix, database
*/
int PetscObjectAppendOptionsPrefix(PetscObject obj,const char prefix[])
{
  char *buf = obj->prefix ;
  int  ierr,len1,len2;

  PetscFunctionBegin;
  if (!prefix) {PetscFunctionReturn(0);}
  if (!buf) {
    ierr = PetscObjectSetOptionsPrefix(obj,prefix);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (prefix[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,1,"Options prefix should not begin with a hypen");

  ierr        = PetscStrlen(prefix,&len1);CHKERRQ(ierr);
  ierr        = PetscStrlen(buf,&len2);CHKERRQ(ierr);
  obj->prefix = (char*)PetscMalloc((1+len1+len2)*sizeof(char));CHKPTRQ(obj->prefix);
  ierr        = PetscStrcpy(obj->prefix,buf);CHKERRQ(ierr);
  ierr        = PetscStrcat(obj->prefix,prefix);CHKERRQ(ierr);
  ierr        = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetOptionsPrefix"
/*
   PetscObjectGetOptionsPrefix - Gets the prefix of the PetscObject.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.keywords: object, get, options, prefix, database
*/
int PetscObjectGetOptionsPrefix(PetscObject obj,char *prefix[])
{
  PetscFunctionBegin;
  *prefix = obj->prefix;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectPrependOptionsPrefix"
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

.keywords: object, append, options, prefix, database
*/
int PetscObjectPrependOptionsPrefix(PetscObject obj,const char prefix[])
{
  char *buf = obj->prefix ;
  int  ierr,len1,len2;

  PetscFunctionBegin;
  if (!prefix) {PetscFunctionReturn(0);}
  if (!buf) {
    ierr = PetscObjectSetOptionsPrefix(obj,prefix);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (prefix[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,1,"Options prefix should not begin with a hypen");

  ierr        = PetscStrlen(prefix,&len1);CHKERRQ(ierr);
  ierr        = PetscStrlen(buf,&len2);CHKERRQ(ierr);
  obj->prefix = (char*)PetscMalloc((1+len1+len2)*sizeof(char));CHKPTRQ(obj->prefix);
  ierr        = PetscStrcpy(obj->prefix,prefix);CHKERRQ(ierr);
  ierr        = PetscStrcat(obj->prefix,buf);CHKERRQ(ierr);
  ierr        = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

