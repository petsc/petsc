#define PETSCDM_DLL

#include "src/dm/ao/aoimpl.h"      /*I "petscao.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "AODataAddAlias" 
/*@C
    AODataAddAlias - Allows accessing a key or field using an alternative
          name.

    Not collective

    Input Parameters:
+   ao - the AOData database
.   alias - substitute name that may be used
-   name - name the alias replaces

   Level: intermediate

.keywords: aliases, keys, fields

.seealso:  
@*/ 
PetscErrorCode PETSCDM_DLLEXPORT AODataAddAlias(AOData ao,char *alias,char *name)
{
  AODataAlias *aoalias,*t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AODATA_COOKIE,1);

  ierr          = PetscNew(AODataAlias,&aoalias);CHKERRQ(ierr);
  ierr          = PetscStrallocpy(alias,&aoalias->alias);CHKERRQ(ierr);
  ierr          = PetscStrallocpy(name,&aoalias->name);CHKERRQ(ierr);
  aoalias->next = PETSC_NULL;

  if (!ao->aliases) {
    ao->aliases = aoalias;
  } else {
    t = ao->aliases;
    while (t->next) t = t->next;
    t->next = aoalias;
  }

  PetscFunctionReturn(0);
}
