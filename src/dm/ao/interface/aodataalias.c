/*$Id: aodataalias.c,v 1.6 2000/05/05 22:19:09 balay Exp bsmith $*/

#include "src/dm/ao/aoimpl.h"      /*I "petscao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AODataAddAlias" 
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
int AODataAddAlias(AOData ao,char *alias,char *name)
{
  AODataAlias *aoalias,*t;
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AODATA_COOKIE);

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
