#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodataalias.c,v 1.1 1999/09/30 19:32:15 bsmith Exp bsmith $";
#endif

#include "src/dm/ao/aoimpl.h"      /*I "ao.h" I*/

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

  aoalias       = PetscNew(AODataAlias);CHKPTRQ(aoalias);
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
