#include <petsc/private/petscimpl.h>  /*I   "petscsys.h"    I*/

typedef struct _FortranCallbackLink *FortranCallbackLink;
struct _FortranCallbackLink {
  char                   *type_name;
  PetscFortranCallbackId max;
  FortranCallbackLink    next;
};

typedef struct {
  PetscInt            basecount;
  PetscInt            maxsubtypecount;
  FortranCallbackLink subtypes;
} FortranCallbackBase;

static FortranCallbackBase *_classbase;
static PetscClassId        _maxclassid = PETSC_SMALLEST_CLASSID;

static PetscErrorCode PetscFortranCallbackFinalize(void)
{
  PetscErrorCode ierr;
  PetscClassId   i;

  PetscFunctionBegin;
  for (i=PETSC_SMALLEST_CLASSID; i<_maxclassid; i++) {
    FortranCallbackBase *base = &_classbase[i-PETSC_SMALLEST_CLASSID];
    FortranCallbackLink next,link = base->subtypes;
    for (; link; link=next) {
      next = link->next;
      ierr = PetscFree(link->type_name);CHKERRQ(ierr);
      ierr = PetscFree(link);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(_classbase);CHKERRQ(ierr);

  _maxclassid = PETSC_SMALLEST_CLASSID;
  PetscFunctionReturn(0);
}

/*@C
   PetscFortranCallbackRegister - register a type+subtype callback

   Not Collective

   Input Parameters:
+  classid - ID of class on which to register callback
-  subtype - subtype string, or NULL for class ids

   Output Parameter:
.  id - callback id

   Level: developer

.seealso: PetscFortranCallbackGetSizes()
@*/
PetscErrorCode PetscFortranCallbackRegister(PetscClassId classid,const char *subtype,PetscFortranCallbackId *id)
{
  PetscErrorCode      ierr;
  FortranCallbackBase *base;
  FortranCallbackLink link;

  PetscFunctionBegin;
  *id = 0;
  if (classid < PETSC_SMALLEST_CLASSID || PETSC_LARGEST_CLASSID < classid) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"ClassId %D corrupt",classid);
  if (classid >= _maxclassid) {
    PetscClassId        newmax = PETSC_SMALLEST_CLASSID + 2*(PETSC_LARGEST_CLASSID-PETSC_SMALLEST_CLASSID);
    FortranCallbackBase *newbase;
    if (!_classbase) {
      ierr = PetscRegisterFinalize(PetscFortranCallbackFinalize);CHKERRQ(ierr);
    }
    ierr = PetscCalloc1(newmax-PETSC_SMALLEST_CLASSID,&newbase);CHKERRQ(ierr);
    ierr = PetscArraycpy(newbase,_classbase,_maxclassid-PETSC_SMALLEST_CLASSID);CHKERRQ(ierr);
    ierr = PetscFree(_classbase);CHKERRQ(ierr);

    _classbase = newbase;
    _maxclassid = newmax;
  }
  base = &_classbase[classid-PETSC_SMALLEST_CLASSID];
  if (!subtype) *id = PETSC_SMALLEST_FORTRAN_CALLBACK + base->basecount++;
  else {
    for (link=base->subtypes; link; link=link->next) { /* look for either both NULL or matching values (implies both non-NULL) */
      PetscBool match;
      ierr = PetscStrcmp(subtype,link->type_name,&match);CHKERRQ(ierr);
      if (match) { /* base type or matching subtype */
        goto found;
      }
    }
    /* Not found. Create node and prepend to class' subtype list */
    ierr = PetscNew(&link);CHKERRQ(ierr);
    ierr = PetscStrallocpy(subtype,&link->type_name);CHKERRQ(ierr);

    link->max      = PETSC_SMALLEST_FORTRAN_CALLBACK;
    link->next     = base->subtypes;
    base->subtypes = link;

found:
    *id = link->max++;

    base->maxsubtypecount = PetscMax(base->maxsubtypecount,link->max-PETSC_SMALLEST_FORTRAN_CALLBACK);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscFortranCallbackGetSizes - get sizes of class and subtype pointer arrays

   Collective

   Input Parameter:
.  classid - class Id

   Output Parameters:
+  numbase - number of registered class callbacks
-  numsubtype - max number of registered subtype callbacks

   Level: developer

.seealso: PetscFortranCallbackRegister()
@*/
PetscErrorCode PetscFortranCallbackGetSizes(PetscClassId classid,PetscInt *numbase,PetscInt *numsubtype)
{

  PetscFunctionBegin;
  if (classid < _maxclassid) {
    FortranCallbackBase *base = &_classbase[classid-PETSC_SMALLEST_CLASSID];
    *numbase    = base->basecount;
    *numsubtype = base->maxsubtypecount;
  } else {                      /* nothing registered */
    *numbase    = 0;
    *numsubtype = 0;
  }
  PetscFunctionReturn(0);
}
