#include <petsc-private/compleximpl.h>   /*I      "petscdmcomplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMComplexGetOffset_Private(DM dm,PetscInt point,PetscInt *offset)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    ierr = PetscSectionGetOffset(dm->defaultSection,point,offset);CHKERRQ(ierr);
  }
#else
  {
    PetscSection s = dm->defaultSection;
    *offset = s->atlasOff[point - s->atlasLayout.pStart];
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetPointLocal"
/*@
   DMComplexGetPointLocal - get location of point data in local Vec

   Not Collective

   Input Arguments:
+  dm - DM defining the topological space
-  point - topological point

   Output Arguments:
+  start - start of point data
-  end - end of point data

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexPointLocalRead()
@*/
PetscErrorCode DMComplexGetPointLocal(DM dm,PetscInt point,PetscInt *start,PetscInt *end)
{
  PetscErrorCode ierr;
  PetscInt offset,dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscSectionGetOffset(dm->defaultSection,point,&offset);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(dm->defaultSection,point,&dof);CHKERRQ(ierr);
  if (start) *start = offset;
  if (end) *end = offset + dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexPointLocalRead"
/*@
   DMComplexPointLocalRead - return read access to a point in local array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
-  array - array to index into

   Output Arguments:
.  ptr - address of read reference to point data, type generic so user can place in structure

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:

$  const struct { PetscScalar foo,bar,baz; } *ptr;
$  DMComplexPointLocalRead(dm,point,array,&ptr);
$  x = 2*ptr->foo + 3*ptr->bar + 5*ptr->baz;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexGetPointLocal()
@*/
PetscErrorCode DMComplexPointLocalRead(DM dm,PetscInt point,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr = DMComplexGetOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(const PetscScalar **)ptr = array + start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexPointLocalRef"
/*@
   DMComplexPointLocalRef - return read/write access to a point in local array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
-  array - array to index into

   Output Arguments:
.  ptr - address of reference to point data, type generic so user can place in structure

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:

$  struct { PetscScalar foo,bar,baz; } *ptr;
$  DMComplexPointLocalRef(dm,point,array,&ptr);
$  ptr->foo = 2; ptr->bar = 3; ptr->baz = 5;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexGetPointLocal()
@*/
PetscErrorCode DMComplexPointLocalRef(DM dm,PetscInt point,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr = DMComplexGetOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(PetscScalar **)ptr = array + start;
  PetscFunctionReturn(0);
}
