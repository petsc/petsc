#include <petsc-private/compleximpl.h>   /*I      "petscdmcomplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetLocalOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMComplexGetLocalOffset_Private(DM dm,PetscInt point,PetscInt *offset)
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
#define __FUNCT__ "DMComplexGetGlobalOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMComplexGetGlobalOffset_Private(DM dm,PetscInt point,PetscInt *offset)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    PetscInt dof,cdof;
    ierr = PetscSectionGetOffset(dm->defaultGlobalSection,point,offset);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(dm->defaultGlobalSection,point,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(dm->defaultGlobalSection,point,&cdof);CHKERRQ(ierr);
    if (dof-cdof <= 0) *offset = -1; /* Indicates no data */
  }
#else
  {
    PetscSection s = dm->defaultGlobalSection;
    PetscInt dof,cdof;
    *offset = s->atlasOff[point - s->atlasLayout.pStart];
    dof = s->atlasDof[point - s->atlasLayout.pStart];
    cdof = s->bc ? s->bc->atlasDof[point - s->bc->atlasLayout.pStart] : 0;
    if (dof-cdof <= 0) *offset = -1;
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

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexPointLocalRead(), DMComplexPointLocalRead(), DMComplexPointLocalRef()
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

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexGetPointLocal(), DMComplexPointGlobalRead()
@*/
PetscErrorCode DMComplexPointLocalRead(DM dm,PetscInt point,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr = DMComplexGetLocalOffset_Private(dm,point,&start);CHKERRQ(ierr);
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

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexGetPointLocal(), DMComplexPointGlobalRef()
@*/
PetscErrorCode DMComplexPointLocalRef(DM dm,PetscInt point,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr = DMComplexGetLocalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(PetscScalar **)ptr = array + start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexGetPointGlobal"
/*@
   DMComplexGetPointGlobal - get location of point data in global Vec

   Not Collective

   Input Arguments:
+  dm - DM defining the topological space
-  point - topological point

   Output Arguments:
+  start - start of point data; returns -(global_start+1) if point is not owned
-  end - end of point data; returns -(global_end+1) if point is not owned

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexPointGlobalRead(), DMComplexGetPointLocal(), DMComplexPointGlobalRead(), DMComplexPointGlobalRef()
@*/
PetscErrorCode DMComplexGetPointGlobal(DM dm,PetscInt point,PetscInt *start,PetscInt *end)
{
  PetscErrorCode ierr;
  PetscInt offset,dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscSectionGetOffset(dm->defaultGlobalSection,point,&offset);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(dm->defaultGlobalSection,point,&dof);CHKERRQ(ierr);
  if (start) *start = offset;
  if (end) *end = offset + dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexPointGlobalRead"
/*@
   DMComplexPointGlobalRead - return read access to a point in global array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
-  array - array to index into

   Output Arguments:
.  ptr - address of read reference to point data, type generic so user can place in structure; returns PETSC_NULL if global point is not owned

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:

$  const struct { PetscScalar foo,bar,baz; } *ptr;
$  DMComplexPointGlobalRead(dm,point,array,&ptr);
$  x = 2*ptr->foo + 3*ptr->bar + 5*ptr->baz;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexGetPointGlobal(), DMComplexPointLocalRead(), DMComplexPointGlobalRef()
@*/
PetscErrorCode DMComplexPointGlobalRead(DM dm,PetscInt point,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr = DMComplexGetGlobalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(const PetscScalar **)ptr = (start >= 0) ? array + start - dm->map->rstart : PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMComplexPointGlobalRef"
/*@
   DMComplexPointGlobalRef - return read/write access to a point in global array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
-  array - array to index into

   Output Arguments:
.  ptr - address of reference to point data, type generic so user can place in structure; returns PETSC_NULL if global point is not owned

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:

$  struct { PetscScalar foo,bar,baz; } *ptr;
$  DMComplexPointGlobalRef(dm,point,array,&ptr);
$  ptr->foo = 2; ptr->bar = 3; ptr->baz = 5;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMComplexGetPointGlobal(), DMComplexPointLocalRef(), DMComplexPointGlobalRead()
@*/
PetscErrorCode DMComplexPointGlobalRef(DM dm,PetscInt point,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr = DMComplexGetGlobalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(PetscScalar **)ptr = (start >= 0) ? array + start - dm->map->rstart : PETSC_NULL;
  PetscFunctionReturn(0);
}
