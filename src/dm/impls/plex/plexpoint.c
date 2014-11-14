#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc-private/isimpl.h>     /* for inline access to atlasOff */

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLocalOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetLocalOffset_Private(DM dm,PetscInt point,PetscInt *offset)
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
    *offset = s->atlasOff[point - s->pStart];
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetLocalFieldOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetLocalFieldOffset_Private(DM dm,PetscInt point,PetscInt field,PetscInt *offset)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    ierr = PetscSectionGetFieldOffset(dm->defaultSection,point,field,offset);CHKERRQ(ierr);
  }
#else
  {
    PetscSection s = dm->defaultSection->field[field];
    *offset = s->atlasOff[point - s->pStart];
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetGlobalOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetGlobalOffset_Private(DM dm,PetscInt point,PetscInt *offset)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    PetscInt       dof,cdof;
    ierr = PetscSectionGetOffset(dm->defaultGlobalSection,point,offset);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(dm->defaultGlobalSection,point,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(dm->defaultGlobalSection,point,&cdof);CHKERRQ(ierr);
    if (dof-cdof <= 0) *offset = -1; /* Indicates no data */
  }
#else
  {
    PetscSection s = dm->defaultGlobalSection;
    PetscInt     dof,cdof;
    *offset = s->atlasOff[point - s->pStart];
    dof     = s->atlasDof[point - s->pStart];
    cdof    = s->bc ? s->bc->atlasDof[point - s->bc->pStart] : 0;
    if (dof-cdof <= 0) *offset = -1;
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetGlobalFieldOffset_Private"
PETSC_STATIC_INLINE PetscErrorCode DMPlexGetGlobalFieldOffset_Private(DM dm,PetscInt point,PetscInt field,PetscInt *offset)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscErrorCode ierr;
    PetscInt       loff,lfoff,dof,cdof;
    ierr = PetscSectionGetOffset(dm->defaultGlobalSection,point,offset);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(dm->defaultSection,point,&loff);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldOffset(dm->defaultSection,point,field,&lfoff);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(dm->defaultSection,point,field,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(dm->defaultSection,point,field,&cdof);CHKERRQ(ierr);
    *offset += lfoff - loff;
    if (dof-cdof <= 0) *offset = -1; /* Indicates no data */
  }
#else
  {
    PetscSection s  = dm->defaultSection;
    PetscSection fs = dm->defaultSection->field[field];
    PetscSection gs = dm->defaultGlobalSection;
    PetscInt     dof,cdof,loff,lfoff;
    loff    = s->atlasOff[point - s->pStart];
    lfoff   = fs->atlasOff[point - s->pStart];
    dof     = s->atlasDof[point - s->pStart];
    cdof    = s->bc ? s->bc->atlasDof[point - s->bc->pStart] : 0;
    *offset = gs->atlasOff[point - s->pStart] + lfoff - loff;
    if (dof-cdof <= 0) *offset = -1;
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetPointLocal"
/*@
   DMPlexGetPointLocal - get location of point data in local Vec

   Not Collective

   Input Arguments:
+  dm - DM defining the topological space
-  point - topological point

   Output Arguments:
+  start - start of point data
-  end - end of point data

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexPointLocalRead(), DMPlexPointLocalRead(), DMPlexPointLocalRef()
@*/
PetscErrorCode DMPlexGetPointLocal(DM dm,PetscInt point,PetscInt *start,PetscInt *end)
{
  PetscErrorCode ierr;
  PetscInt       offset,dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscSectionGetOffset(dm->defaultSection,point,&offset);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(dm->defaultSection,point,&dof);CHKERRQ(ierr);
  if (start) *start = offset;
  if (end) *end = offset + dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointLocalRead"
/*@
   DMPlexPointLocalRead - return read access to a point in local array

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
$  DMPlexPointLocalRead(dm,point,array,&ptr);
$  x = 2*ptr->foo + 3*ptr->bar + 5*ptr->baz;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointLocal(), DMPlexPointGlobalRead()
@*/
PetscErrorCode DMPlexPointLocalRead(DM dm,PetscInt point,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                      = DMPlexGetLocalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(const PetscScalar**)ptr = array + start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointLocalRef"
/*@
   DMPlexPointLocalRef - return read/write access to a point in local array

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
$  DMPlexPointLocalRef(dm,point,array,&ptr);
$  ptr->foo = 2; ptr->bar = 3; ptr->baz = 5;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointLocal(), DMPlexPointGlobalRef()
@*/
PetscErrorCode DMPlexPointLocalRef(DM dm,PetscInt point,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                = DMPlexGetLocalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(PetscScalar**)ptr = array + start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointLocalFieldRead"
/*@
   DMPlexPointLocalFieldRead - return read access to a field on a point in local array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Arguments:
.  ptr - address of read reference to point data, type generic so user can place in structure

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointLocal(), DMPlexPointGlobalRef()
@*/
PetscErrorCode DMPlexPointLocalFieldRead(DM dm,PetscInt point,PetscInt field,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                      = DMPlexGetLocalFieldOffset_Private(dm,point,field,&start);CHKERRQ(ierr);
  *(const PetscScalar**)ptr = array + start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointLocalFieldRef"
/*@
   DMPlexPointLocalFieldRef - return read/write access to a field on a point in local array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Arguments:
.  ptr - address of reference to point data, type generic so user can place in structure

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointLocal(), DMPlexPointGlobalRef()
@*/
PetscErrorCode DMPlexPointLocalFieldRef(DM dm,PetscInt point,PetscInt field,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                = DMPlexGetLocalFieldOffset_Private(dm,point,field,&start);CHKERRQ(ierr);
  *(PetscScalar**)ptr = array + start;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetPointGlobal"
/*@
   DMPlexGetPointGlobal - get location of point data in global Vec

   Not Collective

   Input Arguments:
+  dm - DM defining the topological space
-  point - topological point

   Output Arguments:
+  start - start of point data; returns -(global_start+1) if point is not owned
-  end - end of point data; returns -(global_end+1) if point is not owned

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexPointGlobalRead(), DMPlexGetPointLocal(), DMPlexPointGlobalRead(), DMPlexPointGlobalRef()
@*/
PetscErrorCode DMPlexGetPointGlobal(DM dm,PetscInt point,PetscInt *start,PetscInt *end)
{
  PetscErrorCode ierr;
  PetscInt       offset,dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscSectionGetOffset(dm->defaultGlobalSection,point,&offset);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(dm->defaultGlobalSection,point,&dof);CHKERRQ(ierr);
  if (start) *start = offset;
  if (end) *end = offset + dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointGlobalRead"
/*@
   DMPlexPointGlobalRead - return read access to a point in global array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
-  array - array to index into

   Output Arguments:
.  ptr - address of read reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:

$  const struct { PetscScalar foo,bar,baz; } *ptr;
$  DMPlexPointGlobalRead(dm,point,array,&ptr);
$  x = 2*ptr->foo + 3*ptr->bar + 5*ptr->baz;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointGlobal(), DMPlexPointLocalRead(), DMPlexPointGlobalRef()
@*/
PetscErrorCode DMPlexPointGlobalRead(DM dm,PetscInt point,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                      = DMPlexGetGlobalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(const PetscScalar**)ptr = (start >= 0) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointGlobalRef"
/*@
   DMPlexPointGlobalRef - return read/write access to a point in global array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
-  array - array to index into

   Output Arguments:
.  ptr - address of reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:

$  struct { PetscScalar foo,bar,baz; } *ptr;
$  DMPlexPointGlobalRef(dm,point,array,&ptr);
$  ptr->foo = 2; ptr->bar = 3; ptr->baz = 5;

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointGlobal(), DMPlexPointLocalRef(), DMPlexPointGlobalRead()
@*/
PetscErrorCode DMPlexPointGlobalRef(DM dm,PetscInt point,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                = DMPlexGetGlobalOffset_Private(dm,point,&start);CHKERRQ(ierr);
  *(PetscScalar**)ptr = (start >= 0) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointGlobalFieldRead"
/*@
   DMPlexPointGlobalFieldRead - return read access to a field on a point in global array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Arguments:
.  ptr - address of read reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointGlobal(), DMPlexPointLocalRead(), DMPlexPointGlobalRef()
@*/
PetscErrorCode DMPlexPointGlobalFieldRead(DM dm,PetscInt point,PetscInt field,const PetscScalar *array,const void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                      = DMPlexGetGlobalFieldOffset_Private(dm,point,field,&start);CHKERRQ(ierr);
  *(const PetscScalar**)ptr = (start >= 0) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexPointGlobalFieldRef"
/*@
   DMPlexPointGlobalFieldRef - return read/write access to a field on a point in global array

   Not Collective

   Input Arguments:
+  dm - DM defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Arguments:
.  ptr - address of reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

.seealso: DMGetDefaultSection(), PetscSectionGetOffset(), PetscSectionGetDof(), DMPlexGetPointGlobal(), DMPlexPointLocalRef(), DMPlexPointGlobalRead()
@*/
PetscErrorCode DMPlexPointGlobalFieldRef(DM dm,PetscInt point,PetscInt field,PetscScalar *array,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       start;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidScalarPointer(array,3);
  PetscValidPointer(ptr,4);
  ierr                = DMPlexGetGlobalFieldOffset_Private(dm,point,field,&start);CHKERRQ(ierr);
  *(PetscScalar**)ptr = (start >= 0) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(0);
}
