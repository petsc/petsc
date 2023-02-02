#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/

/*@
   DMPlexGetPointLocal - get location of point data in local `Vec`

   Not Collective

   Input Parameters:
+  dm - `DM` defining the topological space
-  point - topological point

   Output Parameters:
+  start - start of point data
-  end - end of point data

   Level: intermediate

   Note:
   This is a half open interval [start, end)

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexGetPointLocalField()`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexPointLocalRead()`, `DMPlexPointLocalRead()`, `DMPlexPointLocalRef()`
@*/
PetscErrorCode DMPlexGetPointLocal(DM dm, PetscInt point, PetscInt *start, PetscInt *end)
{
  PetscInt s, e;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) PetscValidIntPointer(start, 3);
  if (end) PetscValidIntPointer(end, 4);
  PetscCall(DMGetLocalOffset_Private(dm, point, &s, &e));
  if (start) *start = s;
  if (end) *end = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointLocalRead - return read access to a point in local array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
-  array - array to index into

   Output Parameter:
.  ptr - address of read reference to point data, type generic so user can place in structure

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:
.vb
  const struct { PetscScalar foo,bar,baz; } *ptr;
  DMPlexPointLocalRead(dm,point,array,&ptr);
  x = 2*ptr->foo + 3*ptr->bar + 5*ptr->baz;
.ve

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointLocal()`, `DMPlexPointGlobalRead()`
@*/
PetscErrorCode DMPlexPointLocalRead(DM dm, PetscInt point, const PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 3);
  PetscValidPointer(ptr, 4);
  PetscCall(DMGetLocalOffset_Private(dm, point, &start, &end));
  *(const PetscScalar **)ptr = (start < end) ? array + start : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointLocalRef - return read/write access to a point in local array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
-  array - array to index into

   Output Parameter:
.  ptr - address of reference to point data, type generic so user can place in structure

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:
.vb
  struct { PetscScalar foo,bar,baz; } *ptr;
  DMPlexPointLocalRef(dm,point,array,&ptr);
  ptr->foo = 2; ptr->bar = 3; ptr->baz = 5;
.ve

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointLocal()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexPointLocalRef(DM dm, PetscInt point, PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 3);
  PetscValidPointer(ptr, 4);
  PetscCall(DMGetLocalOffset_Private(dm, point, &start, &end));
  *(PetscScalar **)ptr = (start < end) ? array + start : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetPointLocalField - get location of point field data in local Vec

  Not Collective

  Input Parameters:
+ dm - `DM` defining the topological space
. point - topological point
- field - the field number

  Output Parameters:
+ start - start of point data
- end - end of point data

  Level: intermediate

  Note:
  This is a half open interval [start, end)

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexGetPointLocal()`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexPointLocalRead()`, `DMPlexPointLocalRead()`, `DMPlexPointLocalRef()`
@*/
PetscErrorCode DMPlexGetPointLocalField(DM dm, PetscInt point, PetscInt field, PetscInt *start, PetscInt *end)
{
  PetscInt s, e;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) PetscValidIntPointer(start, 4);
  if (end) PetscValidIntPointer(end, 5);
  PetscCall(DMGetLocalFieldOffset_Private(dm, point, field, &s, &e));
  if (start) *start = s;
  if (end) *end = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointLocalFieldRead - return read access to a field on a point in local array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Parameter:
.  ptr - address of read reference to point data, type generic so user can place in structure

   Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointLocal()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexPointLocalFieldRead(DM dm, PetscInt point, PetscInt field, const PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 4);
  PetscValidPointer(ptr, 5);
  PetscCall(DMGetLocalFieldOffset_Private(dm, point, field, &start, &end));
  *(const PetscScalar **)ptr = array + start;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointLocalFieldRef - return read/write access to a field on a point in local array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Parameter:
.  ptr - address of reference to point data, type generic so user can place in structure

   Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointLocal()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexPointLocalFieldRef(DM dm, PetscInt point, PetscInt field, PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 4);
  PetscValidPointer(ptr, 5);
  PetscCall(DMGetLocalFieldOffset_Private(dm, point, field, &start, &end));
  *(PetscScalar **)ptr = array + start;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetPointGlobal - get location of point data in global Vec

  Not Collective

  Input Parameters:
+ dm - `DM` defining the topological space
- point - topological point

  Output Parameters:
+ start - start of point data; returns -(globalStart+1) if point is not owned
- end - end of point data; returns -(globalEnd+1) if point is not owned

  Level: intermediate

  Note:
  This is a half open interval [start, end)

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexGetPointGlobalField()`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexPointGlobalRead()`, `DMPlexGetPointLocal()`, `DMPlexPointGlobalRead()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexGetPointGlobal(DM dm, PetscInt point, PetscInt *start, PetscInt *end)
{
  PetscInt s, e;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) PetscValidIntPointer(start, 3);
  if (end) PetscValidIntPointer(end, 4);
  PetscCall(DMGetGlobalOffset_Private(dm, point, &s, &e));
  if (start) *start = s;
  if (end) *end = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointGlobalRead - return read access to a point in global array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
-  array - array to index into

   Output Parameter:
.  ptr - address of read reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:
.vb
  const struct { PetscScalar foo,bar,baz; } *ptr;
  DMPlexPointGlobalRead(dm,point,array,&ptr);
  x = 2*ptr->foo + 3*ptr->bar + 5*ptr->baz;
.ve

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointGlobal()`, `DMPlexPointLocalRead()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexPointGlobalRead(DM dm, PetscInt point, const PetscScalar *array, const void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 3);
  PetscValidPointer(ptr, 4);
  PetscCall(DMGetGlobalOffset_Private(dm, point, &start, &end));
  *(const PetscScalar **)ptr = (start < end) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointGlobalRef - return read/write access to a point in global array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
-  array - array to index into

   Output Parameter:
.  ptr - address of reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

   Note:
   A common usage when data sizes are known statically:
.vb
  struct { PetscScalar foo,bar,baz; } *ptr;
  DMPlexPointGlobalRef(dm,point,array,&ptr);
  ptr->foo = 2; ptr->bar = 3; ptr->baz = 5;
.ve

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointGlobal()`, `DMPlexPointLocalRef()`, `DMPlexPointGlobalRead()`
@*/
PetscErrorCode DMPlexPointGlobalRef(DM dm, PetscInt point, PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 3);
  PetscValidPointer(ptr, 4);
  PetscCall(DMGetGlobalOffset_Private(dm, point, &start, &end));
  *(PetscScalar **)ptr = (start < end) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexGetPointGlobalField - get location of point field data in global `Vec`

  Not Collective

  Input Parameters:
+ dm - `DM` defining the topological space
. point - topological point
- field - the field number

  Output Parameters:
+ start - start of point data; returns -(globalStart+1) if point is not owned
- end - end of point data; returns -(globalEnd+1) if point is not owned

  Level: intermediate

  Note:
  This is a half open interval [start, end)

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexGetPointGlobal()`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexPointGlobalRead()`, `DMPlexGetPointLocal()`, `DMPlexPointGlobalRead()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexGetPointGlobalField(DM dm, PetscInt point, PetscInt field, PetscInt *start, PetscInt *end)
{
  PetscInt s, e;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) PetscValidIntPointer(start, 4);
  if (end) PetscValidIntPointer(end, 5);
  PetscCall(DMGetGlobalFieldOffset_Private(dm, point, field, &s, &e));
  if (start) *start = s;
  if (end) *end = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointGlobalFieldRead - return read access to a field on a point in global array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Parameter:
.  ptr - address of read reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointGlobal()`, `DMPlexPointLocalRead()`, `DMPlexPointGlobalRef()`
@*/
PetscErrorCode DMPlexPointGlobalFieldRead(DM dm, PetscInt point, PetscInt field, const PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 4);
  PetscValidPointer(ptr, 5);
  PetscCall(DMGetGlobalFieldOffset_Private(dm, point, field, &start, &end));
  *(const PetscScalar **)ptr = (start < end) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMPlexPointGlobalFieldRef - return read/write access to a field on a point in global array

   Not Collective

   Input Parameters:
+  dm - `DM` defining topological space
.  point - topological point
.  field - field number
-  array - array to index into

   Output Parameter:
.  ptr - address of reference to point data, type generic so user can place in structure; returns NULL if global point is not owned

   Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMGetLocalSection()`, `PetscSectionGetOffset()`, `PetscSectionGetDof()`, `DMPlexGetPointGlobal()`, `DMPlexPointLocalRef()`, `DMPlexPointGlobalRead()`
@*/
PetscErrorCode DMPlexPointGlobalFieldRef(DM dm, PetscInt point, PetscInt field, PetscScalar *array, void *ptr)
{
  PetscInt start, end;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidScalarPointer(array, 4);
  PetscValidPointer(ptr, 5);
  PetscCall(DMGetGlobalFieldOffset_Private(dm, point, field, &start, &end));
  *(PetscScalar **)ptr = (start < end) ? array + start - dm->map->rstart : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
