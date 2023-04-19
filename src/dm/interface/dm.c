#include <petscvec.h>
#include <petsc/private/dmimpl.h>      /*I      "petscdm.h"          I*/
#include <petsc/private/dmlabelimpl.h> /*I      "petscdmlabel.h"     I*/
#include <petsc/private/petscdsimpl.h> /*I      "petscds.h"     I*/
#include <petscdmplex.h>
#include <petscdmfield.h>
#include <petscsf.h>
#include <petscds.h>

#ifdef PETSC_HAVE_LIBCEED
  #include <petscfeceed.h>
#endif

#if !defined(PETSC_HAVE_WINDOWS_COMPILERS)
  #include <petsc/private/valgrind/memcheck.h>
#endif

PetscClassId DM_CLASSID;
PetscClassId DMLABEL_CLASSID;
PetscLogEvent DM_Convert, DM_GlobalToLocal, DM_LocalToGlobal, DM_LocalToLocal, DM_LocatePoints, DM_Coarsen, DM_Refine, DM_CreateInterpolation, DM_CreateRestriction, DM_CreateInjection, DM_CreateMatrix, DM_CreateMassMatrix, DM_Load, DM_AdaptInterpolator;

const char *const DMBoundaryTypes[]          = {"NONE", "GHOSTED", "MIRROR", "PERIODIC", "TWIST", "DMBoundaryType", "DM_BOUNDARY_", NULL};
const char *const DMBoundaryConditionTypes[] = {"INVALID", "ESSENTIAL", "NATURAL", "INVALID", "INVALID", "ESSENTIAL_FIELD", "NATURAL_FIELD", "INVALID", "INVALID", "ESSENTIAL_BD_FIELD", "NATURAL_RIEMANN", "DMBoundaryConditionType", "DM_BC_", NULL};
const char *const DMBlockingTypes[]          = {"TOPOLOGICAL_POINT", "FIELD_NODE", "DMBlockingType", "DM_BLOCKING_", NULL};
const char *const DMPolytopeTypes[]   = {"vertex",  "segment",       "tensor_segment",      "triangle", "quadrilateral", "tensor_quad",    "tetrahedron",  "hexahedron", "triangular_prism", "tensor_triangular_prism", "tensor_quadrilateral_prism",
                                         "pyramid", "FV_ghost_cell", "interior_ghost_cell", "unknown",  "invalid",       "DMPolytopeType", "DM_POLYTOPE_", NULL};
const char *const DMCopyLabelsModes[] = {"replace", "keep", "fail", "DMCopyLabelsMode", "DM_COPY_LABELS_", NULL};

/*@
  DMCreate - Creates an empty `DM` object. `DM`s are the abstract objects in PETSc that mediate between meshes and discretizations and the
  algebraic solvers, time integrators, and optimization algorithms.

  Collective

  Input Parameter:
. comm - The communicator for the `DM` object

  Output Parameter:
. dm - The `DM` object

  Level: beginner

  Notes:
  See `DMType` for a brief summary of available `DM`.

  The type must then be set with `DMSetType()`. If you never call `DMSetType()` it will generate an
  error when you try to use the dm.

.seealso: [](chapter_dmbase), `DM`, `DMSetType()`, `DMType`, `DMDACreate()`, `DMDA`, `DMSLICED`, `DMCOMPOSITE`, `DMPLEX`, `DMMOAB`, `DMNETWORK`
@*/
PetscErrorCode DMCreate(MPI_Comm comm, DM *dm)
{
  DM      v;
  PetscDS ds;

  PetscFunctionBegin;
  PetscValidPointer(dm, 2);
  *dm = NULL;
  PetscCall(DMInitializePackage());

  PetscCall(PetscHeaderCreate(v, DM_CLASSID, "DM", "Distribution Manager", "DM", comm, DMDestroy, DMView));

  ((PetscObject)v)->non_cyclic_references = &DMCountNonCyclicReferences;

  v->setupcalled          = PETSC_FALSE;
  v->setfromoptionscalled = PETSC_FALSE;
  v->ltogmap              = NULL;
  v->bind_below           = 0;
  v->bs                   = 1;
  v->coloringtype         = IS_COLORING_GLOBAL;
  PetscCall(PetscSFCreate(comm, &v->sf));
  PetscCall(PetscSFCreate(comm, &v->sectionSF));
  v->labels                    = NULL;
  v->adjacency[0]              = PETSC_FALSE;
  v->adjacency[1]              = PETSC_TRUE;
  v->depthLabel                = NULL;
  v->celltypeLabel             = NULL;
  v->localSection              = NULL;
  v->globalSection             = NULL;
  v->defaultConstraint.section = NULL;
  v->defaultConstraint.mat     = NULL;
  v->defaultConstraint.bias    = NULL;
  v->coordinates[0].dim        = PETSC_DEFAULT;
  v->coordinates[1].dim        = PETSC_DEFAULT;
  v->sparseLocalize            = PETSC_TRUE;
  v->dim                       = PETSC_DETERMINE;
  {
    PetscInt i;
    for (i = 0; i < 10; ++i) {
      v->nullspaceConstructors[i]     = NULL;
      v->nearnullspaceConstructors[i] = NULL;
    }
  }
  PetscCall(PetscDSCreate(PETSC_COMM_SELF, &ds));
  PetscCall(DMSetRegionDS(v, NULL, NULL, ds, NULL));
  PetscCall(PetscDSDestroy(&ds));
  PetscCall(PetscHMapAuxCreate(&v->auxData));
  v->dmBC              = NULL;
  v->coarseMesh        = NULL;
  v->outputSequenceNum = -1;
  v->outputSequenceVal = 0.0;
  PetscCall(DMSetVecType(v, VECSTANDARD));
  PetscCall(DMSetMatType(v, MATAIJ));

  *dm = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClone - Creates a `DM` object with the same topology as the original.

  Collective

  Input Parameter:
. dm - The original `DM` object

  Output Parameter:
. newdm  - The new `DM` object

  Level: beginner

  Notes:
  For some `DM` implementations this is a shallow clone, the result of which may share (reference counted) information with its parent. For example,
  `DMClone()` applied to a `DMPLEX` object will result in a new `DMPLEX` that shares the topology with the original `DMPLEX`. It does not
  share the `PetscSection` of the original `DM`.

  The clone is considered set up if the original has been set up.

  Use `DMConvert()` for a general way to create new `DM` from a given `DM`

.seealso: [](chapter_dmbase), `DM`, `DMDestroy()`, `DMCreate()`, `DMSetType()`, `DMSetLocalSection()`, `DMSetGlobalSection()`, `DMPLEX`, `DMSetType()`, `DMConvert()`
@*/
PetscErrorCode DMClone(DM dm, DM *newdm)
{
  PetscSF  sf;
  Vec      coords;
  void    *ctx;
  PetscInt dim, cdim, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(newdm, 2);
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), newdm));
  PetscCall(DMCopyLabels(dm, *newdm, PETSC_COPY_VALUES, PETSC_TRUE, DM_COPY_LABELS_FAIL));
  (*newdm)->leveldown     = dm->leveldown;
  (*newdm)->levelup       = dm->levelup;
  (*newdm)->prealloc_only = dm->prealloc_only;
  PetscCall(PetscFree((*newdm)->vectype));
  PetscCall(PetscStrallocpy(dm->vectype, (char **)&(*newdm)->vectype));
  PetscCall(PetscFree((*newdm)->mattype));
  PetscCall(PetscStrallocpy(dm->mattype, (char **)&(*newdm)->mattype));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(*newdm, dim));
  PetscTryTypeMethod(dm, clone, newdm);
  (*newdm)->setupcalled = dm->setupcalled;
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(DMSetPointSF(*newdm, sf));
  PetscCall(DMGetApplicationContext(dm, &ctx));
  PetscCall(DMSetApplicationContext(*newdm, ctx));
  for (i = 0; i < 2; ++i) {
    if (dm->coordinates[i].dm) {
      DM           ncdm;
      PetscSection cs;
      PetscInt     pEnd = -1, pEndMax = -1;

      PetscCall(DMGetLocalSection(dm->coordinates[i].dm, &cs));
      if (cs) PetscCall(PetscSectionGetChart(cs, NULL, &pEnd));
      PetscCall(MPIU_Allreduce(&pEnd, &pEndMax, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
      if (pEndMax >= 0) {
        PetscCall(DMClone(dm->coordinates[i].dm, &ncdm));
        PetscCall(DMCopyDisc(dm->coordinates[i].dm, ncdm));
        PetscCall(DMSetLocalSection(ncdm, cs));
        if (i) PetscCall(DMSetCellCoordinateDM(*newdm, ncdm));
        else PetscCall(DMSetCoordinateDM(*newdm, ncdm));
        PetscCall(DMDestroy(&ncdm));
      }
    }
  }
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMSetCoordinateDim(*newdm, cdim));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  if (coords) {
    PetscCall(DMSetCoordinatesLocal(*newdm, coords));
  } else {
    PetscCall(DMGetCoordinates(dm, &coords));
    if (coords) PetscCall(DMSetCoordinates(*newdm, coords));
  }
  PetscCall(DMGetCellCoordinatesLocal(dm, &coords));
  if (coords) {
    PetscCall(DMSetCellCoordinatesLocal(*newdm, coords));
  } else {
    PetscCall(DMGetCellCoordinates(dm, &coords));
    if (coords) PetscCall(DMSetCellCoordinates(*newdm, coords));
  }
  {
    const PetscReal *maxCell, *Lstart, *L;

    PetscCall(DMGetPeriodicity(dm, &maxCell, &Lstart, &L));
    PetscCall(DMSetPeriodicity(*newdm, maxCell, Lstart, L));
  }
  {
    PetscBool useCone, useClosure;

    PetscCall(DMGetAdjacency(dm, PETSC_DEFAULT, &useCone, &useClosure));
    PetscCall(DMSetAdjacency(*newdm, PETSC_DEFAULT, useCone, useClosure));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
       DMSetVecType - Sets the type of vector created with `DMCreateLocalVector()` and `DMCreateGlobalVector()`

   Logically Collective

   Input Parameters:
+  da - initial distributed array
-  ctype - the vector type, for example `VECSTANDARD`, `VECCUDA`, or `VECVIENNACL`

   Options Database Key:
.   -dm_vec_type ctype - the type of vector to create

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMCreate()`, `DMDestroy()`, `DM`, `DMDAInterpolationType`, `VecType`, `DMGetVecType()`, `DMSetMatType()`, `DMGetMatType()`,
          `VECSTANDARD`, `VECCUDA`, `VECVIENNACL`, `DMCreateLocalVector()`, `DMCreateGlobalVector()`
@*/
PetscErrorCode DMSetVecType(DM da, VecType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscCall(PetscFree(da->vectype));
  PetscCall(PetscStrallocpy(ctype, (char **)&da->vectype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
       DMGetVecType - Gets the type of vector created with `DMCreateLocalVector()` and `DMCreateGlobalVector()`

   Logically Collective

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  ctype - the vector type

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMCreate()`, `DMDestroy()`, `DM`, `DMDAInterpolationType`, `VecType`, `DMSetMatType()`, `DMGetMatType()`, `DMSetVecType()`
@*/
PetscErrorCode DMGetVecType(DM da, VecType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  *ctype = da->vectype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecGetDM - Gets the `DM` defining the data layout of the vector

  Not Collective

  Input Parameter:
. v - The `Vec`

  Output Parameter:
. dm - The `DM`

  Level: intermediate

  Note:
  A `Vec` may not have a `DM` associated with it.

.seealso: [](chapter_dmbase), `DM`, `VecSetDM()`, `DMGetLocalVector()`, `DMGetGlobalVector()`, `DMSetVecType()`
@*/
PetscErrorCode VecGetDM(Vec v, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidPointer(dm, 2);
  PetscCall(PetscObjectQuery((PetscObject)v, "__PETSc_dm", (PetscObject *)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecSetDM - Sets the `DM` defining the data layout of the vector.

  Not Collective

  Input Parameters:
+ v - The `Vec`
- dm - The `DM`

  Level: developer

  Note:
  This is rarely used, generally one uses `DMGetLocalVector()` or  `DMGetGlobalVector()` to create a vector associated with a given `DM`

  This is NOT the same as `DMCreateGlobalVector()` since it does not change the view methods or perform other customization, but merely sets the `DM` member.

.seealso: [](chapter_dmbase), `DM`, `VecGetDM()`, `DMGetLocalVector()`, `DMGetGlobalVector()`, `DMSetVecType()`
@*/
PetscErrorCode VecSetDM(Vec v, DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectCompose((PetscObject)v, "__PETSc_dm", (PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
       DMSetISColoringType - Sets the type of coloring, `IS_COLORING_GLOBAL` or `IS_COLORING_LOCAL` that is created by the `DM`

   Logically Collective

   Input Parameters:
+  dm - the `DM` context
-  ctype - the matrix type

   Options Database Key:
.   -dm_is_coloring_type - global or local

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMSetMatrixPreallocateOnly()`, `MatType`, `DMGetMatType()`,
          `DMGetISColoringType()`, `ISColoringType`, `IS_COLORING_GLOBAL`, `IS_COLORING_LOCAL`
@*/
PetscErrorCode DMSetISColoringType(DM dm, ISColoringType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->coloringtype = ctype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
       DMGetISColoringType - Gets the type of coloring, `IS_COLORING_GLOBAL` or `IS_COLORING_LOCAL` that is created by the `DM`

   Logically Collective

   Input Parameter:
.  dm - the `DM` context

   Output Parameter:
.  ctype - the matrix type

   Options Database Key:
.   -dm_is_coloring_type - global or local

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMSetMatrixPreallocateOnly()`, `MatType`, `DMGetMatType()`,
          `DMGetISColoringType()`, `ISColoringType`, `IS_COLORING_GLOBAL`, `IS_COLORING_LOCAL`
@*/
PetscErrorCode DMGetISColoringType(DM dm, ISColoringType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *ctype = dm->coloringtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
       DMSetMatType - Sets the type of matrix created with `DMCreateMatrix()`

   Logically Collective

   Input Parameters:
+  dm - the `DM` context
-  ctype - the matrix type, for example `MATMPIAIJ`

   Options Database Key:
.   -dm_mat_type ctype - the type of the matrix to create, for example mpiaij

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `MatType`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMSetMatrixPreallocateOnly()`, `MatType`, `DMGetMatType()`, `DMSetMatType()`, `DMGetMatType()`, `DMCreateGlobalVector()`, `DMCreateLocalVector()`
@*/
PetscErrorCode DMSetMatType(DM dm, MatType ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscFree(dm->mattype));
  PetscCall(PetscStrallocpy(ctype, (char **)&dm->mattype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
       DMGetMatType - Gets the type of matrix that would be created with `DMCreateMatrix()`

   Logically Collective

   Input Parameter:
.  dm - the `DM` context

   Output Parameter:
.  ctype - the matrix type

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMSetMatrixPreallocateOnly()`, `MatType`, `DMSetMatType()`, `DMSetMatType()`, `DMGetMatType()`
@*/
PetscErrorCode DMGetMatType(DM dm, MatType *ctype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *ctype = dm->mattype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatGetDM - Gets the `DM` defining the data layout of the matrix

  Not Collective

  Input Parameter:
. A - The `Mat`

  Output Parameter:
. dm - The `DM`

  Level: intermediate

  Note:
  A matrix may not have a `DM` associated with it

  Developer Note:
  Since the `Mat` class doesn't know about the `DM` class the `DM` object is associated with the `Mat` through a `PetscObjectCompose()` operation

.seealso: [](chapter_dmbase), `DM`, `MatSetDM()`, `DMCreateMatrix()`, `DMSetMatType()`
@*/
PetscErrorCode MatGetDM(Mat A, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(dm, 2);
  PetscCall(PetscObjectQuery((PetscObject)A, "__PETSc_dm", (PetscObject *)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSetDM - Sets the `DM` defining the data layout of the matrix

  Not Collective

  Input Parameters:
+ A - The `Mat`
- dm - The `DM`

  Level: developer

  Note:
  This is rarely used in practice, rather `DMCreateMatrix()` is used to create a matrix associated with a particular `DM`

  Developer Note:
  Since the `Mat` class doesn't know about the `DM` class the `DM` object is associated with
  the `Mat` through a `PetscObjectCompose()` operation

.seealso: [](chapter_dmbase), `DM`, `MatGetDM()`, `DMCreateMatrix()`, `DMSetMatType()`
@*/
PetscErrorCode MatSetDM(Mat A, DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectCompose((PetscObject)A, "__PETSc_dm", (PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMSetOptionsPrefix - Sets the prefix prepended to all option names when searching through the options database

   Logically Collective

   Input Parameters:
+  da - the `DM` context
-  prefix - the prefix to prepend

   Level: advanced

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](chapter_dmbase), `DM`, `PetscObjectSetOptionsPrefix()`, `DMSetFromOptions()`
@*/
PetscErrorCode DMSetOptionsPrefix(DM dm, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, prefix));
  if (dm->sf) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm->sf, prefix));
  if (dm->sectionSF) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm->sectionSF, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMAppendOptionsPrefix - Appends an additional string to an already existing prefix used for searching for
   `DM` options in the options database.

   Logically Collective

   Input Parameters:
+  dm - the `DM` context
-  prefix - the string to append to the current prefix

   Level: advanced

   Note:
   If the `DM` does not currently have an options prefix then this value is used alone as the prefix as if `DMSetOptionsPrefix()` had been called.
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](chapter_dmbase), `DM`, `DMSetOptionsPrefix()`, `DMGetOptionsPrefix()`, `PetscObjectAppendOptionsPrefix()`, `DMSetFromOptions()`
@*/
PetscErrorCode DMAppendOptionsPrefix(DM dm, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)dm, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMGetOptionsPrefix - Gets the prefix used for searching for all
   DM options in the options database.

   Not Collective

   Input Parameter:
.  dm - the `DM` context

   Output Parameter:
.  prefix - pointer to the prefix string used is returned

   Level: advanced

   Fortran Note:
   Pass in a string 'prefix' of
   sufficient length to hold the prefix.

.seealso: [](chapter_dmbase), `DM`, `DMSetOptionsPrefix()`, `DMAppendOptionsPrefix()`, `DMSetFromOptions()`
@*/
PetscErrorCode DMGetOptionsPrefix(DM dm, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCountNonCyclicReferences_Internal(DM dm, PetscBool recurseCoarse, PetscBool recurseFine, PetscInt *ncrefct)
{
  PetscInt refct = ((PetscObject)dm)->refct;

  PetscFunctionBegin;
  *ncrefct = 0;
  if (dm->coarseMesh && dm->coarseMesh->fineMesh == dm) {
    refct--;
    if (recurseCoarse) {
      PetscInt coarseCount;

      PetscCall(DMCountNonCyclicReferences_Internal(dm->coarseMesh, PETSC_TRUE, PETSC_FALSE, &coarseCount));
      refct += coarseCount;
    }
  }
  if (dm->fineMesh && dm->fineMesh->coarseMesh == dm) {
    refct--;
    if (recurseFine) {
      PetscInt fineCount;

      PetscCall(DMCountNonCyclicReferences_Internal(dm->fineMesh, PETSC_FALSE, PETSC_TRUE, &fineCount));
      refct += fineCount;
    }
  }
  *ncrefct = refct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Generic wrapper for DMCountNonCyclicReferences_Internal() */
PetscErrorCode DMCountNonCyclicReferences(PetscObject dm, PetscInt *ncrefct)
{
  PetscFunctionBegin;
  PetscCall(DMCountNonCyclicReferences_Internal((DM)dm, PETSC_TRUE, PETSC_TRUE, ncrefct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMDestroyLabelLinkList_Internal(DM dm)
{
  DMLabelLink next = dm->labels;

  PetscFunctionBegin;
  /* destroy the labels */
  while (next) {
    DMLabelLink tmp = next->next;

    if (next->label == dm->depthLabel) dm->depthLabel = NULL;
    if (next->label == dm->celltypeLabel) dm->celltypeLabel = NULL;
    PetscCall(DMLabelDestroy(&next->label));
    PetscCall(PetscFree(next));
    next = tmp;
  }
  dm->labels = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMDestroyCoordinates_Private(DMCoordinates *c)
{
  PetscFunctionBegin;
  c->dim = PETSC_DEFAULT;
  PetscCall(DMDestroy(&c->dm));
  PetscCall(VecDestroy(&c->x));
  PetscCall(VecDestroy(&c->xl));
  PetscCall(DMFieldDestroy(&c->field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMDestroy - Destroys a `DM`.

    Collective

    Input Parameter:
.   dm - the `DM` object to destroy

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreate()`, `DMType`, `DMSetType()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`
@*/
PetscErrorCode DMDestroy(DM *dm)
{
  PetscInt       cnt;
  DMNamedVecLink nlink, nnext;

  PetscFunctionBegin;
  if (!*dm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*dm), DM_CLASSID, 1);

  /* count all non-cyclic references in the doubly-linked list of coarse<->fine meshes */
  PetscCall(DMCountNonCyclicReferences_Internal(*dm, PETSC_TRUE, PETSC_TRUE, &cnt));
  --((PetscObject)(*dm))->refct;
  if (--cnt > 0) {
    *dm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (((PetscObject)(*dm))->refct < 0) PetscFunctionReturn(PETSC_SUCCESS);
  ((PetscObject)(*dm))->refct = 0;

  PetscCall(DMClearGlobalVectors(*dm));
  PetscCall(DMClearLocalVectors(*dm));

  nnext              = (*dm)->namedglobal;
  (*dm)->namedglobal = NULL;
  for (nlink = nnext; nlink; nlink = nnext) { /* Destroy the named vectors */
    nnext = nlink->next;
    PetscCheck(nlink->status == DMVEC_STATUS_IN, ((PetscObject)*dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "DM still has Vec named '%s' checked out", nlink->name);
    PetscCall(PetscFree(nlink->name));
    PetscCall(VecDestroy(&nlink->X));
    PetscCall(PetscFree(nlink));
  }
  nnext             = (*dm)->namedlocal;
  (*dm)->namedlocal = NULL;
  for (nlink = nnext; nlink; nlink = nnext) { /* Destroy the named local vectors */
    nnext = nlink->next;
    PetscCheck(nlink->status == DMVEC_STATUS_IN, ((PetscObject)*dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "DM still has Vec named '%s' checked out", nlink->name);
    PetscCall(PetscFree(nlink->name));
    PetscCall(VecDestroy(&nlink->X));
    PetscCall(PetscFree(nlink));
  }

  /* Destroy the list of hooks */
  {
    DMCoarsenHookLink link, next;
    for (link = (*dm)->coarsenhook; link; link = next) {
      next = link->next;
      PetscCall(PetscFree(link));
    }
    (*dm)->coarsenhook = NULL;
  }
  {
    DMRefineHookLink link, next;
    for (link = (*dm)->refinehook; link; link = next) {
      next = link->next;
      PetscCall(PetscFree(link));
    }
    (*dm)->refinehook = NULL;
  }
  {
    DMSubDomainHookLink link, next;
    for (link = (*dm)->subdomainhook; link; link = next) {
      next = link->next;
      PetscCall(PetscFree(link));
    }
    (*dm)->subdomainhook = NULL;
  }
  {
    DMGlobalToLocalHookLink link, next;
    for (link = (*dm)->gtolhook; link; link = next) {
      next = link->next;
      PetscCall(PetscFree(link));
    }
    (*dm)->gtolhook = NULL;
  }
  {
    DMLocalToGlobalHookLink link, next;
    for (link = (*dm)->ltoghook; link; link = next) {
      next = link->next;
      PetscCall(PetscFree(link));
    }
    (*dm)->ltoghook = NULL;
  }
  /* Destroy the work arrays */
  {
    DMWorkLink link, next;
    PetscCheck(!(*dm)->workout, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Work array still checked out");
    for (link = (*dm)->workin; link; link = next) {
      next = link->next;
      PetscCall(PetscFree(link->mem));
      PetscCall(PetscFree(link));
    }
    (*dm)->workin = NULL;
  }
  /* destroy the labels */
  PetscCall(DMDestroyLabelLinkList_Internal(*dm));
  /* destroy the fields */
  PetscCall(DMClearFields(*dm));
  /* destroy the boundaries */
  {
    DMBoundary next = (*dm)->boundary;
    while (next) {
      DMBoundary b = next;

      next = b->next;
      PetscCall(PetscFree(b));
    }
  }

  PetscCall(PetscObjectDestroy(&(*dm)->dmksp));
  PetscCall(PetscObjectDestroy(&(*dm)->dmsnes));
  PetscCall(PetscObjectDestroy(&(*dm)->dmts));

  if ((*dm)->ctx && (*dm)->ctxdestroy) PetscCall((*(*dm)->ctxdestroy)(&(*dm)->ctx));
  PetscCall(MatFDColoringDestroy(&(*dm)->fd));
  PetscCall(ISLocalToGlobalMappingDestroy(&(*dm)->ltogmap));
  PetscCall(PetscFree((*dm)->vectype));
  PetscCall(PetscFree((*dm)->mattype));

  PetscCall(PetscSectionDestroy(&(*dm)->localSection));
  PetscCall(PetscSectionDestroy(&(*dm)->globalSection));
  PetscCall(PetscLayoutDestroy(&(*dm)->map));
  PetscCall(PetscSectionDestroy(&(*dm)->defaultConstraint.section));
  PetscCall(MatDestroy(&(*dm)->defaultConstraint.mat));
  PetscCall(PetscSFDestroy(&(*dm)->sf));
  PetscCall(PetscSFDestroy(&(*dm)->sectionSF));
  if ((*dm)->useNatural) {
    if ((*dm)->sfNatural) PetscCall(PetscSFDestroy(&(*dm)->sfNatural));
    PetscCall(PetscObjectDereference((PetscObject)(*dm)->sfMigration));
  }
  {
    Vec     *auxData;
    PetscInt n, i, off = 0;

    PetscCall(PetscHMapAuxGetSize((*dm)->auxData, &n));
    PetscCall(PetscMalloc1(n, &auxData));
    PetscCall(PetscHMapAuxGetVals((*dm)->auxData, &off, auxData));
    for (i = 0; i < n; ++i) PetscCall(VecDestroy(&auxData[i]));
    PetscCall(PetscFree(auxData));
    PetscCall(PetscHMapAuxDestroy(&(*dm)->auxData));
  }
  if ((*dm)->coarseMesh && (*dm)->coarseMesh->fineMesh == *dm) PetscCall(DMSetFineDM((*dm)->coarseMesh, NULL));

  PetscCall(DMDestroy(&(*dm)->coarseMesh));
  if ((*dm)->fineMesh && (*dm)->fineMesh->coarseMesh == *dm) PetscCall(DMSetCoarseDM((*dm)->fineMesh, NULL));
  PetscCall(DMDestroy(&(*dm)->fineMesh));
  PetscCall(PetscFree((*dm)->Lstart));
  PetscCall(PetscFree((*dm)->L));
  PetscCall(PetscFree((*dm)->maxCell));
  PetscCall(DMDestroyCoordinates_Private(&(*dm)->coordinates[0]));
  PetscCall(DMDestroyCoordinates_Private(&(*dm)->coordinates[1]));
  if ((*dm)->transformDestroy) PetscCall((*(*dm)->transformDestroy)(*dm, (*dm)->transformCtx));
  PetscCall(DMDestroy(&(*dm)->transformDM));
  PetscCall(VecDestroy(&(*dm)->transform));
  PetscCall(VecScatterDestroy(&(*dm)->periodic.affine_to_local));
  PetscCall(VecDestroy(&(*dm)->periodic.affine));

  PetscCall(DMClearDS(*dm));
  PetscCall(DMDestroy(&(*dm)->dmBC));
  /* if memory was published with SAWs then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*dm));

  if ((*dm)->ops->destroy) PetscCall((*(*dm)->ops->destroy)(*dm));
  PetscCall(DMMonitorCancel(*dm));
#ifdef PETSC_HAVE_LIBCEED
  PetscCallCEED(CeedElemRestrictionDestroy(&(*dm)->ceedERestrict));
  PetscCallCEED(CeedDestroy(&(*dm)->ceed));
#endif
  /* We do not destroy (*dm)->data here so that we can reference count backend objects */
  PetscCall(PetscHeaderDestroy(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMSetUp - sets up the data structures inside a `DM` object

    Collective

    Input Parameter:
.   dm - the `DM` object to setup

    Level: intermediate

    Note:
    This is usually called after various parameter setting operations and `DMSetFromOptions()` are called on the `DM`

.seealso: [](chapter_dmbase), `DM`, `DMCreate()`, `DMSetType()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`
@*/
PetscErrorCode DMSetUp(DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscTryTypeMethod(dm, setup);
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMSetFromOptions - sets parameters in a `DM` from the options database

    Collective

    Input Parameter:
.   dm - the `DM` object to set options for

    Options Database Keys:
+   -dm_preallocate_only - Only preallocate the matrix for `DMCreateMatrix()` and `DMCreateMassMatrix()`, but do not fill it with zeros
.   -dm_vec_type <type>  - type of vector to create inside `DM`
.   -dm_mat_type <type>  - type of matrix to create inside `DM`
.   -dm_is_coloring_type - <global or local>
.   -dm_bind_below <n>   - bind (force execution on CPU) for `Vec` and `Mat` objects with local size (number of vector entries or matrix rows) below n; currently only supported for `DMDA`
. -dm_plex_filename <str>           - File containing a mesh
. -dm_plex_boundary_filename <str>  - File containing a mesh boundary
. -dm_plex_name <str>               - Name of the mesh in the file
. -dm_plex_shape <shape>            - The domain shape, such as `BOX`, `SPHERE`, etc.
. -dm_plex_cell <ct>                - Cell shape
. -dm_plex_reference_cell_domain <bool> - Use a reference cell domain
. -dm_plex_dim <dim>                - Set the topological dimension
. -dm_plex_simplex <bool>           - `PETSC_TRUE` for simplex elements, `PETSC_FALSE` for tensor elements
. -dm_plex_interpolate <bool>       - `PETSC_TRUE` turns on topological interpolation (creating edges and faces)
. -dm_plex_scale <sc>               - Scale factor for mesh coordinates
. -dm_plex_box_faces <m,n,p>        - Number of faces along each dimension
. -dm_plex_box_lower <x,y,z>        - Specify lower-left-bottom coordinates for the box
. -dm_plex_box_upper <x,y,z>        - Specify upper-right-top coordinates for the box
. -dm_plex_box_bd <bx,by,bz>        - Specify the `DMBoundaryType` for each direction
. -dm_plex_sphere_radius <r>        - The sphere radius
. -dm_plex_ball_radius <r>          - Radius of the ball
. -dm_plex_cylinder_bd <bz>         - Boundary type in the z direction
. -dm_plex_cylinder_num_wedges <n>  - Number of wedges around the cylinder
. -dm_plex_reorder <order>          - Reorder the mesh using the specified algorithm
. -dm_refine_pre <n>                - The number of refinements before distribution
. -dm_refine_uniform_pre <bool>     - Flag for uniform refinement before distribution
. -dm_refine_volume_limit_pre <v>   - The maximum cell volume after refinement before distribution
. -dm_refine <n>                    - The number of refinements after distribution
. -dm_extrude <l>                   - Activate extrusion and specify the number of layers to extrude
. -dm_plex_transform_extrude_thickness <t>           - The total thickness of extruded layers
. -dm_plex_transform_extrude_use_tensor <bool>       - Use tensor cells when extruding
. -dm_plex_transform_extrude_symmetric <bool>        - Extrude layers symmetrically about the surface
. -dm_plex_transform_extrude_normal <n0,...,nd>      - Specify the extrusion direction
. -dm_plex_transform_extrude_thicknesses <t0,...,tl> - Specify thickness of each layer
. -dm_plex_create_fv_ghost_cells    - Flag to create finite volume ghost cells on the boundary
. -dm_plex_fv_ghost_cells_label <name> - Label name for ghost cells boundary
. -dm_distribute <bool>             - Flag to redistribute a mesh among processes
. -dm_distribute_overlap <n>        - The size of the overlap halo
. -dm_plex_adj_cone <bool>          - Set adjacency direction
.  -dm_plex_adj_closure <bool>       - Set adjacency size
.   -dm_plex_check_symmetry        - Check that the adjacency information in the mesh is symmetric - `DMPlexCheckSymmetry()`
.   -dm_plex_check_skeleton        - Check that each cell has the correct number of vertices (only for homogeneous simplex or tensor meshes) - `DMPlexCheckSkeleton()`
.   -dm_plex_check_faces           - Check that the faces of each cell give a vertex order this is consistent with what we expect from the cell type - `DMPlexCheckFaces()`
.   -dm_plex_check_geometry        - Check that cells have positive volume - `DMPlexCheckGeometry()`
.   -dm_plex_check_pointsf         - Check some necessary conditions for `PointSF` - `DMPlexCheckPointSF()`
.   -dm_plex_check_interface_cones - Check points on inter-partition interfaces have conforming order of cone points - `DMPlexCheckInterfaceCones()`
-   -dm_plex_check_all             - Perform all the checks above

    Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`,
         `DMPlexCheckSymmetry()`, `DMPlexCheckSkeleton()`, `DMPlexCheckFaces()`, `DMPlexCheckGeometry()`, `DMPlexCheckPointSF()`, `DMPlexCheckInterfaceCones()`,
         `DMSetOptionsPrefix()`, `DM`, `DMType`, `DMPLEX`, `DMDA`
@*/
PetscErrorCode DMSetFromOptions(DM dm)
{
  char      typeName[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->setfromoptionscalled = PETSC_TRUE;
  if (dm->sf) PetscCall(PetscSFSetFromOptions(dm->sf));
  if (dm->sectionSF) PetscCall(PetscSFSetFromOptions(dm->sectionSF));
  if (dm->coordinates[0].dm) PetscCall(DMSetFromOptions(dm->coordinates[0].dm));
  PetscObjectOptionsBegin((PetscObject)dm);
  PetscCall(PetscOptionsBool("-dm_preallocate_only", "only preallocate matrix, but do not set column indices", "DMSetMatrixPreallocateOnly", dm->prealloc_only, &dm->prealloc_only, NULL));
  PetscCall(PetscOptionsFList("-dm_vec_type", "Vector type used for created vectors", "DMSetVecType", VecList, dm->vectype, typeName, 256, &flg));
  if (flg) PetscCall(DMSetVecType(dm, typeName));
  PetscCall(PetscOptionsFList("-dm_mat_type", "Matrix type used for created matrices", "DMSetMatType", MatList, dm->mattype ? dm->mattype : typeName, typeName, sizeof(typeName), &flg));
  if (flg) PetscCall(DMSetMatType(dm, typeName));
  PetscCall(PetscOptionsEnum("-dm_blocking_type", "Topological point or field node blocking", "DMSetBlockingType", DMBlockingTypes, (PetscEnum)dm->blocking_type, (PetscEnum *)&dm->blocking_type, NULL));
  PetscCall(PetscOptionsEnum("-dm_is_coloring_type", "Global or local coloring of Jacobian", "DMSetISColoringType", ISColoringTypes, (PetscEnum)dm->coloringtype, (PetscEnum *)&dm->coloringtype, NULL));
  PetscCall(PetscOptionsInt("-dm_bind_below", "Set the size threshold (in entries) below which the Vec is bound to the CPU", "VecBindToCPU", dm->bind_below, &dm->bind_below, &flg));
  PetscTryTypeMethod(dm, setfromoptions, PetscOptionsObject);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)dm, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMViewFromOptions - View a `DM` in a particular way based on a request in the options database

   Collective

   Input Parameters:
+  dm - the `DM` object
.  obj - optional object that provides the prefix for the options database (if `NULL` then the prefix in obj is used)
-  optionname - option string that is used to activate viewing

   Level: intermediate

   Note:
   See `PetscObjectViewFromOptions()` for a list of values that can be provided in the options database to determine how the `DM` is viewed

.seealso: [](chapter_dmbase), `DM`, `DMView()`, `PetscObjectViewFromOptions()`, `DMCreate()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode DMViewFromOptions(DM dm, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)dm, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMView - Views a `DM`. Depending on the `PetscViewer` and its `PetscViewerFormat` it may print some ASCII information about the `DM` to the screen or a file or
    save the `DM` in a binary file to be loaded later or create a visualization of the `DM`

    Collective

    Input Parameters:
+   dm - the `DM` object to view
-   v - the viewer

    Level: beginner

    Notes:
    Using `PETSCVIEWERHDF5` type with `PETSC_VIEWER_HDF5_PETSC` as the `PetscViewerFormat` one can save multiple `DMPLEX`
    meshes in a single HDF5 file. This in turn requires one to name the `DMPLEX` object with `PetscObjectSetName()`
    before saving it with `DMView()` and before loading it with `DMLoad()` for identification of the mesh object.

.seealso: [](chapter_dmbase), `DM`, `PetscViewer`, `PetscViewerFormat`, `PetscViewerSetFormat()`, `DMDestroy()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMLoad()`, `PetscObjectSetName()`
@*/
PetscErrorCode DMView(DM dm, PetscViewer v)
{
  PetscBool         isbinary;
  PetscMPIInt       size;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!v) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)dm), &v));
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  /* Ideally, we would like to have this test on.
     However, it currently breaks socket viz via GLVis.
     During DMView(parallel_mesh,glvis_viewer), each
     process opens a sequential ASCII socket to visualize
     the local mesh, and PetscObjectView(dm,local_socket)
     is internally called inside VecView_GLVis, incurring
     in an error here */
  /* PetscCheckSameComm(dm,1,v,2); */
  PetscCall(PetscViewerCheckWritable(v));

  PetscCall(PetscViewerGetFormat(v, &format));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)dm, v));
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERBINARY, &isbinary));
  if (isbinary) {
    PetscInt classid = DM_FILE_CLASSID;
    char     type[256];

    PetscCall(PetscViewerBinaryWrite(v, &classid, 1, PETSC_INT));
    PetscCall(PetscStrncpy(type, ((PetscObject)dm)->type_name, sizeof(type)));
    PetscCall(PetscViewerBinaryWrite(v, type, 256, PETSC_CHAR));
  }
  PetscTryTypeMethod(dm, view, v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateGlobalVector - Creates a global vector from a `DM` object. A global vector is a parallel vector that has no duplicate values shared between MPI ranks,
    that is it has no ghost locations.

    Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   vec - the global vector

    Level: beginner

.seealso: [](chapter_dmbase), `DM`, `Vec`, `DMCreateLocalVector()`, `DMGetGlobalVector()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`,
         `DMGlobalToLocalBegin()`, `DMGlobalToLocalEnd()`
@*/
PetscErrorCode DMCreateGlobalVector(DM dm, Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(vec, 2);
  PetscUseTypeMethod(dm, createglobalvector, vec);
  if (PetscDefined(USE_DEBUG)) {
    DM vdm;

    PetscCall(VecGetDM(*vec, &vdm));
    PetscCheck(vdm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "DM type '%s' did not attach the DM to the vector", ((PetscObject)dm)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateLocalVector - Creates a local vector from a `DM` object.

    Not Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   vec - the local vector

    Level: beginner

    Note:
    A local vector usually has ghost locations that contain values that are owned by different MPI ranks. A global vector has no ghost locations.

.seealso: [](chapter_dmbase), `DM`, `Vec`, `DMCreateGlobalVector()`, `DMGetLocalVector()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`
         `DMGlobalToLocalBegin()`, `DMGlobalToLocalEnd()`
@*/
PetscErrorCode DMCreateLocalVector(DM dm, Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(vec, 2);
  PetscUseTypeMethod(dm, createlocalvector, vec);
  if (PetscDefined(USE_DEBUG)) {
    DM vdm;

    PetscCall(VecGetDM(*vec, &vdm));
    PetscCheck(vdm, PETSC_COMM_SELF, PETSC_ERR_LIB, "DM type '%s' did not attach the DM to the vector", ((PetscObject)dm)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMGetLocalToGlobalMapping - Accesses the local-to-global mapping in a `DM`.

   Collective

   Input Parameter:
.  dm - the `DM` that provides the mapping

   Output Parameter:
.  ltog - the mapping

   Level: advanced

   Notes:
   The global to local mapping allows one to set values into the global vector or matrix using `VecSetValuesLocal()` and `MatSetValuesLocal()`

   Vectors obtained with  `DMCreateGlobalVector()` and matrices obtained with `DMCreateMatrix()` already contain the global mapping so you do
   need to use this function with those objects.

   This mapping can then be used by `VecSetLocalToGlobalMapping()` or `MatSetLocalToGlobalMapping()`.

.seealso: [](chapter_dmbase), `DM`, `DMCreateLocalVector()`, `DMCreateLocalVector()`, `DMCreateGlobalVector()`, `VecSetLocalToGlobalMapping()`, `MatSetLocalToGlobalMapping()`,
          `DMCreateMatrix()`
@*/
PetscErrorCode DMGetLocalToGlobalMapping(DM dm, ISLocalToGlobalMapping *ltog)
{
  PetscInt bs = -1, bsLocal[2], bsMinMax[2];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ltog, 2);
  if (!dm->ltogmap) {
    PetscSection section, sectionGlobal;

    PetscCall(DMGetLocalSection(dm, &section));
    if (section) {
      const PetscInt *cdofs;
      PetscInt       *ltog;
      PetscInt        pStart, pEnd, n, p, k, l;

      PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
      PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
      PetscCall(PetscSectionGetStorageSize(section, &n));
      PetscCall(PetscMalloc1(n, &ltog)); /* We want the local+overlap size */
      for (p = pStart, l = 0; p < pEnd; ++p) {
        PetscInt bdof, cdof, dof, off, c, cind;

        /* Should probably use constrained dofs */
        PetscCall(PetscSectionGetDof(section, p, &dof));
        PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
        PetscCall(PetscSectionGetConstraintIndices(section, p, &cdofs));
        PetscCall(PetscSectionGetOffset(sectionGlobal, p, &off));
        /* If you have dofs, and constraints, and they are unequal, we set the blocksize to 1 */
        bdof = cdof && (dof - cdof) ? 1 : dof;
        if (dof) bs = bs < 0 ? bdof : PetscGCD(bs, bdof);

        for (c = 0, cind = 0; c < dof; ++c, ++l) {
          if (cind < cdof && c == cdofs[cind]) {
            ltog[l] = off < 0 ? off - c : -(off + c + 1);
            cind++;
          } else {
            ltog[l] = (off < 0 ? -(off + 1) : off) + c - cind;
          }
        }
      }
      /* Must have same blocksize on all procs (some might have no points) */
      bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs;
      bsLocal[1] = bs;
      PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)dm), bsLocal, bsMinMax));
      if (bsMinMax[0] != bsMinMax[1]) {
        bs = 1;
      } else {
        bs = bsMinMax[0];
      }
      bs = bs < 0 ? 1 : bs;
      /* Must reduce indices by blocksize */
      if (bs > 1) {
        for (l = 0, k = 0; l < n; l += bs, ++k) {
          // Integer division of negative values truncates toward zero(!), not toward negative infinity
          ltog[k] = ltog[l] >= 0 ? ltog[l] / bs : -(-(ltog[l] + 1) / bs + 1);
        }
        n /= bs;
      }
      PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm), bs, n, ltog, PETSC_OWN_POINTER, &dm->ltogmap));
    } else PetscUseTypeMethod(dm, getlocaltoglobalmapping);
  }
  *ltog = dm->ltogmap;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMGetBlockSize - Gets the inherent block size associated with a `DM`

   Not Collective

   Input Parameter:
.  dm - the `DM` with block structure

   Output Parameter:
.  bs - the block size, 1 implies no exploitable block structure

   Level: intermediate

   Note:
   This might be the number of degrees of freedom at each grid point for a structured grid.

   Complex `DM` that represent multiphysics or staggered grids or mixed-methods do not generally have a single inherent block size, but
   rather different locations in the vectors may have a different block size.

.seealso: [](chapter_dmbase), `DM`, `ISCreateBlock()`, `VecSetBlockSize()`, `MatSetBlockSize()`, `DMGetLocalToGlobalMapping()`
@*/
PetscErrorCode DMGetBlockSize(DM dm, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(bs, 2);
  PetscCheck(dm->bs >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DM does not have enough information to provide a block size yet");
  *bs = dm->bs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMCreateInterpolation - Gets the interpolation matrix between two `DM` objects. The resulting matrix map degrees of freedom in the vector obtained by
    `DMCreateGlobalVector()` on the coarse `DM` to similar vectors on the fine grid `DM`.

    Collective

    Input Parameters:
+   dmc - the `DM` object
-   dmf - the second, finer `DM` object

    Output Parameters:
+  mat - the interpolation
-  vec - the scaling (optional), see `DMCreateInterpolationScale()`

    Level: developer

    Notes:
    For `DMDA` objects this only works for "uniform refinement", that is the refined mesh was obtained `DMRefine()` or the coarse mesh was obtained by
    DMCoarsen(). The coordinates set into the `DMDA` are completely ignored in computing the interpolation.

    For `DMDA` objects you can use this interpolation (more precisely the interpolation from the `DMGetCoordinateDM()`) to interpolate the mesh coordinate
    vectors EXCEPT in the periodic case where it does not make sense since the coordinate vectors are not periodic.

.seealso: [](chapter_dmbase), `DM`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMRefine()`, `DMCoarsen()`, `DMCreateRestriction()`, `DMCreateInterpolationScale()`
@*/
PetscErrorCode DMCreateInterpolation(DM dmc, DM dmf, Mat *mat, Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 2);
  PetscValidPointer(mat, 3);
  PetscCall(PetscLogEventBegin(DM_CreateInterpolation, dmc, dmf, 0, 0));
  PetscUseTypeMethod(dmc, createinterpolation, dmf, mat, vec);
  PetscCall(PetscLogEventEnd(DM_CreateInterpolation, dmc, dmf, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateInterpolationScale - Forms L = 1/(R*1) where 1 is the vector of all ones, and R is the transpose of the interpolation between the `DM`.
    xcoarse = diag(L)*R*xfine preserves scale and is thus suitable for state (versus residual) restriction. In other words xcoarse is the coarse
    representation of xfine.

  Input Parameters:
+      dac - `DM` that defines a coarse mesh
.      daf - `DM` that defines a fine mesh
-      mat - the restriction (or interpolation operator) from fine to coarse

  Output Parameter:
.    scale - the scaled vector

  Level: advanced

  Developer Note:
  If the fine-scale `DMDA` has the -dm_bind_below option set to true, then `DMCreateInterpolationScale()` calls `MatSetBindingPropagates()`
  on the restriction/interpolation operator to set the bindingpropagates flag to true.

.seealso: [](chapter_dmbase), `DM`, `MatRestrict()`, `MatInterpolate()`, `DMCreateInterpolation()`, DMCreateRestriction()`, `DMCreateGlobalVector()`
@*/
PetscErrorCode DMCreateInterpolationScale(DM dac, DM daf, Mat mat, Vec *scale)
{
  Vec         fine;
  PetscScalar one = 1.0;
#if defined(PETSC_HAVE_CUDA)
  PetscBool bindingpropagates, isbound;
#endif

  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(daf, &fine));
  PetscCall(DMCreateGlobalVector(dac, scale));
  PetscCall(VecSet(fine, one));
#if defined(PETSC_HAVE_CUDA)
  /* If the 'fine' Vec is bound to the CPU, it makes sense to bind 'mat' as well.
   * Note that we only do this for the CUDA case, right now, but if we add support for MatMultTranspose() via ViennaCL,
   * we'll need to do it for that case, too.*/
  PetscCall(VecGetBindingPropagates(fine, &bindingpropagates));
  if (bindingpropagates) {
    PetscCall(MatSetBindingPropagates(mat, PETSC_TRUE));
    PetscCall(VecBoundToCPU(fine, &isbound));
    PetscCall(MatBindToCPU(mat, isbound));
  }
#endif
  PetscCall(MatRestrict(mat, fine, *scale));
  PetscCall(VecDestroy(&fine));
  PetscCall(VecReciprocal(*scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateRestriction - Gets restriction matrix between two `DM` objects. The resulting matrix map degrees of freedom in the vector obtained by
    `DMCreateGlobalVector()` on the fine `DM` to similar vectors on the coarse grid `DM`.

    Collective

    Input Parameters:
+   dmc - the `DM` object
-   dmf - the second, finer `DM` object

    Output Parameter:
.  mat - the restriction

    Level: developer

    Note:
    This only works for `DMSTAG`. For many situations either the transpose of the operator obtained with `DMCreateInterpolation()` or that
    matrix multiplied by the vector obtained with `DMCreateInterpolationScale()` provides the desired object.

.seealso: [](chapter_dmbase), `DM`, `DMRestrict()`, `DMInterpolate()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMRefine()`, `DMCoarsen()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMCreateRestriction(DM dmc, DM dmf, Mat *mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 2);
  PetscValidPointer(mat, 3);
  PetscCall(PetscLogEventBegin(DM_CreateRestriction, dmc, dmf, 0, 0));
  PetscUseTypeMethod(dmc, createrestriction, dmf, mat);
  PetscCall(PetscLogEventEnd(DM_CreateRestriction, dmc, dmf, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateInjection - Gets injection matrix between two `DM` objects. This is an operator that applied to a vector obtained with
    `DMCreateGlobalVector()` on the fine grid maps the values to a vector on the vector on the coarse `DM` by simply selecting the values
    on the coarse grid points. This compares to the operator obtained by `DMCreateRestriction()` or the transpose of the operator obtained
    by `DMCreateInterpolation()` that uses a "local weighted average" of the values around the coarse grid point as the coarse grid value.

    Collective

    Input Parameters:
+   dac - the `DM` object
-   daf - the second, finer `DM` object

    Output Parameter:
.   mat - the injection

    Level: developer

   Note:
    For `DMDA` objects this only works for "uniform refinement", that is the refined mesh was obtained `DMRefine()` or the coarse mesh was obtained by
        `DMCoarsen()`. The coordinates set into the `DMDA` are completely ignored in computing the injection.

.seealso: [](chapter_dmbase), `DM`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMCreateInterpolation()`,
          `DMCreateRestriction()`, `MatRestrict()`, `MatInterpolate()`
@*/
PetscErrorCode DMCreateInjection(DM dac, DM daf, Mat *mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac, DM_CLASSID, 1);
  PetscValidHeaderSpecific(daf, DM_CLASSID, 2);
  PetscValidPointer(mat, 3);
  PetscCall(PetscLogEventBegin(DM_CreateInjection, dac, daf, 0, 0));
  PetscUseTypeMethod(dac, createinjection, daf, mat);
  PetscCall(PetscLogEventEnd(DM_CreateInjection, dac, daf, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCreateMassMatrix - Gets the mass matrix between two `DM` objects, M_ij = \int \phi_i \psi_j where the \phi are Galerkin basis functions for a
  a Galerkin finite element model on the `DM`

  Collective

  Input Parameters:
+ dmc - the target `DM` object
- dmf - the source `DM` object

  Output Parameter:
. mat - the mass matrix

  Level: developer

  Notes:
  For `DMPLEX` the finite element model for the `DM` must have been already provided.

  if `dmc` is `dmf` then x^t M x is an approximation to the L2 norm of the vector x which is obtained by `DMCreateGlobalVector()`

.seealso: [](chapter_dmbase), `DM`, `DMCreateMassMatrixLumped()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMRefine()`, `DMCoarsen()`, `DMCreateRestriction()`, `DMCreateInterpolation()`, `DMCreateInjection()`
@*/
PetscErrorCode DMCreateMassMatrix(DM dmc, DM dmf, Mat *mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 2);
  PetscValidPointer(mat, 3);
  PetscCall(PetscLogEventBegin(DM_CreateMassMatrix, 0, 0, 0, 0));
  PetscUseTypeMethod(dmc, createmassmatrix, dmf, mat);
  PetscCall(PetscLogEventEnd(DM_CreateMassMatrix, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCreateMassMatrixLumped - Gets the lumped mass matrix for a given `DM`

  Collective

  Input Parameter:
. dm - the `DM` object

  Output Parameter:
. lm - the lumped mass matrix, which is a diagonal matrix, represented as a vector

  Level: developer

  Note:
  See `DMCreateMassMatrix()` for how to create the non-lumped version of the mass matrix.

.seealso: [](chapter_dmbase), `DM`, `DMCreateMassMatrix()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMRefine()`, `DMCoarsen()`, `DMCreateRestriction()`, `DMCreateInterpolation()`, `DMCreateInjection()`
@*/
PetscErrorCode DMCreateMassMatrixLumped(DM dm, Vec *lm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(lm, 2);
  PetscUseTypeMethod(dm, createmassmatrixlumped, lm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateColoring - Gets coloring of a graph associated with the `DM`. Often the graph represents the operator matrix associated with the discretization
    of a PDE on the `DM`.

    Collective

    Input Parameters:
+   dm - the `DM` object
-   ctype - `IS_COLORING_LOCAL` or `IS_COLORING_GLOBAL`

    Output Parameter:
.   coloring - the coloring

    Level: developer

    Notes:
    Coloring of matrices can also be computed directly from the sparse matrix nonzero structure via the `MatColoring` object or from the mesh from which the
    matrix comes from (what this function provides). In general using the mesh produces a more optimal coloring (fewer colors).

    This produces a coloring with the distance of 2, see `MatSetColoringDistance()` which can be used for efficiently computing Jacobians with `MatFDColoringCreate()`
    For `DMDA` in three dimensions with periodic boundary conditions the number of grid points in each dimension must be divisible by 2*stencil_width + 1,
    otherwise an error will be generated.

.seealso: [](chapter_dmbase), `DM`, `ISColoring`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMSetMatType()`, `MatColoring`, `MatFDColoringCreate()`
@*/
PetscErrorCode DMCreateColoring(DM dm, ISColoringType ctype, ISColoring *coloring)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coloring, 3);
  PetscUseTypeMethod(dm, getcoloring, ctype, coloring);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCreateMatrix - Gets an empty matrix for a `DM` that is most commonly used to store the Jacobian of a discrete PDE operator.

    Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   mat - the empty Jacobian

    Options Database Key:
. -dm_preallocate_only - Only preallocate the matrix for `DMCreateMatrix()` and `DMCreateMassMatrix()`, but do not fill it with zeros

    Level: beginner

    Notes:
    This properly preallocates the number of nonzeros in the sparse matrix so you
    do not need to do it yourself.

    By default it also sets the nonzero structure and puts in the zero entries. To prevent setting
    the nonzero pattern call `DMSetMatrixPreallocateOnly()`

    For `DMDA`, when you call `MatView()` on this matrix it is displayed using the global natural ordering, NOT in the ordering used
    internally by PETSc.

    For `DMDA`, in general it is easiest to use `MatSetValuesStencil()` or `MatSetValuesLocal()` to put values into the matrix because
    `MatSetValues()` requires the indices for the global numbering for the `DMDA` which is complic`ated to compute

.seealso: [](chapter_dmbase), `DM`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMSetMatType()`, `DMCreateMassMatrix()`
@*/
PetscErrorCode DMCreateMatrix(DM dm, Mat *mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(mat, 2);
  PetscCall(MatInitializePackage());
  PetscCall(PetscLogEventBegin(DM_CreateMatrix, 0, 0, 0, 0));
  PetscUseTypeMethod(dm, creatematrix, mat);
  if (PetscDefined(USE_DEBUG)) {
    DM mdm;

    PetscCall(MatGetDM(*mat, &mdm));
    PetscCheck(mdm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "DM type '%s' did not attach the DM to the matrix", ((PetscObject)dm)->type_name);
  }
  /* Handle nullspace and near nullspace */
  if (dm->Nf) {
    MatNullSpace nullSpace;
    PetscInt     Nf, f;

    PetscCall(DMGetNumFields(dm, &Nf));
    for (f = 0; f < Nf; ++f) {
      if (dm->nullspaceConstructors[f]) {
        PetscCall((*dm->nullspaceConstructors[f])(dm, f, f, &nullSpace));
        PetscCall(MatSetNullSpace(*mat, nullSpace));
        PetscCall(MatNullSpaceDestroy(&nullSpace));
        break;
      }
    }
    for (f = 0; f < Nf; ++f) {
      if (dm->nearnullspaceConstructors[f]) {
        PetscCall((*dm->nearnullspaceConstructors[f])(dm, f, f, &nullSpace));
        PetscCall(MatSetNearNullSpace(*mat, nullSpace));
        PetscCall(MatNullSpaceDestroy(&nullSpace));
      }
    }
  }
  PetscCall(PetscLogEventEnd(DM_CreateMatrix, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetMatrixPreallocateSkip - When `DMCreateMatrix()` is called the matrix sizes and `ISLocalToGlobalMapping` will be
  properly set, but the data structures to store values in the matrices will not be preallocated. This is most useful to reduce initialization costs when
  `MatSetPreallocationCOO()` and `MatSetValuesCOO()` will be used.

  Logically Collective

  Input Parameters:
+ dm - the `DM`
- skip - `PETSC_TRUE` to skip preallocation

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreateMatrix()`, `DMSetMatrixStructureOnly()`, `DMSetMatrixPreallocateOnly()`
@*/
PetscErrorCode DMSetMatrixPreallocateSkip(DM dm, PetscBool skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->prealloc_skip = skip;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetMatrixPreallocateOnly - When `DMCreateMatrix()` is called the matrix will be properly
    preallocated but the nonzero structure and zero values will not be set.

  Logically Collective

  Input Parameters:
+ dm - the `DM`
- only - `PETSC_TRUE` if only want preallocation

  Options Database Key:
. -dm_preallocate_only - Only preallocate the matrix for `DMCreateMatrix()`, `DMCreateMassMatrix()`, but do not fill it with zeros

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMSetMatrixStructureOnly()`, `DMSetMatrixPreallocateSkip()`
@*/
PetscErrorCode DMSetMatrixPreallocateOnly(DM dm, PetscBool only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->prealloc_only = only;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetMatrixStructureOnly - When `DMCreateMatrix()` is called, the matrix structure will be created
    but the array for numerical values will not be allocated.

  Logically Collective

  Input Parameters:
+ dm - the `DM`
- only - `PETSC_TRUE` if you only want matrix structure

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreateMatrix()`, `DMSetMatrixPreallocateOnly()`, `DMSetMatrixPreallocateSkip()`
@*/
PetscErrorCode DMSetMatrixStructureOnly(DM dm, PetscBool only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->structure_only = only;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetBlockingType - set the blocking granularity to be used for variable block size `DMCreateMatrix()` is called

  Logically Collective

  Input Parameters:
+ dm - the `DM`
- btype - block by topological point or field node

  Options Database Key:
. -dm_blocking_type [topological_point, field_node] - use topological point blocking or field node blocking

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMCreateMatrix()`, `MatSetVariableBlockSizes()`
@*/
PetscErrorCode DMSetBlockingType(DM dm, DMBlockingType btype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->blocking_type = btype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetBlockingType - get the blocking granularity to be used for variable block size `DMCreateMatrix()` is called

  Not Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. btype - block by topological point or field node

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMCreateMatrix()`, `MatSetVariableBlockSizes()`
@*/
PetscErrorCode DMGetBlockingType(DM dm, DMBlockingType *btype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(btype, 2);
  *btype = dm->blocking_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetWorkArray - Gets a work array guaranteed to be at least the input size, restore with `DMRestoreWorkArray()`

  Not Collective

  Input Parameters:
+ dm - the `DM` object
. count - The minimum size
- dtype - MPI data type, often `MPIU_REAL`, `MPIU_SCALAR`, or `MPIU_INT`)

  Output Parameter:
. array - the work array

  Level: developer

  Note:
  A `DM` may stash the array between instantiations so using this routine may be more efficient than calling `PetscMalloc()`

  The array may contain nonzero values

.seealso: [](chapter_dmbase), `DM`, `DMDestroy()`, `DMCreate()`, `DMRestoreWorkArray()`, `PetscMalloc()`
@*/
PetscErrorCode DMGetWorkArray(DM dm, PetscInt count, MPI_Datatype dtype, void *mem)
{
  DMWorkLink  link;
  PetscMPIInt dsize;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(mem, 4);
  if (dm->workin) {
    link       = dm->workin;
    dm->workin = dm->workin->next;
  } else {
    PetscCall(PetscNew(&link));
  }
  PetscCallMPI(MPI_Type_size(dtype, &dsize));
  if (((size_t)dsize * count) > link->bytes) {
    PetscCall(PetscFree(link->mem));
    PetscCall(PetscMalloc(dsize * count, &link->mem));
    link->bytes = dsize * count;
  }
  link->next  = dm->workout;
  dm->workout = link;
#if defined(__MEMCHECK_H) && (defined(PLAT_amd64_linux) || defined(PLAT_x86_linux) || defined(PLAT_amd64_darwin))
  VALGRIND_MAKE_MEM_NOACCESS((char *)link->mem + (size_t)dsize * count, link->bytes - (size_t)dsize * count);
  VALGRIND_MAKE_MEM_UNDEFINED(link->mem, (size_t)dsize * count);
#endif
  *(void **)mem = link->mem;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMRestoreWorkArray - Restores a work array obtained with `DMCreateWorkArray()`

  Not Collective

  Input Parameters:
+ dm - the `DM` object
. count - The minimum size
- dtype - MPI data type, often `MPIU_REAL`, `MPIU_SCALAR`, `MPIU_INT`

  Output Parameter:
. array - the work array

  Level: developer

  Developer Note:
  count and dtype are ignored, they are only needed for `DMGetWorkArray()`

.seealso: [](chapter_dmbase), `DM`, `DMDestroy()`, `DMCreate()`, `DMGetWorkArray()`
@*/
PetscErrorCode DMRestoreWorkArray(DM dm, PetscInt count, MPI_Datatype dtype, void *mem)
{
  DMWorkLink *p, link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(mem, 4);
  for (p = &dm->workout; (link = *p); p = &link->next) {
    if (link->mem == *(void **)mem) {
      *p            = link->next;
      link->next    = dm->workin;
      dm->workin    = link;
      *(void **)mem = NULL;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Array was not checked out");
}

/*@C
  DMSetNullSpaceConstructor - Provide a callback function which constructs the nullspace for a given field, defined with `DMAddField()`, when function spaces
  are joined or split, such as in `DMCreateSubDM()`

  Logically Collective; No Fortran Support

  Input Parameters:
+ dm     - The `DM`
. field  - The field number for the nullspace
- nullsp - A callback to create the nullspace

  Calling sequence of `nullsp`:
.vb
    PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
.ve
+  dm        - The present `DM`
.  origField - The field number given above, in the original `DM`
.  field     - The field number in dm
-  nullSpace - The nullspace for the given field

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMGetNullSpaceConstructor()`, `DMSetNearNullSpaceConstructor()`, `DMGetNearNullSpaceConstructor()`, `DMCreateSubDM()`, `DMCreateSuperDM()`
@*/
PetscErrorCode DMSetNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (*nullsp)(DM, PetscInt, PetscInt, MatNullSpace *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(field < 10, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %" PetscInt_FMT " >= 10 fields", field);
  dm->nullspaceConstructors[field] = nullsp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetNullSpaceConstructor - Return the callback function which constructs the nullspace for a given field, defined with `DMAddField()`

  Not Collective; No Fortran Support

  Input Parameters:
+ dm     - The `DM`
- field  - The field number for the nullspace

  Output Parameter:
. nullsp - A callback to create the nullspace

  Calling sequence of `nullsp`:
.vb
    PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
.ve
+  dm        - The present DM
.  origField - The field number given above, in the original DM
.  field     - The field number in dm
-  nullSpace - The nullspace for the given field

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMGetField()`, `DMSetNullSpaceConstructor()`, `DMSetNearNullSpaceConstructor()`, `DMGetNearNullSpaceConstructor()`, `DMCreateSubDM()`, `DMCreateSuperDM()`
@*/
PetscErrorCode DMGetNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (**nullsp)(DM, PetscInt, PetscInt, MatNullSpace *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(nullsp, 3);
  PetscCheck(field < 10, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %" PetscInt_FMT " >= 10 fields", field);
  *nullsp = dm->nullspaceConstructors[field];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSetNearNullSpaceConstructor - Provide a callback function which constructs the near-nullspace for a given field, defined with `DMAddField()`

  Logically Collective; No Fortran Support

  Input Parameters:
+ dm     - The `DM`
. field  - The field number for the nullspace
- nullsp - A callback to create the near-nullspace

  Calling sequence of `nullsp`:
.vb
    PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
.ve
+  dm        - The present `DM`
.  origField - The field number given above, in the original `DM`
.  field     - The field number in dm
-  nullSpace - The nullspace for the given field

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMGetNearNullSpaceConstructor()`, `DMSetNullSpaceConstructor()`, `DMGetNullSpaceConstructor()`, `DMCreateSubDM()`, `DMCreateSuperDM()`,
          `MatNullSpace`
@*/
PetscErrorCode DMSetNearNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (*nullsp)(DM, PetscInt, PetscInt, MatNullSpace *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(field < 10, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %" PetscInt_FMT " >= 10 fields", field);
  dm->nearnullspaceConstructors[field] = nullsp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetNearNullSpaceConstructor - Return the callback function which constructs the near-nullspace for a given field, defined with `DMAddField()`

  Not Collective; No Fortran Support

  Input Parameters:
+ dm     - The `DM`
- field  - The field number for the nullspace

  Output Parameter:
. nullsp - A callback to create the near-nullspace

  Calling sequence of `nullsp`:
.vb
    PetscErrorCode nullsp(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace)
.ve
+  dm        - The present `DM`
.  origField - The field number given above, in the original `DM`
.  field     - The field number in dm
-  nullSpace - The nullspace for the given field

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMGetField()`, `DMSetNearNullSpaceConstructor()`, `DMSetNullSpaceConstructor()`, `DMGetNullSpaceConstructor()`, `DMCreateSubDM()`,
          `MatNullSpace`, `DMCreateSuperDM()`
@*/
PetscErrorCode DMGetNearNullSpaceConstructor(DM dm, PetscInt field, PetscErrorCode (**nullsp)(DM, PetscInt, PetscInt, MatNullSpace *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(nullsp, 3);
  PetscCheck(field < 10, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %" PetscInt_FMT " >= 10 fields", field);
  *nullsp = dm->nearnullspaceConstructors[field];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateFieldIS - Creates a set of `IS` objects with the global indices of dofs for each field defined with `DMAddField()`

  Not Collective; No Fortran Support

  Input Parameter:
. dm - the `DM` object

  Output Parameters:
+ numFields  - The number of fields (or `NULL` if not requested)
. fieldNames - The number of each field (or `NULL` if not requested)
- fields     - The global indices for each field (or `NULL` if not requested)

  Level: intermediate

  Note:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  `PetscFree()`, every entry of fields should be destroyed with `ISDestroy()`, and both arrays should be freed with
  `PetscFree()`.

  Developer Note:
  It is not clear why both this function and `DMCreateFieldDecomposition()` exist. Having two seems redundant and confusing. This function should
  likely be removed.

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMGetField()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`,
          `DMCreateFieldDecomposition()`
@*/
PetscErrorCode DMCreateFieldIS(DM dm, PetscInt *numFields, char ***fieldNames, IS **fields)
{
  PetscSection section, sectionGlobal;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numFields) {
    PetscValidIntPointer(numFields, 2);
    *numFields = 0;
  }
  if (fieldNames) {
    PetscValidPointer(fieldNames, 3);
    *fieldNames = NULL;
  }
  if (fields) {
    PetscValidPointer(fields, 4);
    *fields = NULL;
  }
  PetscCall(DMGetLocalSection(dm, &section));
  if (section) {
    PetscInt *fieldSizes, *fieldNc, **fieldIndices;
    PetscInt  nF, f, pStart, pEnd, p;

    PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
    PetscCall(PetscSectionGetNumFields(section, &nF));
    PetscCall(PetscMalloc3(nF, &fieldSizes, nF, &fieldNc, nF, &fieldIndices));
    PetscCall(PetscSectionGetChart(sectionGlobal, &pStart, &pEnd));
    for (f = 0; f < nF; ++f) {
      fieldSizes[f] = 0;
      PetscCall(PetscSectionGetFieldComponents(section, f, &fieldNc[f]));
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof;

      PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
      if (gdof > 0) {
        for (f = 0; f < nF; ++f) {
          PetscInt fdof, fcdof, fpdof;

          PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
          PetscCall(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
          fpdof = fdof - fcdof;
          if (fpdof && fpdof != fieldNc[f]) {
            /* Layout does not admit a pointwise block size */
            fieldNc[f] = 1;
          }
          fieldSizes[f] += fpdof;
        }
      }
    }
    for (f = 0; f < nF; ++f) {
      PetscCall(PetscMalloc1(fieldSizes[f], &fieldIndices[f]));
      fieldSizes[f] = 0;
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
      if (gdof > 0) {
        PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
        for (f = 0; f < nF; ++f) {
          PetscInt fdof, fcdof, fc;

          PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
          PetscCall(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
          for (fc = 0; fc < fdof - fcdof; ++fc, ++fieldSizes[f]) fieldIndices[f][fieldSizes[f]] = goff++;
        }
      }
    }
    if (numFields) *numFields = nF;
    if (fieldNames) {
      PetscCall(PetscMalloc1(nF, fieldNames));
      for (f = 0; f < nF; ++f) {
        const char *fieldName;

        PetscCall(PetscSectionGetFieldName(section, f, &fieldName));
        PetscCall(PetscStrallocpy(fieldName, (char **)&(*fieldNames)[f]));
      }
    }
    if (fields) {
      PetscCall(PetscMalloc1(nF, fields));
      for (f = 0; f < nF; ++f) {
        PetscInt bs, in[2], out[2];

        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), fieldSizes[f], fieldIndices[f], PETSC_OWN_POINTER, &(*fields)[f]));
        in[0] = -fieldNc[f];
        in[1] = fieldNc[f];
        PetscCall(MPIU_Allreduce(in, out, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
        bs = (-out[0] == out[1]) ? out[1] : 1;
        PetscCall(ISSetBlockSize((*fields)[f], bs));
      }
    }
    PetscCall(PetscFree3(fieldSizes, fieldNc, fieldIndices));
  } else PetscTryTypeMethod(dm, createfieldis, numFields, fieldNames, fields);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateFieldDecomposition - Returns a list of `IS` objects defining a decomposition of a problem into subproblems
                          corresponding to different fields: each `IS` contains the global indices of the dofs of the
                          corresponding field, defined by `DMAddField()`. The optional list of `DM`s define the `DM` for each subproblem.
                          The same as `DMCreateFieldIS()` but also returns a `DM` for each field.

  Not Collective; No Fortran Support

  Input Parameter:
. dm - the `DM` object

  Output Parameters:
+ len       - The number of fields (or `NULL` if not requested)
. namelist  - The name for each field (or `NULL` if not requested)
. islist    - The global indices for each field (or `NULL` if not requested)
- dmlist    - The `DM`s for each field subproblem (or `NULL`, if not requested; if `NULL` is returned, no `DM`s are defined)

  Level: intermediate

  Note:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  `PetscFree()`, every entry of is should be destroyed with `ISDestroy()`, every entry of dm should be destroyed with `DMDestroy()`,
  and all of the arrays should be freed with `PetscFree()`.

  Developer Note:
  It is not clear why this function and `DMCreateFieldIS()` exist. Having two seems redundant and confusing.

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMCreateFieldIS()`, `DMCreateSubDM()`, `DMCreateDomainDecomposition()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMCreateFieldIS()`
@*/
PetscErrorCode DMCreateFieldDecomposition(DM dm, PetscInt *len, char ***namelist, IS **islist, DM **dmlist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (len) {
    PetscValidIntPointer(len, 2);
    *len = 0;
  }
  if (namelist) {
    PetscValidPointer(namelist, 3);
    *namelist = NULL;
  }
  if (islist) {
    PetscValidPointer(islist, 4);
    *islist = NULL;
  }
  if (dmlist) {
    PetscValidPointer(dmlist, 5);
    *dmlist = NULL;
  }
  /*
   Is it a good idea to apply the following check across all impls?
   Perhaps some impls can have a well-defined decomposition before DMSetUp?
   This, however, follows the general principle that accessors are not well-behaved until the object is set up.
   */
  PetscCheck(dm->setupcalled, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Decomposition defined only after DMSetUp");
  if (!dm->ops->createfielddecomposition) {
    PetscSection section;
    PetscInt     numFields, f;

    PetscCall(DMGetLocalSection(dm, &section));
    if (section) PetscCall(PetscSectionGetNumFields(section, &numFields));
    if (section && numFields && dm->ops->createsubdm) {
      if (len) *len = numFields;
      if (namelist) PetscCall(PetscMalloc1(numFields, namelist));
      if (islist) PetscCall(PetscMalloc1(numFields, islist));
      if (dmlist) PetscCall(PetscMalloc1(numFields, dmlist));
      for (f = 0; f < numFields; ++f) {
        const char *fieldName;

        PetscCall(DMCreateSubDM(dm, 1, &f, islist ? &(*islist)[f] : NULL, dmlist ? &(*dmlist)[f] : NULL));
        if (namelist) {
          PetscCall(PetscSectionGetFieldName(section, f, &fieldName));
          PetscCall(PetscStrallocpy(fieldName, (char **)&(*namelist)[f]));
        }
      }
    } else {
      PetscCall(DMCreateFieldIS(dm, len, namelist, islist));
      /* By default there are no DMs associated with subproblems. */
      if (dmlist) *dmlist = NULL;
    }
  } else PetscUseTypeMethod(dm, createfielddecomposition, len, namelist, islist, dmlist);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateSubDM - Returns an `IS` and `DM` encapsulating a subproblem defined by the fields passed in.
                  The fields are defined by `DMCreateFieldIS()`.

  Not collective

  Input Parameters:
+ dm        - The `DM` object
. numFields - The number of fields to select
- fields    - The field numbers of the selected fields

  Output Parameters:
+ is - The global indices for all the degrees of freedom in the new sub `DM`
- subdm - The `DM` for the subproblem

  Level: intermediate

  Note:
  You need to call `DMPlexSetMigrationSF()` on the original `DM` if you want the Global-To-Natural map to be automatically constructed

.seealso: [](chapter_dmbase), `DM`, `DMCreateFieldIS()`, `DMCreateFieldDecomposition()`, `DMAddField()`, `DMCreateSuperDM()`, `DM`, `IS`, `DMPlexSetMigrationSF()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMCreateFieldIS()`
@*/
PetscErrorCode DMCreateSubDM(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(fields, 3);
  if (is) PetscValidPointer(is, 4);
  if (subdm) PetscValidPointer(subdm, 5);
  PetscUseTypeMethod(dm, createsubdm, numFields, fields, is, subdm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateSuperDM - Returns an arrays of `IS` and `DM` encapsulating a superproblem defined by multiple `DM`s passed in.

  Not collective

  Input Parameters:
+ dms - The `DM` objects
- n - The number of `DM`s

  Output Parameters:
+ is - The global indices for each of subproblem within the super `DM`, or NULL
- superdm - The `DM` for the superproblem

  Level: intermediate

  Note:
  You need to call `DMPlexSetMigrationSF()` on the original `DM` if you want the Global-To-Natural map to be automatically constructed

.seealso: [](chapter_dmbase), `DM`, `DMCreateSubDM()`, `DMPlexSetMigrationSF()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMCreateFieldIS()`
@*/
PetscErrorCode DMCreateSuperDM(DM dms[], PetscInt n, IS **is, DM *superdm)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidPointer(dms, 1);
  for (i = 0; i < n; ++i) PetscValidHeaderSpecific(dms[i], DM_CLASSID, 1);
  if (is) PetscValidPointer(is, 3);
  PetscValidPointer(superdm, 4);
  PetscCheck(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of DMs must be nonnegative: %" PetscInt_FMT, n);
  if (n) {
    DM dm = dms[0];
    PetscCall((*dm->ops->createsuperdm)(dms, n, is, superdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateDomainDecomposition - Returns lists of `IS` objects defining a decomposition of a problem into subproblems
                          corresponding to restrictions to pairs of nested subdomains: each `IS` contains the global
                          indices of the dofs of the corresponding subdomains with in the dofs of the original `DM`.
                          The inner subdomains conceptually define a nonoverlapping covering, while outer subdomains can overlap.
                          The optional list of `DM`s define a `DM` for each subproblem.

  Not Collective

  Input Parameter:
. dm - the `DM` object

  Output Parameters:
+ n           - The number of subproblems in the domain decomposition (or `NULL` if not requested)
. namelist    - The name for each subdomain (or `NULL` if not requested)
. innerislist - The global indices for each inner subdomain (or NULL, if not requested)
. outerislist - The global indices for each outer subdomain (or NULL, if not requested)
- dmlist      - The `DM`s for each subdomain subproblem (or NULL, if not requested; if `NULL` is returned, no `DM`s are defined)

  Level: intermediate

  Note:
  The user is responsible for freeing all requested arrays. In particular, every entry of names should be freed with
  `PetscFree()`, every entry of is should be destroyed with `ISDestroy()`, every entry of dm should be destroyed with `DMDestroy()`,
  and all of the arrays should be freed with `PetscFree()`.

  Questions:
  The `dmlist` is for the inner subdomains or the outer subdomains or all subdomains?

.seealso: [](chapter_dmbase), `DM`, `DMCreateFieldDecomposition()`, `DMDestroy()`, `DMCreateDomainDecompositionScatters()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMCreateFieldDecomposition()`
@*/
PetscErrorCode DMCreateDomainDecomposition(DM dm, PetscInt *n, char ***namelist, IS **innerislist, IS **outerislist, DM **dmlist)
{
  DMSubDomainHookLink link;
  PetscInt            i, l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (n) {
    PetscValidIntPointer(n, 2);
    *n = 0;
  }
  if (namelist) {
    PetscValidPointer(namelist, 3);
    *namelist = NULL;
  }
  if (innerislist) {
    PetscValidPointer(innerislist, 4);
    *innerislist = NULL;
  }
  if (outerislist) {
    PetscValidPointer(outerislist, 5);
    *outerislist = NULL;
  }
  if (dmlist) {
    PetscValidPointer(dmlist, 6);
    *dmlist = NULL;
  }
  /*
   Is it a good idea to apply the following check across all impls?
   Perhaps some impls can have a well-defined decomposition before DMSetUp?
   This, however, follows the general principle that accessors are not well-behaved until the object is set up.
   */
  PetscCheck(dm->setupcalled, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Decomposition defined only after DMSetUp");
  if (dm->ops->createdomaindecomposition) {
    PetscUseTypeMethod(dm, createdomaindecomposition, &l, namelist, innerislist, outerislist, dmlist);
    /* copy subdomain hooks and context over to the subdomain DMs */
    if (dmlist && *dmlist) {
      for (i = 0; i < l; i++) {
        for (link = dm->subdomainhook; link; link = link->next) {
          if (link->ddhook) PetscCall((*link->ddhook)(dm, (*dmlist)[i], link->ctx));
        }
        if (dm->ctx) (*dmlist)[i]->ctx = dm->ctx;
      }
    }
    if (n) *n = l;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateDomainDecompositionScatters - Returns scatters to the subdomain vectors from the global vector

  Not Collective

  Input Parameters:
+ dm - the `DM` object
. n  - the number of subdomain scatters
- subdms - the local subdomains

  Output Parameters:
+ iscat - scatter from global vector to nonoverlapping global vector entries on subdomain
. oscat - scatter from global vector to overlapping global vector entries on subdomain
- gscat - scatter from global vector to local vector on subdomain (fills in ghosts)

  Level: developer

  Note:
    This is an alternative to the iis and ois arguments in `DMCreateDomainDecomposition()` that allow for the solution
  of general nonlinear problems with overlapping subdomain methods.  While merely having index sets that enable subsets
  of the residual equations to be created is fine for linear problems, nonlinear problems require local assembly of
  solution and residual data.

  Questions:
  Can the subdms input be anything or are they exactly the `DM` obtained from `DMCreateDomainDecomposition()`?

.seealso: [](chapter_dmbase), `DM`, `DMCreateDomainDecomposition()`, `DMDestroy()`, `DMView()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMCreateFieldIS()`
@*/
PetscErrorCode DMCreateDomainDecompositionScatters(DM dm, PetscInt n, DM *subdms, VecScatter **iscat, VecScatter **oscat, VecScatter **gscat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subdms, 3);
  PetscUseTypeMethod(dm, createddscatters, n, subdms, iscat, oscat, gscat);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMRefine - Refines a `DM` object using a standard nonadaptive refinement of the underlying mesh

  Collective

  Input Parameters:
+ dm   - the `DM` object
- comm - the communicator to contain the new `DM` object (or `MPI_COMM_NULL`)

  Output Parameter:
. dmf - the refined `DM`, or `NULL`

  Options Database Key:
. -dm_plex_cell_refiner <strategy> - chooses the refinement strategy, e.g. regular, tohex

  Level: developer

  Note:
  If no refinement was done, the return value is `NULL`

.seealso: [](chapter_dmbase), `DM`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMRefine(DM dm, MPI_Comm comm, DM *dmf)
{
  DMRefineHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscLogEventBegin(DM_Refine, dm, 0, 0, 0));
  PetscUseTypeMethod(dm, refine, comm, dmf);
  if (*dmf) {
    (*dmf)->ops->creatematrix = dm->ops->creatematrix;

    PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)dm, (PetscObject)*dmf));

    (*dmf)->ctx       = dm->ctx;
    (*dmf)->leveldown = dm->leveldown;
    (*dmf)->levelup   = dm->levelup + 1;

    PetscCall(DMSetMatType(*dmf, dm->mattype));
    for (link = dm->refinehook; link; link = link->next) {
      if (link->refinehook) PetscCall((*link->refinehook)(dm, *dmf, link->ctx));
    }
  }
  PetscCall(PetscLogEventEnd(DM_Refine, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMRefineHookAdd - adds a callback to be run when interpolating a nonlinear problem to a finer grid

   Logically Collective; No Fortran Support

   Input Parameters:
+  coarse - `DM` on which to run a hook when interpolating to a finer level
.  refinehook - function to run when setting up the finer level
.  interphook - function to run to update data on finer levels (once per `SNESSolve`())
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Calling sequence of `refinehook`:
$  PetscErrorCode refinehook(DM coarse, DM fine, void *ctx);
+  coarse - coarse level `DM`
.  fine - fine level `DM` to interpolate problem to
-  ctx - optional user-defined function context

   Calling sequence of `interphook`:
$  PetscErrorCode interphook(DM coarse, Mat interp, DM fine,void *ctx)
+  coarse - coarse level `DM`
.  interp - matrix interpolating a coarse-level solution to the finer grid
.  fine - fine level `DM` to update
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data that is attached to the `DM`s via, for example, `PetscObjectCompose()`, needs to be
   passed to fine grids while grid sequencing.

   The actual interpolation is done when `DMInterpolate()` is called.

   If this function is called multiple times, the hooks will be run in the order they are added.

.seealso: [](chapter_dmbase), `DM`, `DMCoarsenHookAdd()`, `DMInterpolate()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMRefineHookAdd(DM coarse, PetscErrorCode (*refinehook)(DM, DM, void *), PetscErrorCode (*interphook)(DM, Mat, DM, void *), void *ctx)
{
  DMRefineHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse, DM_CLASSID, 1);
  for (p = &coarse->refinehook; *p; p = &(*p)->next) { /* Scan to the end of the current list of hooks */
    if ((*p)->refinehook == refinehook && (*p)->interphook == interphook && (*p)->ctx == ctx) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscNew(&link));
  link->refinehook = refinehook;
  link->interphook = interphook;
  link->ctx        = ctx;
  link->next       = NULL;
  *p               = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMRefineHookRemove - remove a callback from the list of hooks, that have been set with `DMRefineHookAdd()`, to be run when interpolating
    a nonlinear problem to a finer grid

   Logically Collective; No Fortran Support

   Input Parameters:
+  coarse - the `DM` on which to run a hook when restricting to a coarser level
.  refinehook - function to run when setting up a finer level
.  interphook - function to run to update data on finer levels
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Level: advanced

   Note:
   This function does nothing if the hook is not in the list.

.seealso: [](chapter_dmbase), `DM`, `DMRefineHookAdd()`, `DMCoarsenHookRemove()`, `DMInterpolate()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMRefineHookRemove(DM coarse, PetscErrorCode (*refinehook)(DM, DM, void *), PetscErrorCode (*interphook)(DM, Mat, DM, void *), void *ctx)
{
  DMRefineHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse, DM_CLASSID, 1);
  for (p = &coarse->refinehook; *p; p = &(*p)->next) { /* Search the list of current hooks */
    if ((*p)->refinehook == refinehook && (*p)->interphook == interphook && (*p)->ctx == ctx) {
      link = *p;
      *p   = link->next;
      PetscCall(PetscFree(link));
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMInterpolate - interpolates user-defined problem data attached to a `DM` to a finer `DM` by running hooks registered by `DMRefineHookAdd()`

   Collective if any hooks are

   Input Parameters:
+  coarse - coarser `DM` to use as a base
.  interp - interpolation matrix, apply using `MatInterpolate()`
-  fine - finer `DM` to update

   Level: developer

   Developer Note:
   This routine is called `DMInterpolate()` while the hook is called `DMRefineHookAdd()`. It would be better to have an
   an API with consistent terminology.

.seealso: [](chapter_dmbase), `DM`, `DMRefineHookAdd()`, `MatInterpolate()`
@*/
PetscErrorCode DMInterpolate(DM coarse, Mat interp, DM fine)
{
  DMRefineHookLink link;

  PetscFunctionBegin;
  for (link = fine->refinehook; link; link = link->next) {
    if (link->interphook) PetscCall((*link->interphook)(coarse, interp, fine, link->ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMInterpolateSolution - Interpolates a solution from a coarse mesh to a fine mesh.

   Collective

   Input Parameters:
+  coarse - coarse `DM`
.  fine   - fine `DM`
.  interp - (optional) the matrix computed by `DMCreateInterpolation()`.  Implementations may not need this, but if it
            is available it can avoid some recomputation.  If it is provided, `MatInterpolate()` will be used if
            the coarse `DM` does not have a specialized implementation.
-  coarseSol - solution on the coarse mesh

   Output Parameter:
.  fineSol - the interpolation of coarseSol to the fine mesh

   Level: developer

   Note:
   This function exists because the interpolation of a solution vector between meshes is not always a linear
   map.  For example, if a boundary value problem has an inhomogeneous Dirichlet boundary condition that is compressed
   out of the solution vector.  Or if interpolation is inherently a nonlinear operation, such as a method using
   slope-limiting reconstruction.

   Developer Note:
   This doesn't just interpolate "solutions" so its API name is questionable.

.seealso: [](chapter_dmbase), `DM`, `DMInterpolate()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMInterpolateSolution(DM coarse, DM fine, Mat interp, Vec coarseSol, Vec fineSol)
{
  PetscErrorCode (*interpsol)(DM, DM, Mat, Vec, Vec) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse, DM_CLASSID, 1);
  if (interp) PetscValidHeaderSpecific(interp, MAT_CLASSID, 3);
  PetscValidHeaderSpecific(coarseSol, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(fineSol, VEC_CLASSID, 5);

  PetscCall(PetscObjectQueryFunction((PetscObject)coarse, "DMInterpolateSolution_C", &interpsol));
  if (interpsol) {
    PetscCall((*interpsol)(coarse, fine, interp, coarseSol, fineSol));
  } else if (interp) {
    PetscCall(MatInterpolate(interp, coarseSol, fineSol));
  } else SETERRQ(PetscObjectComm((PetscObject)coarse), PETSC_ERR_SUP, "DM %s does not implement DMInterpolateSolution()", ((PetscObject)coarse)->type_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGetRefineLevel - Gets the number of refinements that have generated this `DM` from some initial `DM`.

    Not Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   level - number of refinements

    Level: developer

    Note:
    This can be used, by example, to set the number of coarser levels associated with this `DM` for a multigrid solver.

.seealso: [](chapter_dmbase), `DM`, `DMRefine()`, `DMCoarsen()`, `DMGetCoarsenLevel()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMGetRefineLevel(DM dm, PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *level = dm->levelup;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMSetRefineLevel - Sets the number of refinements that have generated this `DM`.

    Not Collective

    Input Parameters:
+   dm - the `DM` object
-   level - number of refinements

    Level: advanced

    Notes:
    This value is used by `PCMG` to determine how many multigrid levels to use

    The values are usually set automatically by the process that is causing the refinements of an initial `DM` by calling this routine.

.seealso: [](chapter_dmbase), `DM`, `DMGetRefineLevel()`, `DMCoarsen()`, `DMGetCoarsenLevel()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMSetRefineLevel(DM dm, PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->levelup = level;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMExtrude - Extrude a `DM` object from a surface

  Collective

  Input Parameters:
+ dm     - the `DM` object
- layers - the number of extruded cell layers

  Output Parameter:
. dme - the extruded `DM`, or `NULL`

  Level: developer

  Note:
  If no extrusion was done, the return value is `NULL`

.seealso: [](chapter_dmbase), `DM`, `DMRefine()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`
@*/
PetscErrorCode DMExtrude(DM dm, PetscInt layers, DM *dme)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscUseTypeMethod(dm, extrude, layers, dme);
  if (*dme) {
    (*dme)->ops->creatematrix = dm->ops->creatematrix;
    PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)dm, (PetscObject)*dme));
    (*dme)->ctx = dm->ctx;
    PetscCall(DMSetMatType(*dme, dm->mattype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetBasisTransformDM_Internal(DM dm, DM *tdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(tdm, 2);
  *tdm = dm->transformDM;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetBasisTransformVec_Internal(DM dm, Vec *tv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(tv, 2);
  *tv = dm->transform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMHasBasisTransform - Whether the `DM` employs a basis transformation from functions in global vectors to functions in local vectors

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. flg - `PETSC_TRUE` if a basis transformation should be done

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMPlexGlobalToLocalBasis()`, `DMPlexLocalToGlobalBasis()`, `DMPlexCreateBasisRotation()`
@*/
PetscErrorCode DMHasBasisTransform(DM dm, PetscBool *flg)
{
  Vec tv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  *flg = tv ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMConstructBasisTransform_Internal(DM dm)
{
  PetscSection s, ts;
  PetscScalar *ta;
  PetscInt     cdim, pStart, pEnd, p, Nf, f, Nc, dof;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(PetscSectionGetNumFields(s, &Nf));
  PetscCall(DMClone(dm, &dm->transformDM));
  PetscCall(DMGetLocalSection(dm->transformDM, &ts));
  PetscCall(PetscSectionSetNumFields(ts, Nf));
  PetscCall(PetscSectionSetChart(ts, pStart, pEnd));
  for (f = 0; f < Nf; ++f) {
    PetscCall(PetscSectionGetFieldComponents(s, f, &Nc));
    /* We could start to label fields by their transformation properties */
    if (Nc != cdim) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscCall(PetscSectionGetFieldDof(s, p, f, &dof));
      if (!dof) continue;
      PetscCall(PetscSectionSetFieldDof(ts, p, f, PetscSqr(cdim)));
      PetscCall(PetscSectionAddDof(ts, p, PetscSqr(cdim)));
    }
  }
  PetscCall(PetscSectionSetUp(ts));
  PetscCall(DMCreateLocalVector(dm->transformDM, &dm->transform));
  PetscCall(VecGetArray(dm->transform, &ta));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < Nf; ++f) {
      PetscCall(PetscSectionGetFieldDof(ts, p, f, &dof));
      if (dof) {
        PetscReal          x[3] = {0.0, 0.0, 0.0};
        PetscScalar       *tva;
        const PetscScalar *A;

        /* TODO Get quadrature point for this dual basis vector for coordinate */
        PetscCall((*dm->transformGetMatrix)(dm, x, PETSC_TRUE, &A, dm->transformCtx));
        PetscCall(DMPlexPointLocalFieldRef(dm->transformDM, p, f, ta, (void *)&tva));
        PetscCall(PetscArraycpy(tva, A, PetscSqr(cdim)));
      }
    }
  }
  PetscCall(VecRestoreArray(dm->transform, &ta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCopyTransform(DM dm, DM newdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(newdm, DM_CLASSID, 2);
  newdm->transformCtx       = dm->transformCtx;
  newdm->transformSetUp     = dm->transformSetUp;
  newdm->transformDestroy   = NULL;
  newdm->transformGetMatrix = dm->transformGetMatrix;
  if (newdm->transformSetUp) PetscCall(DMConstructBasisTransform_Internal(newdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMGlobalToLocalHookAdd - adds a callback to be run when `DMGlobalToLocal()` is called

   Logically Collective

   Input Parameters:
+  dm - the `DM`
.  beginhook - function to run at the beginning of `DMGlobalToLocalBegin()`
.  endhook - function to run after `DMGlobalToLocalEnd()` has completed
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Calling sequence of `beginhook`:
$  PetscErrorCode  beginhook(DM fine, VecScatter out, VecScatter in, DM coarse, void *ctx)
+  dm - global DM
.  g - global vector
.  mode - mode
.  l - local vector
-  ctx - optional user-defined function context

   Calling sequence of `endhook`:
$  PetscErrorCode  endhook(DM fine, VecScatter out, VecScatter in, DM coarse, void *ctx)
+  global - global DM
-  ctx - optional user-defined function context

   Level: advanced

   Note:
   The hook may be used to provide, for example, values that represent boundary conditions in the local vectors that do not exist on the global vector.

.seealso: [](chapter_dmbase), `DM`, `DMGlobalToLocal()`, `DMRefineHookAdd()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMGlobalToLocalHookAdd(DM dm, PetscErrorCode (*beginhook)(DM, Vec, InsertMode, Vec, void *), PetscErrorCode (*endhook)(DM, Vec, InsertMode, Vec, void *), void *ctx)
{
  DMGlobalToLocalHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (p = &dm->gtolhook; *p; p = &(*p)->next) { } /* Scan to the end of the current list of hooks */
  PetscCall(PetscNew(&link));
  link->beginhook = beginhook;
  link->endhook   = endhook;
  link->ctx       = ctx;
  link->next      = NULL;
  *p              = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGlobalToLocalHook_Constraints(DM dm, Vec g, InsertMode mode, Vec l, void *ctx)
{
  Mat          cMat;
  Vec          cVec, cBias;
  PetscSection section, cSec;
  PetscInt     pStart, pEnd, p, dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDefaultConstraints(dm, &cSec, &cMat, &cBias));
  if (cMat && (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES)) {
    PetscInt nRows;

    PetscCall(MatGetSize(cMat, &nRows, NULL));
    if (nRows <= 0) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(MatCreateVecs(cMat, NULL, &cVec));
    PetscCall(MatMult(cMat, l, cVec));
    if (cBias) PetscCall(VecAXPY(cVec, 1., cBias));
    PetscCall(PetscSectionGetChart(cSec, &pStart, &pEnd));
    for (p = pStart; p < pEnd; p++) {
      PetscCall(PetscSectionGetDof(cSec, p, &dof));
      if (dof) {
        PetscScalar *vals;
        PetscCall(VecGetValuesSection(cVec, cSec, p, &vals));
        PetscCall(VecSetValuesSection(l, section, p, vals, INSERT_ALL_VALUES));
      }
    }
    PetscCall(VecDestroy(&cVec));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGlobalToLocal - update local vectors from global vector

    Neighbor-wise Collective

    Input Parameters:
+   dm - the `DM` object
.   g - the global vector
.   mode - `INSERT_VALUES` or `ADD_VALUES`
-   l - the local vector

    Level: beginner

    Notes:
    The communication involved in this update can be overlapped with computation by instead using
    `DMGlobalToLocalBegin()` and `DMGlobalToLocalEnd()`.

    `DMGlobalToLocalHookAdd()` may be used to provide additional operations that are performed during the update process.

.seealso: [](chapter_dmbase), `DM`, `DMGlobalToLocalHookAdd()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobal()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobalEnd()`,
          `DMGlobalToLocalBegin()` `DMGlobalToLocalEnd()`
@*/
PetscErrorCode DMGlobalToLocal(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscFunctionBegin;
  PetscCall(DMGlobalToLocalBegin(dm, g, mode, l));
  PetscCall(DMGlobalToLocalEnd(dm, g, mode, l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGlobalToLocalBegin - Begins updating local vectors from global vector

    Neighbor-wise Collective

    Input Parameters:
+   dm - the `DM` object
.   g - the global vector
.   mode - `INSERT_VALUES` or `ADD_VALUES`
-   l - the local vector

    Level: intermediate

    Notes:
    The operation is completed with `DMGlobalToLocalEnd()`

    One can perform local computations between the `DMGlobalToLocalBegin()` and  `DMGlobalToLocalEnd()` to overlap communication and computation

    `DMGlobalToLocal()` is a short form of  `DMGlobalToLocalBegin()` and  `DMGlobalToLocalEnd()`

    `DMGlobalToLocalHookAdd()` may be used to provide additional operations that are performed during the update process.

.seealso: [](chapter_dmbase), `DM`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMGlobalToLocal()`, `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobal()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobalEnd()`
@*/
PetscErrorCode DMGlobalToLocalBegin(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscSF                 sf;
  DMGlobalToLocalHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (link = dm->gtolhook; link; link = link->next) {
    if (link->beginhook) PetscCall((*link->beginhook)(dm, g, mode, l, link->ctx));
  }
  PetscCall(DMGetSectionSF(dm, &sf));
  if (sf) {
    const PetscScalar *gArray;
    PetscScalar       *lArray;
    PetscMemType       lmtype, gmtype;

    PetscCheck(mode != ADD_VALUES, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", (int)mode);
    PetscCall(VecGetArrayAndMemType(l, &lArray, &lmtype));
    PetscCall(VecGetArrayReadAndMemType(g, &gArray, &gmtype));
    PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, gmtype, gArray, lmtype, lArray, MPI_REPLACE));
    PetscCall(VecRestoreArrayAndMemType(l, &lArray));
    PetscCall(VecRestoreArrayReadAndMemType(g, &gArray));
  } else {
    PetscCall((*dm->ops->globaltolocalbegin)(dm, g, mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode), l));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGlobalToLocalEnd - Ends updating local vectors from global vector

    Neighbor-wise Collective

    Input Parameters:
+   dm - the `DM` object
.   g - the global vector
.   mode - `INSERT_VALUES` or `ADD_VALUES`
-   l - the local vector

    Level: intermediate

    Note:
    See `DMGlobalToLocalBegin()` for details.

.seealso: [](chapter_dmbase), `DM`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMGlobalToLocal()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobal()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobalEnd()`
@*/
PetscErrorCode DMGlobalToLocalEnd(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscSF                 sf;
  const PetscScalar      *gArray;
  PetscScalar            *lArray;
  PetscBool               transform;
  DMGlobalToLocalHookLink link;
  PetscMemType            lmtype, gmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetSectionSF(dm, &sf));
  PetscCall(DMHasBasisTransform(dm, &transform));
  if (sf) {
    PetscCheck(mode != ADD_VALUES, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", (int)mode);

    PetscCall(VecGetArrayAndMemType(l, &lArray, &lmtype));
    PetscCall(VecGetArrayReadAndMemType(g, &gArray, &gmtype));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, gArray, lArray, MPI_REPLACE));
    PetscCall(VecRestoreArrayAndMemType(l, &lArray));
    PetscCall(VecRestoreArrayReadAndMemType(g, &gArray));
    if (transform) PetscCall(DMPlexGlobalToLocalBasis(dm, l));
  } else {
    PetscCall((*dm->ops->globaltolocalend)(dm, g, mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode), l));
  }
  PetscCall(DMGlobalToLocalHook_Constraints(dm, g, mode, l, NULL));
  for (link = dm->gtolhook; link; link = link->next) {
    if (link->endhook) PetscCall((*link->endhook)(dm, g, mode, l, link->ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMLocalToGlobalHookAdd - adds a callback to be run when a local to global is called

   Logically Collective

   Input Parameters:
+  dm - the `DM`
.  beginhook - function to run at the beginning of `DMLocalToGlobalBegin()`
.  endhook - function to run after `DMLocalToGlobalEnd()` has completed
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Calling sequence of `beginhook`:
$  PetscErrorCode  beginhook(DM fine, Vec l, InsertMode mode, Vec g, void *ctx)
+  dm - global `DM`
.  l - local vector
.  mode - mode
.  g - global vector
-  ctx - optional user-defined function context

   Calling sequence of `endhook`:
$  PetscErrorCode  endhook(DM fine, Vec l, InsertMode mode, Vec g, void *ctx)
+  global - global `DM`
.  l - local vector
.  mode - mode
.  g - global vector
-  ctx - optional user-defined function context

   Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMLocalToGlobal()`, `DMRefineHookAdd()`, `DMGlobalToLocalHookAdd()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMLocalToGlobalHookAdd(DM dm, PetscErrorCode (*beginhook)(DM, Vec, InsertMode, Vec, void *), PetscErrorCode (*endhook)(DM, Vec, InsertMode, Vec, void *), void *ctx)
{
  DMLocalToGlobalHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (p = &dm->ltoghook; *p; p = &(*p)->next) { } /* Scan to the end of the current list of hooks */
  PetscCall(PetscNew(&link));
  link->beginhook = beginhook;
  link->endhook   = endhook;
  link->ctx       = ctx;
  link->next      = NULL;
  *p              = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLocalToGlobalHook_Constraints(DM dm, Vec l, InsertMode mode, Vec g, void *ctx)
{
  Mat          cMat;
  Vec          cVec;
  PetscSection section, cSec;
  PetscInt     pStart, pEnd, p, dof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDefaultConstraints(dm, &cSec, &cMat, NULL));
  if (cMat && (mode == ADD_VALUES || mode == ADD_ALL_VALUES || mode == ADD_BC_VALUES)) {
    PetscInt nRows;

    PetscCall(MatGetSize(cMat, &nRows, NULL));
    if (nRows <= 0) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(MatCreateVecs(cMat, NULL, &cVec));
    PetscCall(PetscSectionGetChart(cSec, &pStart, &pEnd));
    for (p = pStart; p < pEnd; p++) {
      PetscCall(PetscSectionGetDof(cSec, p, &dof));
      if (dof) {
        PetscInt     d;
        PetscScalar *vals;
        PetscCall(VecGetValuesSection(l, section, p, &vals));
        PetscCall(VecSetValuesSection(cVec, cSec, p, vals, mode));
        /* for this to be the true transpose, we have to zero the values that
         * we just extracted */
        for (d = 0; d < dof; d++) vals[d] = 0.;
      }
    }
    PetscCall(MatMultTransposeAdd(cMat, cVec, l, l));
    PetscCall(VecDestroy(&cVec));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
    DMLocalToGlobal - updates global vectors from local vectors

    Neighbor-wise Collective

    Input Parameters:
+   dm - the `DM` object
.   l - the local vector
.   mode - if `INSERT_VALUES` then no parallel communication is used, if `ADD_VALUES` then all ghost points from the same base point accumulate into that base point.
-   g - the global vector

    Level: beginner

    Notes:
    The communication involved in this update can be overlapped with computation by using
    `DMLocalToGlobalBegin()` and `DMLocalToGlobalEnd()`.

    In the `ADD_VALUES` case you normally would zero the receiving vector before beginning this operation.

    `INSERT_VALUES` is not supported for `DMDA`; in that case simply compute the values directly into a global vector instead of a local one.

    Use `DMLocalToGlobalHookAdd()` to add additional operations that are performed on the data during the update process

.seealso: [](chapter_dmbase), `DM`, `DMLocalToGlobalBegin()`, `DMLocalToGlobalEnd()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMGlobalToLocal()`, `DMGlobalToLocalEnd()`, `DMGlobalToLocalBegin()`, `DMLocalToGlobalHookAdd()`, `DMGlobaToLocallHookAdd()`
@*/
PetscErrorCode DMLocalToGlobal(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscFunctionBegin;
  PetscCall(DMLocalToGlobalBegin(dm, l, mode, g));
  PetscCall(DMLocalToGlobalEnd(dm, l, mode, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMLocalToGlobalBegin - begins updating global vectors from local vectors

    Neighbor-wise Collective

    Input Parameters:
+   dm - the `DM` object
.   l - the local vector
.   mode - if `INSERT_VALUES` then no parallel communication is used, if `ADD_VALUES` then all ghost points from the same base point accumulate into that base point.
-   g - the global vector

    Level: intermediate

    Notes:
    In the `ADD_VALUES` case you normally would zero the receiving vector before beginning this operation.

    `INSERT_VALUES is` not supported for `DMDA`, in that case simply compute the values directly into a global vector instead of a local one.

    Use `DMLocalToGlobalEnd()` to complete the communication process.

    `DMLocalToGlobal()` is a short form of  `DMLocalToGlobalBegin()` and  `DMLocalToGlobalEnd()`

    `DMLocalToGlobalHookAdd()` may be used to provide additional operations that are performed during the update process.

.seealso: [](chapter_dmbase), `DM`, `DMLocalToGlobal()`, `DMLocalToGlobalEnd()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMGlobalToLocal()`, `DMGlobalToLocalEnd()`, `DMGlobalToLocalBegin()`
@*/
PetscErrorCode DMLocalToGlobalBegin(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscSF                 sf;
  PetscSection            s, gs;
  DMLocalToGlobalHookLink link;
  Vec                     tmpl;
  const PetscScalar      *lArray;
  PetscScalar            *gArray;
  PetscBool               isInsert, transform, l_inplace = PETSC_FALSE, g_inplace = PETSC_FALSE;
  PetscMemType            lmtype = PETSC_MEMTYPE_HOST, gmtype = PETSC_MEMTYPE_HOST;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (link = dm->ltoghook; link; link = link->next) {
    if (link->beginhook) PetscCall((*link->beginhook)(dm, l, mode, g, link->ctx));
  }
  PetscCall(DMLocalToGlobalHook_Constraints(dm, l, mode, g, NULL));
  PetscCall(DMGetSectionSF(dm, &sf));
  PetscCall(DMGetLocalSection(dm, &s));
  switch (mode) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
  case INSERT_BC_VALUES:
    isInsert = PETSC_TRUE;
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
  case ADD_BC_VALUES:
    isInsert = PETSC_FALSE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", mode);
  }
  if ((sf && !isInsert) || (s && isInsert)) {
    PetscCall(DMHasBasisTransform(dm, &transform));
    if (transform) {
      PetscCall(DMGetNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl));
      PetscCall(VecCopy(l, tmpl));
      PetscCall(DMPlexLocalToGlobalBasis(dm, tmpl));
      PetscCall(VecGetArrayRead(tmpl, &lArray));
    } else if (isInsert) {
      PetscCall(VecGetArrayRead(l, &lArray));
    } else {
      PetscCall(VecGetArrayReadAndMemType(l, &lArray, &lmtype));
      l_inplace = PETSC_TRUE;
    }
    if (s && isInsert) {
      PetscCall(VecGetArray(g, &gArray));
    } else {
      PetscCall(VecGetArrayAndMemType(g, &gArray, &gmtype));
      g_inplace = PETSC_TRUE;
    }
    if (sf && !isInsert) {
      PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_SCALAR, lmtype, lArray, gmtype, gArray, MPIU_SUM));
    } else if (s && isInsert) {
      PetscInt gStart, pStart, pEnd, p;

      PetscCall(DMGetGlobalSection(dm, &gs));
      PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
      PetscCall(VecGetOwnershipRange(g, &gStart, NULL));
      for (p = pStart; p < pEnd; ++p) {
        PetscInt dof, gdof, cdof, gcdof, off, goff, d, e;

        PetscCall(PetscSectionGetDof(s, p, &dof));
        PetscCall(PetscSectionGetDof(gs, p, &gdof));
        PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
        PetscCall(PetscSectionGetConstraintDof(gs, p, &gcdof));
        PetscCall(PetscSectionGetOffset(s, p, &off));
        PetscCall(PetscSectionGetOffset(gs, p, &goff));
        /* Ignore off-process data and points with no global data */
        if (!gdof || goff < 0) continue;
        PetscCheck(dof == gdof, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes, p: %" PetscInt_FMT " dof: %" PetscInt_FMT " gdof: %" PetscInt_FMT " cdof: %" PetscInt_FMT " gcdof: %" PetscInt_FMT, p, dof, gdof, cdof, gcdof);
        /* If no constraints are enforced in the global vector */
        if (!gcdof) {
          for (d = 0; d < dof; ++d) gArray[goff - gStart + d] = lArray[off + d];
          /* If constraints are enforced in the global vector */
        } else if (cdof == gcdof) {
          const PetscInt *cdofs;
          PetscInt        cind = 0;

          PetscCall(PetscSectionGetConstraintIndices(s, p, &cdofs));
          for (d = 0, e = 0; d < dof; ++d) {
            if ((cind < cdof) && (d == cdofs[cind])) {
              ++cind;
              continue;
            }
            gArray[goff - gStart + e++] = lArray[off + d];
          }
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes, p: %" PetscInt_FMT " dof: %" PetscInt_FMT " gdof: %" PetscInt_FMT " cdof: %" PetscInt_FMT " gcdof: %" PetscInt_FMT, p, dof, gdof, cdof, gcdof);
      }
    }
    if (g_inplace) {
      PetscCall(VecRestoreArrayAndMemType(g, &gArray));
    } else {
      PetscCall(VecRestoreArray(g, &gArray));
    }
    if (transform) {
      PetscCall(VecRestoreArrayRead(tmpl, &lArray));
      PetscCall(DMRestoreNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl));
    } else if (l_inplace) {
      PetscCall(VecRestoreArrayReadAndMemType(l, &lArray));
    } else {
      PetscCall(VecRestoreArrayRead(l, &lArray));
    }
  } else {
    PetscCall((*dm->ops->localtoglobalbegin)(dm, l, mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode), g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMLocalToGlobalEnd - updates global vectors from local vectors

    Neighbor-wise Collective

    Input Parameters:
+   dm - the `DM` object
.   l - the local vector
.   mode - `INSERT_VALUES` or `ADD_VALUES`
-   g - the global vector

    Level: intermediate

    Note:
    See `DMLocalToGlobalBegin()` for full details

.seealso: [](chapter_dmbase), `DM`, `DMLocalToGlobalBegin()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMGlobalToLocalEnd()`, `DMGlobalToLocalEnd()`
@*/
PetscErrorCode DMLocalToGlobalEnd(DM dm, Vec l, InsertMode mode, Vec g)
{
  PetscSF                 sf;
  PetscSection            s;
  DMLocalToGlobalHookLink link;
  PetscBool               isInsert, transform;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetSectionSF(dm, &sf));
  PetscCall(DMGetLocalSection(dm, &s));
  switch (mode) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    isInsert = PETSC_TRUE;
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    isInsert = PETSC_FALSE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insertion mode %d", mode);
  }
  if (sf && !isInsert) {
    const PetscScalar *lArray;
    PetscScalar       *gArray;
    Vec                tmpl;

    PetscCall(DMHasBasisTransform(dm, &transform));
    if (transform) {
      PetscCall(DMGetNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl));
      PetscCall(VecGetArrayRead(tmpl, &lArray));
    } else {
      PetscCall(VecGetArrayReadAndMemType(l, &lArray, NULL));
    }
    PetscCall(VecGetArrayAndMemType(g, &gArray, NULL));
    PetscCall(PetscSFReduceEnd(sf, MPIU_SCALAR, lArray, gArray, MPIU_SUM));
    if (transform) {
      PetscCall(VecRestoreArrayRead(tmpl, &lArray));
      PetscCall(DMRestoreNamedLocalVector(dm, "__petsc_dm_transform_local_copy", &tmpl));
    } else {
      PetscCall(VecRestoreArrayReadAndMemType(l, &lArray));
    }
    PetscCall(VecRestoreArrayAndMemType(g, &gArray));
  } else if (s && isInsert) {
  } else {
    PetscCall((*dm->ops->localtoglobalend)(dm, l, mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode), g));
  }
  for (link = dm->ltoghook; link; link = link->next) {
    if (link->endhook) PetscCall((*link->endhook)(dm, g, mode, l, link->ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMLocalToLocalBegin - Begins the process of mapping values from a local vector (that include ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly from values on other MPI ranks. Must be followed by `DMLocalToLocalEnd()`.

   Neighbor-wise Collective

   Input Parameters:
+  dm - the `DM` object
.  g - the original local vector
-  mode - one of `INSERT_VALUES` or `ADD_VALUES`

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLocalToLocalEnd()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateLocalVector()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMLocalToLocalEnd()`, `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`
@*/
PetscErrorCode DMLocalToLocalBegin(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(l, VEC_CLASSID, 4);
  PetscUseTypeMethod(dm, localtolocalbegin, g, mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode), l);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMLocalToLocalEnd - Maps from a local vector to another local vector where the ghost
   points in the second are set correctly. Must be preceded by `DMLocalToLocalBegin()`.

   Neighbor-wise Collective

   Input Parameters:
+  da - the `DM` object
.  g - the original local vector
-  mode - one of `INSERT_VALUES` or `ADD_VALUES`

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLocalToLocalBegin()`, `DMCoarsen()`, `DMDestroy()`, `DMView()`, `DMCreateLocalVector()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMLocalToLocalBegin()`, `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`
@*/
PetscErrorCode DMLocalToLocalEnd(DM dm, Vec g, InsertMode mode, Vec l)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(l, VEC_CLASSID, 4);
  PetscUseTypeMethod(dm, localtolocalend, g, mode == INSERT_ALL_VALUES ? INSERT_VALUES : (mode == ADD_ALL_VALUES ? ADD_VALUES : mode), l);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMCoarsen - Coarsens a `DM` object using a standard, non-adaptive coarsening of the underlying mesh

    Collective

    Input Parameters:
+   dm - the `DM` object
-   comm - the communicator to contain the new `DM` object (or `MPI_COMM_NULL`)

    Output Parameter:
.   dmc - the coarsened `DM`

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMRefine()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMCoarsen(DM dm, MPI_Comm comm, DM *dmc)
{
  DMCoarsenHookLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscLogEventBegin(DM_Coarsen, dm, 0, 0, 0));
  PetscUseTypeMethod(dm, coarsen, comm, dmc);
  if (*dmc) {
    (*dmc)->bind_below = dm->bind_below; /* Propagate this from parent DM; otherwise -dm_bind_below will be useless for multigrid cases. */
    PetscCall(DMSetCoarseDM(dm, *dmc));
    (*dmc)->ops->creatematrix = dm->ops->creatematrix;
    PetscCall(PetscObjectCopyFortranFunctionPointers((PetscObject)dm, (PetscObject)*dmc));
    (*dmc)->ctx       = dm->ctx;
    (*dmc)->levelup   = dm->levelup;
    (*dmc)->leveldown = dm->leveldown + 1;
    PetscCall(DMSetMatType(*dmc, dm->mattype));
    for (link = dm->coarsenhook; link; link = link->next) {
      if (link->coarsenhook) PetscCall((*link->coarsenhook)(dm, *dmc, link->ctx));
    }
  }
  PetscCall(PetscLogEventEnd(DM_Coarsen, dm, 0, 0, 0));
  PetscCheck(*dmc, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "NULL coarse mesh produced");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMCoarsenHookAdd - adds a callback to be run when restricting a nonlinear problem to the coarse grid

   Logically Collective; No Fortran Support

   Input Parameters:
+  fine - `DM` on which to run a hook when restricting to a coarser level
.  coarsenhook - function to run when setting up a coarser level
.  restricthook - function to run to update data on coarser levels (called once per `SNESSolve()`)
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Calling sequence of `coarsenhook`:
$  PetscErrorCode  coarsenhook(DM fine, DM coarse, void *ctx);
+  fine - fine level `DM`
.  coarse - coarse level `DM` to restrict problem to
-  ctx - optional user-defined function context

   Calling sequence of `restricthook`:
$  PetscErrorCode  restricthook(DM fine, Mat mrestrict, Vec rscale, Mat inject, DM coarse, void *ctx)
+  fine - fine level `DM`
.  mrestrict - matrix restricting a fine-level solution to the coarse grid, usually the transpose of the interpolation
.  rscale - scaling vector for restriction
.  inject - matrix restricting by injection
.  coarse - coarse level DM to update
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data, attached to the `DM` with `PetscObjectCompose()`, needs to be set up or passed from the fine `DM` to the coarse `DM`.

   If this function is called multiple times, the hooks will be run in the order they are added.

   In order to compose with nonlinear preconditioning without duplicating storage, the hook should be implemented to
   extract the finest level information from its context (instead of from the `SNES`).

   The hooks are automatically called by `DMRestrict()`

.seealso: [](chapter_dmbase), `DM`, `DMCoarsenHookRemove()`, `DMRefineHookAdd()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMCoarsenHookAdd(DM fine, PetscErrorCode (*coarsenhook)(DM, DM, void *), PetscErrorCode (*restricthook)(DM, Mat, Vec, Mat, DM, void *), void *ctx)
{
  DMCoarsenHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine, DM_CLASSID, 1);
  for (p = &fine->coarsenhook; *p; p = &(*p)->next) { /* Scan to the end of the current list of hooks */
    if ((*p)->coarsenhook == coarsenhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscNew(&link));
  link->coarsenhook  = coarsenhook;
  link->restricthook = restricthook;
  link->ctx          = ctx;
  link->next         = NULL;
  *p                 = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMCoarsenHookRemove - remove a callback set with `DMCoarsenHookAdd()`

   Logically Collective; No Fortran Support

   Input Parameters:
+  fine - `DM` on which to run a hook when restricting to a coarser level
.  coarsenhook - function to run when setting up a coarser level
.  restricthook - function to run to update data on coarser levels
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Level: advanced

   Note:
   This function does nothing if the hook is not in the list.

.seealso: [](chapter_dmbase), `DM`, `DMCoarsenHookAdd()`, `DMRefineHookAdd()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMCoarsenHookRemove(DM fine, PetscErrorCode (*coarsenhook)(DM, DM, void *), PetscErrorCode (*restricthook)(DM, Mat, Vec, Mat, DM, void *), void *ctx)
{
  DMCoarsenHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fine, DM_CLASSID, 1);
  for (p = &fine->coarsenhook; *p; p = &(*p)->next) { /* Search the list of current hooks */
    if ((*p)->coarsenhook == coarsenhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) {
      link = *p;
      *p   = link->next;
      PetscCall(PetscFree(link));
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMRestrict - restricts user-defined problem data to a coarser `DM` by running hooks registered by `DMCoarsenHookAdd()`

   Collective if any hooks are

   Input Parameters:
+  fine - finer `DM` from which the data is obtained
.  restrct - restriction matrix, apply using `MatRestrict()`, usually the transpose of the interpolation
.  rscale - scaling vector for restriction
.  inject - injection matrix, also use `MatRestrict()`
-  coarse - coarser `DM` to update

   Level: developer

   Developer Note:
   Though this routine is called `DMRestrict()` the hooks are added with `DMCoarsenHookAdd()`, a consistent terminology would be better

.seealso: [](chapter_dmbase), `DM`, `DMCoarsenHookAdd()`, `MatRestrict()`, `DMInterpolate()`, `DMRefineHookAdd()`
@*/
PetscErrorCode DMRestrict(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse)
{
  DMCoarsenHookLink link;

  PetscFunctionBegin;
  for (link = fine->coarsenhook; link; link = link->next) {
    if (link->restricthook) PetscCall((*link->restricthook)(fine, restrct, rscale, inject, coarse, link->ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMSubDomainHookAdd - adds a callback to be run when restricting a problem to the coarse grid

   Logically Collective; No Fortran Support

   Input Parameters:
+  global - global `DM`
.  ddhook - function to run to pass data to the decomposition `DM` upon its creation
.  restricthook - function to run to update data on block solve (at the beginning of the block solve)
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Calling sequence of `ddhook`:
$  PetscErrorCode ddhook(DM global, DM block, void *ctx)
+  global - global `DM`
.  block  - block `DM`
-  ctx - optional user-defined function context

   Calling sequence of `restricthook`:
$  PetscErrorCode restricthook(DM global, VecScatter out, VecScatter in, DM block, void *ctx)
+  global - global `DM`
.  out    - scatter to the outer (with ghost and overlap points) block vector
.  in     - scatter to block vector values only owned locally
.  block  - block `DM`
-  ctx - optional user-defined function context

   Level: advanced

   Notes:
   This function is only needed if auxiliary data needs to be set up on subdomain `DM`s.

   If this function is called multiple times, the hooks will be run in the order they are added.

   In order to compose with nonlinear preconditioning without duplicating storage, the hook should be implemented to
   extract the global information from its context (instead of from the `SNES`).

.seealso: [](chapter_dmbase), `DM`, `DMSubDomainHookRemove()`, `DMRefineHookAdd()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMSubDomainHookAdd(DM global, PetscErrorCode (*ddhook)(DM, DM, void *), PetscErrorCode (*restricthook)(DM, VecScatter, VecScatter, DM, void *), void *ctx)
{
  DMSubDomainHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(global, DM_CLASSID, 1);
  for (p = &global->subdomainhook; *p; p = &(*p)->next) { /* Scan to the end of the current list of hooks */
    if ((*p)->ddhook == ddhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscNew(&link));
  link->restricthook = restricthook;
  link->ddhook       = ddhook;
  link->ctx          = ctx;
  link->next         = NULL;
  *p                 = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMSubDomainHookRemove - remove a callback from the list to be run when restricting a problem to the coarse grid

   Logically Collective; No Fortran Support

   Input Parameters:
+  global - global `DM`
.  ddhook - function to run to pass data to the decomposition `DM` upon its creation
.  restricthook - function to run to update data on block solve (at the beginning of the block solve)
-  ctx - [optional] user-defined context for provide data for the hooks (may be `NULL`)

   Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMSubDomainHookAdd()`, `SNESFASGetInterpolation()`, `SNESFASGetInjection()`, `PetscObjectCompose()`, `PetscContainerCreate()`
@*/
PetscErrorCode DMSubDomainHookRemove(DM global, PetscErrorCode (*ddhook)(DM, DM, void *), PetscErrorCode (*restricthook)(DM, VecScatter, VecScatter, DM, void *), void *ctx)
{
  DMSubDomainHookLink link, *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(global, DM_CLASSID, 1);
  for (p = &global->subdomainhook; *p; p = &(*p)->next) { /* Search the list of current hooks */
    if ((*p)->ddhook == ddhook && (*p)->restricthook == restricthook && (*p)->ctx == ctx) {
      link = *p;
      *p   = link->next;
      PetscCall(PetscFree(link));
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMSubDomainRestrict - restricts user-defined problem data to a block `DM` by running hooks registered by `DMSubDomainHookAdd()`

   Collective if any hooks are

   Input Parameters:
+  fine - finer `DM` to use as a base
.  oscatter - scatter from domain global vector filling subdomain global vector with overlap
.  gscatter - scatter from domain global vector filling subdomain local vector with ghosts
-  coarse - coarser `DM` to update

   Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCoarsenHookAdd()`, `MatRestrict()`
@*/
PetscErrorCode DMSubDomainRestrict(DM global, VecScatter oscatter, VecScatter gscatter, DM subdm)
{
  DMSubDomainHookLink link;

  PetscFunctionBegin;
  for (link = global->subdomainhook; link; link = link->next) {
    if (link->restricthook) PetscCall((*link->restricthook)(global, oscatter, gscatter, subdm, link->ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGetCoarsenLevel - Gets the number of coarsenings that have generated this `DM`.

    Not Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   level - number of coarsenings

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCoarsen()`, `DMSetCoarsenLevel()`, `DMGetRefineLevel()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMGetCoarsenLevel(DM dm, PetscInt *level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(level, 2);
  *level = dm->leveldown;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMSetCoarsenLevel - Sets the number of coarsenings that have generated this `DM`.

    Collective

    Input Parameters:
+   dm - the `DM` object
-   level - number of coarsenings

    Level: developer

    Note:
    This is rarely used directly, the information is automatically set when a `DM` is created with `DMCoarsen()`

.seealso: [](chapter_dmbase), `DM`, `DMSetCoarsenLevel()`, `DMCoarsen()`, `DMGetCoarsenLevel()`, `DMGetRefineLevel()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMSetCoarsenLevel(DM dm, PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->leveldown = level;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMRefineHierarchy - Refines a `DM` object, all levels at once

    Collective

    Input Parameters:
+   dm - the `DM` object
-   nlevels - the number of levels of refinement

    Output Parameter:
.   dmf - the refined `DM` hierarchy

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCoarsen()`, `DMCoarsenHierarchy()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMRefineHierarchy(DM dm, PetscInt nlevels, DM dmf[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(nlevels >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidPointer(dmf, 3);
  if (dm->ops->refinehierarchy) {
    PetscUseTypeMethod(dm, refinehierarchy, nlevels, dmf);
  } else if (dm->ops->refine) {
    PetscInt i;

    PetscCall(DMRefine(dm, PetscObjectComm((PetscObject)dm), &dmf[0]));
    for (i = 1; i < nlevels; i++) PetscCall(DMRefine(dmf[i - 1], PetscObjectComm((PetscObject)dm), &dmf[i]));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No RefineHierarchy for this DM yet");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMCoarsenHierarchy - Coarsens a `DM` object, all levels at once

    Collective

    Input Parameters:
+   dm - the `DM` object
-   nlevels - the number of levels of coarsening

    Output Parameter:
.   dmc - the coarsened `DM` hierarchy

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCoarsen()`, `DMRefineHierarchy()`, `DMDestroy()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`
@*/
PetscErrorCode DMCoarsenHierarchy(DM dm, PetscInt nlevels, DM dmc[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(nlevels >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "nlevels cannot be negative");
  if (nlevels == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidPointer(dmc, 3);
  if (dm->ops->coarsenhierarchy) {
    PetscUseTypeMethod(dm, coarsenhierarchy, nlevels, dmc);
  } else if (dm->ops->coarsen) {
    PetscInt i;

    PetscCall(DMCoarsen(dm, PetscObjectComm((PetscObject)dm), &dmc[0]));
    for (i = 1; i < nlevels; i++) PetscCall(DMCoarsen(dmc[i - 1], PetscObjectComm((PetscObject)dm), &dmc[i]));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No CoarsenHierarchy for this DM yet");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMSetApplicationContextDestroy - Sets a user function that will be called to destroy the application context when the `DM` is destroyed

    Logically Collective if the function is collective

    Input Parameters:
+   dm - the `DM` object
-   destroy - the destroy function

    Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMSetApplicationContext()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMGetApplicationContext()`
@*/
PetscErrorCode DMSetApplicationContextDestroy(DM dm, PetscErrorCode (*destroy)(void **))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->ctxdestroy = destroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMSetApplicationContext - Set a user context into a `DM` object

    Not Collective

    Input Parameters:
+   dm - the `DM` object
-   ctx - the user context

    Level: intermediate

    Note:
    A user context is a way to pass problem specific information that is accessible whenever the `DM` is available

.seealso: [](chapter_dmbase), `DM`, `DMGetApplicationContext()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMGetApplicationContext()`
@*/
PetscErrorCode DMSetApplicationContext(DM dm, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGetApplicationContext - Gets a user context from a `DM` object

    Not Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   ctx - the user context

    Level: intermediate

    Note:
    A user context is a way to pass problem specific information that is accessible whenever the `DM` is available

.seealso: [](chapter_dmbase), `DM`, `DMGetApplicationContext()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMGetApplicationContext()`
@*/
PetscErrorCode DMGetApplicationContext(DM dm, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *(void **)ctx = dm->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMSetVariableBounds - sets a function to compute the lower and upper bound vectors for `SNESVI`.

    Logically Collective

    Input Parameters:
+   dm - the DM object
-   f - the function that computes variable bounds used by SNESVI (use `NULL` to cancel a previous function that was set)

    Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMComputeVariableBounds()`, `DMHasVariableBounds()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMGetApplicationContext()`,
         `DMSetJacobian()`
@*/
PetscErrorCode DMSetVariableBounds(DM dm, PetscErrorCode (*f)(DM, Vec, Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->ops->computevariablebounds = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMHasVariableBounds - does the `DM` object have a variable bounds function?

    Not Collective

    Input Parameter:
.   dm - the `DM` object to destroy

    Output Parameter:
.   flg - `PETSC_TRUE` if the variable bounds function exists

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMComputeVariableBounds()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMGetApplicationContext()`
@*/
PetscErrorCode DMHasVariableBounds(DM dm, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  *flg = (dm->ops->computevariablebounds) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    DMComputeVariableBounds - compute variable bounds used by `SNESVI`.

    Logically Collective

    Input Parameter:
.   dm - the `DM` object

    Output parameters:
+   xl - lower bound
-   xu - upper bound

    Level: advanced

    Note:
    This is generally not called by users. It calls the function provided by the user with DMSetVariableBounds()

.seealso: [](chapter_dmbase), `DM`, `DMHasVariableBounds()`, `DMView()`, `DMCreateGlobalVector()`, `DMCreateInterpolation()`, `DMCreateColoring()`, `DMCreateMatrix()`, `DMCreateMassMatrix()`, `DMGetApplicationContext()`
@*/
PetscErrorCode DMComputeVariableBounds(DM dm, Vec xl, Vec xu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(xl, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(xu, VEC_CLASSID, 3);
  PetscUseTypeMethod(dm, computevariablebounds, xl, xu);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMHasColoring - does the `DM` object have a method of providing a coloring?

    Not Collective

    Input Parameter:
.   dm - the DM object

    Output Parameter:
.   flg - `PETSC_TRUE` if the `DM` has facilities for `DMCreateColoring()`.

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreateColoring()`
@*/
PetscErrorCode DMHasColoring(DM dm, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  *flg = (dm->ops->getcoloring) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMHasCreateRestriction - does the `DM` object have a method of providing a restriction?

    Not Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   flg - `PETSC_TRUE` if the `DM` has facilities for `DMCreateRestriction()`.

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreateRestriction()`, `DMHasCreateInterpolation()`, `DMHasCreateInjection()`
@*/
PetscErrorCode DMHasCreateRestriction(DM dm, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  *flg = (dm->ops->createrestriction) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMHasCreateInjection - does the `DM` object have a method of providing an injection?

    Not Collective

    Input Parameter:
.   dm - the `DM` object

    Output Parameter:
.   flg - `PETSC_TRUE` if the `DM` has facilities for `DMCreateInjection()`.

    Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMCreateInjection()`, `DMHasCreateRestriction()`, `DMHasCreateInterpolation()`
@*/
PetscErrorCode DMHasCreateInjection(DM dm, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  if (dm->ops->hascreateinjection) PetscUseTypeMethod(dm, hascreateinjection, flg);
  else *flg = (dm->ops->createinjection) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscFunctionList DMList              = NULL;
PetscBool         DMRegisterAllCalled = PETSC_FALSE;

/*@C
  DMSetType - Builds a `DM`, for a particular `DM` implementation.

  Collective

  Input Parameters:
+ dm     - The `DM` object
- method - The name of the `DMType`, for example `DMDA`, `DMPLEX`

  Options Database Key:
. -dm_type <type> - Sets the `DM` type; use -help for a list of available types

  Level: intermediate

  Note:
  Of the `DM` is constructed by directly calling a function to construct a particular `DM`, for example, `DMDACreate2d()` or `DMPlexCreateBoxMesh()`

.seealso: [](chapter_dmbase), `DM`, `DMType`, `DMDA`, `DMPLEX`, `DMGetType()`, `DMCreate()`, `DMDACreate2d()`
@*/
PetscErrorCode DMSetType(DM dm, DMType method)
{
  PetscErrorCode (*r)(DM);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, method, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMRegisterAll());
  PetscCall(PetscFunctionListFind(DMList, method, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DM type: %s", method);

  PetscTryTypeMethod(dm, destroy);
  PetscCall(PetscMemzero(dm->ops, sizeof(*dm->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)dm, method));
  PetscCall((*r)(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetType - Gets the `DM` type name (as a string) from the `DM`.

  Not Collective

  Input Parameter:
. dm  - The `DM`

  Output Parameter:
. type - The `DMType` name

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMType`, `DMDA`, `DMPLEX`, `DMSetType()`, `DMCreate()`
@*/
PetscErrorCode DMGetType(DM dm, DMType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(type, 2);
  PetscCall(DMRegisterAll());
  *type = ((PetscObject)dm)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMConvert - Converts a `DM` to another `DM`, either of the same or different type.

  Collective

  Input Parameters:
+ dm - the `DM`
- newtype - new `DM` type (use "same" for the same type)

  Output Parameter:
. M - pointer to new `DM`

  Level: intermediate

  Notes:
  Cannot be used to convert a sequential `DM` to a parallel or a parallel to sequential,
  the MPI communicator of the generated `DM` is always the same as the communicator
  of the input `DM`.

.seealso: [](chapter_dmbase), `DM`, `DMSetType()`, `DMCreate()`, `DMClone()`
@*/
PetscErrorCode DMConvert(DM dm, DMType newtype, DM *M)
{
  DM        B;
  char      convname[256];
  PetscBool sametype /*, issame */;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidType(dm, 1);
  PetscValidPointer(M, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, newtype, &sametype));
  /* PetscCall(PetscStrcmp(newtype, "same", &issame)); */
  if (sametype) {
    *M = dm;
    PetscCall(PetscObjectReference((PetscObject)dm));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    PetscErrorCode (*conv)(DM, DMType, DM *) = NULL;

    /*
       Order of precedence:
       1) See if a specialized converter is known to the current DM.
       2) See if a specialized converter is known to the desired DM class.
       3) See if a good general converter is registered for the desired class
       4) See if a good general converter is known for the current matrix.
       5) Use a really basic converter.
    */

    /* 1) See if a specialized converter is known to the current DM and the desired class */
    PetscCall(PetscStrncpy(convname, "DMConvert_", sizeof(convname)));
    PetscCall(PetscStrlcat(convname, ((PetscObject)dm)->type_name, sizeof(convname)));
    PetscCall(PetscStrlcat(convname, "_", sizeof(convname)));
    PetscCall(PetscStrlcat(convname, newtype, sizeof(convname)));
    PetscCall(PetscStrlcat(convname, "_C", sizeof(convname)));
    PetscCall(PetscObjectQueryFunction((PetscObject)dm, convname, &conv));
    if (conv) goto foundconv;

    /* 2)  See if a specialized converter is known to the desired DM class. */
    PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), &B));
    PetscCall(DMSetType(B, newtype));
    PetscCall(PetscStrncpy(convname, "DMConvert_", sizeof(convname)));
    PetscCall(PetscStrlcat(convname, ((PetscObject)dm)->type_name, sizeof(convname)));
    PetscCall(PetscStrlcat(convname, "_", sizeof(convname)));
    PetscCall(PetscStrlcat(convname, newtype, sizeof(convname)));
    PetscCall(PetscStrlcat(convname, "_C", sizeof(convname)));
    PetscCall(PetscObjectQueryFunction((PetscObject)B, convname, &conv));
    if (conv) {
      PetscCall(DMDestroy(&B));
      goto foundconv;
    }

#if 0
    /* 3) See if a good general converter is registered for the desired class */
    conv = B->ops->convertfrom;
    PetscCall(DMDestroy(&B));
    if (conv) goto foundconv;

    /* 4) See if a good general converter is known for the current matrix */
    if (dm->ops->convert) {
      conv = dm->ops->convert;
    }
    if (conv) goto foundconv;
#endif

    /* 5) Use a really basic converter. */
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No conversion possible between DM types %s and %s", ((PetscObject)dm)->type_name, newtype);

  foundconv:
    PetscCall(PetscLogEventBegin(DM_Convert, dm, 0, 0, 0));
    PetscCall((*conv)(dm, newtype, M));
    /* Things that are independent of DM type: We should consult DMClone() here */
    {
      const PetscReal *maxCell, *Lstart, *L;

      PetscCall(DMGetPeriodicity(dm, &maxCell, &Lstart, &L));
      PetscCall(DMSetPeriodicity(*M, maxCell, Lstart, L));
      (*M)->prealloc_only = dm->prealloc_only;
      PetscCall(PetscFree((*M)->vectype));
      PetscCall(PetscStrallocpy(dm->vectype, (char **)&(*M)->vectype));
      PetscCall(PetscFree((*M)->mattype));
      PetscCall(PetscStrallocpy(dm->mattype, (char **)&(*M)->mattype));
    }
    PetscCall(PetscLogEventEnd(DM_Convert, dm, 0, 0, 0));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)*M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  DMRegister -  Adds a new `DM` type implementation

  Not Collective

  Input Parameters:
+ sname    - The name of a new user-defined creation routine
- function - The creation routine itself

  Level: advanced

  Notes:
  `DMRegister()` may be called multiple times to add several user-defined `DM`s

  Sample usage:
.vb
    DMRegister("my_da", MyDMCreate);
.ve

  Then, your `DM` type can be chosen with the procedural interface via
.vb
    DMCreate(MPI_Comm, DM *);
    DMSetType(DM,"my_da");
.ve
   or at runtime via the option
.vb
    -da_type my_da
.ve

.seealso: [](chapter_dmbase), `DM`, `DMType`, `DMSetType()`, `DMRegisterAll()`, `DMRegisterDestroy()`
@*/
PetscErrorCode DMRegister(const char sname[], PetscErrorCode (*function)(DM))
{
  PetscFunctionBegin;
  PetscCall(DMInitializePackage());
  PetscCall(PetscFunctionListAdd(&DMList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMLoad - Loads a DM that has been stored in binary  with `DMView()`.

  Collective

  Input Parameters:
+ newdm - the newly loaded `DM`, this needs to have been created with `DMCreate()` or
           some related function before a call to `DMLoad()`.
- viewer - binary file viewer, obtained from `PetscViewerBinaryOpen()` or
           `PETSCVIEWERHDF5` file viewer, obtained from `PetscViewerHDF5Open()`

   Level: intermediate

  Notes:
  The type is determined by the data in the file, any type set into the DM before this call is ignored.

  Using `PETSCVIEWERHDF5` type with `PETSC_VIEWER_HDF5_PETSC` format, one can save multiple `DMPLEX`
  meshes in a single HDF5 file. This in turn requires one to name the `DMPLEX` object with `PetscObjectSetName()`
  before saving it with `DMView()` and before loading it with `DMLoad()` for identification of the mesh object.

.seealso: [](chapter_dmbase), `DM`, `PetscViewerBinaryOpen()`, `DMView()`, `MatLoad()`, `VecLoad()`
@*/
PetscErrorCode DMLoad(DM newdm, PetscViewer viewer)
{
  PetscBool isbinary, ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newdm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscViewerCheckReadable(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  PetscCall(PetscLogEventBegin(DM_Load, viewer, 0, 0, 0));
  if (isbinary) {
    PetscInt classid;
    char     type[256];

    PetscCall(PetscViewerBinaryRead(viewer, &classid, 1, NULL, PETSC_INT));
    PetscCheck(classid == DM_FILE_CLASSID, PetscObjectComm((PetscObject)newdm), PETSC_ERR_ARG_WRONG, "Not DM next in file, classid found %d", (int)classid);
    PetscCall(PetscViewerBinaryRead(viewer, type, 256, NULL, PETSC_CHAR));
    PetscCall(DMSetType(newdm, type));
    PetscTryTypeMethod(newdm, load, viewer);
  } else if (ishdf5) {
    PetscTryTypeMethod(newdm, load, viewer);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerBinaryOpen() or PetscViewerHDF5Open()");
  PetscCall(PetscLogEventEnd(DM_Load, viewer, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/******************************** FEM Support **********************************/

PetscErrorCode DMPrintCellVector(PetscInt c, const char name[], PetscInt len, const PetscScalar x[])
{
  PetscInt f;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT " Element %s\n", c, name));
  for (f = 0; f < len; ++f) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  | %g |\n", (double)PetscRealPart(x[f])));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPrintCellMatrix(PetscInt c, const char name[], PetscInt rows, PetscInt cols, const PetscScalar A[])
{
  PetscInt f, g;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT " Element %s\n", c, name));
  for (f = 0; f < rows; ++f) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  |"));
    for (g = 0; g < cols; ++g) PetscCall(PetscPrintf(PETSC_COMM_SELF, " % 9.5g", (double)PetscRealPart(A[f * cols + g])));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " |\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPrintLocalVec(DM dm, const char name[], PetscReal tol, Vec X)
{
  PetscInt           localSize, bs;
  PetscMPIInt        size;
  Vec                x, xglob;
  const PetscScalar *xarray;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(VecDuplicate(X, &x));
  PetscCall(VecCopy(X, x));
  PetscCall(VecChop(x, tol));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s:\n", name));
  if (size > 1) {
    PetscCall(VecGetLocalSize(x, &localSize));
    PetscCall(VecGetArrayRead(x, &xarray));
    PetscCall(VecGetBlockSize(x, &bs));
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm), bs, localSize, PETSC_DETERMINE, xarray, &xglob));
  } else {
    xglob = x;
  }
  PetscCall(VecView(xglob, PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm))));
  if (size > 1) {
    PetscCall(VecDestroy(&xglob));
    PetscCall(VecRestoreArrayRead(x, &xarray));
  }
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetSection - Get the `PetscSection` encoding the local data layout for the `DM`.   This is equivalent to `DMGetLocalSection()`. Deprecated in v3.12

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. section - The `PetscSection`

  Options Database Key:
. -dm_petscsection_view - View the `PetscSection` created by the `DM`

  Level: advanced

  Notes:
  Use `DMGetLocalSection()` in new code.

  This gets a borrowed reference, so the user should not destroy this `PetscSection`.

.seealso: [](chapter_dmbase), `DM`, `DMGetLocalSection()`, `DMSetLocalSection()`, `DMGetGlobalSection()`
@*/
PetscErrorCode DMGetSection(DM dm, PetscSection *section)
{
  PetscFunctionBegin;
  PetscCall(DMGetLocalSection(dm, section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetLocalSection - Get the `PetscSection` encoding the local data layout for the `DM`.

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. section - The `PetscSection`

  Options Database Key:
. -dm_petscsection_view - View the section created by the `DM`

  Level: intermediate

  Note:
  This gets a borrowed reference, so the user should not destroy this `PetscSection`.

.seealso: [](chapter_dmbase), `DM`, `DMSetLocalSection()`, `DMGetGlobalSection()`
@*/
PetscErrorCode DMGetLocalSection(DM dm, PetscSection *section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (!dm->localSection && dm->ops->createlocalsection) {
    PetscInt d;

    if (dm->setfromoptionscalled) {
      PetscObject       obj = (PetscObject)dm;
      PetscViewer       viewer;
      PetscViewerFormat format;
      PetscBool         flg;

      PetscCall(PetscOptionsGetViewer(PetscObjectComm(obj), obj->options, obj->prefix, "-dm_petscds_view", &viewer, &format, &flg));
      if (flg) PetscCall(PetscViewerPushFormat(viewer, format));
      for (d = 0; d < dm->Nds; ++d) {
        PetscCall(PetscDSSetFromOptions(dm->probs[d].ds));
        if (flg) PetscCall(PetscDSView(dm->probs[d].ds, viewer));
      }
      if (flg) {
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerPopFormat(viewer));
        PetscCall(PetscViewerDestroy(&viewer));
      }
    }
    PetscUseTypeMethod(dm, createlocalsection);
    if (dm->localSection) PetscCall(PetscObjectViewFromOptions((PetscObject)dm->localSection, NULL, "-dm_petscsection_view"));
  }
  *section = dm->localSection;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetSection - Set the `PetscSection` encoding the local data layout for the `DM`.  This is equivalent to `DMSetLocalSection()`. Deprecated in v3.12

  Input Parameters:
+ dm - The `DM`
- section - The `PetscSection`

  Level: advanced

  Notes:
  Use `DMSetLocalSection()` in new code.

  Any existing `PetscSection` will be destroyed

.seealso: [](chapter_dmbase), `DM`, `DMSetLocalSection()`, `DMGetLocalSection()`, `DMSetGlobalSection()`
@*/
PetscErrorCode DMSetSection(DM dm, PetscSection section)
{
  PetscFunctionBegin;
  PetscCall(DMSetLocalSection(dm, section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetLocalSection - Set the `PetscSection` encoding the local data layout for the `DM`.

  Input Parameters:
+ dm - The `DM`
- section - The `PetscSection`

  Level: intermediate

  Note:
  Any existing Section will be destroyed

.seealso: [](chapter_dmbase), `DM`, `PetscSection`, `DMGetLocalSection()`, `DMSetGlobalSection()`
@*/
PetscErrorCode DMSetLocalSection(DM dm, PetscSection section)
{
  PetscInt numFields = 0;
  PetscInt f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)section));
  PetscCall(PetscSectionDestroy(&dm->localSection));
  dm->localSection = section;
  if (section) PetscCall(PetscSectionGetNumFields(dm->localSection, &numFields));
  if (numFields) {
    PetscCall(DMSetNumFields(dm, numFields));
    for (f = 0; f < numFields; ++f) {
      PetscObject disc;
      const char *name;

      PetscCall(PetscSectionGetFieldName(dm->localSection, f, &name));
      PetscCall(DMGetField(dm, f, NULL, &disc));
      PetscCall(PetscObjectSetName(disc, name));
    }
  }
  /* The global section will be rebuilt in the next call to DMGetGlobalSection(). */
  PetscCall(PetscSectionDestroy(&dm->globalSection));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetDefaultConstraints - Get the `PetscSection` and `Mat` that specify the local constraint interpolation. See `DMSetDefaultConstraints()` for a description of the purpose of constraint interpolation.

  not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameters:
+ section - The `PetscSection` describing the range of the constraint matrix: relates rows of the constraint matrix to dofs of the default section.  Returns `NULL` if there are no local constraints.
. mat - The `Mat` that interpolates local constraints: its width should be the layout size of the default section.  Returns `NULL` if there are no local constraints.
- bias - Vector containing bias to be added to constrained dofs

  Level: advanced

  Note:
  This gets borrowed references, so the user should not destroy the `PetscSection`, `Mat`, or `Vec`.

.seealso: [](chapter_dmbase), `DM`, `DMSetDefaultConstraints()`
@*/
PetscErrorCode DMGetDefaultConstraints(DM dm, PetscSection *section, Mat *mat, Vec *bias)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->defaultConstraint.section && !dm->defaultConstraint.mat && dm->ops->createdefaultconstraints) PetscUseTypeMethod(dm, createdefaultconstraints);
  if (section) *section = dm->defaultConstraint.section;
  if (mat) *mat = dm->defaultConstraint.mat;
  if (bias) *bias = dm->defaultConstraint.bias;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetDefaultConstraints - Set the `PetscSection` and `Mat` that specify the local constraint interpolation.

  Collective

  Input Parameters:
+ dm - The `DM`
. section - The `PetscSection` describing the range of the constraint matrix: relates rows of the constraint matrix to dofs of the default section.  Must have a local communicator (`PETSC_COMM_SELF` or derivative).
. mat - The `Mat` that interpolates local constraints: its width should be the layout size of the default section:  `NULL` indicates no constraints.  Must have a local communicator (`PETSC_COMM_SELF` or derivative).
- bias - A bias vector to be added to constrained values in the local vector.  `NULL` indicates no bias.  Must have a local communicator (`PETSC_COMM_SELF` or derivative).

  Level: advanced

  Notes:
  If a constraint matrix is specified, then it is applied during `DMGlobalToLocalEnd()` when mode is `INSERT_VALUES`, `INSERT_BC_VALUES`, or `INSERT_ALL_VALUES`.  Without a constraint matrix, the local vector l returned by `DMGlobalToLocalEnd()` contains values that have been scattered from a global vector without modification; with a constraint matrix A, l is modified by computing c = A * l + bias, l[s[i]] = c[i], where the scatter s is defined by the `PetscSection` returned by `DMGetDefaultConstraints()`.

  If a constraint matrix is specified, then its adjoint is applied during `DMLocalToGlobalBegin()` when mode is `ADD_VALUES`, `ADD_BC_VALUES`, or `ADD_ALL_VALUES`.  Without a constraint matrix, the local vector l is accumulated into a global vector without modification; with a constraint matrix A, l is first modified by computing c[i] = l[s[i]], l[s[i]] = 0, l = l + A'*c, which is the adjoint of the operation described above.  Any bias, if specified, is ignored when accumulating.

  This increments the references of the `PetscSection`, `Mat`, and `Vec`, so they user can destroy them.

.seealso: [](chapter_dmbase), `DM`, `DMGetDefaultConstraints()`
@*/
PetscErrorCode DMSetDefaultConstraints(DM dm, PetscSection section, Mat mat, Vec bias)
{
  PetscMPIInt result;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) {
    PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
    PetscCallMPI(MPI_Comm_compare(PETSC_COMM_SELF, PetscObjectComm((PetscObject)section), &result));
    PetscCheck(result == MPI_CONGRUENT || result == MPI_IDENT, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMECOMM, "constraint section must have local communicator");
  }
  if (mat) {
    PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
    PetscCallMPI(MPI_Comm_compare(PETSC_COMM_SELF, PetscObjectComm((PetscObject)mat), &result));
    PetscCheck(result == MPI_CONGRUENT || result == MPI_IDENT, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMECOMM, "constraint matrix must have local communicator");
  }
  if (bias) {
    PetscValidHeaderSpecific(bias, VEC_CLASSID, 4);
    PetscCallMPI(MPI_Comm_compare(PETSC_COMM_SELF, PetscObjectComm((PetscObject)bias), &result));
    PetscCheck(result == MPI_CONGRUENT || result == MPI_IDENT, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMECOMM, "constraint bias must have local communicator");
  }
  PetscCall(PetscObjectReference((PetscObject)section));
  PetscCall(PetscSectionDestroy(&dm->defaultConstraint.section));
  dm->defaultConstraint.section = section;
  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&dm->defaultConstraint.mat));
  dm->defaultConstraint.mat = mat;
  PetscCall(PetscObjectReference((PetscObject)bias));
  PetscCall(VecDestroy(&dm->defaultConstraint.bias));
  dm->defaultConstraint.bias = bias;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_DEBUG)
/*
  DMDefaultSectionCheckConsistency - Check the consistentcy of the global and local sections. Generates and error if they are not consistent.

  Input Parameters:
+ dm - The `DM`
. localSection - `PetscSection` describing the local data layout
- globalSection - `PetscSection` describing the global data layout

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetSectionSF()`, `DMSetSectionSF()`
*/
static PetscErrorCode DMDefaultSectionCheckConsistency_Internal(DM dm, PetscSection localSection, PetscSection globalSection)
{
  MPI_Comm        comm;
  PetscLayout     layout;
  const PetscInt *ranges;
  PetscInt        pStart, pEnd, p, nroots;
  PetscMPIInt     size, rank;
  PetscBool       valid = PETSC_TRUE, gvalid;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSectionGetChart(globalSection, &pStart, &pEnd));
  PetscCall(PetscSectionGetConstrainedStorageSize(globalSection, &nroots));
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetBlockSize(layout, 1));
  PetscCall(PetscLayoutSetLocalSize(layout, nroots));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetRanges(layout, &ranges));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, gdof, gcdof, goff, gsize, d;

    PetscCall(PetscSectionGetDof(localSection, p, &dof));
    PetscCall(PetscSectionGetOffset(localSection, p, &off));
    PetscCall(PetscSectionGetConstraintDof(localSection, p, &cdof));
    PetscCall(PetscSectionGetDof(globalSection, p, &gdof));
    PetscCall(PetscSectionGetConstraintDof(globalSection, p, &gcdof));
    PetscCall(PetscSectionGetOffset(globalSection, p, &goff));
    if (!gdof) continue; /* Censored point */
    if ((gdof < 0 ? -(gdof + 1) : gdof) != dof) {
      PetscCall(PetscSynchronizedPrintf(comm, "[%d]Global dof %" PetscInt_FMT " for point %" PetscInt_FMT " not equal to local dof %" PetscInt_FMT "\n", rank, gdof, p, dof));
      valid = PETSC_FALSE;
    }
    if (gcdof && (gcdof != cdof)) {
      PetscCall(PetscSynchronizedPrintf(comm, "[%d]Global constraints %" PetscInt_FMT " for point %" PetscInt_FMT " not equal to local constraints %" PetscInt_FMT "\n", rank, gcdof, p, cdof));
      valid = PETSC_FALSE;
    }
    if (gdof < 0) {
      gsize = gdof < 0 ? -(gdof + 1) - gcdof : gdof - gcdof;
      for (d = 0; d < gsize; ++d) {
        PetscInt offset = -(goff + 1) + d, r;

        PetscCall(PetscFindInt(offset, size + 1, ranges, &r));
        if (r < 0) r = -(r + 2);
        if ((r < 0) || (r >= size)) {
          PetscCall(PetscSynchronizedPrintf(comm, "[%d]Point %" PetscInt_FMT " mapped to invalid process %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ")\n", rank, p, r, gdof, goff));
          valid = PETSC_FALSE;
          break;
        }
      }
    }
  }
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCall(PetscSynchronizedFlush(comm, NULL));
  PetscCall(MPIU_Allreduce(&valid, &gvalid, 1, MPIU_BOOL, MPI_LAND, comm));
  if (!gvalid) {
    PetscCall(DMView(dm, NULL));
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Inconsistent local and global sections");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode DMGetIsoperiodicPointSF_Internal(DM dm, PetscSF *sf)
{
  PetscErrorCode (*f)(DM, PetscSF *);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  PetscCall(PetscObjectQueryFunction((PetscObject)dm, "DMGetIsoperiodicPointSF_C", &f));
  if (f) PetscCall(f(dm, sf));
  else *sf = dm->sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetGlobalSection - Get the `PetscSection` encoding the global data layout for the `DM`.

  Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. section - The `PetscSection`

  Level: intermediate

  Note:
  This gets a borrowed reference, so the user should not destroy this `PetscSection`.

.seealso: [](chapter_dmbase), `DM`, `DMSetLocalSection()`, `DMGetLocalSection()`
@*/
PetscErrorCode DMGetGlobalSection(DM dm, PetscSection *section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  if (!dm->globalSection) {
    PetscSection s;
    PetscSF      sf;

    PetscCall(DMGetLocalSection(dm, &s));
    PetscCheck(s, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM must have a default PetscSection in order to create a global PetscSection");
    PetscCheck(dm->sf, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM must have a point PetscSF in order to create a global PetscSection");
    PetscCall(DMGetIsoperiodicPointSF_Internal(dm, &sf));
    PetscCall(PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &dm->globalSection));
    PetscCall(PetscLayoutDestroy(&dm->map));
    PetscCall(PetscSectionGetValueLayout(PetscObjectComm((PetscObject)dm), dm->globalSection, &dm->map));
    PetscCall(PetscSectionViewFromOptions(dm->globalSection, NULL, "-global_section_view"));
  }
  *section = dm->globalSection;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetGlobalSection - Set the `PetscSection` encoding the global data layout for the `DM`.

  Input Parameters:
+ dm - The `DM`
- section - The PetscSection, or `NULL`

  Level: intermediate

  Note:
  Any existing `PetscSection` will be destroyed

.seealso: [](chapter_dmbase), `DM`, `DMGetGlobalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMSetGlobalSection(DM dm, PetscSection section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)section));
  PetscCall(PetscSectionDestroy(&dm->globalSection));
  dm->globalSection = section;
#if defined(PETSC_USE_DEBUG)
  if (section) PetscCall(DMDefaultSectionCheckConsistency_Internal(dm, dm->localSection, section));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetSectionSF - Get the `PetscSF` encoding the parallel dof overlap for the `DM`. If it has not been set,
  it is created from the default `PetscSection` layouts in the `DM`.

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. sf - The `PetscSF`

  Level: intermediate

  Note:
  This gets a borrowed reference, so the user should not destroy this `PetscSF`.

.seealso: [](chapter_dmbase), `DM`, `DMSetSectionSF()`, `DMCreateSectionSF()`
@*/
PetscErrorCode DMGetSectionSF(DM dm, PetscSF *sf)
{
  PetscInt nroots;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  if (!dm->sectionSF) PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), &dm->sectionSF));
  PetscCall(PetscSFGetGraph(dm->sectionSF, &nroots, NULL, NULL, NULL));
  if (nroots < 0) {
    PetscSection section, gSection;

    PetscCall(DMGetLocalSection(dm, &section));
    if (section) {
      PetscCall(DMGetGlobalSection(dm, &gSection));
      PetscCall(DMCreateSectionSF(dm, section, gSection));
    } else {
      *sf = NULL;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  *sf = dm->sectionSF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetSectionSF - Set the `PetscSF` encoding the parallel dof overlap for the `DM`

  Input Parameters:
+ dm - The `DM`
- sf - The `PetscSF`

  Level: intermediate

  Note:
  Any previous `PetscSF` is destroyed

.seealso: [](chapter_dmbase), `DM`, `DMGetSectionSF()`, `DMCreateSectionSF()`
@*/
PetscErrorCode DMSetSectionSF(DM dm, PetscSF sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)sf));
  PetscCall(PetscSFDestroy(&dm->sectionSF));
  dm->sectionSF = sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateSectionSF - Create the `PetscSF` encoding the parallel dof overlap for the `DM` based upon the `PetscSection`s
  describing the data layout.

  Input Parameters:
+ dm - The `DM`
. localSection - `PetscSection` describing the local data layout
- globalSection - `PetscSection` describing the global data layout

  Level: developer

  Note:
  One usually uses `DMGetSectionSF()` to obtain the `PetscSF`

  Developer Note:
  Since this routine has for arguments the two sections from the `DM` and puts the resulting `PetscSF`
  directly into the `DM`, perhaps this function should not take the local and global sections as
  input and should just obtain them from the `DM`?

.seealso: [](chapter_dmbase), `DM`, `DMGetSectionSF()`, `DMSetSectionSF()`, `DMGetLocalSection()`, `DMGetGlobalSection()`
@*/
PetscErrorCode DMCreateSectionSF(DM dm, PetscSection localSection, PetscSection globalSection)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscSFSetGraphSection(dm->sectionSF, localSection, globalSection));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetPointSF - Get the `PetscSF` encoding the parallel section point overlap for the `DM`.

  Not collective but the resulting `PetscSF` is collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. sf - The `PetscSF`

  Level: intermediate

  Note:
  This gets a borrowed reference, so the user should not destroy this `PetscSF`.

.seealso: [](chapter_dmbase), `DM`, `DMSetPointSF()`, `DMGetSectionSF()`, `DMSetSectionSF()`, `DMCreateSectionSF()`
@*/
PetscErrorCode DMGetPointSF(DM dm, PetscSF *sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  *sf = dm->sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetPointSF - Set the `PetscSF` encoding the parallel section point overlap for the `DM`.

  Collective

  Input Parameters:
+ dm - The `DM`
- sf - The `PetscSF`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetPointSF()`, `DMGetSectionSF()`, `DMSetSectionSF()`, `DMCreateSectionSF()`
@*/
PetscErrorCode DMSetPointSF(DM dm, PetscSF sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)sf));
  PetscCall(PetscSFDestroy(&dm->sf));
  dm->sf = sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNaturalSF - Get the `PetscSF` encoding the map back to the original mesh ordering

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. sf - The `PetscSF`

  Level: intermediate

  Note:
  This gets a borrowed reference, so the user should not destroy this `PetscSF`.

.seealso: [](chapter_dmbase), `DM`, `DMSetNaturalSF()`, `DMSetUseNatural()`, `DMGetUseNatural()`, `DMPlexCreateGlobalToNaturalSF()`, `DMPlexDistribute()`
@*/
PetscErrorCode DMGetNaturalSF(DM dm, PetscSF *sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(sf, 2);
  *sf = dm->sfNatural;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetNaturalSF - Set the PetscSF encoding the map back to the original mesh ordering

  Input Parameters:
+ dm - The DM
- sf - The PetscSF

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetNaturalSF()`, `DMSetUseNatural()`, `DMGetUseNatural()`, `DMPlexCreateGlobalToNaturalSF()`, `DMPlexDistribute()`
@*/
PetscErrorCode DMSetNaturalSF(DM dm, PetscSF sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)sf));
  PetscCall(PetscSFDestroy(&dm->sfNatural));
  dm->sfNatural = sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSetDefaultAdjacency_Private(DM dm, PetscInt f, PetscObject disc)
{
  PetscClassId id;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetClassId(disc, &id));
  if (id == PETSCFE_CLASSID) {
    PetscCall(DMSetAdjacency(dm, f, PETSC_FALSE, PETSC_TRUE));
  } else if (id == PETSCFV_CLASSID) {
    PetscCall(DMSetAdjacency(dm, f, PETSC_TRUE, PETSC_FALSE));
  } else {
    PetscCall(DMSetAdjacency(dm, f, PETSC_FALSE, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldEnlarge_Static(DM dm, PetscInt NfNew)
{
  RegionField *tmpr;
  PetscInt     Nf = dm->Nf, f;

  PetscFunctionBegin;
  if (Nf >= NfNew) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscMalloc1(NfNew, &tmpr));
  for (f = 0; f < Nf; ++f) tmpr[f] = dm->fields[f];
  for (f = Nf; f < NfNew; ++f) {
    tmpr[f].disc        = NULL;
    tmpr[f].label       = NULL;
    tmpr[f].avoidTensor = PETSC_FALSE;
  }
  PetscCall(PetscFree(dm->fields));
  dm->Nf     = NfNew;
  dm->fields = tmpr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClearFields - Remove all fields from the `DM`

  Logically Collective

  Input Parameter:
. dm - The `DM`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetNumFields()`, `DMSetNumFields()`, `DMSetField()`
@*/
PetscErrorCode DMClearFields(DM dm)
{
  PetscInt f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (f = 0; f < dm->Nf; ++f) {
    PetscCall(PetscObjectDestroy(&dm->fields[f].disc));
    PetscCall(DMLabelDestroy(&dm->fields[f].label));
  }
  PetscCall(PetscFree(dm->fields));
  dm->fields = NULL;
  dm->Nf     = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNumFields - Get the number of fields in the `DM`

  Not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. Nf - The number of fields

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMSetNumFields()`, `DMSetField()`
@*/
PetscErrorCode DMGetNumFields(DM dm, PetscInt *numFields)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(numFields, 2);
  *numFields = dm->Nf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetNumFields - Set the number of fields in the `DM`

  Logically Collective

  Input Parameters:
+ dm - The `DM`
- Nf - The number of fields

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetNumFields()`, `DMSetField()`
@*/
PetscErrorCode DMSetNumFields(DM dm, PetscInt numFields)
{
  PetscInt Nf, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetNumFields(dm, &Nf));
  for (f = Nf; f < numFields; ++f) {
    PetscContainer obj;

    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)dm), &obj));
    PetscCall(DMAddField(dm, NULL, (PetscObject)obj));
    PetscCall(PetscContainerDestroy(&obj));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetField - Return the `DMLabel` and discretization object for a given `DM` field

  Not Collective

  Input Parameters:
+ dm - The `DM`
- f  - The field number

  Output Parameters:
+ label - The label indicating the support of the field, or `NULL` for the entire mesh (pass in `NULL` if not needed)
- disc - The discretization object (pass in `NULL` if not needed)

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMSetField()`
@*/
PetscErrorCode DMGetField(DM dm, PetscInt f, DMLabel *label, PetscObject *disc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(disc, 4);
  PetscCheck((f >= 0) && (f < dm->Nf), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %" PetscInt_FMT " must be in [0, %" PetscInt_FMT ")", f, dm->Nf);
  if (label) *label = dm->fields[f].label;
  if (disc) *disc = dm->fields[f].disc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Does not clear the DS */
PetscErrorCode DMSetField_Internal(DM dm, PetscInt f, DMLabel label, PetscObject disc)
{
  PetscFunctionBegin;
  PetscCall(DMFieldEnlarge_Static(dm, f + 1));
  PetscCall(DMLabelDestroy(&dm->fields[f].label));
  PetscCall(PetscObjectDestroy(&dm->fields[f].disc));
  dm->fields[f].label = label;
  dm->fields[f].disc  = disc;
  PetscCall(PetscObjectReference((PetscObject)label));
  PetscCall(PetscObjectReference((PetscObject)disc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSetField - Set the discretization object for a given `DM` field. Usually one would call `DMAddField()` which automatically handles
  the field numbering.

  Logically Collective

  Input Parameters:
+ dm    - The `DM`
. f     - The field number
. label - The label indicating the support of the field, or `NULL` for the entire mesh
- disc - The discretization object

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMGetField()`
@*/
PetscErrorCode DMSetField(DM dm, PetscInt f, DMLabel label, PetscObject disc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 3);
  PetscValidHeader(disc, 4);
  PetscCheck(f >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %" PetscInt_FMT " must be non-negative", f);
  PetscCall(DMSetField_Internal(dm, f, label, disc));
  PetscCall(DMSetDefaultAdjacency_Private(dm, f, disc));
  PetscCall(DMClearDS(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAddField - Add a field to a `DM` object. A field is a function space defined by of a set of discretization points (geometric entities)
  and a discretization object that defines the function space associated with those points.

  Logically Collective

  Input Parameters:
+ dm    - The `DM`
. label - The label indicating the support of the field, or `NULL` for the entire mesh
- disc - The discretization object

  Level: intermediate

  Notes:
  The label already exists or will be added to the `DM` with `DMSetLabel()`.

  For example, a piecewise continuous pressure field can be defined by coefficients at the cell centers of a mesh and piecewise constant functions
  within each cell. Thus a specific function in the space is defined by the combination of a `Vec` containing the coefficients, a `DM` defining the
  geometry entities, a `DMLabel` indicating a subset of those geometric entities, and a discretization object, such as a `PetscFE`.

.seealso: [](chapter_dmbase), `DM`, `DMSetLabel()`, `DMSetField()`, `DMGetField()`, `PetscFE`
@*/
PetscErrorCode DMAddField(DM dm, DMLabel label, PetscObject disc)
{
  PetscInt Nf = dm->Nf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  PetscValidHeader(disc, 3);
  PetscCall(DMFieldEnlarge_Static(dm, Nf + 1));
  dm->fields[Nf].label = label;
  dm->fields[Nf].disc  = disc;
  PetscCall(PetscObjectReference((PetscObject)label));
  PetscCall(PetscObjectReference((PetscObject)disc));
  PetscCall(DMSetDefaultAdjacency_Private(dm, Nf, disc));
  PetscCall(DMClearDS(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetFieldAvoidTensor - Set flag to avoid defining the field on tensor cells

  Logically Collective

  Input Parameters:
+ dm          - The `DM`
. f           - The field index
- avoidTensor - `PETSC_TRUE` to skip defining the field on tensor cells

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetFieldAvoidTensor()`, `DMSetField()`, `DMGetField()`
@*/
PetscErrorCode DMSetFieldAvoidTensor(DM dm, PetscInt f, PetscBool avoidTensor)
{
  PetscFunctionBegin;
  PetscCheck((f >= 0) && (f < dm->Nf), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", f, dm->Nf);
  dm->fields[f].avoidTensor = avoidTensor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetFieldAvoidTensor - Get flag to avoid defining the field on tensor cells

  Not Collective

  Input Parameters:
+ dm          - The `DM`
- f           - The field index

  Output Parameter:
. avoidTensor - The flag to avoid defining the field on tensor cells

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMAddField()`, `DMSetField()`, `DMGetField()`, `DMSetFieldAvoidTensor()`, `DMSetField()`, `DMGetField()`
@*/
PetscErrorCode DMGetFieldAvoidTensor(DM dm, PetscInt f, PetscBool *avoidTensor)
{
  PetscFunctionBegin;
  PetscCheck((f >= 0) && (f < dm->Nf), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Field %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", f, dm->Nf);
  *avoidTensor = dm->fields[f].avoidTensor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCopyFields - Copy the discretizations for the `DM` into another `DM`

  Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. newdm - The `DM`

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMGetField()`, `DMSetField()`, `DMAddField()`, `DMCopyDS()`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMCopyFields(DM dm, DM newdm)
{
  PetscInt Nf, f;

  PetscFunctionBegin;
  if (dm == newdm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMClearFields(newdm));
  for (f = 0; f < Nf; ++f) {
    DMLabel     label;
    PetscObject field;
    PetscBool   useCone, useClosure;

    PetscCall(DMGetField(dm, f, &label, &field));
    PetscCall(DMSetField(newdm, f, label, field));
    PetscCall(DMGetAdjacency(dm, f, &useCone, &useClosure));
    PetscCall(DMSetAdjacency(newdm, f, useCone, useClosure));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetAdjacency - Returns the flags for determining variable influence

  Not Collective

  Input Parameters:
+ dm - The `DM` object
- f  - The field number, or `PETSC_DEFAULT` for the default adjacency

  Output Parameters:
+ useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Level: developer

  Notes:
.vb
     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE
.ve
  Further explanation can be found in the User's Manual Section on the Influence of Variables on One Another.

.seealso: [](chapter_dmbase), `DM`, `DMSetAdjacency()`, `DMGetField()`, `DMSetField()`
@*/
PetscErrorCode DMGetAdjacency(DM dm, PetscInt f, PetscBool *useCone, PetscBool *useClosure)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (useCone) PetscValidBoolPointer(useCone, 3);
  if (useClosure) PetscValidBoolPointer(useClosure, 4);
  if (f < 0) {
    if (useCone) *useCone = dm->adjacency[0];
    if (useClosure) *useClosure = dm->adjacency[1];
  } else {
    PetscInt Nf;

    PetscCall(DMGetNumFields(dm, &Nf));
    PetscCheck(f < Nf, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %" PetscInt_FMT " must be in [0, %" PetscInt_FMT ")", f, Nf);
    if (useCone) *useCone = dm->fields[f].adjacency[0];
    if (useClosure) *useClosure = dm->fields[f].adjacency[1];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetAdjacency - Set the flags for determining variable influence

  Not Collective

  Input Parameters:
+ dm         - The `DM` object
. f          - The field number
. useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Level: developer

  Notes:
.vb
     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE
.ve
  Further explanation can be found in the User's Manual Section on the Influence of Variables on One Another.

.seealso: [](chapter_dmbase), `DM`, `DMGetAdjacency()`, `DMGetField()`, `DMSetField()`
@*/
PetscErrorCode DMSetAdjacency(DM dm, PetscInt f, PetscBool useCone, PetscBool useClosure)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (f < 0) {
    dm->adjacency[0] = useCone;
    dm->adjacency[1] = useClosure;
  } else {
    PetscInt Nf;

    PetscCall(DMGetNumFields(dm, &Nf));
    PetscCheck(f < Nf, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Field number %" PetscInt_FMT " must be in [0, %" PetscInt_FMT ")", f, Nf);
    dm->fields[f].adjacency[0] = useCone;
    dm->fields[f].adjacency[1] = useClosure;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetBasicAdjacency - Returns the flags for determining variable influence, using either the default or field 0 if it is defined

  Not collective

  Input Parameter:
. dm - The `DM` object

  Output Parameters:
+ useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Level: developer

  Notes:
.vb
     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE
.ve

.seealso: [](chapter_dmbase), `DM`, `DMSetBasicAdjacency()`, `DMGetField()`, `DMSetField()`
@*/
PetscErrorCode DMGetBasicAdjacency(DM dm, PetscBool *useCone, PetscBool *useClosure)
{
  PetscInt Nf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (useCone) PetscValidBoolPointer(useCone, 2);
  if (useClosure) PetscValidBoolPointer(useClosure, 3);
  PetscCall(DMGetNumFields(dm, &Nf));
  if (!Nf) {
    PetscCall(DMGetAdjacency(dm, PETSC_DEFAULT, useCone, useClosure));
  } else {
    PetscCall(DMGetAdjacency(dm, 0, useCone, useClosure));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetBasicAdjacency - Set the flags for determining variable influence, using either the default or field 0 if it is defined

  Not Collective

  Input Parameters:
+ dm         - The `DM` object
. useCone    - Flag for variable influence starting with the cone operation
- useClosure - Flag for variable influence using transitive closure

  Level: developer

  Notes:
.vb
     FEM:   Two points p and q are adjacent if q \in closure(star(p)),   useCone = PETSC_FALSE, useClosure = PETSC_TRUE
     FVM:   Two points p and q are adjacent if q \in support(p+cone(p)), useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
     FVM++: Two points p and q are adjacent if q \in star(closure(p)),   useCone = PETSC_TRUE,  useClosure = PETSC_TRUE
.ve

.seealso: [](chapter_dmbase), `DM`, `DMGetBasicAdjacency()`, `DMGetField()`, `DMSetField()`
@*/
PetscErrorCode DMSetBasicAdjacency(DM dm, PetscBool useCone, PetscBool useClosure)
{
  PetscInt Nf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetNumFields(dm, &Nf));
  if (!Nf) {
    PetscCall(DMSetAdjacency(dm, PETSC_DEFAULT, useCone, useClosure));
  } else {
    PetscCall(DMSetAdjacency(dm, 0, useCone, useClosure));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCompleteBCLabels_Internal(DM dm)
{
  DM           plex;
  DMLabel     *labels, *glabels;
  const char **names;
  char        *sendNames, *recvNames;
  PetscInt     Nds, s, maxLabels = 0, maxLen = 0, gmaxLen, Nl = 0, gNl, l, gl, m;
  size_t       len;
  MPI_Comm     comm;
  PetscMPIInt  rank, size, p, *counts, *displs;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS  dsBC;
    PetscInt numBd;

    PetscCall(DMGetRegionNumDS(dm, s, NULL, NULL, &dsBC, NULL));
    PetscCall(PetscDSGetNumBoundary(dsBC, &numBd));
    maxLabels += numBd;
  }
  PetscCall(PetscCalloc1(maxLabels, &labels));
  /* Get list of labels to be completed */
  for (s = 0; s < Nds; ++s) {
    PetscDS  dsBC;
    PetscInt numBd, bd;

    PetscCall(DMGetRegionNumDS(dm, s, NULL, NULL, &dsBC, NULL));
    PetscCall(PetscDSGetNumBoundary(dsBC, &numBd));
    for (bd = 0; bd < numBd; ++bd) {
      DMLabel      label;
      PetscInt     field;
      PetscObject  obj;
      PetscClassId id;

      PetscCall(PetscDSGetBoundary(dsBC, bd, NULL, NULL, NULL, &label, NULL, NULL, &field, NULL, NULL, NULL, NULL, NULL));
      PetscCall(DMGetField(dm, field, NULL, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (!(id == PETSCFE_CLASSID) || !label) continue;
      for (l = 0; l < Nl; ++l)
        if (labels[l] == label) break;
      if (l == Nl) labels[Nl++] = label;
    }
  }
  /* Get label names */
  PetscCall(PetscMalloc1(Nl, &names));
  for (l = 0; l < Nl; ++l) PetscCall(PetscObjectGetName((PetscObject)labels[l], &names[l]));
  for (l = 0; l < Nl; ++l) {
    PetscCall(PetscStrlen(names[l], &len));
    maxLen = PetscMax(maxLen, (PetscInt)len + 2);
  }
  PetscCall(PetscFree(labels));
  PetscCall(MPIU_Allreduce(&maxLen, &gmaxLen, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscCalloc1(Nl * gmaxLen, &sendNames));
  for (l = 0; l < Nl; ++l) PetscCall(PetscStrncpy(&sendNames[gmaxLen * l], names[l], gmaxLen));
  PetscCall(PetscFree(names));
  /* Put all names on all processes */
  PetscCall(PetscCalloc2(size, &counts, size + 1, &displs));
  PetscCallMPI(MPI_Allgather(&Nl, 1, MPI_INT, counts, 1, MPI_INT, comm));
  for (p = 0; p < size; ++p) displs[p + 1] = displs[p] + counts[p];
  gNl = displs[size];
  for (p = 0; p < size; ++p) {
    counts[p] *= gmaxLen;
    displs[p] *= gmaxLen;
  }
  PetscCall(PetscCalloc2(gNl * gmaxLen, &recvNames, gNl, &glabels));
  PetscCallMPI(MPI_Allgatherv(sendNames, counts[rank], MPI_CHAR, recvNames, counts, displs, MPI_CHAR, comm));
  PetscCall(PetscFree2(counts, displs));
  PetscCall(PetscFree(sendNames));
  for (l = 0, gl = 0; l < gNl; ++l) {
    PetscCall(DMGetLabel(dm, &recvNames[l * gmaxLen], &glabels[gl]));
    PetscCheck(glabels[gl], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Label %s missing on rank %d", &recvNames[l * gmaxLen], rank);
    for (m = 0; m < gl; ++m)
      if (glabels[m] == glabels[gl]) continue;
    PetscCall(DMConvert(dm, DMPLEX, &plex));
    PetscCall(DMPlexLabelComplete(plex, glabels[gl]));
    PetscCall(DMDestroy(&plex));
    ++gl;
  }
  PetscCall(PetscFree2(recvNames, glabels));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDSEnlarge_Static(DM dm, PetscInt NdsNew)
{
  DMSpace *tmpd;
  PetscInt Nds = dm->Nds, s;

  PetscFunctionBegin;
  if (Nds >= NdsNew) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscMalloc1(NdsNew, &tmpd));
  for (s = 0; s < Nds; ++s) tmpd[s] = dm->probs[s];
  for (s = Nds; s < NdsNew; ++s) {
    tmpd[s].ds     = NULL;
    tmpd[s].label  = NULL;
    tmpd[s].fields = NULL;
  }
  PetscCall(PetscFree(dm->probs));
  dm->Nds   = NdsNew;
  dm->probs = tmpd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNumDS - Get the number of discrete systems in the `DM`

  Not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. Nds - The number of `PetscDS` objects

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMGetNumDS(DM dm, PetscInt *Nds)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(Nds, 2);
  *Nds = dm->Nds;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMClearDS - Remove all discrete systems from the `DM`

  Logically Collective

  Input Parameter:
. dm - The `DM`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetNumDS()`, `DMGetDS()`, `DMSetField()`
@*/
PetscErrorCode DMClearDS(DM dm)
{
  PetscInt s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (s = 0; s < dm->Nds; ++s) {
    PetscCall(PetscDSDestroy(&dm->probs[s].ds));
    PetscCall(PetscDSDestroy(&dm->probs[s].dsIn));
    PetscCall(DMLabelDestroy(&dm->probs[s].label));
    PetscCall(ISDestroy(&dm->probs[s].fields));
  }
  PetscCall(PetscFree(dm->probs));
  dm->probs = NULL;
  dm->Nds   = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetDS - Get the default `PetscDS`

  Not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. ds - The default `PetscDS`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMGetCellDS()`, `DMGetRegionDS()`
@*/
PetscErrorCode DMGetDS(DM dm, PetscDS *ds)
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ds, 2);
  PetscCheck(dm->Nds > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Need to call DMCreateDS() before calling DMGetDS()");
  *ds = dm->probs[0].ds;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellDS - Get the `PetscDS` defined on a given cell

  Not Collective

  Input Parameters:
+ dm    - The `DM`
- point - Cell for the `PetscDS`

  Output Parameters:
+ ds   - The `PetscDS` defined on the given cell
- dsIn - The `PetscDS` for input on the given cell, or NULL if the same ds

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMGetDS()`, `DMSetRegionDS()`
@*/
PetscErrorCode DMGetCellDS(DM dm, PetscInt point, PetscDS *ds, PetscDS *dsIn)
{
  PetscDS  dsDef = NULL;
  PetscInt s;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ds) PetscValidPointer(ds, 3);
  if (dsIn) PetscValidPointer(dsIn, 4);
  PetscCheck(point >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Mesh point cannot be negative: %" PetscInt_FMT, point);
  if (ds) *ds = NULL;
  if (dsIn) *dsIn = NULL;
  for (s = 0; s < dm->Nds; ++s) {
    PetscInt val;

    if (!dm->probs[s].label) {
      dsDef = dm->probs[s].ds;
    } else {
      PetscCall(DMLabelGetValue(dm->probs[s].label, point, &val));
      if (val >= 0) {
        if (ds) *ds = dm->probs[s].ds;
        if (dsIn) *dsIn = dm->probs[s].dsIn;
        break;
      }
    }
  }
  if (ds && !*ds) *ds = dsDef;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetRegionDS - Get the `PetscDS` for a given mesh region, defined by a `DMLabel`

  Not Collective

  Input Parameters:
+ dm    - The `DM`
- label - The `DMLabel` defining the mesh region, or `NULL` for the entire mesh

  Output Parameters:
+ fields - The `IS` containing the `DM` field numbers for the fields in this `PetscDS`, or `NULL`
. ds     - The `PetscDS` defined on the given region, or `NULL`
- dsIn   - The `PetscDS` for input in the given region, or `NULL`

  Level: advanced

  Note:
  If a non-`NULL` label is given, but there is no `PetscDS` on that specific label,
  the `PetscDS` for the full domain (if present) is returned. Returns with
  fields = `NULL` and ds = `NULL` if there is no `PetscDS` for the full domain.

.seealso: [](chapter_dmbase), `DM`, `DMGetRegionNumDS()`, `DMSetRegionDS()`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMGetRegionDS(DM dm, DMLabel label, IS *fields, PetscDS *ds, PetscDS *dsIn)
{
  PetscInt Nds = dm->Nds, s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  if (fields) {
    PetscValidPointer(fields, 3);
    *fields = NULL;
  }
  if (ds) {
    PetscValidPointer(ds, 4);
    *ds = NULL;
  }
  if (dsIn) {
    PetscValidPointer(dsIn, 5);
    *dsIn = NULL;
  }
  for (s = 0; s < Nds; ++s) {
    if (dm->probs[s].label == label || !dm->probs[s].label) {
      if (fields) *fields = dm->probs[s].fields;
      if (ds) *ds = dm->probs[s].ds;
      if (dsIn) *dsIn = dm->probs[s].dsIn;
      if (dm->probs[s].label) PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetRegionDS - Set the `PetscDS` for a given mesh region, defined by a `DMLabel`

  Collective

  Input Parameters:
+ dm     - The `DM`
. label  - The `DMLabel` defining the mesh region, or `NULL` for the entire mesh
. fields - The `IS` containing the `DM` field numbers for the fields in this `PetscDS`, or `NULL` for all fields
. ds     - The `PetscDS` defined on the given region
- dsIn   - The `PetscDS` for input on the given cell, or `NULL` if it is the same `PetscDS`

  Level: advanced

  Note:
  If the label has a `PetscDS` defined, it will be replaced. Otherwise, it will be added to the `DM`. If the `PetscDS` is replaced,
  the fields argument is ignored.

.seealso: [](chapter_dmbase), `DM`, `DMGetRegionDS()`, `DMSetRegionNumDS()`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMSetRegionDS(DM dm, DMLabel label, IS fields, PetscDS ds, PetscDS dsIn)
{
  PetscInt Nds = dm->Nds, s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  if (fields) PetscValidHeaderSpecific(fields, IS_CLASSID, 3);
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 4);
  if (dsIn) PetscValidHeaderSpecific(dsIn, PETSCDS_CLASSID, 5);
  for (s = 0; s < Nds; ++s) {
    if (dm->probs[s].label == label) {
      PetscCall(PetscDSDestroy(&dm->probs[s].ds));
      PetscCall(PetscDSDestroy(&dm->probs[s].dsIn));
      dm->probs[s].ds   = ds;
      dm->probs[s].dsIn = dsIn;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCall(DMDSEnlarge_Static(dm, Nds + 1));
  PetscCall(PetscObjectReference((PetscObject)label));
  PetscCall(PetscObjectReference((PetscObject)fields));
  PetscCall(PetscObjectReference((PetscObject)ds));
  PetscCall(PetscObjectReference((PetscObject)dsIn));
  if (!label) {
    /* Put the NULL label at the front, so it is returned as the default */
    for (s = Nds - 1; s >= 0; --s) dm->probs[s + 1] = dm->probs[s];
    Nds = 0;
  }
  dm->probs[Nds].label  = label;
  dm->probs[Nds].fields = fields;
  dm->probs[Nds].ds     = ds;
  dm->probs[Nds].dsIn   = dsIn;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetRegionNumDS - Get the `PetscDS` for a given mesh region, defined by the region number

  Not Collective

  Input Parameters:
+ dm  - The `DM`
- num - The region number, in [0, Nds)

  Output Parameters:
+ label  - The region label, or `NULL`
. fields - The `IS` containing the `DM` field numbers for the fields in this `PetscDS`, or `NULL`
. ds     - The `PetscDS` defined on the given region, or `NULL`
- dsIn   - The `PetscDS` for input in the given region, or `NULL`

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMGetRegionDS()`, `DMSetRegionDS()`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMGetRegionNumDS(DM dm, PetscInt num, DMLabel *label, IS *fields, PetscDS *ds, PetscDS *dsIn)
{
  PetscInt Nds;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetNumDS(dm, &Nds));
  PetscCheck((num >= 0) && (num < Nds), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Region number %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", num, Nds);
  if (label) {
    PetscValidPointer(label, 3);
    *label = dm->probs[num].label;
  }
  if (fields) {
    PetscValidPointer(fields, 4);
    *fields = dm->probs[num].fields;
  }
  if (ds) {
    PetscValidPointer(ds, 5);
    *ds = dm->probs[num].ds;
  }
  if (dsIn) {
    PetscValidPointer(dsIn, 6);
    *dsIn = dm->probs[num].dsIn;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetRegionNumDS - Set the `PetscDS` for a given mesh region, defined by the region number

  Not Collective

  Input Parameters:
+ dm     - The `DM`
. num    - The region number, in [0, Nds)
. label  - The region label, or `NULL`
. fields - The `IS` containing the `DM` field numbers for the fields in this `PetscDS`, or `NULL` to prevent setting
. ds     - The `PetscDS` defined on the given region, or `NULL` to prevent setting
- dsIn   - The `PetscDS` for input on the given cell, or `NULL` if it is the same `PetscDS`

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMGetRegionDS()`, `DMSetRegionDS()`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMSetRegionNumDS(DM dm, PetscInt num, DMLabel label, IS fields, PetscDS ds, PetscDS dsIn)
{
  PetscInt Nds;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 3);
  PetscCall(DMGetNumDS(dm, &Nds));
  PetscCheck((num >= 0) && (num < Nds), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Region number %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", num, Nds);
  PetscCall(PetscObjectReference((PetscObject)label));
  PetscCall(DMLabelDestroy(&dm->probs[num].label));
  dm->probs[num].label = label;
  if (fields) {
    PetscValidHeaderSpecific(fields, IS_CLASSID, 4);
    PetscCall(PetscObjectReference((PetscObject)fields));
    PetscCall(ISDestroy(&dm->probs[num].fields));
    dm->probs[num].fields = fields;
  }
  if (ds) {
    PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 5);
    PetscCall(PetscObjectReference((PetscObject)ds));
    PetscCall(PetscDSDestroy(&dm->probs[num].ds));
    dm->probs[num].ds = ds;
  }
  if (dsIn) {
    PetscValidHeaderSpecific(dsIn, PETSCDS_CLASSID, 6);
    PetscCall(PetscObjectReference((PetscObject)dsIn));
    PetscCall(PetscDSDestroy(&dm->probs[num].dsIn));
    dm->probs[num].dsIn = dsIn;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMFindRegionNum - Find the region number for a given `PetscDS`, or -1 if it is not found.

  Not Collective

  Input Parameters:
+ dm  - The `DM`
- ds  - The `PetscDS` defined on the given region

  Output Parameter:
. num - The region number, in [0, Nds), or -1 if not found

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMGetRegionNumDS()`, `DMGetRegionDS()`, `DMSetRegionDS()`, `DMGetDS()`, `DMGetCellDS()`
@*/
PetscErrorCode DMFindRegionNum(DM dm, PetscDS ds, PetscInt *num)
{
  PetscInt Nds, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(ds, PETSCDS_CLASSID, 2);
  PetscValidIntPointer(num, 3);
  PetscCall(DMGetNumDS(dm, &Nds));
  for (n = 0; n < Nds; ++n)
    if (ds == dm->probs[n].ds) break;
  if (n >= Nds) *num = -1;
  else *num = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateFEDefault - Create a `PetscFE` based on the celltype for the mesh

  Not Collective

  Input Parameters:
+ dm     - The `DM`
. Nc     - The number of components for the field
. prefix - The options prefix for the output `PetscFE`, or `NULL`
- qorder - The quadrature order or `PETSC_DETERMINE` to use `PetscSpace` polynomial degree

  Output Parameter:
. fem - The `PetscFE`

  Level: intermediate

  Note:
  This is a convenience method that just calls `PetscFECreateByCell()` underneath.

.seealso: [](chapter_dmbase), `DM`, `PetscFECreateByCell()`, `DMAddField()`, `DMCreateDS()`, `DMGetCellDS()`, `DMGetRegionDS()`
@*/
PetscErrorCode DMCreateFEDefault(DM dm, PetscInt Nc, const char prefix[], PetscInt qorder, PetscFE *fem)
{
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, Nc, 2);
  if (prefix) PetscValidCharPointer(prefix, 3);
  PetscValidLogicalCollectiveInt(dm, qorder, 4);
  PetscValidPointer(fem, 5);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, Nc, ct, prefix, qorder, fem));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCreateDS - Create the discrete systems for the `DM` based upon the fields added to the `DM`

  Collective

  Input Parameter:
. dm - The `DM`

  Options Database Key:
. -dm_petscds_view - View all the `PetscDS` objects in this `DM`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMSetField`, `DMAddField()`, `DMGetDS()`, `DMGetCellDS()`, `DMGetRegionDS()`, `DMSetRegionDS()`
@*/
PetscErrorCode DMCreateDS(DM dm)
{
  MPI_Comm  comm;
  PetscDS   dsDef;
  DMLabel  *labelSet;
  PetscInt  dE, Nf = dm->Nf, f, s, Nl, l, Ndef, k;
  PetscBool doSetup = PETSC_TRUE, flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->fields) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  /* Determine how many regions we have */
  PetscCall(PetscMalloc1(Nf, &labelSet));
  Nl   = 0;
  Ndef = 0;
  for (f = 0; f < Nf; ++f) {
    DMLabel  label = dm->fields[f].label;
    PetscInt l;

#ifdef PETSC_HAVE_LIBCEED
    /* Move CEED context to discretizations */
    {
      PetscClassId id;

      PetscCall(PetscObjectGetClassId(dm->fields[f].disc, &id));
      if (id == PETSCFE_CLASSID) {
        Ceed ceed;

        PetscCall(DMGetCeed(dm, &ceed));
        PetscCall(PetscFESetCeed((PetscFE)dm->fields[f].disc, ceed));
      }
    }
#endif
    if (!label) {
      ++Ndef;
      continue;
    }
    for (l = 0; l < Nl; ++l)
      if (label == labelSet[l]) break;
    if (l < Nl) continue;
    labelSet[Nl++] = label;
  }
  /* Create default DS if there are no labels to intersect with */
  PetscCall(DMGetRegionDS(dm, NULL, NULL, &dsDef, NULL));
  if (!dsDef && Ndef && !Nl) {
    IS        fields;
    PetscInt *fld, nf;

    for (f = 0, nf = 0; f < Nf; ++f)
      if (!dm->fields[f].label) ++nf;
    PetscCheck(nf, comm, PETSC_ERR_PLIB, "All fields have labels, but we are trying to create a default DS");
    PetscCall(PetscMalloc1(nf, &fld));
    for (f = 0, nf = 0; f < Nf; ++f)
      if (!dm->fields[f].label) fld[nf++] = f;
    PetscCall(ISCreate(PETSC_COMM_SELF, &fields));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fields, "dm_fields_"));
    PetscCall(ISSetType(fields, ISGENERAL));
    PetscCall(ISGeneralSetIndices(fields, nf, fld, PETSC_OWN_POINTER));

    PetscCall(PetscDSCreate(PETSC_COMM_SELF, &dsDef));
    PetscCall(DMSetRegionDS(dm, NULL, fields, dsDef, NULL));
    PetscCall(PetscDSDestroy(&dsDef));
    PetscCall(ISDestroy(&fields));
  }
  PetscCall(DMGetRegionDS(dm, NULL, NULL, &dsDef, NULL));
  if (dsDef) PetscCall(PetscDSSetCoordinateDimension(dsDef, dE));
  /* Intersect labels with default fields */
  if (Ndef && Nl) {
    DM              plex;
    DMLabel         cellLabel;
    IS              fieldIS, allcellIS, defcellIS = NULL;
    PetscInt       *fields;
    const PetscInt *cells;
    PetscInt        depth, nf = 0, n, c;

    PetscCall(DMConvert(dm, DMPLEX, &plex));
    PetscCall(DMPlexGetDepth(plex, &depth));
    PetscCall(DMGetStratumIS(plex, "dim", depth, &allcellIS));
    if (!allcellIS) PetscCall(DMGetStratumIS(plex, "depth", depth, &allcellIS));
    /* TODO This looks like it only works for one label */
    for (l = 0; l < Nl; ++l) {
      DMLabel label = labelSet[l];
      IS      pointIS;

      PetscCall(ISDestroy(&defcellIS));
      PetscCall(DMLabelGetStratumIS(label, 1, &pointIS));
      PetscCall(ISDifference(allcellIS, pointIS, &defcellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISDestroy(&allcellIS));

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "defaultCells", &cellLabel));
    PetscCall(ISGetLocalSize(defcellIS, &n));
    PetscCall(ISGetIndices(defcellIS, &cells));
    for (c = 0; c < n; ++c) PetscCall(DMLabelSetValue(cellLabel, cells[c], 1));
    PetscCall(ISRestoreIndices(defcellIS, &cells));
    PetscCall(ISDestroy(&defcellIS));
    PetscCall(DMPlexLabelComplete(plex, cellLabel));

    PetscCall(PetscMalloc1(Ndef, &fields));
    for (f = 0; f < Nf; ++f)
      if (!dm->fields[f].label) fields[nf++] = f;
    PetscCall(ISCreate(PETSC_COMM_SELF, &fieldIS));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fieldIS, "dm_fields_"));
    PetscCall(ISSetType(fieldIS, ISGENERAL));
    PetscCall(ISGeneralSetIndices(fieldIS, nf, fields, PETSC_OWN_POINTER));

    PetscCall(PetscDSCreate(PETSC_COMM_SELF, &dsDef));
    PetscCall(DMSetRegionDS(dm, cellLabel, fieldIS, dsDef, NULL));
    PetscCall(PetscDSSetCoordinateDimension(dsDef, dE));
    PetscCall(DMLabelDestroy(&cellLabel));
    PetscCall(PetscDSDestroy(&dsDef));
    PetscCall(ISDestroy(&fieldIS));
    PetscCall(DMDestroy(&plex));
  }
  /* Create label DSes
     - WE ONLY SUPPORT IDENTICAL OR DISJOINT LABELS
  */
  /* TODO Should check that labels are disjoint */
  for (l = 0; l < Nl; ++l) {
    DMLabel   label = labelSet[l];
    PetscDS   ds, dsIn = NULL;
    IS        fields;
    PetscInt *fld, nf;

    PetscCall(PetscDSCreate(PETSC_COMM_SELF, &ds));
    for (f = 0, nf = 0; f < Nf; ++f)
      if (label == dm->fields[f].label || !dm->fields[f].label) ++nf;
    PetscCall(PetscMalloc1(nf, &fld));
    for (f = 0, nf = 0; f < Nf; ++f)
      if (label == dm->fields[f].label || !dm->fields[f].label) fld[nf++] = f;
    PetscCall(ISCreate(PETSC_COMM_SELF, &fields));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fields, "dm_fields_"));
    PetscCall(ISSetType(fields, ISGENERAL));
    PetscCall(ISGeneralSetIndices(fields, nf, fld, PETSC_OWN_POINTER));
    PetscCall(PetscDSSetCoordinateDimension(ds, dE));
    {
      DMPolytopeType ct;
      PetscInt       lStart, lEnd;
      PetscBool      isCohesiveLocal = PETSC_FALSE, isCohesive;

      PetscCall(DMLabelGetBounds(label, &lStart, &lEnd));
      if (lStart >= 0) {
        PetscCall(DMPlexGetCellType(dm, lStart, &ct));
        switch (ct) {
        case DM_POLYTOPE_POINT_PRISM_TENSOR:
        case DM_POLYTOPE_SEG_PRISM_TENSOR:
        case DM_POLYTOPE_TRI_PRISM_TENSOR:
        case DM_POLYTOPE_QUAD_PRISM_TENSOR:
          isCohesiveLocal = PETSC_TRUE;
          break;
        default:
          break;
        }
      }
      PetscCall(MPIU_Allreduce(&isCohesiveLocal, &isCohesive, 1, MPIU_BOOL, MPI_LOR, comm));
      if (isCohesive) {
        PetscCall(PetscDSCreate(PETSC_COMM_SELF, &dsIn));
        PetscCall(PetscDSSetCoordinateDimension(dsIn, dE));
      }
      for (f = 0, nf = 0; f < Nf; ++f) {
        if (label == dm->fields[f].label || !dm->fields[f].label) {
          if (label == dm->fields[f].label) {
            PetscCall(PetscDSSetDiscretization(ds, nf, NULL));
            PetscCall(PetscDSSetCohesive(ds, nf, isCohesive));
            if (dsIn) {
              PetscCall(PetscDSSetDiscretization(dsIn, nf, NULL));
              PetscCall(PetscDSSetCohesive(dsIn, nf, isCohesive));
            }
          }
          ++nf;
        }
      }
    }
    PetscCall(DMSetRegionDS(dm, label, fields, ds, dsIn));
    PetscCall(ISDestroy(&fields));
    PetscCall(PetscDSDestroy(&ds));
    PetscCall(PetscDSDestroy(&dsIn));
  }
  PetscCall(PetscFree(labelSet));
  /* Set fields in DSes */
  for (s = 0; s < dm->Nds; ++s) {
    PetscDS         ds     = dm->probs[s].ds;
    PetscDS         dsIn   = dm->probs[s].dsIn;
    IS              fields = dm->probs[s].fields;
    const PetscInt *fld;
    PetscInt        nf, dsnf;
    PetscBool       isCohesive;

    PetscCall(PetscDSGetNumFields(ds, &dsnf));
    PetscCall(PetscDSIsCohesive(ds, &isCohesive));
    PetscCall(ISGetLocalSize(fields, &nf));
    PetscCall(ISGetIndices(fields, &fld));
    for (f = 0; f < nf; ++f) {
      PetscObject  disc = dm->fields[fld[f]].disc;
      PetscBool    isCohesiveField;
      PetscClassId id;

      /* Handle DS with no fields */
      if (dsnf) PetscCall(PetscDSGetCohesive(ds, f, &isCohesiveField));
      /* If this is a cohesive cell, then regular fields need the lower dimensional discretization */
      if (isCohesive) {
        if (!isCohesiveField) {
          PetscObject bdDisc;

          PetscCall(PetscFEGetHeightSubspace((PetscFE)disc, 1, (PetscFE *)&bdDisc));
          PetscCall(PetscDSSetDiscretization(ds, f, bdDisc));
          PetscCall(PetscDSSetDiscretization(dsIn, f, disc));
        } else {
          PetscCall(PetscDSSetDiscretization(ds, f, disc));
          PetscCall(PetscDSSetDiscretization(dsIn, f, disc));
        }
      } else {
        PetscCall(PetscDSSetDiscretization(ds, f, disc));
      }
      /* We allow people to have placeholder fields and construct the Section by hand */
      PetscCall(PetscObjectGetClassId(disc, &id));
      if ((id != PETSCFE_CLASSID) && (id != PETSCFV_CLASSID)) doSetup = PETSC_FALSE;
    }
    PetscCall(ISRestoreIndices(fields, &fld));
  }
  /* Allow k-jet tabulation */
  PetscCall(PetscOptionsGetInt(NULL, ((PetscObject)dm)->prefix, "-dm_ds_jet_degree", &k, &flg));
  if (flg) {
    for (s = 0; s < dm->Nds; ++s) {
      PetscDS  ds   = dm->probs[s].ds;
      PetscDS  dsIn = dm->probs[s].dsIn;
      PetscInt Nf, f;

      PetscCall(PetscDSGetNumFields(ds, &Nf));
      for (f = 0; f < Nf; ++f) {
        PetscCall(PetscDSSetJetDegree(ds, f, k));
        if (dsIn) PetscCall(PetscDSSetJetDegree(dsIn, f, k));
      }
    }
  }
  /* Setup DSes */
  if (doSetup) {
    for (s = 0; s < dm->Nds; ++s) {
      PetscCall(PetscDSSetUp(dm->probs[s].ds));
      if (dm->probs[s].dsIn) PetscCall(PetscDSSetUp(dm->probs[s].dsIn));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMComputeExactSolution - Compute the exact solution for a given `DM`, using the `PetscDS` information.

  Collective

  Input Parameters:
+ dm   - The `DM`
- time - The time

  Output Parameters:
+ u    - The vector will be filled with exact solution values, or `NULL`
- u_t  - The vector will be filled with the time derivative of exact solution values, or `NULL`

  Level: developer

  Note:
  The user must call `PetscDSSetExactSolution()` before using this routine

.seealso: [](chapter_dmbase), `DM`, `PetscDSSetExactSolution()`
@*/
PetscErrorCode DMComputeExactSolution(DM dm, PetscReal time, Vec u, Vec u_t)
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void   **ectxs;
  Vec      locu, locu_t;
  PetscInt Nf, Nds, s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (u) {
    PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
    PetscCall(DMGetLocalVector(dm, &locu));
    PetscCall(VecSet(locu, 0.));
  }
  if (u_t) {
    PetscValidHeaderSpecific(u_t, VEC_CLASSID, 4);
    PetscCall(DMGetLocalVector(dm, &locu_t));
    PetscCall(VecSet(locu_t, 0.));
  }
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscMalloc2(Nf, &exacts, Nf, &ectxs));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS         ds;
    DMLabel         label;
    IS              fieldIS;
    const PetscInt *fields, id = 1;
    PetscInt        dsNf, f;

    PetscCall(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds, NULL));
    PetscCall(PetscDSGetNumFields(ds, &dsNf));
    PetscCall(ISGetIndices(fieldIS, &fields));
    PetscCall(PetscArrayzero(exacts, Nf));
    PetscCall(PetscArrayzero(ectxs, Nf));
    if (u) {
      for (f = 0; f < dsNf; ++f) PetscCall(PetscDSGetExactSolution(ds, fields[f], &exacts[fields[f]], &ectxs[fields[f]]));
      if (label) PetscCall(DMProjectFunctionLabelLocal(dm, time, label, 1, &id, 0, NULL, exacts, ectxs, INSERT_ALL_VALUES, locu));
      else PetscCall(DMProjectFunctionLocal(dm, time, exacts, ectxs, INSERT_ALL_VALUES, locu));
    }
    if (u_t) {
      PetscCall(PetscArrayzero(exacts, Nf));
      PetscCall(PetscArrayzero(ectxs, Nf));
      for (f = 0; f < dsNf; ++f) PetscCall(PetscDSGetExactSolutionTimeDerivative(ds, fields[f], &exacts[fields[f]], &ectxs[fields[f]]));
      if (label) PetscCall(DMProjectFunctionLabelLocal(dm, time, label, 1, &id, 0, NULL, exacts, ectxs, INSERT_ALL_VALUES, locu_t));
      else PetscCall(DMProjectFunctionLocal(dm, time, exacts, ectxs, INSERT_ALL_VALUES, locu_t));
    }
    PetscCall(ISRestoreIndices(fieldIS, &fields));
  }
  if (u) {
    PetscCall(PetscObjectSetName((PetscObject)u, "Exact Solution"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)u, "exact_"));
  }
  if (u_t) {
    PetscCall(PetscObjectSetName((PetscObject)u, "Exact Solution Time Derivative"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)u_t, "exact_t_"));
  }
  PetscCall(PetscFree2(exacts, ectxs));
  if (u) {
    PetscCall(DMLocalToGlobalBegin(dm, locu, INSERT_ALL_VALUES, u));
    PetscCall(DMLocalToGlobalEnd(dm, locu, INSERT_ALL_VALUES, u));
    PetscCall(DMRestoreLocalVector(dm, &locu));
  }
  if (u_t) {
    PetscCall(DMLocalToGlobalBegin(dm, locu_t, INSERT_ALL_VALUES, u_t));
    PetscCall(DMLocalToGlobalEnd(dm, locu_t, INSERT_ALL_VALUES, u_t));
    PetscCall(DMRestoreLocalVector(dm, &locu_t));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTransferDS_Internal(DM dm, DMLabel label, IS fields, PetscDS ds, PetscDS dsIn)
{
  PetscDS dsNew, dsInNew = NULL;

  PetscFunctionBegin;
  PetscCall(PetscDSCreate(PetscObjectComm((PetscObject)ds), &dsNew));
  PetscCall(PetscDSCopy(ds, dm, dsNew));
  if (dsIn) {
    PetscCall(PetscDSCreate(PetscObjectComm((PetscObject)dsIn), &dsInNew));
    PetscCall(PetscDSCopy(dsIn, dm, dsInNew));
  }
  PetscCall(DMSetRegionDS(dm, label, fields, dsNew, dsInNew));
  PetscCall(PetscDSDestroy(&dsNew));
  PetscCall(PetscDSDestroy(&dsInNew));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCopyDS - Copy the discrete systems for the `DM` into another `DM`

  Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. newdm - The `DM`

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMCopyFields()`, `DMAddField()`, `DMGetDS()`, `DMGetCellDS()`, `DMGetRegionDS()`, `DMSetRegionDS()`
@*/
PetscErrorCode DMCopyDS(DM dm, DM newdm)
{
  PetscInt Nds, s;

  PetscFunctionBegin;
  if (dm == newdm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetNumDS(dm, &Nds));
  PetscCall(DMClearDS(newdm));
  for (s = 0; s < Nds; ++s) {
    DMLabel  label;
    IS       fields;
    PetscDS  ds, dsIn, newds;
    PetscInt Nbd, bd;

    PetscCall(DMGetRegionNumDS(dm, s, &label, &fields, &ds, &dsIn));
    /* TODO: We need to change all keys from labels in the old DM to labels in the new DM */
    PetscCall(DMTransferDS_Internal(newdm, label, fields, ds, dsIn));
    /* Complete new labels in the new DS */
    PetscCall(DMGetRegionDS(newdm, label, NULL, &newds, NULL));
    PetscCall(PetscDSGetNumBoundary(newds, &Nbd));
    for (bd = 0; bd < Nbd; ++bd) {
      PetscWeakForm wf;
      DMLabel       label;
      PetscInt      field;

      PetscCall(PetscDSGetBoundary(newds, bd, &wf, NULL, NULL, &label, NULL, NULL, &field, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscWeakFormReplaceLabel(wf, label));
    }
  }
  PetscCall(DMCompleteBCLabels_Internal(newdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCopyDisc - Copy the fields and discrete systems for the `DM` into another `DM`

  Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. newdm - The `DM`

  Level: advanced

  Developer Note:
  Really ugly name, nothing in PETSc is called a `Disc` plus it is an ugly abbreviation

.seealso: [](chapter_dmbase), `DM`, `DMCopyFields()`, `DMCopyDS()`
@*/
PetscErrorCode DMCopyDisc(DM dm, DM newdm)
{
  PetscFunctionBegin;
  PetscCall(DMCopyFields(dm, newdm));
  PetscCall(DMCopyDS(dm, newdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetDimension - Return the topological dimension of the `DM`

  Not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. dim - The topological dimension

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMSetDimension()`, `DMCreate()`
@*/
PetscErrorCode DMGetDimension(DM dm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(dim, 2);
  *dim = dm->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetDimension - Set the topological dimension of the `DM`

  Collective

  Input Parameters:
+ dm - The `DM`
- dim - The topological dimension

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMGetDimension()`, `DMCreate()`
@*/
PetscErrorCode DMSetDimension(DM dm, PetscInt dim)
{
  PetscDS  ds;
  PetscInt Nds, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, dim, 2);
  dm->dim = dim;
  if (dm->dim >= 0) {
    PetscCall(DMGetNumDS(dm, &Nds));
    for (n = 0; n < Nds; ++n) {
      PetscCall(DMGetRegionNumDS(dm, n, NULL, NULL, &ds, NULL));
      if (ds->dimEmbed < 0) PetscCall(PetscDSSetCoordinateDimension(ds, dim));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetDimPoints - Get the half-open interval for all points of a given dimension

  Collective

  Input Parameters:
+ dm - the `DM`
- dim - the dimension

  Output Parameters:
+ pStart - The first point of the given dimension
- pEnd - The first point following points of the given dimension

  Level: intermediate

  Note:
  The points are vertices in the Hasse diagram encoding the topology. This is explained in
  https://arxiv.org/abs/0908.4427. If no points exist of this dimension in the storage scheme,
  then the interval is empty.

.seealso: [](chapter_dmbase), `DM`, `DMPLEX`, `DMPlexGetDepthStratum()`, `DMPlexGetHeightStratum()`
@*/
PetscErrorCode DMGetDimPoints(DM dm, PetscInt dim, PetscInt *pStart, PetscInt *pEnd)
{
  PetscInt d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDimension(dm, &d));
  PetscCheck((dim >= 0) && (dim <= d), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %" PetscInt_FMT, dim);
  PetscUseTypeMethod(dm, getdimpoints, dim, pStart, pEnd);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetOutputDM - Retrieve the `DM` associated with the layout for output

  Collective

  Input Parameter:
. dm - The original `DM`

  Output Parameter:
. odm - The `DM` which provides the layout for output

  Level: intermediate

  Note:
  In some situations the vector obtained with `DMCreateGlobalVector()` excludes points for degrees of freedom that are associated with fixed (Dirichelet) boundary
  conditions since the algebraic solver does not solve for those variables. The output `DM` includes these excluded points and its global vector contains the
  locations for those dof so that they can be output to a file or other viewer along with the unconstrained dof.

.seealso: [](chapter_dmbase), `DM`, `VecView()`, `DMGetGlobalSection()`, `DMCreateGlobalVector()`, `PetscSectionHasConstraints()`, `DMSetGlobalSection()`
@*/
PetscErrorCode DMGetOutputDM(DM dm, DM *odm)
{
  PetscSection section;
  PetscBool    hasConstraints, ghasConstraints;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(odm, 2);
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionHasConstraints(section, &hasConstraints));
  PetscCall(MPIU_Allreduce(&hasConstraints, &ghasConstraints, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)dm)));
  if (!ghasConstraints) {
    *odm = dm;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!dm->dmBC) {
    PetscSection newSection, gsection;
    PetscSF      sf;

    PetscCall(DMClone(dm, &dm->dmBC));
    PetscCall(DMCopyDisc(dm, dm->dmBC));
    PetscCall(PetscSectionClone(section, &newSection));
    PetscCall(DMSetLocalSection(dm->dmBC, newSection));
    PetscCall(PetscSectionDestroy(&newSection));
    PetscCall(DMGetPointSF(dm->dmBC, &sf));
    PetscCall(PetscSectionCreateGlobalSection(section, sf, PETSC_TRUE, PETSC_FALSE, &gsection));
    PetscCall(DMSetGlobalSection(dm->dmBC, gsection));
    PetscCall(PetscSectionDestroy(&gsection));
  }
  *odm = dm->dmBC;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetOutputSequenceNumber - Retrieve the sequence number/value for output

  Input Parameter:
. dm - The original `DM`

  Output Parameters:
+ num - The output sequence number
- val - The output sequence value

  Level: intermediate

  Note:
  This is intended for output that should appear in sequence, for instance
  a set of timesteps in an `PETSCVIEWERHDF5` file, or a set of realizations of a stochastic system.

  Developer Note:
  The `DM` serves as a convenient place to store the current iteration value. The iteration is not
  not directly related to the `DM`.

.seealso: [](chapter_dmbase), `DM`, `VecView()`
@*/
PetscErrorCode DMGetOutputSequenceNumber(DM dm, PetscInt *num, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (num) {
    PetscValidIntPointer(num, 2);
    *num = dm->outputSequenceNum;
  }
  if (val) {
    PetscValidRealPointer(val, 3);
    *val = dm->outputSequenceVal;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetOutputSequenceNumber - Set the sequence number/value for output

  Input Parameters:
+ dm - The original `DM`
. num - The output sequence number
- val - The output sequence value

  Level: intermediate

  Note:
  This is intended for output that should appear in sequence, for instance
  a set of timesteps in an `PETSCVIEWERHDF5` file, or a set of realizations of a stochastic system.

.seealso: [](chapter_dmbase), `DM`, `VecView()`
@*/
PetscErrorCode DMSetOutputSequenceNumber(DM dm, PetscInt num, PetscReal val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->outputSequenceNum = num;
  dm->outputSequenceVal = val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
 DMOutputSequenceLoad - Retrieve the sequence value from a `PetscViewer`

  Input Parameters:
+ dm   - The original `DM`
. name - The sequence name
- num  - The output sequence number

  Output Parameter:
. val  - The output sequence value

  Level: intermediate

  Note:
  This is intended for output that should appear in sequence, for instance
  a set of timesteps in an `PETSCVIEWERHDF5` file, or a set of realizations of a stochastic system.

  Developer Note:
  It is unclear at the user API level why a `DM` is needed as input

.seealso: [](chapter_dmbase), `DM`, `DMGetOutputSequenceNumber()`, `DMSetOutputSequenceNumber()`, `VecView()`
@*/
PetscErrorCode DMOutputSequenceLoad(DM dm, PetscViewer viewer, const char *name, PetscInt num, PetscReal *val)
{
  PetscBool ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidRealPointer(val, 5);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscScalar value;

    PetscCall(DMSequenceLoad_HDF5_Internal(dm, name, num, &value, viewer));
    *val = PetscRealPart(value);
#endif
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerHDF5Open()");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetUseNatural - Get the flag for creating a mapping to the natural order when a `DM` is (re)distributed in parallel

  Not Collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. useNatural - `PETSC_TRUE` to build the mapping to a natural order during distribution

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMSetUseNatural()`, `DMCreate()`
@*/
PetscErrorCode DMGetUseNatural(DM dm, PetscBool *useNatural)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(useNatural, 2);
  *useNatural = dm->useNatural;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetUseNatural - Set the flag for creating a mapping to the natural order when a `DM` is (re)distributed in parallel

  Collective

  Input Parameters:
 + dm - The `DM`
- useNatural - `PETSC_TRUE` to build the mapping to a natural order during distribution

  Note:
  This also causes the map to be build after `DMCreateSubDM()` and `DMCreateSuperDM()`

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMGetUseNatural()`, `DMCreate()`, `DMPlexDistribute()`, `DMCreateSubDM()`, `DMCreateSuperDM()`
@*/
PetscErrorCode DMSetUseNatural(DM dm, PetscBool useNatural)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(dm, useNatural, 2);
  dm->useNatural = useNatural;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateLabel - Create a label of the given name if it does not already exist in the `DM`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - The label name

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLabelCreate()`, `DMHasLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMCreateLabel(DM dm, const char name[])
{
  PetscBool flg;
  DMLabel   label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscCall(DMHasLabel(dm, name, &flg));
  if (!flg) {
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, name, &label));
    PetscCall(DMAddLabel(dm, label));
    PetscCall(DMLabelDestroy(&label));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCreateLabelAtIndex - Create a label of the given name at the given index. If it already exists in the `DM`, move it to this index.

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
. l    - The index for the label
- name - The label name

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMCreateLabel()`, `DMLabelCreate()`, `DMHasLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMCreateLabelAtIndex(DM dm, PetscInt l, const char name[])
{
  DMLabelLink orig, prev = NULL;
  DMLabel     label;
  PetscInt    Nl, m;
  PetscBool   flg, match;
  const char *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 3);
  PetscCall(DMHasLabel(dm, name, &flg));
  if (!flg) {
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, name, &label));
    PetscCall(DMAddLabel(dm, label));
    PetscCall(DMLabelDestroy(&label));
  }
  PetscCall(DMGetNumLabels(dm, &Nl));
  PetscCheck(l < Nl, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label index %" PetscInt_FMT " must be in [0, %" PetscInt_FMT ")", l, Nl);
  for (m = 0, orig = dm->labels; m < Nl; ++m, prev = orig, orig = orig->next) {
    PetscCall(PetscObjectGetName((PetscObject)orig->label, &lname));
    PetscCall(PetscStrcmp(name, lname, &match));
    if (match) break;
  }
  if (m == l) PetscFunctionReturn(PETSC_SUCCESS);
  if (!m) dm->labels = orig->next;
  else prev->next = orig->next;
  if (!l) {
    orig->next = dm->labels;
    dm->labels = orig;
  } else {
    for (m = 0, prev = dm->labels; m < l - 1; ++m, prev = prev->next)
      ;
    orig->next = prev->next;
    prev->next = orig;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabelValue - Get the value in a `DMLabel` for the given point, with -1 as the default

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
. name - The label name
- point - The mesh point

  Output Parameter:
. value - The label value for this point, or -1 if the point is not in the label

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabelGetValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMGetLabelValue(DM dm, const char name[], PetscInt point, PetscInt *value)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCheck(label, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No label named %s was found", name);
  PetscCall(DMLabelGetValue(label, point, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSetLabelValue - Add a point to a `DMLabel` with given value

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Output Parameter:

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabelSetValue()`, `DMGetStratumIS()`, `DMClearLabelValue()`
@*/
PetscErrorCode DMSetLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscCall(DMGetLabel(dm, name, &label));
  if (!label) {
    PetscCall(DMCreateLabel(dm, name));
    PetscCall(DMGetLabel(dm, name, &label));
  }
  PetscCall(DMLabelSetValue(label, point, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMClearLabelValue - Remove a point from a `DMLabel` with given value

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
. name - The label name
. point - The mesh point
- value - The label value for this point

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabelClearValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMClearLabelValue(DM dm, const char name[], PetscInt point, PetscInt value)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscCall(DMGetLabel(dm, name, &label));
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelClearValue(label, point, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabelSize - Get the value of `DMLabelGetNumValues()` of a `DMLabel` in the `DM`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - The label name

  Output Parameter:
. size - The number of different integer ids, or 0 if the label does not exist

  Level: beginner

  Developer Note:
  This should be renamed to something like `DMGetLabelNumValues()` or removed.

.seealso: [](chapter_dmbase), `DM`, `DMLabelGetNumValues()`, `DMSetLabelValue()`, `DMGetLabel()`
@*/
PetscErrorCode DMGetLabelSize(DM dm, const char name[], PetscInt *size)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(size, 3);
  PetscCall(DMGetLabel(dm, name, &label));
  *size = 0;
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelGetNumValues(label, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabelIdIS - Get the `DMLabelGetValueIS()` from a `DMLabel` in the `DM`

  Not Collective

  Input Parameters:
+ mesh - The `DM` object
- name - The label name

  Output Parameter:
. ids - The integer ids, or `NULL` if the label does not exist

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabelGetValueIS()`, `DMGetLabelSize()`
@*/
PetscErrorCode DMGetLabelIdIS(DM dm, const char name[], IS *ids)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(ids, 3);
  PetscCall(DMGetLabel(dm, name, &label));
  *ids = NULL;
  if (label) {
    PetscCall(DMLabelGetValueIS(label, ids));
  } else {
    /* returning an empty IS */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_USE_POINTER, ids));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetStratumSize - Get the number of points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The `DM` object
. name - The label name
- value - The stratum value

  Output Parameter:
. size - The number of points, also called the stratum size

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabelGetStratumSize()`, `DMGetLabelSize()`, `DMGetLabelIds()`
@*/
PetscErrorCode DMGetStratumSize(DM dm, const char name[], PetscInt value, PetscInt *size)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(size, 4);
  PetscCall(DMGetLabel(dm, name, &label));
  *size = 0;
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelGetStratumSize(label, value, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetStratumIS - Get the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The `DM` object
. name - The label name
- value - The stratum value

  Output Parameter:
. points - The stratum points, or `NULL` if the label does not exist or does not have that value

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabelGetStratumIS()`, `DMGetStratumSize()`
@*/
PetscErrorCode DMGetStratumIS(DM dm, const char name[], PetscInt value, IS *points)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(points, 4);
  PetscCall(DMGetLabel(dm, name, &label));
  *points = NULL;
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelGetStratumIS(label, value, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSetStratumIS - Set the points in a label stratum

  Not Collective

  Input Parameters:
+ dm - The `DM` object
. name - The label name
. value - The stratum value
- points - The stratum points

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMClearLabelStratum()`, `DMLabelClearStratum()`, `DMLabelSetStratumIS()`, `DMGetStratumSize()`
@*/
PetscErrorCode DMSetStratumIS(DM dm, const char name[], PetscInt value, IS points)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(points, 4);
  PetscCall(DMGetLabel(dm, name, &label));
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelSetStratumIS(label, value, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMClearLabelStratum - Remove all points from a stratum from a `DMLabel`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
. name - The label name
- value - The label value for this point

  Output Parameter:

  Level: beginner

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMLabelClearStratum()`, `DMSetLabelValue()`, `DMGetStratumIS()`, `DMClearLabelValue()`
@*/
PetscErrorCode DMClearLabelStratum(DM dm, const char name[], PetscInt value)
{
  DMLabel label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscCall(DMGetLabel(dm, name, &label));
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelClearStratum(label, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNumLabels - Return the number of labels defined by on the `DM`

  Not Collective

  Input Parameter:
. dm   - The `DM` object

  Output Parameter:
. numLabels - the number of Labels

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMGetLabelByNum()`, `DMGetLabelName()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMGetNumLabels(DM dm, PetscInt *numLabels)
{
  DMLabelLink next = dm->labels;
  PetscInt    n    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(numLabels, 2);
  while (next) {
    ++n;
    next = next->next;
  }
  *numLabels = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabelName - Return the name of nth label

  Not Collective

  Input Parameters:
+ dm - The `DM` object
- n  - the label number

  Output Parameter:
. name - the label name

  Level: intermediate

  Developer Note:
  Some of the functions that appropriate on labels using their number have the suffix ByNum, others do not.

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMGetLabelByNum()`, `DMGetLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMGetLabelName(DM dm, PetscInt n, const char **name)
{
  DMLabelLink next = dm->labels;
  PetscInt    l    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(name, 3);
  while (next) {
    if (l == n) {
      PetscCall(PetscObjectGetName((PetscObject)next->label, name));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    ++l;
    next = next->next;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %" PetscInt_FMT " does not exist in this DM", n);
}

/*@C
  DMHasLabel - Determine whether the `DM` has a label of a given name

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - The label name

  Output Parameter:
. hasLabel - `PETSC_TRUE` if the label is present

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMGetLabel()`, `DMGetLabelByNum()`, `DMCreateLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMHasLabel(DM dm, const char name[], PetscBool *hasLabel)
{
  DMLabelLink next = dm->labels;
  const char *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidBoolPointer(hasLabel, 3);
  *hasLabel = PETSC_FALSE;
  while (next) {
    PetscCall(PetscObjectGetName((PetscObject)next->label, &lname));
    PetscCall(PetscStrcmp(name, lname, hasLabel));
    if (*hasLabel) break;
    next = next->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabel - Return the label of a given name, or `NULL`, from a `DM`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - The label name

  Output Parameter:
. label - The `DMLabel`, or `NULL` if the label is absent

  Default labels in a `DMPLEX`:
+   "depth"       - Holds the depth (co-dimension) of each mesh point
.   "celltype"    - Holds the topological type of each cell
.   "ghost"       - If the DM is distributed with overlap, this marks the cells and faces in the overlap
.   "Cell Sets"   - Mirrors the cell sets defined by GMsh and ExodusII
.   "Face Sets"   - Mirrors the face sets defined by GMsh and ExodusII
-  "Vertex Sets" - Mirrors the vertex sets defined by GMsh

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMHasLabel()`, `DMGetLabelByNum()`, `DMAddLabel()`, `DMCreateLabel()`, `DMHasLabel()`, `DMPlexGetDepthLabel()`, `DMPlexGetCellType()`
@*/
PetscErrorCode DMGetLabel(DM dm, const char name[], DMLabel *label)
{
  DMLabelLink next = dm->labels;
  PetscBool   hasLabel;
  const char *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidPointer(label, 3);
  *label = NULL;
  while (next) {
    PetscCall(PetscObjectGetName((PetscObject)next->label, &lname));
    PetscCall(PetscStrcmp(name, lname, &hasLabel));
    if (hasLabel) {
      *label = next->label;
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabelByNum - Return the nth label on a `DM`

  Not Collective

  Input Parameters:
+ dm - The `DM` object
- n  - the label number

  Output Parameter:
. label - the label

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMAddLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMGetLabelByNum(DM dm, PetscInt n, DMLabel *label)
{
  DMLabelLink next = dm->labels;
  PetscInt    l    = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 3);
  while (next) {
    if (l == n) {
      *label = next->label;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    ++l;
    next = next->next;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %" PetscInt_FMT " does not exist in this DM", n);
}

/*@C
  DMAddLabel - Add the label to this `DM`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- label - The `DMLabel`

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMLabel`, `DMCreateLabel()`, `DMHasLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMAddLabel(DM dm, DMLabel label)
{
  DMLabelLink l, *p, tmpLabel;
  PetscBool   hasLabel;
  const char *lname;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectGetName((PetscObject)label, &lname));
  PetscCall(DMHasLabel(dm, lname, &hasLabel));
  PetscCheck(!hasLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in this DM", lname);
  PetscCall(PetscCalloc1(1, &tmpLabel));
  tmpLabel->label  = label;
  tmpLabel->output = PETSC_TRUE;
  for (p = &dm->labels; (l = *p); p = &l->next) { }
  *p = tmpLabel;
  PetscCall(PetscObjectReference((PetscObject)label));
  PetscCall(PetscStrcmp(lname, "depth", &flg));
  if (flg) dm->depthLabel = label;
  PetscCall(PetscStrcmp(lname, "celltype", &flg));
  if (flg) dm->celltypeLabel = label;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSetLabel - Replaces the label of a given name, or ignores it if the name is not present

  Not Collective

  Input Parameters:
+ dm    - The `DM` object
- label - The `DMLabel`, having the same name, to substitute

  Default labels in a `DMPLEX`:
+  "depth"       - Holds the depth (co-dimension) of each mesh point
.  "celltype"    - Holds the topological type of each cell
.  "ghost"       - If the DM is distributed with overlap, this marks the cells and faces in the overlap
.  "Cell Sets"   - Mirrors the cell sets defined by GMsh and ExodusII
.  "Face Sets"   - Mirrors the face sets defined by GMsh and ExodusII
- "Vertex Sets" - Mirrors the vertex sets defined by GMsh

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMCreateLabel()`, `DMHasLabel()`, `DMPlexGetDepthLabel()`, `DMPlexGetCellType()`
@*/
PetscErrorCode DMSetLabel(DM dm, DMLabel label)
{
  DMLabelLink next = dm->labels;
  PetscBool   hasLabel, flg;
  const char *name, *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  PetscCall(PetscObjectGetName((PetscObject)label, &name));
  while (next) {
    PetscCall(PetscObjectGetName((PetscObject)next->label, &lname));
    PetscCall(PetscStrcmp(name, lname, &hasLabel));
    if (hasLabel) {
      PetscCall(PetscObjectReference((PetscObject)label));
      PetscCall(PetscStrcmp(lname, "depth", &flg));
      if (flg) dm->depthLabel = label;
      PetscCall(PetscStrcmp(lname, "celltype", &flg));
      if (flg) dm->celltypeLabel = label;
      PetscCall(DMLabelDestroy(&next->label));
      next->label = label;
      break;
    }
    next = next->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMRemoveLabel - Remove the label given by name from this `DM`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - The label name

  Output Parameter:
. label - The `DMLabel`, or `NULL` if the label is absent. Pass in `NULL` to call `DMLabelDestroy()` on the label, otherwise the
          caller is responsible for calling `DMLabelDestroy()`.

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMCreateLabel()`, `DMHasLabel()`, `DMGetLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMLabelDestroy()`, `DMRemoveLabelBySelf()`
@*/
PetscErrorCode DMRemoveLabel(DM dm, const char name[], DMLabel *label)
{
  DMLabelLink link, *pnext;
  PetscBool   hasLabel;
  const char *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (label) {
    PetscValidPointer(label, 3);
    *label = NULL;
  }
  for (pnext = &dm->labels; (link = *pnext); pnext = &link->next) {
    PetscCall(PetscObjectGetName((PetscObject)link->label, &lname));
    PetscCall(PetscStrcmp(name, lname, &hasLabel));
    if (hasLabel) {
      *pnext = link->next; /* Remove from list */
      PetscCall(PetscStrcmp(name, "depth", &hasLabel));
      if (hasLabel) dm->depthLabel = NULL;
      PetscCall(PetscStrcmp(name, "celltype", &hasLabel));
      if (hasLabel) dm->celltypeLabel = NULL;
      if (label) *label = link->label;
      else PetscCall(DMLabelDestroy(&link->label));
      PetscCall(PetscFree(link));
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMRemoveLabelBySelf - Remove the label from this `DM`

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
. label - The `DMLabel` to be removed from the `DM`
- failNotFound - Should it fail if the label is not found in the `DM`?

  Level: developer

  Note:
  Only exactly the same instance is removed if found, name match is ignored.
  If the `DM` has an exclusive reference to the label, the label gets destroyed and
  *label nullified.

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMCreateLabel()`, `DMHasLabel()`, `DMGetLabel()` `DMGetLabelValue()`, `DMSetLabelValue()`, `DMLabelDestroy()`, `DMRemoveLabel()`
@*/
PetscErrorCode DMRemoveLabelBySelf(DM dm, DMLabel *label, PetscBool failNotFound)
{
  DMLabelLink link, *pnext;
  PetscBool   hasLabel = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 2);
  if (!*label && !failNotFound) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*label, DMLABEL_CLASSID, 2);
  PetscValidLogicalCollectiveBool(dm, failNotFound, 3);
  for (pnext = &dm->labels; (link = *pnext); pnext = &link->next) {
    if (*label == link->label) {
      hasLabel = PETSC_TRUE;
      *pnext   = link->next; /* Remove from list */
      if (*label == dm->depthLabel) dm->depthLabel = NULL;
      if (*label == dm->celltypeLabel) dm->celltypeLabel = NULL;
      if (((PetscObject)link->label)->refct < 2) *label = NULL; /* nullify if exclusive reference */
      PetscCall(DMLabelDestroy(&link->label));
      PetscCall(PetscFree(link));
      break;
    }
  }
  PetscCheck(hasLabel || !failNotFound, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Given label not found in DM");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetLabelOutput - Get the output flag for a given label

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - The label name

  Output Parameter:
. output - The flag for output

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMSetLabelOutput()`, `DMCreateLabel()`, `DMHasLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMGetLabelOutput(DM dm, const char name[], PetscBool *output)
{
  DMLabelLink next = dm->labels;
  const char *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  PetscValidBoolPointer(output, 3);
  while (next) {
    PetscBool flg;

    PetscCall(PetscObjectGetName((PetscObject)next->label, &lname));
    PetscCall(PetscStrcmp(name, lname, &flg));
    if (flg) {
      *output = next->output;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    next = next->next;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this dm", name);
}

/*@C
  DMSetLabelOutput - Set if a given label should be saved to a `PetscViewer` in calls to `DMView()`

  Not Collective

  Input Parameters:
+ dm     - The `DM` object
. name   - The label name
- output - `PETSC_TRUE` to save the label to the viewer

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMGetOutputFlag()`, `DMGetLabelOutput()`, `DMCreateLabel()`, `DMHasLabel()`, `DMGetLabelValue()`, `DMSetLabelValue()`, `DMGetStratumIS()`
@*/
PetscErrorCode DMSetLabelOutput(DM dm, const char name[], PetscBool output)
{
  DMLabelLink next = dm->labels;
  const char *lname;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  while (next) {
    PetscBool flg;

    PetscCall(PetscObjectGetName((PetscObject)next->label, &lname));
    PetscCall(PetscStrcmp(name, lname, &flg));
    if (flg) {
      next->output = output;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    next = next->next;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No label named %s was present in this dm", name);
}

/*@
  DMCopyLabels - Copy labels from one `DM` mesh to another `DM` with a superset of the points

  Collective

  Input Parameters:
+ dmA - The `DM` object with initial labels
. dmB - The `DM` object to which labels are copied
. mode - Copy labels by pointers (`PETSC_OWN_POINTER`) or duplicate them (`PETSC_COPY_VALUES`)
. all  - Copy all labels including "depth", "dim", and "celltype" (`PETSC_TRUE`) which are otherwise ignored (`PETSC_FALSE`)
- emode - How to behave when a `DMLabel` in the source and destination `DM`s with the same name is encountered (see `DMCopyLabelsMode`)

  Level: intermediate

  Note:
  This is typically used when interpolating or otherwise adding to a mesh, or testing.

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMAddLabel()`, `DMCopyLabelsMode`
@*/
PetscErrorCode DMCopyLabels(DM dmA, DM dmB, PetscCopyMode mode, PetscBool all, DMCopyLabelsMode emode)
{
  DMLabel     label, labelNew, labelOld;
  const char *name;
  PetscBool   flg;
  DMLabelLink link;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  PetscValidLogicalCollectiveEnum(dmA, mode, 3);
  PetscValidLogicalCollectiveBool(dmA, all, 4);
  PetscCheck(mode != PETSC_USE_POINTER, PetscObjectComm((PetscObject)dmA), PETSC_ERR_SUP, "PETSC_USE_POINTER not supported for objects");
  if (dmA == dmB) PetscFunctionReturn(PETSC_SUCCESS);
  for (link = dmA->labels; link; link = link->next) {
    label = link->label;
    PetscCall(PetscObjectGetName((PetscObject)label, &name));
    if (!all) {
      PetscCall(PetscStrcmp(name, "depth", &flg));
      if (flg) continue;
      PetscCall(PetscStrcmp(name, "dim", &flg));
      if (flg) continue;
      PetscCall(PetscStrcmp(name, "celltype", &flg));
      if (flg) continue;
    }
    PetscCall(DMGetLabel(dmB, name, &labelOld));
    if (labelOld) {
      switch (emode) {
      case DM_COPY_LABELS_KEEP:
        continue;
      case DM_COPY_LABELS_REPLACE:
        PetscCall(DMRemoveLabelBySelf(dmB, &labelOld, PETSC_TRUE));
        break;
      case DM_COPY_LABELS_FAIL:
        SETERRQ(PetscObjectComm((PetscObject)dmA), PETSC_ERR_ARG_OUTOFRANGE, "Label %s already exists in destination DM", name);
      default:
        SETERRQ(PetscObjectComm((PetscObject)dmA), PETSC_ERR_ARG_OUTOFRANGE, "Unhandled DMCopyLabelsMode %d", (int)emode);
      }
    }
    if (mode == PETSC_COPY_VALUES) {
      PetscCall(DMLabelDuplicate(label, &labelNew));
    } else {
      labelNew = label;
    }
    PetscCall(DMAddLabel(dmB, labelNew));
    if (mode == PETSC_COPY_VALUES) PetscCall(DMLabelDestroy(&labelNew));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCompareLabels - Compare labels of two `DMPLEX` meshes

  Collective; No Fortran Support

  Input Parameters:
+ dm0 - First `DM` object
- dm1 - Second `DM` object

  Output Parameters
+ equal   - (Optional) Flag whether labels of dm0 and dm1 are the same
- message - (Optional) Message describing the difference, or `NULL` if there is no difference

  Level: intermediate

  Notes:
  The output flag equal will be the same on all processes.

  If equal is passed as `NULL` and difference is found, an error is thrown on all processes.

  Make sure to pass equal is `NULL` on all processes or none of them.

  The output message is set independently on each rank.

  message must be freed with `PetscFree()`

  If message is passed as `NULL` and a difference is found, the difference description is printed to stderr in synchronized manner.

  Make sure to pass message as `NULL` on all processes or no processes.

  Labels are matched by name. If the number of labels and their names are equal,
  `DMLabelCompare()` is used to compare each pair of labels with the same name.

.seealso: [](chapter_dmbase), `DM`, `DMLabel`, `DMAddLabel()`, `DMCopyLabelsMode`, `DMLabelCompare()`
@*/
PetscErrorCode DMCompareLabels(DM dm0, DM dm1, PetscBool *equal, char **message)
{
  PetscInt    n, i;
  char        msg[PETSC_MAX_PATH_LEN] = "";
  PetscBool   eq;
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm0, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm1, DM_CLASSID, 2);
  PetscCheckSameComm(dm0, 1, dm1, 2);
  if (equal) PetscValidBoolPointer(equal, 3);
  if (message) PetscValidPointer(message, 4);
  PetscCall(PetscObjectGetComm((PetscObject)dm0, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  {
    PetscInt n1;

    PetscCall(DMGetNumLabels(dm0, &n));
    PetscCall(DMGetNumLabels(dm1, &n1));
    eq = (PetscBool)(n == n1);
    if (!eq) PetscCall(PetscSNPrintf(msg, sizeof(msg), "Number of labels in dm0 = %" PetscInt_FMT " != %" PetscInt_FMT " = Number of labels in dm1", n, n1));
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
    if (!eq) goto finish;
  }
  for (i = 0; i < n; i++) {
    DMLabel     l0, l1;
    const char *name;
    char       *msgInner;

    /* Ignore label order */
    PetscCall(DMGetLabelByNum(dm0, i, &l0));
    PetscCall(PetscObjectGetName((PetscObject)l0, &name));
    PetscCall(DMGetLabel(dm1, name, &l1));
    if (!l1) {
      PetscCall(PetscSNPrintf(msg, sizeof(msg), "Label \"%s\" (#%" PetscInt_FMT " in dm0) not found in dm1", name, i));
      eq = PETSC_FALSE;
      break;
    }
    PetscCall(DMLabelCompare(comm, l0, l1, &eq, &msgInner));
    PetscCall(PetscStrncpy(msg, msgInner, sizeof(msg)));
    PetscCall(PetscFree(msgInner));
    if (!eq) break;
  }
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &eq, 1, MPIU_BOOL, MPI_LAND, comm));
finish:
  /* If message output arg not set, print to stderr */
  if (message) {
    *message = NULL;
    if (msg[0]) PetscCall(PetscStrallocpy(msg, message));
  } else {
    if (msg[0]) PetscCall(PetscSynchronizedFPrintf(comm, PETSC_STDERR, "[%d] %s\n", rank, msg));
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDERR));
  }
  /* If same output arg not ser and labels are not equal, throw error */
  if (equal) *equal = eq;
  else PetscCheck(eq, comm, PETSC_ERR_ARG_INCOMP, "DMLabels are not the same in dm0 and dm1");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSetLabelValue_Fast(DM dm, DMLabel *label, const char name[], PetscInt point, PetscInt value)
{
  PetscFunctionBegin;
  PetscValidPointer(label, 2);
  if (!*label) {
    PetscCall(DMCreateLabel(dm, name));
    PetscCall(DMGetLabel(dm, name, label));
  }
  PetscCall(DMLabelSetValue(*label, point, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Many mesh programs, such as Triangle and TetGen, allow only a single label for each mesh point. Therefore, we would
  like to encode all label IDs using a single, universal label. We can do this by assigning an integer to every
  (label, id) pair in the DM.

  However, a mesh point can have multiple labels, so we must separate all these values. We will assign a bit range to
  each label.
*/
PetscErrorCode DMUniversalLabelCreate(DM dm, DMUniversalLabel *universal)
{
  DMUniversalLabel ul;
  PetscBool       *active;
  PetscInt         pStart, pEnd, p, Nl, l, m;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(1, &ul));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "universal", &ul->label));
  PetscCall(DMGetNumLabels(dm, &Nl));
  PetscCall(PetscCalloc1(Nl, &active));
  ul->Nl = 0;
  for (l = 0; l < Nl; ++l) {
    PetscBool   isdepth, iscelltype;
    const char *name;

    PetscCall(DMGetLabelName(dm, l, &name));
    PetscCall(PetscStrncmp(name, "depth", 6, &isdepth));
    PetscCall(PetscStrncmp(name, "celltype", 9, &iscelltype));
    active[l] = !(isdepth || iscelltype) ? PETSC_TRUE : PETSC_FALSE;
    if (active[l]) ++ul->Nl;
  }
  PetscCall(PetscCalloc5(ul->Nl, &ul->names, ul->Nl, &ul->indices, ul->Nl + 1, &ul->offsets, ul->Nl + 1, &ul->bits, ul->Nl, &ul->masks));
  ul->Nv = 0;
  for (l = 0, m = 0; l < Nl; ++l) {
    DMLabel     label;
    PetscInt    nv;
    const char *name;

    if (!active[l]) continue;
    PetscCall(DMGetLabelName(dm, l, &name));
    PetscCall(DMGetLabelByNum(dm, l, &label));
    PetscCall(DMLabelGetNumValues(label, &nv));
    PetscCall(PetscStrallocpy(name, &ul->names[m]));
    ul->indices[m] = l;
    ul->Nv += nv;
    ul->offsets[m + 1] = nv;
    ul->bits[m + 1]    = PetscCeilReal(PetscLog2Real(nv + 1));
    ++m;
  }
  for (l = 1; l <= ul->Nl; ++l) {
    ul->offsets[l] = ul->offsets[l - 1] + ul->offsets[l];
    ul->bits[l]    = ul->bits[l - 1] + ul->bits[l];
  }
  for (l = 0; l < ul->Nl; ++l) {
    PetscInt b;

    ul->masks[l] = 0;
    for (b = ul->bits[l]; b < ul->bits[l + 1]; ++b) ul->masks[l] |= 1 << b;
  }
  PetscCall(PetscMalloc1(ul->Nv, &ul->values));
  for (l = 0, m = 0; l < Nl; ++l) {
    DMLabel         label;
    IS              valueIS;
    const PetscInt *varr;
    PetscInt        nv, v;

    if (!active[l]) continue;
    PetscCall(DMGetLabelByNum(dm, l, &label));
    PetscCall(DMLabelGetNumValues(label, &nv));
    PetscCall(DMLabelGetValueIS(label, &valueIS));
    PetscCall(ISGetIndices(valueIS, &varr));
    for (v = 0; v < nv; ++v) ul->values[ul->offsets[m] + v] = varr[v];
    PetscCall(ISRestoreIndices(valueIS, &varr));
    PetscCall(ISDestroy(&valueIS));
    PetscCall(PetscSortInt(nv, &ul->values[ul->offsets[m]]));
    ++m;
  }
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  uval   = 0;
    PetscBool marked = PETSC_FALSE;

    for (l = 0, m = 0; l < Nl; ++l) {
      DMLabel  label;
      PetscInt val, defval, loc, nv;

      if (!active[l]) continue;
      PetscCall(DMGetLabelByNum(dm, l, &label));
      PetscCall(DMLabelGetValue(label, p, &val));
      PetscCall(DMLabelGetDefaultValue(label, &defval));
      if (val == defval) {
        ++m;
        continue;
      }
      nv     = ul->offsets[m + 1] - ul->offsets[m];
      marked = PETSC_TRUE;
      PetscCall(PetscFindInt(val, nv, &ul->values[ul->offsets[m]], &loc));
      PetscCheck(loc >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Label value %" PetscInt_FMT " not found in compression array", val);
      uval += (loc + 1) << ul->bits[m];
      ++m;
    }
    if (marked) PetscCall(DMLabelSetValue(ul->label, p, uval));
  }
  PetscCall(PetscFree(active));
  *universal = ul;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMUniversalLabelDestroy(DMUniversalLabel *universal)
{
  PetscInt l;

  PetscFunctionBegin;
  for (l = 0; l < (*universal)->Nl; ++l) PetscCall(PetscFree((*universal)->names[l]));
  PetscCall(DMLabelDestroy(&(*universal)->label));
  PetscCall(PetscFree5((*universal)->names, (*universal)->indices, (*universal)->offsets, (*universal)->bits, (*universal)->masks));
  PetscCall(PetscFree((*universal)->values));
  PetscCall(PetscFree(*universal));
  *universal = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMUniversalLabelGetLabel(DMUniversalLabel ul, DMLabel *ulabel)
{
  PetscFunctionBegin;
  PetscValidPointer(ulabel, 2);
  *ulabel = ul->label;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMUniversalLabelCreateLabels(DMUniversalLabel ul, PetscBool preserveOrder, DM dm)
{
  PetscInt Nl = ul->Nl, l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 3);
  for (l = 0; l < Nl; ++l) {
    if (preserveOrder) PetscCall(DMCreateLabelAtIndex(dm, ul->indices[l], ul->names[l]));
    else PetscCall(DMCreateLabel(dm, ul->names[l]));
  }
  if (preserveOrder) {
    for (l = 0; l < ul->Nl; ++l) {
      const char *name;
      PetscBool   match;

      PetscCall(DMGetLabelName(dm, ul->indices[l], &name));
      PetscCall(PetscStrcmp(name, ul->names[l], &match));
      PetscCheck(match, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Label %" PetscInt_FMT " name %s does not match new name %s", l, name, ul->names[l]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMUniversalLabelSetLabelValue(DMUniversalLabel ul, DM dm, PetscBool useIndex, PetscInt p, PetscInt value)
{
  PetscInt l;

  PetscFunctionBegin;
  for (l = 0; l < ul->Nl; ++l) {
    DMLabel  label;
    PetscInt lval = (value & ul->masks[l]) >> ul->bits[l];

    if (lval) {
      if (useIndex) PetscCall(DMGetLabelByNum(dm, ul->indices[l], &label));
      else PetscCall(DMGetLabel(dm, ul->names[l], &label));
      PetscCall(DMLabelSetValue(label, p, ul->values[ul->offsets[l] + lval - 1]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoarseDM - Get the coarse `DM`from which this `DM` was obtained by refinement

  Not Collective

  Input Parameter:
. dm - The `DM` object

  Output Parameter:
. cdm - The coarse `DM`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMSetCoarseDM()`, `DMCoarsen()`
@*/
PetscErrorCode DMGetCoarseDM(DM dm, DM *cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cdm, 2);
  *cdm = dm->coarseMesh;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCoarseDM - Set the coarse `DM` from which this `DM` was obtained by refinement

  Input Parameters:
+ dm - The `DM` object
- cdm - The coarse `DM`

  Level: intermediate

  Note:
  Normally this is set automatically by `DMRefine()`

.seealso: [](chapter_dmbase), `DM`, `DMGetCoarseDM()`, `DMCoarsen()`, `DMSetRefine()`, `DMSetFineDM()`
@*/
PetscErrorCode DMSetCoarseDM(DM dm, DM cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cdm) PetscValidHeaderSpecific(cdm, DM_CLASSID, 2);
  if (dm == cdm) cdm = NULL;
  PetscCall(PetscObjectReference((PetscObject)cdm));
  PetscCall(DMDestroy(&dm->coarseMesh));
  dm->coarseMesh = cdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetFineDM - Get the fine mesh from which this `DM` was obtained by coarsening

  Input Parameter:
. dm - The `DM` object

  Output Parameter:
. fdm - The fine `DM`

  Level: intermediate

.seealso: [](chapter_dmbase), `DM`, `DMSetFineDM()`, `DMCoarsen()`, `DMRefine()`
@*/
PetscErrorCode DMGetFineDM(DM dm, DM *fdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(fdm, 2);
  *fdm = dm->fineMesh;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetFineDM - Set the fine mesh from which this was obtained by coarsening

  Input Parameters:
+ dm - The `DM` object
- fdm - The fine `DM`

  Level: developer

  Note:
  Normally this is set automatically by `DMCoarsen()`

.seealso: [](chapter_dmbase), `DM`, `DMGetFineDM()`, `DMCoarsen()`, `DMRefine()`
@*/
PetscErrorCode DMSetFineDM(DM dm, DM fdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (fdm) PetscValidHeaderSpecific(fdm, DM_CLASSID, 2);
  if (dm == fdm) fdm = NULL;
  PetscCall(PetscObjectReference((PetscObject)fdm));
  PetscCall(DMDestroy(&dm->fineMesh));
  dm->fineMesh = fdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMAddBoundary - Add a boundary condition to a model represented by a `DM`

  Collective

  Input Parameters:
+ dm       - The `DM`, with a `PetscDS` that matches the problem being constrained
. type     - The type of condition, e.g. `DM_BC_ESSENTIAL_ANALYTIC`, `DM_BC_ESSENTIAL_FIELD` (Dirichlet), or `DM_BC_NATURAL` (Neumann)
. name     - The BC name
. label    - The label defining constrained points
. Nv       - The number of `DMLabel` values for constrained points
. values   - An array of values for constrained points
. field    - The field to constrain
. Nc       - The number of constrained field components (0 will constrain all fields)
. comps    - An array of constrained component numbers
. bcFunc   - A pointwise function giving boundary values
. bcFunc_t - A pointwise function giving the time deriative of the boundary values, or NULL
- ctx      - An optional user context for bcFunc

  Output Parameter:
. bd          - (Optional) Boundary number

  Options Database Keys:
+ -bc_<boundary name> <num> - Overrides the boundary ids
- -bc_<boundary name>_comp <num> - Overrides the boundary components

  Level: intermediate

  Notes:
  Both bcFunc abd bcFunc_t will depend on the boundary condition type. If the type if `DM_BC_ESSENTIAL`, then the calling sequence is:

$ void bcFunc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[])

  If the type is `DM_BC_ESSENTIAL_FIELD` or other _FIELD value, then the calling sequence is:

.vb
  void bcFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
              PetscReal time, const PetscReal x[], PetscScalar bcval[])
.ve
+ dim - the spatial dimension
. Nf - the number of fields
. uOff - the offset into u[] and u_t[] for each field
. uOff_x - the offset into u_x[] for each field
. u - each field evaluated at the current point
. u_t - the time derivative of each field evaluated at the current point
. u_x - the gradient of each field evaluated at the current point
. aOff - the offset into a[] and a_t[] for each auxiliary field
. aOff_x - the offset into a_x[] for each auxiliary field
. a - each auxiliary field evaluated at the current point
. a_t - the time derivative of each auxiliary field evaluated at the current point
. a_x - the gradient of auxiliary each field evaluated at the current point
. t - current time
. x - coordinates of the current point
. numConstants - number of constant parameters
. constants - constant parameters
- bcval - output values at the current point

.seealso: [](chapter_dmbase), `DM`, `DSGetBoundary()`, `PetscDSAddBoundary()`
@*/
PetscErrorCode DMAddBoundary(DM dm, DMBoundaryConditionType type, const char name[], DMLabel label, PetscInt Nv, const PetscInt values[], PetscInt field, PetscInt Nc, const PetscInt comps[], void (*bcFunc)(void), void (*bcFunc_t)(void), void *ctx, PetscInt *bd)
{
  PetscDS ds;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(dm, type, 2);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 4);
  PetscValidLogicalCollectiveInt(dm, Nv, 5);
  PetscValidLogicalCollectiveInt(dm, field, 7);
  PetscValidLogicalCollectiveInt(dm, Nc, 8);
  PetscCheck(!dm->localSection, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Cannot add boundary to DM after creating local section");
  PetscCall(DMGetDS(dm, &ds));
  /* Complete label */
  if (label) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      DM plex;

      PetscCall(DMConvert(dm, DMPLEX, &plex));
      if (plex) PetscCall(DMPlexLabelComplete(plex, label));
      PetscCall(DMDestroy(&plex));
    }
  }
  PetscCall(PetscDSAddBoundary(ds, type, name, label, Nv, values, field, Nc, comps, bcFunc, bcFunc_t, ctx, bd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO Remove this since now the structures are the same */
static PetscErrorCode DMPopulateBoundary(DM dm)
{
  PetscDS     ds;
  DMBoundary *lastnext;
  DSBoundary  dsbound;

  PetscFunctionBegin;
  PetscCall(DMGetDS(dm, &ds));
  dsbound = ds->boundary;
  if (dm->boundary) {
    DMBoundary next = dm->boundary;

    /* quick check to see if the PetscDS has changed */
    if (next->dsboundary == dsbound) PetscFunctionReturn(PETSC_SUCCESS);
    /* the PetscDS has changed: tear down and rebuild */
    while (next) {
      DMBoundary b = next;

      next = b->next;
      PetscCall(PetscFree(b));
    }
    dm->boundary = NULL;
  }

  lastnext = &(dm->boundary);
  while (dsbound) {
    DMBoundary dmbound;

    PetscCall(PetscNew(&dmbound));
    dmbound->dsboundary = dsbound;
    dmbound->label      = dsbound->label;
    /* push on the back instead of the front so that it is in the same order as in the PetscDS */
    *lastnext = dmbound;
    lastnext  = &(dmbound->next);
    dsbound   = dsbound->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: missing manual page */
PetscErrorCode DMIsBoundaryPoint(DM dm, PetscInt point, PetscBool *isBd)
{
  DMBoundary b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(isBd, 3);
  *isBd = PETSC_FALSE;
  PetscCall(DMPopulateBoundary(dm));
  b = dm->boundary;
  while (b && !(*isBd)) {
    DMLabel    label = b->label;
    DSBoundary dsb   = b->dsboundary;
    PetscInt   i;

    if (label) {
      for (i = 0; i < dsb->Nv && !(*isBd); ++i) PetscCall(DMLabelStratumHasPoint(label, dsb->values[i], point, isBd));
    }
    b = b->next;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFunction - This projects the given function into the function space provided by a `DM`, putting the coefficients in a global vector.

  Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

   Calling sequence of `funcs`:
$  PetscErrorCode funcs(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
+  dim - The spatial dimension
.  time - The time at which to sample
.  x   - The coordinates
.  Nc  - The number of components
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunctionLocal()`, `DMProjectFunctionLabel()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFunction(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec X)
{
  Vec localX;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(VecSet(localX, 0.));
  PetscCall(DMProjectFunctionLocal(dm, time, funcs, ctxs, mode, localX));
  PetscCall(DMLocalToGlobalBegin(dm, localX, mode, X));
  PetscCall(DMLocalToGlobalEnd(dm, localX, mode, X));
  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFunctionLocal - This projects the given function into the function space provided by a `DM`, putting the coefficients in a local vector.

  Not Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. localX - vector

   Calling sequence of `funcs`:
$  PetscErrorCode funcs(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
+  dim - The spatial dimension
.  x   - The coordinates
.  Nc  - The number of components
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunction()`, `DMProjectFunctionLabel()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFunctionLocal(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(localX, VEC_CLASSID, 6);
  PetscCall((dm->ops->projectfunctionlocal)(dm, time, funcs, ctxs, mode, localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFunctionLabel - This projects the given function into the function space provided by the `DM`, putting the coefficients in a global vector, setting values only for points in the given label.

  Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. label   - The `DMLabel` selecting the portion of the mesh for projection
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs may be null.
- mode    - The insertion mode for values

  Output Parameter:
. X - vector

   Calling sequence of `funcs`:
$  PetscErrorCode funcs(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
+  dim - The spatial dimension
.  x   - The coordinates
.  Nc  - The number of components
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunction()`, `DMProjectFunctionLocal()`, `DMProjectFunctionLabelLocal()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFunctionLabel(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec X)
{
  Vec localX;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(VecSet(localX, 0.));
  PetscCall(DMProjectFunctionLabelLocal(dm, time, label, numIds, ids, Nc, comps, funcs, ctxs, mode, localX));
  PetscCall(DMLocalToGlobalBegin(dm, localX, mode, X));
  PetscCall(DMLocalToGlobalEnd(dm, localX, mode, X));
  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFunctionLabelLocal - This projects the given function into the function space provided by the `DM`, putting the coefficients in a local vector, setting values only for points in the given label.

  Not Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. label   - The `DMLabel` selecting the portion of the mesh for projection
. funcs   - The coordinate functions to evaluate, one per field
. ctxs    - Optional array of contexts to pass to each coordinate function.  ctxs itself may be null.
- mode    - The insertion mode for values

  Output Parameter:
. localX - vector

   Calling sequence of `funcs`:
$  PetscErrorCode funcs(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx);
+  dim - The spatial dimension
.  x   - The coordinates
.  Nc  - The number of components
.  u   - The output field values
-  ctx - optional user-defined function context

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunction()`, `DMProjectFunctionLocal()`, `DMProjectFunctionLabel()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFunctionLabelLocal(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(localX, VEC_CLASSID, 11);
  PetscCall((dm->ops->projectfunctionlabellocal)(dm, time, label, numIds, ids, Nc, comps, funcs, ctxs, mode, localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFieldLocal - This projects the given function of the input fields into the function space provided by the `DM`, putting the coefficients in a local vector.

  Not Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. localU  - The input field vector; may be `NULL` if projection is defined purely by coordinates
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. localX  - The output vector

   Calling sequence of `funcs`:
.vb
   void funcs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);
.ve
+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note:
  There are three different `DM`s that potentially interact in this function. The output `DM`, dm, specifies the layout of the values calculates by funcs.
  The input `DM`, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary `DM`, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectField()`, `DMProjectFieldLabelLocal()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFieldLocal(DM dm, PetscReal time, Vec localU, void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (localU) PetscValidHeaderSpecific(localU, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(localX, VEC_CLASSID, 6);
  PetscCall((dm->ops->projectfieldlocal)(dm, time, localU, funcs, mode, localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFieldLabelLocal - This projects the given function of the input fields into the function space provided, putting the coefficients in a local vector, calculating only over the portion of the domain specified by the label.

  Not Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. label   - The `DMLabel` marking the portion of the domain to output
. numIds  - The number of label ids to use
. ids     - The label ids to use for marking
. Nc      - The number of components to set in the output, or `PETSC_DETERMINE` for all components
. comps   - The components to set in the output, or `NULL` for all components
. localU  - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. localX  - The output vector

   Calling sequence of `funcs`:
.vb
   void funcs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);
.ve
+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note:
  There are three different `DM`s that potentially interact in this function. The output `DM`, dm, specifies the layout of the values calculates by funcs.
  The input `DM`, attached to localU, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary `DM`, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectField()`, `DMProjectFieldLabel()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFieldLabelLocal(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], Vec localU, void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(localU, VEC_CLASSID, 8);
  PetscValidHeaderSpecific(localX, VEC_CLASSID, 11);
  PetscCall((dm->ops->projectfieldlabellocal)(dm, time, label, numIds, ids, Nc, comps, localU, funcs, mode, localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectFieldLabel - This projects the given function of the input fields into the function space provided, putting the coefficients in a global vector, calculating only over the portion of the domain specified by the label.

  Not Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. label   - The `DMLabel` marking the portion of the domain to output
. numIds  - The number of label ids to use
. ids     - The label ids to use for marking
. Nc      - The number of components to set in the output, or `PETSC_DETERMINE` for all components
. comps   - The components to set in the output, or `NULL` for all components
. U       - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. X       - The output vector

   Calling sequence of `funcs`:
.vb
  void func(PetscInt dim, PetscInt Nf, PetscInt NfAux,
            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
            PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);
.ve
+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note:
  There are three different `DM`s that potentially interact in this function. The output `DM`, dm, specifies the layout of the values calculates by funcs.
  The input `DM`, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary `DM`, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectField()`, `DMProjectFieldLabelLocal()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectFieldLabel(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], Vec U, void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), InsertMode mode, Vec X)
{
  DM  dmIn;
  Vec localU, localX;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(VecGetDM(U, &dmIn));
  PetscCall(DMGetLocalVector(dmIn, &localU));
  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(VecSet(localX, 0.));
  PetscCall(DMGlobalToLocalBegin(dmIn, U, mode, localU));
  PetscCall(DMGlobalToLocalEnd(dmIn, U, mode, localU));
  PetscCall(DMProjectFieldLabelLocal(dm, time, label, numIds, ids, Nc, comps, localU, funcs, mode, localX));
  PetscCall(DMLocalToGlobalBegin(dm, localX, mode, X));
  PetscCall(DMLocalToGlobalEnd(dm, localX, mode, X));
  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(DMRestoreLocalVector(dmIn, &localU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectBdFieldLabelLocal - This projects the given function of the input fields into the function space provided, putting the coefficients in a local vector, calculating only over the portion of the domain boundary specified by the label.

  Not Collective

  Input Parameters:
+ dm      - The `DM`
. time    - The time
. label   - The `DMLabel` marking the portion of the domain boundary to output
. numIds  - The number of label ids to use
. ids     - The label ids to use for marking
. Nc      - The number of components to set in the output, or `PETSC_DETERMINE` for all components
. comps   - The components to set in the output, or `NULL` for all components
. localU  - The input field vector
. funcs   - The functions to evaluate, one per field
- mode    - The insertion mode for values

  Output Parameter:
. localX  - The output vector

   Calling sequence of `funcs`:
.vb
   void funcs(PetscInt dim, PetscInt Nf, PetscInt NfAux,
              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);
.ve
+  dim          - The spatial dimension
.  Nf           - The number of input fields
.  NfAux        - The number of input auxiliary fields
.  uOff         - The offset of each field in u[]
.  uOff_x       - The offset of each field in u_x[]
.  u            - The field values at this point in space
.  u_t          - The field time derivative at this point in space (or NULL)
.  u_x          - The field derivatives at this point in space
.  aOff         - The offset of each auxiliary field in u[]
.  aOff_x       - The offset of each auxiliary field in u_x[]
.  a            - The auxiliary field values at this point in space
.  a_t          - The auxiliary field time derivative at this point in space (or NULL)
.  a_x          - The auxiliary field derivatives at this point in space
.  t            - The current time
.  x            - The coordinates of this point
.  n            - The face normal
.  numConstants - The number of constants
.  constants    - The value of each constant
-  f            - The value of the function at this point in space

  Note:
  There are three different `DM`s that potentially interact in this function. The output `DM`, dm, specifies the layout of the values calculates by funcs.
  The input `DM`, attached to U, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary `DM`, attached to the
  auxiliary field vector, which is attached to dm, can also be different. It can have a different topology, number of fields, and discretizations.

  Level: intermediate

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectField()`, `DMProjectFieldLabelLocal()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectBdFieldLabelLocal(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Nc, const PetscInt comps[], Vec localU, void (**funcs)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(localU, VEC_CLASSID, 8);
  PetscValidHeaderSpecific(localX, VEC_CLASSID, 11);
  PetscCall((dm->ops->projectbdfieldlabellocal)(dm, time, label, numIds, ids, Nc, comps, localU, funcs, mode, localX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMComputeL2Diff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h.

  Collective

  Input Parameters:
+ dm    - The `DM`
. time  - The time
. funcs - The functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h, a global vector

  Output Parameter:
. diff - The diff ||u - u_h||_2

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunction()`, `DMComputeL2FieldDiff()`, `DMComputeL2GradientDiff()`
@*/
PetscErrorCode DMComputeL2Diff(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 5);
  PetscCall((dm->ops->computel2diff)(dm, time, funcs, ctxs, X, diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMComputeL2GradientDiff - This function computes the L_2 difference between the gradient of a function u and an FEM interpolant solution grad u_h.

  Collective

  Input Parameters:
+ dm    - The `DM`
, time  - The time
. funcs - The gradient functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
. X     - The coefficient vector u_h, a global vector
- n     - The vector to project along

  Output Parameter:
. diff - The diff ||(grad u - grad u_h) . n||_2

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunction()`, `DMComputeL2Diff()`, `DMComputeL2FieldDiff()`
@*/
PetscErrorCode DMComputeL2GradientDiff(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, const PetscReal n[], PetscReal *diff)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 5);
  PetscCall((dm->ops->computel2gradientdiff)(dm, time, funcs, ctxs, X, n, diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMComputeL2FieldDiff - This function computes the L_2 difference between a function u and an FEM interpolant solution u_h, separated into field components.

  Collective

  Input Parameters:
+ dm    - The `DM`
. time  - The time
. funcs - The functions to evaluate for each field component
. ctxs  - Optional array of contexts to pass to each function, or NULL.
- X     - The coefficient vector u_h, a global vector

  Output Parameter:
. diff - The array of differences, ||u^f - u^f_h||_2

  Level: developer

  Developer Notes:
  This API is specific to only particular usage of `DM`

  The notes need to provide some information about what has to be provided to the `DM` to be able to perform the computation.

.seealso: [](chapter_dmbase), `DM`, `DMProjectFunction()`, `DMComputeL2FieldDiff()`, `DMComputeL2GradientDiff()`
@*/
PetscErrorCode DMComputeL2FieldDiff(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal diff[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 5);
  PetscCall((dm->ops->computel2fielddiff)(dm, time, funcs, ctxs, X, diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
 DMGetNeighbors - Gets an array containing the MPI ranks of all the processes neighbors

 Not Collective

 Input Parameter:
.  dm    - The `DM`

 Output Parameters:
+  nranks - the number of neighbours
-  ranks - the neighbors ranks

 Level: beginner

 Note:
 Do not free the array, it is freed when the `DM` is destroyed.

.seealso: [](chapter_dmbase), `DM`, `DMDAGetNeighbors()`, `PetscSFGetRootRanks()`
@*/
PetscErrorCode DMGetNeighbors(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall((dm->ops->getneighbors)(dm, nranks, ranks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/matimpl.h> /* Needed because of coloring->ctype below */

/*
    Converts the input vector to a ghosted vector and then calls the standard coloring code.
    This has be a different function because it requires DM which is not defined in the Mat library
*/
PetscErrorCode MatFDColoringApply_AIJDM(Mat J, MatFDColoring coloring, Vec x1, void *sctx)
{
  PetscFunctionBegin;
  if (coloring->ctype == IS_COLORING_LOCAL) {
    Vec x1local;
    DM  dm;
    PetscCall(MatGetDM(J, &dm));
    PetscCheck(dm, PetscObjectComm((PetscObject)J), PETSC_ERR_ARG_INCOMP, "IS_COLORING_LOCAL requires a DM");
    PetscCall(DMGetLocalVector(dm, &x1local));
    PetscCall(DMGlobalToLocalBegin(dm, x1, INSERT_VALUES, x1local));
    PetscCall(DMGlobalToLocalEnd(dm, x1, INSERT_VALUES, x1local));
    x1 = x1local;
  }
  PetscCall(MatFDColoringApply_AIJ(J, coloring, x1, sctx));
  if (coloring->ctype == IS_COLORING_LOCAL) {
    DM dm;
    PetscCall(MatGetDM(J, &dm));
    PetscCall(DMRestoreLocalVector(dm, &x1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    MatFDColoringUseDM - allows a `MatFDColoring` object to use the `DM` associated with the matrix to compute a `IS_COLORING_LOCAL` coloring

    Input Parameter:
.    coloring - the `MatFDColoring` object

    Level: advanced

    Developer Note:
    this routine exists because the PETSc `Mat` library does not know about the `DM` objects

.seealso: [](chapter_dmbase), `DM`, `MatFDColoring`, `MatFDColoringCreate()`, `ISColoringType`
@*/
PetscErrorCode MatFDColoringUseDM(Mat coloring, MatFDColoring fdcoloring)
{
  PetscFunctionBegin;
  coloring->ops->fdcoloringapply = MatFDColoringApply_AIJDM;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    DMGetCompatibility - determine if two `DM`s are compatible

    Collective

    Input Parameters:
+    dm1 - the first `DM`
-    dm2 - the second `DM`

    Output Parameters:
+    compatible - whether or not the two `DM`s are compatible
-    set - whether or not the compatible value was actually determined and set

    Level: advanced

    Notes:
    Two `DM`s are deemed compatible if they represent the same parallel decomposition
    of the same topology. This implies that the section (field data) on one
    "makes sense" with respect to the topology and parallel decomposition of the other.
    Loosely speaking, compatible `DM`s represent the same domain and parallel
    decomposition, but hold different data.

    Typically, one would confirm compatibility if intending to simultaneously iterate
    over a pair of vectors obtained from different `DM`s.

    For example, two `DMDA` objects are compatible if they have the same local
    and global sizes and the same stencil width. They can have different numbers
    of degrees of freedom per node. Thus, one could use the node numbering from
    either `DM` in bounds for a loop over vectors derived from either `DM`.

    Consider the operation of summing data living on a 2-dof `DMDA` to data living
    on a 1-dof `DMDA`, which should be compatible, as in the following snippet.
.vb
  ...
  PetscCall(DMGetCompatibility(da1,da2,&compatible,&set));
  if (set && compatible)  {
    PetscCall(DMDAVecGetArrayDOF(da1,vec1,&arr1));
    PetscCall(DMDAVecGetArrayDOF(da2,vec2,&arr2));
    PetscCall(DMDAGetCorners(da1,&x,&y,NULL,&m,&n,NULL));
    for (j=y; j<y+n; ++j) {
      for (i=x; i<x+m, ++i) {
        arr1[j][i][0] = arr2[j][i][0] + arr2[j][i][1];
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(da1,vec1,&arr1));
    PetscCall(DMDAVecRestoreArrayDOF(da2,vec2,&arr2));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)da1,PETSC_ERR_ARG_INCOMP,"DMDA objects incompatible");
  }
  ...
.ve

    Checking compatibility might be expensive for a given implementation of `DM`,
    or might be impossible to unambiguously confirm or deny. For this reason,
    this function may decline to determine compatibility, and hence users should
    always check the "set" output parameter.

    A `DM` is always compatible with itself.

    In the current implementation, `DM`s which live on "unequal" communicators
    (MPI_UNEQUAL in the terminology of MPI_Comm_compare()) are always deemed
    incompatible.

    This function is labeled "Collective," as information about all subdomains
    is required on each rank. However, in `DM` implementations which store all this
    information locally, this function may be merely "Logically Collective".

    Developer Note:
    Compatibility is assumed to be a symmetric concept; `DM` A is compatible with `DM` B
    iff B is compatible with A. Thus, this function checks the implementations
    of both dm and dmc (if they are of different types), attempting to determine
    compatibility. It is left to `DM` implementers to ensure that symmetry is
    preserved. The simplest way to do this is, when implementing type-specific
    logic for this function, is to check for existing logic in the implementation
    of other `DM` types and let *set = PETSC_FALSE if found.

.seealso: [](chapter_dmbase), `DM`, `DMDACreateCompatibleDMDA()`, `DMStagCreateCompatibleDMStag()`
@*/
PetscErrorCode DMGetCompatibility(DM dm1, DM dm2, PetscBool *compatible, PetscBool *set)
{
  PetscMPIInt compareResult;
  DMType      type, type2;
  PetscBool   sameType;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm2, DM_CLASSID, 2);

  /* Declare a DM compatible with itself */
  if (dm1 == dm2) {
    *set        = PETSC_TRUE;
    *compatible = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Declare a DM incompatible with a DM that lives on an "unequal"
     communicator. Note that this does not preclude compatibility with
     DMs living on "congruent" or "similar" communicators, but this must be
     determined by the implementation-specific logic */
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)dm1), PetscObjectComm((PetscObject)dm2), &compareResult));
  if (compareResult == MPI_UNEQUAL) {
    *set        = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Pass to the implementation-specific routine, if one exists. */
  if (dm1->ops->getcompatibility) {
    PetscUseTypeMethod(dm1, getcompatibility, dm2, compatible, set);
    if (*set) PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* If dm1 and dm2 are of different types, then attempt to check compatibility
     with an implementation of this function from dm2 */
  PetscCall(DMGetType(dm1, &type));
  PetscCall(DMGetType(dm2, &type2));
  PetscCall(PetscStrcmp(type, type2, &sameType));
  if (!sameType && dm2->ops->getcompatibility) {
    PetscUseTypeMethod(dm2, getcompatibility, dm1, compatible, set); /* Note argument order */
  } else {
    *set = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMMonitorSet - Sets an additional monitor function that is to be used after a solve to monitor discretization performance.

  Logically Collective

  Input Parameters:
+ DM - the `DM`
. f - the monitor function
. mctx - [optional] user-defined context for private data for the monitor routine (use `NULL` if no context is desired)
- monitordestroy - [optional] routine that frees monitor context (may be `NULL`)

  Options Database Key:
- -dm_monitor_cancel - cancels all monitors that have been hardwired into a code by calls to `DMMonitorSet()`, but
                            does not cancel those set via the options database.

  Level: intermediate

  Note:
  Several different monitoring routines may be set by calling
  `DMMonitorSet()` multiple times or with `DMMonitorSetFromOptions()`; all will be called in the
  order in which they were set.

  Fortran Note:
  Only a single monitor function can be set for each `DM` object

  Developer Note:
  This API has a generic name but seems specific to a very particular aspect of the use of `DM`

.seealso: [](chapter_dmbase), `DM`, `DMMonitorCancel()`, `DMMonitorSetFromOptions()`, `DMMonitor()`
@*/
PetscErrorCode DMMonitorSet(DM dm, PetscErrorCode (*f)(DM, void *), void *mctx, PetscErrorCode (*monitordestroy)(void **))
{
  PetscInt m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (m = 0; m < dm->numbermonitors; ++m) {
    PetscBool identical;

    PetscCall(PetscMonitorCompare((PetscErrorCode(*)(void))f, mctx, monitordestroy, (PetscErrorCode(*)(void))dm->monitor[m], dm->monitorcontext[m], dm->monitordestroy[m], &identical));
    if (identical) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(dm->numbermonitors < MAXDMMONITORS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many monitors set");
  dm->monitor[dm->numbermonitors]          = f;
  dm->monitordestroy[dm->numbermonitors]   = monitordestroy;
  dm->monitorcontext[dm->numbermonitors++] = (void *)mctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMMonitorCancel - Clears all the monitor functions for a `DM` object.

  Logically Collective

  Input Parameter:
. dm - the DM

  Options Database Key:
. -dm_monitor_cancel - cancels all monitors that have been hardwired
  into a code by calls to `DMonitorSet()`, but does not cancel those
  set via the options database

  Level: intermediate

  Note:
  There is no way to clear one specific monitor from a `DM` object.

.seealso: [](chapter_dmbase), `DM`, `DMMonitorSet()`, `DMMonitorSetFromOptions()`, `DMMonitor()`
@*/
PetscErrorCode DMMonitorCancel(DM dm)
{
  PetscInt m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (m = 0; m < dm->numbermonitors; ++m) {
    if (dm->monitordestroy[m]) PetscCall((*dm->monitordestroy[m])(&dm->monitorcontext[m]));
  }
  dm->numbermonitors = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

  Collective

  Input Parameters:
+ dm   - `DM` object you wish to monitor
. name - the monitor type one is seeking
. help - message indicating what monitoring is done
. manual - manual page for the monitor
. monitor - the monitor function
- monitorsetup - a function that is called once ONLY if the user selected this monitor that may set additional features of the `DM` or `PetscViewer` objects

  Output Parameter:
. flg - Flag set if the monitor was created

  Level: developer

.seealso: [](chapter_dmbase), `DM`, `PetscOptionsGetViewer()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `DMMonitor()`, `DMMonitorSet()`
@*/
PetscErrorCode DMMonitorSetFromOptions(DM dm, const char name[], const char help[], const char manual[], PetscErrorCode (*monitor)(DM, void *), PetscErrorCode (*monitorsetup)(DM, PetscViewerAndFormat *), PetscBool *flg)
{
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)dm), ((PetscObject)dm)->options, ((PetscObject)dm)->prefix, name, &viewer, &format, flg));
  if (*flg) {
    PetscViewerAndFormat *vf;

    PetscCall(PetscViewerAndFormatCreate(viewer, format, &vf));
    PetscCall(PetscObjectDereference((PetscObject)viewer));
    if (monitorsetup) PetscCall((*monitorsetup)(dm, vf));
    PetscCall(DMMonitorSet(dm, (PetscErrorCode(*)(DM, void *))monitor, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   DMMonitor - runs the user provided monitor routines, if they exist

   Collective

   Input Parameter:
.  dm - The `DM`

   Level: developer

   Question:
   Note should indicate when during the life of the `DM` the monitor is run. It appears to be related to the discretization process seems rather specialized
   since some `DM` have no concept of discretization

.seealso: [](chapter_dmbase), `DM`, `DMMonitorSet()`, `DMMonitorSetFromOptions()`
@*/
PetscErrorCode DMMonitor(DM dm)
{
  PetscInt m;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for (m = 0; m < dm->numbermonitors; ++m) PetscCall((*dm->monitor[m])(dm, dm->monitorcontext[m]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMComputeError - Computes the error assuming the user has provided the exact solution functions

  Collective

  Input Parameters:
+ dm     - The `DM`
- sol    - The solution vector

  Input/Output Parameter:
. errors - An array of length Nf, the number of fields, or `NULL` for no output; on output
           contains the error in each field

  Output Parameter:
. errorVec - A vector to hold the cellwise error (may be `NULL`)

  Level: developer

  Note:
  The exact solutions come from the `PetscDS` object, and the time comes from `DMGetOutputSequenceNumber()`.

.seealso: [](chapter_dmbase), `DM`, `DMMonitorSet()`, `DMGetRegionNumDS()`, `PetscDSGetExactSolution()`, `DMGetOutputSequenceNumber()`
@*/
PetscErrorCode DMComputeError(DM dm, Vec sol, PetscReal errors[], Vec *errorVec)
{
  PetscErrorCode (**exactSol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  void    **ctxs;
  PetscReal time;
  PetscInt  Nf, f, Nds, s;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscCalloc2(Nf, &exactSol, Nf, &ctxs));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS         ds;
    DMLabel         label;
    IS              fieldIS;
    const PetscInt *fields;
    PetscInt        dsNf;

    PetscCall(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds, NULL));
    PetscCall(PetscDSGetNumFields(ds, &dsNf));
    if (fieldIS) PetscCall(ISGetIndices(fieldIS, &fields));
    for (f = 0; f < dsNf; ++f) {
      const PetscInt field = fields[f];
      PetscCall(PetscDSGetExactSolution(ds, field, &exactSol[field], &ctxs[field]));
    }
    if (fieldIS) PetscCall(ISRestoreIndices(fieldIS, &fields));
  }
  for (f = 0; f < Nf; ++f) PetscCheck(exactSol[f], PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "DS must contain exact solution functions in order to calculate error, missing for field %" PetscInt_FMT, f);
  PetscCall(DMGetOutputSequenceNumber(dm, NULL, &time));
  if (errors) PetscCall(DMComputeL2FieldDiff(dm, time, exactSol, ctxs, sol, errors));
  if (errorVec) {
    DM             edm;
    DMPolytopeType ct;
    PetscBool      simplex;
    PetscInt       dim, cStart, Nf;

    PetscCall(DMClone(dm, &edm));
    PetscCall(DMGetDimension(edm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
    PetscCall(DMPlexGetCellType(dm, cStart, &ct));
    simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscCall(DMGetNumFields(dm, &Nf));
    for (f = 0; f < Nf; ++f) {
      PetscFE         fe, efe;
      PetscQuadrature q;
      const char     *name;

      PetscCall(DMGetField(dm, f, NULL, (PetscObject *)&fe));
      PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, Nf, simplex, 0, PETSC_DETERMINE, &efe));
      PetscCall(PetscObjectGetName((PetscObject)fe, &name));
      PetscCall(PetscObjectSetName((PetscObject)efe, name));
      PetscCall(PetscFEGetQuadrature(fe, &q));
      PetscCall(PetscFESetQuadrature(efe, q));
      PetscCall(DMSetField(edm, f, NULL, (PetscObject)efe));
      PetscCall(PetscFEDestroy(&efe));
    }
    PetscCall(DMCreateDS(edm));

    PetscCall(DMCreateGlobalVector(edm, errorVec));
    PetscCall(PetscObjectSetName((PetscObject)*errorVec, "Error"));
    PetscCall(DMPlexComputeL2DiffVec(dm, time, exactSol, ctxs, sol, *errorVec));
    PetscCall(DMDestroy(&edm));
  }
  PetscCall(PetscFree2(exactSol, ctxs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetNumAuxiliaryVec - Get the number of auxiliary vectors associated with this `DM`

  Not Collective

  Input Parameter:
. dm     - The `DM`

  Output Parameter:
. numAux - The number of auxiliary data vectors

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMSetAuxiliaryVec()`, `DMGetAuxiliaryLabels()`, `DMGetAuxiliaryVec()`, `DMSetAuxiliaryVec()`
@*/
PetscErrorCode DMGetNumAuxiliaryVec(DM dm, PetscInt *numAux)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscHMapAuxGetSize(dm->auxData, numAux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetAuxiliaryVec - Get the auxiliary vector for region specified by the given label and value, and equation part

  Not Collective

  Input Parameters:
+ dm     - The `DM`
. label  - The `DMLabel`
. value  - The label value indicating the region
- part   - The equation part, or 0 if unused

  Output Parameter:
. aux    - The `Vec` holding auxiliary field data

  Level: advanced

  Note:
  If no auxiliary vector is found for this (label, value), (NULL, 0, 0) is checked as well.

.seealso: [](chapter_dmbase), `DM`, `DMSetAuxiliaryVec()`, `DMGetNumAuxiliaryVec()`, `DMGetAuxiliaryLabels()`
@*/
PetscErrorCode DMGetAuxiliaryVec(DM dm, DMLabel label, PetscInt value, PetscInt part, Vec *aux)
{
  PetscHashAuxKey key, wild = {NULL, 0, 0};
  PetscBool       has;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  key.label = label;
  key.value = value;
  key.part  = part;
  PetscCall(PetscHMapAuxHas(dm->auxData, key, &has));
  if (has) PetscCall(PetscHMapAuxGet(dm->auxData, key, aux));
  else PetscCall(PetscHMapAuxGet(dm->auxData, wild, aux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetAuxiliaryVec - Set an auxiliary vector for region specified by the given label and value, and equation part

  Not Collective because auxiliary vectors are not parallel

  Input Parameters:
+ dm     - The `DM`
. label  - The `DMLabel`
. value  - The label value indicating the region
. part   - The equation part, or 0 if unused
- aux    - The `Vec` holding auxiliary field data

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMGetAuxiliaryVec()`, `DMGetAuxiliaryLabels()`, `DMCopyAuxiliaryVec()`
@*/
PetscErrorCode DMSetAuxiliaryVec(DM dm, DMLabel label, PetscInt value, PetscInt part, Vec aux)
{
  Vec             old;
  PetscHashAuxKey key;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (label) PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 2);
  key.label = label;
  key.value = value;
  key.part  = part;
  PetscCall(PetscHMapAuxGet(dm->auxData, key, &old));
  PetscCall(PetscObjectReference((PetscObject)aux));
  PetscCall(PetscObjectDereference((PetscObject)old));
  if (!aux) PetscCall(PetscHMapAuxDel(dm->auxData, key));
  else PetscCall(PetscHMapAuxSet(dm->auxData, key, aux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetAuxiliaryLabels - Get the labels, values, and parts for all auxiliary vectors in this `DM`

  Not Collective

  Input Parameter:
. dm      - The `DM`

  Output Parameters:
+ labels  - The `DMLabel`s for each `Vec`
. values  - The label values for each `Vec`
- parts   - The equation parts for each `Vec`

  Level: advanced

  Note:
  The arrays passed in must be at least as large as `DMGetNumAuxiliaryVec()`.

.seealso: [](chapter_dmbase), `DM`, `DMGetNumAuxiliaryVec()`, `DMGetAuxiliaryVec()`, `DMGetNumAuxiliaryVec()`, `DMSetAuxiliaryVec()`, DMCopyAuxiliaryVec()`
@*/
PetscErrorCode DMGetAuxiliaryLabels(DM dm, DMLabel labels[], PetscInt values[], PetscInt parts[])
{
  PetscHashAuxKey *keys;
  PetscInt         n, i, off = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(labels, 2);
  PetscValidIntPointer(values, 3);
  PetscValidIntPointer(parts, 4);
  PetscCall(DMGetNumAuxiliaryVec(dm, &n));
  PetscCall(PetscMalloc1(n, &keys));
  PetscCall(PetscHMapAuxGetKeys(dm->auxData, &off, keys));
  for (i = 0; i < n; ++i) {
    labels[i] = keys[i].label;
    values[i] = keys[i].value;
    parts[i]  = keys[i].part;
  }
  PetscCall(PetscFree(keys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCopyAuxiliaryVec - Copy the auxiliary vector data on a `DM` to a new `DM`

  Not Collective

  Input Parameter:
. dm    - The `DM`

  Output Parameter:
. dmNew - The new `DM`, now with the same auxiliary data

  Level: advanced

  Note:
  This is a shallow copy of the auxiliary vectors

.seealso: [](chapter_dmbase), `DM`, `DMGetNumAuxiliaryVec()`, `DMGetAuxiliaryVec()`, `DMSetAuxiliaryVec()`
@*/
PetscErrorCode DMCopyAuxiliaryVec(DM dm, DM dmNew)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscHMapAuxDestroy(&dmNew->auxData));
  PetscCall(PetscHMapAuxDuplicate(dm->auxData, &dmNew->auxData));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPolytopeMatchOrientation - Determine an orientation (transformation) that takes the source face arrangement to the target face arrangement

  Not Collective

  Input Parameters:
+ ct         - The `DMPolytopeType`
. sourceCone - The source arrangement of faces
- targetCone - The target arrangement of faces

  Output Parameters:
+ ornt  - The orientation (transformation) which will take the source arrangement to the target arrangement
- found - Flag indicating that a suitable orientation was found

  Level: advanced

  Note:
  An arrangement is a face order combined with an orientation for each face

  Each orientation (transformation) is labeled with an integer from negative `DMPolytopeTypeGetNumArrangments(ct)`/2 to `DMPolytopeTypeGetNumArrangments(ct)`/2
  that labels each arrangement (face ordering plus orientation for each face).

  See `DMPolytopeMatchVertexOrientation()` to find a new vertex orientation that takes the source vertex arrangement to the target vertex arrangement

.seealso: [](chapter_dmbase), `DM`, `DMPolytopeGetOrientation()`, `DMPolytopeMatchVertexOrientation()`, `DMPolytopeGetVertexOrientation()`
@*/
PetscErrorCode DMPolytopeMatchOrientation(DMPolytopeType ct, const PetscInt sourceCone[], const PetscInt targetCone[], PetscInt *ornt, PetscBool *found)
{
  const PetscInt cS = DMPolytopeTypeGetConeSize(ct);
  const PetscInt nO = DMPolytopeTypeGetNumArrangments(ct) / 2;
  PetscInt       o, c;

  PetscFunctionBegin;
  if (!nO) {
    *ornt  = 0;
    *found = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (o = -nO; o < nO; ++o) {
    const PetscInt *arr = DMPolytopeTypeGetArrangment(ct, o);

    for (c = 0; c < cS; ++c)
      if (sourceCone[arr[c * 2]] != targetCone[c]) break;
    if (c == cS) {
      *ornt = o;
      break;
    }
  }
  *found = o == nO ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPolytopeGetOrientation - Determine an orientation (transformation) that takes the source face arrangement to the target face arrangement

  Not Collective

  Input Parameters:
+ ct         - The `DMPolytopeType`
. sourceCone - The source arrangement of faces
- targetCone - The target arrangement of faces

  Output Parameter:
. ornt  - The orientation (transformation) which will take the source arrangement to the target arrangement

  Level: advanced

  Note:
  This function is the same as `DMPolytopeMatchOrientation()` except it will generate an error if no suitable orientation can be found.

  Developer Note:
  It is unclear why this function needs to exist since one can simply call `DMPolytopeMatchOrientation()` and error if none is found

.seealso: [](chapter_dmbase), `DM`, `DMPolytopeType`, `DMPolytopeMatchOrientation()`, `DMPolytopeGetVertexOrientation()`, `DMPolytopeMatchVertexOrientation()`
@*/
PetscErrorCode DMPolytopeGetOrientation(DMPolytopeType ct, const PetscInt sourceCone[], const PetscInt targetCone[], PetscInt *ornt)
{
  PetscBool found;

  PetscFunctionBegin;
  PetscCall(DMPolytopeMatchOrientation(ct, sourceCone, targetCone, ornt, &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find orientation for %s", DMPolytopeTypes[ct]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPolytopeMatchVertexOrientation - Determine an orientation (transformation) that takes the source vertex arrangement to the target vertex arrangement

  Not Collective

  Input Parameters:
+ ct         - The `DMPolytopeType`
. sourceVert - The source arrangement of vertices
- targetVert - The target arrangement of vertices

  Output Parameters:
+ ornt  - The orientation (transformation) which will take the source arrangement to the target arrangement
- found - Flag indicating that a suitable orientation was found

  Level: advanced

  Note:
  An arrangement is a vertex order

  Each orientation (transformation) is labeled with an integer from negative `DMPolytopeTypeGetNumArrangments(ct)`/2 to `DMPolytopeTypeGetNumArrangments(ct)`/2
  that labels each arrangement (vertex ordering).

  See `DMPolytopeMatchOrientation()` to find a new face orientation that takes the source face arrangement to the target face arrangement

.seealso: [](chapter_dmbase), `DM`, `DMPolytopeType`, `DMPolytopeGetOrientation()`, `DMPolytopeMatchOrientation()`, `DMPolytopeTypeGetNumVertices()`, `DMPolytopeTypeGetVertexArrangment()`
@*/
PetscErrorCode DMPolytopeMatchVertexOrientation(DMPolytopeType ct, const PetscInt sourceVert[], const PetscInt targetVert[], PetscInt *ornt, PetscBool *found)
{
  const PetscInt cS = DMPolytopeTypeGetNumVertices(ct);
  const PetscInt nO = DMPolytopeTypeGetNumArrangments(ct) / 2;
  PetscInt       o, c;

  PetscFunctionBegin;
  if (!nO) {
    *ornt  = 0;
    *found = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (o = -nO; o < nO; ++o) {
    const PetscInt *arr = DMPolytopeTypeGetVertexArrangment(ct, o);

    for (c = 0; c < cS; ++c)
      if (sourceVert[arr[c]] != targetVert[c]) break;
    if (c == cS) {
      *ornt = o;
      break;
    }
  }
  *found = o == nO ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPolytopeGetVertexOrientation - Determine an orientation (transformation) that takes the source vertex arrangement to the target vertex arrangement

  Not Collective

  Input Parameters:
+ ct         - The `DMPolytopeType`
. sourceCone - The source arrangement of vertices
- targetCone - The target arrangement of vertices

  Output Parameter:
. ornt  - The orientation (transformation) which will take the source arrangement to the target arrangement

  Level: advanced

  Note:
  This function is the same as `DMPolytopeMatchVertexOrientation()` except it errors if not orientation is possible.

  Developer Note:
  It is unclear why this function needs to exist since one can simply call `DMPolytopeMatchVertexOrientation()` and error if none is found

.seealso: [](chapter_dmbase), `DM`, `DMPolytopeType`, `DMPolytopeMatchVertexOrientation()`, `DMPolytopeGetOrientation()`
@*/
PetscErrorCode DMPolytopeGetVertexOrientation(DMPolytopeType ct, const PetscInt sourceCone[], const PetscInt targetCone[], PetscInt *ornt)
{
  PetscBool found;

  PetscFunctionBegin;
  PetscCall(DMPolytopeMatchVertexOrientation(ct, sourceCone, targetCone, ornt, &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not find orientation for %s", DMPolytopeTypes[ct]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPolytopeInCellTest - Check whether a point lies inside the reference cell of given type

  Not Collective

  Input Parameters:
+ ct    - The `DMPolytopeType`
- point - Coordinates of the point

  Output Parameter:
. inside  - Flag indicating whether the point is inside the reference cell of given type

  Level: advanced

.seealso: [](chapter_dmbase), `DM`, `DMPolytopeType`, `DMLocatePoints()`
@*/
PetscErrorCode DMPolytopeInCellTest(DMPolytopeType ct, const PetscReal point[], PetscBool *inside)
{
  PetscReal sum = 0.0;
  PetscInt  d;

  PetscFunctionBegin;
  *inside = PETSC_TRUE;
  switch (ct) {
  case DM_POLYTOPE_TRIANGLE:
  case DM_POLYTOPE_TETRAHEDRON:
    for (d = 0; d < DMPolytopeTypeGetDim(ct); ++d) {
      if (point[d] < -1.0) {
        *inside = PETSC_FALSE;
        break;
      }
      sum += point[d];
    }
    if (sum > PETSC_SMALL) {
      *inside = PETSC_FALSE;
      break;
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
    for (d = 0; d < DMPolytopeTypeGetDim(ct); ++d)
      if (PetscAbsReal(point[d]) > 1. + PETSC_SMALL) {
        *inside = PETSC_FALSE;
        break;
      }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported polytope type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
