#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/

PetscFunctionList DMGeomModelList              = NULL;
PetscBool         DMGeomModelRegisterAllCalled = PETSC_FALSE;

#if defined(PETSC_HAVE_EGADS)
PETSC_INTERN PetscErrorCode DMSnapToGeomModel_EGADS(DM, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
#endif

static PetscErrorCode DMSnapToGeomModelBall(DM dm, PetscInt p, PetscInt dE, const PetscScalar mcoords[], PetscScalar gcoords[])
{
  PetscInt val;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabelValue(dm, "marker", p, &val));
  if (val >= 0) {
    PetscReal norm = 0.;

    for (PetscInt d = 0; d < dE; ++d) norm += PetscSqr(PetscRealPart(mcoords[d]));
    norm = PetscSqrtReal(norm);
    for (PetscInt d = 0; d < dE; ++d) gcoords[d] = mcoords[d] / norm;
  } else {
    for (PetscInt d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSnapToGeomModelCylinder(DM dm, PetscInt p, PetscInt dE, const PetscScalar mcoords[], PetscScalar gcoords[])
{
  PetscReal gmin[3], gmax[3];
  PetscInt  val;

  PetscFunctionBeginUser;
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMGetLabelValue(dm, "generatrix", p, &val));
  if (val >= 0) {
    PetscReal norm = 0.;

    for (PetscInt d = 0; d < dE - 1; ++d) norm += PetscSqr(PetscRealPart(mcoords[d]));
    norm = PetscSqrtReal(norm);
    for (PetscInt d = 0; d < dE - 1; ++d) gcoords[d] = mcoords[d] * gmax[0] / norm;
    gcoords[dE - 1] = mcoords[dE - 1];
  } else {
    for (PetscInt d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGeomModelRegisterAll - Registers all of the geometry model methods in the `DM` package.

  Not Collective

  Level: advanced

.seealso: `DM`, `DMGeomModelRegisterDestroy()`
@*/
PetscErrorCode DMGeomModelRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMGeomModelRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DMGeomModelRegisterAllCalled = PETSC_TRUE;
  PetscCall(DMGeomModelRegister("ball", DMSnapToGeomModelBall));
  PetscCall(DMGeomModelRegister("cylinder", DMSnapToGeomModelCylinder));
#if defined(PETSC_HAVE_EGADS)
  // FIXME: Brandon uses DMPlexSnapToGeomModel() here instead
  PetscCall(DMGeomModelRegister("egads", DMSnapToGeomModel_EGADS));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGeomModelRegister -  Adds a geometry model to `DM`

  Not Collective, No Fortran Support

  Input Parameters:
+ sname - name of a new user-defined geometry model
- fnc   - geometry model function

  Example Usage:
.vb
   DMGeomModelRegister("my_geom_model", MySnapToGeomModel);
.ve

  Then, your generator can be chosen with the procedural interface via
.vb
  DMSetGeomModel(dm, "my_geom_model",...)
.ve
  or at runtime via the option
.vb
  -dm_geom_model my_geom_model
.ve

  Level: advanced

  Note:
  `DMGeomModelRegister()` may be called multiple times to add several user-defined generators

.seealso: `DM`, `DMGeomModelRegisterAll()`, `DMPlexGeomModel()`, `DMGeomModelRegisterDestroy()`
@*/
PetscErrorCode DMGeomModelRegister(const char sname[], PetscErrorCode (*fnc)(DM, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&DMGeomModelList, sname, fnc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGeomModelRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMGeomModelList));
  DMGeomModelRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetSnapToGeomModel - Choose a geometry model for this `DM`.

  Not Collective

  Input Parameters:
+ dm   - The `DM` object
- name - A geometry model name, or `NULL` for the default

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexCreate()`, `DMSnapToGeomModel()`
@*/
PetscErrorCode DMSetSnapToGeomModel(DM dm, const char name[])
{
  char      geomname[PETSC_MAX_PATH_LEN];
  PetscBool flg;

  PetscFunctionBegin;
  if (!name && dm->ops->snaptogeommodel) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscOptionsGetString(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_geom_model", geomname, sizeof(geomname), &flg));
  if (flg) name = geomname;
  if (!name) {
    PetscObject modelObj;

    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", &modelObj));
    if (modelObj) name = "egads";
    else {
      PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", &modelObj));
      if (modelObj) name = "egads";
    }
  }
  if (!name) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(DMGeomModelList, name, &dm->ops->snaptogeommodel));
  PetscCheck(dm->ops->snaptogeommodel, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Geometry model %s not registered; you may need to add --download-%s to your ./configure options", name, name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSnapToGeomModel - Given a coordinate point 'mcoords' on the mesh point 'p', return the closest coordinate point 'gcoords' on the geometry model associated with that point.

  Not Collective

  Input Parameters:
+ dm      - The `DMPLEX` object
. p       - The mesh point
. dE      - The coordinate dimension
- mcoords - A coordinate point lying on the mesh point

  Output Parameter:
. gcoords - The closest coordinate point on the geometry model associated with 'p' to the given point

  Level: intermediate

  Note:
  Returns the original coordinates if no geometry model is found.

  The coordinate dimension may be different from the coordinate dimension of the `dm`, for example if the transformation is extrusion.

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMRefine()`, `DMPlexCreate()`, `DMPlexSetRefinementUniform()`
@*/
PetscErrorCode DMSnapToGeomModel(DM dm, PetscInt p, PetscInt dE, const PetscScalar mcoords[], PetscScalar gcoords[])
{
  PetscFunctionBegin;
  if (!dm->ops->snaptogeommodel)
    for (PetscInt d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
  else PetscUseTypeMethod(dm, snaptogeommodel, p, dE, mcoords, gcoords);
  PetscFunctionReturn(PETSC_SUCCESS);
}
