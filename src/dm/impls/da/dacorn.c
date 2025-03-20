/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/
#include <petscdmfield.h>

PetscErrorCode DMCreateCoordinateDM_DA(DM dm, DM *cdm)
{
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(DMDACreateCompatibleDMDA(dm, dm->dim, cdm));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*cdm, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*cdm, "cdm_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDASetFieldName - Sets the names of individual field components in multicomponent
  vectors associated with a `DMDA`.

  Logically Collective; name must contain a common value

  Input Parameters:
+ da   - the `DMDA`
. nf   - field number for the `DMDA` (0, 1, ... dof-1), where dof indicates the
         number of degrees of freedom per node within the `DMDA`
- name - the name of the field (component)

  Level: intermediate

  Note:
  It must be called after having called `DMSetUp()`.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetFieldName()`, `DMDASetCoordinateName()`, `DMDAGetCoordinateName()`, `DMDASetFieldNames()`, `DMSetUp()`
@*/
PetscErrorCode DMDASetFieldName(DM da, PetscInt nf, const char name[])
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  PetscCheck(nf >= 0 && nf < dd->w, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid field number: %" PetscInt_FMT, nf);
  PetscCheck(dd->fieldname, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "You should call DMSetUp() first");
  PetscCall(PetscFree(dd->fieldname[nf]));
  PetscCall(PetscStrallocpy(name, &dd->fieldname[nf]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDAGetFieldNames - Gets the name of all the components in the vector associated with the `DMDA`

  Not Collective; names will contain a common value; No Fortran Support

  Input Parameter:
. da - the `DMDA` object

  Output Parameter:
. names - the names of the components, final string is `NULL`, will have the same number of entries as the dof used in creating the `DMDA`

  Level: intermediate

  Fortran Note:
  Use `DMDAGetFieldName()`

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetFieldName()`, `DMDASetCoordinateName()`, `DMDAGetCoordinateName()`, `DMDASetFieldName()`, `DMDASetFieldNames()`
@*/
PetscErrorCode DMDAGetFieldNames(DM da, const char *const **names)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  *names = (const char *const *)dd->fieldname;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASetFieldNames - Sets the name of each component in the vector associated with the `DMDA`

  Logically Collective; names must contain a common value; No Fortran Support

  Input Parameters:
+ da    - the `DMDA` object
- names - the names of the components, final string must be `NULL`, must have the same number of entries as the dof used in creating the `DMDA`

  Level: intermediate

  Note:
  It must be called after having called `DMSetUp()`.

  Fortran Note:
  Use `DMDASetFieldName()`

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetFieldName()`, `DMDASetCoordinateName()`, `DMDAGetCoordinateName()`, `DMDASetFieldName()`, `DMSetUp()`
@*/
PetscErrorCode DMDASetFieldNames(DM da, const char *const names[])
{
  DM_DA   *dd = (DM_DA *)da->data;
  char   **fieldname;
  PetscInt nf = 0;

  PetscFunctionBegin;
  PetscCheck(dd->fieldname, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "You should call DMSetUp() first");
  while (names[nf++]) { }
  PetscCheck(nf == dd->w + 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of fields %" PetscInt_FMT, nf - 1);
  PetscCall(PetscStrArrayallocpy(names, &fieldname));
  PetscCall(PetscStrArrayDestroy(&dd->fieldname));
  dd->fieldname = fieldname;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetFieldName - Gets the names of individual field components in multicomponent
  vectors associated with a `DMDA`.

  Not Collective; name will contain a common value

  Input Parameters:
+ da - the `DMDA`
- nf - field number for the `DMDA` (0, 1, ... dof-1), where dof indicates the
       number of degrees of freedom per node within the `DMDA`

  Output Parameter:
. name - the name of the field (component)

  Level: intermediate

  Note:
  It must be called after having called `DMSetUp()`.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDASetFieldName()`, `DMDASetCoordinateName()`, `DMDAGetCoordinateName()`, `DMSetUp()`
@*/
PetscErrorCode DMDAGetFieldName(DM da, PetscInt nf, const char *name[])
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  PetscAssertPointer(name, 3);
  PetscCheck(nf >= 0 && nf < dd->w, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid field number: %" PetscInt_FMT, nf);
  PetscCheck(dd->fieldname, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "You should call DMSetUp() first");
  *name = dd->fieldname[nf];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDASetCoordinateName - Sets the name of the coordinate directions associated with a `DMDA`, for example "x" or "y"

  Logically Collective; name must contain a common value; No Fortran Support

  Input Parameters:
+ dm   - the `DMDA`
. nf   - coordinate number for the `DMDA` (0, 1, ... dim-1),
- name - the name of the coordinate

  Level: intermediate

  Note:
  Must be called after having called `DMSetUp()`.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetCoordinateName()`, `DMDASetFieldName()`, `DMDAGetFieldName()`, `DMSetUp()`
@*/
PetscErrorCode DMDASetCoordinateName(DM dm, PetscInt nf, const char name[])
{
  DM_DA *dd = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  PetscCheck(nf >= 0 && nf < dm->dim, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid coordinate number: %" PetscInt_FMT, nf);
  PetscCheck(dd->coordinatename, PetscObjectComm((PetscObject)dm), PETSC_ERR_ORDER, "You should call DMSetUp() first");
  PetscCall(PetscFree(dd->coordinatename[nf]));
  PetscCall(PetscStrallocpy(name, &dd->coordinatename[nf]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetCoordinateName - Gets the name of a coordinate direction associated with a `DMDA`.

  Not Collective; name will contain a common value; No Fortran Support

  Input Parameters:
+ dm - the `DMDA`
- nf - number for the `DMDA` (0, 1, ... dim-1)

  Output Parameter:
. name - the name of the coordinate direction

  Level: intermediate

  Note:
  It must be called after having called `DMSetUp()`.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDASetCoordinateName()`, `DMDASetFieldName()`, `DMDAGetFieldName()`, `DMSetUp()`
@*/
PetscErrorCode DMDAGetCoordinateName(DM dm, PetscInt nf, const char *name[])
{
  DM_DA *dd = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMDA);
  PetscAssertPointer(name, 3);
  PetscCheck(nf >= 0 && nf < dm->dim, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid coordinate number: %" PetscInt_FMT, nf);
  PetscCheck(dd->coordinatename, PetscObjectComm((PetscObject)dm), PETSC_ERR_ORDER, "You should call DMSetUp() first");
  *name = dd->coordinatename[nf];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDAGetCorners - Returns the global (`x`,`y`,`z`) indices of the lower left
  corner and size of the local region, excluding ghost points.

  Not Collective

  Input Parameter:
. da - the `DMDA`

  Output Parameters:
+ x - the corner index for the first dimension
. y - the corner index for the second dimension (only used in 2D and 3D problems)
. z - the corner index for the third dimension (only used in 3D problems)
. m - the width in the first dimension
. n - the width in the second dimension (only used in 2D and 3D problems)
- p - the width in the third dimension (only used in 3D problems)

  Level: beginner

  Notes:
  Any of `y`, `z`, `n`, and `p` can be passed in as `NULL` if not needed.

  The corner information is independent of the number of degrees of
  freedom per node set with the `DMDACreateXX()` routine.  Thus the `x`, `y`, and `z`
  can be thought of as the lower left coordinates of the patch of values on process on a logical grid and `m`, `n`, and `p` as the
  extent of the patch, where each grid point has (potentially) several degrees of freedom.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetGhostCorners()`, `DMDAGetOwnershipRanges()`, `DMStagGetCorners()`, `DMSTAG`
@*/
PetscErrorCode DMDAGetCorners(DM da, PeOp PetscInt *x, PeOp PetscInt *y, PeOp PetscInt *z, PeOp PetscInt *m, PeOp PetscInt *n, PeOp PetscInt *p)
{
  PetscInt w;
  DM_DA   *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  /* since the xs, xe ... have all been multiplied by the number of degrees
     of freedom per cell, w = dd->w, we divide that out before returning.*/
  w = dd->w;
  if (x) *x = dd->xs / w + dd->xo;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = dd->ys + dd->yo;
  if (z) *z = dd->zs + dd->zo;
  if (m) *m = (dd->xe - dd->xs) / w;
  if (n) *n = (dd->ye - dd->ys);
  if (p) *p = (dd->ze - dd->zs);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetLocalBoundingIndices_DMDA(DM dm, PetscReal lmin[], PetscReal lmax[])
{
  DMDALocalInfo info;

  PetscFunctionBegin;
  PetscCall(DMDAGetLocalInfo(dm, &info));
  lmin[0] = info.xs;
  lmin[1] = info.ys;
  lmin[2] = info.zs;
  lmax[0] = info.xs + info.xm - 1;
  lmax[1] = info.ys + info.ym - 1;
  lmax[2] = info.zs + info.zm - 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma ignore: -fdoc-*
/*@
  DMDAGetReducedDMDA - Deprecated; use DMDACreateCompatibleDMDA()

  Level: deprecated
@*/
PetscErrorCode DMDAGetReducedDMDA(DM da, PetscInt nfields, DM *nda)
{
  PetscFunctionBegin;
  PetscCall(DMDACreateCompatibleDMDA(da, nfields, nda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDACreateCompatibleDMDA - Creates a `DMDA` with the same layout as given `DMDA` but with fewer or more fields

  Collective

  Input Parameters:
+ da      - the `DMDA`
- nfields - number of fields in new `DMDA`

  Output Parameter:
. nda - the new `DMDA`

  Level: intermediate

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetGhostCorners()`, `DMSetCoordinates()`, `DMDASetUniformCoordinates()`, `DMGetCoordinates()`, `DMDAGetGhostedCoordinates()`,
          `DMStagCreateCompatibleDMStag()`
@*/
PetscErrorCode DMDACreateCompatibleDMDA(DM da, PetscInt nfields, DM *nda)
{
  DM_DA          *dd = (DM_DA *)da->data;
  PetscInt        s, m, n, p, M, N, P, dim, Mo, No, Po;
  const PetscInt *lx, *ly, *lz;
  DMBoundaryType  bx, by, bz;
  DMDAStencilType stencil_type;
  Vec             coords;
  PetscInt        ox, oy, oz;
  PetscInt        cl, rl;

  PetscFunctionBegin;
  dim = da->dim;
  M   = dd->M;
  N   = dd->N;
  P   = dd->P;
  m   = dd->m;
  n   = dd->n;
  p   = dd->p;
  s   = dd->s;
  bx  = dd->bx;
  by  = dd->by;
  bz  = dd->bz;

  stencil_type = dd->stencil_type;

  PetscCall(DMDAGetOwnershipRanges(da, &lx, &ly, &lz));
  if (dim == 1) {
    PetscCall(DMDACreate1d(PetscObjectComm((PetscObject)da), bx, M, nfields, s, dd->lx, nda));
  } else if (dim == 2) {
    PetscCall(DMDACreate2d(PetscObjectComm((PetscObject)da), bx, by, stencil_type, M, N, m, n, nfields, s, lx, ly, nda));
  } else if (dim == 3) {
    PetscCall(DMDACreate3d(PetscObjectComm((PetscObject)da), bx, by, bz, stencil_type, M, N, P, m, n, p, nfields, s, lx, ly, lz, nda));
  }
  PetscCall(DMSetUp(*nda));
  PetscCall(DMGetCoordinates(da, &coords));
  PetscCall(DMSetCoordinates(*nda, coords));

  /* allow for getting a reduced DA corresponding to a domain decomposition */
  PetscCall(DMDAGetOffset(da, &ox, &oy, &oz, &Mo, &No, &Po));
  PetscCall(DMDASetOffset(*nda, ox, oy, oz, Mo, No, Po));

  /* allow for getting a reduced DA corresponding to a coarsened DA */
  PetscCall(DMGetCoarsenLevel(da, &cl));
  PetscCall(DMGetRefineLevel(da, &rl));

  (*nda)->levelup   = rl;
  (*nda)->leveldown = cl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDAGetCoordinateArray - Gets an array containing the coordinates of the `DMDA`

  Not Collective; No Fortran Support

  Input Parameter:
. dm - the `DMDA`

  Output Parameter:
. xc - the coordinates

  Level: intermediate

  Note:
  Use  `DMDARestoreCoordinateArray()` to return the array

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDASetCoordinateName()`, `DMDASetFieldName()`, `DMDAGetFieldName()`, `DMDARestoreCoordinateArray()`
@*/
PetscErrorCode DMDAGetCoordinateArray(DM dm, void *xc)
{
  DM  cdm;
  Vec x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetCoordinates(dm, &x));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMDAVecGetArray(cdm, x, xc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDARestoreCoordinateArray - Returns an array containing the coordinates of the `DMDA` obtained with `DMDAGetCoordinateArray()`

  Not Collective; No Fortran Support

  Input Parameters:
+ dm - the `DMDA`
- xc - the coordinates

  Level: intermediate

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDASetCoordinateName()`, `DMDASetFieldName()`, `DMDAGetFieldName()`, `DMDAGetCoordinateArray()`
@*/
PetscErrorCode DMDARestoreCoordinateArray(DM dm, void *xc)
{
  DM  cdm;
  Vec x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetCoordinates(dm, &x));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMDAVecRestoreArray(cdm, x, xc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
