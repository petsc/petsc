#include <petsc/private/dmimpl.h>           /*I      "petscdm.h"          I*/

#include <petscdmplex.h>

/*@C
  DMGetPeriodicity - Get the description of mesh periodicity

  Input Parameter:
. dm      - The DM object

  Output Parameters:
+ maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates
. Lstart  - If we assume the mesh is a torus, this is the start of each coordinate, or NULL for 0.0
- L       - If we assume the mesh is a torus, this is the length of each coordinate, otherwise it is < 0.0

  Level: developer

.seealso: `DMGetPeriodicity()`
@*/
PetscErrorCode DMGetPeriodicity(DM dm, const PetscReal **maxCell, const PetscReal **Lstart, const PetscReal **L)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (maxCell) *maxCell = dm->maxCell;
  if (Lstart)  *Lstart  = dm->Lstart;
  if (L)       *L       = dm->L;
  PetscFunctionReturn(0);
}

/*@C
  DMSetPeriodicity - Set the description of mesh periodicity

  Input Parameters:
+ dm      - The DM object
. maxCell - Over distances greater than this, we can assume a point has crossed over to another sheet, when trying to localize cell coordinates. Pass NULL to remove such information.
. Lstart  - If we assume the mesh is a torus, this is the start of each coordinate, or NULL for 0.0
- L       - If we assume the mesh is a torus, this is the length of each coordinate, otherwise it is < 0.0

  Level: developer

.seealso: `DMGetPeriodicity()`
@*/
PetscErrorCode DMSetPeriodicity(DM dm, const PetscReal maxCell[], const PetscReal Lstart[], const PetscReal L[])
{
  PetscInt dim, d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (maxCell) {PetscValidRealPointer(maxCell,2);}
  if (Lstart)  {PetscValidRealPointer(Lstart,3);}
  if (L)       {PetscValidRealPointer(L,4);}
  PetscCall(DMGetDimension(dm, &dim));
  if (maxCell) {
    if (!dm->maxCell) PetscCall(PetscMalloc1(dim, &dm->maxCell));
    for (d = 0; d < dim; ++d) dm->maxCell[d] = maxCell[d];
  } else { /* remove maxCell information to disable automatic computation of localized vertices */
    PetscCall(PetscFree(dm->maxCell));
    dm->maxCell = NULL;
  }
  if (Lstart) {
    if (!dm->Lstart) PetscCall(PetscMalloc1(dim, &dm->Lstart));
    for (d = 0; d < dim; ++d) dm->Lstart[d] = Lstart[d];
  } else { /* remove L information to disable automatic computation of localized vertices */
    PetscCall(PetscFree(dm->Lstart));
    dm->Lstart = NULL;
  }
  if (L) {
    if (!dm->L) PetscCall(PetscMalloc1(dim, &dm->L));
    for (d = 0; d < dim; ++d) dm->L[d] = L[d];
  } else { /* remove L information to disable automatic computation of localized vertices */
    PetscCall(PetscFree(dm->L));
    dm->L = NULL;
  }
  PetscCheck((dm->maxCell && dm->L) || (!dm->maxCell && !dm->L), PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Cannot set only one of maxCell/L");
  PetscFunctionReturn(0);
}

/*@
  DMLocalizeCoordinate - If a mesh is periodic (a torus with lengths L_i, some of which can be infinite), project the coordinate onto [0, L_i) in each dimension.

  Input Parameters:
+ dm     - The DM
. in     - The input coordinate point (dim numbers)
- endpoint - Include the endpoint L_i

  Output Parameter:
. out - The localized coordinate point

  Level: developer

.seealso: `DMLocalizeCoordinates()`, `DMLocalizeAddCoordinate()`
@*/
PetscErrorCode DMLocalizeCoordinate(DM dm, const PetscScalar in[], PetscBool endpoint, PetscScalar out[])
{
  PetscInt       dim, d;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &dim));
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    if (endpoint) {
      for (d = 0; d < dim; ++d) {
        if ((PetscAbsReal(PetscRealPart(in[d])/dm->L[d] - PetscFloorReal(PetscRealPart(in[d])/dm->L[d])) < PETSC_SMALL) && (PetscRealPart(in[d])/dm->L[d] > PETSC_SMALL)) {
          out[d] = in[d] - dm->L[d]*(PetscFloorReal(PetscRealPart(in[d])/dm->L[d]) - 1);
        } else {
          out[d] = in[d] - dm->L[d]*PetscFloorReal(PetscRealPart(in[d])/dm->L[d]);
        }
      }
    } else {
      for (d = 0; d < dim; ++d) {
        out[d] = in[d] - dm->L[d]*PetscFloorReal(PetscRealPart(in[d])/dm->L[d]);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMLocalizeCoordinate_Internal - If a mesh is periodic, and the input point is far from the anchor, pick the coordinate sheet of the torus which moves it closer.

  Input Parameters:
+ dm     - The DM
. dim    - The spatial dimension
. anchor - The anchor point, the input point can be no more than maxCell away from it
- in     - The input coordinate point (dim numbers)

  Output Parameter:
. out - The localized coordinate point

  Level: developer

  Note: This is meant to get a set of coordinates close to each other, as in a cell. The anchor is usually the one of the vertices on a containing cell

.seealso: `DMLocalizeCoordinates()`, `DMLocalizeAddCoordinate()`
*/
PetscErrorCode DMLocalizeCoordinate_Internal(DM dm, PetscInt dim, const PetscScalar anchor[], const PetscScalar in[], PetscScalar out[])
{
  PetscInt d;

  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    for (d = 0; d < dim; ++d) {
      if ((dm->L[d] > 0.0) && (PetscAbsScalar(anchor[d] - in[d]) > dm->maxCell[d])) {
        out[d] = PetscRealPart(anchor[d]) > PetscRealPart(in[d]) ? dm->L[d] + in[d] : in[d] - dm->L[d];
      } else {
        out[d] = in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMLocalizeCoordinateReal_Internal(DM dm, PetscInt dim, const PetscReal anchor[], const PetscReal in[], PetscReal out[])
{
  PetscInt d;

  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    for (d = 0; d < dim; ++d) {
      if ((dm->L[d] > 0.0) && (PetscAbsReal(anchor[d] - in[d]) > dm->maxCell[d])) {
        out[d] = anchor[d] > in[d] ? dm->L[d] + in[d] : in[d] - dm->L[d];
      } else {
        out[d] = in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMLocalizeAddCoordinate_Internal - If a mesh is periodic, and the input point is far from the anchor, pick the coordinate sheet of the torus which moves it closer.

  Input Parameters:
+ dm     - The DM
. dim    - The spatial dimension
. anchor - The anchor point, the input point can be no more than maxCell away from it
. in     - The input coordinate delta (dim numbers)
- out    - The input coordinate point (dim numbers)

  Output Parameter:
. out    - The localized coordinate in + out

  Level: developer

  Note: This is meant to get a set of coordinates close to each other, as in a cell. The anchor is usually the one of the vertices on a containing cell

.seealso: `DMLocalizeCoordinates()`, `DMLocalizeCoordinate()`
*/
PetscErrorCode DMLocalizeAddCoordinate_Internal(DM dm, PetscInt dim, const PetscScalar anchor[], const PetscScalar in[], PetscScalar out[])
{
  PetscInt d;

  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] += in[d];
  } else {
    for (d = 0; d < dim; ++d) {
      const PetscReal maxC = dm->maxCell[d];

      if ((dm->L[d] > 0.0) && (PetscAbsScalar(anchor[d] - in[d]) > maxC)) {
        const PetscScalar newCoord = PetscRealPart(anchor[d]) > PetscRealPart(in[d]) ? dm->L[d] + in[d] : in[d] - dm->L[d];

        if (PetscAbsScalar(newCoord - anchor[d]) > maxC)
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT "-Coordinate %g more than %g away from anchor %g", d, (double) PetscRealPart(in[d]), (double) maxC, (double) PetscRealPart(anchor[d]));
        out[d] += newCoord;
      } else {
        out[d] += in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalizedLocal - Check if the DM coordinates have been localized for cells on this process

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
  areLocalized - True if localized

  Level: developer

.seealso: `DMLocalizeCoordinates()`, `DMGetCoordinatesLocalized()`, `DMSetPeriodicity()`
@*/
PetscErrorCode DMGetCoordinatesLocalizedLocal(DM dm, PetscBool *areLocalized)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(areLocalized, 2);
  *areLocalized = dm->coordinates[1].dim < 0 ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalized - Check if the DM coordinates have been localized for cells

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameter:
  areLocalized - True if localized

  Level: developer

.seealso: `DMLocalizeCoordinates()`, `DMSetPeriodicity()`, `DMGetCoordinatesLocalizedLocal()`
@*/
PetscErrorCode DMGetCoordinatesLocalized(DM dm,PetscBool *areLocalized)
{
  PetscBool localized;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(areLocalized, 2);
  PetscCall(DMGetCoordinatesLocalizedLocal(dm, &localized));
  PetscCall(MPIU_Allreduce(&localized, areLocalized, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm)));
  PetscFunctionReturn(0);
}

/*@
  DMLocalizeCoordinates - If a mesh is periodic, create local coordinates for cells having periodic faces

  Collective on dm

  Input Parameter:
. dm - The DM

  Level: developer

.seealso: `DMSetPeriodicity()`, `DMLocalizeCoordinate()`, `DMLocalizeAddCoordinate()`
@*/
PetscErrorCode DMLocalizeCoordinates(DM dm)
{
  DM               cdm, cdgdm, cplex, plex;
  PetscSection     cs, csDG;
  Vec              coordinates, cVec;
  PetscScalar     *coordsDG, *anchor, *localized;
  const PetscReal *Lstart, *L;
  PetscInt         Nc, vStart, vEnd, sStart, sEnd, newStart = PETSC_MAX_INT, newEnd = PETSC_MIN_INT, bs, coordSize;
  PetscBool        isLocalized, sparseLocalize = dm->sparseLocalize, useDG = PETSC_FALSE, useDGGlobal;
  PetscInt         maxHeight = 0, h;
  PetscInt        *pStart = NULL, *pEnd = NULL;
  MPI_Comm         comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetPeriodicity(dm, NULL, &Lstart, &L));
  /* Cannot automatically localize without L and maxCell right now */
  if (!L) PetscFunctionReturn(0);
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetCoordinatesLocalized(dm, &isLocalized));
  if (isLocalized) PetscFunctionReturn(0);

  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMConvert(dm,  DMPLEX, &plex));
  PetscCall(DMConvert(cdm, DMPLEX, &cplex));
  if (cplex) {
    PetscCall(DMPlexGetDepthStratum(cplex, 0, &vStart, &vEnd));
    PetscCall(DMPlexGetMaxProjectionHeight(cplex, &maxHeight));
    PetscCall(DMGetWorkArray(dm, 2*(maxHeight + 1), MPIU_INT, &pStart));
    pEnd     = &pStart[maxHeight + 1];
    newStart = vStart;
    newEnd   = vEnd;
    for (h = 0; h <= maxHeight; h++) {
      PetscCall(DMPlexGetHeightStratum(cplex, h, &pStart[h], &pEnd[h]));
      newStart = PetscMin(newStart, pStart[h]);
      newEnd   = PetscMax(newEnd, pEnd[h]);
    }
  } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Coordinate localization requires a DMPLEX coordinate DM");
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCheck(coordinates, comm, PETSC_ERR_SUP, "Missing local coordinate vector");
  PetscCall(DMGetCoordinateSection(dm, &cs));
  PetscCall(VecGetBlockSize(coordinates, &bs));
  PetscCall(PetscSectionGetChart(cs, &sStart, &sEnd));

  PetscCall(PetscSectionCreate(comm, &csDG));
  PetscCall(PetscSectionSetNumFields(csDG, 1));
  PetscCall(PetscSectionGetFieldComponents(cs, 0, &Nc));
  PetscCall(PetscSectionSetFieldComponents(csDG, 0, Nc));
  PetscCall(PetscSectionSetChart(csDG, newStart, newEnd));
  PetscCheck(bs == Nc, comm, PETSC_ERR_ARG_INCOMP, "Coordinate block size %" PetscInt_FMT " != %" PetscInt_FMT " number of components", bs, Nc);

  PetscCall(DMGetWorkArray(dm, 2 * Nc, MPIU_SCALAR, &anchor));
  localized = &anchor[Nc];
  for (h = 0; h <= maxHeight; h++) {
    PetscInt cStart = pStart[h], cEnd = pEnd[h], c;

    for (c = cStart; c < cEnd; ++c) {
      PetscScalar   *cellCoords = NULL;
      DMPolytopeType ct;
      PetscInt       dof, d, p;

      PetscCall(DMPlexGetCellType(plex, c, &ct));
      if (ct == DM_POLYTOPE_FV_GHOST) continue;
      PetscCall(DMPlexVecGetClosure(cplex, cs, coordinates, c, &dof, &cellCoords));
      PetscCheck(!(dof % Nc), comm, PETSC_ERR_ARG_INCOMP, "Coordinate size on cell %" PetscInt_FMT " closure %" PetscInt_FMT " not divisible by %" PetscInt_FMT " number of components", c, dof, Nc);
      for (d = 0; d < Nc; ++d) anchor[d] = cellCoords[d];
      for (p = 0; p < dof/Nc; ++p) {
        PetscCall(DMLocalizeCoordinate_Internal(dm, Nc, anchor, &cellCoords[p*Nc], localized));
        for (d = 0; d < Nc; ++d) if (cellCoords[p*Nc + d] != localized[d]) break;
        if (d < Nc) break;
      }
      if (p < dof/Nc) useDG = PETSC_TRUE;
      if (p < dof/Nc || !sparseLocalize) {
        PetscCall(PetscSectionSetDof(csDG, c, dof));
        PetscCall(PetscSectionSetFieldDof(csDG, c, 0, dof));
      }
      PetscCall(DMPlexVecRestoreClosure(cplex, cs, coordinates, c, &dof, &cellCoords));
    }
  }
  PetscCallMPI(MPI_Allreduce(&useDG, &useDGGlobal, 1, MPIU_BOOL, MPI_LOR, comm));
  if (!useDGGlobal) goto end;

  PetscCall(PetscSectionSetUp(csDG));
  PetscCall(PetscSectionGetStorageSize(csDG, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &cVec));
  PetscCall(PetscObjectSetName((PetscObject) cVec, "coordinates"));
  PetscCall(VecSetBlockSize(cVec, bs));
  PetscCall(VecSetSizes(cVec, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetType(cVec, VECSTANDARD));
  PetscCall(VecGetArray(cVec, &coordsDG));
  for (h = 0; h <= maxHeight; h++) {
    PetscInt cStart = pStart[h], cEnd = pEnd[h], c;

    for (c = cStart; c < cEnd; ++c) {
      PetscScalar *cellCoords = NULL;
      PetscInt     p = 0, q, dof, cdof, d, offDG;

      PetscCall(PetscSectionGetDof(csDG, c, &cdof));
      if (!cdof) continue;
      PetscCall(DMPlexVecGetClosure(cplex, cs, coordinates, c, &dof, &cellCoords));
      PetscCall(PetscSectionGetOffset(csDG, c, &offDG));
      // TODO The coordinates are set in closure order, which might not be the tensor order
      for (q = 0; q < dof/Nc; ++q) {
        // Select a trial anchor
        for (d = 0; d < Nc; ++d) anchor[d] = cellCoords[q*Nc+d];
        for (p = 0; p < dof/Nc; ++p) {
          PetscCall(DMLocalizeCoordinate_Internal(dm, Nc, anchor, &cellCoords[p*Nc], &coordsDG[offDG + p*Nc]));
          // We need the cell to fit into the torus [lower, lower+L)
          for (d = 0; d < Nc; ++d)
            if (L[d] > 0. && ((PetscRealPart(coordsDG[offDG + p*Nc + d]) < (Lstart ? Lstart[d] : 0.)) || (PetscRealPart(coordsDG[offDG + p*Nc + d]) > (Lstart ? Lstart[d] : 0.)+L[d]))) break;
          if (d < Nc) break;
        }
        if (p == dof/Nc) break;
      }
      PetscCheck(p == dof/Nc, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " does not fit into the torus %s[0, L]", c, Lstart ? "Lstart + " : "");
      PetscCall(DMPlexVecRestoreClosure(cplex, cs, coordinates, c, &dof, &cellCoords));
    }
  }
  PetscCall(VecRestoreArray(cVec, &coordsDG));
  PetscCall(DMClone(cdm, &cdgdm));
  PetscCall(DMSetCellCoordinateDM(dm, cdgdm));
  PetscCall(DMSetCellCoordinateSection(dm, PETSC_DETERMINE, csDG));
  PetscCall(DMSetCellCoordinatesLocal(dm, cVec));
  PetscCall(VecDestroy(&cVec));
  // Convert the discretization
  {
    PetscFE         fe, dgfe;
    PetscSpace      P;
    PetscDualSpace  Q, dgQ;
    PetscQuadrature q, fq;
    PetscClassId    id;

    PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *) &fe));
    PetscCall(PetscObjectGetClassId((PetscObject) fe, &id));
    if (id == PETSCFE_CLASSID) {
      PetscCall(PetscFEGetBasisSpace(fe, &P));
      PetscCall(PetscObjectReference((PetscObject) P));
      PetscCall(PetscFEGetDualSpace(fe, &Q));
      PetscCall(PetscDualSpaceDuplicate(Q, &dgQ));
      PetscCall(PetscDualSpaceLagrangeSetContinuity(dgQ, PETSC_FALSE));
      PetscCall(PetscDualSpaceSetUp(dgQ));
      PetscCall(PetscFEGetQuadrature(fe, &q));
      PetscCall(PetscObjectReference((PetscObject) q));
      PetscCall(PetscFEGetFaceQuadrature(fe, &fq));
      PetscCall(PetscObjectReference((PetscObject) fq));
      PetscCall(PetscFECreateFromSpaces(P, dgQ, q, fq, &dgfe));
      PetscCall(DMSetField(cdgdm, 0, NULL, (PetscObject) dgfe));
      PetscCall(PetscFEDestroy(&dgfe));
      PetscCall(DMCreateDS(cdgdm));
    }
  }
  PetscCall(DMDestroy(&cdgdm));

end:
  PetscCall(DMRestoreWorkArray(dm, 2 * bs, MPIU_SCALAR, &anchor));
  PetscCall(DMRestoreWorkArray(dm, 2*(maxHeight + 1), MPIU_INT, &pStart));
  PetscCall(PetscSectionDestroy(&csDG));
  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&cplex));
  PetscFunctionReturn(0);
}
