#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petscsf.h>
#include <petscds.h>
#include <petscdraw.h>
#include <petscdmfield.h>
#include <petscdmplextransform.h>

/* Logging support */
PetscLogEvent DMPLEX_Interpolate, DMPLEX_Partition, DMPLEX_Distribute, DMPLEX_DistributeCones, DMPLEX_DistributeLabels, DMPLEX_DistributeSF, DMPLEX_DistributeOverlap, DMPLEX_DistributeField, DMPLEX_DistributeData, DMPLEX_Migrate, DMPLEX_InterpolateSF, DMPLEX_GlobalToNaturalBegin, DMPLEX_GlobalToNaturalEnd, DMPLEX_NaturalToGlobalBegin, DMPLEX_NaturalToGlobalEnd, DMPLEX_Stratify, DMPLEX_Symmetrize, DMPLEX_Preallocate, DMPLEX_ResidualFEM, DMPLEX_JacobianFEM, DMPLEX_InterpolatorFEM, DMPLEX_InjectorFEM, DMPLEX_IntegralFEM, DMPLEX_CreateGmsh, DMPLEX_RebalanceSharedPoints, DMPLEX_PartSelf, DMPLEX_PartLabelInvert, DMPLEX_PartLabelCreateSF, DMPLEX_PartStratSF, DMPLEX_CreatePointSF,DMPLEX_LocatePoints,DMPLEX_TopologyView,DMPLEX_LabelsView,DMPLEX_CoordinatesView,DMPLEX_SectionView,DMPLEX_GlobalVectorView,DMPLEX_LocalVectorView,DMPLEX_TopologyLoad,DMPLEX_LabelsLoad,DMPLEX_CoordinatesLoad,DMPLEX_SectionLoad,DMPLEX_GlobalVectorLoad,DMPLEX_LocalVectorLoad;

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

/*@
  DMPlexIsSimplex - Is the first cell in this mesh a simplex?

  Input Parameter:
. dm      - The DMPlex object

  Output Parameter:
. simplex - Flag checking for a simplex

  Note: This just gives the first range of cells found. If the mesh has several cell types, it will only give the first.
  If the mesh has no cells, this returns PETSC_FALSE.

  Level: intermediate

.seealso DMPlexGetSimplexOrBoxCells(), DMPlexGetCellType(), DMPlexGetHeightStratum(), DMPolytopeTypeGetNumVertices()
@*/
PetscErrorCode DMPlexIsSimplex(DM dm, PetscBool *simplex)
{
  DMPolytopeType ct;
  PetscInt       cStart, cEnd;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  if (cEnd <= cStart) {*simplex = PETSC_FALSE; PetscFunctionReturn(0);}
  CHKERRQ(DMPlexGetCellType(dm, cStart, &ct));
  *simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetSimplexOrBoxCells - Get the range of cells which are neither prisms nor ghost FV cells

  Input Parameters:
+ dm     - The DMPlex object
- height - The cell height in the Plex, 0 is the default

  Output Parameters:
+ cStart - The first "normal" cell
- cEnd   - The upper bound on "normal"" cells

  Note: This just gives the first range of cells found. If the mesh has several cell types, it will only give the first.

  Level: developer

.seealso DMPlexConstructGhostCells(), DMPlexGetGhostCellStratum()
@*/
PetscErrorCode DMPlexGetSimplexOrBoxCells(DM dm, PetscInt height, PetscInt *cStart, PetscInt *cEnd)
{
  DMPolytopeType ct = DM_POLYTOPE_UNKNOWN;
  PetscInt       cS, cE, c;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetHeightStratum(dm, PetscMax(height, 0), &cS, &cE));
  for (c = cS; c < cE; ++c) {
    DMPolytopeType cct;

    CHKERRQ(DMPlexGetCellType(dm, c, &cct));
    if ((PetscInt) cct < 0) break;
    switch (cct) {
      case DM_POLYTOPE_POINT:
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_TRIANGLE:
      case DM_POLYTOPE_QUADRILATERAL:
      case DM_POLYTOPE_TETRAHEDRON:
      case DM_POLYTOPE_HEXAHEDRON:
        ct = cct;
        break;
      default: break;
    }
    if (ct != DM_POLYTOPE_UNKNOWN) break;
  }
  if (ct != DM_POLYTOPE_UNKNOWN) {
    DMLabel ctLabel;

    CHKERRQ(DMPlexGetCellTypeLabel(dm, &ctLabel));
    CHKERRQ(DMLabelGetStratumBounds(ctLabel, ct, &cS, &cE));
  }
  if (cStart) *cStart = cS;
  if (cEnd)   *cEnd   = cE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetFieldType_Internal(DM dm, PetscSection section, PetscInt field, PetscInt *sStart, PetscInt *sEnd, PetscViewerVTKFieldType *ft)
{
  PetscInt       cdim, pStart, pEnd, vStart, vEnd, cStart, cEnd;
  PetscInt       vcdof[2] = {0,0}, globalvcdof[2];

  PetscFunctionBegin;
  *ft  = PETSC_VTK_INVALID;
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  if (field >= 0) {
    if ((vStart >= pStart) && (vStart < pEnd)) CHKERRQ(PetscSectionGetFieldDof(section, vStart, field, &vcdof[0]));
    if ((cStart >= pStart) && (cStart < pEnd)) CHKERRQ(PetscSectionGetFieldDof(section, cStart, field, &vcdof[1]));
  } else {
    if ((vStart >= pStart) && (vStart < pEnd)) CHKERRQ(PetscSectionGetDof(section, vStart, &vcdof[0]));
    if ((cStart >= pStart) && (cStart < pEnd)) CHKERRQ(PetscSectionGetDof(section, cStart, &vcdof[1]));
  }
  CHKERRMPI(MPI_Allreduce(vcdof, globalvcdof, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  if (globalvcdof[0]) {
    *sStart = vStart;
    *sEnd   = vEnd;
    if (globalvcdof[0] == cdim) *ft = PETSC_VTK_POINT_VECTOR_FIELD;
    else                        *ft = PETSC_VTK_POINT_FIELD;
  } else if (globalvcdof[1]) {
    *sStart = cStart;
    *sEnd   = cEnd;
    if (globalvcdof[1] == cdim) *ft = PETSC_VTK_CELL_VECTOR_FIELD;
    else                        *ft = PETSC_VTK_CELL_FIELD;
  } else {
    if (field >= 0) {
      const char *fieldname;

      CHKERRQ(PetscSectionGetFieldName(section, field, &fieldname));
      CHKERRQ(PetscInfo((PetscObject) dm, "Could not classify VTK output type of section field %D \"%s\"\n", field, fieldname));
    } else {
      CHKERRQ(PetscInfo((PetscObject) dm, "Could not classify VTK output type of section\"%s\"\n"));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexVecView1D - Plot many 1D solutions on the same line graph

  Collective on dm

  Input Parameters:
+ dm - The DMPlex
. n  - The number of vectors
. u  - The array of local vectors
- viewer - The Draw viewer

  Level: advanced

.seealso: VecViewFromOptions(), VecView()
@*/
PetscErrorCode DMPlexVecView1D(DM dm, PetscInt n, Vec u[], PetscViewer viewer)
{
  PetscDS            ds;
  PetscDraw          draw = NULL;
  PetscDrawLG        lg;
  Vec                coordinates;
  const PetscScalar *coords, **sol;
  PetscReal         *vals;
  PetscInt          *Nc;
  PetscInt           Nf, f, c, Nl, l, i, vStart, vEnd, v;
  char             **names;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(PetscDSGetTotalComponents(ds, &Nl));
  CHKERRQ(PetscDSGetComponents(ds, &Nc));

  CHKERRQ(PetscViewerDrawGetDraw(viewer, 0, &draw));
  if (!draw) PetscFunctionReturn(0);
  CHKERRQ(PetscDrawLGCreate(draw, n*Nl, &lg));

  CHKERRQ(PetscMalloc3(n, &sol, n*Nl, &names, n*Nl, &vals));
  for (i = 0, l = 0; i < n; ++i) {
    const char *vname;

    CHKERRQ(PetscObjectGetName((PetscObject) u[i], &vname));
    for (f = 0; f < Nf; ++f) {
      PetscObject disc;
      const char *fname;
      char        tmpname[PETSC_MAX_PATH_LEN];

      CHKERRQ(PetscDSGetDiscretization(ds, f, &disc));
      /* TODO Create names for components */
      for (c = 0; c < Nc[f]; ++c, ++l) {
        CHKERRQ(PetscObjectGetName(disc, &fname));
        CHKERRQ(PetscStrcpy(tmpname, vname));
        CHKERRQ(PetscStrlcat(tmpname, ":", PETSC_MAX_PATH_LEN));
        CHKERRQ(PetscStrlcat(tmpname, fname, PETSC_MAX_PATH_LEN));
        CHKERRQ(PetscStrallocpy(tmpname, &names[l]));
      }
    }
  }
  CHKERRQ(PetscDrawLGSetLegend(lg, (const char *const *) names));
  /* Just add P_1 support for now */
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(VecGetArrayRead(coordinates, &coords));
  for (i = 0; i < n; ++i) CHKERRQ(VecGetArrayRead(u[i], &sol[i]));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *x, *svals;

    CHKERRQ(DMPlexPointLocalRead(dm, v, coords, &x));
    for (i = 0; i < n; ++i) {
      CHKERRQ(DMPlexPointLocalRead(dm, v, sol[i], &svals));
      for (l = 0; l < Nl; ++l) vals[i*Nl + l] = PetscRealPart(svals[l]);
    }
    CHKERRQ(PetscDrawLGAddCommonPoint(lg, PetscRealPart(x[0]), vals));
  }
  CHKERRQ(VecRestoreArrayRead(coordinates, &coords));
  for (i = 0; i < n; ++i) CHKERRQ(VecRestoreArrayRead(u[i], &sol[i]));
  for (l = 0; l < n*Nl; ++l) CHKERRQ(PetscFree(names[l]));
  CHKERRQ(PetscFree3(sol, names, vals));

  CHKERRQ(PetscDrawLGDraw(lg));
  CHKERRQ(PetscDrawLGDestroy(&lg));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Plex_Local_Draw_1D(Vec u, PetscViewer viewer)
{
  DM             dm;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(u, &dm));
  CHKERRQ(DMPlexVecView1D(dm, 1, &u, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Plex_Local_Draw_2D(Vec v, PetscViewer viewer)
{
  DM                 dm;
  PetscSection       s;
  PetscDraw          draw, popup;
  DM                 cdm;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *array;
  PetscReal          bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscReal          vbound[2], time;
  PetscBool          flg;
  PetscInt           dim, Nf, f, Nc, comp, vStart, vEnd, cStart, cEnd, c, N, level, step, w = 0;
  const char        *name;
  char               title[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(PetscViewerDrawGetDraw(viewer, 0, &draw));
  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMGetCoordinateDim(dm, &dim));
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscSectionGetNumFields(s, &Nf));
  CHKERRQ(DMGetCoarsenLevel(dm, &level));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
  CHKERRQ(DMGetOutputSequenceNumber(dm, &step, &time));

  CHKERRQ(VecGetLocalSize(coordinates, &N));
  CHKERRQ(VecGetArrayRead(coordinates, &coords));
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));   bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c+1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c+1]));
  }
  CHKERRQ(VecRestoreArrayRead(coordinates, &coords));
  CHKERRQ(PetscDrawClear(draw));

  /* Could implement something like DMDASelectFields() */
  for (f = 0; f < Nf; ++f) {
    DM   fdm = dm;
    Vec  fv  = v;
    IS   fis;
    char prefix[PETSC_MAX_PATH_LEN];
    const char *fname;

    CHKERRQ(PetscSectionGetFieldComponents(s, f, &Nc));
    CHKERRQ(PetscSectionGetFieldName(s, f, &fname));

    if (v->hdr.prefix) CHKERRQ(PetscStrncpy(prefix, v->hdr.prefix,sizeof(prefix)));
    else               {prefix[0] = '\0';}
    if (Nf > 1) {
      CHKERRQ(DMCreateSubDM(dm, 1, &f, &fis, &fdm));
      CHKERRQ(VecGetSubVector(v, fis, &fv));
      CHKERRQ(PetscStrlcat(prefix, fname,sizeof(prefix)));
      CHKERRQ(PetscStrlcat(prefix, "_",sizeof(prefix)));
    }
    for (comp = 0; comp < Nc; ++comp, ++w) {
      PetscInt nmax = 2;

      CHKERRQ(PetscViewerDrawGetDraw(viewer, w, &draw));
      if (Nc > 1) CHKERRQ(PetscSNPrintf(title, sizeof(title), "%s:%s_%D Step: %D Time: %.4g", name, fname, comp, step, time));
      else        CHKERRQ(PetscSNPrintf(title, sizeof(title), "%s:%s Step: %D Time: %.4g", name, fname, step, time));
      CHKERRQ(PetscDrawSetTitle(draw, title));

      /* TODO Get max and min only for this component */
      CHKERRQ(PetscOptionsGetRealArray(NULL, prefix, "-vec_view_bounds", vbound, &nmax, &flg));
      if (!flg) {
        CHKERRQ(VecMin(fv, NULL, &vbound[0]));
        CHKERRQ(VecMax(fv, NULL, &vbound[1]));
        if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
      }
      CHKERRQ(PetscDrawGetPopup(draw, &popup));
      CHKERRQ(PetscDrawScalePopup(popup, vbound[0], vbound[1]));
      CHKERRQ(PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]));

      CHKERRQ(VecGetArrayRead(fv, &array));
      for (c = cStart; c < cEnd; ++c) {
        PetscScalar *coords = NULL, *a = NULL;
        PetscInt     numCoords, color[4] = {-1,-1,-1,-1};

        CHKERRQ(DMPlexPointLocalRead(fdm, c, array, &a));
        if (a) {
          color[0] = PetscDrawRealToColor(PetscRealPart(a[comp]), vbound[0], vbound[1]);
          color[1] = color[2] = color[3] = color[0];
        } else {
          PetscScalar *vals = NULL;
          PetscInt     numVals, va;

          CHKERRQ(DMPlexVecGetClosure(fdm, NULL, fv, c, &numVals, &vals));
          PetscCheckFalse(numVals % Nc,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of components %D does not divide the number of values in the closure %D", Nc, numVals);
          switch (numVals/Nc) {
          case 3: /* P1 Triangle */
          case 4: /* P1 Quadrangle */
            for (va = 0; va < numVals/Nc; ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc+comp]), vbound[0], vbound[1]);
            break;
          case 6: /* P2 Triangle */
          case 8: /* P2 Quadrangle */
            for (va = 0; va < numVals/(Nc*2); ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc+comp + numVals/(Nc*2)]), vbound[0], vbound[1]);
            break;
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of values for cell closure %D cannot be handled", numVals/Nc);
          }
          CHKERRQ(DMPlexVecRestoreClosure(fdm, NULL, fv, c, &numVals, &vals));
        }
        CHKERRQ(DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords));
        switch (numCoords) {
        case 6:
        case 12: /* Localized triangle */
          CHKERRQ(PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]));
          break;
        case 8:
        case 16: /* Localized quadrilateral */
          CHKERRQ(PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]));
          CHKERRQ(PetscDrawTriangle(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), color[2], color[3], color[0]));
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells with %D coordinates", numCoords);
        }
        CHKERRQ(DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords));
      }
      CHKERRQ(VecRestoreArrayRead(fv, &array));
      CHKERRQ(PetscDrawFlush(draw));
      CHKERRQ(PetscDrawPause(draw));
      CHKERRQ(PetscDrawSave(draw));
    }
    if (Nf > 1) {
      CHKERRQ(VecRestoreSubVector(v, fis, &fv));
      CHKERRQ(ISDestroy(&fis));
      CHKERRQ(DMDestroy(&fdm));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Plex_Local_Draw(Vec v, PetscViewer viewer)
{
  DM        dm;
  PetscDraw draw;
  PetscInt  dim;
  PetscBool isnull;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerDrawGetDraw(viewer, 0, &draw));
  CHKERRQ(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(0);

  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMGetCoordinateDim(dm, &dim));
  switch (dim) {
  case 1: CHKERRQ(VecView_Plex_Local_Draw_1D(v, viewer));break;
  case 2: CHKERRQ(VecView_Plex_Local_Draw_2D(v, viewer));break;
  default: SETERRQ(PetscObjectComm((PetscObject) v), PETSC_ERR_SUP, "Cannot draw meshes of dimension %" PetscInt_FMT ". Try PETSCVIEWERGLVIS", dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Plex_Local_VTK(Vec v, PetscViewer viewer)
{
  DM                      dm;
  Vec                     locv;
  const char              *name;
  PetscSection            section;
  PetscInt                pStart, pEnd;
  PetscInt                numFields;
  PetscViewerVTKFieldType ft;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMCreateLocalVector(dm, &locv)); /* VTK viewer requires exclusive ownership of the vector */
  CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
  CHKERRQ(PetscObjectSetName((PetscObject) locv, name));
  CHKERRQ(VecCopy(v, locv));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  if (!numFields) {
    CHKERRQ(DMPlexGetFieldType_Internal(dm, section, PETSC_DETERMINE, &pStart, &pEnd, &ft));
    CHKERRQ(PetscViewerVTKAddField(viewer, (PetscObject) dm, DMPlexVTKWriteAll, PETSC_DEFAULT, ft, PETSC_TRUE,(PetscObject) locv));
  } else {
    PetscInt f;

    for (f = 0; f < numFields; f++) {
      CHKERRQ(DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft));
      if (ft == PETSC_VTK_INVALID) continue;
      CHKERRQ(PetscObjectReference((PetscObject)locv));
      CHKERRQ(PetscViewerVTKAddField(viewer, (PetscObject) dm, DMPlexVTKWriteAll, f, ft, PETSC_TRUE,(PetscObject) locv));
    }
    CHKERRQ(VecDestroy(&locv));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_Local(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isglvis;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis));
  if (isvtk || ishdf5 || isdraw || isglvis) {
    PetscInt    i,numFields;
    PetscObject fe;
    PetscBool   fem = PETSC_FALSE;
    Vec         locv = v;
    const char  *name;
    PetscInt    step;
    PetscReal   time;

    CHKERRQ(DMGetNumFields(dm, &numFields));
    for (i=0; i<numFields; i++) {
      CHKERRQ(DMGetField(dm, i, NULL, &fe));
      if (fe->classid == PETSCFE_CLASSID) { fem = PETSC_TRUE; break; }
    }
    if (fem) {
      PetscObject isZero;

      CHKERRQ(DMGetLocalVector(dm, &locv));
      CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
      CHKERRQ(PetscObjectSetName((PetscObject) locv, name));
      CHKERRQ(PetscObjectQuery((PetscObject) v, "__Vec_bc_zero__", &isZero));
      CHKERRQ(PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", isZero));
      CHKERRQ(VecCopy(v, locv));
      CHKERRQ(DMGetOutputSequenceNumber(dm, NULL, &time));
      CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL));
    }
    if (isvtk) {
      CHKERRQ(VecView_Plex_Local_VTK(locv, viewer));
    } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      CHKERRQ(VecView_Plex_Local_HDF5_Internal(locv, viewer));
#else
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
    } else if (isdraw) {
      CHKERRQ(VecView_Plex_Local_Draw(locv, viewer));
    } else if (isglvis) {
      CHKERRQ(DMGetOutputSequenceNumber(dm, &step, NULL));
      CHKERRQ(PetscViewerGLVisSetSnapId(viewer, step));
      CHKERRQ(VecView_GLVis(locv, viewer));
    }
    if (fem) {
      CHKERRQ(PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", NULL));
      CHKERRQ(DMRestoreLocalVector(dm, &locv));
    }
  } else {
    PetscBool isseq;

    CHKERRQ(PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq));
    if (isseq) CHKERRQ(VecView_Seq(v, viewer));
    else       CHKERRQ(VecView_MPI(v, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex(Vec v, PetscViewer viewer)
{
  DM        dm;
  PetscBool isvtk, ishdf5, isdraw, isglvis, isexodusii;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,      &isvtk));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,     &ishdf5));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,     &isdraw));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS,    &isglvis));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWEREXODUSII, &isexodusii));
  if (isvtk || isdraw || isglvis) {
    Vec         locv;
    PetscObject isZero;
    const char *name;

    CHKERRQ(DMGetLocalVector(dm, &locv));
    CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
    CHKERRQ(PetscObjectSetName((PetscObject) locv, name));
    CHKERRQ(DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv));
    CHKERRQ(DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv));
    CHKERRQ(PetscObjectQuery((PetscObject) v, "__Vec_bc_zero__", &isZero));
    CHKERRQ(PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", isZero));
    CHKERRQ(VecView_Plex_Local(locv, viewer));
    CHKERRQ(PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", NULL));
    CHKERRQ(DMRestoreLocalVector(dm, &locv));
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(VecView_Plex_HDF5_Internal(v, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isexodusii) {
#if defined(PETSC_HAVE_EXODUSII)
    CHKERRQ(VecView_PlexExodusII_Internal(v, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "ExodusII not supported in this build.\nPlease reconfigure using --download-exodusii");
#endif
  } else {
    PetscBool isseq;

    CHKERRQ(PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq));
    if (isseq) CHKERRQ(VecView_Seq(v, viewer));
    else       CHKERRQ(VecView_MPI(v, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_Native(Vec originalv, PetscViewer viewer)
{
  DM                dm;
  MPI_Comm          comm;
  PetscViewerFormat format;
  Vec               v;
  PetscBool         isvtk, ishdf5;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(originalv, &dm));
  CHKERRQ(PetscObjectGetComm((PetscObject) originalv, &comm));
  PetscCheck(dm,comm, PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,  &isvtk));
  if (format == PETSC_VIEWER_NATIVE) {
    /* Natural ordering is the common case for DMDA, NATIVE means plain vector, for PLEX is the opposite */
    /* this need a better fix */
    if (dm->useNatural) {
      if (dm->sfNatural) {
        const char *vecname;
        PetscInt    n, nroots;

        CHKERRQ(VecGetLocalSize(originalv, &n));
        CHKERRQ(PetscSFGetGraph(dm->sfNatural, &nroots, NULL, NULL, NULL));
        if (n == nroots) {
          CHKERRQ(DMGetGlobalVector(dm, &v));
          CHKERRQ(DMPlexGlobalToNaturalBegin(dm, originalv, v));
          CHKERRQ(DMPlexGlobalToNaturalEnd(dm, originalv, v));
          CHKERRQ(PetscObjectGetName((PetscObject) originalv, &vecname));
          CHKERRQ(PetscObjectSetName((PetscObject) v, vecname));
        } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "DM global to natural SF only handles global vectors");
      } else SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created");
    } else v = originalv;
  } else v = originalv;

  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(VecView_Plex_HDF5_Native_Internal(v, viewer));
#else
    SETERRQ(comm, PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isvtk) {
    SETERRQ(comm, PETSC_ERR_SUP, "VTK format does not support viewing in natural order. Please switch to HDF5.");
  } else {
    PetscBool isseq;

    CHKERRQ(PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq));
    if (isseq) CHKERRQ(VecView_Seq(v, viewer));
    else       CHKERRQ(VecView_MPI(v, viewer));
  }
  if (v != originalv) CHKERRQ(DMRestoreGlobalVector(dm, &v));
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_Local(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      ishdf5;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  if (ishdf5) {
    DM          dmBC;
    Vec         gv;
    const char *name;

    CHKERRQ(DMGetOutputDM(dm, &dmBC));
    CHKERRQ(DMGetGlobalVector(dmBC, &gv));
    CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
    CHKERRQ(PetscObjectSetName((PetscObject) gv, name));
    CHKERRQ(VecLoad_Default(gv, viewer));
    CHKERRQ(DMGlobalToLocalBegin(dmBC, gv, INSERT_VALUES, v));
    CHKERRQ(DMGlobalToLocalEnd(dmBC, gv, INSERT_VALUES, v));
    CHKERRQ(DMRestoreGlobalVector(dmBC, &gv));
  } else {
    CHKERRQ(VecLoad_Default(v, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      ishdf5,isexodusii;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,     &ishdf5));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWEREXODUSII, &isexodusii));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(VecLoad_Plex_HDF5_Internal(v, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isexodusii) {
#if defined(PETSC_HAVE_EXODUSII)
    CHKERRQ(VecLoad_PlexExodusII_Internal(v, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "ExodusII not supported in this build.\nPlease reconfigure using --download-exodusii");
#endif
  } else {
    CHKERRQ(VecLoad_Default(v, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_Native(Vec originalv, PetscViewer viewer)
{
  DM                dm;
  PetscViewerFormat format;
  PetscBool         ishdf5;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(originalv, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject) originalv), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  if (format == PETSC_VIEWER_NATIVE) {
    if (dm->useNatural) {
      if (dm->sfNatural) {
        if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
          Vec         v;
          const char *vecname;

          CHKERRQ(DMGetGlobalVector(dm, &v));
          CHKERRQ(PetscObjectGetName((PetscObject) originalv, &vecname));
          CHKERRQ(PetscObjectSetName((PetscObject) v, vecname));
          CHKERRQ(VecLoad_Plex_HDF5_Native_Internal(v, viewer));
          CHKERRQ(DMPlexNaturalToGlobalBegin(dm, v, originalv));
          CHKERRQ(DMPlexNaturalToGlobalEnd(dm, v, originalv));
          CHKERRQ(DMRestoreGlobalVector(dm, &v));
#else
          SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
        } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Reading in natural order is not supported for anything but HDF5.");
      }
    } else {
      CHKERRQ(VecLoad_Default(originalv, viewer));
    }
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexView_Ascii_Geometry(DM dm, PetscViewer viewer)
{
  PetscSection       coordSection;
  Vec                coordinates;
  DMLabel            depthLabel, celltypeLabel;
  const char        *name[4];
  const PetscScalar *a;
  PetscInt           dim, pStart, pEnd, cStart, cEnd, c;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &celltypeLabel));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscSectionGetChart(coordSection, &pStart, &pEnd));
  CHKERRQ(VecGetArrayRead(coordinates, &a));
  name[0]     = "vertex";
  name[1]     = "edge";
  name[dim-1] = "face";
  name[dim]   = "cell";
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl, ct;

    CHKERRQ(DMLabelGetValue(celltypeLabel, c, &ct));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Geometry for cell %D polytope type %s:\n", c, DMPolytopeTypes[ct]));
    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      PetscInt point = closure[cl], depth, dof, off, d, p;

      if ((point < pStart) || (point >= pEnd)) continue;
      CHKERRQ(PetscSectionGetDof(coordSection, point, &dof));
      if (!dof) continue;
      CHKERRQ(DMLabelGetValue(depthLabel, point, &depth));
      CHKERRQ(PetscSectionGetOffset(coordSection, point, &off));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s %D coords:", name[depth], point));
      for (p = 0; p < dof/dim; ++p) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, " ("));
        for (d = 0; d < dim; ++d) {
          if (d > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, ", "));
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "%g", (double) PetscRealPart(a[off+p*dim+d])));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer, ")"));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  CHKERRQ(VecRestoreArrayRead(coordinates, &a));
  PetscFunctionReturn(0);
}

typedef enum {CS_CARTESIAN, CS_POLAR, CS_CYLINDRICAL, CS_SPHERICAL} CoordSystem;
const char *CoordSystems[] = {"cartesian", "polar", "cylindrical", "spherical", "CoordSystem", "CS_", NULL};

static PetscErrorCode DMPlexView_Ascii_Coordinates(PetscViewer viewer, CoordSystem cs, PetscInt dim, const PetscScalar x[])
{
  PetscInt       i;

  PetscFunctionBegin;
  if (dim > 3) {
    for (i = 0; i < dim; ++i) CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double) PetscRealPart(x[i])));
  } else {
    PetscReal coords[3], trcoords[3];

    for (i = 0; i < dim; ++i) coords[i] = PetscRealPart(x[i]);
    switch (cs) {
      case CS_CARTESIAN: for (i = 0; i < dim; ++i) trcoords[i] = coords[i];break;
      case CS_POLAR:
        PetscCheckFalse(dim != 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Polar coordinates are for 2 dimension, not %D", dim);
        trcoords[0] = PetscSqrtReal(PetscSqr(coords[0]) + PetscSqr(coords[1]));
        trcoords[1] = PetscAtan2Real(coords[1], coords[0]);
        break;
      case CS_CYLINDRICAL:
        PetscCheckFalse(dim != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cylindrical coordinates are for 3 dimension, not %D", dim);
        trcoords[0] = PetscSqrtReal(PetscSqr(coords[0]) + PetscSqr(coords[1]));
        trcoords[1] = PetscAtan2Real(coords[1], coords[0]);
        trcoords[2] = coords[2];
        break;
      case CS_SPHERICAL:
        PetscCheckFalse(dim != 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Spherical coordinates are for 3 dimension, not %D", dim);
        trcoords[0] = PetscSqrtReal(PetscSqr(coords[0]) + PetscSqr(coords[1]) + PetscSqr(coords[2]));
        trcoords[1] = PetscAtan2Real(PetscSqrtReal(PetscSqr(coords[0]) + PetscSqr(coords[1])), coords[2]);
        trcoords[2] = PetscAtan2Real(coords[1], coords[0]);
        break;
    }
    for (i = 0; i < dim; ++i) CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, " %g", (double) trcoords[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Plex          *mesh = (DM_Plex*) dm->data;
  DM                cdm;
  PetscSection      coordSection;
  Vec               coordinates;
  PetscViewerFormat format;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    const char *name;
    PetscInt    dim, cellHeight, maxConeSize, maxSupportSize;
    PetscInt    pStart, pEnd, p, numLabels, l;
    PetscMPIInt rank, size;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
    CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
    if (name) CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s in %D dimension%s:\n", name, dim, dim == 1 ? "" : "s"));
    else      CHKERRQ(PetscViewerASCIIPrintf(viewer, "Mesh in %D dimension%s:\n", dim, dim == 1 ? "" : "s"));
    if (cellHeight) CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Cells are at height %D\n", cellHeight));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Supports:\n", name));
    CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Max support size: %D\n", rank, maxSupportSize));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      CHKERRQ(PetscSectionGetDof(mesh->supportSection, p, &dof));
      CHKERRQ(PetscSectionGetOffset(mesh->supportSection, p, &off));
      for (s = off; s < off+dof; ++s) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D ----> %D\n", rank, p, mesh->supports[s]));
      }
    }
    CHKERRQ(PetscViewerFlush(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Cones:\n", name));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Max cone size: %D\n", rank, maxConeSize));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
      CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
      for (c = off; c < off+dof; ++c) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D <---- %D (%D)\n", rank, p, mesh->cones[c], mesh->coneOrientations[c]));
      }
    }
    CHKERRQ(PetscViewerFlush(viewer));
    CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
    if (coordSection && coordinates) {
      CoordSystem        cs = CS_CARTESIAN;
      const PetscScalar *array;
      PetscInt           Nf, Nc, pStart, pEnd, p;
      PetscMPIInt        rank;
      const char        *name;

      CHKERRQ(PetscOptionsGetEnum(((PetscObject) viewer)->options, ((PetscObject) viewer)->prefix, "-dm_plex_view_coord_system", CoordSystems, (PetscEnum *) &cs, NULL));
      CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
      CHKERRQ(PetscSectionGetNumFields(coordSection, &Nf));
      PetscCheckFalse(Nf != 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate section should have 1 field, not %D", Nf);
      CHKERRQ(PetscSectionGetFieldComponents(coordSection, 0, &Nc));
      CHKERRQ(PetscSectionGetChart(coordSection, &pStart, &pEnd));
      CHKERRQ(PetscObjectGetName((PetscObject) coordinates, &name));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s with %D fields\n", name, Nf));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  field 0 with %D components\n", Nc));
      if (cs != CS_CARTESIAN) CHKERRQ(PetscViewerASCIIPrintf(viewer, "  output coordinate system: %s\n", CoordSystems[cs]));

      CHKERRQ(VecGetArrayRead(coordinates, &array));
      CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "Process %d:\n", rank));
      for (p = pStart; p < pEnd; ++p) {
        PetscInt dof, off;

        CHKERRQ(PetscSectionGetDof(coordSection, p, &dof));
        CHKERRQ(PetscSectionGetOffset(coordSection, p, &off));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "  (%4D) dim %2D offset %3D", p, dof, off));
        CHKERRQ(DMPlexView_Ascii_Coordinates(viewer, cs, dof, &array[off]));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
      }
      CHKERRQ(PetscViewerFlush(viewer));
      CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
      CHKERRQ(VecRestoreArrayRead(coordinates, &array));
    }
    CHKERRQ(DMGetNumLabels(dm, &numLabels));
    if (numLabels) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Labels:\n"));
    for (l = 0; l < numLabels; ++l) {
      DMLabel     label;
      PetscBool   isdepth;
      const char *name;

      CHKERRQ(DMGetLabelName(dm, l, &name));
      CHKERRQ(PetscStrcmp(name, "depth", &isdepth));
      if (isdepth) continue;
      CHKERRQ(DMGetLabel(dm, name, &label));
      CHKERRQ(DMLabelView(label, viewer));
    }
    if (size > 1) {
      PetscSF sf;

      CHKERRQ(DMGetPointSF(dm, &sf));
      CHKERRQ(PetscSFView(sf, viewer));
    }
    CHKERRQ(PetscViewerFlush(viewer));
  } else if (format == PETSC_VIEWER_ASCII_LATEX) {
    const char  *name, *color;
    const char  *defcolors[3]  = {"gray", "orange", "green"};
    const char  *deflcolors[4] = {"blue", "cyan", "red", "magenta"};
    char         lname[PETSC_MAX_PATH_LEN];
    PetscReal    scale         = 2.0;
    PetscReal    tikzscale     = 1.0;
    PetscBool    useNumbers    = PETSC_TRUE, drawNumbers[4], drawColors[4], useLabels, useColors, plotEdges, drawHasse = PETSC_FALSE;
    double       tcoords[3];
    PetscScalar *coords;
    PetscInt     numLabels, l, numColors, numLColors, dim, d, depth, cStart, cEnd, c, vStart, vEnd, v, eStart = 0, eEnd = 0, e, p, n;
    PetscMPIInt  rank, size;
    char         **names, **colors, **lcolors;
    PetscBool    flg, lflg;
    PetscBT      wp = NULL;
    PetscInt     pEnd, pStart;

    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMPlexGetDepth(dm, &depth));
    CHKERRQ(DMGetNumLabels(dm, &numLabels));
    numLabels  = PetscMax(numLabels, 10);
    numColors  = 10;
    numLColors = 10;
    CHKERRQ(PetscCalloc3(numLabels, &names, numColors, &colors, numLColors, &lcolors));
    CHKERRQ(PetscOptionsGetReal(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_scale", &scale, NULL));
    CHKERRQ(PetscOptionsGetReal(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_tikzscale", &tikzscale, NULL));
    CHKERRQ(PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_numbers", &useNumbers, NULL));
    for (d = 0; d < 4; ++d) drawNumbers[d] = useNumbers;
    for (d = 0; d < 4; ++d) drawColors[d]  = PETSC_TRUE;
    n = 4;
    CHKERRQ(PetscOptionsGetBoolArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_numbers_depth", drawNumbers, &n, &flg));
    PetscCheckFalse(flg && n != dim+1,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Number of flags %D != %D dim+1", n, dim+1);
    CHKERRQ(PetscOptionsGetBoolArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_colors_depth", drawColors, &n, &flg));
    PetscCheckFalse(flg && n != dim+1,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Number of flags %D != %D dim+1", n, dim+1);
    CHKERRQ(PetscOptionsGetStringArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_labels", names, &numLabels, &useLabels));
    if (!useLabels) numLabels = 0;
    CHKERRQ(PetscOptionsGetStringArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_colors", colors, &numColors, &useColors));
    if (!useColors) {
      numColors = 3;
      for (c = 0; c < numColors; ++c) CHKERRQ(PetscStrallocpy(defcolors[c], &colors[c]));
    }
    CHKERRQ(PetscOptionsGetStringArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_lcolors", lcolors, &numLColors, &useColors));
    if (!useColors) {
      numLColors = 4;
      for (c = 0; c < numLColors; ++c) CHKERRQ(PetscStrallocpy(deflcolors[c], &lcolors[c]));
    }
    CHKERRQ(PetscOptionsGetString(((PetscObject) viewer)->options, ((PetscObject) viewer)->prefix, "-dm_plex_view_label_filter", lname, sizeof(lname), &lflg));
    plotEdges = (PetscBool)(depth > 1 && drawNumbers[1] && dim < 3);
    CHKERRQ(PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_edges", &plotEdges, &flg));
    PetscCheckFalse(flg && plotEdges && depth < dim,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Mesh must be interpolated");
    if (depth < dim) plotEdges = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(((PetscObject) viewer)->options, ((PetscObject) viewer)->prefix, "-dm_plex_view_hasse", &drawHasse, NULL));

    /* filter points with labelvalue != labeldefaultvalue */
    CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    CHKERRQ(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    if (lflg) {
      DMLabel lbl;

      CHKERRQ(DMGetLabel(dm, lname, &lbl));
      if (lbl) {
        PetscInt val, defval;

        CHKERRQ(DMLabelGetDefaultValue(lbl, &defval));
        CHKERRQ(PetscBTCreate(pEnd-pStart, &wp));
        for (c = pStart;  c < pEnd; c++) {
          PetscInt *closure = NULL;
          PetscInt  closureSize;

          CHKERRQ(DMLabelGetValue(lbl, c, &val));
          if (val == defval) continue;

          CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
          for (p = 0; p < closureSize*2; p += 2) {
            CHKERRQ(PetscBTSet(wp, closure[p] - pStart));
          }
          CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
        }
      }
    }

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
    CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "\
\\documentclass[tikz]{standalone}\n\n\
\\usepackage{pgflibraryshapes}\n\
\\usetikzlibrary{backgrounds}\n\
\\usetikzlibrary{arrows}\n\
\\begin{document}\n"));
    if (size > 1) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s for process ", name));
      for (p = 0; p < size; ++p) {
        if (p > 0 && p == size-1) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, ", and ", colors[p%numColors], p));
        } else if (p > 0) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, ", ", colors[p%numColors], p));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "{\\textcolor{%s}%D}", colors[p%numColors], p));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer, ".\n\n\n"));
    }
    if (drawHasse) {
      PetscInt maxStratum = PetscMax(vEnd-vStart, PetscMax(eEnd-eStart, cEnd-cStart));

      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\vStart}{%D}\n", vStart));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\vEnd}{%D}\n", vEnd-1));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\numVertices}{%D}\n", vEnd-vStart));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\vShift}{%.2f}\n", 3 + (maxStratum-(vEnd-vStart))/2.));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\eStart}{%D}\n", eStart));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\eEnd}{%D}\n", eEnd-1));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\eShift}{%.2f}\n", 3 + (maxStratum-(eEnd-eStart))/2.));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\numEdges}{%D}\n", eEnd-eStart));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\cStart}{%D}\n", cStart));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\cEnd}{%D}\n", cEnd-1));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\numCells}{%D}\n", cEnd-cStart));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\newcommand{\\cShift}{%.2f}\n", 3 + (maxStratum-(cEnd-cStart))/2.));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\begin{tikzpicture}[scale = %g,font=\\fontsize{8}{8}\\selectfont]\n", (double) tikzscale));

    /* Plot vertices */
    CHKERRQ(VecGetArray(coordinates, &coords));
    CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  off, dof, d;
      PetscBool isLabeled = PETSC_FALSE;

      if (wp && !PetscBTLookup(wp,v - pStart)) continue;
      CHKERRQ(PetscSectionGetDof(coordSection, v, &dof));
      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\\path ("));
      PetscCheck(dof <= 3,PETSC_COMM_SELF,PETSC_ERR_PLIB,"coordSection vertex %D has dof %D > 3",v,dof);
      for (d = 0; d < dof; ++d) {
        tcoords[d] = (double) (scale*PetscRealPart(coords[off+d]));
        tcoords[d] = PetscAbs(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
      }
      /* Rotate coordinates since PGF makes z point out of the page instead of up */
      if (dim == 3) {PetscReal tmp = tcoords[1]; tcoords[1] = tcoords[2]; tcoords[2] = -tmp;}
      for (d = 0; d < dof; ++d) {
        if (d > 0) CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ","));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "%g", (double) tcoords[d]));
      }
      if (drawHasse) color = colors[0%numColors];
      else           color = colors[rank%numColors];
      for (l = 0; l < numLabels; ++l) {
        PetscInt val;
        CHKERRQ(DMGetLabelValue(dm, names[l], v, &val));
        if (val >= 0) {color = lcolors[l%numLColors]; isLabeled = PETSC_TRUE; break;}
      }
      if (drawNumbers[0]) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [draw,shape=circle,color=%s] {%D};\n", v, rank, color, v));
      } else if (drawColors[0]) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [fill,inner sep=%dpt,shape=circle,color=%s] {};\n", v, rank, !isLabeled ? 1 : 2, color));
      } else {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [] {};\n", v, rank));
      }
    }
    CHKERRQ(VecRestoreArray(coordinates, &coords));
    CHKERRQ(PetscViewerFlush(viewer));
    /* Plot edges */
    if (plotEdges) {
      CHKERRQ(VecGetArray(coordinates, &coords));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\path\n"));
      for (e = eStart; e < eEnd; ++e) {
        const PetscInt *cone;
        PetscInt        coneSize, offA, offB, dof, d;

        if (wp && !PetscBTLookup(wp,e - pStart)) continue;
        CHKERRQ(DMPlexGetConeSize(dm, e, &coneSize));
        PetscCheckFalse(coneSize != 2,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Edge %D cone should have two vertices, not %D", e, coneSize);
        CHKERRQ(DMPlexGetCone(dm, e, &cone));
        CHKERRQ(PetscSectionGetDof(coordSection, cone[0], &dof));
        CHKERRQ(PetscSectionGetOffset(coordSection, cone[0], &offA));
        CHKERRQ(PetscSectionGetOffset(coordSection, cone[1], &offB));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "("));
        for (d = 0; d < dof; ++d) {
          tcoords[d] = (double) (0.5*scale*PetscRealPart(coords[offA+d]+coords[offB+d]));
          tcoords[d] = PetscAbs(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
        }
        /* Rotate coordinates since PGF makes z point out of the page instead of up */
        if (dim == 3) {PetscReal tmp = tcoords[1]; tcoords[1] = tcoords[2]; tcoords[2] = -tmp;}
        for (d = 0; d < dof; ++d) {
          if (d > 0) CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ","));
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "%g", (double)tcoords[d]));
        }
        if (drawHasse) color = colors[1%numColors];
        else           color = colors[rank%numColors];
        for (l = 0; l < numLabels; ++l) {
          PetscInt val;
          CHKERRQ(DMGetLabelValue(dm, names[l], v, &val));
          if (val >= 0) {color = lcolors[l%numLColors]; break;}
        }
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [draw,shape=circle,color=%s] {%D} --\n", e, rank, color, e));
      }
      CHKERRQ(VecRestoreArray(coordinates, &coords));
      CHKERRQ(PetscViewerFlush(viewer));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "(0,0);\n"));
    }
    /* Plot cells */
    if (dim == 3 || !drawNumbers[1]) {
      for (e = eStart; e < eEnd; ++e) {
        const PetscInt *cone;

        if (wp && !PetscBTLookup(wp,e - pStart)) continue;
        color = colors[rank%numColors];
        for (l = 0; l < numLabels; ++l) {
          PetscInt val;
          CHKERRQ(DMGetLabelValue(dm, names[l], e, &val));
          if (val >= 0) {color = lcolors[l%numLColors]; break;}
        }
        CHKERRQ(DMPlexGetCone(dm, e, &cone));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] (%D_%d) -- (%D_%d);\n", color, cone[0], rank, cone[1], rank));
      }
    } else {
       DMPolytopeType ct;

      /* Drawing a 2D polygon */
      for (c = cStart; c < cEnd; ++c) {
        if (wp && !PetscBTLookup(wp, c - pStart)) continue;
        CHKERRQ(DMPlexGetCellType(dm, c, &ct));
        if (ct == DM_POLYTOPE_SEG_PRISM_TENSOR ||
            ct == DM_POLYTOPE_TRI_PRISM_TENSOR ||
            ct == DM_POLYTOPE_QUAD_PRISM_TENSOR) {
          const PetscInt *cone;
          PetscInt        coneSize, e;

          CHKERRQ(DMPlexGetCone(dm, c, &cone));
          CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
          for (e = 0; e < coneSize; ++e) {
            const PetscInt *econe;

            CHKERRQ(DMPlexGetCone(dm, cone[e], &econe));
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] (%D_%d) -- (%D_%d) -- (%D_%d);\n", colors[rank%numColors], econe[0], rank, cone[e], rank, econe[1], rank));
          }
        } else {
          PetscInt *closure = NULL;
          PetscInt  closureSize, Nv = 0, v;

          CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
          for (p = 0; p < closureSize*2; p += 2) {
            const PetscInt point = closure[p];

            if ((point >= vStart) && (point < vEnd)) closure[Nv++] = point;
          }
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] ", colors[rank%numColors]));
          for (v = 0; v <= Nv; ++v) {
            const PetscInt vertex = closure[v%Nv];

            if (v > 0) {
              if (plotEdges) {
                const PetscInt *edge;
                PetscInt        endpoints[2], ne;

                endpoints[0] = closure[v-1]; endpoints[1] = vertex;
                CHKERRQ(DMPlexGetJoin(dm, 2, endpoints, &ne, &edge));
                PetscCheckFalse(ne != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find edge for vertices %D, %D", endpoints[0], endpoints[1]);
                CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, " -- (%D_%d) -- ", edge[0], rank));
                CHKERRQ(DMPlexRestoreJoin(dm, 2, endpoints, &ne, &edge));
              } else {
                CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, " -- "));
              }
            }
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "(%D_%d)", vertex, rank));
          }
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ";\n"));
          CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
        }
      }
    }
    CHKERRQ(VecGetArray(coordinates, &coords));
    for (c = cStart; c < cEnd; ++c) {
      double    ccoords[3] = {0.0, 0.0, 0.0};
      PetscBool isLabeled  = PETSC_FALSE;
      PetscInt *closure    = NULL;
      PetscInt  closureSize, dof, d, n = 0;

      if (wp && !PetscBTLookup(wp,c - pStart)) continue;
      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "\\path ("));
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = closure[p];
        PetscInt       off;

        if ((point < vStart) || (point >= vEnd)) continue;
        CHKERRQ(PetscSectionGetDof(coordSection, point, &dof));
        CHKERRQ(PetscSectionGetOffset(coordSection, point, &off));
        for (d = 0; d < dof; ++d) {
          tcoords[d] = (double) (scale*PetscRealPart(coords[off+d]));
          tcoords[d] = PetscAbs(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
        }
        /* Rotate coordinates since PGF makes z point out of the page instead of up */
        if (dof == 3) {PetscReal tmp = tcoords[1]; tcoords[1] = tcoords[2]; tcoords[2] = -tmp;}
        for (d = 0; d < dof; ++d) {ccoords[d] += tcoords[d];}
        ++n;
      }
      for (d = 0; d < dof; ++d) {ccoords[d] /= n;}
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (d = 0; d < dof; ++d) {
        if (d > 0) CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ","));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "%g", (double) ccoords[d]));
      }
      if (drawHasse) color = colors[depth%numColors];
      else           color = colors[rank%numColors];
      for (l = 0; l < numLabels; ++l) {
        PetscInt val;
        CHKERRQ(DMGetLabelValue(dm, names[l], c, &val));
        if (val >= 0) {color = lcolors[l%numLColors]; isLabeled = PETSC_TRUE; break;}
      }
      if (drawNumbers[dim]) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [draw,shape=circle,color=%s] {%D};\n", c, rank, color, c));
      } else if (drawColors[dim]) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [fill,inner sep=%dpt,shape=circle,color=%s] {};\n", c, rank, !isLabeled ? 1 : 2, color));
      } else {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [] {};\n", c, rank));
      }
    }
    CHKERRQ(VecRestoreArray(coordinates, &coords));
    if (drawHasse) {
      color = colors[depth%numColors];
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%% Cells\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\foreach \\c in {\\cStart,...,\\cEnd}\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "{\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  \\node(\\c_%d) [draw,shape=circle,color=%s,minimum size = 6mm] at (\\cShift+\\c-\\cStart,0) {\\c};\n", rank, color));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "}\n"));

      color = colors[1%numColors];
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%% Edges\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\foreach \\e in {\\eStart,...,\\eEnd}\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "{\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  \\node(\\e_%d) [draw,shape=circle,color=%s,minimum size = 6mm] at (\\eShift+\\e-\\eStart,1) {\\e};\n", rank, color));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "}\n"));

      color = colors[0%numColors];
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "%% Vertices\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\foreach \\v in {\\vStart,...,\\vEnd}\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "{\n"));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  \\node(\\v_%d) [draw,shape=circle,color=%s,minimum size = 6mm] at (\\vShift+\\v-\\vStart,2) {\\v};\n", rank, color));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "}\n"));

      for (p = pStart; p < pEnd; ++p) {
        const PetscInt *cone;
        PetscInt        coneSize, cp;

        CHKERRQ(DMPlexGetCone(dm, p, &cone));
        CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
        for (cp = 0; cp < coneSize; ++cp) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\draw[->, shorten >=1pt] (%D_%d) -- (%D_%d);\n", cone[cp], rank, p, rank));
        }
      }
    }
    CHKERRQ(PetscViewerFlush(viewer));
    CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\end{tikzpicture}\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "\\end{document}\n", name));
    for (l = 0; l < numLabels;  ++l) CHKERRQ(PetscFree(names[l]));
    for (c = 0; c < numColors;  ++c) CHKERRQ(PetscFree(colors[c]));
    for (c = 0; c < numLColors; ++c) CHKERRQ(PetscFree(lcolors[c]));
    CHKERRQ(PetscFree3(names, colors, lcolors));
    CHKERRQ(PetscBTDestroy(&wp));
  } else if (format == PETSC_VIEWER_LOAD_BALANCE) {
    Vec                    cown,acown;
    VecScatter             sct;
    ISLocalToGlobalMapping g2l;
    IS                     gid,acis;
    MPI_Comm               comm,ncomm = MPI_COMM_NULL;
    MPI_Group              ggroup,ngroup;
    PetscScalar            *array,nid;
    const PetscInt         *idxs;
    PetscInt               *idxs2,*start,*adjacency,*work;
    PetscInt64             lm[3],gm[3];
    PetscInt               i,c,cStart,cEnd,cum,numVertices,ect,ectn,cellHeight;
    PetscMPIInt            d1,d2,rank;

    CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
    CHKERRMPI(MPI_Comm_rank(comm,&rank));
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    CHKERRMPI(MPI_Comm_split_type(comm,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&ncomm));
#endif
    if (ncomm != MPI_COMM_NULL) {
      CHKERRMPI(MPI_Comm_group(comm,&ggroup));
      CHKERRMPI(MPI_Comm_group(ncomm,&ngroup));
      d1   = 0;
      CHKERRMPI(MPI_Group_translate_ranks(ngroup,1,&d1,ggroup,&d2));
      nid  = d2;
      CHKERRMPI(MPI_Group_free(&ggroup));
      CHKERRMPI(MPI_Group_free(&ngroup));
      CHKERRMPI(MPI_Comm_free(&ncomm));
    } else nid = 0.0;

    /* Get connectivity */
    CHKERRQ(DMPlexGetVTKCellHeight(dm,&cellHeight));
    CHKERRQ(DMPlexCreatePartitionerGraph(dm,cellHeight,&numVertices,&start,&adjacency,&gid));

    /* filter overlapped local cells */
    CHKERRQ(DMPlexGetHeightStratum(dm,cellHeight,&cStart,&cEnd));
    CHKERRQ(ISGetIndices(gid,&idxs));
    CHKERRQ(ISGetLocalSize(gid,&cum));
    CHKERRQ(PetscMalloc1(cum,&idxs2));
    for (c = cStart, cum = 0; c < cEnd; c++) {
      if (idxs[c-cStart] < 0) continue;
      idxs2[cum++] = idxs[c-cStart];
    }
    CHKERRQ(ISRestoreIndices(gid,&idxs));
    PetscCheckFalse(numVertices != cum,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected %D != %D",numVertices,cum);
    CHKERRQ(ISDestroy(&gid));
    CHKERRQ(ISCreateGeneral(comm,numVertices,idxs2,PETSC_OWN_POINTER,&gid));

    /* support for node-aware cell locality */
    CHKERRQ(ISCreateGeneral(comm,start[numVertices],adjacency,PETSC_USE_POINTER,&acis));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,start[numVertices],&acown));
    CHKERRQ(VecCreateMPI(comm,numVertices,PETSC_DECIDE,&cown));
    CHKERRQ(VecGetArray(cown,&array));
    for (c = 0; c < numVertices; c++) array[c] = nid;
    CHKERRQ(VecRestoreArray(cown,&array));
    CHKERRQ(VecScatterCreate(cown,acis,acown,NULL,&sct));
    CHKERRQ(VecScatterBegin(sct,cown,acown,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(sct,cown,acown,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(ISDestroy(&acis));
    CHKERRQ(VecScatterDestroy(&sct));
    CHKERRQ(VecDestroy(&cown));

    /* compute edgeCut */
    for (c = 0, cum = 0; c < numVertices; c++) cum = PetscMax(cum,start[c+1]-start[c]);
    CHKERRQ(PetscMalloc1(cum,&work));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(gid,&g2l));
    CHKERRQ(ISLocalToGlobalMappingSetType(g2l,ISLOCALTOGLOBALMAPPINGHASH));
    CHKERRQ(ISDestroy(&gid));
    CHKERRQ(VecGetArray(acown,&array));
    for (c = 0, ect = 0, ectn = 0; c < numVertices; c++) {
      PetscInt totl;

      totl = start[c+1]-start[c];
      CHKERRQ(ISGlobalToLocalMappingApply(g2l,IS_GTOLM_MASK,totl,adjacency+start[c],NULL,work));
      for (i = 0; i < totl; i++) {
        if (work[i] < 0) {
          ect  += 1;
          ectn += (array[i + start[c]] != nid) ? 0 : 1;
        }
      }
    }
    CHKERRQ(PetscFree(work));
    CHKERRQ(VecRestoreArray(acown,&array));
    lm[0] = numVertices > 0 ?  numVertices : PETSC_MAX_INT;
    lm[1] = -numVertices;
    CHKERRMPI(MPIU_Allreduce(lm,gm,2,MPIU_INT64,MPI_MIN,comm));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Cell balance: %.2f (max %D, min %D",-((double)gm[1])/((double)gm[0]),-(PetscInt)gm[1],(PetscInt)gm[0]));
    lm[0] = ect; /* edgeCut */
    lm[1] = ectn; /* node-aware edgeCut */
    lm[2] = numVertices > 0 ? 0 : 1; /* empty processes */
    CHKERRMPI(MPIU_Allreduce(lm,gm,3,MPIU_INT64,MPI_SUM,comm));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,", empty %D)\n",(PetscInt)gm[2]));
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Edge Cut: %D (on node %.3f)\n",(PetscInt)(gm[0]/2),gm[0] ? ((double)(gm[1]))/((double)gm[0]) : 1.));
#else
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Edge Cut: %D (on node %.3f)\n",(PetscInt)(gm[0]/2),0.0));
#endif
    CHKERRQ(ISLocalToGlobalMappingDestroy(&g2l));
    CHKERRQ(PetscFree(start));
    CHKERRQ(PetscFree(adjacency));
    CHKERRQ(VecDestroy(&acown));
  } else {
    const char    *name;
    PetscInt      *sizes, *hybsizes, *ghostsizes;
    PetscInt       locDepth, depth, cellHeight, dim, d;
    PetscInt       pStart, pEnd, p, gcStart, gcEnd, gcNum;
    PetscInt       numLabels, l, maxSize = 17;
    DMPolytopeType ct0 = DM_POLYTOPE_UNKNOWN;
    MPI_Comm       comm;
    PetscMPIInt    size, rank;

    CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
    CHKERRMPI(MPI_Comm_size(comm, &size));
    CHKERRMPI(MPI_Comm_rank(comm, &rank));
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
    CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
    if (name) CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s in %D dimension%s:\n", name, dim, dim == 1 ? "" : "s"));
    else      CHKERRQ(PetscViewerASCIIPrintf(viewer, "Mesh in %D dimension%s:\n", dim, dim == 1 ? "" : "s"));
    if (cellHeight) CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Cells are at height %D\n", cellHeight));
    CHKERRQ(DMPlexGetDepth(dm, &locDepth));
    CHKERRMPI(MPIU_Allreduce(&locDepth, &depth, 1, MPIU_INT, MPI_MAX, comm));
    CHKERRQ(DMPlexGetGhostCellStratum(dm, &gcStart, &gcEnd));
    gcNum = gcEnd - gcStart;
    if (size < maxSize) CHKERRQ(PetscCalloc3(size, &sizes, size, &hybsizes, size, &ghostsizes));
    else                CHKERRQ(PetscCalloc3(3,    &sizes, 3,    &hybsizes, 3,    &ghostsizes));
    for (d = 0; d <= depth; d++) {
      PetscInt Nc[2] = {0, 0}, ict;

      CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
      if (pStart < pEnd) CHKERRQ(DMPlexGetCellType(dm, pStart, &ct0));
      ict  = ct0;
      CHKERRMPI(MPI_Bcast(&ict, 1, MPIU_INT, 0, comm));
      ct0  = (DMPolytopeType) ict;
      for (p = pStart; p < pEnd; ++p) {
        DMPolytopeType ct;

        CHKERRQ(DMPlexGetCellType(dm, p, &ct));
        if (ct == ct0) ++Nc[0];
        else           ++Nc[1];
      }
      if (size < maxSize) {
        CHKERRMPI(MPI_Gather(&Nc[0], 1, MPIU_INT, sizes,    1, MPIU_INT, 0, comm));
        CHKERRMPI(MPI_Gather(&Nc[1], 1, MPIU_INT, hybsizes, 1, MPIU_INT, 0, comm));
        if (d == depth) CHKERRMPI(MPI_Gather(&gcNum, 1, MPIU_INT, ghostsizes, 1, MPIU_INT, 0, comm));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Number of %D-cells per rank:", (depth == 1) && d ? dim : d));
        for (p = 0; p < size; ++p) {
          if (rank == 0) {
            CHKERRQ(PetscViewerASCIIPrintf(viewer, " %D", sizes[p]+hybsizes[p]));
            if (hybsizes[p]   > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, " (%D)", hybsizes[p]));
            if (ghostsizes[p] > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, " [%D]", ghostsizes[p]));
          }
        }
      } else {
        PetscInt locMinMax[2];

        locMinMax[0] = Nc[0]+Nc[1]; locMinMax[1] = Nc[0]+Nc[1];
        CHKERRQ(PetscGlobalMinMaxInt(comm, locMinMax, sizes));
        locMinMax[0] = Nc[1]; locMinMax[1] = Nc[1];
        CHKERRQ(PetscGlobalMinMaxInt(comm, locMinMax, hybsizes));
        if (d == depth) {
          locMinMax[0] = gcNum; locMinMax[1] = gcNum;
          CHKERRQ(PetscGlobalMinMaxInt(comm, locMinMax, ghostsizes));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Min/Max of %D-cells per rank:", (depth == 1) && d ? dim : d));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT "/%" PetscInt_FMT, sizes[0], sizes[1]));
        if (hybsizes[0]   > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, " (%" PetscInt_FMT "/%" PetscInt_FMT ")", hybsizes[0], hybsizes[1]));
        if (ghostsizes[0] > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, " [%" PetscInt_FMT "/%" PetscInt_FMT "]", ghostsizes[0], ghostsizes[1]));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    CHKERRQ(PetscFree3(sizes, hybsizes, ghostsizes));
    {
      const PetscReal      *maxCell;
      const PetscReal      *L;
      const DMBoundaryType *bd;
      PetscBool             per, localized;

      CHKERRQ(DMGetPeriodicity(dm, &per, &maxCell, &L, &bd));
      CHKERRQ(DMGetCoordinatesLocalized(dm, &localized));
      if (per) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "Periodic mesh ("));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
        for (d = 0; d < dim; ++d) {
          if (bd && d > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, ", "));
          if (bd)    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s", DMBoundaryTypes[bd[d]]));
        }
        CHKERRQ(PetscViewerASCIIPrintf(viewer, ") coordinates %s\n", localized ? "localized" : "not localized"));
        CHKERRQ(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      }
    }
    CHKERRQ(DMGetNumLabels(dm, &numLabels));
    if (numLabels) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Labels:\n"));
    for (l = 0; l < numLabels; ++l) {
      DMLabel         label;
      const char     *name;
      IS              valueIS;
      const PetscInt *values;
      PetscInt        numValues, v;

      CHKERRQ(DMGetLabelName(dm, l, &name));
      CHKERRQ(DMGetLabel(dm, name, &label));
      CHKERRQ(DMLabelGetNumValues(label, &numValues));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  %s: %D strata with value/size (", name, numValues));
      CHKERRQ(DMLabelGetValueIS(label, &valueIS));
      CHKERRQ(ISGetIndices(valueIS, &values));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      for (v = 0; v < numValues; ++v) {
        PetscInt size;

        CHKERRQ(DMLabelGetStratumSize(label, values[v], &size));
        if (v > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, ", "));
        CHKERRQ(PetscViewerASCIIPrintf(viewer, "%D (%D)", values[v], size));
      }
      CHKERRQ(PetscViewerASCIIPrintf(viewer, ")\n"));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      CHKERRQ(ISRestoreIndices(valueIS, &values));
      CHKERRQ(ISDestroy(&valueIS));
    }
    {
      char    **labelNames;
      PetscInt  Nl = numLabels;
      PetscBool flg;

      CHKERRQ(PetscMalloc1(Nl, &labelNames));
      CHKERRQ(PetscOptionsGetStringArray(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_plex_view_labels", labelNames, &Nl, &flg));
      for (l = 0; l < Nl; ++l) {
        DMLabel label;

        CHKERRQ(DMHasLabel(dm, labelNames[l], &flg));
        if (flg) {
          CHKERRQ(DMGetLabel(dm, labelNames[l], &label));
          CHKERRQ(DMLabelView(label, viewer));
        }
        CHKERRQ(PetscFree(labelNames[l]));
      }
      CHKERRQ(PetscFree(labelNames));
    }
    /* If no fields are specified, people do not want to see adjacency */
    if (dm->Nf) {
      PetscInt f;

      for (f = 0; f < dm->Nf; ++f) {
        const char *name;

        CHKERRQ(PetscObjectGetName(dm->fields[f].disc, &name));
        if (numLabels) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Field %s:\n", name));
        CHKERRQ(PetscViewerASCIIPushTab(viewer));
        if (dm->fields[f].label) CHKERRQ(DMLabelView(dm->fields[f].label, viewer));
        if (dm->fields[f].adjacency[0]) {
          if (dm->fields[f].adjacency[1]) CHKERRQ(PetscViewerASCIIPrintf(viewer, "adjacency FVM++\n"));
          else                            CHKERRQ(PetscViewerASCIIPrintf(viewer, "adjacency FVM\n"));
        } else {
          if (dm->fields[f].adjacency[1]) CHKERRQ(PetscViewerASCIIPrintf(viewer, "adjacency FEM\n"));
          else                            CHKERRQ(PetscViewerASCIIPrintf(viewer, "adjacency FUNKY\n"));
        }
        CHKERRQ(PetscViewerASCIIPopTab(viewer));
      }
    }
    CHKERRQ(DMGetCoarseDM(dm, &cdm));
    if (cdm) {
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ(DMPlexView_Ascii(cdm, viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDrawCell(DM dm, PetscDraw draw, PetscInt cell, const PetscScalar coords[])
{
  DMPolytopeType ct;
  PetscMPIInt    rank;
  PetscInt       cdim;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMPlexGetCellType(dm, cell, &ct));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  switch (ct) {
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
    switch (cdim) {
    case 1:
    {
      const PetscReal y  = 0.5;  /* TODO Put it in the middle of the viewport */
      const PetscReal dy = 0.05; /* TODO Make it a fraction of the total length */

      CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[0]), y,    PetscRealPart(coords[1]), y,    PETSC_DRAW_BLACK));
      CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[0]), y+dy, PetscRealPart(coords[0]), y-dy, PETSC_DRAW_BLACK));
      CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[1]), y+dy, PetscRealPart(coords[1]), y-dy, PETSC_DRAW_BLACK));
    }
    break;
    case 2:
    {
      const PetscReal dx = (PetscRealPart(coords[3]) - PetscRealPart(coords[1]));
      const PetscReal dy = (PetscRealPart(coords[2]) - PetscRealPart(coords[0]));
      const PetscReal l  = 0.1/PetscSqrtReal(dx*dx + dy*dy);

      CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK));
      CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[0])+l*dx, PetscRealPart(coords[1])+l*dy, PetscRealPart(coords[0])-l*dx, PetscRealPart(coords[1])-l*dy, PETSC_DRAW_BLACK));
      CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[2])+l*dx, PetscRealPart(coords[3])+l*dy, PetscRealPart(coords[2])-l*dx, PetscRealPart(coords[3])-l*dy, PETSC_DRAW_BLACK));
    }
    break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of dimension %D", cdim);
    }
    break;
  case DM_POLYTOPE_TRIANGLE:
    CHKERRQ(PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK));
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    CHKERRQ(PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2));
    CHKERRQ(PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]),
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                              PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PETSC_DRAW_BLACK));
    CHKERRQ(PetscDrawLine(draw, PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK));
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDrawCellHighOrder(DM dm, PetscDraw draw, PetscInt cell, const PetscScalar coords[], PetscInt edgeDiv, PetscReal refCoords[], PetscReal edgeCoords[])
{
  DMPolytopeType ct;
  PetscReal      centroid[2] = {0., 0.};
  PetscMPIInt    rank;
  PetscInt       fillColor, v, e, d;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMPlexGetCellType(dm, cell, &ct));
  fillColor = PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2;
  switch (ct) {
  case DM_POLYTOPE_TRIANGLE:
    {
      PetscReal refVertices[6] = {-1., -1., 1., -1., -1., 1.};

      for (v = 0; v < 3; ++v) {centroid[0] += PetscRealPart(coords[v*2+0])/3.;centroid[1] += PetscRealPart(coords[v*2+1])/3.;}
      for (e = 0; e < 3; ++e) {
        refCoords[0] = refVertices[e*2+0];
        refCoords[1] = refVertices[e*2+1];
        for (d = 1; d <= edgeDiv; ++d) {
          refCoords[d*2+0] = refCoords[0] + (refVertices[(e+1)%3 * 2 + 0] - refCoords[0])*d/edgeDiv;
          refCoords[d*2+1] = refCoords[1] + (refVertices[(e+1)%3 * 2 + 1] - refCoords[1])*d/edgeDiv;
        }
        CHKERRQ(DMPlexReferenceToCoordinates(dm, cell, edgeDiv+1, refCoords, edgeCoords));
        for (d = 0; d < edgeDiv; ++d) {
          CHKERRQ(PetscDrawTriangle(draw, centroid[0], centroid[1], edgeCoords[d*2+0], edgeCoords[d*2+1], edgeCoords[(d+1)*2+0], edgeCoords[(d+1)*2+1], fillColor, fillColor, fillColor));
          CHKERRQ(PetscDrawLine(draw, edgeCoords[d*2+0], edgeCoords[d*2+1], edgeCoords[(d+1)*2+0], edgeCoords[(d+1)*2+1], PETSC_DRAW_BLACK));
        }
      }
    }
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexView_Draw(DM dm, PetscViewer viewer)
{
  PetscDraw          draw;
  DM                 cdm;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  PetscReal          xyl[2],xyr[2],bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscReal         *refCoords, *edgeCoords;
  PetscBool          isnull, drawAffine = PETSC_TRUE;
  PetscInt           dim, vStart, vEnd, cStart, cEnd, c, N, edgeDiv = 4;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinateDim(dm, &dim));
  PetscCheckFalse(dim > 2,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D", dim);
  CHKERRQ(PetscOptionsGetBool(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_view_draw_affine", &drawAffine, NULL));
  if (!drawAffine) CHKERRQ(PetscMalloc2((edgeDiv+1)*dim, &refCoords, (edgeDiv+1)*dim, &edgeCoords));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  CHKERRQ(PetscViewerDrawGetDraw(viewer, 0, &draw));
  CHKERRQ(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRQ(PetscDrawSetTitle(draw, "Mesh"));

  CHKERRQ(VecGetLocalSize(coordinates, &N));
  CHKERRQ(VecGetArrayRead(coordinates, &coords));
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));   bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c+1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c+1]));
  }
  CHKERRQ(VecRestoreArrayRead(coordinates, &coords));
  CHKERRMPI(MPIU_Allreduce(&bound[0],xyl,2,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)dm)));
  CHKERRMPI(MPIU_Allreduce(&bound[2],xyr,2,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)dm)));
  CHKERRQ(PetscDrawSetCoordinates(draw, xyl[0], xyl[1], xyr[0], xyr[1]));
  CHKERRQ(PetscDrawClear(draw));

  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *coords = NULL;
    PetscInt     numCoords;

    CHKERRQ(DMPlexVecGetClosureAtDepth_Internal(dm, coordSection, coordinates, c, 0, &numCoords, &coords));
    if (drawAffine) {
      CHKERRQ(DMPlexDrawCell(dm, draw, c, coords));
    } else {
      CHKERRQ(DMPlexDrawCellHighOrder(dm, draw, c, coords, edgeDiv, refCoords, edgeCoords));
    }
    CHKERRQ(DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords));
  }
  if (!drawAffine) CHKERRQ(PetscFree2(refCoords, edgeCoords));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#include <petscviewerexodusii.h>
#endif

PetscErrorCode DMView_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, ishdf5, isvtk, isdraw, flg, isglvis, isexodus;
  char           name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII,    &iascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,      &isvtk));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,     &ishdf5));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,     &isdraw));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS,    &isglvis));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWEREXODUSII, &isexodus));
  if (iascii) {
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_GLVIS) {
      CHKERRQ(DMPlexView_GLVis(dm, viewer));
    } else {
      CHKERRQ(DMPlexView_Ascii(dm, viewer));
    }
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexView_HDF5_Internal(dm, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isvtk) {
    CHKERRQ(DMPlexVTKWriteAll((PetscObject) dm,viewer));
  } else if (isdraw) {
    CHKERRQ(DMPlexView_Draw(dm, viewer));
  } else if (isglvis) {
    CHKERRQ(DMPlexView_GLVis(dm, viewer));
#if defined(PETSC_HAVE_EXODUSII)
  } else if (isexodus) {
/*
      exodusII requires that all sets be part of exactly one cell set.
      If the dm does not have a "Cell Sets" label defined, we create one
      with ID 1, containig all cells.
      Note that if the Cell Sets label is defined but does not cover all cells,
      we may still have a problem. This should probably be checked here or in the viewer;
    */
    PetscInt numCS;
    CHKERRQ(DMGetLabelSize(dm,"Cell Sets",&numCS));
    if (!numCS) {
      PetscInt cStart, cEnd, c;
      CHKERRQ(DMCreateLabel(dm, "Cell Sets"));
      CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      for (c = cStart; c < cEnd; ++c) CHKERRQ(DMSetLabelValue(dm, "Cell Sets", c, 1));
    }
    CHKERRQ(DMView_PlexExodusII(dm, viewer));
#endif
  } else {
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlex writing", ((PetscObject)viewer)->type_name);
  }
  /* Optionally view the partition */
  CHKERRQ(PetscOptionsHasName(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_partition_view", &flg));
  if (flg) {
    Vec ranks;
    CHKERRQ(DMPlexCreateRankField(dm, &ranks));
    CHKERRQ(VecView(ranks, viewer));
    CHKERRQ(VecDestroy(&ranks));
  }
  /* Optionally view a label */
  CHKERRQ(PetscOptionsGetString(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_label_view", name, sizeof(name), &flg));
  if (flg) {
    DMLabel label;
    Vec     val;

    CHKERRQ(DMGetLabel(dm, name, &label));
    PetscCheck(label,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Label %s provided to -dm_label_view does not exist in this DM", name);
    CHKERRQ(DMPlexCreateLabelField(dm, label, &val));
    CHKERRQ(VecView(val, viewer));
    CHKERRQ(VecDestroy(&val));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexTopologyView - Saves a DMPlex topology into a file

  Collective on DM

  Input Parameters:
+ dm     - The DM whose topology is to be saved
- viewer - The PetscViewer for saving

  Level: advanced

.seealso: DMView(), DMPlexCoordinatesView(), DMPlexLabelsView(), DMPlexTopologyLoad()
@*/
PetscErrorCode DMPlexTopologyView(DM dm, PetscViewer viewer)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_TopologyView,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      IS globalPointNumbering;

      CHKERRQ(DMPlexCreatePointNumbering(dm, &globalPointNumbering));
      CHKERRQ(DMPlexTopologyView_HDF5_Internal(dm, globalPointNumbering, viewer));
      CHKERRQ(ISDestroy(&globalPointNumbering));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 output.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_TopologyView,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCoordinatesView - Saves DMPlex coordinates into a file

  Collective on DM

  Input Parameters:
+ dm     - The DM whose coordinates are to be saved
- viewer - The PetscViewer for saving

  Level: advanced

.seealso: DMView(), DMPlexTopologyView(), DMPlexLabelsView(), DMPlexCoordinatesLoad()
@*/
PetscErrorCode DMPlexCoordinatesView(DM dm, PetscViewer viewer)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_CoordinatesView,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      CHKERRQ(DMPlexCoordinatesView_HDF5_Internal(dm, viewer));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_CoordinatesView,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelsView - Saves DMPlex labels into a file

  Collective on DM

  Input Parameters:
+ dm     - The DM whose labels are to be saved
- viewer - The PetscViewer for saving

  Level: advanced

.seealso: DMView(), DMPlexTopologyView(), DMPlexCoordinatesView(), DMPlexLabelsLoad()
@*/
PetscErrorCode DMPlexLabelsView(DM dm, PetscViewer viewer)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_LabelsView,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    IS                globalPointNumbering;
    PetscViewerFormat format;

    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      CHKERRQ(DMPlexCreatePointNumbering(dm, &globalPointNumbering));
      CHKERRQ(DMPlexLabelsView_HDF5_Internal(dm, globalPointNumbering, viewer));
      CHKERRQ(ISDestroy(&globalPointNumbering));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_LabelsView,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSectionView - Saves a section associated with a DMPlex

  Collective on DM

  Input Parameters:
+ dm         - The DM that contains the topology on which the section to be saved is defined
. viewer     - The PetscViewer for saving
- sectiondm  - The DM that contains the section to be saved

  Level: advanced

  Notes:
  This function is a wrapper around PetscSectionView(); in addition to the raw section, it saves information that associates the section points to the topology (dm) points. When the topology (dm) and the section are later loaded with DMPlexTopologyLoad() and DMPlexSectionLoad(), respectively, this information is used to match section points with topology points.

  In general dm and sectiondm are two different objects, the former carrying the topology and the latter carrying the section, and have been given a topology name and a section name, respectively, with PetscObjectSetName(). In practice, however, they can be the same object if it carries both topology and section; in that case the name of the object is used as both the topology name and the section name.

.seealso: DMView(), DMPlexTopologyView(), DMPlexCoordinatesView(), DMPlexLabelsView(), DMPlexGlobalVectorView(), DMPlexLocalVectorView(), PetscSectionView(), DMPlexSectionLoad()
@*/
PetscErrorCode DMPlexSectionView(DM dm, PetscViewer viewer, DM sectiondm)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(sectiondm, DM_CLASSID, 3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_SectionView,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexSectionView_HDF5_Internal(dm, viewer, sectiondm));
#else
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_SectionView,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalVectorView - Saves a global vector

  Collective on DM

  Input Parameters:
+ dm        - The DM that represents the topology
. viewer    - The PetscViewer to save data with
. sectiondm - The DM that contains the global section on which vec is defined
- vec       - The global vector to be saved

  Level: advanced

  Notes:
  In general dm and sectiondm are two different objects, the former carrying the topology and the latter carrying the section, and have been given a topology name and a section name, respectively, with PetscObjectSetName(). In practice, however, they can be the same object if it carries both topology and section; in that case the name of the object is used as both the topology name and the section name.

  Typical calling sequence
$       DMCreate(PETSC_COMM_WORLD, &dm);
$       DMSetType(dm, DMPLEX);
$       PetscObjectSetName((PetscObject)dm, "topologydm_name");
$       DMClone(dm, &sectiondm);
$       PetscObjectSetName((PetscObject)sectiondm, "sectiondm_name");
$       PetscSectionCreate(PETSC_COMM_WORLD, &section);
$       DMPlexGetChart(sectiondm, &pStart, &pEnd);
$       PetscSectionSetChart(section, pStart, pEnd);
$       PetscSectionSetUp(section);
$       DMSetLocalSection(sectiondm, section);
$       PetscSectionDestroy(&section);
$       DMGetGlobalVector(sectiondm, &vec);
$       PetscObjectSetName((PetscObject)vec, "vec_name");
$       DMPlexTopologyView(dm, viewer);
$       DMPlexSectionView(dm, viewer, sectiondm);
$       DMPlexGlobalVectorView(dm, viewer, sectiondm, vec);
$       DMRestoreGlobalVector(sectiondm, &vec);
$       DMDestroy(&sectiondm);
$       DMDestroy(&dm);

.seealso: DMPlexTopologyView(), DMPlexSectionView(), DMPlexLocalVectorView(), DMPlexGlobalVectorLoad(), DMPlexLocalVectorLoad()
@*/
PetscErrorCode DMPlexGlobalVectorView(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  PetscBool       ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(sectiondm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 4);
  /* Check consistency */
  {
    PetscSection  section;
    PetscBool     includesConstraints;
    PetscInt      m, m1;

    CHKERRQ(VecGetLocalSize(vec, &m1));
    CHKERRQ(DMGetGlobalSection(sectiondm, &section));
    CHKERRQ(PetscSectionGetIncludesConstraints(section, &includesConstraints));
    if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(section, &m));
    else CHKERRQ(PetscSectionGetConstrainedStorageSize(section, &m));
    PetscCheckFalse(m1 != m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Global vector size (%D) != global section storage size (%D)", m1, m);
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_GlobalVectorView,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexGlobalVectorView_HDF5_Internal(dm, viewer, sectiondm, vec));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_GlobalVectorView,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexLocalVectorView - Saves a local vector

  Collective on DM

  Input Parameters:
+ dm        - The DM that represents the topology
. viewer    - The PetscViewer to save data with
. sectiondm - The DM that contains the local section on which vec is defined; may be the same as dm
- vec       - The local vector to be saved

  Level: advanced

  Notes:
  In general dm and sectiondm are two different objects, the former carrying the topology and the latter carrying the section, and have been given a topology name and a section name, respectively, with PetscObjectSetName(). In practice, however, they can be the same object if it carries both topology and section; in that case the name of the object is used as both the topology name and the section name.

  Typical calling sequence
$       DMCreate(PETSC_COMM_WORLD, &dm);
$       DMSetType(dm, DMPLEX);
$       PetscObjectSetName((PetscObject)dm, "topologydm_name");
$       DMClone(dm, &sectiondm);
$       PetscObjectSetName((PetscObject)sectiondm, "sectiondm_name");
$       PetscSectionCreate(PETSC_COMM_WORLD, &section);
$       DMPlexGetChart(sectiondm, &pStart, &pEnd);
$       PetscSectionSetChart(section, pStart, pEnd);
$       PetscSectionSetUp(section);
$       DMSetLocalSection(sectiondm, section);
$       DMGetLocalVector(sectiondm, &vec);
$       PetscObjectSetName((PetscObject)vec, "vec_name");
$       DMPlexTopologyView(dm, viewer);
$       DMPlexSectionView(dm, viewer, sectiondm);
$       DMPlexLocalVectorView(dm, viewer, sectiondm, vec);
$       DMRestoreLocalVector(sectiondm, &vec);
$       DMDestroy(&sectiondm);
$       DMDestroy(&dm);

.seealso: DMPlexTopologyView(), DMPlexSectionView(), DMPlexGlobalVectorView(), DMPlexGlobalVectorLoad(), DMPlexLocalVectorLoad()
@*/
PetscErrorCode DMPlexLocalVectorView(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  PetscBool       ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(sectiondm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 4);
  /* Check consistency */
  {
    PetscSection  section;
    PetscBool     includesConstraints;
    PetscInt      m, m1;

    CHKERRQ(VecGetLocalSize(vec, &m1));
    CHKERRQ(DMGetLocalSection(sectiondm, &section));
    CHKERRQ(PetscSectionGetIncludesConstraints(section, &includesConstraints));
    if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(section, &m));
    else CHKERRQ(PetscSectionGetConstrainedStorageSize(section, &m));
    PetscCheckFalse(m1 != m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local vector size (%D) != local section storage size (%D)", m1, m);
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_LocalVectorView,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexLocalVectorView_HDF5_Internal(dm, viewer, sectiondm, vec));
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_LocalVectorView,viewer,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMLoad_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,   &ishdf5));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_XDMF || format == PETSC_VIEWER_HDF5_VIZ) {
      CHKERRQ(DMPlexLoad_HDF5_Xdmf_Internal(dm, viewer));
    } else if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      CHKERRQ(DMPlexLoad_HDF5_Internal(dm, viewer));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
    PetscFunctionReturn(0);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlex loading", ((PetscObject)viewer)->type_name);
}

/*@
  DMPlexTopologyLoad - Loads a topology into a DMPlex

  Collective on DM

  Input Parameters:
+ dm     - The DM into which the topology is loaded
- viewer - The PetscViewer for the saved topology

  Output Parameters:
. globalToLocalPointSF - The PetscSF that pushes points in [0, N) to the associated points in the loaded plex, where N is the global number of points; NULL if unneeded

  Level: advanced

.seealso: DMLoad(), DMPlexCoordinatesLoad(), DMPlexLabelsLoad(), DMView(), PetscViewerHDF5Open(), PetscViewerPushFormat()
@*/
PetscErrorCode DMPlexTopologyLoad(DM dm, PetscViewer viewer, PetscSF *globalToLocalPointSF)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (globalToLocalPointSF) PetscValidPointer(globalToLocalPointSF, 3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_TopologyLoad,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      CHKERRQ(DMPlexTopologyLoad_HDF5_Internal(dm, viewer, globalToLocalPointSF));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_TopologyLoad,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCoordinatesLoad - Loads coordinates into a DMPlex

  Collective on DM

  Input Parameters:
+ dm     - The DM into which the coordinates are loaded
. viewer - The PetscViewer for the saved coordinates
- globalToLocalPointSF - The SF returned by DMPlexTopologyLoad() when loading dm from viewer

  Level: advanced

.seealso: DMLoad(), DMPlexTopologyLoad(), DMPlexLabelsLoad(), DMView(), PetscViewerHDF5Open(), PetscViewerPushFormat()
@*/
PetscErrorCode DMPlexCoordinatesLoad(DM dm, PetscViewer viewer, PetscSF globalToLocalPointSF)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(globalToLocalPointSF, PETSCSF_CLASSID, 3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_CoordinatesLoad,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      CHKERRQ(DMPlexCoordinatesLoad_HDF5_Internal(dm, viewer, globalToLocalPointSF));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_CoordinatesLoad,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexLabelsLoad - Loads labels into a DMPlex

  Collective on DM

  Input Parameters:
+ dm     - The DM into which the labels are loaded
. viewer - The PetscViewer for the saved labels
- globalToLocalPointSF - The SF returned by DMPlexTopologyLoad() when loading dm from viewer

  Level: advanced

  Notes:
  The PetscSF argument must not be NULL if the DM is distributed, otherwise an error occurs.

.seealso: DMLoad(), DMPlexTopologyLoad(), DMPlexCoordinatesLoad(), DMView(), PetscViewerHDF5Open(), PetscViewerPushFormat()
@*/
PetscErrorCode DMPlexLabelsLoad(DM dm, PetscViewer viewer, PetscSF globalToLocalPointSF)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  if (globalToLocalPointSF) PetscValidHeaderSpecific(globalToLocalPointSF, PETSCSF_CLASSID, 3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_LabelsLoad,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;

    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      CHKERRQ(DMPlexLabelsLoad_HDF5_Internal(dm, viewer, globalToLocalPointSF));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_LabelsLoad,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSectionLoad - Loads section into a DMPlex

  Collective on DM

  Input Parameters:
+ dm          - The DM that represents the topology
. viewer      - The PetscViewer that represents the on-disk section (sectionA)
. sectiondm   - The DM into which the on-disk section (sectionA) is migrated
- globalToLocalPointSF - The SF returned by DMPlexTopologyLoad() when loading dm from viewer

  Output Parameters
+ globalDofSF - The SF that migrates any on-disk Vec data associated with sectionA into a global Vec associated with the sectiondm's global section (NULL if not needed)
- localDofSF  - The SF that migrates any on-disk Vec data associated with sectionA into a local Vec associated with the sectiondm's local section (NULL if not needed)

  Level: advanced

  Notes:
  This function is a wrapper around PetscSectionLoad(); it loads, in addition to the raw section, a list of global point numbers that associates each on-disk section point with a global point number in [0, NX), where NX is the number of topology points in dm. Noting that globalToLocalPointSF associates each topology point in dm with a global number in [0, NX), one can readily establish an association of the on-disk section points with the topology points.

  In general dm and sectiondm are two different objects, the former carrying the topology and the latter carrying the section, and have been given a topology name and a section name, respectively, with PetscObjectSetName(). In practice, however, they can be the same object if it carries both topology and section; in that case the name of the object is used as both the topology name and the section name.

  The output parameter, globalDofSF (localDofSF), can later be used with DMPlexGlobalVectorLoad() (DMPlexLocalVectorLoad()) to load on-disk vectors into global (local) vectors associated with sectiondm's global (local) section.

  Example using 2 processes:
$  NX (number of points on dm): 4
$  sectionA                   : the on-disk section
$  vecA                       : a vector associated with sectionA
$  sectionB                   : sectiondm's local section constructed in this function
$  vecB (local)               : a vector associated with sectiondm's local section
$  vecB (global)              : a vector associated with sectiondm's global section
$
$                                     rank 0    rank 1
$  vecA (global)                  : [.0 .4 .1 | .2 .3]        <- to be loaded in DMPlexGlobalVectorLoad() or DMPlexLocalVectorLoad()
$  sectionA->atlasOff             :       0 2 | 1             <- loaded in PetscSectionLoad()
$  sectionA->atlasDof             :       1 3 | 1             <- loaded in PetscSectionLoad()
$  sectionA's global point numbers:       0 2 | 3             <- loaded in DMPlexSectionLoad()
$  [0, NX)                        :       0 1 | 2 3           <- conceptual partition used in globalToLocalPointSF
$  sectionB's global point numbers:     0 1 3 | 3 2           <- associated with [0, NX) by globalToLocalPointSF
$  sectionB->atlasDof             :     1 0 1 | 1 3
$  sectionB->atlasOff (no perm)   :     0 1 1 | 0 1
$  vecB (local)                   :   [.0 .4] | [.4 .1 .2 .3] <- to be constructed by calling DMPlexLocalVectorLoad() with localDofSF
$  vecB (global)                  :    [.0 .4 | .1 .2 .3]     <- to be constructed by calling DMPlexGlobalVectorLoad() with globalDofSF
$
$  where "|" represents a partition of loaded data, and global point 3 is assumed to be owned by rank 0.

.seealso: DMLoad(), DMPlexTopologyLoad(), DMPlexCoordinatesLoad(), DMPlexLabelsLoad(), DMPlexGlobalVectorLoad(), DMPlexLocalVectorLoad(), PetscSectionLoad(), DMPlexSectionView()
@*/
PetscErrorCode DMPlexSectionLoad(DM dm, PetscViewer viewer, DM sectiondm, PetscSF globalToLocalPointSF, PetscSF *globalDofSF, PetscSF *localDofSF)
{
  PetscBool      ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(sectiondm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(globalToLocalPointSF, PETSCSF_CLASSID, 4);
  if (globalDofSF) PetscValidPointer(globalDofSF, 5);
  if (localDofSF) PetscValidPointer(localDofSF, 6);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_SectionLoad,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexSectionLoad_HDF5_Internal(dm, viewer, sectiondm, globalToLocalPointSF, globalDofSF, localDofSF));
#else
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_SectionLoad,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalVectorLoad - Loads on-disk vector data into a global vector

  Collective on DM

  Input Parameters:
+ dm        - The DM that represents the topology
. viewer    - The PetscViewer that represents the on-disk vector data
. sectiondm - The DM that contains the global section on which vec is defined
. sf        - The SF that migrates the on-disk vector data into vec
- vec       - The global vector to set values of

  Level: advanced

  Notes:
  In general dm and sectiondm are two different objects, the former carrying the topology and the latter carrying the section, and have been given a topology name and a section name, respectively, with PetscObjectSetName(). In practice, however, they can be the same object if it carries both topology and section; in that case the name of the object is used as both the topology name and the section name.

  Typical calling sequence
$       DMCreate(PETSC_COMM_WORLD, &dm);
$       DMSetType(dm, DMPLEX);
$       PetscObjectSetName((PetscObject)dm, "topologydm_name");
$       DMPlexTopologyLoad(dm, viewer, &sfX);
$       DMClone(dm, &sectiondm);
$       PetscObjectSetName((PetscObject)sectiondm, "sectiondm_name");
$       DMPlexSectionLoad(dm, viewer, sectiondm, sfX, &gsf, NULL);
$       DMGetGlobalVector(sectiondm, &vec);
$       PetscObjectSetName((PetscObject)vec, "vec_name");
$       DMPlexGlobalVectorLoad(dm, viewer, sectiondm, gsf, vec);
$       DMRestoreGlobalVector(sectiondm, &vec);
$       PetscSFDestroy(&gsf);
$       PetscSFDestroy(&sfX);
$       DMDestroy(&sectiondm);
$       DMDestroy(&dm);

.seealso: DMPlexTopologyLoad(), DMPlexSectionLoad(), DMPlexLocalVectorLoad(), DMPlexGlobalVectorView(), DMPlexLocalVectorView()
@*/
PetscErrorCode DMPlexGlobalVectorLoad(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sf, Vec vec)
{
  PetscBool       ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(sectiondm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 4);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 5);
  /* Check consistency */
  {
    PetscSection  section;
    PetscBool     includesConstraints;
    PetscInt      m, m1;

    CHKERRQ(VecGetLocalSize(vec, &m1));
    CHKERRQ(DMGetGlobalSection(sectiondm, &section));
    CHKERRQ(PetscSectionGetIncludesConstraints(section, &includesConstraints));
    if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(section, &m));
    else CHKERRQ(PetscSectionGetConstrainedStorageSize(section, &m));
    PetscCheckFalse(m1 != m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Global vector size (%D) != global section storage size (%D)", m1, m);
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_GlobalVectorLoad,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexVecLoad_HDF5_Internal(dm, viewer, sectiondm, sf, vec));
#else
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_GlobalVectorLoad,viewer,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexLocalVectorLoad - Loads on-disk vector data into a local vector

  Collective on DM

  Input Parameters:
+ dm        - The DM that represents the topology
. viewer    - The PetscViewer that represents the on-disk vector data
. sectiondm - The DM that contains the local section on which vec is defined
. sf        - The SF that migrates the on-disk vector data into vec
- vec       - The local vector to set values of

  Level: advanced

  Notes:
  In general dm and sectiondm are two different objects, the former carrying the topology and the latter carrying the section, and have been given a topology name and a section name, respectively, with PetscObjectSetName(). In practice, however, they can be the same object if it carries both topology and section; in that case the name of the object is used as both the topology name and the section name.

  Typical calling sequence
$       DMCreate(PETSC_COMM_WORLD, &dm);
$       DMSetType(dm, DMPLEX);
$       PetscObjectSetName((PetscObject)dm, "topologydm_name");
$       DMPlexTopologyLoad(dm, viewer, &sfX);
$       DMClone(dm, &sectiondm);
$       PetscObjectSetName((PetscObject)sectiondm, "sectiondm_name");
$       DMPlexSectionLoad(dm, viewer, sectiondm, sfX, NULL, &lsf);
$       DMGetLocalVector(sectiondm, &vec);
$       PetscObjectSetName((PetscObject)vec, "vec_name");
$       DMPlexLocalVectorLoad(dm, viewer, sectiondm, lsf, vec);
$       DMRestoreLocalVector(sectiondm, &vec);
$       PetscSFDestroy(&lsf);
$       PetscSFDestroy(&sfX);
$       DMDestroy(&sectiondm);
$       DMDestroy(&dm);

.seealso: DMPlexTopologyLoad(), DMPlexSectionLoad(), DMPlexGlobalVectorLoad(), DMPlexGlobalVectorView(), DMPlexLocalVectorView()
@*/
PetscErrorCode DMPlexLocalVectorLoad(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sf, Vec vec)
{
  PetscBool       ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(sectiondm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 4);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 5);
  /* Check consistency */
  {
    PetscSection  section;
    PetscBool     includesConstraints;
    PetscInt      m, m1;

    CHKERRQ(VecGetLocalSize(vec, &m1));
    CHKERRQ(DMGetLocalSection(sectiondm, &section));
    CHKERRQ(PetscSectionGetIncludesConstraints(section, &includesConstraints));
    if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(section, &m));
    else CHKERRQ(PetscSectionGetConstrainedStorageSize(section, &m));
    PetscCheckFalse(m1 != m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local vector size (%D) != local section storage size (%D)", m1, m);
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
  CHKERRQ(PetscLogEventBegin(DMPLEX_LocalVectorLoad,viewer,0,0,0));
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    CHKERRQ(DMPlexVecLoad_HDF5_Internal(dm, viewer, sectiondm, sf, vec));
#else
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  }
  CHKERRQ(PetscLogEventEnd(DMPLEX_LocalVectorLoad,viewer,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Plex(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertBoundaryValues_C", NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMCreateNeumannOverlap_C", NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"DMInterpolateSolution_C", NULL));
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  CHKERRQ(PetscSectionDestroy(&mesh->coneSection));
  CHKERRQ(PetscFree(mesh->cones));
  CHKERRQ(PetscFree(mesh->coneOrientations));
  CHKERRQ(PetscSectionDestroy(&mesh->supportSection));
  CHKERRQ(PetscSectionDestroy(&mesh->subdomainSection));
  CHKERRQ(PetscFree(mesh->supports));
  CHKERRQ(PetscFree(mesh->facesTmp));
  CHKERRQ(PetscFree(mesh->tetgenOpts));
  CHKERRQ(PetscFree(mesh->triangleOpts));
  CHKERRQ(PetscFree(mesh->transformType));
  CHKERRQ(PetscPartitionerDestroy(&mesh->partitioner));
  CHKERRQ(DMLabelDestroy(&mesh->subpointMap));
  CHKERRQ(ISDestroy(&mesh->subpointIS));
  CHKERRQ(ISDestroy(&mesh->globalVertexNumbers));
  CHKERRQ(ISDestroy(&mesh->globalCellNumbers));
  CHKERRQ(PetscSectionDestroy(&mesh->anchorSection));
  CHKERRQ(ISDestroy(&mesh->anchorIS));
  CHKERRQ(PetscSectionDestroy(&mesh->parentSection));
  CHKERRQ(PetscFree(mesh->parents));
  CHKERRQ(PetscFree(mesh->childIDs));
  CHKERRQ(PetscSectionDestroy(&mesh->childSection));
  CHKERRQ(PetscFree(mesh->children));
  CHKERRQ(DMDestroy(&mesh->referenceTree));
  CHKERRQ(PetscGridHashDestroy(&mesh->lbox));
  CHKERRQ(PetscFree(mesh->neighbors));
  if (mesh->metricCtx) CHKERRQ(PetscFree(mesh->metricCtx));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  CHKERRQ(PetscFree(mesh));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Plex(DM dm, Mat *J)
{
  PetscSection           sectionGlobal;
  PetscInt               bs = -1, mbs;
  PetscInt               localSize;
  PetscBool              isShell, isBlock, isSeqBlock, isMPIBlock, isSymBlock, isSymSeqBlock, isSymMPIBlock, isMatIS;
  MatType                mtype;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  CHKERRQ(MatInitializePackage());
  mtype = dm->mattype;
  CHKERRQ(DMGetGlobalSection(dm, &sectionGlobal));
  /* CHKERRQ(PetscSectionGetStorageSize(sectionGlobal, &localSize)); */
  CHKERRQ(PetscSectionGetConstrainedStorageSize(sectionGlobal, &localSize));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)dm), J));
  CHKERRQ(MatSetSizes(*J, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE));
  CHKERRQ(MatSetType(*J, mtype));
  CHKERRQ(MatSetFromOptions(*J));
  CHKERRQ(MatGetBlockSize(*J, &mbs));
  if (mbs > 1) bs = mbs;
  CHKERRQ(PetscStrcmp(mtype, MATSHELL, &isShell));
  CHKERRQ(PetscStrcmp(mtype, MATBAIJ, &isBlock));
  CHKERRQ(PetscStrcmp(mtype, MATSEQBAIJ, &isSeqBlock));
  CHKERRQ(PetscStrcmp(mtype, MATMPIBAIJ, &isMPIBlock));
  CHKERRQ(PetscStrcmp(mtype, MATSBAIJ, &isSymBlock));
  CHKERRQ(PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock));
  CHKERRQ(PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock));
  CHKERRQ(PetscStrcmp(mtype, MATIS, &isMatIS));
  if (!isShell) {
    PetscBool fillMatrix = (PetscBool)(!dm->prealloc_only && !isMatIS);
    PetscInt  *dnz, *onz, *dnzu, *onzu, bsLocal[2], bsMinMax[2];
    PetscInt  pStart, pEnd, p, dof, cdof;

    CHKERRQ(DMGetLocalToGlobalMapping(dm,&ltog));
    CHKERRQ(PetscSectionGetChart(sectionGlobal, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt bdof;

      CHKERRQ(PetscSectionGetDof(sectionGlobal, p, &dof));
      CHKERRQ(PetscSectionGetConstraintDof(sectionGlobal, p, &cdof));
      dof  = dof < 0 ? -(dof+1) : dof;
      bdof = cdof && (dof-cdof) ? 1 : dof;
      if (dof) {
        if (bs < 0)          {bs = bdof;}
        else if (bs != bdof) {bs = 1; break;}
      }
    }
    /* Must have same blocksize on all procs (some might have no points) */
    bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs;
    bsLocal[1] = bs;
    CHKERRQ(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax));
    if (bsMinMax[0] != bsMinMax[1]) bs = 1;
    else bs = bsMinMax[0];
    bs = PetscMax(1,bs);
    CHKERRQ(MatSetLocalToGlobalMapping(*J,ltog,ltog));
    if (dm->prealloc_skip) { // User will likely use MatSetPreallocationCOO(), but still set structural parameters
      CHKERRQ(MatSetBlockSize(*J, bs));
      CHKERRQ(MatSetUp(*J));
    } else {
      CHKERRQ(PetscCalloc4(localSize/bs, &dnz, localSize/bs, &onz, localSize/bs, &dnzu, localSize/bs, &onzu));
      CHKERRQ(DMPlexPreallocateOperator(dm, bs, dnz, onz, dnzu, onzu, *J, fillMatrix));
      CHKERRQ(PetscFree4(dnz, onz, dnzu, onzu));
    }
  }
  CHKERRQ(MatSetDM(*J, dm));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetSubdomainSection - Returns the section associated with the subdomain

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameters:
. subsection - The subdomain section

  Level: developer

.seealso:
@*/
PetscErrorCode DMPlexGetSubdomainSection(DM dm, PetscSection *subsection)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->subdomainSection) {
    PetscSection section;
    PetscSF      sf;

    CHKERRQ(PetscSFCreate(PETSC_COMM_SELF,&sf));
    CHKERRQ(DMGetLocalSection(dm,&section));
    CHKERRQ(PetscSectionCreateGlobalSection(section,sf,PETSC_FALSE,PETSC_TRUE,&mesh->subdomainSection));
    CHKERRQ(PetscSFDestroy(&sf));
  }
  *subsection = mesh->subdomainSection;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetChart - Return the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameters:
+ pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetChart(DM dm, PetscInt *pStart, PetscInt *pEnd)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->coneSection, pStart, pEnd));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetChart - Set the interval for all mesh points [pStart, pEnd)

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. pStart - The first mesh point
- pEnd   - The upper bound for mesh points

  Output Parameters:

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetChart()
@*/
PetscErrorCode DMPlexSetChart(DM dm, PetscInt pStart, PetscInt pEnd)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionSetChart(mesh->coneSection, pStart, pEnd));
  CHKERRQ(PetscSectionSetChart(mesh->supportSection, pStart, pEnd));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetConeSize - Return the number of in-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. size - The cone size for point p

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetConeSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, size));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetConeSize - Set the number of in-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- size - The cone size for point p

  Output Parameter:

  Note:
  This should be called after DMPlexSetChart().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexSetConeSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionSetDof(mesh->coneSection, p, size));

  mesh->maxConeSize = PetscMax(mesh->maxConeSize, size);
  PetscFunctionReturn(0);
}

/*@
  DMPlexAddConeSize - Add the given number of in-edges to this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- size - The additional cone size for point p

  Output Parameter:

  Note:
  This should be called after DMPlexSetChart().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexGetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexAddConeSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       csize;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionAddDof(mesh->coneSection, p, size));
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &csize));

  mesh->maxConeSize = PetscMax(mesh->maxConeSize, csize);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetCone - Return the points on the in-edges for this point in the DAG

  Not collective

  Input Parameters:
+ dm - The DMPlex
- p - The point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. cone - An array of points which are on the in-edges for point p

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.
  You must also call DMPlexRestoreCone() after you finish using the returned array.
  DMPlexRestoreCone() is not needed/available in C.

.seealso: DMPlexGetConeSize(), DMPlexSetCone(), DMPlexGetConeTuple(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetCone(DM dm, PetscInt p, const PetscInt *cone[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
  *cone = &mesh->cones[off];
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetConeTuple - Return the points on the in-edges of several points in the DAG

  Not collective

  Input Parameters:
+ dm - The DMPlex
- p - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameters:
+ pConesSection - PetscSection describing the layout of pCones
- pCones - An array of points which are on the in-edges for the point set p

  Level: intermediate

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexGetConeRecursive(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetConeTuple(DM dm, IS p, PetscSection *pConesSection, IS *pCones)
{
  PetscSection        cs, newcs;
  PetscInt            *cones;
  PetscInt            *newarr=NULL;
  PetscInt            n;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetCones(dm, &cones));
  CHKERRQ(DMPlexGetConeSection(dm, &cs));
  CHKERRQ(PetscSectionExtractDofsFromArray(cs, MPIU_INT, cones, p, &newcs, pCones ? ((void**)&newarr) : NULL));
  if (pConesSection) *pConesSection = newcs;
  if (pCones) {
    CHKERRQ(PetscSectionGetStorageSize(newcs, &n));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)p), n, newarr, PETSC_OWN_POINTER, pCones));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetConeRecursiveVertices - Expand each given point into its cone points and do that recursively until we end up just with vertices.

  Not collective

  Input Parameters:
+ dm - The DMPlex
- points - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. expandedPoints - An array of vertices recursively expanded from input points

  Level: advanced

  Notes:
  Like DMPlexGetConeRecursive but returns only the 0-depth IS (i.e. vertices only) and no sections.
  There is no corresponding Restore function, just call ISDestroy() on the returned IS to deallocate.

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexGetConeTuple(), DMPlexGetConeRecursive(), DMPlexRestoreConeRecursive(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexGetConeRecursiveVertices(DM dm, IS points, IS *expandedPoints)
{
  IS                  *expandedPointsAll;
  PetscInt            depth;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(points, IS_CLASSID, 2);
  PetscValidPointer(expandedPoints, 3);
  CHKERRQ(DMPlexGetConeRecursive(dm, points, &depth, &expandedPointsAll, NULL));
  *expandedPoints = expandedPointsAll[0];
  CHKERRQ(PetscObjectReference((PetscObject)expandedPointsAll[0]));
  CHKERRQ(DMPlexRestoreConeRecursive(dm, points, &depth, &expandedPointsAll, NULL));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetConeRecursive - Expand each given point into its cone points and do that recursively until we end up just with vertices (DAG points of depth 0, i.e. without cones).

  Not collective

  Input Parameters:
+ dm - The DMPlex
- points - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameters:
+ depth - (optional) Size of the output arrays, equal to DMPlex depth, returned by DMPlexGetDepth()
. expandedPoints - (optional) An array of index sets with recursively expanded cones
- sections - (optional) An array of sections which describe mappings from points to their cone points

  Level: advanced

  Notes:
  Like DMPlexGetConeTuple() but recursive.

  Array expandedPoints has size equal to depth. Each expandedPoints[d] contains DAG points with maximum depth d, recursively cone-wise expanded from the input points.
  For example, for d=0 it contains only vertices, for d=1 it can contain vertices and edges, etc.

  Array section has size equal to depth.  Each PetscSection sections[d] realizes mapping from expandedPoints[d+1] (section points) to expandedPoints[d] (section dofs) as follows:
  (1) DAG points in expandedPoints[d+1] with depth d+1 to their cone points in expandedPoints[d];
  (2) DAG points in expandedPoints[d+1] with depth in [0,d] to the same points in expandedPoints[d].

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexGetConeTuple(), DMPlexRestoreConeRecursive(), DMPlexGetConeRecursiveVertices(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexGetConeRecursive(DM dm, IS points, PetscInt *depth, IS *expandedPoints[], PetscSection *sections[])
{
  const PetscInt      *arr0=NULL, *cone=NULL;
  PetscInt            *arr=NULL, *newarr=NULL;
  PetscInt            d, depth_, i, n, newn, cn, co, start, end;
  IS                  *expandedPoints_;
  PetscSection        *sections_;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(points, IS_CLASSID, 2);
  if (depth) PetscValidIntPointer(depth, 3);
  if (expandedPoints) PetscValidPointer(expandedPoints, 4);
  if (sections) PetscValidPointer(sections, 5);
  CHKERRQ(ISGetLocalSize(points, &n));
  CHKERRQ(ISGetIndices(points, &arr0));
  CHKERRQ(DMPlexGetDepth(dm, &depth_));
  CHKERRQ(PetscCalloc1(depth_, &expandedPoints_));
  CHKERRQ(PetscCalloc1(depth_, &sections_));
  arr = (PetscInt*) arr0; /* this is ok because first generation of arr is not modified */
  for (d=depth_-1; d>=0; d--) {
    CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &sections_[d]));
    CHKERRQ(PetscSectionSetChart(sections_[d], 0, n));
    for (i=0; i<n; i++) {
      CHKERRQ(DMPlexGetDepthStratum(dm, d+1, &start, &end));
      if (arr[i] >= start && arr[i] < end) {
        CHKERRQ(DMPlexGetConeSize(dm, arr[i], &cn));
        CHKERRQ(PetscSectionSetDof(sections_[d], i, cn));
      } else {
        CHKERRQ(PetscSectionSetDof(sections_[d], i, 1));
      }
    }
    CHKERRQ(PetscSectionSetUp(sections_[d]));
    CHKERRQ(PetscSectionGetStorageSize(sections_[d], &newn));
    CHKERRQ(PetscMalloc1(newn, &newarr));
    for (i=0; i<n; i++) {
      CHKERRQ(PetscSectionGetDof(sections_[d], i, &cn));
      CHKERRQ(PetscSectionGetOffset(sections_[d], i, &co));
      if (cn > 1) {
        CHKERRQ(DMPlexGetCone(dm, arr[i], &cone));
        CHKERRQ(PetscMemcpy(&newarr[co], cone, cn*sizeof(PetscInt)));
      } else {
        newarr[co] = arr[i];
      }
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, newn, newarr, PETSC_OWN_POINTER, &expandedPoints_[d]));
    arr = newarr;
    n = newn;
  }
  CHKERRQ(ISRestoreIndices(points, &arr0));
  *depth = depth_;
  if (expandedPoints) *expandedPoints = expandedPoints_;
  else {
    for (d=0; d<depth_; d++) CHKERRQ(ISDestroy(&expandedPoints_[d]));
    CHKERRQ(PetscFree(expandedPoints_));
  }
  if (sections) *sections = sections_;
  else {
    for (d=0; d<depth_; d++) CHKERRQ(PetscSectionDestroy(&sections_[d]));
    CHKERRQ(PetscFree(sections_));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexRestoreConeRecursive - Deallocates arrays created by DMPlexGetConeRecursive

  Not collective

  Input Parameters:
+ dm - The DMPlex
- points - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameters:
+ depth - (optional) Size of the output arrays, equal to DMPlex depth, returned by DMPlexGetDepth()
. expandedPoints - (optional) An array of recursively expanded cones
- sections - (optional) An array of sections which describe mappings from points to their cone points

  Level: advanced

  Notes:
  See DMPlexGetConeRecursive() for details.

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexGetConeTuple(), DMPlexGetConeRecursive(), DMPlexGetConeRecursiveVertices(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexRestoreConeRecursive(DM dm, IS points, PetscInt *depth, IS *expandedPoints[], PetscSection *sections[])
{
  PetscInt            d, depth_;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(dm, &depth_));
  PetscCheckFalse(depth && *depth != depth_,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "depth changed since last call to DMPlexGetConeRecursive");
  if (depth) *depth = 0;
  if (expandedPoints) {
    for (d=0; d<depth_; d++) CHKERRQ(ISDestroy(&((*expandedPoints)[d])));
    CHKERRQ(PetscFree(*expandedPoints));
  }
  if (sections)  {
    for (d=0; d<depth_; d++) CHKERRQ(PetscSectionDestroy(&((*sections)[d])));
    CHKERRQ(PetscFree(*sections));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetCone - Set the points on the in-edges for this point in the DAG; that is these are the points that cover the specific point

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- cone - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp(), DMPlexSetSupport(), DMPlexSetSupportSize()
@*/
PetscErrorCode DMPlexSetCone(DM dm, PetscInt p, const PetscInt cone[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
  if (dof) PetscValidPointer(cone, 3);
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
  PetscCheckFalse((p < pStart) || (p >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    PetscCheckFalse((cone[c] < pStart) || (cone[c] >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", cone[c], pStart, pEnd);
    mesh->cones[off+c] = cone[c];
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetConeOrientation - Return the orientations on the in-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. coneOrientation - An array of orientations which are on the in-edges for point p. An orientation is an
                    integer giving the prescription for cone traversal.

  Level: beginner

  Notes:
  The number indexes the symmetry transformations for the cell type (see manual). Orientation 0 is always
  the identity transformation. Negative orientation indicates reflection so that -(o+1) is the reflection
  of o, however it is not necessarily the inverse. To get the inverse, use DMPolytopeTypeComposeOrientationInv()
  with the identity.

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.
  You must also call DMPlexRestoreConeOrientation() after you finish using the returned array.
  DMPlexRestoreConeOrientation() is not needed/available in C.

.seealso: DMPolytopeTypeComposeOrientation(), DMPolytopeTypeComposeOrientationInv(), DMPlexCreate(), DMPlexGetCone(), DMPlexSetCone(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetConeOrientation(DM dm, PetscInt p, const PetscInt *coneOrientation[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt dof;
    CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
    if (dof) PetscValidPointer(coneOrientation, 3);
  }
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));

  *coneOrientation = &mesh->coneOrientations[off];
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetConeOrientation - Set the orientations on the in-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- coneOrientation - An array of orientations
  Output Parameter:

  Notes:
  This should be called after all calls to DMPlexSetConeSize() and DMSetUp().

  The meaning of coneOrientation is detailed in DMPlexGetConeOrientation().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetConeOrientation(), DMPlexSetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexSetConeOrientation(DM dm, PetscInt p, const PetscInt coneOrientation[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
  if (dof) PetscValidPointer(coneOrientation, 3);
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
  PetscCheckFalse((p < pStart) || (p >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    PetscInt cdof, o = coneOrientation[c];

    CHKERRQ(PetscSectionGetDof(mesh->coneSection, mesh->cones[off+c], &cdof));
    PetscCheckFalse(o && ((o < -(cdof+1)) || (o >= cdof)),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone orientation %D is not in the valid range [%D. %D)", o, -(cdof+1), cdof);
    mesh->coneOrientations[off+c] = o;
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertCone - Insert a point into the in-edges for the point p in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
. conePos - The local index in the cone where the point should be put
- conePoint - The mesh point to insert

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexInsertCone(DM dm, PetscInt p, PetscInt conePos, PetscInt conePoint)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd));
  PetscCheckFalse((p < pStart) || (p >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  PetscCheckFalse((conePoint < pStart) || (conePoint >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", conePoint, pStart, pEnd);
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
  PetscCheckFalse((conePos < 0) || (conePos >= dof),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
  mesh->cones[off+conePos] = conePoint;
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertConeOrientation - Insert a point orientation for the in-edge for the point p in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
. conePos - The local index in the cone where the point should be put
- coneOrientation - The point orientation to insert

  Level: beginner

  Notes:
  The meaning of coneOrientation values is detailed in DMPlexGetConeOrientation().

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexInsertConeOrientation(DM dm, PetscInt p, PetscInt conePos, PetscInt coneOrientation)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd));
  PetscCheckFalse((p < pStart) || (p >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
  PetscCheckFalse((conePos < 0) || (conePos >= dof),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
  mesh->coneOrientations[off+conePos] = coneOrientation;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetSupportSize - Return the number of out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. size - The support size for point p

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart(), DMPlexGetConeSize()
@*/
PetscErrorCode DMPlexGetSupportSize(DM dm, PetscInt p, PetscInt *size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  CHKERRQ(PetscSectionGetDof(mesh->supportSection, p, size));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetSupportSize - Set the number of out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- size - The support size for point p

  Output Parameter:

  Note:
  This should be called after DMPlexSetChart().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetSupportSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexSetSupportSize(DM dm, PetscInt p, PetscInt size)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionSetDof(mesh->supportSection, p, size));

  mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, size);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetSupport - Return the points on the out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. support - An array of points which are on the out-edges for point p

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.
  You must also call DMPlexRestoreSupport() after you finish using the returned array.
  DMPlexRestoreSupport() is not needed/available in C.

.seealso: DMPlexGetSupportSize(), DMPlexSetSupport(), DMPlexGetCone(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetSupport(DM dm, PetscInt p, const PetscInt *support[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(support, 3);
  CHKERRQ(PetscSectionGetOffset(mesh->supportSection, p, &off));
  *support = &mesh->supports[off];
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetSupport - Set the points on the out-edges for this point in the DAG, that is the list of points that this point covers

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- support - An array of points which are on the out-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetSupportSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexSetCone(), DMPlexSetConeSize(), DMPlexCreate(), DMPlexGetSupport(), DMPlexSetChart(), DMPlexSetSupportSize(), DMSetUp()
@*/
PetscErrorCode DMPlexSetSupport(DM dm, PetscInt p, const PetscInt support[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetDof(mesh->supportSection, p, &dof));
  if (dof) PetscValidPointer(support, 3);
  CHKERRQ(PetscSectionGetOffset(mesh->supportSection, p, &off));
  PetscCheckFalse((p < pStart) || (p >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    PetscCheckFalse((support[c] < pStart) || (support[c] >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", support[c], pStart, pEnd);
    mesh->supports[off+c] = support[c];
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexInsertSupport - Insert a point into the out-edges for the point p in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
. supportPos - The local index in the cone where the point should be put
- supportPoint - The mesh point to insert

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexInsertSupport(DM dm, PetscInt p, PetscInt supportPos, PetscInt supportPoint)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetDof(mesh->supportSection, p, &dof));
  CHKERRQ(PetscSectionGetOffset(mesh->supportSection, p, &off));
  PetscCheckFalse((p < pStart) || (p >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  PetscCheckFalse((supportPoint < pStart) || (supportPoint >= pEnd),PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", supportPoint, pStart, pEnd);
  PetscCheckFalse(supportPos >= dof,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support position %D of point %D is not in the valid range [0, %D)", supportPos, p, dof);
  mesh->supports[off+supportPos] = supportPoint;
  PetscFunctionReturn(0);
}

/* Converts an orientation o in the current numbering to the previous scheme used in Plex */
PetscInt DMPolytopeConvertNewOrientation_Internal(DMPolytopeType ct, PetscInt o)
{
  switch (ct) {
    case DM_POLYTOPE_SEGMENT:
      if (o == -1) return -2;
      break;
    case DM_POLYTOPE_TRIANGLE:
      if (o == -3) return -1;
      if (o == -2) return -3;
      if (o == -1) return -2;
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      if (o == -4) return -2;
      if (o == -3) return -1;
      if (o == -2) return -4;
      if (o == -1) return -3;
      break;
    default: return o;
  }
  return o;
}

/* Converts an orientation o in the previous scheme used in Plex to the current numbering */
PetscInt DMPolytopeConvertOldOrientation_Internal(DMPolytopeType ct, PetscInt o)
{
  switch (ct) {
    case DM_POLYTOPE_SEGMENT:
      if ((o == -2) || (o == 1)) return -1;
      if (o == -1) return 0;
      break;
    case DM_POLYTOPE_TRIANGLE:
      if (o == -3) return -2;
      if (o == -2) return -1;
      if (o == -1) return -3;
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      if (o == -4) return -2;
      if (o == -3) return -1;
      if (o == -2) return -4;
      if (o == -1) return -3;
      break;
    default: return o;
  }
  return o;
}

/* Takes in a mesh whose orientations are in the previous scheme and converts them all to the current numbering */
PetscErrorCode DMPlexConvertOldOrientations_Internal(DM dm)
{
  PetscInt       pStart, pEnd, p;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *ornt;
    PetscInt        coneSize, c;

    CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, p, &cone));
    CHKERRQ(DMPlexGetConeOrientation(dm, p, &ornt));
    for (c = 0; c < coneSize; ++c) {
      DMPolytopeType ct;
      const PetscInt o = ornt[c];

      CHKERRQ(DMPlexGetCellType(dm, cone[c], &ct));
      switch (ct) {
        case DM_POLYTOPE_SEGMENT:
          if ((o == -2) || (o == 1)) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -1));
          if (o == -1) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, 0));
          break;
        case DM_POLYTOPE_TRIANGLE:
          if (o == -3) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -2));
          if (o == -2) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -1));
          if (o == -1) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -3));
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          if (o == -4) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -2));
          if (o == -3) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -1));
          if (o == -2) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -4));
          if (o == -1) CHKERRQ(DMPlexInsertConeOrientation(dm, p, c, -3));
          break;
        default: break;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetTransitiveClosure_Depth1_Private(DM dm, PetscInt p, PetscInt ornt, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DMPolytopeType  ct = DM_POLYTOPE_UNKNOWN;
  PetscInt       *closure;
  const PetscInt *tmp = NULL, *tmpO = NULL;
  PetscInt        off = 0, tmpSize, t;

  PetscFunctionBeginHot;
  if (ornt) {
    CHKERRQ(DMPlexGetCellType(dm, p, &ct));
    if (ct == DM_POLYTOPE_FV_GHOST || ct == DM_POLYTOPE_INTERIOR_GHOST || ct == DM_POLYTOPE_UNKNOWN) ct = DM_POLYTOPE_UNKNOWN;
  }
  if (*points) {
    closure = *points;
  } else {
    PetscInt maxConeSize, maxSupportSize;
    CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
    CHKERRQ(DMGetWorkArray(dm, 2*(PetscMax(maxConeSize, maxSupportSize)+1), MPIU_INT, &closure));
  }
  if (useCone) {
    CHKERRQ(DMPlexGetConeSize(dm, p, &tmpSize));
    CHKERRQ(DMPlexGetCone(dm, p, &tmp));
    CHKERRQ(DMPlexGetConeOrientation(dm, p, &tmpO));
  } else {
    CHKERRQ(DMPlexGetSupportSize(dm, p, &tmpSize));
    CHKERRQ(DMPlexGetSupport(dm, p, &tmp));
  }
  if (ct == DM_POLYTOPE_UNKNOWN) {
    closure[off++] = p;
    closure[off++] = 0;
    for (t = 0; t < tmpSize; ++t) {
      closure[off++] = tmp[t];
      closure[off++] = tmpO ? tmpO[t] : 0;
    }
  } else {
    const PetscInt *arr = DMPolytopeTypeGetArrangment(ct, ornt);

    /* We assume that cells with a valid type have faces with a valid type */
    closure[off++] = p;
    closure[off++] = ornt;
    for (t = 0; t < tmpSize; ++t) {
      DMPolytopeType ft;

      CHKERRQ(DMPlexGetCellType(dm, tmp[t], &ft));
      closure[off++] = tmp[arr[t]];
      closure[off++] = tmpO ? DMPolytopeTypeComposeOrientation(ft, ornt, tmpO[t]) : 0;
    }
  }
  if (numPoints) *numPoints = tmpSize+1;
  if (points)    *points    = closure;
  PetscFunctionReturn(0);
}

/* We need a special tensor verison becasue we want to allow duplicate points in the endcaps for hybrid cells */
static PetscErrorCode DMPlexTransitiveClosure_Tensor_Internal(DM dm, PetscInt point, DMPolytopeType ct, PetscInt o, PetscBool useCone, PetscInt *numPoints, PetscInt **points)
{
  const PetscInt *arr = DMPolytopeTypeGetArrangment(ct, o);
  const PetscInt *cone, *ornt;
  PetscInt       *pts,  *closure = NULL;
  DMPolytopeType  ft;
  PetscInt        maxConeSize, maxSupportSize, coneSeries, supportSeries, maxSize;
  PetscInt        dim, coneSize, c, d, clSize, cl;

  PetscFunctionBeginHot;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
  CHKERRQ(DMPlexGetCone(dm, point, &cone));
  CHKERRQ(DMPlexGetConeOrientation(dm, point, &ornt));
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  coneSeries    = (maxConeSize    > 1) ? ((PetscPowInt(maxConeSize,    dim+1)-1)/(maxConeSize-1))    : dim+1;
  supportSeries = (maxSupportSize > 1) ? ((PetscPowInt(maxSupportSize, dim+1)-1)/(maxSupportSize-1)) : dim+1;
  maxSize       = PetscMax(coneSeries, supportSeries);
  if (*points) {pts  = *points;}
  else         CHKERRQ(DMGetWorkArray(dm, 2*maxSize, MPIU_INT, &pts));
  c    = 0;
  pts[c++] = point;
  pts[c++] = o;
  CHKERRQ(DMPlexGetCellType(dm, cone[arr[0*2+0]], &ft));
  CHKERRQ(DMPlexGetTransitiveClosure_Internal(dm, cone[arr[0*2+0]], DMPolytopeTypeComposeOrientation(ft, arr[0*2+1], ornt[0]), useCone, &clSize, &closure));
  for (cl = 0; cl < clSize*2; cl += 2) {pts[c++] = closure[cl]; pts[c++] = closure[cl+1];}
  CHKERRQ(DMPlexGetTransitiveClosure_Internal(dm, cone[arr[1*2+0]], DMPolytopeTypeComposeOrientation(ft, arr[1*2+1], ornt[1]), useCone, &clSize, &closure));
  for (cl = 0; cl < clSize*2; cl += 2) {pts[c++] = closure[cl]; pts[c++] = closure[cl+1];}
  CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cone[0], useCone, &clSize, &closure));
  for (d = 2; d < coneSize; ++d) {
    CHKERRQ(DMPlexGetCellType(dm, cone[arr[d*2+0]], &ft));
    pts[c++] = cone[arr[d*2+0]];
    pts[c++] = DMPolytopeTypeComposeOrientation(ft, arr[d*2+1], ornt[d]);
  }
  if (dim >= 3) {
    for (d = 2; d < coneSize; ++d) {
      const PetscInt  fpoint = cone[arr[d*2+0]];
      const PetscInt *fcone, *fornt;
      PetscInt        fconeSize, fc, i;

      CHKERRQ(DMPlexGetCellType(dm, fpoint, &ft));
      const PetscInt *farr = DMPolytopeTypeGetArrangment(ft, DMPolytopeTypeComposeOrientation(ft, arr[d*2+1], ornt[d]));
      CHKERRQ(DMPlexGetConeSize(dm, fpoint, &fconeSize));
      CHKERRQ(DMPlexGetCone(dm, fpoint, &fcone));
      CHKERRQ(DMPlexGetConeOrientation(dm, fpoint, &fornt));
      for (fc = 0; fc < fconeSize; ++fc) {
        const PetscInt cp = fcone[farr[fc*2+0]];
        const PetscInt co = farr[fc*2+1];

        for (i = 0; i < c; i += 2) if (pts[i] == cp) break;
        if (i == c) {
          CHKERRQ(DMPlexGetCellType(dm, cp, &ft));
          pts[c++] = cp;
          pts[c++] = DMPolytopeTypeComposeOrientation(ft, co, fornt[farr[fc*2+0]]);
        }
      }
    }
  }
  *numPoints = c/2;
  *points    = pts;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetTransitiveClosure_Internal(DM dm, PetscInt p, PetscInt ornt, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DMPolytopeType ct;
  PetscInt      *closure, *fifo;
  PetscInt       closureSize = 0, fifoStart = 0, fifoSize = 0;
  PetscInt       maxConeSize, maxSupportSize, coneSeries, supportSeries;
  PetscInt       depth, maxSize;

  PetscFunctionBeginHot;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (depth == 1) {
    CHKERRQ(DMPlexGetTransitiveClosure_Depth1_Private(dm, p, ornt, useCone, numPoints, points));
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMPlexGetCellType(dm, p, &ct));
  if (ct == DM_POLYTOPE_FV_GHOST || ct == DM_POLYTOPE_INTERIOR_GHOST || ct == DM_POLYTOPE_UNKNOWN) ct = DM_POLYTOPE_UNKNOWN;
  if (ct == DM_POLYTOPE_SEG_PRISM_TENSOR || ct == DM_POLYTOPE_TRI_PRISM_TENSOR || ct == DM_POLYTOPE_QUAD_PRISM_TENSOR) {
    CHKERRQ(DMPlexTransitiveClosure_Tensor_Internal(dm, p, ct, ornt, useCone, numPoints, points));
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));
  coneSeries    = (maxConeSize    > 1) ? ((PetscPowInt(maxConeSize,    depth+1)-1)/(maxConeSize-1))    : depth+1;
  supportSeries = (maxSupportSize > 1) ? ((PetscPowInt(maxSupportSize, depth+1)-1)/(maxSupportSize-1)) : depth+1;
  maxSize       = PetscMax(coneSeries, supportSeries);
  CHKERRQ(DMGetWorkArray(dm, 3*maxSize, MPIU_INT, &fifo));
  if (*points) {closure = *points;}
  else         CHKERRQ(DMGetWorkArray(dm, 2*maxSize, MPIU_INT, &closure));
  closure[closureSize++] = p;
  closure[closureSize++] = ornt;
  fifo[fifoSize++]       = p;
  fifo[fifoSize++]       = ornt;
  fifo[fifoSize++]       = ct;
  /* Should kick out early when depth is reached, rather than checking all vertices for empty cones */
  while (fifoSize - fifoStart) {
    const PetscInt       q    = fifo[fifoStart++];
    const PetscInt       o    = fifo[fifoStart++];
    const DMPolytopeType qt   = (DMPolytopeType) fifo[fifoStart++];
    const PetscInt      *qarr = DMPolytopeTypeGetArrangment(qt, o);
    const PetscInt      *tmp, *tmpO;
    PetscInt             tmpSize, t;

    if (PetscDefined(USE_DEBUG)) {
      PetscInt nO = DMPolytopeTypeGetNumArrangments(qt)/2;
      PetscCheckFalse(o && (o >= nO || o < -nO),PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid orientation %D not in [%D,%D) for %s %D", o, -nO, nO, DMPolytopeTypes[qt], q);
    }
    if (useCone) {
      CHKERRQ(DMPlexGetConeSize(dm, q, &tmpSize));
      CHKERRQ(DMPlexGetCone(dm, q, &tmp));
      CHKERRQ(DMPlexGetConeOrientation(dm, q, &tmpO));
    } else {
      CHKERRQ(DMPlexGetSupportSize(dm, q, &tmpSize));
      CHKERRQ(DMPlexGetSupport(dm, q, &tmp));
      tmpO = NULL;
    }
    for (t = 0; t < tmpSize; ++t) {
      const PetscInt ip = useCone && qarr ? qarr[t*2]   : t;
      const PetscInt io = useCone && qarr ? qarr[t*2+1] : 0;
      const PetscInt cp = tmp[ip];
      CHKERRQ(DMPlexGetCellType(dm, cp, &ct));
      const PetscInt co = tmpO ? DMPolytopeTypeComposeOrientation(ct, io, tmpO[ip]) : 0;
      PetscInt       c;

      /* Check for duplicate */
      for (c = 0; c < closureSize; c += 2) {
        if (closure[c] == cp) break;
      }
      if (c == closureSize) {
        closure[closureSize++] = cp;
        closure[closureSize++] = co;
        fifo[fifoSize++]       = cp;
        fifo[fifoSize++]       = co;
        fifo[fifoSize++]       = ct;
      }
    }
  }
  CHKERRQ(DMRestoreWorkArray(dm, 3*maxSize, MPIU_INT, &fifo));
  if (numPoints) *numPoints = closureSize/2;
  if (points)    *points    = closure;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetTransitiveClosure - Return the points on the transitive closure of the in-edges or out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ dm      - The DMPlex
. p       - The mesh point
- useCone - PETSC_TRUE for the closure, otherwise return the star

  Input/Output Parameter:
. points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...];
           if NULL on input, internal storage will be returned, otherwise the provided array is used

  Output Parameter:
. numPoints - The number of points in the closure, so points[] is of size 2*numPoints

  Note:
  If using internal storage (points is NULL on input), each call overwrites the last output.

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexRestoreTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numPoints) PetscValidIntPointer(numPoints, 4);
  if (points)    PetscValidPointer(points, 5);
  CHKERRQ(DMPlexGetTransitiveClosure_Internal(dm, p, 0, useCone, numPoints, points));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreTransitiveClosure - Restore the array of points on the transitive closure of the in-edges or out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ dm        - The DMPlex
. p         - The mesh point
. useCone   - PETSC_TRUE for the closure, otherwise return the star
. numPoints - The number of points in the closure, so points[] is of size 2*numPoints
- points    - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...]

  Note:
  If not using internal storage (points is not NULL on input), this call is unnecessary

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexGetTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexRestoreTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numPoints) *numPoints = 0;
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, points));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMaxSizes - Return the maximum number of in-edges (cone) and out-edges (support) for any point in the DAG

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameters:
+ maxConeSize - The maximum number of in-edges
- maxSupportSize - The maximum number of out-edges

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetMaxSizes(DM dm, PetscInt *maxConeSize, PetscInt *maxSupportSize)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (maxConeSize)    *maxConeSize    = mesh->maxConeSize;
  if (maxSupportSize) *maxSupportSize = mesh->maxSupportSize;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUp_Plex(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionSetUp(mesh->coneSection));
  CHKERRQ(PetscSectionGetStorageSize(mesh->coneSection, &size));
  CHKERRQ(PetscMalloc1(size, &mesh->cones));
  CHKERRQ(PetscCalloc1(size, &mesh->coneOrientations));
  CHKERRQ(PetscLogObjectMemory((PetscObject) dm, size*2*sizeof(PetscInt)));
  if (mesh->maxSupportSize) {
    CHKERRQ(PetscSectionSetUp(mesh->supportSection));
    CHKERRQ(PetscSectionGetStorageSize(mesh->supportSection, &size));
    CHKERRQ(PetscMalloc1(size, &mesh->supports));
    CHKERRQ(PetscLogObjectMemory((PetscObject) dm, size*sizeof(PetscInt)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_Plex(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscFunctionBegin;
  if (subdm) CHKERRQ(DMClone(dm, subdm));
  CHKERRQ(DMCreateSectionSubDM(dm, numFields, fields, is, subdm));
  if (subdm) {(*subdm)->useNatural = dm->useNatural;}
  if (dm->useNatural && dm->sfMigration) {
    PetscSF        sfMigrationInv,sfNatural;
    PetscSection   section, sectionSeq;

    (*subdm)->sfMigration = dm->sfMigration;
    CHKERRQ(PetscObjectReference((PetscObject) dm->sfMigration));
    CHKERRQ(DMGetLocalSection((*subdm), &section));
    CHKERRQ(PetscSFCreateInverseSF((*subdm)->sfMigration, &sfMigrationInv));
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) (*subdm)), &sectionSeq));
    CHKERRQ(PetscSFDistributeSection(sfMigrationInv, section, NULL, sectionSeq));

    CHKERRQ(DMPlexCreateGlobalToNaturalSF(*subdm, sectionSeq, (*subdm)->sfMigration, &sfNatural));
    (*subdm)->sfNatural = sfNatural;
    CHKERRQ(PetscSectionDestroy(&sectionSeq));
    CHKERRQ(PetscSFDestroy(&sfMigrationInv));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSuperDM_Plex(DM dms[], PetscInt len, IS **is, DM *superdm)
{
  PetscInt       i = 0;

  PetscFunctionBegin;
  CHKERRQ(DMClone(dms[0], superdm));
  CHKERRQ(DMCreateSectionSuperDM(dms, len, is, superdm));
  (*superdm)->useNatural = PETSC_FALSE;
  for (i = 0; i < len; i++) {
    if (dms[i]->useNatural && dms[i]->sfMigration) {
      PetscSF        sfMigrationInv,sfNatural;
      PetscSection   section, sectionSeq;

      (*superdm)->sfMigration = dms[i]->sfMigration;
      CHKERRQ(PetscObjectReference((PetscObject) dms[i]->sfMigration));
      (*superdm)->useNatural = PETSC_TRUE;
      CHKERRQ(DMGetLocalSection((*superdm), &section));
      CHKERRQ(PetscSFCreateInverseSF((*superdm)->sfMigration, &sfMigrationInv));
      CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) (*superdm)), &sectionSeq));
      CHKERRQ(PetscSFDistributeSection(sfMigrationInv, section, NULL, sectionSeq));

      CHKERRQ(DMPlexCreateGlobalToNaturalSF(*superdm, sectionSeq, (*superdm)->sfMigration, &sfNatural));
      (*superdm)->sfNatural = sfNatural;
      CHKERRQ(PetscSectionDestroy(&sectionSeq));
      CHKERRQ(PetscSFDestroy(&sfMigrationInv));
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexSymmetrize - Create support (out-edge) information from cone (in-edge) information

  Not collective

  Input Parameter:
. mesh - The DMPlex

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetCone()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetChart(), DMPlexSetConeSize(), DMPlexSetCone()
@*/
PetscErrorCode DMPlexSymmetrize(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *offsets;
  PetscInt       supportSize;
  PetscInt       pStart, pEnd, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(!mesh->supports,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Supports were already setup in this DMPlex");
  CHKERRQ(PetscLogEventBegin(DMPLEX_Symmetrize,dm,0,0,0));
  /* Calculate support sizes */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
    CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
    for (c = off; c < off+dof; ++c) {
      CHKERRQ(PetscSectionAddDof(mesh->supportSection, mesh->cones[c], 1));
    }
  }
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    CHKERRQ(PetscSectionGetDof(mesh->supportSection, p, &dof));

    mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, dof);
  }
  CHKERRQ(PetscSectionSetUp(mesh->supportSection));
  /* Calculate supports */
  CHKERRQ(PetscSectionGetStorageSize(mesh->supportSection, &supportSize));
  CHKERRQ(PetscMalloc1(supportSize, &mesh->supports));
  CHKERRQ(PetscCalloc1(pEnd - pStart, &offsets));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    CHKERRQ(PetscSectionGetDof(mesh->coneSection, p, &dof));
    CHKERRQ(PetscSectionGetOffset(mesh->coneSection, p, &off));
    for (c = off; c < off+dof; ++c) {
      const PetscInt q = mesh->cones[c];
      PetscInt       offS;

      CHKERRQ(PetscSectionGetOffset(mesh->supportSection, q, &offS));

      mesh->supports[offS+offsets[q]] = p;
      ++offsets[q];
    }
  }
  CHKERRQ(PetscFree(offsets));
  CHKERRQ(PetscLogEventEnd(DMPLEX_Symmetrize,dm,0,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateDepthStratum(DM dm, DMLabel label, PetscInt depth, PetscInt pStart, PetscInt pEnd)
{
  IS             stratumIS;

  PetscFunctionBegin;
  if (pStart >= pEnd) PetscFunctionReturn(0);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt  qStart, qEnd, numLevels, level;
    PetscBool overlap = PETSC_FALSE;
    CHKERRQ(DMLabelGetNumValues(label, &numLevels));
    for (level = 0; level < numLevels; level++) {
      CHKERRQ(DMLabelGetStratumBounds(label, level, &qStart, &qEnd));
      if ((pStart >= qStart && pStart < qEnd) || (pEnd > qStart && pEnd <= qEnd)) {overlap = PETSC_TRUE; break;}
    }
    PetscCheck(!overlap,PETSC_COMM_SELF, PETSC_ERR_PLIB, "New depth %D range [%D,%D) overlaps with depth %D range [%D,%D)", depth, pStart, pEnd, level, qStart, qEnd);
  }
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, pEnd-pStart, pStart, 1, &stratumIS));
  CHKERRQ(DMLabelSetStratumIS(label, depth, stratumIS));
  CHKERRQ(ISDestroy(&stratumIS));
  PetscFunctionReturn(0);
}

/*@
  DMPlexStratify - The DAG for most topologies is a graded poset (https://en.wikipedia.org/wiki/Graded_poset), and
  can be illustrated by a Hasse Diagram (https://en.wikipedia.org/wiki/Hasse_diagram). The strata group all points of the
  same grade, and this function calculates the strata. This grade can be seen as the height (or depth) of the point in
  the DAG.

  Collective on dm

  Input Parameter:
. mesh - The DMPlex

  Output Parameter:

  Notes:
  Concretely, DMPlexStratify() creates a new label named "depth" containing the depth in the DAG of each point. For cell-vertex
  meshes, vertices are depth 0 and cells are depth 1. For fully interpolated meshes, depth 0 for vertices, 1 for edges, and so on
  until cells have depth equal to the dimension of the mesh. The depth label can be accessed through DMPlexGetDepthLabel() or DMPlexGetDepthStratum(), or
  manually via DMGetLabel().  The height is defined implicitly by height = maxDimension - depth, and can be accessed
  via DMPlexGetHeightStratum().  For example, cells have height 0 and faces have height 1.

  The depth of a point is calculated by executing a breadth-first search (BFS) on the DAG. This could produce surprising results
  if run on a partially interpolated mesh, meaning one that had some edges and faces, but not others. For example, suppose that
  we had a mesh consisting of one triangle (c0) and three vertices (v0, v1, v2), and only one edge is on the boundary so we choose
  to interpolate only that one (e0), so that
$  cone(c0) = {e0, v2}
$  cone(e0) = {v0, v1}
  If DMPlexStratify() is run on this mesh, it will give depths
$  depth 0 = {v0, v1, v2}
$  depth 1 = {e0, c0}
  where the triangle has been given depth 1, instead of 2, because it is reachable from vertex v2.

  DMPlexStratify() should be called after all calls to DMPlexSymmetrize()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSymmetrize(), DMPlexComputeCellTypes()
@*/
PetscErrorCode DMPlexStratify(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  DMLabel        label;
  PetscInt       pStart, pEnd, p;
  PetscInt       numRoots = 0, numLeaves = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscLogEventBegin(DMPLEX_Stratify,dm,0,0,0));

  /* Create depth label */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(DMCreateLabel(dm, "depth"));
  CHKERRQ(DMPlexGetDepthLabel(dm, &label));

  {
    /* Initialize roots and count leaves */
    PetscInt sMin = PETSC_MAX_INT;
    PetscInt sMax = PETSC_MIN_INT;
    PetscInt coneSize, supportSize;

    for (p = pStart; p < pEnd; ++p) {
      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      CHKERRQ(DMPlexGetSupportSize(dm, p, &supportSize));
      if (!coneSize && supportSize) {
        sMin = PetscMin(p, sMin);
        sMax = PetscMax(p, sMax);
        ++numRoots;
      } else if (!supportSize && coneSize) {
        ++numLeaves;
      } else if (!supportSize && !coneSize) {
        /* Isolated points */
        sMin = PetscMin(p, sMin);
        sMax = PetscMax(p, sMax);
      }
    }
    CHKERRQ(DMPlexCreateDepthStratum(dm, label, 0, sMin, sMax+1));
  }

  if (numRoots + numLeaves == (pEnd - pStart)) {
    PetscInt sMin = PETSC_MAX_INT;
    PetscInt sMax = PETSC_MIN_INT;
    PetscInt coneSize, supportSize;

    for (p = pStart; p < pEnd; ++p) {
      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      CHKERRQ(DMPlexGetSupportSize(dm, p, &supportSize));
      if (!supportSize && coneSize) {
        sMin = PetscMin(p, sMin);
        sMax = PetscMax(p, sMax);
      }
    }
    CHKERRQ(DMPlexCreateDepthStratum(dm, label, 1, sMin, sMax+1));
  } else {
    PetscInt level = 0;
    PetscInt qStart, qEnd, q;

    CHKERRQ(DMLabelGetStratumBounds(label, level, &qStart, &qEnd));
    while (qEnd > qStart) {
      PetscInt sMin = PETSC_MAX_INT;
      PetscInt sMax = PETSC_MIN_INT;

      for (q = qStart; q < qEnd; ++q) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        CHKERRQ(DMPlexGetSupportSize(dm, q, &supportSize));
        CHKERRQ(DMPlexGetSupport(dm, q, &support));
        for (s = 0; s < supportSize; ++s) {
          sMin = PetscMin(support[s], sMin);
          sMax = PetscMax(support[s], sMax);
        }
      }
      CHKERRQ(DMLabelGetNumValues(label, &level));
      CHKERRQ(DMPlexCreateDepthStratum(dm, label, level, sMin, sMax+1));
      CHKERRQ(DMLabelGetStratumBounds(label, level, &qStart, &qEnd));
    }
  }
  { /* just in case there is an empty process */
    PetscInt numValues, maxValues = 0, v;

    CHKERRQ(DMLabelGetNumValues(label, &numValues));
    CHKERRMPI(MPI_Allreduce(&numValues,&maxValues,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm)));
    for (v = numValues; v < maxValues; v++) {
      CHKERRQ(DMLabelAddStratum(label, v));
    }
  }
  CHKERRQ(PetscObjectStateGet((PetscObject) label, &mesh->depthState));
  CHKERRQ(PetscLogEventEnd(DMPLEX_Stratify,dm,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeCellType_Internal(DM dm, PetscInt p, PetscInt pdepth, DMPolytopeType *pt)
{
  DMPolytopeType ct = DM_POLYTOPE_UNKNOWN;
  PetscInt       dim, depth, pheight, coneSize;

  PetscFunctionBeginHot;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
  pheight = depth - pdepth;
  if (depth <= 1) {
    switch (pdepth) {
      case 0: ct = DM_POLYTOPE_POINT;break;
      case 1:
        switch (coneSize) {
          case 2: ct = DM_POLYTOPE_SEGMENT;break;
          case 3: ct = DM_POLYTOPE_TRIANGLE;break;
          case 4:
          switch (dim) {
            case 2: ct = DM_POLYTOPE_QUADRILATERAL;break;
            case 3: ct = DM_POLYTOPE_TETRAHEDRON;break;
            default: break;
          }
          break;
        case 5: ct = DM_POLYTOPE_PYRAMID;break;
        case 6: ct = DM_POLYTOPE_TRI_PRISM_TENSOR;break;
        case 8: ct = DM_POLYTOPE_HEXAHEDRON;break;
        default: break;
      }
    }
  } else {
    if (pdepth == 0) {
      ct = DM_POLYTOPE_POINT;
    } else if (pheight == 0) {
      switch (dim) {
        case 1:
          switch (coneSize) {
            case 2: ct = DM_POLYTOPE_SEGMENT;break;
            default: break;
          }
          break;
        case 2:
          switch (coneSize) {
            case 3: ct = DM_POLYTOPE_TRIANGLE;break;
            case 4: ct = DM_POLYTOPE_QUADRILATERAL;break;
            default: break;
          }
          break;
        case 3:
          switch (coneSize) {
            case 4: ct = DM_POLYTOPE_TETRAHEDRON;break;
            case 5:
            {
              const PetscInt *cone;
              PetscInt        faceConeSize;

              CHKERRQ(DMPlexGetCone(dm, p, &cone));
              CHKERRQ(DMPlexGetConeSize(dm, cone[0], &faceConeSize));
              switch (faceConeSize) {
                case 3: ct = DM_POLYTOPE_TRI_PRISM_TENSOR;break;
                case 4: ct = DM_POLYTOPE_PYRAMID;break;
              }
            }
            break;
            case 6: ct = DM_POLYTOPE_HEXAHEDRON;break;
            default: break;
          }
          break;
        default: break;
      }
    } else if (pheight > 0) {
      switch (coneSize) {
        case 2: ct = DM_POLYTOPE_SEGMENT;break;
        case 3: ct = DM_POLYTOPE_TRIANGLE;break;
        case 4: ct = DM_POLYTOPE_QUADRILATERAL;break;
        default: break;
      }
    }
  }
  *pt = ct;
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeCellTypes - Infer the polytope type of every cell using its dimension and cone size.

  Collective on dm

  Input Parameter:
. mesh - The DMPlex

  DMPlexComputeCellTypes() should be called after all calls to DMPlexSymmetrize() and DMPlexStratify()

  Level: developer

  Note: This function is normally called automatically by Plex when a cell type is requested. It creates an
  internal DMLabel named "celltype" which can be directly accessed using DMGetLabel(). A user may disable
  automatic creation by creating the label manually, using DMCreateLabel(dm, "celltype").

.seealso: DMPlexCreate(), DMPlexSymmetrize(), DMPlexStratify(), DMGetLabel(), DMCreateLabel()
@*/
PetscErrorCode DMPlexComputeCellTypes(DM dm)
{
  DM_Plex       *mesh;
  DMLabel        ctLabel;
  PetscInt       pStart, pEnd, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh = (DM_Plex *) dm->data;
  CHKERRQ(DMCreateLabel(dm, "celltype"));
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &ctLabel));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct = DM_POLYTOPE_UNKNOWN;
    PetscInt       pdepth;

    CHKERRQ(DMPlexGetPointDepth(dm, p, &pdepth));
    CHKERRQ(DMPlexComputeCellType_Internal(dm, p, pdepth, &ct));
    PetscCheckFalse(ct == DM_POLYTOPE_UNKNOWN,PETSC_COMM_SELF, PETSC_ERR_SUP, "Point %D is screwed up", p);
    CHKERRQ(DMLabelSetValue(ctLabel, p, ct));
  }
  CHKERRQ(PetscObjectStateGet((PetscObject) ctLabel, &mesh->celltypeState));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) ctLabel, NULL, "-dm_plex_celltypes_view"));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetJoin - Get an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Level: intermediate

  Note: Currently, this is restricted to a single level join

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.seealso: DMPlexRestoreJoin(), DMPlexGetMeet()
@*/
PetscErrorCode DMPlexGetJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *join[2];
  PetscInt       joinSize, i = 0;
  PetscInt       dof, off, p, c, m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(points, 3);
  PetscValidIntPointer(numCoveredPoints, 4);
  PetscValidPointer(coveredPoints, 5);
  CHKERRQ(DMGetWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[0]));
  CHKERRQ(DMGetWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[1]));
  /* Copy in support of first point */
  CHKERRQ(PetscSectionGetDof(mesh->supportSection, points[0], &dof));
  CHKERRQ(PetscSectionGetOffset(mesh->supportSection, points[0], &off));
  for (joinSize = 0; joinSize < dof; ++joinSize) {
    join[i][joinSize] = mesh->supports[off+joinSize];
  }
  /* Check each successive support */
  for (p = 1; p < numPoints; ++p) {
    PetscInt newJoinSize = 0;

    CHKERRQ(PetscSectionGetDof(mesh->supportSection, points[p], &dof));
    CHKERRQ(PetscSectionGetOffset(mesh->supportSection, points[p], &off));
    for (c = 0; c < dof; ++c) {
      const PetscInt point = mesh->supports[off+c];

      for (m = 0; m < joinSize; ++m) {
        if (point == join[i][m]) {
          join[1-i][newJoinSize++] = point;
          break;
        }
      }
    }
    joinSize = newJoinSize;
    i        = 1-i;
  }
  *numCoveredPoints = joinSize;
  *coveredPoints    = join[i];
  CHKERRQ(DMRestoreWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[1-i]));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreJoin - Restore an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.seealso: DMPlexGetJoin(), DMPlexGetFullJoin(), DMPlexGetMeet()
@*/
PetscErrorCode DMPlexRestoreJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (points) PetscValidIntPointer(points,3);
  if (numCoveredPoints) PetscValidIntPointer(numCoveredPoints,4);
  PetscValidPointer(coveredPoints, 5);
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, (void*) coveredPoints));
  if (numCoveredPoints) *numCoveredPoints = 0;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetFullJoin - Get an array for the join of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the join
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the join
- coveredPoints - The points in the join

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.seealso: DMPlexGetJoin(), DMPlexRestoreJoin(), DMPlexGetMeet()
@*/
PetscErrorCode DMPlexGetFullJoin(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *offsets, **closures;
  PetscInt      *join[2];
  PetscInt       depth = 0, maxSize, joinSize = 0, i = 0;
  PetscInt       p, d, c, m, ms;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(points, 3);
  PetscValidIntPointer(numCoveredPoints, 4);
  PetscValidPointer(coveredPoints, 5);

  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(PetscCalloc1(numPoints, &closures));
  CHKERRQ(DMGetWorkArray(dm, numPoints*(depth+2), MPIU_INT, &offsets));
  ms      = mesh->maxSupportSize;
  maxSize = (ms > 1) ? ((PetscPowInt(ms,depth+1)-1)/(ms-1)) : depth + 1;
  CHKERRQ(DMGetWorkArray(dm, maxSize, MPIU_INT, &join[0]));
  CHKERRQ(DMGetWorkArray(dm, maxSize, MPIU_INT, &join[1]));

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closures[p]));

    offsets[p*(depth+2)+0] = 0;
    for (d = 0; d < depth+1; ++d) {
      PetscInt pStart, pEnd, i;

      CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
      for (i = offsets[p*(depth+2)+d]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(depth+2)+d+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(depth+2)+d+1] = i;
    }
    PetscCheckFalse(offsets[p*(depth+2)+depth+1] != closureSize,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(depth+2)+depth+1], closureSize);
  }
  for (d = 0; d < depth+1; ++d) {
    PetscInt dof;

    /* Copy in support of first point */
    dof = offsets[d+1] - offsets[d];
    for (joinSize = 0; joinSize < dof; ++joinSize) {
      join[i][joinSize] = closures[0][(offsets[d]+joinSize)*2];
    }
    /* Check each successive cone */
    for (p = 1; p < numPoints && joinSize; ++p) {
      PetscInt newJoinSize = 0;

      dof = offsets[p*(depth+2)+d+1] - offsets[p*(depth+2)+d];
      for (c = 0; c < dof; ++c) {
        const PetscInt point = closures[p][(offsets[p*(depth+2)+d]+c)*2];

        for (m = 0; m < joinSize; ++m) {
          if (point == join[i][m]) {
            join[1-i][newJoinSize++] = point;
            break;
          }
        }
      }
      joinSize = newJoinSize;
      i        = 1-i;
    }
    if (joinSize) break;
  }
  *numCoveredPoints = joinSize;
  *coveredPoints    = join[i];
  for (p = 0; p < numPoints; ++p) {
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, NULL, &closures[p]));
  }
  CHKERRQ(PetscFree(closures));
  CHKERRQ(DMRestoreWorkArray(dm, numPoints*(depth+2), MPIU_INT, &offsets));
  CHKERRQ(DMRestoreWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[1-i]));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetMeet - Get an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Note: Currently, this is restricted to a single level meet

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.seealso: DMPlexRestoreMeet(), DMPlexGetJoin()
@*/
PetscErrorCode DMPlexGetMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveringPoints, const PetscInt **coveringPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *meet[2];
  PetscInt       meetSize, i = 0;
  PetscInt       dof, off, p, c, m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 3);
  PetscValidPointer(numCoveringPoints, 4);
  PetscValidPointer(coveringPoints, 5);
  CHKERRQ(DMGetWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[0]));
  CHKERRQ(DMGetWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[1]));
  /* Copy in cone of first point */
  CHKERRQ(PetscSectionGetDof(mesh->coneSection, points[0], &dof));
  CHKERRQ(PetscSectionGetOffset(mesh->coneSection, points[0], &off));
  for (meetSize = 0; meetSize < dof; ++meetSize) {
    meet[i][meetSize] = mesh->cones[off+meetSize];
  }
  /* Check each successive cone */
  for (p = 1; p < numPoints; ++p) {
    PetscInt newMeetSize = 0;

    CHKERRQ(PetscSectionGetDof(mesh->coneSection, points[p], &dof));
    CHKERRQ(PetscSectionGetOffset(mesh->coneSection, points[p], &off));
    for (c = 0; c < dof; ++c) {
      const PetscInt point = mesh->cones[off+c];

      for (m = 0; m < meetSize; ++m) {
        if (point == meet[i][m]) {
          meet[1-i][newMeetSize++] = point;
          break;
        }
      }
    }
    meetSize = newMeetSize;
    i        = 1-i;
  }
  *numCoveringPoints = meetSize;
  *coveringPoints    = meet[i];
  CHKERRQ(DMRestoreWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[1-i]));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreMeet - Restore an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.seealso: DMPlexGetMeet(), DMPlexGetFullMeet(), DMPlexGetJoin()
@*/
PetscErrorCode DMPlexRestoreMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (points) PetscValidIntPointer(points,3);
  if (numCoveredPoints) PetscValidIntPointer(numCoveredPoints,4);
  PetscValidPointer(coveredPoints,5);
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, (void*) coveredPoints));
  if (numCoveredPoints) *numCoveredPoints = 0;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetFullMeet - Get an array for the meet of the set of points

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
. numPoints - The number of input points for the meet
- points - The input points

  Output Parameters:
+ numCoveredPoints - The number of points in the meet
- coveredPoints - The points in the meet

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numCoveredPoints argument is not present in the Fortran 90 binding since it is internal to the array.

.seealso: DMPlexGetMeet(), DMPlexRestoreMeet(), DMPlexGetJoin()
@*/
PetscErrorCode DMPlexGetFullMeet(DM dm, PetscInt numPoints, const PetscInt points[], PetscInt *numCoveredPoints, const PetscInt **coveredPoints)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt      *offsets, **closures;
  PetscInt      *meet[2];
  PetscInt       height = 0, maxSize, meetSize = 0, i = 0;
  PetscInt       p, h, c, m, mc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 3);
  PetscValidPointer(numCoveredPoints, 4);
  PetscValidPointer(coveredPoints, 5);

  CHKERRQ(DMPlexGetDepth(dm, &height));
  CHKERRQ(PetscMalloc1(numPoints, &closures));
  CHKERRQ(DMGetWorkArray(dm, numPoints*(height+2), MPIU_INT, &offsets));
  mc      = mesh->maxConeSize;
  maxSize = (mc > 1) ? ((PetscPowInt(mc,height+1)-1)/(mc-1)) : height + 1;
  CHKERRQ(DMGetWorkArray(dm, maxSize, MPIU_INT, &meet[0]));
  CHKERRQ(DMGetWorkArray(dm, maxSize, MPIU_INT, &meet[1]));

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closures[p]));

    offsets[p*(height+2)+0] = 0;
    for (h = 0; h < height+1; ++h) {
      PetscInt pStart, pEnd, i;

      CHKERRQ(DMPlexGetHeightStratum(dm, h, &pStart, &pEnd));
      for (i = offsets[p*(height+2)+h]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(height+2)+h+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(height+2)+h+1] = i;
    }
    PetscCheckFalse(offsets[p*(height+2)+height+1] != closureSize,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(height+2)+height+1], closureSize);
  }
  for (h = 0; h < height+1; ++h) {
    PetscInt dof;

    /* Copy in cone of first point */
    dof = offsets[h+1] - offsets[h];
    for (meetSize = 0; meetSize < dof; ++meetSize) {
      meet[i][meetSize] = closures[0][(offsets[h]+meetSize)*2];
    }
    /* Check each successive cone */
    for (p = 1; p < numPoints && meetSize; ++p) {
      PetscInt newMeetSize = 0;

      dof = offsets[p*(height+2)+h+1] - offsets[p*(height+2)+h];
      for (c = 0; c < dof; ++c) {
        const PetscInt point = closures[p][(offsets[p*(height+2)+h]+c)*2];

        for (m = 0; m < meetSize; ++m) {
          if (point == meet[i][m]) {
            meet[1-i][newMeetSize++] = point;
            break;
          }
        }
      }
      meetSize = newMeetSize;
      i        = 1-i;
    }
    if (meetSize) break;
  }
  *numCoveredPoints = meetSize;
  *coveredPoints    = meet[i];
  for (p = 0; p < numPoints; ++p) {
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, NULL, &closures[p]));
  }
  CHKERRQ(PetscFree(closures));
  CHKERRQ(DMRestoreWorkArray(dm, numPoints*(height+2), MPIU_INT, &offsets));
  CHKERRQ(DMRestoreWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[1-i]));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexEqual - Determine if two DMs have the same topology

  Not Collective

  Input Parameters:
+ dmA - A DMPlex object
- dmB - A DMPlex object

  Output Parameters:
. equal - PETSC_TRUE if the topologies are identical

  Level: intermediate

  Notes:
  We are not solving graph isomorphism, so we do not permutation.

.seealso: DMPlexGetCone()
@*/
PetscErrorCode DMPlexEqual(DM dmA, DM dmB, PetscBool *equal)
{
  PetscInt       depth, depthB, pStart, pEnd, pStartB, pEndB, p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  PetscValidPointer(equal, 3);

  *equal = PETSC_FALSE;
  CHKERRQ(DMPlexGetDepth(dmA, &depth));
  CHKERRQ(DMPlexGetDepth(dmB, &depthB));
  if (depth != depthB) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetChart(dmA, &pStart,  &pEnd));
  CHKERRQ(DMPlexGetChart(dmB, &pStartB, &pEndB));
  if ((pStart != pStartB) || (pEnd != pEndB)) PetscFunctionReturn(0);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *coneB, *ornt, *orntB, *support, *supportB;
    PetscInt        coneSize, coneSizeB, c, supportSize, supportSizeB, s;

    CHKERRQ(DMPlexGetConeSize(dmA, p, &coneSize));
    CHKERRQ(DMPlexGetCone(dmA, p, &cone));
    CHKERRQ(DMPlexGetConeOrientation(dmA, p, &ornt));
    CHKERRQ(DMPlexGetConeSize(dmB, p, &coneSizeB));
    CHKERRQ(DMPlexGetCone(dmB, p, &coneB));
    CHKERRQ(DMPlexGetConeOrientation(dmB, p, &orntB));
    if (coneSize != coneSizeB) PetscFunctionReturn(0);
    for (c = 0; c < coneSize; ++c) {
      if (cone[c] != coneB[c]) PetscFunctionReturn(0);
      if (ornt[c] != orntB[c]) PetscFunctionReturn(0);
    }
    CHKERRQ(DMPlexGetSupportSize(dmA, p, &supportSize));
    CHKERRQ(DMPlexGetSupport(dmA, p, &support));
    CHKERRQ(DMPlexGetSupportSize(dmB, p, &supportSizeB));
    CHKERRQ(DMPlexGetSupport(dmB, p, &supportB));
    if (supportSize != supportSizeB) PetscFunctionReturn(0);
    for (s = 0; s < supportSize; ++s) {
      if (support[s] != supportB[s]) PetscFunctionReturn(0);
    }
  }
  *equal = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetNumFaceVertices - Returns the number of vertices on a face

  Not Collective

  Input Parameters:
+ dm         - The DMPlex
. cellDim    - The cell dimension
- numCorners - The number of vertices on a cell

  Output Parameters:
. numFaceVertices - The number of vertices on a face

  Level: developer

  Notes:
  Of course this can only work for a restricted set of symmetric shapes

.seealso: DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetNumFaceVertices(DM dm, PetscInt cellDim, PetscInt numCorners, PetscInt *numFaceVertices)
{
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscValidPointer(numFaceVertices,4);
  switch (cellDim) {
  case 0:
    *numFaceVertices = 0;
    break;
  case 1:
    *numFaceVertices = 1;
    break;
  case 2:
    switch (numCorners) {
    case 3: /* triangle */
      *numFaceVertices = 2; /* Edge has 2 vertices */
      break;
    case 4: /* quadrilateral */
      *numFaceVertices = 2; /* Edge has 2 vertices */
      break;
    case 6: /* quadratic triangle, tri and quad cohesive Lagrange cells */
      *numFaceVertices = 3; /* Edge has 3 vertices */
      break;
    case 9: /* quadratic quadrilateral, quadratic quad cohesive Lagrange cells */
      *numFaceVertices = 3; /* Edge has 3 vertices */
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %D for dimension %D", numCorners, cellDim);
    }
    break;
  case 3:
    switch (numCorners) {
    case 4: /* tetradehdron */
      *numFaceVertices = 3; /* Face has 3 vertices */
      break;
    case 6: /* tet cohesive cells */
      *numFaceVertices = 4; /* Face has 4 vertices */
      break;
    case 8: /* hexahedron */
      *numFaceVertices = 4; /* Face has 4 vertices */
      break;
    case 9: /* tet cohesive Lagrange cells */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 10: /* quadratic tetrahedron */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 12: /* hex cohesive Lagrange cells */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 18: /* quadratic tet cohesive Lagrange cells */
      *numFaceVertices = 6; /* Face has 6 vertices */
      break;
    case 27: /* quadratic hexahedron, quadratic hex cohesive Lagrange cells */
      *numFaceVertices = 9; /* Face has 9 vertices */
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %D for dimension %D", numCorners, cellDim);
    }
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell dimension %D", cellDim);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetDepthLabel - Get the DMLabel recording the depth of each point

  Not Collective

  Input Parameter:
. dm    - The DMPlex object

  Output Parameter:
. depthLabel - The DMLabel recording point depth

  Level: developer

.seealso: DMPlexGetDepth(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum(), DMPlexGetPointDepth(),
@*/
PetscErrorCode DMPlexGetDepthLabel(DM dm, DMLabel *depthLabel)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depthLabel, 2);
  *depthLabel = dm->depthLabel;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetDepth - Get the depth of the DAG representing this mesh

  Not Collective

  Input Parameter:
. dm    - The DMPlex object

  Output Parameter:
. depth - The number of strata (breadth first levels) in the DAG

  Level: developer

  Notes:
  This returns maximum of point depths over all points, i.e. maximum value of the label returned by DMPlexGetDepthLabel().
  The point depth is described more in detail in DMPlexGetDepthStratum().
  An empty mesh gives -1.

.seealso: DMPlexGetDepthLabel(), DMPlexGetDepthStratum(), DMPlexGetPointDepth(), DMPlexSymmetrize()
@*/
PetscErrorCode DMPlexGetDepth(DM dm, PetscInt *depth)
{
  DMLabel        label;
  PetscInt       d = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depth, 2);
  CHKERRQ(DMPlexGetDepthLabel(dm, &label));
  if (label) CHKERRQ(DMLabelGetNumValues(label, &d));
  *depth = d-1;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetDepthStratum - Get the bounds [start, end) for all points at a certain depth.

  Not Collective

  Input Parameters:
+ dm           - The DMPlex object
- stratumValue - The requested depth

  Output Parameters:
+ start - The first point at this depth
- end   - One beyond the last point at this depth

  Notes:
  Depth indexing is related to topological dimension.  Depth stratum 0 contains the lowest topological dimension points,
  often "vertices".  If the mesh is "interpolated" (see DMPlexInterpolate()), then depth stratum 1 contains the next
  higher dimension, e.g., "edges".

  Level: developer

.seealso: DMPlexGetHeightStratum(), DMPlexGetDepth(), DMPlexGetDepthLabel(), DMPlexGetPointDepth(), DMPlexSymmetrize(), DMPlexInterpolate()
@*/
PetscErrorCode DMPlexGetDepthStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end)
{
  DMLabel        label;
  PetscInt       pStart, pEnd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) {PetscValidPointer(start, 3); *start = 0;}
  if (end)   {PetscValidPointer(end,   4); *end   = 0;}
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  if (pStart == pEnd) PetscFunctionReturn(0);
  if (stratumValue < 0) {
    if (start) *start = pStart;
    if (end)   *end   = pEnd;
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMPlexGetDepthLabel(dm, &label));
  PetscCheck(label,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");
  CHKERRQ(DMLabelGetStratumBounds(label, stratumValue, start, end));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetHeightStratum - Get the bounds [start, end) for all points at a certain height.

  Not Collective

  Input Parameters:
+ dm           - The DMPlex object
- stratumValue - The requested height

  Output Parameters:
+ start - The first point at this height
- end   - One beyond the last point at this height

  Notes:
  Height indexing is related to topological codimension.  Height stratum 0 contains the highest topological dimension
  points, often called "cells" or "elements".  If the mesh is "interpolated" (see DMPlexInterpolate()), then height
  stratum 1 contains the boundary of these "cells", often called "faces" or "facets".

  Level: developer

.seealso: DMPlexGetDepthStratum(), DMPlexGetDepth(), DMPlexGetPointHeight()
@*/
PetscErrorCode DMPlexGetHeightStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end)
{
  DMLabel        label;
  PetscInt       depth, pStart, pEnd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) {PetscValidPointer(start, 3); *start = 0;}
  if (end)   {PetscValidPointer(end,   4); *end   = 0;}
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  if (pStart == pEnd) PetscFunctionReturn(0);
  if (stratumValue < 0) {
    if (start) *start = pStart;
    if (end)   *end   = pEnd;
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMPlexGetDepthLabel(dm, &label));
  PetscCheck(label,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");
  CHKERRQ(DMLabelGetNumValues(label, &depth));
  CHKERRQ(DMLabelGetStratumBounds(label, depth-1-stratumValue, start, end));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPointDepth - Get the depth of a given point

  Not Collective

  Input Parameters:
+ dm    - The DMPlex object
- point - The point

  Output Parameter:
. depth - The depth of the point

  Level: intermediate

.seealso: DMPlexGetCellType(), DMPlexGetDepthLabel(), DMPlexGetDepth(), DMPlexGetPointHeight()
@*/
PetscErrorCode DMPlexGetPointDepth(DM dm, PetscInt point, PetscInt *depth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(depth, 3);
  CHKERRQ(DMLabelGetValue(dm->depthLabel, point, depth));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPointHeight - Get the height of a given point

  Not Collective

  Input Parameters:
+ dm    - The DMPlex object
- point - The point

  Output Parameter:
. height - The height of the point

  Level: intermediate

.seealso: DMPlexGetCellType(), DMPlexGetDepthLabel(), DMPlexGetDepth(), DMPlexGetPointDepth()
@*/
PetscErrorCode DMPlexGetPointHeight(DM dm, PetscInt point, PetscInt *height)
{
  PetscInt       n, pDepth;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(height, 3);
  CHKERRQ(DMLabelGetNumValues(dm->depthLabel, &n));
  CHKERRQ(DMLabelGetValue(dm->depthLabel, point, &pDepth));
  *height = n - 1 - pDepth;  /* DAG depth is n-1 */
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetCellTypeLabel - Get the DMLabel recording the polytope type of each cell

  Not Collective

  Input Parameter:
. dm - The DMPlex object

  Output Parameter:
. celltypeLabel - The DMLabel recording cell polytope type

  Note: This function will trigger automatica computation of cell types. This can be disabled by calling
  DMCreateLabel(dm, "celltype") beforehand.

  Level: developer

.seealso: DMPlexGetCellType(), DMPlexGetDepthLabel(), DMCreateLabel()
@*/
PetscErrorCode DMPlexGetCellTypeLabel(DM dm, DMLabel *celltypeLabel)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(celltypeLabel, 2);
  if (!dm->celltypeLabel) CHKERRQ(DMPlexComputeCellTypes(dm));
  *celltypeLabel = dm->celltypeLabel;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetCellType - Get the polytope type of a given cell

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
- cell - The cell

  Output Parameter:
. celltype - The polytope type of the cell

  Level: intermediate

.seealso: DMPlexGetCellTypeLabel(), DMPlexGetDepthLabel(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexGetCellType(DM dm, PetscInt cell, DMPolytopeType *celltype)
{
  DMLabel        label;
  PetscInt       ct;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(celltype, 3);
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &label));
  CHKERRQ(DMLabelGetValue(label, cell, &ct));
  PetscCheckFalse(ct < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cell %D has not been assigned a cell type", cell);
  *celltype = (DMPolytopeType) ct;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetCellType - Set the polytope type of a given cell

  Not Collective

  Input Parameters:
+ dm   - The DMPlex object
. cell - The cell
- celltype - The polytope type of the cell

  Note: By default, cell types will be automatically computed using DMPlexComputeCellTypes() before this function
  is executed. This function will override the computed type. However, if automatic classification will not succeed
  and a user wants to manually specify all types, the classification must be disabled by calling
  DMCreaateLabel(dm, "celltype") before getting or setting any cell types.

  Level: advanced

.seealso: DMPlexGetCellTypeLabel(), DMPlexGetDepthLabel(), DMPlexGetDepth(), DMPlexComputeCellTypes(), DMCreateLabel()
@*/
PetscErrorCode DMPlexSetCellType(DM dm, PetscInt cell, DMPolytopeType celltype)
{
  DMLabel        label;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &label));
  CHKERRQ(DMLabelSetValue(label, cell, celltype));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateCoordinateDM_Plex(DM dm, DM *cdm)
{
  PetscSection   section, s;
  Mat            m;
  PetscInt       maxHeight;

  PetscFunctionBegin;
  CHKERRQ(DMClone(dm, cdm));
  CHKERRQ(DMPlexGetMaxProjectionHeight(dm, &maxHeight));
  CHKERRQ(DMPlexSetMaxProjectionHeight(*cdm, maxHeight));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  CHKERRQ(DMSetLocalSection(*cdm, section));
  CHKERRQ(PetscSectionDestroy(&section));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &s));
  CHKERRQ(MatCreate(PETSC_COMM_SELF, &m));
  CHKERRQ(DMSetDefaultConstraints(*cdm, s, m, NULL));
  CHKERRQ(PetscSectionDestroy(&s));
  CHKERRQ(MatDestroy(&m));

  CHKERRQ(DMSetNumFields(*cdm, 1));
  CHKERRQ(DMCreateDS(*cdm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateCoordinateField_Plex(DM dm, DMField *field)
{
  Vec            coordsLocal;
  DM             coordsDM;

  PetscFunctionBegin;
  *field = NULL;
  CHKERRQ(DMGetCoordinatesLocal(dm,&coordsLocal));
  CHKERRQ(DMGetCoordinateDM(dm,&coordsDM));
  if (coordsLocal && coordsDM) {
    CHKERRQ(DMFieldCreateDS(coordsDM, 0, coordsLocal, field));
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetConeSection - Return a section which describes the layout of cone data

  Not Collective

  Input Parameters:
. dm        - The DMPlex object

  Output Parameter:
. section - The PetscSection object

  Level: developer

.seealso: DMPlexGetSupportSection(), DMPlexGetCones(), DMPlexGetConeOrientations()
@*/
PetscErrorCode DMPlexGetConeSection(DM dm, PetscSection *section)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) *section = mesh->coneSection;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetSupportSection - Return a section which describes the layout of support data

  Not Collective

  Input Parameters:
. dm        - The DMPlex object

  Output Parameter:
. section - The PetscSection object

  Level: developer

.seealso: DMPlexGetConeSection()
@*/
PetscErrorCode DMPlexGetSupportSection(DM dm, PetscSection *section)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (section) *section = mesh->supportSection;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetCones - Return cone data

  Not Collective

  Input Parameters:
. dm        - The DMPlex object

  Output Parameter:
. cones - The cone for each point

  Level: developer

.seealso: DMPlexGetConeSection()
@*/
PetscErrorCode DMPlexGetCones(DM dm, PetscInt *cones[])
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cones) *cones = mesh->cones;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetConeOrientations - Return cone orientation data

  Not Collective

  Input Parameters:
. dm        - The DMPlex object

  Output Parameter:
. coneOrientations - The array of cone orientations for all points

  Level: developer

  Notes:
  The PetscSection returned by DMPlexGetConeSection() partitions coneOrientations into cone orientations of particular points as returned by DMPlexGetConeOrientation().

  The meaning of coneOrientations values is detailed in DMPlexGetConeOrientation().

.seealso: DMPlexGetConeSection(), DMPlexGetConeOrientation()
@*/
PetscErrorCode DMPlexGetConeOrientations(DM dm, PetscInt *coneOrientations[])
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (coneOrientations) *coneOrientations = mesh->coneOrientations;
  PetscFunctionReturn(0);
}

/******************************** FEM Support **********************************/

/*
 Returns number of components and tensor degree for the field.  For interpolated meshes, line should be a point
 representing a line in the section.
*/
static PetscErrorCode PetscSectionFieldGetTensorDegree_Private(PetscSection section,PetscInt field,PetscInt line,PetscBool vertexchart,PetscInt *Nc,PetscInt *k)
{
  PetscFunctionBeginHot;
  CHKERRQ(PetscSectionGetFieldComponents(section, field, Nc));
  if (line < 0) {
    *k = 0;
    *Nc = 0;
  } else if (vertexchart) {            /* If we only have a vertex chart, we must have degree k=1 */
    *k = 1;
  } else {                      /* Assume the full interpolated mesh is in the chart; lines in particular */
    /* An order k SEM disc has k-1 dofs on an edge */
    CHKERRQ(PetscSectionGetFieldDof(section, line, field, k));
    *k = *k / *Nc + 1;
  }
  PetscFunctionReturn(0);
}

/*@

  DMPlexSetClosurePermutationTensor - Create a permutation from the default (BFS) point ordering in the closure, to a
  lexicographic ordering over the tensor product cell (i.e., line, quad, hex, etc.), and set this permutation in the
  section provided (or the section of the DM).

  Input Parameters:
+ dm      - The DM
. point   - Either a cell (highest dim point) or an edge (dim 1 point), or PETSC_DETERMINE
- section - The PetscSection to reorder, or NULL for the default section

  Note: The point is used to determine the number of dofs/field on an edge. For SEM, this is related to the polynomial
  degree of the basis.

  Example:
  A typical interpolated single-quad mesh might order points as
.vb
  [c0, v1, v2, v3, v4, e5, e6, e7, e8]

  v4 -- e6 -- v3
  |           |
  e7    c0    e8
  |           |
  v1 -- e5 -- v2
.ve

  (There is no significance to the ordering described here.)  The default section for a Q3 quad might typically assign
  dofs in the order of points, e.g.,
.vb
    c0 -> [0,1,2,3]
    v1 -> [4]
    ...
    e5 -> [8, 9]
.ve

  which corresponds to the dofs
.vb
    6   10  11  7
    13  2   3   15
    12  0   1   14
    4   8   9   5
.ve

  The closure in BFS ordering works through height strata (cells, edges, vertices) to produce the ordering
.vb
  0 1 2 3 8 9 14 15 11 10 13 12 4 5 7 6
.ve

  After calling DMPlexSetClosurePermutationTensor(), the closure will be ordered lexicographically,
.vb
   4 8 9 5 12 0 1 14 13 2 3 15 6 10 11 7
.ve

  Level: developer

.seealso: DMGetLocalSection(), PetscSectionSetClosurePermutation(), DMSetGlobalSection()
@*/
PetscErrorCode DMPlexSetClosurePermutationTensor(DM dm, PetscInt point, PetscSection section)
{
  DMLabel        label;
  PetscInt       dim, depth = -1, eStart = -1, Nf;
  PetscBool      vertexchart;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  if (dim < 1) PetscFunctionReturn(0);
  if (point < 0) {
    PetscInt sStart,sEnd;

    CHKERRQ(DMPlexGetDepthStratum(dm, 1, &sStart, &sEnd));
    point = sEnd-sStart ? sStart : point;
  }
  CHKERRQ(DMPlexGetDepthLabel(dm, &label));
  if (point >= 0) CHKERRQ(DMLabelGetValue(label, point, &depth));
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  if (depth == 1) {eStart = point;}
  else if  (depth == dim) {
    const PetscInt *cone;

    CHKERRQ(DMPlexGetCone(dm, point, &cone));
    if (dim == 2) eStart = cone[0];
    else if (dim == 3) {
      const PetscInt *cone2;
      CHKERRQ(DMPlexGetCone(dm, cone[0], &cone2));
      eStart = cone2[0];
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D of depth %D cannot be used to bootstrap spectral ordering for dim %D", point, depth, dim);
  } else PetscCheckFalse(depth >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D of depth %D cannot be used to bootstrap spectral ordering for dim %D", point, depth, dim);
  {                             /* Determine whether the chart covers all points or just vertices. */
    PetscInt pStart,pEnd,cStart,cEnd;
    CHKERRQ(DMPlexGetDepthStratum(dm,0,&pStart,&pEnd));
    CHKERRQ(PetscSectionGetChart(section,&cStart,&cEnd));
    if (pStart == cStart && pEnd == cEnd) vertexchart = PETSC_TRUE;      /* Only vertices are in the chart */
    else if (cStart <= point && point < cEnd) vertexchart = PETSC_FALSE; /* Some interpolated points exist in the chart */
    else vertexchart = PETSC_TRUE;                                       /* Some interpolated points are not in chart; assume dofs only at cells and vertices */
  }
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  for (PetscInt d=1; d<=dim; d++) {
    PetscInt k, f, Nc, c, i, j, size = 0, offset = 0, foffset = 0;
    PetscInt *perm;

    for (f = 0; f < Nf; ++f) {
      CHKERRQ(PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k));
      size += PetscPowInt(k+1, d)*Nc;
    }
    CHKERRQ(PetscMalloc1(size, &perm));
    for (f = 0; f < Nf; ++f) {
      switch (d) {
      case 1:
        CHKERRQ(PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k));
        /*
         Original ordering is [ edge of length k-1; vtx0; vtx1 ]
         We want              [ vtx0; edge of length k-1; vtx1 ]
         */
        for (c=0; c<Nc; c++,offset++) perm[offset] = (k-1)*Nc + c + foffset;
        for (i=0; i<k-1; i++) for (c=0; c<Nc; c++,offset++) perm[offset] = i*Nc + c + foffset;
        for (c=0; c<Nc; c++,offset++) perm[offset] = k*Nc + c + foffset;
        foffset = offset;
        break;
      case 2:
        /* The original quad closure is oriented clockwise, {f, e_b, e_r, e_t, e_l, v_lb, v_rb, v_tr, v_tl} */
        CHKERRQ(PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k));
        /* The SEM order is

         v_lb, {e_b}, v_rb,
         e^{(k-1)-i}_l, {f^{i*(k-1)}}, e^i_r,
         v_lt, reverse {e_t}, v_rt
         */
        {
          const PetscInt of   = 0;
          const PetscInt oeb  = of   + PetscSqr(k-1);
          const PetscInt oer  = oeb  + (k-1);
          const PetscInt oet  = oer  + (k-1);
          const PetscInt oel  = oet  + (k-1);
          const PetscInt ovlb = oel  + (k-1);
          const PetscInt ovrb = ovlb + 1;
          const PetscInt ovrt = ovrb + 1;
          const PetscInt ovlt = ovrt + 1;
          PetscInt       o;

          /* bottom */
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovlb*Nc + c + foffset;
          for (o = oeb; o < oer; ++o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovrb*Nc + c + foffset;
          /* middle */
          for (i = 0; i < k-1; ++i) {
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oel+(k-2)-i)*Nc + c + foffset;
            for (o = of+(k-1)*i; o < of+(k-1)*(i+1); ++o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oer+i)*Nc + c + foffset;
          }
          /* top */
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovlt*Nc + c + foffset;
          for (o = oel-1; o >= oet; --o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovrt*Nc + c + foffset;
          foffset = offset;
        }
        break;
      case 3:
        /* The original hex closure is

         {c,
         f_b, f_t, f_f, f_b, f_r, f_l,
         e_bl, e_bb, e_br, e_bf,  e_tf, e_tr, e_tb, e_tl,  e_rf, e_lf, e_lb, e_rb,
         v_blf, v_blb, v_brb, v_brf, v_tlf, v_trf, v_trb, v_tlb}
         */
        CHKERRQ(PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k));
        /* The SEM order is
         Bottom Slice
         v_blf, {e^{(k-1)-n}_bf}, v_brf,
         e^{i}_bl, f^{n*(k-1)+(k-1)-i}_b, e^{(k-1)-i}_br,
         v_blb, {e_bb}, v_brb,

         Middle Slice (j)
         {e^{(k-1)-j}_lf}, {f^{j*(k-1)+n}_f}, e^j_rf,
         f^{i*(k-1)+j}_l, {c^{(j*(k-1) + i)*(k-1)+n}_t}, f^{j*(k-1)+i}_r,
         e^j_lb, {f^{j*(k-1)+(k-1)-n}_b}, e^{(k-1)-j}_rb,

         Top Slice
         v_tlf, {e_tf}, v_trf,
         e^{(k-1)-i}_tl, {f^{i*(k-1)}_t}, e^{i}_tr,
         v_tlb, {e^{(k-1)-n}_tb}, v_trb,
         */
        {
          const PetscInt oc    = 0;
          const PetscInt ofb   = oc    + PetscSqr(k-1)*(k-1);
          const PetscInt oft   = ofb   + PetscSqr(k-1);
          const PetscInt off   = oft   + PetscSqr(k-1);
          const PetscInt ofk   = off   + PetscSqr(k-1);
          const PetscInt ofr   = ofk   + PetscSqr(k-1);
          const PetscInt ofl   = ofr   + PetscSqr(k-1);
          const PetscInt oebl  = ofl   + PetscSqr(k-1);
          const PetscInt oebb  = oebl  + (k-1);
          const PetscInt oebr  = oebb  + (k-1);
          const PetscInt oebf  = oebr  + (k-1);
          const PetscInt oetf  = oebf  + (k-1);
          const PetscInt oetr  = oetf  + (k-1);
          const PetscInt oetb  = oetr  + (k-1);
          const PetscInt oetl  = oetb  + (k-1);
          const PetscInt oerf  = oetl  + (k-1);
          const PetscInt oelf  = oerf  + (k-1);
          const PetscInt oelb  = oelf  + (k-1);
          const PetscInt oerb  = oelb  + (k-1);
          const PetscInt ovblf = oerb  + (k-1);
          const PetscInt ovblb = ovblf + 1;
          const PetscInt ovbrb = ovblb + 1;
          const PetscInt ovbrf = ovbrb + 1;
          const PetscInt ovtlf = ovbrf + 1;
          const PetscInt ovtrf = ovtlf + 1;
          const PetscInt ovtrb = ovtrf + 1;
          const PetscInt ovtlb = ovtrb + 1;
          PetscInt       o, n;

          /* Bottom Slice */
          /*   bottom */
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovblf*Nc + c + foffset;
          for (o = oetf-1; o >= oebf; --o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovbrf*Nc + c + foffset;
          /*   middle */
          for (i = 0; i < k-1; ++i) {
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oebl+i)*Nc + c + foffset;
            for (n = 0; n < k-1; ++n) {o = ofb+n*(k-1)+i; for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;}
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oebr+(k-2)-i)*Nc + c + foffset;
          }
          /*   top */
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovblb*Nc + c + foffset;
          for (o = oebb; o < oebr; ++o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovbrb*Nc + c + foffset;

          /* Middle Slice */
          for (j = 0; j < k-1; ++j) {
            /*   bottom */
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oelf+(k-2)-j)*Nc + c + foffset;
            for (o = off+j*(k-1); o < off+(j+1)*(k-1); ++o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oerf+j)*Nc + c + foffset;
            /*   middle */
            for (i = 0; i < k-1; ++i) {
              for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (ofl+i*(k-1)+j)*Nc + c + foffset;
              for (n = 0; n < k-1; ++n) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oc+(j*(k-1)+i)*(k-1)+n)*Nc + c + foffset;
              for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (ofr+j*(k-1)+i)*Nc + c + foffset;
            }
            /*   top */
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oelb+j)*Nc + c + foffset;
            for (o = ofk+j*(k-1)+(k-2); o >= ofk+j*(k-1); --o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oerb+(k-2)-j)*Nc + c + foffset;
          }

          /* Top Slice */
          /*   bottom */
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovtlf*Nc + c + foffset;
          for (o = oetf; o < oetr; ++o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovtrf*Nc + c + foffset;
          /*   middle */
          for (i = 0; i < k-1; ++i) {
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oetl+(k-2)-i)*Nc + c + foffset;
            for (n = 0; n < k-1; ++n) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oft+i*(k-1)+n)*Nc + c + foffset;
            for (c = 0; c < Nc; ++c, ++offset) perm[offset] = (oetr+i)*Nc + c + foffset;
          }
          /*   top */
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovtlb*Nc + c + foffset;
          for (o = oetl-1; o >= oetb; --o) for (c = 0; c < Nc; ++c, ++offset) perm[offset] = o*Nc + c + foffset;
          for (c = 0; c < Nc; ++c, ++offset) perm[offset] = ovtrb*Nc + c + foffset;

          foffset = offset;
        }
        break;
      default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No spectral ordering for dimension %D", d);
      }
    }
    PetscCheckFalse(offset != size,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Number of permutation entries %D != %D", offset, size);
    /* Check permutation */
    {
      PetscInt *check;

      CHKERRQ(PetscMalloc1(size, &check));
      for (i = 0; i < size; ++i) {check[i] = -1; PetscCheckFalse(perm[i] < 0 || perm[i] >= size,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid permutation index p[%D] = %D", i, perm[i]);}
      for (i = 0; i < size; ++i) check[perm[i]] = i;
      for (i = 0; i < size; ++i) {PetscCheckFalse(check[i] < 0,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Missing permutation index %D", i);}
      CHKERRQ(PetscFree(check));
    }
    CHKERRQ(PetscSectionSetClosurePermutation_Internal(section, (PetscObject) dm, d, size, PETSC_OWN_POINTER, perm));
    if (d == dim) { // Add permutation for localized (in case this is a coordinate DM)
      PetscInt *loc_perm;
      CHKERRQ(PetscMalloc1(size*2, &loc_perm));
      for (PetscInt i=0; i<size; i++) {
        loc_perm[i] = perm[i];
        loc_perm[size+i] = size + perm[i];
      }
      CHKERRQ(PetscSectionSetClosurePermutation_Internal(section, (PetscObject) dm, d, size*2, PETSC_OWN_POINTER, loc_perm));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetPointDualSpaceFEM(DM dm, PetscInt point, PetscInt field, PetscDualSpace *dspace)
{
  PetscDS        prob;
  PetscInt       depth, Nf, h;
  DMLabel        label;

  PetscFunctionBeginHot;
  CHKERRQ(DMGetDS(dm, &prob));
  Nf      = prob->Nf;
  label   = dm->depthLabel;
  *dspace = NULL;
  if (field < Nf) {
    PetscObject disc = prob->disc[field];

    if (disc->classid == PETSCFE_CLASSID) {
      PetscDualSpace dsp;

      CHKERRQ(PetscFEGetDualSpace((PetscFE)disc,&dsp));
      CHKERRQ(DMLabelGetNumValues(label,&depth));
      CHKERRQ(DMLabelGetValue(label,point,&h));
      h    = depth - 1 - h;
      if (h) {
        CHKERRQ(PetscDualSpaceGetHeightSubspace(dsp,h,dspace));
      } else {
        *dspace = dsp;
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMPlexVecGetClosure_Depth1_Static(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, PetscScalar *values[])
{
  PetscScalar    *array, *vArray;
  const PetscInt *cone, *coneO;
  PetscInt        pStart, pEnd, p, numPoints, size = 0, offset = 0;

  PetscFunctionBeginHot;
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  CHKERRQ(DMPlexGetConeSize(dm, point, &numPoints));
  CHKERRQ(DMPlexGetCone(dm, point, &cone));
  CHKERRQ(DMPlexGetConeOrientation(dm, point, &coneO));
  if (!values || !*values) {
    if ((point >= pStart) && (point < pEnd)) {
      PetscInt dof;

      CHKERRQ(PetscSectionGetDof(section, point, &dof));
      size += dof;
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt cp = cone[p];
      PetscInt       dof;

      if ((cp < pStart) || (cp >= pEnd)) continue;
      CHKERRQ(PetscSectionGetDof(section, cp, &dof));
      size += dof;
    }
    if (!values) {
      if (csize) *csize = size;
      PetscFunctionReturn(0);
    }
    CHKERRQ(DMGetWorkArray(dm, size, MPIU_SCALAR, &array));
  } else {
    array = *values;
  }
  size = 0;
  CHKERRQ(VecGetArray(v, &vArray));
  if ((point >= pStart) && (point < pEnd)) {
    PetscInt     dof, off, d;
    PetscScalar *varr;

    CHKERRQ(PetscSectionGetDof(section, point, &dof));
    CHKERRQ(PetscSectionGetOffset(section, point, &off));
    varr = &vArray[off];
    for (d = 0; d < dof; ++d, ++offset) {
      array[offset] = varr[d];
    }
    size += dof;
  }
  for (p = 0; p < numPoints; ++p) {
    const PetscInt cp = cone[p];
    PetscInt       o  = coneO[p];
    PetscInt       dof, off, d;
    PetscScalar   *varr;

    if ((cp < pStart) || (cp >= pEnd)) continue;
    CHKERRQ(PetscSectionGetDof(section, cp, &dof));
    CHKERRQ(PetscSectionGetOffset(section, cp, &off));
    varr = &vArray[off];
    if (o >= 0) {
      for (d = 0; d < dof; ++d, ++offset) {
        array[offset] = varr[d];
      }
    } else {
      for (d = dof-1; d >= 0; --d, ++offset) {
        array[offset] = varr[d];
      }
    }
    size += dof;
  }
  CHKERRQ(VecRestoreArray(v, &vArray));
  if (!*values) {
    if (csize) *csize = size;
    *values = array;
  } else {
    PetscCheckFalse(size > *csize,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Size of input array %D < actual size %D", *csize, size);
    *csize = size;
  }
  PetscFunctionReturn(0);
}

/* Compress out points not in the section */
static inline PetscErrorCode CompressPoints_Private(PetscSection section, PetscInt *numPoints, PetscInt points[])
{
  const PetscInt np = *numPoints;
  PetscInt       pStart, pEnd, p, q;

  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  for (p = 0, q = 0; p < np; ++p) {
    const PetscInt r = points[p*2];
    if ((r >= pStart) && (r < pEnd)) {
      points[q*2]   = r;
      points[q*2+1] = points[p*2+1];
      ++q;
    }
  }
  *numPoints = q;
  return 0;
}

/* Compressed closure does not apply closure permutation */
PetscErrorCode DMPlexGetCompressedClosure(DM dm, PetscSection section, PetscInt point, PetscInt *numPoints, PetscInt **points, PetscSection *clSec, IS *clPoints, const PetscInt **clp)
{
  const PetscInt *cla = NULL;
  PetscInt       np, *pts = NULL;

  PetscFunctionBeginHot;
  CHKERRQ(PetscSectionGetClosureIndex(section, (PetscObject) dm, clSec, clPoints));
  if (*clPoints) {
    PetscInt dof, off;

    CHKERRQ(PetscSectionGetDof(*clSec, point, &dof));
    CHKERRQ(PetscSectionGetOffset(*clSec, point, &off));
    CHKERRQ(ISGetIndices(*clPoints, &cla));
    np   = dof/2;
    pts  = (PetscInt *) &cla[off];
  } else {
    CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &np, &pts));
    CHKERRQ(CompressPoints_Private(section, &np, pts));
  }
  *numPoints = np;
  *points    = pts;
  *clp       = cla;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexRestoreCompressedClosure(DM dm, PetscSection section, PetscInt point, PetscInt *numPoints, PetscInt **points, PetscSection *clSec, IS *clPoints, const PetscInt **clp)
{
  PetscFunctionBeginHot;
  if (!*clPoints) {
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, numPoints, points));
  } else {
    CHKERRQ(ISRestoreIndices(*clPoints, clp));
  }
  *numPoints = 0;
  *points    = NULL;
  *clSec     = NULL;
  *clPoints  = NULL;
  *clp       = NULL;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMPlexVecGetClosure_Static(DM dm, PetscSection section, PetscInt numPoints, const PetscInt points[], const PetscInt clperm[], const PetscScalar vArray[], PetscInt *size, PetscScalar array[])
{
  PetscInt          offset = 0, p;
  const PetscInt    **perms = NULL;
  const PetscScalar **flips = NULL;

  PetscFunctionBeginHot;
  *size = 0;
  CHKERRQ(PetscSectionGetPointSyms(section,numPoints,points,&perms,&flips));
  for (p = 0; p < numPoints; p++) {
    const PetscInt    point = points[2*p];
    const PetscInt    *perm = perms ? perms[p] : NULL;
    const PetscScalar *flip = flips ? flips[p] : NULL;
    PetscInt          dof, off, d;
    const PetscScalar *varr;

    CHKERRQ(PetscSectionGetDof(section, point, &dof));
    CHKERRQ(PetscSectionGetOffset(section, point, &off));
    varr = &vArray[off];
    if (clperm) {
      if (perm) {
        for (d = 0; d < dof; d++) array[clperm[offset + perm[d]]]  = varr[d];
      } else {
        for (d = 0; d < dof; d++) array[clperm[offset +      d ]]  = varr[d];
      }
      if (flip) {
        for (d = 0; d < dof; d++) array[clperm[offset +      d ]] *= flip[d];
      }
    } else {
      if (perm) {
        for (d = 0; d < dof; d++) array[offset + perm[d]]  = varr[d];
      } else {
        for (d = 0; d < dof; d++) array[offset +      d ]  = varr[d];
      }
      if (flip) {
        for (d = 0; d < dof; d++) array[offset +      d ] *= flip[d];
      }
    }
    offset += dof;
  }
  CHKERRQ(PetscSectionRestorePointSyms(section,numPoints,points,&perms,&flips));
  *size = offset;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMPlexVecGetClosure_Fields_Static(DM dm, PetscSection section, PetscInt numPoints, const PetscInt points[], PetscInt numFields, const PetscInt clperm[], const PetscScalar vArray[], PetscInt *size, PetscScalar array[])
{
  PetscInt          offset = 0, f;

  PetscFunctionBeginHot;
  *size = 0;
  for (f = 0; f < numFields; ++f) {
    PetscInt          p;
    const PetscInt    **perms = NULL;
    const PetscScalar **flips = NULL;

    CHKERRQ(PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms,&flips));
    for (p = 0; p < numPoints; p++) {
      const PetscInt    point = points[2*p];
      PetscInt          fdof, foff, b;
      const PetscScalar *varr;
      const PetscInt    *perm = perms ? perms[p] : NULL;
      const PetscScalar *flip = flips ? flips[p] : NULL;

      CHKERRQ(PetscSectionGetFieldDof(section, point, f, &fdof));
      CHKERRQ(PetscSectionGetFieldOffset(section, point, f, &foff));
      varr = &vArray[foff];
      if (clperm) {
        if (perm) {for (b = 0; b < fdof; b++) {array[clperm[offset + perm[b]]]  = varr[b];}}
        else      {for (b = 0; b < fdof; b++) {array[clperm[offset +      b ]]  = varr[b];}}
        if (flip) {for (b = 0; b < fdof; b++) {array[clperm[offset +      b ]] *= flip[b];}}
      } else {
        if (perm) {for (b = 0; b < fdof; b++) {array[offset + perm[b]]  = varr[b];}}
        else      {for (b = 0; b < fdof; b++) {array[offset +      b ]  = varr[b];}}
        if (flip) {for (b = 0; b < fdof; b++) {array[offset +      b ] *= flip[b];}}
      }
      offset += fdof;
    }
    CHKERRQ(PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms,&flips));
  }
  *size = offset;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexVecGetClosure - Get an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. v - The local vector
- point - The point in the DM

  Input/Output Parameters:
+ csize  - The size of the input values array, or NULL; on output the number of values in the closure
- values - An array to use for the values, or NULL to have it allocated automatically;
           if the user provided NULL, it is a borrowed array and should not be freed

$ Note that DMPlexVecGetClosure/DMPlexVecRestoreClosure only allocates the values array if it set to NULL in the
$ calling function. This is because DMPlexVecGetClosure() is typically called in the inner loop of a Vec or Mat
$ assembly function, and a user may already have allocated storage for this operation.
$
$ A typical use could be
$
$  values = NULL;
$  CHKERRQ(DMPlexVecGetClosure(dm, NULL, v, p, &clSize, &values));
$  for (cl = 0; cl < clSize; ++cl) {
$    <Compute on closure>
$  }
$  CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, v, p, &clSize, &values));
$
$ or
$
$  PetscMalloc1(clMaxSize, &values);
$  for (p = pStart; p < pEnd; ++p) {
$    clSize = clMaxSize;
$    CHKERRQ(DMPlexVecGetClosure(dm, NULL, v, p, &clSize, &values));
$    for (cl = 0; cl < clSize; ++cl) {
$      <Compute on closure>
$    }
$  }
$  PetscFree(values);

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The csize argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.seealso DMPlexVecRestoreClosure(), DMPlexVecSetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexVecGetClosure(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, PetscScalar *values[])
{
  PetscSection       clSection;
  IS                 clPoints;
  PetscInt          *points = NULL;
  const PetscInt    *clp, *perm;
  PetscInt           depth, numFields, numPoints, asize;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  if (depth == 1 && numFields < 2) {
    CHKERRQ(DMPlexVecGetClosure_Depth1_Static(dm, section, v, point, csize, values));
    PetscFunctionReturn(0);
  }
  /* Get points */
  CHKERRQ(DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  /* Get sizes */
  asize = 0;
  for (PetscInt p = 0; p < numPoints*2; p += 2) {
    PetscInt dof;
    CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
    asize += dof;
  }
  if (values) {
    const PetscScalar *vArray;
    PetscInt          size;

    if (*values) {
      PetscCheckFalse(*csize < asize,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Provided array size %D not sufficient to hold closure size %D", *csize, asize);
    } else CHKERRQ(DMGetWorkArray(dm, asize, MPIU_SCALAR, values));
    CHKERRQ(PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, depth, asize, &perm));
    CHKERRQ(VecGetArrayRead(v, &vArray));
    /* Get values */
    if (numFields > 0) CHKERRQ(DMPlexVecGetClosure_Fields_Static(dm, section, numPoints, points, numFields, perm, vArray, &size, *values));
    else               CHKERRQ(DMPlexVecGetClosure_Static(dm, section, numPoints, points, perm, vArray, &size, *values));
    PetscCheckFalse(asize != size,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Section size %D does not match Vec closure size %D", asize, size);
    /* Cleanup array */
    CHKERRQ(VecRestoreArrayRead(v, &vArray));
  }
  if (csize) *csize = asize;
  /* Cleanup points */
  CHKERRQ(DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexVecGetClosureAtDepth_Internal(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt depth, PetscInt *csize, PetscScalar *values[])
{
  DMLabel            depthLabel;
  PetscSection       clSection;
  IS                 clPoints;
  PetscScalar       *array;
  const PetscScalar *vArray;
  PetscInt          *points = NULL;
  const PetscInt    *clp, *perm = NULL;
  PetscInt           mdepth, numFields, numPoints, Np = 0, p, clsize, size;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  CHKERRQ(DMPlexGetDepth(dm, &mdepth));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  if (mdepth == 1 && numFields < 2) {
    CHKERRQ(DMPlexVecGetClosure_Depth1_Static(dm, section, v, point, csize, values));
    PetscFunctionReturn(0);
  }
  /* Get points */
  CHKERRQ(DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  for (clsize=0,p=0; p<Np; p++) {
    PetscInt dof;
    CHKERRQ(PetscSectionGetDof(section, points[2*p], &dof));
    clsize += dof;
  }
  CHKERRQ(PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, depth, clsize, &perm));
  /* Filter points */
  for (p = 0; p < numPoints*2; p += 2) {
    PetscInt dep;

    CHKERRQ(DMLabelGetValue(depthLabel, points[p], &dep));
    if (dep != depth) continue;
    points[Np*2+0] = points[p];
    points[Np*2+1] = points[p+1];
    ++Np;
  }
  /* Get array */
  if (!values || !*values) {
    PetscInt asize = 0, dof;

    for (p = 0; p < Np*2; p += 2) {
      CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
      asize += dof;
    }
    if (!values) {
      CHKERRQ(DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
      if (csize) *csize = asize;
      PetscFunctionReturn(0);
    }
    CHKERRQ(DMGetWorkArray(dm, asize, MPIU_SCALAR, &array));
  } else {
    array = *values;
  }
  CHKERRQ(VecGetArrayRead(v, &vArray));
  /* Get values */
  if (numFields > 0) CHKERRQ(DMPlexVecGetClosure_Fields_Static(dm, section, Np, points, numFields, perm, vArray, &size, array));
  else               CHKERRQ(DMPlexVecGetClosure_Static(dm, section, Np, points, perm, vArray, &size, array));
  /* Cleanup points */
  CHKERRQ(DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  /* Cleanup array */
  CHKERRQ(VecRestoreArrayRead(v, &vArray));
  if (!*values) {
    if (csize) *csize = size;
    *values = array;
  } else {
    PetscCheckFalse(size > *csize,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Size of input array %D < actual size %D", *csize, size);
    *csize = size;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexVecRestoreClosure - Restore the array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. v - The local vector
. point - The point in the DM
. csize - The number of values in the closure, or NULL
- values - The array of values, which is a borrowed array and should not be freed

  Note that the array values are discarded and not copied back into v. In order to copy values back to v, use DMPlexVecSetClosure()

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The csize argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexVecSetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexVecRestoreClosure(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, PetscScalar *values[])
{
  PetscInt       size = 0;

  PetscFunctionBegin;
  /* Should work without recalculating size */
  CHKERRQ(DMRestoreWorkArray(dm, size, MPIU_SCALAR, (void*) values));
  *values = NULL;
  PetscFunctionReturn(0);
}

static inline void add   (PetscScalar *x, PetscScalar y) {*x += y;}
static inline void insert(PetscScalar *x, PetscScalar y) {*x  = y;}

static inline PetscErrorCode updatePoint_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar*, PetscScalar), PetscBool setBC, const PetscInt perm[], const PetscScalar flip[], const PetscInt clperm[], const PetscScalar values[], PetscInt offset, PetscScalar array[])
{
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar    *a;
  PetscInt        off, cind = 0, k;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetConstraintDof(section, point, &cdof));
  CHKERRQ(PetscSectionGetOffset(section, point, &off));
  a    = &array[off];
  if (!cdof || setBC) {
    if (clperm) {
      if (perm) {for (k = 0; k < dof; ++k) {fuse(&a[k], values[clperm[offset+perm[k]]] * (flip ? flip[perm[k]] : 1.));}}
      else      {for (k = 0; k < dof; ++k) {fuse(&a[k], values[clperm[offset+     k ]] * (flip ? flip[     k ] : 1.));}}
    } else {
      if (perm) {for (k = 0; k < dof; ++k) {fuse(&a[k], values[offset+perm[k]] * (flip ? flip[perm[k]] : 1.));}}
      else      {for (k = 0; k < dof; ++k) {fuse(&a[k], values[offset+     k ] * (flip ? flip[     k ] : 1.));}}
    }
  } else {
    CHKERRQ(PetscSectionGetConstraintIndices(section, point, &cdofs));
    if (clperm) {
      if (perm) {for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
          fuse(&a[k], values[clperm[offset+perm[k]]] * (flip ? flip[perm[k]] : 1.));
        }
      } else {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
          fuse(&a[k], values[clperm[offset+     k ]] * (flip ? flip[     k ] : 1.));
        }
      }
    } else {
      if (perm) {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
          fuse(&a[k], values[offset+perm[k]] * (flip ? flip[perm[k]] : 1.));
        }
      } else {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
          fuse(&a[k], values[offset+     k ] * (flip ? flip[     k ] : 1.));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode updatePointBC_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar*, PetscScalar), const PetscInt perm[], const PetscScalar flip[], const PetscInt clperm[], const PetscScalar values[], PetscInt offset, PetscScalar array[])
{
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar    *a;
  PetscInt        off, cind = 0, k;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetConstraintDof(section, point, &cdof));
  CHKERRQ(PetscSectionGetOffset(section, point, &off));
  a    = &array[off];
  if (cdof) {
    CHKERRQ(PetscSectionGetConstraintIndices(section, point, &cdofs));
    if (clperm) {
      if (perm) {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {
            fuse(&a[k], values[clperm[offset+perm[k]]] * (flip ? flip[perm[k]] : 1.));
            cind++;
          }
        }
      } else {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {
            fuse(&a[k], values[clperm[offset+     k ]] * (flip ? flip[     k ] : 1.));
            cind++;
          }
        }
      }
    } else {
      if (perm) {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {
            fuse(&a[k], values[offset+perm[k]] * (flip ? flip[perm[k]] : 1.));
            cind++;
          }
        }
      } else {
        for (k = 0; k < dof; ++k) {
          if ((cind < cdof) && (k == cdofs[cind])) {
            fuse(&a[k], values[offset+     k ] * (flip ? flip[     k ] : 1.));
            cind++;
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode updatePointFields_private(PetscSection section, PetscInt point, const PetscInt *perm, const PetscScalar *flip, PetscInt f, void (*fuse)(PetscScalar*, PetscScalar), PetscBool setBC, const PetscInt clperm[], const PetscScalar values[], PetscInt *offset, PetscScalar array[])
{
  PetscScalar    *a;
  PetscInt        fdof, foff, fcdof, foffset = *offset;
  const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
  PetscInt        cind = 0, b;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetFieldDof(section, point, f, &fdof));
  CHKERRQ(PetscSectionGetFieldConstraintDof(section, point, f, &fcdof));
  CHKERRQ(PetscSectionGetFieldOffset(section, point, f, &foff));
  a    = &array[foff];
  if (!fcdof || setBC) {
    if (clperm) {
      if (perm) {for (b = 0; b < fdof; b++) {fuse(&a[b], values[clperm[foffset+perm[b]]] * (flip ? flip[perm[b]] : 1.));}}
      else      {for (b = 0; b < fdof; b++) {fuse(&a[b], values[clperm[foffset+     b ]] * (flip ? flip[     b ] : 1.));}}
    } else {
      if (perm) {for (b = 0; b < fdof; b++) {fuse(&a[b], values[foffset+perm[b]] * (flip ? flip[perm[b]] : 1.));}}
      else      {for (b = 0; b < fdof; b++) {fuse(&a[b], values[foffset+     b ] * (flip ? flip[     b ] : 1.));}}
    }
  } else {
    CHKERRQ(PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs));
    if (clperm) {
      if (perm) {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {++cind; continue;}
          fuse(&a[b], values[clperm[foffset+perm[b]]] * (flip ? flip[perm[b]] : 1.));
        }
      } else {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {++cind; continue;}
          fuse(&a[b], values[clperm[foffset+     b ]] * (flip ? flip[     b ] : 1.));
        }
      }
    } else {
      if (perm) {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {++cind; continue;}
          fuse(&a[b], values[foffset+perm[b]] * (flip ? flip[perm[b]] : 1.));
        }
      } else {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {++cind; continue;}
          fuse(&a[b], values[foffset+     b ] * (flip ? flip[     b ] : 1.));
        }
      }
    }
  }
  *offset += fdof;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode updatePointFieldsBC_private(PetscSection section, PetscInt point, const PetscInt perm[], const PetscScalar flip[], PetscInt f, PetscInt Ncc, const PetscInt comps[], void (*fuse)(PetscScalar*, PetscScalar), const PetscInt clperm[], const PetscScalar values[], PetscInt *offset, PetscScalar array[])
{
  PetscScalar    *a;
  PetscInt        fdof, foff, fcdof, foffset = *offset;
  const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
  PetscInt        Nc, cind = 0, ncind = 0, b;
  PetscBool       ncSet, fcSet;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetFieldComponents(section, f, &Nc));
  CHKERRQ(PetscSectionGetFieldDof(section, point, f, &fdof));
  CHKERRQ(PetscSectionGetFieldConstraintDof(section, point, f, &fcdof));
  CHKERRQ(PetscSectionGetFieldOffset(section, point, f, &foff));
  a    = &array[foff];
  if (fcdof) {
    /* We just override fcdof and fcdofs with Ncc and comps */
    CHKERRQ(PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs));
    if (clperm) {
      if (perm) {
        if (comps) {
          for (b = 0; b < fdof; b++) {
            ncSet = fcSet = PETSC_FALSE;
            if (b%Nc == comps[ncind]) {ncind = (ncind+1)%Ncc; ncSet = PETSC_TRUE;}
            if ((cind < fcdof) && (b == fcdofs[cind])) {++cind;  fcSet = PETSC_TRUE;}
            if (ncSet && fcSet) {fuse(&a[b], values[clperm[foffset+perm[b]]] * (flip ? flip[perm[b]] : 1.));}
          }
        } else {
          for (b = 0; b < fdof; b++) {
            if ((cind < fcdof) && (b == fcdofs[cind])) {
              fuse(&a[b], values[clperm[foffset+perm[b]]] * (flip ? flip[perm[b]] : 1.));
              ++cind;
            }
          }
        }
      } else {
        if (comps) {
          for (b = 0; b < fdof; b++) {
            ncSet = fcSet = PETSC_FALSE;
            if (b%Nc == comps[ncind]) {ncind = (ncind+1)%Ncc; ncSet = PETSC_TRUE;}
            if ((cind < fcdof) && (b == fcdofs[cind])) {++cind;  fcSet = PETSC_TRUE;}
            if (ncSet && fcSet) {fuse(&a[b], values[clperm[foffset+     b ]] * (flip ? flip[     b ] : 1.));}
          }
        } else {
          for (b = 0; b < fdof; b++) {
            if ((cind < fcdof) && (b == fcdofs[cind])) {
              fuse(&a[b], values[clperm[foffset+     b ]] * (flip ? flip[     b ] : 1.));
              ++cind;
            }
          }
        }
      }
    } else {
      if (perm) {
        if (comps) {
          for (b = 0; b < fdof; b++) {
            ncSet = fcSet = PETSC_FALSE;
            if (b%Nc == comps[ncind]) {ncind = (ncind+1)%Ncc; ncSet = PETSC_TRUE;}
            if ((cind < fcdof) && (b == fcdofs[cind])) {++cind;  fcSet = PETSC_TRUE;}
            if (ncSet && fcSet) {fuse(&a[b], values[foffset+perm[b]] * (flip ? flip[perm[b]] : 1.));}
          }
        } else {
          for (b = 0; b < fdof; b++) {
            if ((cind < fcdof) && (b == fcdofs[cind])) {
              fuse(&a[b], values[foffset+perm[b]] * (flip ? flip[perm[b]] : 1.));
              ++cind;
            }
          }
        }
      } else {
        if (comps) {
          for (b = 0; b < fdof; b++) {
            ncSet = fcSet = PETSC_FALSE;
            if (b%Nc == comps[ncind]) {ncind = (ncind+1)%Ncc; ncSet = PETSC_TRUE;}
            if ((cind < fcdof) && (b == fcdofs[cind])) {++cind;  fcSet = PETSC_TRUE;}
            if (ncSet && fcSet) {fuse(&a[b], values[foffset+     b ] * (flip ? flip[     b ] : 1.));}
          }
        } else {
          for (b = 0; b < fdof; b++) {
            if ((cind < fcdof) && (b == fcdofs[cind])) {
              fuse(&a[b], values[foffset+     b ] * (flip ? flip[     b ] : 1.));
              ++cind;
            }
          }
        }
      }
    }
  }
  *offset += fdof;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMPlexVecSetClosure_Depth1_Static(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  PetscScalar    *array;
  const PetscInt *cone, *coneO;
  PetscInt        pStart, pEnd, p, numPoints, off, dof;

  PetscFunctionBeginHot;
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  CHKERRQ(DMPlexGetConeSize(dm, point, &numPoints));
  CHKERRQ(DMPlexGetCone(dm, point, &cone));
  CHKERRQ(DMPlexGetConeOrientation(dm, point, &coneO));
  CHKERRQ(VecGetArray(v, &array));
  for (p = 0, off = 0; p <= numPoints; ++p, off += dof) {
    const PetscInt cp = !p ? point : cone[p-1];
    const PetscInt o  = !p ? 0     : coneO[p-1];

    if ((cp < pStart) || (cp >= pEnd)) {dof = 0; continue;}
    CHKERRQ(PetscSectionGetDof(section, cp, &dof));
    /* ADD_VALUES */
    {
      const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
      PetscScalar    *a;
      PetscInt        cdof, coff, cind = 0, k;

      CHKERRQ(PetscSectionGetConstraintDof(section, cp, &cdof));
      CHKERRQ(PetscSectionGetOffset(section, cp, &coff));
      a    = &array[coff];
      if (!cdof) {
        if (o >= 0) {
          for (k = 0; k < dof; ++k) {
            a[k] += values[off+k];
          }
        } else {
          for (k = 0; k < dof; ++k) {
            a[k] += values[off+dof-k-1];
          }
        }
      } else {
        CHKERRQ(PetscSectionGetConstraintIndices(section, cp, &cdofs));
        if (o >= 0) {
          for (k = 0; k < dof; ++k) {
            if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
            a[k] += values[off+k];
          }
        } else {
          for (k = 0; k < dof; ++k) {
            if ((cind < cdof) && (k == cdofs[cind])) {++cind; continue;}
            a[k] += values[off+dof-k-1];
          }
        }
      }
    }
  }
  CHKERRQ(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexVecSetClosure - Set an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. v - The local vector
. point - The point in the DM
. values - The array of values
- mode - The insert mode. One of INSERT_ALL_VALUES, ADD_ALL_VALUES, INSERT_VALUES, ADD_VALUES, INSERT_BC_VALUES, and ADD_BC_VALUES,
         where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions.

  Fortran Notes:
  This routine is only available in Fortran 90, and you must include petsc.h90 in your code.

  Level: intermediate

.seealso DMPlexVecGetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexVecSetClosure(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  PetscSection    clSection;
  IS              clPoints;
  PetscScalar    *array;
  PetscInt       *points = NULL;
  const PetscInt *clp, *clperm = NULL;
  PetscInt        depth, numFields, numPoints, p, clsize;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  if (depth == 1 && numFields < 2 && mode == ADD_VALUES) {
    CHKERRQ(DMPlexVecSetClosure_Depth1_Static(dm, section, v, point, values, mode));
    PetscFunctionReturn(0);
  }
  /* Get points */
  CHKERRQ(DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  for (clsize=0,p=0; p<numPoints; p++) {
    PetscInt dof;
    CHKERRQ(PetscSectionGetDof(section, points[2*p], &dof));
    clsize += dof;
  }
  CHKERRQ(PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, depth, clsize, &clperm));
  /* Get array */
  CHKERRQ(VecGetArray(v, &array));
  /* Get values */
  if (numFields > 0) {
    PetscInt offset = 0, f;
    for (f = 0; f < numFields; ++f) {
      const PetscInt    **perms = NULL;
      const PetscScalar **flips = NULL;

      CHKERRQ(PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms,&flips));
      switch (mode) {
      case INSERT_VALUES:
        for (p = 0; p < numPoints; p++) {
          const PetscInt    point = points[2*p];
          const PetscInt    *perm = perms ? perms[p] : NULL;
          const PetscScalar *flip = flips ? flips[p] : NULL;
          updatePointFields_private(section, point, perm, flip, f, insert, PETSC_FALSE, clperm, values, &offset, array);
        } break;
      case INSERT_ALL_VALUES:
        for (p = 0; p < numPoints; p++) {
          const PetscInt    point = points[2*p];
          const PetscInt    *perm = perms ? perms[p] : NULL;
          const PetscScalar *flip = flips ? flips[p] : NULL;
          updatePointFields_private(section, point, perm, flip, f, insert, PETSC_TRUE, clperm, values, &offset, array);
        } break;
      case INSERT_BC_VALUES:
        for (p = 0; p < numPoints; p++) {
          const PetscInt    point = points[2*p];
          const PetscInt    *perm = perms ? perms[p] : NULL;
          const PetscScalar *flip = flips ? flips[p] : NULL;
          updatePointFieldsBC_private(section, point, perm, flip, f, -1, NULL, insert, clperm, values, &offset, array);
        } break;
      case ADD_VALUES:
        for (p = 0; p < numPoints; p++) {
          const PetscInt    point = points[2*p];
          const PetscInt    *perm = perms ? perms[p] : NULL;
          const PetscScalar *flip = flips ? flips[p] : NULL;
          updatePointFields_private(section, point, perm, flip, f, add, PETSC_FALSE, clperm, values, &offset, array);
        } break;
      case ADD_ALL_VALUES:
        for (p = 0; p < numPoints; p++) {
          const PetscInt    point = points[2*p];
          const PetscInt    *perm = perms ? perms[p] : NULL;
          const PetscScalar *flip = flips ? flips[p] : NULL;
          updatePointFields_private(section, point, perm, flip, f, add, PETSC_TRUE, clperm, values, &offset, array);
        } break;
      case ADD_BC_VALUES:
        for (p = 0; p < numPoints; p++) {
          const PetscInt    point = points[2*p];
          const PetscInt    *perm = perms ? perms[p] : NULL;
          const PetscScalar *flip = flips ? flips[p] : NULL;
          updatePointFieldsBC_private(section, point, perm, flip, f, -1, NULL, add, clperm, values, &offset, array);
        } break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
      }
      CHKERRQ(PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms,&flips));
    }
  } else {
    PetscInt dof, off;
    const PetscInt    **perms = NULL;
    const PetscScalar **flips = NULL;

    CHKERRQ(PetscSectionGetPointSyms(section,numPoints,points,&perms,&flips));
    switch (mode) {
    case INSERT_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        CHKERRQ(PetscSectionGetDof(section, point, &dof));
        updatePoint_private(section, point, dof, insert, PETSC_FALSE, perm, flip, clperm, values, off, array);
      } break;
    case INSERT_ALL_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        CHKERRQ(PetscSectionGetDof(section, point, &dof));
        updatePoint_private(section, point, dof, insert, PETSC_TRUE,  perm, flip, clperm, values, off, array);
      } break;
    case INSERT_BC_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        CHKERRQ(PetscSectionGetDof(section, point, &dof));
        updatePointBC_private(section, point, dof, insert,  perm, flip, clperm, values, off, array);
      } break;
    case ADD_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        CHKERRQ(PetscSectionGetDof(section, point, &dof));
        updatePoint_private(section, point, dof, add,    PETSC_FALSE, perm, flip, clperm, values, off, array);
      } break;
    case ADD_ALL_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        CHKERRQ(PetscSectionGetDof(section, point, &dof));
        updatePoint_private(section, point, dof, add,    PETSC_TRUE,  perm, flip, clperm, values, off, array);
      } break;
    case ADD_BC_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        CHKERRQ(PetscSectionGetDof(section, point, &dof));
        updatePointBC_private(section, point, dof, add,  perm, flip, clperm, values, off, array);
      } break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
    }
    CHKERRQ(PetscSectionRestorePointSyms(section,numPoints,points,&perms,&flips));
  }
  /* Cleanup points */
  CHKERRQ(DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  /* Cleanup array */
  CHKERRQ(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

/* Check whether the given point is in the label. If not, update the offset to skip this point */
static inline PetscErrorCode CheckPoint_Private(DMLabel label, PetscInt labelId, PetscSection section, PetscInt point, PetscInt f, PetscInt *offset)
{
  PetscFunctionBegin;
  if (label) {
    PetscInt       val, fdof;

    /* There is a problem with this:
         Suppose we have two label values, defining surfaces, interecting along a line in 3D. When we add cells to the label, the cells that
       touch both surfaces must pick a label value. Thus we miss setting values for the surface with that other value intersecting that cell.
       Thus I am only going to check val != -1, not val != labelId
    */
    CHKERRQ(DMLabelGetValue(label, point, &val));
    if (val < 0) {
      CHKERRQ(PetscSectionGetFieldDof(section, point, f, &fdof));
      *offset += fdof;
      PetscFunctionReturn(1);
    }
  }
  PetscFunctionReturn(0);
}

/* Unlike DMPlexVecSetClosure(), this uses plex-native closure permutation, not a user-specified permutation such as DMPlexSetClosurePermutationTensor(). */
PetscErrorCode DMPlexVecSetFieldClosure_Internal(DM dm, PetscSection section, Vec v, PetscBool fieldActive[], PetscInt point, PetscInt Ncc, const PetscInt comps[], DMLabel label, PetscInt labelId, const PetscScalar values[], InsertMode mode)
{
  PetscSection    clSection;
  IS              clPoints;
  PetscScalar    *array;
  PetscInt       *points = NULL;
  const PetscInt *clp;
  PetscInt        numFields, numPoints, p;
  PetscInt        offset = 0, f;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  /* Get points */
  CHKERRQ(DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  /* Get array */
  CHKERRQ(VecGetArray(v, &array));
  /* Get values */
  for (f = 0; f < numFields; ++f) {
    const PetscInt    **perms = NULL;
    const PetscScalar **flips = NULL;

    if (!fieldActive[f]) {
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt fdof;
        CHKERRQ(PetscSectionGetFieldDof(section, points[p], f, &fdof));
        offset += fdof;
      }
      continue;
    }
    CHKERRQ(PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms,&flips));
    switch (mode) {
    case INSERT_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        if (CheckPoint_Private(label, labelId, section, point, f, &offset)) continue;
        CHKERRQ(updatePointFields_private(section, point, perm, flip, f, insert, PETSC_FALSE, NULL, values, &offset, array));
      } break;
    case INSERT_ALL_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        if (CheckPoint_Private(label, labelId, section, point, f, &offset)) continue;
        CHKERRQ(updatePointFields_private(section, point, perm, flip, f, insert, PETSC_TRUE, NULL, values, &offset, array));
      } break;
    case INSERT_BC_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        if (CheckPoint_Private(label, labelId, section, point, f, &offset)) continue;
        CHKERRQ(updatePointFieldsBC_private(section, point, perm, flip, f, Ncc, comps, insert, NULL, values, &offset, array));
      } break;
    case ADD_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        if (CheckPoint_Private(label, labelId, section, point, f, &offset)) continue;
        CHKERRQ(updatePointFields_private(section, point, perm, flip, f, add, PETSC_FALSE, NULL, values, &offset, array));
      } break;
    case ADD_ALL_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        if (CheckPoint_Private(label, labelId, section, point, f, &offset)) continue;
        CHKERRQ(updatePointFields_private(section, point, perm, flip, f, add, PETSC_TRUE, NULL, values, &offset, array));
      } break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
    }
    CHKERRQ(PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms,&flips));
  }
  /* Cleanup points */
  CHKERRQ(DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp));
  /* Cleanup array */
  CHKERRQ(VecRestoreArray(v, &array));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexPrintMatSetValues(PetscViewer viewer, Mat A, PetscInt point, PetscInt numRIndices, const PetscInt rindices[], PetscInt numCIndices, const PetscInt cindices[], const PetscScalar values[])
{
  PetscMPIInt    rank;
  PetscInt       i, j;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "[%d]mat for point %D\n", rank, point));
  for (i = 0; i < numRIndices; i++) CHKERRQ(PetscViewerASCIIPrintf(viewer, "[%d]mat row indices[%D] = %D\n", rank, i, rindices[i]));
  for (i = 0; i < numCIndices; i++) CHKERRQ(PetscViewerASCIIPrintf(viewer, "[%d]mat col indices[%D] = %D\n", rank, i, cindices[i]));
  numCIndices = numCIndices ? numCIndices : numRIndices;
  if (!values) PetscFunctionReturn(0);
  for (i = 0; i < numRIndices; i++) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "[%d]", rank));
    for (j = 0; j < numCIndices; j++) {
#if defined(PETSC_USE_COMPLEX)
      CHKERRQ(PetscViewerASCIIPrintf(viewer, " (%g,%g)", (double)PetscRealPart(values[i*numCIndices+j]), (double)PetscImaginaryPart(values[i*numCIndices+j])));
#else
      CHKERRQ(PetscViewerASCIIPrintf(viewer, " %g", (double)values[i*numCIndices+j]));
#endif
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscFunctionReturn(0);
}

/*
  DMPlexGetIndicesPoint_Internal - Add the indices for dofs on a point to an index array

  Input Parameters:
+ section - The section for this data layout
. islocal - Is the section (and thus indices being requested) local or global?
. point   - The point contributing dofs with these indices
. off     - The global offset of this point
. loff    - The local offset of each field
. setBC   - The flag determining whether to include indices of boundary values
. perm    - A permutation of the dofs on this point, or NULL
- indperm - A permutation of the entire indices array, or NULL

  Output Parameter:
. indices - Indices for dofs on this point

  Level: developer

  Note: The indices could be local or global, depending on the value of 'off'.
*/
PetscErrorCode DMPlexGetIndicesPoint_Internal(PetscSection section, PetscBool islocal,PetscInt point, PetscInt off, PetscInt *loff, PetscBool setBC, const PetscInt perm[], const PetscInt indperm[], PetscInt indices[])
{
  PetscInt        dof;   /* The number of unknowns on this point */
  PetscInt        cdof;  /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscInt        cind = 0, k;

  PetscFunctionBegin;
  PetscCheckFalse(!islocal && setBC,PetscObjectComm((PetscObject)section),PETSC_ERR_ARG_INCOMP,"setBC incompatible with global indices; use a local section or disable setBC");
  CHKERRQ(PetscSectionGetDof(section, point, &dof));
  CHKERRQ(PetscSectionGetConstraintDof(section, point, &cdof));
  if (!cdof || setBC) {
    for (k = 0; k < dof; ++k) {
      const PetscInt preind = perm ? *loff+perm[k] : *loff+k;
      const PetscInt ind    = indperm ? indperm[preind] : preind;

      indices[ind] = off + k;
    }
  } else {
    CHKERRQ(PetscSectionGetConstraintIndices(section, point, &cdofs));
    for (k = 0; k < dof; ++k) {
      const PetscInt preind = perm ? *loff+perm[k] : *loff+k;
      const PetscInt ind    = indperm ? indperm[preind] : preind;

      if ((cind < cdof) && (k == cdofs[cind])) {
        /* Insert check for returning constrained indices */
        indices[ind] = -(off+k+1);
        ++cind;
      } else {
        indices[ind] = off + k - (islocal ? 0 : cind);
      }
    }
  }
  *loff += dof;
  PetscFunctionReturn(0);
}

/*
 DMPlexGetIndicesPointFields_Internal - gets section indices for a point in its canonical ordering.

 Input Parameters:
+ section - a section (global or local)
- islocal - PETSC_TRUE if requesting local indices (i.e., section is local); PETSC_FALSE for global
. point - point within section
. off - The offset of this point in the (local or global) indexed space - should match islocal and (usually) the section
. foffs - array of length numFields containing the offset in canonical point ordering (the location in indices) of each field
. setBC - identify constrained (boundary condition) points via involution.
. perms - perms[f][permsoff][:] is a permutation of dofs within each field
. permsoff - offset
- indperm - index permutation

 Output Parameter:
. foffs - each entry is incremented by the number of (unconstrained if setBC=FALSE) dofs in that field
. indices - array to hold indices (as defined by section) of each dof associated with point

 Notes:
 If section is local and setBC=true, there is no distinction between constrained and unconstrained dofs.
 If section is local and setBC=false, the indices for constrained points are the involution -(i+1) of their position
 in the local vector.

 If section is global and setBC=false, the indices for constrained points are negative (and their value is not
 significant).  It is invalid to call with a global section and setBC=true.

 Developer Note:
 The section is only used for field layout, so islocal is technically a statement about the offset (off).  At some point
 in the future, global sections may have fields set, in which case we could pass the global section and obtain the
 offset could be obtained from the section instead of passing it explicitly as we do now.

 Example:
 Suppose a point contains one field with three components, and for which the unconstrained indices are {10, 11, 12}.
 When the middle component is constrained, we get the array {10, -12, 12} for (islocal=TRUE, setBC=FALSE).
 Note that -12 is the involution of 11, so the user can involute negative indices to recover local indices.
 The global vector does not store constrained dofs, so when this function returns global indices, say {110, -112, 111}, the value of -112 is an arbitrary flag that should not be interpreted beyond its sign.

 Level: developer
*/
PetscErrorCode DMPlexGetIndicesPointFields_Internal(PetscSection section, PetscBool islocal, PetscInt point, PetscInt off, PetscInt foffs[], PetscBool setBC, const PetscInt ***perms, PetscInt permsoff, const PetscInt indperm[], PetscInt indices[])
{
  PetscInt       numFields, foff, f;

  PetscFunctionBegin;
  PetscCheckFalse(!islocal && setBC,PetscObjectComm((PetscObject)section),PETSC_ERR_ARG_INCOMP,"setBC incompatible with global indices; use a local section or disable setBC");
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  for (f = 0, foff = 0; f < numFields; ++f) {
    PetscInt        fdof, cfdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, b;
    const PetscInt  *perm = (perms && perms[f]) ? perms[f][permsoff] : NULL;

    CHKERRQ(PetscSectionGetFieldDof(section, point, f, &fdof));
    CHKERRQ(PetscSectionGetFieldConstraintDof(section, point, f, &cfdof));
    if (!cfdof || setBC) {
      for (b = 0; b < fdof; ++b) {
        const PetscInt preind = perm ? foffs[f]+perm[b] : foffs[f]+b;
        const PetscInt ind    = indperm ? indperm[preind] : preind;

        indices[ind] = off+foff+b;
      }
    } else {
      CHKERRQ(PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs));
      for (b = 0; b < fdof; ++b) {
        const PetscInt preind = perm ? foffs[f]+perm[b] : foffs[f]+b;
        const PetscInt ind    = indperm ? indperm[preind] : preind;

        if ((cind < cfdof) && (b == fcdofs[cind])) {
          indices[ind] = -(off+foff+b+1);
          ++cind;
        } else {
          indices[ind] = off + foff + b - (islocal ? 0 : cind);
        }
      }
    }
    foff     += (setBC || islocal ? fdof : (fdof - cfdof));
    foffs[f] += fdof;
  }
  PetscFunctionReturn(0);
}

/*
  This version believes the globalSection offsets for each field, rather than just the point offset

 . foffs - The offset into 'indices' for each field, since it is segregated by field

 Notes:
 The semantics of this function relate to that of setBC=FALSE in DMPlexGetIndicesPointFields_Internal.
 Since this function uses global indices, setBC=TRUE would be invalid, so no such argument exists.
*/
static PetscErrorCode DMPlexGetIndicesPointFieldsSplit_Internal(PetscSection section, PetscSection globalSection, PetscInt point, PetscInt foffs[], const PetscInt ***perms, PetscInt permsoff, const PetscInt indperm[], PetscInt indices[])
{
  PetscInt       numFields, foff, f;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));
  for (f = 0; f < numFields; ++f) {
    PetscInt        fdof, cfdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, b;
    const PetscInt  *perm = (perms && perms[f]) ? perms[f][permsoff] : NULL;

    CHKERRQ(PetscSectionGetFieldDof(section, point, f, &fdof));
    CHKERRQ(PetscSectionGetFieldConstraintDof(section, point, f, &cfdof));
    CHKERRQ(PetscSectionGetFieldOffset(globalSection, point, f, &foff));
    if (!cfdof) {
      for (b = 0; b < fdof; ++b) {
        const PetscInt preind = perm ? foffs[f]+perm[b] : foffs[f]+b;
        const PetscInt ind    = indperm ? indperm[preind] : preind;

        indices[ind] = foff+b;
      }
    } else {
      CHKERRQ(PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs));
      for (b = 0; b < fdof; ++b) {
        const PetscInt preind = perm ? foffs[f]+perm[b] : foffs[f]+b;
        const PetscInt ind    = indperm ? indperm[preind] : preind;

        if ((cind < cfdof) && (b == fcdofs[cind])) {
          indices[ind] = -(foff+b+1);
          ++cind;
        } else {
          indices[ind] = foff+b-cind;
        }
      }
    }
    foffs[f] += fdof;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexAnchorsModifyMat(DM dm, PetscSection section, PetscInt numPoints, PetscInt numIndices, const PetscInt points[], const PetscInt ***perms, const PetscScalar values[], PetscInt *outNumPoints, PetscInt *outNumIndices, PetscInt *outPoints[], PetscScalar *outValues[], PetscInt offsets[], PetscBool multiplyLeft)
{
  Mat             cMat;
  PetscSection    aSec, cSec;
  IS              aIS;
  PetscInt        aStart = -1, aEnd = -1;
  const PetscInt  *anchors;
  PetscInt        numFields, f, p, q, newP = 0;
  PetscInt        newNumPoints = 0, newNumIndices = 0;
  PetscInt        *newPoints, *indices, *newIndices;
  PetscInt        maxAnchor, maxDof;
  PetscInt        newOffsets[32];
  PetscInt        *pointMatOffsets[32];
  PetscInt        *newPointOffsets[32];
  PetscScalar     *pointMat[32];
  PetscScalar     *newValues=NULL,*tmpValues;
  PetscBool       anyConstrained = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  CHKERRQ(PetscSectionGetNumFields(section, &numFields));

  CHKERRQ(DMPlexGetAnchors(dm,&aSec,&aIS));
  /* if there are point-to-point constraints */
  if (aSec) {
    CHKERRQ(PetscArrayzero(newOffsets, 32));
    CHKERRQ(ISGetIndices(aIS,&anchors));
    CHKERRQ(PetscSectionGetChart(aSec,&aStart,&aEnd));
    /* figure out how many points are going to be in the new element matrix
     * (we allow double counting, because it's all just going to be summed
     * into the global matrix anyway) */
    for (p = 0; p < 2*numPoints; p+=2) {
      PetscInt b    = points[p];
      PetscInt bDof = 0, bSecDof;

      CHKERRQ(PetscSectionGetDof(section,b,&bSecDof));
      if (!bSecDof) {
        continue;
      }
      if (b >= aStart && b < aEnd) {
        CHKERRQ(PetscSectionGetDof(aSec,b,&bDof));
      }
      if (bDof) {
        /* this point is constrained */
        /* it is going to be replaced by its anchors */
        PetscInt bOff, q;

        anyConstrained = PETSC_TRUE;
        newNumPoints  += bDof;
        CHKERRQ(PetscSectionGetOffset(aSec,b,&bOff));
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q];
          PetscInt aDof;

          CHKERRQ(PetscSectionGetDof(section,a,&aDof));
          newNumIndices += aDof;
          for (f = 0; f < numFields; ++f) {
            PetscInt fDof;

            CHKERRQ(PetscSectionGetFieldDof(section, a, f, &fDof));
            newOffsets[f+1] += fDof;
          }
        }
      }
      else {
        /* this point is not constrained */
        newNumPoints++;
        newNumIndices += bSecDof;
        for (f = 0; f < numFields; ++f) {
          PetscInt fDof;

          CHKERRQ(PetscSectionGetFieldDof(section, b, f, &fDof));
          newOffsets[f+1] += fDof;
        }
      }
    }
  }
  if (!anyConstrained) {
    if (outNumPoints)  *outNumPoints  = 0;
    if (outNumIndices) *outNumIndices = 0;
    if (outPoints)     *outPoints     = NULL;
    if (outValues)     *outValues     = NULL;
    if (aSec) CHKERRQ(ISRestoreIndices(aIS,&anchors));
    PetscFunctionReturn(0);
  }

  if (outNumPoints)  *outNumPoints  = newNumPoints;
  if (outNumIndices) *outNumIndices = newNumIndices;

  for (f = 0; f < numFields; ++f) newOffsets[f+1] += newOffsets[f];

  if (!outPoints && !outValues) {
    if (offsets) {
      for (f = 0; f <= numFields; f++) {
        offsets[f] = newOffsets[f];
      }
    }
    if (aSec) CHKERRQ(ISRestoreIndices(aIS,&anchors));
    PetscFunctionReturn(0);
  }

  PetscCheckFalse(numFields && newOffsets[numFields] != newNumIndices,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", newOffsets[numFields], newNumIndices);

  CHKERRQ(DMGetDefaultConstraints(dm, &cSec, &cMat, NULL));

  /* workspaces */
  if (numFields) {
    for (f = 0; f < numFields; f++) {
      CHKERRQ(DMGetWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[f]));
      CHKERRQ(DMGetWorkArray(dm,numPoints+1,MPIU_INT,&newPointOffsets[f]));
    }
  }
  else {
    CHKERRQ(DMGetWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[0]));
    CHKERRQ(DMGetWorkArray(dm,numPoints,MPIU_INT,&newPointOffsets[0]));
  }

  /* get workspaces for the point-to-point matrices */
  if (numFields) {
    PetscInt totalOffset, totalMatOffset;

    for (p = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt bDof = 0, bSecDof;

      CHKERRQ(PetscSectionGetDof(section,b,&bSecDof));
      if (!bSecDof) {
        for (f = 0; f < numFields; f++) {
          newPointOffsets[f][p + 1] = 0;
          pointMatOffsets[f][p + 1] = 0;
        }
        continue;
      }
      if (b >= aStart && b < aEnd) {
        CHKERRQ(PetscSectionGetDof(aSec, b, &bDof));
      }
      if (bDof) {
        for (f = 0; f < numFields; f++) {
          PetscInt fDof, q, bOff, allFDof = 0;

          CHKERRQ(PetscSectionGetFieldDof(section, b, f, &fDof));
          CHKERRQ(PetscSectionGetOffset(aSec, b, &bOff));
          for (q = 0; q < bDof; q++) {
            PetscInt a = anchors[bOff + q];
            PetscInt aFDof;

            CHKERRQ(PetscSectionGetFieldDof(section, a, f, &aFDof));
            allFDof += aFDof;
          }
          newPointOffsets[f][p+1] = allFDof;
          pointMatOffsets[f][p+1] = fDof * allFDof;
        }
      }
      else {
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          CHKERRQ(PetscSectionGetFieldDof(section, b, f, &fDof));
          newPointOffsets[f][p+1] = fDof;
          pointMatOffsets[f][p+1] = 0;
        }
      }
    }
    for (f = 0, totalOffset = 0, totalMatOffset = 0; f < numFields; f++) {
      newPointOffsets[f][0] = totalOffset;
      pointMatOffsets[f][0] = totalMatOffset;
      for (p = 0; p < numPoints; p++) {
        newPointOffsets[f][p+1] += newPointOffsets[f][p];
        pointMatOffsets[f][p+1] += pointMatOffsets[f][p];
      }
      totalOffset    = newPointOffsets[f][numPoints];
      totalMatOffset = pointMatOffsets[f][numPoints];
      CHKERRQ(DMGetWorkArray(dm,pointMatOffsets[f][numPoints],MPIU_SCALAR,&pointMat[f]));
    }
  }
  else {
    for (p = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt bDof = 0, bSecDof;

      CHKERRQ(PetscSectionGetDof(section,b,&bSecDof));
      if (!bSecDof) {
        newPointOffsets[0][p + 1] = 0;
        pointMatOffsets[0][p + 1] = 0;
        continue;
      }
      if (b >= aStart && b < aEnd) {
        CHKERRQ(PetscSectionGetDof(aSec, b, &bDof));
      }
      if (bDof) {
        PetscInt bOff, q, allDof = 0;

        CHKERRQ(PetscSectionGetOffset(aSec, b, &bOff));
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aDof;

          CHKERRQ(PetscSectionGetDof(section, a, &aDof));
          allDof += aDof;
        }
        newPointOffsets[0][p+1] = allDof;
        pointMatOffsets[0][p+1] = bSecDof * allDof;
      }
      else {
        newPointOffsets[0][p+1] = bSecDof;
        pointMatOffsets[0][p+1] = 0;
      }
    }
    newPointOffsets[0][0] = 0;
    pointMatOffsets[0][0] = 0;
    for (p = 0; p < numPoints; p++) {
      newPointOffsets[0][p+1] += newPointOffsets[0][p];
      pointMatOffsets[0][p+1] += pointMatOffsets[0][p];
    }
    CHKERRQ(DMGetWorkArray(dm,pointMatOffsets[0][numPoints],MPIU_SCALAR,&pointMat[0]));
  }

  /* output arrays */
  CHKERRQ(DMGetWorkArray(dm,2*newNumPoints,MPIU_INT,&newPoints));

  /* get the point-to-point matrices; construct newPoints */
  CHKERRQ(PetscSectionGetMaxDof(aSec, &maxAnchor));
  CHKERRQ(PetscSectionGetMaxDof(section, &maxDof));
  CHKERRQ(DMGetWorkArray(dm,maxDof,MPIU_INT,&indices));
  CHKERRQ(DMGetWorkArray(dm,maxAnchor*maxDof,MPIU_INT,&newIndices));
  if (numFields) {
    for (p = 0, newP = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt o    = points[2*p+1];
      PetscInt bDof = 0, bSecDof;

      CHKERRQ(PetscSectionGetDof(section, b, &bSecDof));
      if (!bSecDof) {
        continue;
      }
      if (b >= aStart && b < aEnd) {
        CHKERRQ(PetscSectionGetDof(aSec, b, &bDof));
      }
      if (bDof) {
        PetscInt fStart[32], fEnd[32], fAnchorStart[32], fAnchorEnd[32], bOff, q;

        fStart[0] = 0;
        fEnd[0]   = 0;
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          CHKERRQ(PetscSectionGetFieldDof(cSec, b, f, &fDof));
          fStart[f+1] = fStart[f] + fDof;
          fEnd[f+1]   = fStart[f+1];
        }
        CHKERRQ(PetscSectionGetOffset(cSec, b, &bOff));
        CHKERRQ(DMPlexGetIndicesPointFields_Internal(cSec, PETSC_TRUE, b, bOff, fEnd, PETSC_TRUE, perms, p, NULL, indices));

        fAnchorStart[0] = 0;
        fAnchorEnd[0]   = 0;
        for (f = 0; f < numFields; f++) {
          PetscInt fDof = newPointOffsets[f][p + 1] - newPointOffsets[f][p];

          fAnchorStart[f+1] = fAnchorStart[f] + fDof;
          fAnchorEnd[f+1]   = fAnchorStart[f + 1];
        }
        CHKERRQ(PetscSectionGetOffset(aSec, b, &bOff));
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aOff;

          /* we take the orientation of ap into account in the order that we constructed the indices above: the newly added points have no orientation */
          newPoints[2*(newP + q)]     = a;
          newPoints[2*(newP + q) + 1] = 0;
          CHKERRQ(PetscSectionGetOffset(section, a, &aOff));
          CHKERRQ(DMPlexGetIndicesPointFields_Internal(section, PETSC_TRUE, a, aOff, fAnchorEnd, PETSC_TRUE, NULL, -1, NULL, newIndices));
        }
        newP += bDof;

        if (outValues) {
          /* get the point-to-point submatrix */
          for (f = 0; f < numFields; f++) {
            CHKERRQ(MatGetValues(cMat,fEnd[f]-fStart[f],indices + fStart[f],fAnchorEnd[f] - fAnchorStart[f],newIndices + fAnchorStart[f],pointMat[f] + pointMatOffsets[f][p]));
          }
        }
      }
      else {
        newPoints[2 * newP]     = b;
        newPoints[2 * newP + 1] = o;
        newP++;
      }
    }
  } else {
    for (p = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt o    = points[2*p+1];
      PetscInt bDof = 0, bSecDof;

      CHKERRQ(PetscSectionGetDof(section, b, &bSecDof));
      if (!bSecDof) {
        continue;
      }
      if (b >= aStart && b < aEnd) {
        CHKERRQ(PetscSectionGetDof(aSec, b, &bDof));
      }
      if (bDof) {
        PetscInt bEnd = 0, bAnchorEnd = 0, bOff;

        CHKERRQ(PetscSectionGetOffset(cSec, b, &bOff));
        CHKERRQ(DMPlexGetIndicesPoint_Internal(cSec, PETSC_TRUE, b, bOff, &bEnd, PETSC_TRUE, (perms && perms[0]) ? perms[0][p] : NULL, NULL, indices));

        CHKERRQ(PetscSectionGetOffset (aSec, b, &bOff));
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aOff;

          /* we take the orientation of ap into account in the order that we constructed the indices above: the newly added points have no orientation */

          newPoints[2*(newP + q)]     = a;
          newPoints[2*(newP + q) + 1] = 0;
          CHKERRQ(PetscSectionGetOffset(section, a, &aOff));
          CHKERRQ(DMPlexGetIndicesPoint_Internal(section, PETSC_TRUE, a, aOff, &bAnchorEnd, PETSC_TRUE, NULL, NULL, newIndices));
        }
        newP += bDof;

        /* get the point-to-point submatrix */
        if (outValues) {
          CHKERRQ(MatGetValues(cMat,bEnd,indices,bAnchorEnd,newIndices,pointMat[0] + pointMatOffsets[0][p]));
        }
      }
      else {
        newPoints[2 * newP]     = b;
        newPoints[2 * newP + 1] = o;
        newP++;
      }
    }
  }

  if (outValues) {
    CHKERRQ(DMGetWorkArray(dm,newNumIndices*numIndices,MPIU_SCALAR,&tmpValues));
    CHKERRQ(PetscArrayzero(tmpValues,newNumIndices*numIndices));
    /* multiply constraints on the right */
    if (numFields) {
      for (f = 0; f < numFields; f++) {
        PetscInt oldOff = offsets[f];

        for (p = 0; p < numPoints; p++) {
          PetscInt cStart = newPointOffsets[f][p];
          PetscInt b      = points[2 * p];
          PetscInt c, r, k;
          PetscInt dof;

          CHKERRQ(PetscSectionGetFieldDof(section,b,f,&dof));
          if (!dof) {
            continue;
          }
          if (pointMatOffsets[f][p] < pointMatOffsets[f][p + 1]) {
            PetscInt nCols         = newPointOffsets[f][p+1]-cStart;
            const PetscScalar *mat = pointMat[f] + pointMatOffsets[f][p];

            for (r = 0; r < numIndices; r++) {
              for (c = 0; c < nCols; c++) {
                for (k = 0; k < dof; k++) {
                  tmpValues[r * newNumIndices + cStart + c] += values[r * numIndices + oldOff + k] * mat[k * nCols + c];
                }
              }
            }
          }
          else {
            /* copy this column as is */
            for (r = 0; r < numIndices; r++) {
              for (c = 0; c < dof; c++) {
                tmpValues[r * newNumIndices + cStart + c] = values[r * numIndices + oldOff + c];
              }
            }
          }
          oldOff += dof;
        }
      }
    }
    else {
      PetscInt oldOff = 0;
      for (p = 0; p < numPoints; p++) {
        PetscInt cStart = newPointOffsets[0][p];
        PetscInt b      = points[2 * p];
        PetscInt c, r, k;
        PetscInt dof;

        CHKERRQ(PetscSectionGetDof(section,b,&dof));
        if (!dof) {
          continue;
        }
        if (pointMatOffsets[0][p] < pointMatOffsets[0][p + 1]) {
          PetscInt nCols         = newPointOffsets[0][p+1]-cStart;
          const PetscScalar *mat = pointMat[0] + pointMatOffsets[0][p];

          for (r = 0; r < numIndices; r++) {
            for (c = 0; c < nCols; c++) {
              for (k = 0; k < dof; k++) {
                tmpValues[r * newNumIndices + cStart + c] += mat[k * nCols + c] * values[r * numIndices + oldOff + k];
              }
            }
          }
        }
        else {
          /* copy this column as is */
          for (r = 0; r < numIndices; r++) {
            for (c = 0; c < dof; c++) {
              tmpValues[r * newNumIndices + cStart + c] = values[r * numIndices + oldOff + c];
            }
          }
        }
        oldOff += dof;
      }
    }

    if (multiplyLeft) {
      CHKERRQ(DMGetWorkArray(dm,newNumIndices*newNumIndices,MPIU_SCALAR,&newValues));
      CHKERRQ(PetscArrayzero(newValues,newNumIndices*newNumIndices));
      /* multiply constraints transpose on the left */
      if (numFields) {
        for (f = 0; f < numFields; f++) {
          PetscInt oldOff = offsets[f];

          for (p = 0; p < numPoints; p++) {
            PetscInt rStart = newPointOffsets[f][p];
            PetscInt b      = points[2 * p];
            PetscInt c, r, k;
            PetscInt dof;

            CHKERRQ(PetscSectionGetFieldDof(section,b,f,&dof));
            if (pointMatOffsets[f][p] < pointMatOffsets[f][p + 1]) {
              PetscInt nRows                        = newPointOffsets[f][p+1]-rStart;
              const PetscScalar *PETSC_RESTRICT mat = pointMat[f] + pointMatOffsets[f][p];

              for (r = 0; r < nRows; r++) {
                for (c = 0; c < newNumIndices; c++) {
                  for (k = 0; k < dof; k++) {
                    newValues[(rStart + r) * newNumIndices + c] += mat[k * nRows + r] * tmpValues[(oldOff + k) * newNumIndices + c];
                  }
                }
              }
            }
            else {
              /* copy this row as is */
              for (r = 0; r < dof; r++) {
                for (c = 0; c < newNumIndices; c++) {
                  newValues[(rStart + r) * newNumIndices + c] = tmpValues[(oldOff + r) * newNumIndices + c];
                }
              }
            }
            oldOff += dof;
          }
        }
      }
      else {
        PetscInt oldOff = 0;

        for (p = 0; p < numPoints; p++) {
          PetscInt rStart = newPointOffsets[0][p];
          PetscInt b      = points[2 * p];
          PetscInt c, r, k;
          PetscInt dof;

          CHKERRQ(PetscSectionGetDof(section,b,&dof));
          if (pointMatOffsets[0][p] < pointMatOffsets[0][p + 1]) {
            PetscInt nRows                        = newPointOffsets[0][p+1]-rStart;
            const PetscScalar *PETSC_RESTRICT mat = pointMat[0] + pointMatOffsets[0][p];

            for (r = 0; r < nRows; r++) {
              for (c = 0; c < newNumIndices; c++) {
                for (k = 0; k < dof; k++) {
                  newValues[(rStart + r) * newNumIndices + c] += mat[k * nRows + r] * tmpValues[(oldOff + k) * newNumIndices + c];
                }
              }
            }
          }
          else {
            /* copy this row as is */
            for (r = 0; r < dof; r++) {
              for (c = 0; c < newNumIndices; c++) {
                newValues[(rStart + r) * newNumIndices + c] = tmpValues[(oldOff + r) * newNumIndices + c];
              }
            }
          }
          oldOff += dof;
        }
      }

      CHKERRQ(DMRestoreWorkArray(dm,newNumIndices*numIndices,MPIU_SCALAR,&tmpValues));
    }
    else {
      newValues = tmpValues;
    }
  }

  /* clean up */
  CHKERRQ(DMRestoreWorkArray(dm,maxDof,MPIU_INT,&indices));
  CHKERRQ(DMRestoreWorkArray(dm,maxAnchor*maxDof,MPIU_INT,&newIndices));

  if (numFields) {
    for (f = 0; f < numFields; f++) {
      CHKERRQ(DMRestoreWorkArray(dm,pointMatOffsets[f][numPoints],MPIU_SCALAR,&pointMat[f]));
      CHKERRQ(DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[f]));
      CHKERRQ(DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&newPointOffsets[f]));
    }
  }
  else {
    CHKERRQ(DMRestoreWorkArray(dm,pointMatOffsets[0][numPoints],MPIU_SCALAR,&pointMat[0]));
    CHKERRQ(DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[0]));
    CHKERRQ(DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&newPointOffsets[0]));
  }
  CHKERRQ(ISRestoreIndices(aIS,&anchors));

  /* output */
  if (outPoints) {
    *outPoints = newPoints;
  }
  else {
    CHKERRQ(DMRestoreWorkArray(dm,2*newNumPoints,MPIU_INT,&newPoints));
  }
  if (outValues) {
    *outValues = newValues;
  }
  for (f = 0; f <= numFields; f++) {
    offsets[f] = newOffsets[f];
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetClosureIndices - Gets the global dof indices associated with the closure of the given point within the provided sections.

  Not collective

  Input Parameters:
+ dm         - The DM
. section    - The PetscSection describing the points (a local section)
. idxSection - The PetscSection from which to obtain indices (may be local or global)
. point      - The point defining the closure
- useClPerm  - Use the closure point permutation if available

  Output Parameters:
+ numIndices - The number of dof indices in the closure of point with the input sections
. indices    - The dof indices
. outOffsets - Array to write the field offsets into, or NULL
- values     - The input values, which may be modified if sign flips are induced by the point symmetries, or NULL

  Notes:
  Must call DMPlexRestoreClosureIndices() to free allocated memory

  If idxSection is global, any constrained dofs (see DMAddBoundary(), for example) will get negative indices.  The value
  of those indices is not significant.  If idxSection is local, the constrained dofs will yield the involution -(idx+1)
  of their index in a local vector.  A caller who does not wish to distinguish those points may recover the nonnegative
  indices via involution, -(-(idx+1)+1)==idx.  Local indices are provided when idxSection == section, otherwise global
  indices (with the above semantics) are implied.

  Level: advanced

.seealso DMPlexRestoreClosureIndices(), DMPlexVecGetClosure(), DMPlexMatSetClosure(), DMGetLocalSection(), DMGetGlobalSection()
@*/
PetscErrorCode DMPlexGetClosureIndices(DM dm, PetscSection section, PetscSection idxSection, PetscInt point, PetscBool useClPerm,
                                       PetscInt *numIndices, PetscInt *indices[], PetscInt outOffsets[], PetscScalar *values[])
{
  /* Closure ordering */
  PetscSection        clSection;
  IS                  clPoints;
  const PetscInt     *clp;
  PetscInt           *points;
  const PetscInt     *clperm = NULL;
  /* Dof permutation and sign flips */
  const PetscInt    **perms[32] = {NULL};
  const PetscScalar **flips[32] = {NULL};
  PetscScalar        *valCopy   = NULL;
  /* Hanging node constraints */
  PetscInt           *pointsC = NULL;
  PetscScalar        *valuesC = NULL;
  PetscInt            NclC, NiC;

  PetscInt           *idx;
  PetscInt            Nf, Ncl, Ni = 0, offsets[32], p, f;
  PetscBool           isLocal = (section == idxSection) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(idxSection, PETSC_SECTION_CLASSID, 3);
  if (numIndices) PetscValidPointer(numIndices, 6);
  if (indices)    PetscValidPointer(indices, 7);
  if (outOffsets) PetscValidPointer(outOffsets, 8);
  if (values)     PetscValidPointer(values, 9);
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  PetscCheckFalse(Nf > 31,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", Nf);
  CHKERRQ(PetscArrayzero(offsets, 32));
  /* 1) Get points in closure */
  CHKERRQ(DMPlexGetCompressedClosure(dm, section, point, &Ncl, &points, &clSection, &clPoints, &clp));
  if (useClPerm) {
    PetscInt depth, clsize;
    CHKERRQ(DMPlexGetPointDepth(dm, point, &depth));
    for (clsize=0,p=0; p<Ncl; p++) {
      PetscInt dof;
      CHKERRQ(PetscSectionGetDof(section, points[2*p], &dof));
      clsize += dof;
    }
    CHKERRQ(PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, depth, clsize, &clperm));
  }
  /* 2) Get number of indices on these points and field offsets from section */
  for (p = 0; p < Ncl*2; p += 2) {
    PetscInt dof, fdof;

    CHKERRQ(PetscSectionGetDof(section, points[p], &dof));
    for (f = 0; f < Nf; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(section, points[p], f, &fdof));
      offsets[f+1] += fdof;
    }
    Ni += dof;
  }
  for (f = 1; f < Nf; ++f) offsets[f+1] += offsets[f];
  PetscCheckFalse(Nf && offsets[Nf] != Ni,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", offsets[Nf], Ni);
  /* 3) Get symmetries and sign flips. Apply sign flips to values if passed in (only works for square values matrix) */
  for (f = 0; f < PetscMax(1, Nf); ++f) {
    if (Nf) CHKERRQ(PetscSectionGetFieldPointSyms(section, f, Ncl, points, &perms[f], &flips[f]));
    else    CHKERRQ(PetscSectionGetPointSyms(section, Ncl, points, &perms[f], &flips[f]));
    /* may need to apply sign changes to the element matrix */
    if (values && flips[f]) {
      PetscInt foffset = offsets[f];

      for (p = 0; p < Ncl; ++p) {
        PetscInt           pnt  = points[2*p], fdof;
        const PetscScalar *flip = flips[f] ? flips[f][p] : NULL;

        if (!Nf) CHKERRQ(PetscSectionGetDof(section, pnt, &fdof));
        else     CHKERRQ(PetscSectionGetFieldDof(section, pnt, f, &fdof));
        if (flip) {
          PetscInt i, j, k;

          if (!valCopy) {
            CHKERRQ(DMGetWorkArray(dm, Ni*Ni, MPIU_SCALAR, &valCopy));
            for (j = 0; j < Ni * Ni; ++j) valCopy[j] = (*values)[j];
            *values = valCopy;
          }
          for (i = 0; i < fdof; ++i) {
            PetscScalar fval = flip[i];

            for (k = 0; k < Ni; ++k) {
              valCopy[Ni * (foffset + i) + k] *= fval;
              valCopy[Ni * k + (foffset + i)] *= fval;
            }
          }
        }
        foffset += fdof;
      }
    }
  }
  /* 4) Apply hanging node constraints. Get new symmetries and replace all storage with constrained storage */
  CHKERRQ(DMPlexAnchorsModifyMat(dm, section, Ncl, Ni, points, perms, values ? *values : NULL, &NclC, &NiC, &pointsC, values ? &valuesC : NULL, offsets, PETSC_TRUE));
  if (NclC) {
    if (valCopy) CHKERRQ(DMRestoreWorkArray(dm, Ni*Ni, MPIU_SCALAR, &valCopy));
    for (f = 0; f < PetscMax(1, Nf); ++f) {
      if (Nf) CHKERRQ(PetscSectionRestoreFieldPointSyms(section, f, Ncl, points, &perms[f], &flips[f]));
      else    CHKERRQ(PetscSectionRestorePointSyms(section, Ncl, points, &perms[f], &flips[f]));
    }
    for (f = 0; f < PetscMax(1, Nf); ++f) {
      if (Nf) CHKERRQ(PetscSectionGetFieldPointSyms(section, f, NclC, pointsC, &perms[f], &flips[f]));
      else    CHKERRQ(PetscSectionGetPointSyms(section, NclC, pointsC, &perms[f], &flips[f]));
    }
    CHKERRQ(DMPlexRestoreCompressedClosure(dm, section, point, &Ncl, &points, &clSection, &clPoints, &clp));
    Ncl     = NclC;
    Ni      = NiC;
    points  = pointsC;
    if (values) *values = valuesC;
  }
  /* 5) Calculate indices */
  CHKERRQ(DMGetWorkArray(dm, Ni, MPIU_INT, &idx));
  if (Nf) {
    PetscInt  idxOff;
    PetscBool useFieldOffsets;

    if (outOffsets) {for (f = 0; f <= Nf; f++) outOffsets[f] = offsets[f];}
    CHKERRQ(PetscSectionGetUseFieldOffsets(idxSection, &useFieldOffsets));
    if (useFieldOffsets) {
      for (p = 0; p < Ncl; ++p) {
        const PetscInt pnt = points[p*2];

        CHKERRQ(DMPlexGetIndicesPointFieldsSplit_Internal(section, idxSection, pnt, offsets, perms, p, clperm, idx));
      }
    } else {
      for (p = 0; p < Ncl; ++p) {
        const PetscInt pnt = points[p*2];

        CHKERRQ(PetscSectionGetOffset(idxSection, pnt, &idxOff));
        /* Note that we pass a local section even though we're using global offsets.  This is because global sections do
         * not (at the time of this writing) have fields set. They probably should, in which case we would pass the
         * global section. */
        CHKERRQ(DMPlexGetIndicesPointFields_Internal(section, isLocal, pnt, idxOff < 0 ? -(idxOff+1) : idxOff, offsets, PETSC_FALSE, perms, p, clperm, idx));
      }
    }
  } else {
    PetscInt off = 0, idxOff;

    for (p = 0; p < Ncl; ++p) {
      const PetscInt  pnt  = points[p*2];
      const PetscInt *perm = perms[0] ? perms[0][p] : NULL;

      CHKERRQ(PetscSectionGetOffset(idxSection, pnt, &idxOff));
      /* Note that we pass a local section even though we're using global offsets.  This is because global sections do
       * not (at the time of this writing) have fields set. They probably should, in which case we would pass the global section. */
      CHKERRQ(DMPlexGetIndicesPoint_Internal(section, isLocal, pnt, idxOff < 0 ? -(idxOff+1) : idxOff, &off, PETSC_FALSE, perm, clperm, idx));
    }
  }
  /* 6) Cleanup */
  for (f = 0; f < PetscMax(1, Nf); ++f) {
    if (Nf) CHKERRQ(PetscSectionRestoreFieldPointSyms(section, f, Ncl, points, &perms[f], &flips[f]));
    else    CHKERRQ(PetscSectionRestorePointSyms(section, Ncl, points, &perms[f], &flips[f]));
  }
  if (NclC) {
    CHKERRQ(DMRestoreWorkArray(dm, NclC*2, MPIU_INT, &pointsC));
  } else {
    CHKERRQ(DMPlexRestoreCompressedClosure(dm, section, point, &Ncl, &points, &clSection, &clPoints, &clp));
  }

  if (numIndices) *numIndices = Ni;
  if (indices)    *indices    = idx;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreClosureIndices - Restores the global dof indices associated with the closure of the given point within the provided sections.

  Not collective

  Input Parameters:
+ dm         - The DM
. section    - The PetscSection describing the points (a local section)
. idxSection - The PetscSection from which to obtain indices (may be local or global)
. point      - The point defining the closure
- useClPerm  - Use the closure point permutation if available

  Output Parameters:
+ numIndices - The number of dof indices in the closure of point with the input sections
. indices    - The dof indices
. outOffsets - Array to write the field offsets into, or NULL
- values     - The input values, which may be modified if sign flips are induced by the point symmetries, or NULL

  Notes:
  If values were modified, the user is responsible for calling DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &values).

  If idxSection is global, any constrained dofs (see DMAddBoundary(), for example) will get negative indices.  The value
  of those indices is not significant.  If idxSection is local, the constrained dofs will yield the involution -(idx+1)
  of their index in a local vector.  A caller who does not wish to distinguish those points may recover the nonnegative
  indices via involution, -(-(idx+1)+1)==idx.  Local indices are provided when idxSection == section, otherwise global
  indices (with the above semantics) are implied.

  Level: advanced

.seealso DMPlexGetClosureIndices(), DMPlexVecGetClosure(), DMPlexMatSetClosure(), DMGetLocalSection(), DMGetGlobalSection()
@*/
PetscErrorCode DMPlexRestoreClosureIndices(DM dm, PetscSection section, PetscSection idxSection, PetscInt point, PetscBool useClPerm,
                                           PetscInt *numIndices, PetscInt *indices[], PetscInt outOffsets[], PetscScalar *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(indices, 7);
  CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_INT, indices));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexMatSetClosure - Set an array of the values on the closure of 'point'

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. globalSection - The section describing the layout in v, or NULL to use the default global section
. A - The matrix
. point - The point in the DM
. values - The array of values
- mode - The insert mode, where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions

  Fortran Notes:
  This routine is only available in Fortran 90, and you must include petsc.h90 in your code.

  Level: intermediate

.seealso DMPlexMatSetClosureGeneral(), DMPlexVecGetClosure(), DMPlexVecSetClosure()
@*/
PetscErrorCode DMPlexMatSetClosure(DM dm, PetscSection section, PetscSection globalSection, Mat A, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  DM_Plex           *mesh = (DM_Plex*) dm->data;
  PetscInt          *indices;
  PetscInt           numIndices;
  const PetscScalar *valuesOrig = values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) CHKERRQ(DMGetLocalSection(dm, &section));
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  if (!globalSection) CHKERRQ(DMGetGlobalSection(dm, &globalSection));
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 4);

  CHKERRQ(DMPlexGetClosureIndices(dm, section, globalSection, point, PETSC_TRUE, &numIndices, &indices, NULL, (PetscScalar **) &values));

  if (mesh->printSetValues) CHKERRQ(DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numIndices, indices, 0, NULL, values));
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, values, mode);
  if (ierr) {
    PetscMPIInt    rank;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    CHKERRQ((*PetscErrorPrintf)("[%d]ERROR in DMPlexMatSetClosure\n", rank));
    CHKERRQ(DMPlexPrintMatSetValues(PETSC_VIEWER_STDERR_SELF, A, point, numIndices, indices, 0, NULL, values));
    CHKERRQ(DMPlexRestoreClosureIndices(dm, section, globalSection, point, PETSC_TRUE, &numIndices, &indices, NULL, (PetscScalar **) &values));
    if (values != valuesOrig) CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &values));
    SETERRQ(PetscObjectComm((PetscObject)dm),ierr,"Not possible to set matrix values");
  }
  if (mesh->printFEM > 1) {
    PetscInt i;
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Indices:"));
    for (i = 0; i < numIndices; ++i) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %D", indices[i]));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
  }

  CHKERRQ(DMPlexRestoreClosureIndices(dm, section, globalSection, point, PETSC_TRUE, &numIndices, &indices, NULL, (PetscScalar **) &values));
  if (values != valuesOrig) CHKERRQ(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &values));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexMatSetClosure - Set an array of the values on the closure of 'point' using a different row and column section

  Not collective

  Input Parameters:
+ dmRow - The DM for the row fields
. sectionRow - The section describing the layout, or NULL to use the default section in dmRow
. globalSectionRow - The section describing the layout, or NULL to use the default global section in dmRow
. dmCol - The DM for the column fields
. sectionCol - The section describing the layout, or NULL to use the default section in dmCol
. globalSectionCol - The section describing the layout, or NULL to use the default global section in dmCol
. A - The matrix
. point - The point in the DMs
. values - The array of values
- mode - The insert mode, where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions

  Level: intermediate

.seealso DMPlexMatSetClosure(), DMPlexVecGetClosure(), DMPlexVecSetClosure()
@*/
PetscErrorCode DMPlexMatSetClosureGeneral(DM dmRow, PetscSection sectionRow, PetscSection globalSectionRow, DM dmCol, PetscSection sectionCol, PetscSection globalSectionCol, Mat A, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  DM_Plex           *mesh = (DM_Plex*) dmRow->data;
  PetscInt          *indicesRow, *indicesCol;
  PetscInt           numIndicesRow, numIndicesCol;
  const PetscScalar *valuesOrig = values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmRow, DM_CLASSID, 1);
  if (!sectionRow) CHKERRQ(DMGetLocalSection(dmRow, &sectionRow));
  PetscValidHeaderSpecific(sectionRow, PETSC_SECTION_CLASSID, 2);
  if (!globalSectionRow) CHKERRQ(DMGetGlobalSection(dmRow, &globalSectionRow));
  PetscValidHeaderSpecific(globalSectionRow, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(dmCol, DM_CLASSID, 4);
  if (!sectionCol) CHKERRQ(DMGetLocalSection(dmCol, &sectionCol));
  PetscValidHeaderSpecific(sectionCol, PETSC_SECTION_CLASSID, 5);
  if (!globalSectionCol) CHKERRQ(DMGetGlobalSection(dmCol, &globalSectionCol));
  PetscValidHeaderSpecific(globalSectionCol, PETSC_SECTION_CLASSID, 6);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 7);

  CHKERRQ(DMPlexGetClosureIndices(dmRow, sectionRow, globalSectionRow, point, PETSC_TRUE, &numIndicesRow, &indicesRow, NULL, (PetscScalar **) &values));
  CHKERRQ(DMPlexGetClosureIndices(dmCol, sectionCol, globalSectionCol, point, PETSC_TRUE, &numIndicesCol, &indicesCol, NULL, (PetscScalar **) &values));

  if (mesh->printSetValues) CHKERRQ(DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numIndicesRow, indicesRow, numIndicesCol, indicesCol, values));
  ierr = MatSetValues(A, numIndicesRow, indicesRow, numIndicesCol, indicesCol, values, mode);
  if (ierr) {
    PetscMPIInt    rank;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    CHKERRQ((*PetscErrorPrintf)("[%d]ERROR in DMPlexMatSetClosure\n", rank));
    CHKERRQ(DMPlexPrintMatSetValues(PETSC_VIEWER_STDERR_SELF, A, point, numIndicesRow, indicesRow, numIndicesCol, indicesCol, values));
    CHKERRQ(DMPlexRestoreClosureIndices(dmRow, sectionRow, globalSectionRow, point, PETSC_TRUE, &numIndicesRow, &indicesRow, NULL, (PetscScalar **) &values));
    CHKERRQ(DMPlexRestoreClosureIndices(dmCol, sectionCol, globalSectionCol, point, PETSC_TRUE, &numIndicesCol, &indicesRow, NULL, (PetscScalar **) &values));
    if (values != valuesOrig) CHKERRQ(DMRestoreWorkArray(dmRow, 0, MPIU_SCALAR, &values));
    CHKERRQ(ierr);
  }

  CHKERRQ(DMPlexRestoreClosureIndices(dmRow, sectionRow, globalSectionRow, point, PETSC_TRUE, &numIndicesRow, &indicesRow, NULL, (PetscScalar **) &values));
  CHKERRQ(DMPlexRestoreClosureIndices(dmCol, sectionCol, globalSectionCol, point, PETSC_TRUE, &numIndicesCol, &indicesCol, NULL, (PetscScalar **) &values));
  if (values != valuesOrig) CHKERRQ(DMRestoreWorkArray(dmRow, 0, MPIU_SCALAR, &values));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexMatSetClosureRefined(DM dmf, PetscSection fsection, PetscSection globalFSection, DM dmc, PetscSection csection, PetscSection globalCSection, Mat A, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  DM_Plex        *mesh   = (DM_Plex*) dmf->data;
  PetscInt       *fpoints = NULL, *ftotpoints = NULL;
  PetscInt       *cpoints = NULL;
  PetscInt       *findices, *cindices;
  const PetscInt *fclperm = NULL, *cclperm = NULL; /* Closure permutations cannot work here */
  PetscInt        foffsets[32], coffsets[32];
  DMPolytopeType  ct;
  PetscInt        numFields, numSubcells, maxFPoints, numFPoints, numCPoints, numFIndices, numCIndices, dof, off, globalOff, pStart, pEnd, p, q, r, s, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 4);
  if (!fsection) CHKERRQ(DMGetLocalSection(dmf, &fsection));
  PetscValidHeaderSpecific(fsection, PETSC_SECTION_CLASSID, 2);
  if (!csection) CHKERRQ(DMGetLocalSection(dmc, &csection));
  PetscValidHeaderSpecific(csection, PETSC_SECTION_CLASSID, 5);
  if (!globalFSection) CHKERRQ(DMGetGlobalSection(dmf, &globalFSection));
  PetscValidHeaderSpecific(globalFSection, PETSC_SECTION_CLASSID, 3);
  if (!globalCSection) CHKERRQ(DMGetGlobalSection(dmc, &globalCSection));
  PetscValidHeaderSpecific(globalCSection, PETSC_SECTION_CLASSID, 6);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 7);
  CHKERRQ(PetscSectionGetNumFields(fsection, &numFields));
  PetscCheckFalse(numFields > 31,PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  CHKERRQ(PetscArrayzero(foffsets, 32));
  CHKERRQ(PetscArrayzero(coffsets, 32));
  /* Column indices */
  CHKERRQ(DMPlexGetTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints));
  maxFPoints = numCPoints;
  /* Compress out points not in the section */
  /*   TODO: Squeeze out points with 0 dof as well */
  CHKERRQ(PetscSectionGetChart(csection, &pStart, &pEnd));
  for (p = 0, q = 0; p < numCPoints*2; p += 2) {
    if ((cpoints[p] >= pStart) && (cpoints[p] < pEnd)) {
      cpoints[q*2]   = cpoints[p];
      cpoints[q*2+1] = cpoints[p+1];
      ++q;
    }
  }
  numCPoints = q;
  for (p = 0, numCIndices = 0; p < numCPoints*2; p += 2) {
    PetscInt fdof;

    CHKERRQ(PetscSectionGetDof(csection, cpoints[p], &dof));
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(csection, cpoints[p], f, &fdof));
      coffsets[f+1] += fdof;
    }
    numCIndices += dof;
  }
  for (f = 1; f < numFields; ++f) coffsets[f+1] += coffsets[f];
  /* Row indices */
  CHKERRQ(DMPlexGetCellType(dmc, point, &ct));
  {
    DMPlexTransform tr;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt, Nt;

    CHKERRQ(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
    CHKERRQ(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
    CHKERRQ(DMPlexTransformCellTransform(tr, ct, point, NULL, &Nt, &rct, &rsize, &rcone, &rornt));
    numSubcells = rsize[Nt-1];
    CHKERRQ(DMPlexTransformDestroy(&tr));
  }
  CHKERRQ(DMGetWorkArray(dmf, maxFPoints*2*numSubcells, MPIU_INT, &ftotpoints));
  for (r = 0, q = 0; r < numSubcells; ++r) {
    /* TODO Map from coarse to fine cells */
    CHKERRQ(DMPlexGetTransitiveClosure(dmf, point*numSubcells + r, PETSC_TRUE, &numFPoints, &fpoints));
    /* Compress out points not in the section */
    CHKERRQ(PetscSectionGetChart(fsection, &pStart, &pEnd));
    for (p = 0; p < numFPoints*2; p += 2) {
      if ((fpoints[p] >= pStart) && (fpoints[p] < pEnd)) {
        CHKERRQ(PetscSectionGetDof(fsection, fpoints[p], &dof));
        if (!dof) continue;
        for (s = 0; s < q; ++s) if (fpoints[p] == ftotpoints[s*2]) break;
        if (s < q) continue;
        ftotpoints[q*2]   = fpoints[p];
        ftotpoints[q*2+1] = fpoints[p+1];
        ++q;
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dmf, point, PETSC_TRUE, &numFPoints, &fpoints));
  }
  numFPoints = q;
  for (p = 0, numFIndices = 0; p < numFPoints*2; p += 2) {
    PetscInt fdof;

    CHKERRQ(PetscSectionGetDof(fsection, ftotpoints[p], &dof));
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(fsection, ftotpoints[p], f, &fdof));
      foffsets[f+1] += fdof;
    }
    numFIndices += dof;
  }
  for (f = 1; f < numFields; ++f) foffsets[f+1] += foffsets[f];

  PetscCheckFalse(numFields && foffsets[numFields] != numFIndices,PetscObjectComm((PetscObject)dmf), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", foffsets[numFields], numFIndices);
  PetscCheckFalse(numFields && coffsets[numFields] != numCIndices,PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", coffsets[numFields], numCIndices);
  CHKERRQ(DMGetWorkArray(dmf, numFIndices, MPIU_INT, &findices));
  CHKERRQ(DMGetWorkArray(dmc, numCIndices, MPIU_INT, &cindices));
  if (numFields) {
    const PetscInt **permsF[32] = {NULL};
    const PetscInt **permsC[32] = {NULL};

    for (f = 0; f < numFields; f++) {
      CHKERRQ(PetscSectionGetFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL));
      CHKERRQ(PetscSectionGetFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL));
    }
    for (p = 0; p < numFPoints; p++) {
      CHKERRQ(PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, foffsets, PETSC_FALSE, permsF, p, fclperm, findices));
    }
    for (p = 0; p < numCPoints; p++) {
      CHKERRQ(PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, coffsets, PETSC_FALSE, permsC, p, cclperm, cindices));
    }
    for (f = 0; f < numFields; f++) {
      CHKERRQ(PetscSectionRestoreFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL));
      CHKERRQ(PetscSectionRestoreFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL));
    }
  } else {
    const PetscInt **permsF = NULL;
    const PetscInt **permsC = NULL;

    CHKERRQ(PetscSectionGetPointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL));
    CHKERRQ(PetscSectionGetPointSyms(csection,numCPoints,cpoints,&permsC,NULL));
    for (p = 0, off = 0; p < numFPoints; p++) {
      const PetscInt *perm = permsF ? permsF[p] : NULL;

      CHKERRQ(PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPoint_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, fclperm, findices));
    }
    for (p = 0, off = 0; p < numCPoints; p++) {
      const PetscInt *perm = permsC ? permsC[p] : NULL;

      CHKERRQ(PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPoint_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, cclperm, cindices));
    }
    CHKERRQ(PetscSectionRestorePointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL));
    CHKERRQ(PetscSectionRestorePointSyms(csection,numCPoints,cpoints,&permsC,NULL));
  }
  if (mesh->printSetValues) CHKERRQ(DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numFIndices, findices, numCIndices, cindices, values));
  /* TODO: flips */
  ierr = MatSetValues(A, numFIndices, findices, numCIndices, cindices, values, mode);
  if (ierr) {
    PetscMPIInt    rank;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    CHKERRQ((*PetscErrorPrintf)("[%d]ERROR in DMPlexMatSetClosure\n", rank));
    CHKERRQ(DMPlexPrintMatSetValues(PETSC_VIEWER_STDERR_SELF, A, point, numFIndices, findices, numCIndices, cindices, values));
    CHKERRQ(DMRestoreWorkArray(dmf, numFIndices, MPIU_INT, &findices));
    CHKERRQ(DMRestoreWorkArray(dmc, numCIndices, MPIU_INT, &cindices));
    CHKERRQ(ierr);
  }
  CHKERRQ(DMRestoreWorkArray(dmf, numCPoints*2*4, MPIU_INT, &ftotpoints));
  CHKERRQ(DMPlexRestoreTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints));
  CHKERRQ(DMRestoreWorkArray(dmf, numFIndices, MPIU_INT, &findices));
  CHKERRQ(DMRestoreWorkArray(dmc, numCIndices, MPIU_INT, &cindices));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexMatGetClosureIndicesRefined(DM dmf, PetscSection fsection, PetscSection globalFSection, DM dmc, PetscSection csection, PetscSection globalCSection, PetscInt point, PetscInt cindices[], PetscInt findices[])
{
  PetscInt      *fpoints = NULL, *ftotpoints = NULL;
  PetscInt      *cpoints = NULL;
  PetscInt       foffsets[32], coffsets[32];
  const PetscInt *fclperm = NULL, *cclperm = NULL; /* Closure permutations cannot work here */
  DMPolytopeType ct;
  PetscInt       numFields, numSubcells, maxFPoints, numFPoints, numCPoints, numFIndices, numCIndices, dof, off, globalOff, pStart, pEnd, p, q, r, s, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 4);
  if (!fsection) CHKERRQ(DMGetLocalSection(dmf, &fsection));
  PetscValidHeaderSpecific(fsection, PETSC_SECTION_CLASSID, 2);
  if (!csection) CHKERRQ(DMGetLocalSection(dmc, &csection));
  PetscValidHeaderSpecific(csection, PETSC_SECTION_CLASSID, 5);
  if (!globalFSection) CHKERRQ(DMGetGlobalSection(dmf, &globalFSection));
  PetscValidHeaderSpecific(globalFSection, PETSC_SECTION_CLASSID, 3);
  if (!globalCSection) CHKERRQ(DMGetGlobalSection(dmc, &globalCSection));
  PetscValidHeaderSpecific(globalCSection, PETSC_SECTION_CLASSID, 6);
  CHKERRQ(PetscSectionGetNumFields(fsection, &numFields));
  PetscCheckFalse(numFields > 31,PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  CHKERRQ(PetscArrayzero(foffsets, 32));
  CHKERRQ(PetscArrayzero(coffsets, 32));
  /* Column indices */
  CHKERRQ(DMPlexGetTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints));
  maxFPoints = numCPoints;
  /* Compress out points not in the section */
  /*   TODO: Squeeze out points with 0 dof as well */
  CHKERRQ(PetscSectionGetChart(csection, &pStart, &pEnd));
  for (p = 0, q = 0; p < numCPoints*2; p += 2) {
    if ((cpoints[p] >= pStart) && (cpoints[p] < pEnd)) {
      cpoints[q*2]   = cpoints[p];
      cpoints[q*2+1] = cpoints[p+1];
      ++q;
    }
  }
  numCPoints = q;
  for (p = 0, numCIndices = 0; p < numCPoints*2; p += 2) {
    PetscInt fdof;

    CHKERRQ(PetscSectionGetDof(csection, cpoints[p], &dof));
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(csection, cpoints[p], f, &fdof));
      coffsets[f+1] += fdof;
    }
    numCIndices += dof;
  }
  for (f = 1; f < numFields; ++f) coffsets[f+1] += coffsets[f];
  /* Row indices */
  CHKERRQ(DMPlexGetCellType(dmc, point, &ct));
  {
    DMPlexTransform tr;
    DMPolytopeType *rct;
    PetscInt       *rsize, *rcone, *rornt, Nt;

    CHKERRQ(DMPlexTransformCreate(PETSC_COMM_SELF, &tr));
    CHKERRQ(DMPlexTransformSetType(tr, DMPLEXREFINEREGULAR));
    CHKERRQ(DMPlexTransformCellTransform(tr, ct, point, NULL, &Nt, &rct, &rsize, &rcone, &rornt));
    numSubcells = rsize[Nt-1];
    CHKERRQ(DMPlexTransformDestroy(&tr));
  }
  CHKERRQ(DMGetWorkArray(dmf, maxFPoints*2*numSubcells, MPIU_INT, &ftotpoints));
  for (r = 0, q = 0; r < numSubcells; ++r) {
    /* TODO Map from coarse to fine cells */
    CHKERRQ(DMPlexGetTransitiveClosure(dmf, point*numSubcells + r, PETSC_TRUE, &numFPoints, &fpoints));
    /* Compress out points not in the section */
    CHKERRQ(PetscSectionGetChart(fsection, &pStart, &pEnd));
    for (p = 0; p < numFPoints*2; p += 2) {
      if ((fpoints[p] >= pStart) && (fpoints[p] < pEnd)) {
        CHKERRQ(PetscSectionGetDof(fsection, fpoints[p], &dof));
        if (!dof) continue;
        for (s = 0; s < q; ++s) if (fpoints[p] == ftotpoints[s*2]) break;
        if (s < q) continue;
        ftotpoints[q*2]   = fpoints[p];
        ftotpoints[q*2+1] = fpoints[p+1];
        ++q;
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dmf, point, PETSC_TRUE, &numFPoints, &fpoints));
  }
  numFPoints = q;
  for (p = 0, numFIndices = 0; p < numFPoints*2; p += 2) {
    PetscInt fdof;

    CHKERRQ(PetscSectionGetDof(fsection, ftotpoints[p], &dof));
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      CHKERRQ(PetscSectionGetFieldDof(fsection, ftotpoints[p], f, &fdof));
      foffsets[f+1] += fdof;
    }
    numFIndices += dof;
  }
  for (f = 1; f < numFields; ++f) foffsets[f+1] += foffsets[f];

  PetscCheckFalse(numFields && foffsets[numFields] != numFIndices,PetscObjectComm((PetscObject)dmf), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", foffsets[numFields], numFIndices);
  PetscCheckFalse(numFields && coffsets[numFields] != numCIndices,PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", coffsets[numFields], numCIndices);
  if (numFields) {
    const PetscInt **permsF[32] = {NULL};
    const PetscInt **permsC[32] = {NULL};

    for (f = 0; f < numFields; f++) {
      CHKERRQ(PetscSectionGetFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL));
      CHKERRQ(PetscSectionGetFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL));
    }
    for (p = 0; p < numFPoints; p++) {
      CHKERRQ(PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, foffsets, PETSC_FALSE, permsF, p, fclperm, findices));
    }
    for (p = 0; p < numCPoints; p++) {
      CHKERRQ(PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, coffsets, PETSC_FALSE, permsC, p, cclperm, cindices));
    }
    for (f = 0; f < numFields; f++) {
      CHKERRQ(PetscSectionRestoreFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL));
      CHKERRQ(PetscSectionRestoreFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL));
    }
  } else {
    const PetscInt **permsF = NULL;
    const PetscInt **permsC = NULL;

    CHKERRQ(PetscSectionGetPointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL));
    CHKERRQ(PetscSectionGetPointSyms(csection,numCPoints,cpoints,&permsC,NULL));
    for (p = 0, off = 0; p < numFPoints; p++) {
      const PetscInt *perm = permsF ? permsF[p] : NULL;

      CHKERRQ(PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPoint_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, fclperm, findices));
    }
    for (p = 0, off = 0; p < numCPoints; p++) {
      const PetscInt *perm = permsC ? permsC[p] : NULL;

      CHKERRQ(PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff));
      CHKERRQ(DMPlexGetIndicesPoint_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, cclperm, cindices));
    }
    CHKERRQ(PetscSectionRestorePointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL));
    CHKERRQ(PetscSectionRestorePointSyms(csection,numCPoints,cpoints,&permsC,NULL));
  }
  CHKERRQ(DMRestoreWorkArray(dmf, numCPoints*2*4, MPIU_INT, &ftotpoints));
  CHKERRQ(DMPlexRestoreTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetVTKCellHeight - Returns the height in the DAG used to determine which points are cells (normally 0)

  Input Parameter:
. dm   - The DMPlex object

  Output Parameter:
. cellHeight - The height of a cell

  Level: developer

.seealso DMPlexSetVTKCellHeight()
@*/
PetscErrorCode DMPlexGetVTKCellHeight(DM dm, PetscInt *cellHeight)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cellHeight, 2);
  *cellHeight = mesh->vtkCellHeight;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexSetVTKCellHeight - Sets the height in the DAG used to determine which points are cells (normally 0)

  Input Parameters:
+ dm   - The DMPlex object
- cellHeight - The height of a cell

  Level: developer

.seealso DMPlexGetVTKCellHeight()
@*/
PetscErrorCode DMPlexSetVTKCellHeight(DM dm, PetscInt cellHeight)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->vtkCellHeight = cellHeight;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGhostCellStratum - Get the range of cells which are used to enforce FV boundary conditions

  Input Parameter:
. dm - The DMPlex object

  Output Parameters:
+ gcStart - The first ghost cell, or NULL
- gcEnd   - The upper bound on ghost cells, or NULL

  Level: advanced

.seealso DMPlexConstructGhostCells(), DMPlexGetGhostCellStratum()
@*/
PetscErrorCode DMPlexGetGhostCellStratum(DM dm, PetscInt *gcStart, PetscInt *gcEnd)
{
  DMLabel        ctLabel;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &ctLabel));
  CHKERRQ(DMLabelGetStratumBounds(ctLabel, DM_POLYTOPE_FV_GHOST, gcStart, gcEnd));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateNumbering_Plex(DM dm, PetscInt pStart, PetscInt pEnd, PetscInt shift, PetscInt *globalSize, PetscSF sf, IS *numbering)
{
  PetscSection   section, globalSection;
  PetscInt      *numbers, p;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  CHKERRQ(PetscSectionSetChart(section, pStart, pEnd));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(PetscSectionSetDof(section, p, 1));
  }
  CHKERRQ(PetscSectionSetUp(section));
  CHKERRQ(PetscSectionCreateGlobalSection(section, sf, PETSC_FALSE, PETSC_FALSE, &globalSection));
  CHKERRQ(PetscMalloc1(pEnd - pStart, &numbers));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetOffset(globalSection, p, &numbers[p-pStart]));
    if (numbers[p-pStart] < 0) numbers[p-pStart] -= shift;
    else                       numbers[p-pStart] += shift;
  }
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd - pStart, numbers, PETSC_OWN_POINTER, numbering));
  if (globalSize) {
    PetscLayout layout;
    CHKERRQ(PetscSectionGetPointLayout(PetscObjectComm((PetscObject) dm), globalSection, &layout));
    CHKERRQ(PetscLayoutGetSize(layout, globalSize));
    CHKERRQ(PetscLayoutDestroy(&layout));
  }
  CHKERRQ(PetscSectionDestroy(&section));
  CHKERRQ(PetscSectionDestroy(&globalSection));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateCellNumbering_Internal(DM dm, PetscBool includeHybrid, IS *globalCellNumbers)
{
  PetscInt       cellHeight, cStart, cEnd;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  if (includeHybrid) CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  else               CHKERRQ(DMPlexGetSimplexOrBoxCells(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMPlexCreateNumbering_Plex(dm, cStart, cEnd, 0, NULL, dm->sf, globalCellNumbers));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetCellNumbering - Get a global cell numbering for all cells on this process

  Input Parameter:
. dm   - The DMPlex object

  Output Parameter:
. globalCellNumbers - Global cell numbers for all cells on this process

  Level: developer

.seealso DMPlexGetVertexNumbering()
@*/
PetscErrorCode DMPlexGetCellNumbering(DM dm, IS *globalCellNumbers)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalCellNumbers) CHKERRQ(DMPlexCreateCellNumbering_Internal(dm, PETSC_FALSE, &mesh->globalCellNumbers));
  *globalCellNumbers = mesh->globalCellNumbers;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateVertexNumbering_Internal(DM dm, PetscBool includeHybrid, IS *globalVertexNumbers)
{
  PetscInt       vStart, vEnd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexCreateNumbering_Plex(dm, vStart, vEnd, 0, NULL, dm->sf, globalVertexNumbers));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetVertexNumbering - Get a global vertex numbering for all vertices on this process

  Input Parameter:
. dm   - The DMPlex object

  Output Parameter:
. globalVertexNumbers - Global vertex numbers for all vertices on this process

  Level: developer

.seealso DMPlexGetCellNumbering()
@*/
PetscErrorCode DMPlexGetVertexNumbering(DM dm, IS *globalVertexNumbers)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalVertexNumbers) CHKERRQ(DMPlexCreateVertexNumbering_Internal(dm, PETSC_FALSE, &mesh->globalVertexNumbers));
  *globalVertexNumbers = mesh->globalVertexNumbers;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreatePointNumbering - Create a global numbering for all points on this process

  Input Parameter:
. dm   - The DMPlex object

  Output Parameter:
. globalPointNumbers - Global numbers for all points on this process

  Level: developer

.seealso DMPlexGetCellNumbering()
@*/
PetscErrorCode DMPlexCreatePointNumbering(DM dm, IS *globalPointNumbers)
{
  IS             nums[4];
  PetscInt       depths[4], gdepths[4], starts[4];
  PetscInt       depth, d, shift = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  /* For unstratified meshes use dim instead of depth */
  if (depth < 0) CHKERRQ(DMGetDimension(dm, &depth));
  for (d = 0; d <= depth; ++d) {
    PetscInt end;

    depths[d] = depth-d;
    CHKERRQ(DMPlexGetDepthStratum(dm, depths[d], &starts[d], &end));
    if (!(starts[d]-end)) { starts[d] = depths[d] = -1; }
  }
  CHKERRQ(PetscSortIntWithArray(depth+1, starts, depths));
  CHKERRMPI(MPIU_Allreduce(depths, gdepths, depth+1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm)));
  for (d = 0; d <= depth; ++d) {
    PetscCheckFalse(starts[d] >= 0 && depths[d] != gdepths[d],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected depth %D, found %D",depths[d],gdepths[d]);
  }
  for (d = 0; d <= depth; ++d) {
    PetscInt pStart, pEnd, gsize;

    CHKERRQ(DMPlexGetDepthStratum(dm, gdepths[d], &pStart, &pEnd));
    CHKERRQ(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, shift, &gsize, dm->sf, &nums[d]));
    shift += gsize;
  }
  CHKERRQ(ISConcatenate(PetscObjectComm((PetscObject) dm), depth+1, nums, globalPointNumbers));
  for (d = 0; d <= depth; ++d) CHKERRQ(ISDestroy(&nums[d]));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateRankField - Create a cell field whose value is the rank of the owner

  Input Parameter:
. dm - The DMPlex object

  Output Parameter:
. ranks - The rank field

  Options Database Keys:
. -dm_partition_view - Adds the rank field into the DM output from -dm_view using the same viewer

  Level: intermediate

.seealso: DMView()
@*/
PetscErrorCode DMPlexCreateRankField(DM dm, Vec *ranks)
{
  DM             rdm;
  PetscFE        fe;
  PetscScalar   *r;
  PetscMPIInt    rank;
  DMPolytopeType ct;
  PetscInt       dim, cStart, cEnd, c;
  PetscBool      simplex;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ranks, 2);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMClone(dm, &rdm));
  CHKERRQ(DMGetDimension(rdm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "PETSc___rank_", -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "rank"));
  CHKERRQ(DMSetField(rdm, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(rdm));
  CHKERRQ(DMCreateGlobalVector(rdm, ranks));
  CHKERRQ(PetscObjectSetName((PetscObject) *ranks, "partition"));
  CHKERRQ(VecGetArray(*ranks, &r));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *lr;

    CHKERRQ(DMPlexPointGlobalRef(rdm, c, r, &lr));
    if (lr) *lr = rank;
  }
  CHKERRQ(VecRestoreArray(*ranks, &r));
  CHKERRQ(DMDestroy(&rdm));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateLabelField - Create a cell field whose value is the label value for that cell

  Input Parameters:
+ dm    - The DMPlex
- label - The DMLabel

  Output Parameter:
. val - The label value field

  Options Database Keys:
. -dm_label_view - Adds the label value field into the DM output from -dm_view using the same viewer

  Level: intermediate

.seealso: DMView()
@*/
PetscErrorCode DMPlexCreateLabelField(DM dm, DMLabel label, Vec *val)
{
  DM             rdm;
  PetscFE        fe;
  PetscScalar   *v;
  PetscInt       dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 2);
  PetscValidPointer(val, 3);
  CHKERRQ(DMClone(dm, &rdm));
  CHKERRQ(DMGetDimension(rdm, &dim));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) rdm), dim, 1, PETSC_TRUE, "PETSc___label_value_", -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "label_value"));
  CHKERRQ(DMSetField(rdm, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(rdm));
  CHKERRQ(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
  CHKERRQ(DMCreateGlobalVector(rdm, val));
  CHKERRQ(PetscObjectSetName((PetscObject) *val, "label_value"));
  CHKERRQ(VecGetArray(*val, &v));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *lv;
    PetscInt     cval;

    CHKERRQ(DMPlexPointGlobalRef(rdm, c, v, &lv));
    CHKERRQ(DMLabelGetValue(label, c, &cval));
    *lv = cval;
  }
  CHKERRQ(VecRestoreArray(*val, &v));
  CHKERRQ(DMDestroy(&rdm));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckSymmetry - Check that the adjacency information in the mesh is symmetric.

  Input Parameter:
. dm - The DMPlex object

  Notes:
  This is a useful diagnostic when creating meshes programmatically.

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Level: developer

.seealso: DMCreate(), DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckSymmetry(DM dm)
{
  PetscSection    coneSection, supportSection;
  const PetscInt *cone, *support;
  PetscInt        coneSize, c, supportSize, s;
  PetscInt        pStart, pEnd, p, pp, csize, ssize;
  PetscBool       storagecheck = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMViewFromOptions(dm, NULL, "-sym_dm_view"));
  CHKERRQ(DMPlexGetConeSection(dm, &coneSection));
  CHKERRQ(DMPlexGetSupportSection(dm, &supportSection));
  /* Check that point p is found in the support of its cone points, and vice versa */
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, p, &cone));
    for (c = 0; c < coneSize; ++c) {
      PetscBool dup = PETSC_FALSE;
      PetscInt  d;
      for (d = c-1; d >= 0; --d) {
        if (cone[c] == cone[d]) {dup = PETSC_TRUE; break;}
      }
      CHKERRQ(DMPlexGetSupportSize(dm, cone[c], &supportSize));
      CHKERRQ(DMPlexGetSupport(dm, cone[c], &support));
      for (s = 0; s < supportSize; ++s) {
        if (support[s] == p) break;
      }
      if ((s >= supportSize) || (dup && (support[s+1] != p))) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "p: %D cone: ", p));
        for (s = 0; s < coneSize; ++s) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, ", cone[s]));
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "p: %D support: ", cone[c]));
        for (s = 0; s < supportSize; ++s) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, ", support[s]));
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
        PetscCheck(!dup,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not repeatedly found in support of repeated cone point %D", p, cone[c]);
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not found in support of cone point %D", p, cone[c]);
      }
    }
    CHKERRQ(DMPlexGetTreeParent(dm, p, &pp, NULL));
    if (p != pp) { storagecheck = PETSC_FALSE; continue; }
    CHKERRQ(DMPlexGetSupportSize(dm, p, &supportSize));
    CHKERRQ(DMPlexGetSupport(dm, p, &support));
    for (s = 0; s < supportSize; ++s) {
      CHKERRQ(DMPlexGetConeSize(dm, support[s], &coneSize));
      CHKERRQ(DMPlexGetCone(dm, support[s], &cone));
      for (c = 0; c < coneSize; ++c) {
        CHKERRQ(DMPlexGetTreeParent(dm, cone[c], &pp, NULL));
        if (cone[c] != pp) { c = 0; break; }
        if (cone[c] == p) break;
      }
      if (c >= coneSize) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "p: %D support: ", p));
        for (c = 0; c < supportSize; ++c) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, ", support[c]));
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "p: %D cone: ", support[s]));
        for (c = 0; c < coneSize; ++c) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, ", cone[c]));
        }
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not found in cone of support point %D", p, support[s]);
      }
    }
  }
  if (storagecheck) {
    CHKERRQ(PetscSectionGetStorageSize(coneSection, &csize));
    CHKERRQ(PetscSectionGetStorageSize(supportSection, &ssize));
    PetscCheckFalse(csize != ssize,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Total cone size %D != Total support size %D", csize, ssize);
  }
  PetscFunctionReturn(0);
}

/*
  For submeshes with cohesive cells (see DMPlexConstructCohesiveCells()), we allow a special case where some of the boundary of a face (edges and vertices) are not duplicated. We call these special boundary points "unsplit", since the same edge or vertex appears in both copies of the face. These unsplit points throw off our counting, so we have to explicitly account for them here.
*/
static PetscErrorCode DMPlexCellUnsplitVertices_Private(DM dm, PetscInt c, DMPolytopeType ct, PetscInt *unsplit)
{
  DMPolytopeType  cct;
  PetscInt        ptpoints[4];
  const PetscInt *cone, *ccone, *ptcone;
  PetscInt        coneSize, cp, cconeSize, ccp, npt = 0, pt;

  PetscFunctionBegin;
  *unsplit = 0;
  switch (ct) {
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      ptpoints[npt++] = c;
      break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
      CHKERRQ(DMPlexGetCone(dm, c, &cone));
      CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
      for (cp = 0; cp < coneSize; ++cp) {
        CHKERRQ(DMPlexGetCellType(dm, cone[cp], &cct));
        if (cct == DM_POLYTOPE_POINT_PRISM_TENSOR) ptpoints[npt++] = cone[cp];
      }
      break;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      CHKERRQ(DMPlexGetCone(dm, c, &cone));
      CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
      for (cp = 0; cp < coneSize; ++cp) {
        CHKERRQ(DMPlexGetCone(dm, cone[cp], &ccone));
        CHKERRQ(DMPlexGetConeSize(dm, cone[cp], &cconeSize));
        for (ccp = 0; ccp < cconeSize; ++ccp) {
          CHKERRQ(DMPlexGetCellType(dm, ccone[ccp], &cct));
          if (cct == DM_POLYTOPE_POINT_PRISM_TENSOR) {
            PetscInt p;
            for (p = 0; p < npt; ++p) if (ptpoints[p] == ccone[ccp]) break;
            if (p == npt) ptpoints[npt++] = ccone[ccp];
          }
        }
      }
      break;
    default: break;
  }
  for (pt = 0; pt < npt; ++pt) {
    CHKERRQ(DMPlexGetCone(dm, ptpoints[pt], &ptcone));
    if (ptcone[0] == ptcone[1]) ++(*unsplit);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckSkeleton - Check that each cell has the correct number of vertices

  Input Parameters:
+ dm - The DMPlex object
- cellHeight - Normally 0

  Notes:
  This is a useful diagnostic when creating meshes programmatically.
  Currently applicable only to homogeneous simplex or tensor meshes.

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Level: developer

.seealso: DMCreate(), DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckSkeleton(DM dm, PetscInt cellHeight)
{
  DMPlexInterpolatedFlag interp;
  DMPolytopeType         ct;
  PetscInt               vStart, vEnd, cStart, cEnd, c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexIsInterpolated(dm, &interp));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  coneSize, closureSize, cl, Nv = 0;

    CHKERRQ(DMPlexGetCellType(dm, c, &ct));
    PetscCheckFalse((PetscInt) ct < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %D has no cell type", c);
    if (ct == DM_POLYTOPE_UNKNOWN) continue;
    if (interp == DMPLEX_INTERPOLATED_FULL) {
      CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
      PetscCheckFalse(coneSize != DMPolytopeTypeGetConeSize(ct),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %D of type %s has cone size %D != %D", c, DMPolytopeTypes[ct], coneSize, DMPolytopeTypeGetConeSize(ct));
    }
    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];
      if ((p >= vStart) && (p < vEnd)) ++Nv;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    /* Special Case: Tensor faces with identified vertices */
    if (Nv < DMPolytopeTypeGetNumVertices(ct)) {
      PetscInt unsplit;

      CHKERRQ(DMPlexCellUnsplitVertices_Private(dm, c, ct, &unsplit));
      if (Nv + unsplit == DMPolytopeTypeGetNumVertices(ct)) continue;
    }
    PetscCheckFalse(Nv != DMPolytopeTypeGetNumVertices(ct),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %D of type %s has %D vertices != %D", c, DMPolytopeTypes[ct], Nv, DMPolytopeTypeGetNumVertices(ct));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckFaces - Check that the faces of each cell give a vertex order this is consistent with what we expect from the cell type

  Not Collective

  Input Parameters:
+ dm - The DMPlex object
- cellHeight - Normally 0

  Notes:
  This is a useful diagnostic when creating meshes programmatically.
  This routine is only relevant for meshes that are fully interpolated across all ranks.
  It will error out if a partially interpolated mesh is given on some rank.
  It will do nothing for locally uninterpolated mesh (as there is nothing to check).

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Level: developer

.seealso: DMCreate(), DMPlexGetVTKCellHeight(), DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckFaces(DM dm, PetscInt cellHeight)
{
  PetscInt       dim, depth, vStart, vEnd, cStart, cEnd, c, h;
  DMPlexInterpolatedFlag interpEnum;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexIsInterpolated(dm, &interpEnum));
  if (interpEnum == DMPLEX_INTERPOLATED_NONE) PetscFunctionReturn(0);
  if (interpEnum == DMPLEX_INTERPOLATED_PARTIAL) {
    PetscMPIInt rank;
    MPI_Comm    comm;

    CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
    CHKERRMPI(MPI_Comm_rank(comm, &rank));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mesh is only partially interpolated on rank %d, this is currently not supported", rank);
  }

  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (h = cellHeight; h < PetscMin(depth, dim); ++h) {
    CHKERRQ(DMPlexGetHeightStratum(dm, h, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt      *cone, *ornt, *faceSizes, *faces;
      const DMPolytopeType *faceTypes;
      DMPolytopeType        ct;
      PetscInt              numFaces, coneSize, f;
      PetscInt             *closure = NULL, closureSize, cl, numCorners = 0, fOff = 0, unsplit;

      CHKERRQ(DMPlexGetCellType(dm, c, &ct));
      CHKERRQ(DMPlexCellUnsplitVertices_Private(dm, c, ct, &unsplit));
      if (unsplit) continue;
      CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
      CHKERRQ(DMPlexGetCone(dm, c, &cone));
      CHKERRQ(DMPlexGetConeOrientation(dm, c, &ornt));
      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt p = closure[cl];
        if ((p >= vStart) && (p < vEnd)) closure[numCorners++] = p;
      }
      CHKERRQ(DMPlexGetRawFaces_Internal(dm, ct, closure, &numFaces, &faceTypes, &faceSizes, &faces));
      PetscCheckFalse(coneSize != numFaces,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %D of type %s has %D faces but should have %D", c, DMPolytopeTypes[ct], coneSize, numFaces);
      for (f = 0; f < numFaces; ++f) {
        DMPolytopeType fct;
        PetscInt       *fclosure = NULL, fclosureSize, cl, fnumCorners = 0, v;

        CHKERRQ(DMPlexGetCellType(dm, cone[f], &fct));
        CHKERRQ(DMPlexGetTransitiveClosure_Internal(dm, cone[f], ornt[f], PETSC_TRUE, &fclosureSize, &fclosure));
        for (cl = 0; cl < fclosureSize*2; cl += 2) {
          const PetscInt p = fclosure[cl];
          if ((p >= vStart) && (p < vEnd)) fclosure[fnumCorners++] = p;
        }
        PetscCheckFalse(fnumCorners != faceSizes[f],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D of type %s (cone idx %D) of cell %D of type %s has %D vertices but should have %D", cone[f], DMPolytopeTypes[fct], f, c, DMPolytopeTypes[ct], fnumCorners, faceSizes[f]);
        for (v = 0; v < fnumCorners; ++v) {
          if (fclosure[v] != faces[fOff+v]) {
            PetscInt v1;

            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "face closure:"));
            for (v1 = 0; v1 < fnumCorners; ++v1) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %D", fclosure[v1]));
            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\ncell face:"));
            for (v1 = 0; v1 < fnumCorners; ++v1) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %D", faces[fOff+v1]));
            CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D of type %s (cone idx %d, ornt %D) of cell %D of type %s vertex %D, %D != %D", cone[f], DMPolytopeTypes[fct], f, ornt[f], c, DMPolytopeTypes[ct], v, fclosure[v], faces[fOff+v]);
          }
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cone[f], PETSC_TRUE, &fclosureSize, &fclosure));
        fOff += faceSizes[f];
      }
      CHKERRQ(DMPlexRestoreRawFaces_Internal(dm, ct, closure, &numFaces, &faceTypes, &faceSizes, &faces));
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckGeometry - Check the geometry of mesh cells

  Input Parameter:
. dm - The DMPlex object

  Notes:
  This is a useful diagnostic when creating meshes programmatically.

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Level: developer

.seealso: DMCreate(), DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckGeometry(DM dm)
{
  Vec            coordinates;
  PetscReal      detJ, J[9], refVol = 1.0;
  PetscReal      vol;
  PetscBool      periodic;
  PetscInt       dim, depth, dE, d, cStart, cEnd, c;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateDim(dm, &dE));
  if (dim != dE) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetPeriodicity(dm, &periodic, NULL, NULL, NULL));
  for (d = 0; d < dim; ++d) refVol *= 2.0;
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* Make sure local coordinates are created, because that step is collective */
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  for (c = cStart; c < cEnd; ++c) {
    DMPolytopeType ct;
    PetscInt       unsplit;
    PetscBool      ignoreZeroVol = PETSC_FALSE;

    CHKERRQ(DMPlexGetCellType(dm, c, &ct));
    switch (ct) {
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
      case DM_POLYTOPE_TRI_PRISM_TENSOR:
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
        ignoreZeroVol = PETSC_TRUE; break;
      default: break;
    }
    switch (ct) {
      case DM_POLYTOPE_TRI_PRISM:
      case DM_POLYTOPE_TRI_PRISM_TENSOR:
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      case DM_POLYTOPE_PYRAMID:
        continue;
      default: break;
    }
    CHKERRQ(DMPlexCellUnsplitVertices_Private(dm, c, ct, &unsplit));
    if (unsplit) continue;
    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ));
    PetscCheckFalse(detJ < -PETSC_SMALL || (detJ <= 0.0 && !ignoreZeroVol),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D of type %s is inverted, |J| = %g", c, DMPolytopeTypes[ct], (double) detJ);
    CHKERRQ(PetscInfo(dm, "Cell %D FEM Volume %g\n", c, (double) detJ*refVol));
    if (depth > 1 && !periodic) {
      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
      PetscCheckFalse(vol < -PETSC_SMALL || (vol <= 0.0 && !ignoreZeroVol),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D of type %s is inverted, vol = %g", c, DMPolytopeTypes[ct], (double) vol);
      CHKERRQ(PetscInfo(dm, "Cell %D FVM Volume %g\n", c, (double) vol));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckPointSF - Check that several necessary conditions are met for the point SF of this plex.

  Input Parameters:
. dm - The DMPlex object

  Notes:
  This is mainly intended for debugging/testing purposes.
  It currently checks only meshes with no partition overlapping.

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Level: developer

.seealso: DMGetPointSF(), DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckPointSF(DM dm)
{
  PetscSF         pointSF;
  PetscInt        cellHeight, cStart, cEnd, l, nleaves, nroots, overlap;
  const PetscInt *locals, *rootdegree;
  PetscBool       distributed;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMGetPointSF(dm, &pointSF));
  CHKERRQ(DMPlexIsDistributed(dm, &distributed));
  if (!distributed) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetOverlap(dm, &overlap));
  if (overlap) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)dm), "Warning: DMPlexCheckPointSF() is currently not implemented for meshes with partition overlapping"));
    PetscFunctionReturn(0);
  }
  PetscCheck(pointSF,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but does not have PointSF attached");
  CHKERRQ(PetscSFGetGraph(pointSF, &nroots, &nleaves, &locals, NULL));
  PetscCheckFalse(nroots < 0,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but its PointSF has no graph set");
  CHKERRQ(PetscSFComputeDegreeBegin(pointSF, &rootdegree));
  CHKERRQ(PetscSFComputeDegreeEnd(pointSF, &rootdegree));

  /* 1) check there are no faces in 2D, cells in 3D, in interface */
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  for (l = 0; l < nleaves; ++l) {
    const PetscInt point = locals[l];

    PetscCheckFalse(point >= cStart && point < cEnd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D which is a cell", point);
  }

  /* 2) if some point is in interface, then all its cone points must be also in interface (either as leaves or roots) */
  for (l = 0; l < nleaves; ++l) {
    const PetscInt  point = locals[l];
    const PetscInt *cone;
    PetscInt        coneSize, c, idx;

    CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
    CHKERRQ(DMPlexGetCone(dm, point, &cone));
    for (c = 0; c < coneSize; ++c) {
      if (!rootdegree[cone[c]]) {
        CHKERRQ(PetscFindInt(cone[c], nleaves, locals, &idx));
        PetscCheckFalse(idx < 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D but not %D from its cone", point, cone[c]);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCheckAll_Internal(DM dm, PetscInt cellHeight)
{
  PetscFunctionBegin;
  CHKERRQ(DMPlexCheckSymmetry(dm));
  CHKERRQ(DMPlexCheckSkeleton(dm, cellHeight));
  CHKERRQ(DMPlexCheckFaces(dm, cellHeight));
  CHKERRQ(DMPlexCheckGeometry(dm));
  CHKERRQ(DMPlexCheckPointSF(dm));
  CHKERRQ(DMPlexCheckInterfaceCones(dm));
  PetscFunctionReturn(0);
}

typedef struct cell_stats
{
  PetscReal min, max, sum, squaresum;
  PetscInt  count;
} cell_stats_t;

static void MPIAPI cell_stats_reduce(void *a, void *b, int * len, MPI_Datatype *datatype)
{
  PetscInt i, N = *len;

  for (i = 0; i < N; i++) {
    cell_stats_t *A = (cell_stats_t *) a;
    cell_stats_t *B = (cell_stats_t *) b;

    B->min = PetscMin(A->min,B->min);
    B->max = PetscMax(A->max,B->max);
    B->sum += A->sum;
    B->squaresum += A->squaresum;
    B->count += A->count;
  }
}

/*@
  DMPlexCheckCellShape - Checks the Jacobian of the mapping from reference to real cells and computes some minimal statistics.

  Collective on dm

  Input Parameters:
+ dm        - The DMPlex object
. output    - If true, statistics will be displayed on stdout
- condLimit - Display all cells above this condition number, or PETSC_DETERMINE for no cell output

  Notes:
  This is mainly intended for debugging/testing purposes.

  For the complete list of DMPlexCheck* functions, see DMSetFromOptions().

  Level: developer

.seealso: DMSetFromOptions(), DMPlexComputeOrthogonalQuality()
@*/
PetscErrorCode DMPlexCheckCellShape(DM dm, PetscBool output, PetscReal condLimit)
{
  DM             dmCoarse;
  cell_stats_t   stats, globalStats;
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  PetscReal      *J, *invJ, min = 0, max = 0, mean = 0, stdev = 0;
  PetscReal      limit = condLimit > 0 ? condLimit : PETSC_MAX_REAL;
  PetscInt       cdim, cStart, cEnd, c, eStart, eEnd, count = 0;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  stats.min   = PETSC_MAX_REAL;
  stats.max   = PETSC_MIN_REAL;
  stats.sum   = stats.squaresum = 0.;
  stats.count = 0;

  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMGetCoordinateDim(dm,&cdim));
  CHKERRQ(PetscMalloc2(PetscSqr(cdim), &J, PetscSqr(cdim), &invJ));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(dm,0,&cStart,&cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm,1,&eStart,&eEnd));
  for (c = cStart; c < cEnd; c++) {
    PetscInt  i;
    PetscReal frobJ = 0., frobInvJ = 0., cond2, cond, detJ;

    CHKERRQ(DMPlexComputeCellGeometryAffineFEM(dm,c,NULL,J,invJ,&detJ));
    PetscCheckFalse(detJ < 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted", c);
    for (i = 0; i < PetscSqr(cdim); ++i) {
      frobJ    += J[i] * J[i];
      frobInvJ += invJ[i] * invJ[i];
    }
    cond2 = frobJ * frobInvJ;
    cond  = PetscSqrtReal(cond2);

    stats.min        = PetscMin(stats.min,cond);
    stats.max        = PetscMax(stats.max,cond);
    stats.sum       += cond;
    stats.squaresum += cond2;
    stats.count++;
    if (output && cond > limit) {
      PetscSection coordSection;
      Vec          coordsLocal;
      PetscScalar *coords = NULL;
      PetscInt     Nv, d, clSize, cl, *closure = NULL;

      CHKERRQ(DMGetCoordinatesLocal(dm, &coordsLocal));
      CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
      CHKERRQ(DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, &Nv, &coords));
      CHKERRQ(PetscSynchronizedPrintf(comm, "[%d] Cell %D cond %g\n", rank, c, (double) cond));
      for (i = 0; i < Nv/cdim; ++i) {
        CHKERRQ(PetscSynchronizedPrintf(comm, "  Vertex %D: (", i));
        for (d = 0; d < cdim; ++d) {
          if (d > 0) CHKERRQ(PetscSynchronizedPrintf(comm, ", "));
          CHKERRQ(PetscSynchronizedPrintf(comm, "%g", (double) PetscRealPart(coords[i*cdim+d])));
        }
        CHKERRQ(PetscSynchronizedPrintf(comm, ")\n"));
      }
      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
      for (cl = 0; cl < clSize*2; cl += 2) {
        const PetscInt edge = closure[cl];

        if ((edge >= eStart) && (edge < eEnd)) {
          PetscReal len;

          CHKERRQ(DMPlexComputeCellGeometryFVM(dm, edge, &len, NULL, NULL));
          CHKERRQ(PetscSynchronizedPrintf(comm, "  Edge %D: length %g\n", edge, (double) len));
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
      CHKERRQ(DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, &Nv, &coords));
    }
  }
  if (output) CHKERRQ(PetscSynchronizedFlush(comm, NULL));

  if (size > 1) {
    PetscMPIInt   blockLengths[2] = {4,1};
    MPI_Aint      blockOffsets[2] = {offsetof(cell_stats_t,min),offsetof(cell_stats_t,count)};
    MPI_Datatype  blockTypes[2]   = {MPIU_REAL,MPIU_INT}, statType;
    MPI_Op        statReduce;

    CHKERRMPI(MPI_Type_create_struct(2,blockLengths,blockOffsets,blockTypes,&statType));
    CHKERRMPI(MPI_Type_commit(&statType));
    CHKERRMPI(MPI_Op_create(cell_stats_reduce, PETSC_TRUE, &statReduce));
    CHKERRMPI(MPI_Reduce(&stats,&globalStats,1,statType,statReduce,0,comm));
    CHKERRMPI(MPI_Op_free(&statReduce));
    CHKERRMPI(MPI_Type_free(&statType));
  } else {
    CHKERRQ(PetscArraycpy(&globalStats,&stats,1));
  }
  if (rank == 0) {
    count = globalStats.count;
    min   = globalStats.min;
    max   = globalStats.max;
    mean  = globalStats.sum / globalStats.count;
    stdev = globalStats.count > 1 ? PetscSqrtReal(PetscMax((globalStats.squaresum - globalStats.count * mean * mean) / (globalStats.count - 1),0)) : 0.0;
  }

  if (output) {
    CHKERRQ(PetscPrintf(comm,"Mesh with %D cells, shape condition numbers: min = %g, max = %g, mean = %g, stddev = %g\n", count, (double) min, (double) max, (double) mean, (double) stdev));
  }
  CHKERRQ(PetscFree2(J,invJ));

  CHKERRQ(DMGetCoarseDM(dm,&dmCoarse));
  if (dmCoarse) {
    PetscBool isplex;

    CHKERRQ(PetscObjectTypeCompare((PetscObject)dmCoarse,DMPLEX,&isplex));
    if (isplex) {
      CHKERRQ(DMPlexCheckCellShape(dmCoarse,output,condLimit));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeOrthogonalQuality - Compute cell-wise orthogonal quality mesh statistic. Optionally tags all cells with
  orthogonal quality below given tolerance.

  Collective on dm

  Input Parameters:
+ dm   - The DMPlex object
. fv   - Optional PetscFV object for pre-computed cell/face centroid information
- atol - [0, 1] Absolute tolerance for tagging cells.

  Output Parameters:
+ OrthQual      - Vec containing orthogonal quality per cell
- OrthQualLabel - DMLabel tagging cells below atol with DM_ADAPT_REFINE

  Options Database Keys:
+ -dm_plex_orthogonal_quality_label_view - view OrthQualLabel if label is requested. Currently only PETSCVIEWERASCII is
supported.
- -dm_plex_orthogonal_quality_vec_view - view OrthQual vector.

  Notes:
  Orthogonal quality is given by the following formula:

  \min \left[ \frac{A_i \cdot f_i}{\|A_i\| \|f_i\|} , \frac{A_i \cdot c_i}{\|A_i\| \|c_i\|} \right]

  Where A_i is the i'th face-normal vector, f_i is the vector from the cell centroid to the i'th face centroid, and c_i
  is the vector from the current cells centroid to the centroid of its i'th neighbor (which shares a face with the
  current cell). This computes the vector similarity between each cell face and its corresponding neighbor centroid by
  calculating the cosine of the angle between these vectors.

  Orthogonal quality ranges from 1 (best) to 0 (worst).

  This routine is mainly useful for FVM, however is not restricted to only FVM. The PetscFV object is optionally used to check for
  pre-computed FVM cell data, but if it is not passed in then this data will be computed.

  Cells are tagged if they have an orthogonal quality less than or equal to the absolute tolerance.

  Level: intermediate

.seealso: DMPlexCheckCellShape(), DMCreateLabel()
@*/
PetscErrorCode DMPlexComputeOrthogonalQuality(DM dm, PetscFV fv, PetscReal atol, Vec *OrthQual, DMLabel *OrthQualLabel)
{
  PetscInt                nc, cellHeight, cStart, cEnd, cell, cellIter = 0;
  PetscInt                *idx;
  PetscScalar             *oqVals;
  const PetscScalar       *cellGeomArr, *faceGeomArr;
  PetscReal               *ci, *fi, *Ai;
  MPI_Comm                comm;
  Vec                     cellgeom, facegeom;
  DM                      dmFace, dmCell;
  IS                      glob;
  ISLocalToGlobalMapping  ltog;
  PetscViewer             vwr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (fv) {PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 2);}
  PetscValidPointer(OrthQual, 4);
  PetscCheck(atol >= 0.0 && atol <= 1.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %g not in [0,1]",(double)atol);
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDimension(dm, &nc));
  PetscCheckFalse(nc < 2,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DM must have dimension >= 2 (current %D)", nc);
  {
    DMPlexInterpolatedFlag interpFlag;

    CHKERRQ(DMPlexIsInterpolated(dm, &interpFlag));
    if (interpFlag != DMPLEX_INTERPOLATED_FULL) {
      PetscMPIInt rank;

      CHKERRMPI(MPI_Comm_rank(comm, &rank));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DM must be fully interpolated, DM on rank %d is not fully interpolated", rank);
    }
  }
  if (OrthQualLabel) {
    PetscValidPointer(OrthQualLabel, 5);
    CHKERRQ(DMCreateLabel(dm, "Orthogonal_Quality"));
    CHKERRQ(DMGetLabel(dm, "Orthogonal_Quality", OrthQualLabel));
  } else {*OrthQualLabel = NULL;}
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, &glob));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(glob, &ltog));
  CHKERRQ(ISLocalToGlobalMappingSetType(ltog, ISLOCALTOGLOBALMAPPINGHASH));
  CHKERRQ(VecCreate(comm, OrthQual));
  CHKERRQ(VecSetType(*OrthQual, VECSTANDARD));
  CHKERRQ(VecSetSizes(*OrthQual, cEnd-cStart, PETSC_DETERMINE));
  CHKERRQ(VecSetLocalToGlobalMapping(*OrthQual, ltog));
  CHKERRQ(VecSetUp(*OrthQual));
  CHKERRQ(ISDestroy(&glob));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&ltog));
  CHKERRQ(DMPlexGetDataFVM(dm, fv, &cellgeom, &facegeom, NULL));
  CHKERRQ(VecGetArrayRead(cellgeom, &cellGeomArr));
  CHKERRQ(VecGetArrayRead(facegeom, &faceGeomArr));
  CHKERRQ(VecGetDM(cellgeom, &dmCell));
  CHKERRQ(VecGetDM(facegeom, &dmFace));
  CHKERRQ(PetscMalloc5(cEnd-cStart, &idx, cEnd-cStart, &oqVals, nc, &ci, nc, &fi, nc, &Ai));
  for (cell = cStart; cell < cEnd; cellIter++,cell++) {
    PetscInt           cellneigh, cellneighiter = 0, adjSize = PETSC_DETERMINE;
    PetscInt           cellarr[2], *adj = NULL;
    PetscScalar        *cArr, *fArr;
    PetscReal          minvalc = 1.0, minvalf = 1.0;
    PetscFVCellGeom    *cg;

    idx[cellIter] = cell-cStart;
    cellarr[0] = cell;
    /* Make indexing into cellGeom easier */
    CHKERRQ(DMPlexPointLocalRead(dmCell, cell, cellGeomArr, &cg));
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &adjSize, &adj));
    /* Technically 1 too big, but easier than fiddling with empty adjacency array */
    CHKERRQ(PetscCalloc2(adjSize, &cArr, adjSize, &fArr));
    for (cellneigh = 0; cellneigh < adjSize; cellneighiter++,cellneigh++) {
      PetscInt         i;
      const PetscInt   neigh = adj[cellneigh];
      PetscReal        normci = 0, normfi = 0, normai = 0;
      PetscFVCellGeom  *cgneigh;
      PetscFVFaceGeom  *fg;

      /* Don't count ourselves in the neighbor list */
      if (neigh == cell) continue;
      CHKERRQ(DMPlexPointLocalRead(dmCell, neigh, cellGeomArr, &cgneigh));
      cellarr[1] = neigh;
      {
        PetscInt       numcovpts;
        const PetscInt *covpts;

        CHKERRQ(DMPlexGetMeet(dm, 2, cellarr, &numcovpts, &covpts));
        CHKERRQ(DMPlexPointLocalRead(dmFace, covpts[0], faceGeomArr, &fg));
        CHKERRQ(DMPlexRestoreMeet(dm, 2, cellarr, &numcovpts, &covpts));
      }

      /* Compute c_i, f_i and their norms */
      for (i = 0; i < nc; i++) {
        ci[i] = cgneigh->centroid[i] - cg->centroid[i];
        fi[i] = fg->centroid[i] - cg->centroid[i];
        Ai[i] = fg->normal[i];
        normci += PetscPowReal(ci[i], 2);
        normfi += PetscPowReal(fi[i], 2);
        normai += PetscPowReal(Ai[i], 2);
      }
      normci = PetscSqrtReal(normci);
      normfi = PetscSqrtReal(normfi);
      normai = PetscSqrtReal(normai);

      /* Normalize and compute for each face-cell-normal pair */
      for (i = 0; i < nc; i++) {
        ci[i] = ci[i]/normci;
        fi[i] = fi[i]/normfi;
        Ai[i] = Ai[i]/normai;
        /* PetscAbs because I don't know if normals are guaranteed to point out */
        cArr[cellneighiter] += PetscAbs(Ai[i]*ci[i]);
        fArr[cellneighiter] += PetscAbs(Ai[i]*fi[i]);
      }
      if (PetscRealPart(cArr[cellneighiter]) < minvalc) {
        minvalc = PetscRealPart(cArr[cellneighiter]);
      }
      if (PetscRealPart(fArr[cellneighiter]) < minvalf) {
        minvalf = PetscRealPart(fArr[cellneighiter]);
      }
    }
    CHKERRQ(PetscFree(adj));
    CHKERRQ(PetscFree2(cArr, fArr));
    /* Defer to cell if they're equal */
    oqVals[cellIter] = PetscMin(minvalf, minvalc);
    if (OrthQualLabel) {
      if (PetscRealPart(oqVals[cellIter]) <= atol) CHKERRQ(DMLabelSetValue(*OrthQualLabel, cell, DM_ADAPT_REFINE));
    }
  }
  CHKERRQ(VecSetValuesLocal(*OrthQual, cEnd-cStart, idx, oqVals, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(*OrthQual));
  CHKERRQ(VecAssemblyEnd(*OrthQual));
  CHKERRQ(VecRestoreArrayRead(cellgeom, &cellGeomArr));
  CHKERRQ(VecRestoreArrayRead(facegeom, &faceGeomArr));
  CHKERRQ(PetscOptionsGetViewer(comm, NULL, NULL, "-dm_plex_orthogonal_quality_label_view", &vwr, NULL, NULL));
  if (OrthQualLabel) {
    if (vwr) CHKERRQ(DMLabelView(*OrthQualLabel, vwr));
  }
  CHKERRQ(PetscFree5(idx, oqVals, ci, fi, Ai));
  CHKERRQ(PetscViewerDestroy(&vwr));
  CHKERRQ(VecViewFromOptions(*OrthQual, NULL, "-dm_plex_orthogonal_quality_vec_view"));
  PetscFunctionReturn(0);
}

/* this is here insead of DMGetOutputDM because output DM still has constraints in the local indices that affect
 * interpolator construction */
static PetscErrorCode DMGetFullDM(DM dm, DM *odm)
{
  PetscSection   section, newSection, gsection;
  PetscSF        sf;
  PetscBool      hasConstraints, ghasConstraints;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(odm,2);
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscSectionHasConstraints(section, &hasConstraints));
  CHKERRMPI(MPI_Allreduce(&hasConstraints, &ghasConstraints, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm)));
  if (!ghasConstraints) {
    CHKERRQ(PetscObjectReference((PetscObject)dm));
    *odm = dm;
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMClone(dm, odm));
  CHKERRQ(DMCopyFields(dm, *odm));
  CHKERRQ(DMGetLocalSection(*odm, &newSection));
  CHKERRQ(DMGetPointSF(*odm, &sf));
  CHKERRQ(PetscSectionCreateGlobalSection(newSection, sf, PETSC_TRUE, PETSC_FALSE, &gsection));
  CHKERRQ(DMSetGlobalSection(*odm, gsection));
  CHKERRQ(PetscSectionDestroy(&gsection));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateAffineInterpolationCorrection_Plex(DM dmc, DM dmf, Vec *shift)
{
  DM             dmco, dmfo;
  Mat            interpo;
  Vec            rscale;
  Vec            cglobalo, clocal;
  Vec            fglobal, fglobalo, flocal;
  PetscBool      regular;

  PetscFunctionBegin;
  CHKERRQ(DMGetFullDM(dmc, &dmco));
  CHKERRQ(DMGetFullDM(dmf, &dmfo));
  CHKERRQ(DMSetCoarseDM(dmfo, dmco));
  CHKERRQ(DMPlexGetRegularRefinement(dmf, &regular));
  CHKERRQ(DMPlexSetRegularRefinement(dmfo, regular));
  CHKERRQ(DMCreateInterpolation(dmco, dmfo, &interpo, &rscale));
  CHKERRQ(DMCreateGlobalVector(dmco, &cglobalo));
  CHKERRQ(DMCreateLocalVector(dmc, &clocal));
  CHKERRQ(VecSet(cglobalo, 0.));
  CHKERRQ(VecSet(clocal, 0.));
  CHKERRQ(DMCreateGlobalVector(dmf, &fglobal));
  CHKERRQ(DMCreateGlobalVector(dmfo, &fglobalo));
  CHKERRQ(DMCreateLocalVector(dmf, &flocal));
  CHKERRQ(VecSet(fglobal, 0.));
  CHKERRQ(VecSet(fglobalo, 0.));
  CHKERRQ(VecSet(flocal, 0.));
  CHKERRQ(DMPlexInsertBoundaryValues(dmc, PETSC_TRUE, clocal, 0., NULL, NULL, NULL));
  CHKERRQ(DMLocalToGlobalBegin(dmco, clocal, INSERT_VALUES, cglobalo));
  CHKERRQ(DMLocalToGlobalEnd(dmco, clocal, INSERT_VALUES, cglobalo));
  CHKERRQ(MatMult(interpo, cglobalo, fglobalo));
  CHKERRQ(DMGlobalToLocalBegin(dmfo, fglobalo, INSERT_VALUES, flocal));
  CHKERRQ(DMGlobalToLocalEnd(dmfo, fglobalo, INSERT_VALUES, flocal));
  CHKERRQ(DMLocalToGlobalBegin(dmf, flocal, INSERT_VALUES, fglobal));
  CHKERRQ(DMLocalToGlobalEnd(dmf, flocal, INSERT_VALUES, fglobal));
  *shift = fglobal;
  CHKERRQ(VecDestroy(&flocal));
  CHKERRQ(VecDestroy(&fglobalo));
  CHKERRQ(VecDestroy(&clocal));
  CHKERRQ(VecDestroy(&cglobalo));
  CHKERRQ(VecDestroy(&rscale));
  CHKERRQ(MatDestroy(&interpo));
  CHKERRQ(DMDestroy(&dmfo));
  CHKERRQ(DMDestroy(&dmco));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMInterpolateSolution_Plex(DM coarse, DM fine, Mat interp, Vec coarseSol, Vec fineSol)
{
  PetscObject    shifto;
  Vec            shift;

  PetscFunctionBegin;
  if (!interp) {
    Vec rscale;

    CHKERRQ(DMCreateInterpolation(coarse, fine, &interp, &rscale));
    CHKERRQ(VecDestroy(&rscale));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)interp));
  }
  CHKERRQ(PetscObjectQuery((PetscObject)interp, "_DMInterpolateSolution_Plex_Vec", &shifto));
  if (!shifto) {
    CHKERRQ(DMCreateAffineInterpolationCorrection_Plex(coarse, fine, &shift));
    CHKERRQ(PetscObjectCompose((PetscObject)interp, "_DMInterpolateSolution_Plex_Vec", (PetscObject) shift));
    shifto = (PetscObject) shift;
    CHKERRQ(VecDestroy(&shift));
  }
  shift = (Vec) shifto;
  CHKERRQ(MatInterpolate(interp, coarseSol, fineSol));
  CHKERRQ(VecAXPY(fineSol, 1.0, shift));
  CHKERRQ(MatDestroy(&interp));
  PetscFunctionReturn(0);
}

/* Pointwise interpolation
     Just code FEM for now
     u^f = I u^c
     sum_k u^f_k phi^f_k = I sum_j u^c_j phi^c_j
     u^f_i = sum_j psi^f_i I phi^c_j u^c_j
     I_{ij} = psi^f_i phi^c_j
*/
PetscErrorCode DMCreateInterpolation_Plex(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  void          *ctx;
  DM             cdm;
  PetscBool      regular, ismatis, isRefined = dmCoarse->data == dmFine->data ? PETSC_FALSE : PETSC_TRUE;

  PetscFunctionBegin;
  CHKERRQ(DMGetGlobalSection(dmFine, &gsf));
  CHKERRQ(PetscSectionGetConstrainedStorageSize(gsf, &m));
  CHKERRQ(DMGetGlobalSection(dmCoarse, &gsc));
  CHKERRQ(PetscSectionGetConstrainedStorageSize(gsc, &n));

  CHKERRQ(PetscStrcmp(dmCoarse->mattype, MATIS, &ismatis));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject) dmCoarse), interpolation));
  CHKERRQ(MatSetSizes(*interpolation, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  CHKERRQ(MatSetType(*interpolation, ismatis ? MATAIJ : dmCoarse->mattype));
  CHKERRQ(DMGetApplicationContext(dmFine, &ctx));

  CHKERRQ(DMGetCoarseDM(dmFine, &cdm));
  CHKERRQ(DMPlexGetRegularRefinement(dmFine, &regular));
  if (!isRefined || (regular && cdm == dmCoarse)) CHKERRQ(DMPlexComputeInterpolatorNested(dmCoarse, dmFine, isRefined, *interpolation, ctx));
  else                                            CHKERRQ(DMPlexComputeInterpolatorGeneral(dmCoarse, dmFine, *interpolation, ctx));
  CHKERRQ(MatViewFromOptions(*interpolation, NULL, "-interp_mat_view"));
  if (scaling) {
    /* Use naive scaling */
    CHKERRQ(DMCreateInterpolationScale(dmCoarse, dmFine, *interpolation, scaling));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInjection_Plex(DM dmCoarse, DM dmFine, Mat *mat)
{
  VecScatter     ctx;

  PetscFunctionBegin;
  CHKERRQ(DMPlexComputeInjectorFEM(dmCoarse, dmFine, &ctx, NULL));
  CHKERRQ(MatCreateScatter(PetscObjectComm((PetscObject)ctx), ctx, mat));
  CHKERRQ(VecScatterDestroy(&ctx));
  PetscFunctionReturn(0);
}

static void g0_identity_private(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];
  PetscInt       c;
  for (c = 0; c < Nc; ++c) g0[c*Nc+c] = 1.0;
}

PetscErrorCode DMCreateMassMatrixLumped_Plex(DM dm, Vec *mass)
{
  DM             dmc;
  PetscDS        ds;
  Vec            ones, locmass;
  IS             cellIS;
  PetscFormKey   key;
  PetscInt       depth;

  PetscFunctionBegin;
  CHKERRQ(DMClone(dm, &dmc));
  CHKERRQ(DMCopyDisc(dm, dmc));
  CHKERRQ(DMGetDS(dmc, &ds));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_identity_private, NULL, NULL, NULL));
  CHKERRQ(DMCreateGlobalVector(dmc, mass));
  CHKERRQ(DMGetLocalVector(dmc, &ones));
  CHKERRQ(DMGetLocalVector(dmc, &locmass));
  CHKERRQ(DMPlexGetDepth(dmc, &depth));
  CHKERRQ(DMGetStratumIS(dmc, "depth", depth, &cellIS));
  CHKERRQ(VecSet(locmass, 0.0));
  CHKERRQ(VecSet(ones, 1.0));
  key.label = NULL;
  key.value = 0;
  key.field = 0;
  key.part  = 0;
  CHKERRQ(DMPlexComputeJacobian_Action_Internal(dmc, key, cellIS, 0.0, 0.0, ones, NULL, ones, locmass, NULL));
  CHKERRQ(ISDestroy(&cellIS));
  CHKERRQ(VecSet(*mass, 0.0));
  CHKERRQ(DMLocalToGlobalBegin(dmc, locmass, ADD_VALUES, *mass));
  CHKERRQ(DMLocalToGlobalEnd(dmc, locmass, ADD_VALUES, *mass));
  CHKERRQ(DMRestoreLocalVector(dmc, &ones));
  CHKERRQ(DMRestoreLocalVector(dmc, &locmass));
  CHKERRQ(DMDestroy(&dmc));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMassMatrix_Plex(DM dmCoarse, DM dmFine, Mat *mass)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  void          *ctx;
  DM             cdm;
  PetscBool      regular;

  PetscFunctionBegin;
  if (dmFine == dmCoarse) {
    DM            dmc;
    PetscDS       ds;
    PetscWeakForm wf;
    Vec           u;
    IS            cellIS;
    PetscFormKey  key;
    PetscInt      depth;

    CHKERRQ(DMClone(dmFine, &dmc));
    CHKERRQ(DMCopyDisc(dmFine, dmc));
    CHKERRQ(DMGetDS(dmc, &ds));
    CHKERRQ(PetscDSGetWeakForm(ds, &wf));
    CHKERRQ(PetscWeakFormClear(wf));
    CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_identity_private, NULL, NULL, NULL));
    CHKERRQ(DMCreateMatrix(dmc, mass));
    CHKERRQ(DMGetGlobalVector(dmc, &u));
    CHKERRQ(DMPlexGetDepth(dmc, &depth));
    CHKERRQ(DMGetStratumIS(dmc, "depth", depth, &cellIS));
    CHKERRQ(MatZeroEntries(*mass));
    key.label = NULL;
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    CHKERRQ(DMPlexComputeJacobian_Internal(dmc, key, cellIS, 0.0, 0.0, u, NULL, *mass, *mass, NULL));
    CHKERRQ(ISDestroy(&cellIS));
    CHKERRQ(DMRestoreGlobalVector(dmc, &u));
    CHKERRQ(DMDestroy(&dmc));
  } else {
    CHKERRQ(DMGetGlobalSection(dmFine, &gsf));
    CHKERRQ(PetscSectionGetConstrainedStorageSize(gsf, &m));
    CHKERRQ(DMGetGlobalSection(dmCoarse, &gsc));
    CHKERRQ(PetscSectionGetConstrainedStorageSize(gsc, &n));

    CHKERRQ(MatCreate(PetscObjectComm((PetscObject) dmCoarse), mass));
    CHKERRQ(MatSetSizes(*mass, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
    CHKERRQ(MatSetType(*mass, dmCoarse->mattype));
    CHKERRQ(DMGetApplicationContext(dmFine, &ctx));

    CHKERRQ(DMGetCoarseDM(dmFine, &cdm));
    CHKERRQ(DMPlexGetRegularRefinement(dmFine, &regular));
    if (regular && cdm == dmCoarse) CHKERRQ(DMPlexComputeMassMatrixNested(dmCoarse, dmFine, *mass, ctx));
    else                            CHKERRQ(DMPlexComputeMassMatrixGeneral(dmCoarse, dmFine, *mass, ctx));
  }
  CHKERRQ(MatViewFromOptions(*mass, NULL, "-mass_mat_view"));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetRegularRefinement - Get the flag indicating that this mesh was obtained by regular refinement from its coarse mesh

  Input Parameter:
. dm - The DMPlex object

  Output Parameter:
. regular - The flag

  Level: intermediate

.seealso: DMPlexSetRegularRefinement()
@*/
PetscErrorCode DMPlexGetRegularRefinement(DM dm, PetscBool *regular)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(regular, 2);
  *regular = ((DM_Plex *) dm->data)->regularRefinement;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetRegularRefinement - Set the flag indicating that this mesh was obtained by regular refinement from its coarse mesh

  Input Parameters:
+ dm - The DMPlex object
- regular - The flag

  Level: intermediate

.seealso: DMPlexGetRegularRefinement()
@*/
PetscErrorCode DMPlexSetRegularRefinement(DM dm, PetscBool regular)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ((DM_Plex *) dm->data)->regularRefinement = regular;
  PetscFunctionReturn(0);
}

/* anchors */
/*@
  DMPlexGetAnchors - Get the layout of the anchor (point-to-point) constraints.  Typically, the user will not have to
  call DMPlexGetAnchors() directly: if there are anchors, then DMPlexGetAnchors() is called during DMGetDefaultConstraints().

  not collective

  Input Parameter:
. dm - The DMPlex object

  Output Parameters:
+ anchorSection - If not NULL, set to the section describing which points anchor the constrained points.
- anchorIS - If not NULL, set to the list of anchors indexed by anchorSection

  Level: intermediate

.seealso: DMPlexSetAnchors(), DMGetDefaultConstraints(), DMSetDefaultConstraints()
@*/
PetscErrorCode DMPlexGetAnchors(DM dm, PetscSection *anchorSection, IS *anchorIS)
{
  DM_Plex *plex = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!plex->anchorSection && !plex->anchorIS && plex->createanchors) CHKERRQ((*plex->createanchors)(dm));
  if (anchorSection) *anchorSection = plex->anchorSection;
  if (anchorIS) *anchorIS = plex->anchorIS;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetAnchors - Set the layout of the local anchor (point-to-point) constraints.  Unlike boundary conditions,
  when a point's degrees of freedom in a section are constrained to an outside value, the anchor constraints set a
  point's degrees of freedom to be a linear combination of other points' degrees of freedom.

  After specifying the layout of constraints with DMPlexSetAnchors(), one specifies the constraints by calling
  DMGetDefaultConstraints() and filling in the entries in the constraint matrix.

  collective on dm

  Input Parameters:
+ dm - The DMPlex object
. anchorSection - The section that describes the mapping from constrained points to the anchor points listed in anchorIS.  Must have a local communicator (PETSC_COMM_SELF or derivative).
- anchorIS - The list of all anchor points.  Must have a local communicator (PETSC_COMM_SELF or derivative).

  The reference counts of anchorSection and anchorIS are incremented.

  Level: intermediate

.seealso: DMPlexGetAnchors(), DMGetDefaultConstraints(), DMSetDefaultConstraints()
@*/
PetscErrorCode DMPlexSetAnchors(DM dm, PetscSection anchorSection, IS anchorIS)
{
  DM_Plex        *plex = (DM_Plex *)dm->data;
  PetscMPIInt    result;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (anchorSection) {
    PetscValidHeaderSpecific(anchorSection,PETSC_SECTION_CLASSID,2);
    CHKERRMPI(MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)anchorSection),&result));
    PetscCheckFalse(result != MPI_CONGRUENT && result != MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"anchor section must have local communicator");
  }
  if (anchorIS) {
    PetscValidHeaderSpecific(anchorIS,IS_CLASSID,3);
    CHKERRMPI(MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)anchorIS),&result));
    PetscCheckFalse(result != MPI_CONGRUENT && result != MPI_IDENT,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"anchor IS must have local communicator");
  }

  CHKERRQ(PetscObjectReference((PetscObject)anchorSection));
  CHKERRQ(PetscSectionDestroy(&plex->anchorSection));
  plex->anchorSection = anchorSection;

  CHKERRQ(PetscObjectReference((PetscObject)anchorIS));
  CHKERRQ(ISDestroy(&plex->anchorIS));
  plex->anchorIS = anchorIS;

  if (PetscUnlikelyDebug(anchorIS && anchorSection)) {
    PetscInt size, a, pStart, pEnd;
    const PetscInt *anchors;

    CHKERRQ(PetscSectionGetChart(anchorSection,&pStart,&pEnd));
    CHKERRQ(ISGetLocalSize(anchorIS,&size));
    CHKERRQ(ISGetIndices(anchorIS,&anchors));
    for (a = 0; a < size; a++) {
      PetscInt p;

      p = anchors[a];
      if (p >= pStart && p < pEnd) {
        PetscInt dof;

        CHKERRQ(PetscSectionGetDof(anchorSection,p,&dof));
        if (dof) {

          CHKERRQ(ISRestoreIndices(anchorIS,&anchors));
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Point %D cannot be constrained and an anchor",p);
        }
      }
    }
    CHKERRQ(ISRestoreIndices(anchorIS,&anchors));
  }
  /* reset the generic constraints */
  CHKERRQ(DMSetDefaultConstraints(dm,NULL,NULL,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateConstraintSection_Anchors(DM dm, PetscSection section, PetscSection *cSec)
{
  PetscSection anchorSection;
  PetscInt pStart, pEnd, sStart, sEnd, p, dof, numFields, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetAnchors(dm,&anchorSection,NULL));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF,cSec));
  CHKERRQ(PetscSectionGetNumFields(section,&numFields));
  if (numFields) {
    PetscInt f;
    CHKERRQ(PetscSectionSetNumFields(*cSec,numFields));

    for (f = 0; f < numFields; f++) {
      PetscInt numComp;

      CHKERRQ(PetscSectionGetFieldComponents(section,f,&numComp));
      CHKERRQ(PetscSectionSetFieldComponents(*cSec,f,numComp));
    }
  }
  CHKERRQ(PetscSectionGetChart(anchorSection,&pStart,&pEnd));
  CHKERRQ(PetscSectionGetChart(section,&sStart,&sEnd));
  pStart = PetscMax(pStart,sStart);
  pEnd   = PetscMin(pEnd,sEnd);
  pEnd   = PetscMax(pStart,pEnd);
  CHKERRQ(PetscSectionSetChart(*cSec,pStart,pEnd));
  for (p = pStart; p < pEnd; p++) {
    CHKERRQ(PetscSectionGetDof(anchorSection,p,&dof));
    if (dof) {
      CHKERRQ(PetscSectionGetDof(section,p,&dof));
      CHKERRQ(PetscSectionSetDof(*cSec,p,dof));
      for (f = 0; f < numFields; f++) {
        CHKERRQ(PetscSectionGetFieldDof(section,p,f,&dof));
        CHKERRQ(PetscSectionSetFieldDof(*cSec,p,f,dof));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(*cSec));
  CHKERRQ(PetscObjectSetName((PetscObject) *cSec, "Constraint Section"));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateConstraintMatrix_Anchors(DM dm, PetscSection section, PetscSection cSec, Mat *cMat)
{
  PetscSection   aSec;
  PetscInt       pStart, pEnd, p, sStart, sEnd, dof, aDof, aOff, off, nnz, annz, m, n, q, a, offset, *i, *j;
  const PetscInt *anchors;
  PetscInt       numFields, f;
  IS             aIS;
  MatType        mtype;
  PetscBool      iscuda,iskokkos;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionGetStorageSize(cSec, &m));
  CHKERRQ(PetscSectionGetStorageSize(section, &n));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,cMat));
  CHKERRQ(MatSetSizes(*cMat,m,n,m,n));
  CHKERRQ(PetscStrcmp(dm->mattype,MATSEQAIJCUSPARSE,&iscuda));
  if (!iscuda) CHKERRQ(PetscStrcmp(dm->mattype,MATMPIAIJCUSPARSE,&iscuda));
  CHKERRQ(PetscStrcmp(dm->mattype,MATSEQAIJKOKKOS,&iskokkos));
  if (!iskokkos) CHKERRQ(PetscStrcmp(dm->mattype,MATMPIAIJKOKKOS,&iskokkos));
  if (iscuda) mtype = MATSEQAIJCUSPARSE;
  else if (iskokkos) mtype = MATSEQAIJKOKKOS;
  else mtype = MATSEQAIJ;
  CHKERRQ(MatSetType(*cMat,mtype));
  CHKERRQ(DMPlexGetAnchors(dm,&aSec,&aIS));
  CHKERRQ(ISGetIndices(aIS,&anchors));
  /* cSec will be a subset of aSec and section */
  CHKERRQ(PetscSectionGetChart(cSec,&pStart,&pEnd));
  CHKERRQ(PetscSectionGetChart(section,&sStart,&sEnd));
  CHKERRQ(PetscMalloc1(m+1,&i));
  i[0] = 0;
  CHKERRQ(PetscSectionGetNumFields(section,&numFields));
  for (p = pStart; p < pEnd; p++) {
    PetscInt rDof, rOff, r;

    CHKERRQ(PetscSectionGetDof(aSec,p,&rDof));
    if (!rDof) continue;
    CHKERRQ(PetscSectionGetOffset(aSec,p,&rOff));
    if (numFields) {
      for (f = 0; f < numFields; f++) {
        annz = 0;
        for (r = 0; r < rDof; r++) {
          a = anchors[rOff + r];
          if (a < sStart || a >= sEnd) continue;
          CHKERRQ(PetscSectionGetFieldDof(section,a,f,&aDof));
          annz += aDof;
        }
        CHKERRQ(PetscSectionGetFieldDof(cSec,p,f,&dof));
        CHKERRQ(PetscSectionGetFieldOffset(cSec,p,f,&off));
        for (q = 0; q < dof; q++) {
          i[off + q + 1] = i[off + q] + annz;
        }
      }
    } else {
      annz = 0;
      CHKERRQ(PetscSectionGetDof(cSec,p,&dof));
      for (q = 0; q < dof; q++) {
        a = anchors[rOff + q];
        if (a < sStart || a >= sEnd) continue;
        CHKERRQ(PetscSectionGetDof(section,a,&aDof));
        annz += aDof;
      }
      CHKERRQ(PetscSectionGetDof(cSec,p,&dof));
      CHKERRQ(PetscSectionGetOffset(cSec,p,&off));
      for (q = 0; q < dof; q++) {
        i[off + q + 1] = i[off + q] + annz;
      }
    }
  }
  nnz = i[m];
  CHKERRQ(PetscMalloc1(nnz,&j));
  offset = 0;
  for (p = pStart; p < pEnd; p++) {
    if (numFields) {
      for (f = 0; f < numFields; f++) {
        CHKERRQ(PetscSectionGetFieldDof(cSec,p,f,&dof));
        for (q = 0; q < dof; q++) {
          PetscInt rDof, rOff, r;
          CHKERRQ(PetscSectionGetDof(aSec,p,&rDof));
          CHKERRQ(PetscSectionGetOffset(aSec,p,&rOff));
          for (r = 0; r < rDof; r++) {
            PetscInt s;

            a = anchors[rOff + r];
            if (a < sStart || a >= sEnd) continue;
            CHKERRQ(PetscSectionGetFieldDof(section,a,f,&aDof));
            CHKERRQ(PetscSectionGetFieldOffset(section,a,f,&aOff));
            for (s = 0; s < aDof; s++) {
              j[offset++] = aOff + s;
            }
          }
        }
      }
    } else {
      CHKERRQ(PetscSectionGetDof(cSec,p,&dof));
      for (q = 0; q < dof; q++) {
        PetscInt rDof, rOff, r;
        CHKERRQ(PetscSectionGetDof(aSec,p,&rDof));
        CHKERRQ(PetscSectionGetOffset(aSec,p,&rOff));
        for (r = 0; r < rDof; r++) {
          PetscInt s;

          a = anchors[rOff + r];
          if (a < sStart || a >= sEnd) continue;
          CHKERRQ(PetscSectionGetDof(section,a,&aDof));
          CHKERRQ(PetscSectionGetOffset(section,a,&aOff));
          for (s = 0; s < aDof; s++) {
            j[offset++] = aOff + s;
          }
        }
      }
    }
  }
  CHKERRQ(MatSeqAIJSetPreallocationCSR(*cMat,i,j,NULL));
  CHKERRQ(PetscFree(i));
  CHKERRQ(PetscFree(j));
  CHKERRQ(ISRestoreIndices(aIS,&anchors));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateDefaultConstraints_Plex(DM dm)
{
  DM_Plex        *plex = (DM_Plex *)dm->data;
  PetscSection   anchorSection, section, cSec;
  Mat            cMat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetAnchors(dm,&anchorSection,NULL));
  if (anchorSection) {
    PetscInt Nf;

    CHKERRQ(DMGetLocalSection(dm,&section));
    CHKERRQ(DMPlexCreateConstraintSection_Anchors(dm,section,&cSec));
    CHKERRQ(DMPlexCreateConstraintMatrix_Anchors(dm,section,cSec,&cMat));
    CHKERRQ(DMGetNumFields(dm,&Nf));
    if (Nf && plex->computeanchormatrix) CHKERRQ((*plex->computeanchormatrix)(dm,section,cSec,cMat));
    CHKERRQ(DMSetDefaultConstraints(dm,cSec,cMat,NULL));
    CHKERRQ(PetscSectionDestroy(&cSec));
    CHKERRQ(MatDestroy(&cMat));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDomainDM_Plex(DM dm, DMLabel label, PetscInt value, IS *is, DM *subdm)
{
  IS             subis;
  PetscSection   section, subsection;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalSection(dm, &section));
  PetscCheck(section,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting subdomain");
  PetscCheck(subdm,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Must set output subDM for splitting subdomain");
  /* Create subdomain */
  CHKERRQ(DMPlexFilter(dm, label, value, subdm));
  /* Create submodel */
  CHKERRQ(DMPlexGetSubpointIS(*subdm, &subis));
  CHKERRQ(PetscSectionCreateSubmeshSection(section, subis, &subsection));
  CHKERRQ(DMSetLocalSection(*subdm, subsection));
  CHKERRQ(PetscSectionDestroy(&subsection));
  CHKERRQ(DMCopyDisc(dm, *subdm));
  /* Create map from submodel to global model */
  if (is) {
    PetscSection    sectionGlobal, subsectionGlobal;
    IS              spIS;
    const PetscInt *spmap;
    PetscInt       *subIndices;
    PetscInt        subSize = 0, subOff = 0, pStart, pEnd, p;
    PetscInt        Nf, f, bs = -1, bsLocal[2], bsMinMax[2];

    CHKERRQ(DMPlexGetSubpointIS(*subdm, &spIS));
    CHKERRQ(ISGetIndices(spIS, &spmap));
    CHKERRQ(PetscSectionGetNumFields(section, &Nf));
    CHKERRQ(DMGetGlobalSection(dm, &sectionGlobal));
    CHKERRQ(DMGetGlobalSection(*subdm, &subsectionGlobal));
    CHKERRQ(PetscSectionGetChart(subsection, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, pSubSize  = 0;

      CHKERRQ(PetscSectionGetDof(sectionGlobal, p, &gdof));
      if (gdof > 0) {
        for (f = 0; f < Nf; ++f) {
          PetscInt fdof, fcdof;

          CHKERRQ(PetscSectionGetFieldDof(subsection, p, f, &fdof));
          CHKERRQ(PetscSectionGetFieldConstraintDof(subsection, p, f, &fcdof));
          pSubSize += fdof-fcdof;
        }
        subSize += pSubSize;
        if (pSubSize) {
          if (bs < 0) {
            bs = pSubSize;
          } else if (bs != pSubSize) {
            /* Layout does not admit a pointwise block size */
            bs = 1;
          }
        }
      }
    }
    /* Must have same blocksize on all procs (some might have no points) */
    bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs; bsLocal[1] = bs;
    CHKERRQ(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax));
    if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
    else                            {bs = bsMinMax[0];}
    CHKERRQ(PetscMalloc1(subSize, &subIndices));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      CHKERRQ(PetscSectionGetDof(subsectionGlobal, p, &gdof));
      if (gdof > 0) {
        const PetscInt point = spmap[p];

        CHKERRQ(PetscSectionGetOffset(sectionGlobal, point, &goff));
        for (f = 0; f < Nf; ++f) {
          PetscInt fdof, fcdof, fc, f2, poff = 0;

          /* Can get rid of this loop by storing field information in the global section */
          for (f2 = 0; f2 < f; ++f2) {
            CHKERRQ(PetscSectionGetFieldDof(section, p, f2, &fdof));
            CHKERRQ(PetscSectionGetFieldConstraintDof(section, p, f2, &fcdof));
            poff += fdof-fcdof;
          }
          CHKERRQ(PetscSectionGetFieldDof(section, p, f, &fdof));
          CHKERRQ(PetscSectionGetFieldConstraintDof(section, p, f, &fcdof));
          for (fc = 0; fc < fdof-fcdof; ++fc, ++subOff) {
            subIndices[subOff] = goff+poff+fc;
          }
        }
      }
    }
    CHKERRQ(ISRestoreIndices(spIS, &spmap));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm), subSize, subIndices, PETSC_OWN_POINTER, is));
    if (bs > 1) {
      /* We need to check that the block size does not come from non-contiguous fields */
      PetscInt i, j, set = 1;
      for (i = 0; i < subSize; i += bs) {
        for (j = 0; j < bs; ++j) {
          if (subIndices[i+j] != subIndices[i]+j) {set = 0; break;}
        }
      }
      if (set) CHKERRQ(ISSetBlockSize(*is, bs));
    }
    /* Attach nullspace */
    for (f = 0; f < Nf; ++f) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[f];
      if ((*subdm)->nullspaceConstructors[f]) break;
    }
    if (f < Nf) {
      MatNullSpace nullSpace;
      CHKERRQ((*(*subdm)->nullspaceConstructors[f])(*subdm, f, f, &nullSpace));

      CHKERRQ(PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace));
      CHKERRQ(MatNullSpaceDestroy(&nullSpace));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexMonitorThroughput - Report the cell throughput of FE integration

  Input Parameter:
- dm - The DM

  Level: developer

  Options Database Keys:
. -dm_plex_monitor_throughput - Activate the monitor

.seealso: DMSetFromOptions(), DMPlexCreate()
@*/
PetscErrorCode DMPlexMonitorThroughput(DM dm, void *dummy)
{
#if defined(PETSC_USE_LOG)
  PetscStageLog      stageLog;
  PetscLogEvent      event;
  PetscLogStage      stage;
  PetscEventPerfInfo eventInfo;
  PetscReal          cellRate, flopRate;
  PetscInt           cStart, cEnd, Nf, N;
  const char        *name;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
#if defined(PETSC_USE_LOG)
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(PetscLogGetStageLog(&stageLog));
  CHKERRQ(PetscStageLogGetCurrent(stageLog, &stage));
  CHKERRQ(PetscLogEventGetId("DMPlexResidualFE", &event));
  CHKERRQ(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
  N        = (cEnd - cStart)*Nf*eventInfo.count;
  flopRate = eventInfo.flops/eventInfo.time;
  cellRate = N/eventInfo.time;
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "DM (%s) FE Residual Integration: %D integrals %D reps\n  Cell rate: %.2g/s flop rate: %.2g MF/s\n", name ? name : "unknown", N, eventInfo.count, (double) cellRate, (double) (flopRate/1.e6)));
#else
  SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Plex Throughput Monitor is not supported if logging is turned off. Reconfigure using --with-log.");
#endif
  PetscFunctionReturn(0);
}
