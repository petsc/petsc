#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petscsf.h>
#include <petscds.h>
#include <petscdraw.h>
#include <petscdmfield.h>

/* Logging support */
PetscLogEvent DMPLEX_Interpolate, DMPLEX_Partition, DMPLEX_Distribute, DMPLEX_DistributeCones, DMPLEX_DistributeLabels, DMPLEX_DistributeSF, DMPLEX_DistributeOverlap, DMPLEX_DistributeField, DMPLEX_DistributeData, DMPLEX_Migrate, DMPLEX_InterpolateSF, DMPLEX_GlobalToNaturalBegin, DMPLEX_GlobalToNaturalEnd, DMPLEX_NaturalToGlobalBegin, DMPLEX_NaturalToGlobalEnd, DMPLEX_Stratify, DMPLEX_Symmetrize, DMPLEX_Preallocate, DMPLEX_ResidualFEM, DMPLEX_JacobianFEM, DMPLEX_InterpolatorFEM, DMPLEX_InjectorFEM, DMPLEX_IntegralFEM, DMPLEX_CreateGmsh, DMPLEX_RebalanceSharedPoints, DMPLEX_PartSelf, DMPLEX_PartLabelInvert, DMPLEX_PartLabelCreateSF, DMPLEX_PartStratSF, DMPLEX_CreatePointSF;

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

/*@
  DMPlexRefineSimplexToTensor - Uniformly refines simplicial cells into tensor product cells.
  3 quadrilaterals per triangle in 2D and 4 hexahedra per tetrahedron in 3D.

  Collective

  Input Parameters:
. dm - The DMPlex object

  Output Parameters:
. dmRefined - The refined DMPlex object

  Note: Returns NULL if the mesh is already a tensor product mesh.

  Level: intermediate

.seealso: DMPlexCreate(), DMPlexSetRefinementUniform()
@*/
PetscErrorCode DMPlexRefineSimplexToTensor(DM dm, DM *dmRefined)
{
  CellRefiner      cellRefiner;
  DMPolytopeType   ct;
  PetscInt         dim, cMax, fMax, cStart, cEnd;
  PetscBool        lop, allnoop, localized;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmRefined, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm,&cMax,&fMax,NULL,NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  if (!(cEnd - cStart)) cellRefiner = REFINER_NOOP;
  else {
    ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
    switch (ct) {
      case DM_POLYTOPE_POINT:
      case DM_POLYTOPE_SEGMENT:
        cellRefiner = REFINER_NOOP;break;
      case DM_POLYTOPE_TRIANGLE:
        if (cMax >= 0) cellRefiner = REFINER_HYBRID_SIMPLEX_TO_HEX_2D;
        else           cellRefiner = REFINER_SIMPLEX_TO_HEX_2D;
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        if (cMax >= 0) cellRefiner = REFINER_HYBRID_SIMPLEX_TO_HEX_2D;
        else           cellRefiner = REFINER_NOOP;
        break;
      case DM_POLYTOPE_TETRAHEDRON:
        if (cMax >= 0) cellRefiner = REFINER_HYBRID_SIMPLEX_TO_HEX_3D;
        else           cellRefiner = REFINER_SIMPLEX_TO_HEX_3D;
        break;
      case DM_POLYTOPE_TRI_PRISM_TENSOR:
        cellRefiner = REFINER_HYBRID_SIMPLEX_TO_HEX_3D;break;
      case DM_POLYTOPE_HEXAHEDRON:
        if (cMax >= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Simplex2Tensor in 3D with Hybrid mesh not yet done");
        else           cellRefiner = REFINER_NOOP;
        break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle cell polytope type %s", DMPolytopeTypes[ct]);
    }
  }
  /* return if we don't need to refine */
  lop = (cellRefiner == REFINER_NOOP) ? PETSC_TRUE : PETSC_FALSE;
  ierr = MPIU_Allreduce(&lop,&allnoop,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  if (allnoop) {
    *dmRefined = NULL;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexRefineUniform_Internal(dm, cellRefiner, dmRefined);CHKERRQ(ierr);
  ierr = DMCopyBoundary(dm, *dmRefined);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  if (localized) {
    ierr = DMLocalizeCoordinates(*dmRefined);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetFieldType_Internal(DM dm, PetscSection section, PetscInt field, PetscInt *sStart, PetscInt *sEnd, PetscViewerVTKFieldType *ft)
{
  PetscInt       cdim, pStart, pEnd, vStart, vEnd, cStart, cEnd, cMax;
  PetscInt       vcdof[2] = {0,0}, globalvcdof[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ft  = PETSC_VTK_INVALID;
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cMax < 0 ? cEnd : cMax;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  if (field >= 0) {
    if ((vStart >= pStart) && (vStart < pEnd)) {ierr = PetscSectionGetFieldDof(section, vStart, field, &vcdof[0]);CHKERRQ(ierr);}
    if ((cStart >= pStart) && (cStart < pEnd)) {ierr = PetscSectionGetFieldDof(section, cStart, field, &vcdof[1]);CHKERRQ(ierr);}
  } else {
    if ((vStart >= pStart) && (vStart < pEnd)) {ierr = PetscSectionGetDof(section, vStart, &vcdof[0]);CHKERRQ(ierr);}
    if ((cStart >= pStart) && (cStart < pEnd)) {ierr = PetscSectionGetDof(section, cStart, &vcdof[1]);CHKERRQ(ierr);}
  }
  ierr = MPI_Allreduce(vcdof, globalvcdof, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
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

      ierr = PetscSectionGetFieldName(section, field, &fieldname);CHKERRQ(ierr);
      ierr = PetscInfo2((PetscObject) dm, "Could not classify VTK output type of section field %D \"%s\"\n", field, fieldname);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo((PetscObject) dm, "Could not classify VTK output typp of section\"%s\"\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_Plex_Local_Draw(Vec v, PetscViewer viewer)
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
  PetscBool          isnull, flg;
  PetscInt           dim, Nf, f, Nc, comp, vStart, vEnd, cStart, cEnd, c, N, level, step, w = 0;
  const char        *name;
  char               title[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw, &isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D. Use PETSCVIEWERGLVIS", dim);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  ierr = DMGetCoarsenLevel(dm, &level);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &step, &time);CHKERRQ(ierr);

  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));   bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c+1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c+1]));
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  /* Could implement something like DMDASelectFields() */
  for (f = 0; f < Nf; ++f) {
    DM   fdm = dm;
    Vec  fv  = v;
    IS   fis;
    char prefix[PETSC_MAX_PATH_LEN];
    const char *fname;

    ierr = PetscSectionGetFieldComponents(s, f, &Nc);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldName(s, f, &fname);CHKERRQ(ierr);

    if (v->hdr.prefix) {ierr = PetscStrncpy(prefix, v->hdr.prefix,sizeof(prefix));CHKERRQ(ierr);}
    else               {prefix[0] = '\0';}
    if (Nf > 1) {
      ierr = DMCreateSubDM(dm, 1, &f, &fis, &fdm);CHKERRQ(ierr);
      ierr = VecGetSubVector(v, fis, &fv);CHKERRQ(ierr);
      ierr = PetscStrlcat(prefix, fname,sizeof(prefix));CHKERRQ(ierr);
      ierr = PetscStrlcat(prefix, "_",sizeof(prefix));CHKERRQ(ierr);
    }
    for (comp = 0; comp < Nc; ++comp, ++w) {
      PetscInt nmax = 2;

      ierr = PetscViewerDrawGetDraw(viewer, w, &draw);CHKERRQ(ierr);
      if (Nc > 1) {ierr = PetscSNPrintf(title, sizeof(title), "%s:%s_%D Step: %D Time: %.4g", name, fname, comp, step, time);CHKERRQ(ierr);}
      else        {ierr = PetscSNPrintf(title, sizeof(title), "%s:%s Step: %D Time: %.4g", name, fname, step, time);CHKERRQ(ierr);}
      ierr = PetscDrawSetTitle(draw, title);CHKERRQ(ierr);

      /* TODO Get max and min only for this component */
      ierr = PetscOptionsGetRealArray(NULL, prefix, "-vec_view_bounds", vbound, &nmax, &flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = VecMin(fv, NULL, &vbound[0]);CHKERRQ(ierr);
        ierr = VecMax(fv, NULL, &vbound[1]);CHKERRQ(ierr);
        if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
      }
      ierr = PetscDrawGetPopup(draw, &popup);CHKERRQ(ierr);
      ierr = PetscDrawScalePopup(popup, vbound[0], vbound[1]);CHKERRQ(ierr);
      ierr = PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]);CHKERRQ(ierr);

      ierr = VecGetArrayRead(fv, &array);CHKERRQ(ierr);
      for (c = cStart; c < cEnd; ++c) {
        PetscScalar *coords = NULL, *a = NULL;
        PetscInt     numCoords, color[4] = {-1,-1,-1,-1};

        ierr = DMPlexPointLocalRead(fdm, c, array, &a);CHKERRQ(ierr);
        if (a) {
          color[0] = PetscDrawRealToColor(PetscRealPart(a[comp]), vbound[0], vbound[1]);
          color[1] = color[2] = color[3] = color[0];
        } else {
          PetscScalar *vals = NULL;
          PetscInt     numVals, va;

          ierr = DMPlexVecGetClosure(fdm, NULL, fv, c, &numVals, &vals);CHKERRQ(ierr);
          if (numVals % Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of components %D does not divide the number of values in the closure %D", Nc, numVals);
          switch (numVals/Nc) {
          case 3: /* P1 Triangle */
          case 4: /* P1 Quadrangle */
            for (va = 0; va < numVals/Nc; ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc+comp]), vbound[0], vbound[1]);
            break;
          case 6: /* P2 Triangle */
          case 8: /* P2 Quadrangle */
            for (va = 0; va < numVals/(Nc*2); ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc+comp + numVals/(Nc*2)]), vbound[0], vbound[1]);
            break;
          default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of values for cell closure %D cannot be handled", numVals/Nc);
          }
          ierr = DMPlexVecRestoreClosure(fdm, NULL, fv, c, &numVals, &vals);CHKERRQ(ierr);
        }
        ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
        switch (numCoords) {
        case 6:
          ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]);CHKERRQ(ierr);
          break;
        case 8:
          ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]);CHKERRQ(ierr);
          ierr = PetscDrawTriangle(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), color[2], color[3], color[0]);CHKERRQ(ierr);
          break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells with %D coordinates", numCoords);
        }
        ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(fv, &array);CHKERRQ(ierr);
      ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
      ierr = PetscDrawPause(draw);CHKERRQ(ierr);
      ierr = PetscDrawSave(draw);CHKERRQ(ierr);
    }
    if (Nf > 1) {
      ierr = VecRestoreSubVector(v, fis, &fv);CHKERRQ(ierr);
      ierr = ISDestroy(&fis);CHKERRQ(ierr);
      ierr = DMDestroy(&fdm);CHKERRQ(ierr);
    }
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
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &locv);CHKERRQ(ierr); /* VTK viewer requires exclusive ownership of the vector */
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
  ierr = VecCopy(v, locv);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (!numFields) {
    ierr = DMPlexGetFieldType_Internal(dm, section, PETSC_DETERMINE, &pStart, &pEnd, &ft);CHKERRQ(ierr);
    ierr = PetscViewerVTKAddField(viewer, (PetscObject) dm, DMPlexVTKWriteAll, PETSC_DEFAULT, ft, PETSC_TRUE,(PetscObject) locv);CHKERRQ(ierr);
  } else {
    PetscInt f;

    for (f = 0; f < numFields; f++) {
      ierr = DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft);CHKERRQ(ierr);
      if (ft == PETSC_VTK_INVALID) continue;
      ierr = PetscObjectReference((PetscObject)locv);CHKERRQ(ierr);
      ierr = PetscViewerVTKAddField(viewer, (PetscObject) dm, DMPlexVTKWriteAll, f, ft, PETSC_TRUE,(PetscObject) locv);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&locv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_Local(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if (isvtk || ishdf5 || isdraw || isglvis) {
    PetscInt    i,numFields;
    PetscObject fe;
    PetscBool   fem = PETSC_FALSE;
    Vec         locv = v;
    const char  *name;
    PetscInt    step;
    PetscReal   time;

    ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
    for (i=0; i<numFields; i++) {
      ierr = DMGetField(dm, i, NULL, &fe);CHKERRQ(ierr);
      if (fe->classid == PETSCFE_CLASSID) { fem = PETSC_TRUE; break; }
    }
    if (fem) {
      ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
      ierr = VecCopy(v, locv);CHKERRQ(ierr);
      ierr = DMGetOutputSequenceNumber(dm, NULL, &time);CHKERRQ(ierr);
      ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL);CHKERRQ(ierr);
    }
    if (isvtk) {
      ierr = VecView_Plex_Local_VTK(locv, viewer);CHKERRQ(ierr);
    } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
      ierr = VecView_Plex_Local_HDF5_Internal(locv, viewer);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
    } else if (isdraw) {
      ierr = VecView_Plex_Local_Draw(locv, viewer);CHKERRQ(ierr);
    } else if (isglvis) {
      ierr = DMGetOutputSequenceNumber(dm, &step, NULL);CHKERRQ(ierr);
      ierr = PetscViewerGLVisSetSnapId(viewer, step);CHKERRQ(ierr);
      ierr = VecView_GLVis(locv, viewer);CHKERRQ(ierr);
    }
    if (fem) {ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);}
  } else {
    PetscBool isseq;

    ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
    if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
    else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if (isvtk || isdraw || isglvis) {
    Vec         locv;
    const char *name;

    ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = VecView_Plex_Local(locv, viewer);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = VecView_Plex_HDF5_Internal(v, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    PetscBool isseq;

    ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
    if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
    else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(originalv, &dm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) originalv, &comm);CHKERRQ(ierr);
  if (!dm) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,  &isvtk);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_NATIVE) {
    /* Natural ordering is the common case for DMDA, NATIVE means plain vector, for PLEX is the opposite */
    /* this need a better fix */
    if (dm->useNatural) {
      if (dm->sfNatural) {
        const char *vecname;
        PetscInt    n, nroots;

        ierr = VecGetLocalSize(originalv, &n);CHKERRQ(ierr);
        ierr = PetscSFGetGraph(dm->sfNatural, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
        if (n == nroots) {
          ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
          ierr = DMPlexGlobalToNaturalBegin(dm, originalv, v);CHKERRQ(ierr);
          ierr = DMPlexGlobalToNaturalEnd(dm, originalv, v);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) originalv, &vecname);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) v, vecname);CHKERRQ(ierr);
        } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "DM global to natural SF only handles global vectors");
      } else SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created");
    } else v = originalv;
  } else v = originalv;

  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = VecView_Plex_HDF5_Native_Internal(v, viewer);CHKERRQ(ierr);
#else
    SETERRQ(comm, PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isvtk) {
    SETERRQ(comm, PETSC_ERR_SUP, "VTK format does not support viewing in natural order. Please switch to HDF5.");
  } else {
    PetscBool isseq;

    ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
    if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
    else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
  }
  if (v != originalv) {ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_Local(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5);CHKERRQ(ierr);
  if (ishdf5) {
    DM          dmBC;
    Vec         gv;
    const char *name;

    ierr = DMGetOutputDM(dm, &dmBC);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmBC, &gv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) gv, name);CHKERRQ(ierr);
    ierr = VecLoad_Default(gv, viewer);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmBC, gv, INSERT_VALUES, v);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmBC, gv, INSERT_VALUES, v);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmBC, &gv);CHKERRQ(ierr);
  } else {
    ierr = VecLoad_Default(v, viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5);CHKERRQ(ierr);
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = VecLoad_Plex_HDF5_Internal(v, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    ierr = VecLoad_Default(v, viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_Native(Vec originalv, PetscViewer viewer)
{
  DM                dm;
  PetscViewerFormat format;
  PetscBool         ishdf5;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(originalv, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject) originalv), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_NATIVE) {
    if (dm->useNatural) {
      if (dm->sfNatural) {
        if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
          Vec         v;
          const char *vecname;

          ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
          ierr = PetscObjectGetName((PetscObject) originalv, &vecname);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) v, vecname);CHKERRQ(ierr);
          ierr = VecLoad_Plex_HDF5_Native_Internal(v, viewer);CHKERRQ(ierr);
          ierr = DMPlexNaturalToGlobalBegin(dm, v, originalv);CHKERRQ(ierr);
          ierr = DMPlexNaturalToGlobalEnd(dm, v, originalv);CHKERRQ(ierr);
          ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);
#else
          SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
        } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Reading in natural order is not supported for anything but HDF5.");
      }
    } else {
      ierr = VecLoad_Default(originalv, viewer);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMPlexGetCellTypeLabel(dm, &celltypeLabel);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &a);CHKERRQ(ierr);
  name[0]     = "vertex";
  name[1]     = "edge";
  name[dim-1] = "face";
  name[dim]   = "cell";
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl, ct;

    ierr = DMLabelGetValue(celltypeLabel, c, &ct);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Geometry for cell %D polytope type %s:\n", c, DMPolytopeTypes[ct]);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      PetscInt point = closure[cl], depth, dof, off, d, p;

      if ((point < pStart) || (point >= pEnd)) continue;
      ierr = PetscSectionGetDof(coordSection, point, &dof);CHKERRQ(ierr);
      if (!dof) continue;
      ierr = DMLabelGetValue(depthLabel, point, &depth);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, point, &off);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "%s %D coords:", name[depth], point);CHKERRQ(ierr);
      for (p = 0; p < dof/dim; ++p) {
        ierr = PetscViewerASCIIPrintf(viewer, " (");CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) {
          if (d > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
          ierr = PetscViewerASCIIPrintf(viewer, "%g", (double) PetscRealPart(a[off+p*dim+d]));CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, ")");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(coordinates, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Plex          *mesh = (DM_Plex*) dm->data;
  DM                cdm;
  DMLabel           markers;
  PetscSection      coordSection;
  Vec               coordinates;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    const char *name;
    PetscInt    dim, cellHeight, maxConeSize, maxSupportSize;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt rank, size;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
    if (name) {ierr = PetscViewerASCIIPrintf(viewer, "%s in %D dimension%s:\n", name, dim, dim == 1 ? "" : "s");CHKERRQ(ierr);}
    else      {ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %D dimension%s:\n", dim, dim == 1 ? "" : "s");CHKERRQ(ierr);}
    if (cellHeight) {ierr = PetscViewerASCIIPrintf(viewer, "  Cells are at height %D\n", cellHeight);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "Supports:\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Max support size: %D\n", rank, maxSupportSize);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      for (s = off; s < off+dof; ++s) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D ----> %D\n", rank, p, mesh->supports[s]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Cones:\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Max cone size: %D\n", rank, maxConeSize);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, c;

      ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
      for (c = off; c < off+dof; ++c) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D <---- %D (%D)\n", rank, p, mesh->cones[c], mesh->coneOrientations[c]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    if (coordSection && coordinates) {
      ierr = PetscSectionVecView(coordSection, coordinates, viewer);CHKERRQ(ierr);
    }
    ierr = DMGetLabel(dm, "marker", &markers);CHKERRQ(ierr);
    if (markers) {ierr = DMLabelView(markers,viewer);CHKERRQ(ierr);}
    if (size > 1) {
      PetscSF sf;

      ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
      ierr = PetscSFView(sf, viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_LATEX) {
    const char  *name, *color;
    const char  *defcolors[3]  = {"gray", "orange", "green"};
    const char  *deflcolors[4] = {"blue", "cyan", "red", "magenta"};
    char         lname[PETSC_MAX_PATH_LEN];
    PetscReal    scale         = 2.0;
    PetscReal    tikzscale     = 1.0;
    PetscBool    useNumbers    = PETSC_TRUE, useLabels, useColors;
    double       tcoords[3];
    PetscScalar *coords;
    PetscInt     numLabels, l, numColors, numLColors, dim, depth, cStart, cEnd, c, vStart, vEnd, v, eStart = 0, eEnd = 0, e, p;
    PetscMPIInt  rank, size;
    char         **names, **colors, **lcolors;
    PetscBool    plotEdges, flg, lflg;
    PetscBT      wp = NULL;
    PetscInt     pEnd, pStart;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
    numLabels  = PetscMax(numLabels, 10);
    numColors  = 10;
    numLColors = 10;
    ierr = PetscCalloc3(numLabels, &names, numColors, &colors, numLColors, &lcolors);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_scale", &scale, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_tikzscale", &tikzscale, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_numbers", &useNumbers, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetStringArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_labels", names, &numLabels, &useLabels);CHKERRQ(ierr);
    if (!useLabels) numLabels = 0;
    ierr = PetscOptionsGetStringArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_colors", colors, &numColors, &useColors);CHKERRQ(ierr);
    if (!useColors) {
      numColors = 3;
      for (c = 0; c < numColors; ++c) {ierr = PetscStrallocpy(defcolors[c], &colors[c]);CHKERRQ(ierr);}
    }
    ierr = PetscOptionsGetStringArray(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_lcolors", lcolors, &numLColors, &useColors);CHKERRQ(ierr);
    if (!useColors) {
      numLColors = 4;
      for (c = 0; c < numLColors; ++c) {ierr = PetscStrallocpy(deflcolors[c], &lcolors[c]);CHKERRQ(ierr);}
    }
    ierr = PetscOptionsGetString(((PetscObject) viewer)->options, ((PetscObject) viewer)->prefix, "-dm_plex_view_label_filter", lname, PETSC_MAX_PATH_LEN, &lflg);CHKERRQ(ierr);
    plotEdges = (PetscBool)(depth > 1 && useNumbers && dim < 3);
    ierr = PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_edges", &plotEdges, &flg);CHKERRQ(ierr);
    if (flg && plotEdges && depth < dim) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Mesh must be interpolated");
    if (depth < dim) plotEdges = PETSC_FALSE;

    /* filter points with labelvalue != labeldefaultvalue */
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (lflg) {
      DMLabel lbl;

      ierr = DMGetLabel(dm, lname, &lbl);CHKERRQ(ierr);
      if (lbl) {
        PetscInt val, defval;

        ierr = DMLabelGetDefaultValue(lbl, &defval);CHKERRQ(ierr);
        ierr = PetscBTCreate(pEnd-pStart, &wp);CHKERRQ(ierr);
        for (c = pStart;  c < pEnd; c++) {
          PetscInt *closure = NULL;
          PetscInt  closureSize;

          ierr = DMLabelGetValue(lbl, c, &val);CHKERRQ(ierr);
          if (val == defval) continue;

          ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          for (p = 0; p < closureSize*2; p += 2) {
            ierr = PetscBTSet(wp, closure[p] - pStart);CHKERRQ(ierr);
          }
          ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        }
      }
    }

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\
\\documentclass[tikz]{standalone}\n\n\
\\usepackage{pgflibraryshapes}\n\
\\usetikzlibrary{backgrounds}\n\
\\usetikzlibrary{arrows}\n\
\\begin{document}\n");CHKERRQ(ierr);
    if (size > 1) {
      ierr = PetscViewerASCIIPrintf(viewer, "%s for process ", name);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {
        if (p > 0 && p == size-1) {
          ierr = PetscViewerASCIIPrintf(viewer, ", and ", colors[p%numColors], p);CHKERRQ(ierr);
        } else if (p > 0) {
          ierr = PetscViewerASCIIPrintf(viewer, ", ", colors[p%numColors], p);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "{\\textcolor{%s}%D}", colors[p%numColors], p);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, ".\n\n\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\\begin{tikzpicture}[scale = %g,font=\\fontsize{8}{8}\\selectfont]\n", (double) tikzscale);CHKERRQ(ierr);

    /* Plot vertices */
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  off, dof, d;
      PetscBool isLabeled = PETSC_FALSE;

      if (wp && !PetscBTLookup(wp,v - pStart)) continue;
      ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\path (");CHKERRQ(ierr);
      if (PetscUnlikely(dof > 3)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"coordSection vertex %D has dof %D > 3",v,dof);
      for (d = 0; d < dof; ++d) {
        tcoords[d] = (double) (scale*PetscRealPart(coords[off+d]));
        tcoords[d] = PetscAbs(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
      }
      /* Rotate coordinates since PGF makes z point out of the page instead of up */
      if (dim == 3) {PetscReal tmp = tcoords[1]; tcoords[1] = tcoords[2]; tcoords[2] = -tmp;}
      for (d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%g", (double) tcoords[d]);CHKERRQ(ierr);
      }
      color = colors[rank%numColors];
      for (l = 0; l < numLabels; ++l) {
        PetscInt val;
        ierr = DMGetLabelValue(dm, names[l], v, &val);CHKERRQ(ierr);
        if (val >= 0) {color = lcolors[l%numLColors]; isLabeled = PETSC_TRUE; break;}
      }
      if (useNumbers) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [draw,shape=circle,color=%s] {%D};\n", v, rank, color, v);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [fill,inner sep=%dpt,shape=circle,color=%s] {};\n", v, rank, !isLabeled ? 1 : 2, color);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    /* Plot cells */
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
    if (dim == 3 || !useNumbers) {
      for (e = eStart; e < eEnd; ++e) {
        const PetscInt *cone;

        if (wp && !PetscBTLookup(wp,e - pStart)) continue;
        color = colors[rank%numColors];
        for (l = 0; l < numLabels; ++l) {
          PetscInt val;
          ierr = DMGetLabelValue(dm, names[l], e, &val);CHKERRQ(ierr);
          if (val >= 0) {color = lcolors[l%numLColors]; break;}
        }
        ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] (%D_%d) -- (%D_%d);\n", color, cone[0], rank, cone[1], rank);CHKERRQ(ierr);
      }
    } else {
      for (c = cStart; c < cEnd; ++c) {
        PetscInt *closure = NULL;
        PetscInt  closureSize, firstPoint = -1;

        if (wp && !PetscBTLookup(wp,c - pStart)) continue;
        ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\draw[color=%s] ", colors[rank%numColors]);CHKERRQ(ierr);
        for (p = 0; p < closureSize*2; p += 2) {
          const PetscInt point = closure[p];

          if ((point < vStart) || (point >= vEnd)) continue;
          if (firstPoint >= 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- ");CHKERRQ(ierr);}
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(%D_%d)", point, rank);CHKERRQ(ierr);
          if (firstPoint < 0) firstPoint = point;
        }
        /* Why doesn't this work? ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- cycle;\n");CHKERRQ(ierr); */
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, " -- (%D_%d);\n", firstPoint, rank);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
    }
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      double    ccoords[3] = {0.0, 0.0, 0.0};
      PetscBool isLabeled  = PETSC_FALSE;
      PetscInt *closure    = NULL;
      PetscInt  closureSize, dof, d, n = 0;

      if (wp && !PetscBTLookup(wp,c - pStart)) continue;
      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\path (");CHKERRQ(ierr);
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = closure[p];
        PetscInt       off;

        if ((point < vStart) || (point >= vEnd)) continue;
        ierr = PetscSectionGetDof(coordSection, point, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, point, &off);CHKERRQ(ierr);
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
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%g", (double) ccoords[d]);CHKERRQ(ierr);
      }
      color = colors[rank%numColors];
      for (l = 0; l < numLabels; ++l) {
        PetscInt val;
        ierr = DMGetLabelValue(dm, names[l], c, &val);CHKERRQ(ierr);
        if (val >= 0) {color = lcolors[l%numLColors]; isLabeled = PETSC_TRUE; break;}
      }
      if (useNumbers) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [draw,shape=circle,color=%s] {%D};\n", c, rank, color, c);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [fill,inner sep=%dpt,shape=circle,color=%s] {};\n", c, rank, !isLabeled ? 1 : 2, color);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    /* Plot edges */
    if (plotEdges) {
      ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "\\path\n");CHKERRQ(ierr);
      for (e = eStart; e < eEnd; ++e) {
        const PetscInt *cone;
        PetscInt        coneSize, offA, offB, dof, d;

        if (wp && !PetscBTLookup(wp,e - pStart)) continue;
        ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
        if (coneSize != 2) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Edge %D cone should have two vertices, not %D", e, coneSize);
        ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(coordSection, cone[0], &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cone[0], &offA);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cone[1], &offB);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(");CHKERRQ(ierr);
        for (d = 0; d < dof; ++d) {
          tcoords[d] = (double) (0.5*scale*PetscRealPart(coords[offA+d]+coords[offB+d]));
          tcoords[d] = PetscAbs(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
        }
        /* Rotate coordinates since PGF makes z point out of the page instead of up */
        if (dim == 3) {PetscReal tmp = tcoords[1]; tcoords[1] = tcoords[2]; tcoords[2] = -tmp;}
        for (d = 0; d < dof; ++d) {
          if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%g", (double)tcoords[d]);CHKERRQ(ierr);
        }
        color = colors[rank%numColors];
        for (l = 0; l < numLabels; ++l) {
          PetscInt val;
          ierr = DMGetLabelValue(dm, names[l], v, &val);CHKERRQ(ierr);
          if (val >= 0) {color = lcolors[l%numLColors]; break;}
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, ") node(%D_%d) [draw,shape=circle,color=%s] {%D} --\n", e, rank, color, e);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "(0,0);\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{tikzpicture}\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{document}\n", name);CHKERRQ(ierr);
    for (l = 0; l < numLabels;  ++l) {ierr = PetscFree(names[l]);CHKERRQ(ierr);}
    for (c = 0; c < numColors;  ++c) {ierr = PetscFree(colors[c]);CHKERRQ(ierr);}
    for (c = 0; c < numLColors; ++c) {ierr = PetscFree(lcolors[c]);CHKERRQ(ierr);}
    ierr = PetscFree3(names, colors, lcolors);CHKERRQ(ierr);
    ierr = PetscBTDestroy(&wp);CHKERRQ(ierr);
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

    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    ierr = MPI_Comm_split_type(comm,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&ncomm);CHKERRQ(ierr);
#endif
    if (ncomm != MPI_COMM_NULL) {
      ierr = MPI_Comm_group(comm,&ggroup);CHKERRQ(ierr);
      ierr = MPI_Comm_group(ncomm,&ngroup);CHKERRQ(ierr);
      d1   = 0;
      ierr = MPI_Group_translate_ranks(ngroup,1,&d1,ggroup,&d2);CHKERRQ(ierr);
      nid  = d2;
      ierr = MPI_Group_free(&ggroup);CHKERRQ(ierr);
      ierr = MPI_Group_free(&ngroup);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&ncomm);CHKERRQ(ierr);
    } else nid = 0.0;

    /* Get connectivity */
    ierr = DMPlexGetVTKCellHeight(dm,&cellHeight);CHKERRQ(ierr);
    ierr = DMPlexCreatePartitionerGraph(dm,cellHeight,&numVertices,&start,&adjacency,&gid);CHKERRQ(ierr);

    /* filter overlapped local cells */
    ierr = DMPlexGetHeightStratum(dm,cellHeight,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = ISGetIndices(gid,&idxs);CHKERRQ(ierr);
    ierr = ISGetLocalSize(gid,&cum);CHKERRQ(ierr);
    ierr = PetscMalloc1(cum,&idxs2);CHKERRQ(ierr);
    for (c = cStart, cum = 0; c < cEnd; c++) {
      if (idxs[c-cStart] < 0) continue;
      idxs2[cum++] = idxs[c-cStart];
    }
    ierr = ISRestoreIndices(gid,&idxs);CHKERRQ(ierr);
    if (numVertices != cum) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected %D != %D",numVertices,cum);
    ierr = ISDestroy(&gid);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,numVertices,idxs2,PETSC_OWN_POINTER,&gid);CHKERRQ(ierr);

    /* support for node-aware cell locality */
    ierr = ISCreateGeneral(comm,start[numVertices],adjacency,PETSC_USE_POINTER,&acis);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,start[numVertices],&acown);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,numVertices,PETSC_DECIDE,&cown);CHKERRQ(ierr);
    ierr = VecGetArray(cown,&array);CHKERRQ(ierr);
    for (c = 0; c < numVertices; c++) array[c] = nid;
    ierr = VecRestoreArray(cown,&array);CHKERRQ(ierr);
    ierr = VecScatterCreate(cown,acis,acown,NULL,&sct);CHKERRQ(ierr);
    ierr = VecScatterBegin(sct,cown,acown,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sct,cown,acown,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = ISDestroy(&acis);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&sct);CHKERRQ(ierr);
    ierr = VecDestroy(&cown);CHKERRQ(ierr);

    /* compute edgeCut */
    for (c = 0, cum = 0; c < numVertices; c++) cum = PetscMax(cum,start[c+1]-start[c]);
    ierr = PetscMalloc1(cum,&work);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(gid,&g2l);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingSetType(g2l,ISLOCALTOGLOBALMAPPINGHASH);CHKERRQ(ierr);
    ierr = ISDestroy(&gid);CHKERRQ(ierr);
    ierr = VecGetArray(acown,&array);CHKERRQ(ierr);
    for (c = 0, ect = 0, ectn = 0; c < numVertices; c++) {
      PetscInt totl;

      totl = start[c+1]-start[c];
      ierr = ISGlobalToLocalMappingApply(g2l,IS_GTOLM_MASK,totl,adjacency+start[c],NULL,work);CHKERRQ(ierr);
      for (i = 0; i < totl; i++) {
        if (work[i] < 0) {
          ect  += 1;
          ectn += (array[i + start[c]] != nid) ? 0 : 1;
        }
      }
    }
    ierr  = PetscFree(work);CHKERRQ(ierr);
    ierr  = VecRestoreArray(acown,&array);CHKERRQ(ierr);
    lm[0] = numVertices > 0 ?  numVertices : PETSC_MAX_INT;
    lm[1] = -numVertices;
    ierr  = MPIU_Allreduce(lm,gm,2,MPIU_INT64,MPI_MIN,comm);CHKERRQ(ierr);
    ierr  = PetscViewerASCIIPrintf(viewer,"  Cell balance: %.2f (max %D, min %D",-((double)gm[1])/((double)gm[0]),-(PetscInt)gm[1],(PetscInt)gm[0]);CHKERRQ(ierr);
    lm[0] = ect; /* edgeCut */
    lm[1] = ectn; /* node-aware edgeCut */
    lm[2] = numVertices > 0 ? 0 : 1; /* empty processes */
    ierr  = MPIU_Allreduce(lm,gm,3,MPIU_INT64,MPI_SUM,comm);CHKERRQ(ierr);
    ierr  = PetscViewerASCIIPrintf(viewer,", empty %D)\n",(PetscInt)gm[2]);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    ierr  = PetscViewerASCIIPrintf(viewer,"  Edge Cut: %D (on node %.3f)\n",(PetscInt)(gm[0]/2),gm[0] ? ((double)(gm[1]))/((double)gm[0]) : 1.);CHKERRQ(ierr);
#else
    ierr  = PetscViewerASCIIPrintf(viewer,"  Edge Cut: %D (on node %.3f)\n",(PetscInt)(gm[0]/2),0.0);CHKERRQ(ierr);
#endif
    ierr  = ISLocalToGlobalMappingDestroy(&g2l);CHKERRQ(ierr);
    ierr  = PetscFree(start);CHKERRQ(ierr);
    ierr  = PetscFree(adjacency);CHKERRQ(ierr);
    ierr  = VecDestroy(&acown);CHKERRQ(ierr);
  } else {
    MPI_Comm    comm;
    PetscInt   *sizes, *hybsizes, *ghostsizes;
    PetscInt    locDepth, depth, cellHeight, dim, d, pMax[4];
    PetscInt    pStart, pEnd, p, gcStart, gcEnd, gcNum;
    PetscInt    numLabels, l;
    const char *name;
    PetscMPIInt size;

    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    if (name) {ierr = PetscViewerASCIIPrintf(viewer, "%s in %D dimension%s:\n", name, dim, dim == 1 ? "" : "s");CHKERRQ(ierr);}
    else      {ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %D dimension%s:\n", dim, dim == 1 ? "" : "s");CHKERRQ(ierr);}
    if (cellHeight) {ierr = PetscViewerASCIIPrintf(viewer, "  Cells are at height %D\n", cellHeight);CHKERRQ(ierr);}
    ierr = DMPlexGetDepth(dm, &locDepth);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&locDepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(dm, &pMax[depth], depth > 0 ? &pMax[depth-1] : NULL, depth > 1 ? &pMax[depth - 2] : NULL, &pMax[0]);CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &gcStart, &gcEnd);CHKERRQ(ierr);
    gcNum = gcEnd - gcStart;
    ierr = PetscCalloc3(size,&sizes,size,&hybsizes,size,&ghostsizes);CHKERRQ(ierr);
    if (depth == 1) {
      ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      pMax[0] -= pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = MPI_Gather(&pMax[0], 1, MPIU_INT, hybsizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = MPI_Gather(&gcNum, 1, MPIU_INT, ghostsizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", 0);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {
        if (hybsizes[p] >= 0) {ierr = PetscViewerASCIIPrintf(viewer, " %D (%D)", sizes[p], sizes[p] - hybsizes[p]);CHKERRQ(ierr);}
        else                  {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      pMax[depth] -= pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = MPI_Gather(&pMax[depth], 1, MPIU_INT, hybsizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", dim);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {
        ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);
        if (hybsizes[p] >= 0)   {ierr = PetscViewerASCIIPrintf(viewer, " (%D)", sizes[p] - hybsizes[p]);CHKERRQ(ierr);}
        if (ghostsizes[p] > 0) {ierr = PetscViewerASCIIPrintf(viewer, " [%D]", ghostsizes[p]);CHKERRQ(ierr);}
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    } else {
      PetscMPIInt rank;
      ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
      for (d = 0; d <= dim; d++) {
        ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        pEnd    -= pStart;
        pMax[d] -= pStart;
        ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = MPI_Gather(&pMax[d], 1, MPIU_INT, hybsizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
        if (d == dim) {ierr = MPI_Gather(&gcNum, 1, MPIU_INT, ghostsizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", d);CHKERRQ(ierr);
        for (p = 0; p < size; ++p) {
          if (!rank) {
            ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);
            if (hybsizes[p] >= 0) {ierr = PetscViewerASCIIPrintf(viewer, " (%D)", sizes[p] - hybsizes[p]);CHKERRQ(ierr);}
            if (d == dim && ghostsizes[p] > 0) {ierr = PetscViewerASCIIPrintf(viewer, " [%D]", ghostsizes[p]);CHKERRQ(ierr);}
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscFree3(sizes,hybsizes,ghostsizes);CHKERRQ(ierr);
    ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
    if (numLabels) {ierr = PetscViewerASCIIPrintf(viewer, "Labels:\n");CHKERRQ(ierr);}
    for (l = 0; l < numLabels; ++l) {
      DMLabel         label;
      const char     *name;
      IS              valueIS;
      const PetscInt *values;
      PetscInt        numValues, v;

      ierr = DMGetLabelName(dm, l, &name);CHKERRQ(ierr);
      ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
      ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %s: %D strata with value/size (", name, numValues);CHKERRQ(ierr);
      ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
      ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
      for (v = 0; v < numValues; ++v) {
        PetscInt size;

        ierr = DMLabelGetStratumSize(label, values[v], &size);CHKERRQ(ierr);
        if (v > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPrintf(viewer, "%D (%D)", values[v], size);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, ")\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
      ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
      ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    }
    /* If no fields are specified, people do not want to see adjacency */
    if (dm->Nf) {
      PetscInt f;

      for (f = 0; f < dm->Nf; ++f) {
        const char *name;

        ierr = PetscObjectGetName(dm->fields[f].disc, &name);CHKERRQ(ierr);
        if (numLabels) {ierr = PetscViewerASCIIPrintf(viewer, "Field %s:\n", name);CHKERRQ(ierr);}
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        if (dm->fields[f].label) {ierr = DMLabelView(dm->fields[f].label, viewer);CHKERRQ(ierr);}
        if (dm->fields[f].adjacency[0]) {
          if (dm->fields[f].adjacency[1]) {ierr = PetscViewerASCIIPrintf(viewer, "adjacency FVM++\n");CHKERRQ(ierr);}
          else                            {ierr = PetscViewerASCIIPrintf(viewer, "adjacency FVM\n");CHKERRQ(ierr);}
        } else {
          if (dm->fields[f].adjacency[1]) {ierr = PetscViewerASCIIPrintf(viewer, "adjacency FEM\n");CHKERRQ(ierr);}
          else                            {ierr = PetscViewerASCIIPrintf(viewer, "adjacency FUNKY\n");CHKERRQ(ierr);}
        }
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
    ierr = DMGetCoarseDM(dm, &cdm);CHKERRQ(ierr);
    if (cdm) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = DMPlexView_Ascii(cdm, viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
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
  PetscBool          isnull;
  PetscInt           dim, vStart, vEnd, cStart, cEnd, c, N;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D", dim);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw, &isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = PetscDrawSetTitle(draw, "Mesh");CHKERRQ(ierr);

  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));   bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c+1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c+1]));
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&bound[0],xyl,2,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&bound[2],xyr,2,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw, xyl[0], xyl[1], xyr[0], xyr[1]);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar   *coords = NULL;
    DMPolytopeType ct;
    PetscInt       numCoords;

    ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
      ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2);CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                                PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                                PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                                PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2);CHKERRQ(ierr);
      ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]),
                                PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                                PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2,
                                PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS-2) + 2);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar   *coords = NULL;
    DMPolytopeType ct;
    PetscInt       numCoords;

    ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
  }
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

PetscErrorCode DMView_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, ishdf5, isvtk, isdraw, flg, isglvis, isexodus;
  char           name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII,    &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,      &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,     &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,     &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS,    &isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWEREXODUSII, &isexodus);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_GLVIS) {
      ierr = DMPlexView_GLVis(dm, viewer);CHKERRQ(ierr);
    } else {
      ierr = DMPlexView_Ascii(dm, viewer);CHKERRQ(ierr);
    }
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = DMPlexView_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isvtk) {
    ierr = DMPlexVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = DMPlexView_Draw(dm, viewer);CHKERRQ(ierr);
  } else if (isglvis) {
    ierr = DMPlexView_GLVis(dm, viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_EXODUSII)
  } else if (isexodus) {
    int exoid;
    PetscInt cStart, cEnd, c;

    ierr = DMCreateLabel(dm, "Cell Sets");CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {ierr = DMSetLabelValue(dm, "Cell Sets", c, 1);CHKERRQ(ierr);}
    ierr = PetscViewerExodusIIGetId(viewer, &exoid);CHKERRQ(ierr);
    ierr = DMPlexView_ExodusII_Internal(dm, exoid, 1);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlex writing", ((PetscObject)viewer)->type_name);
  }
  /* Optionally view the partition */
  ierr = PetscOptionsHasName(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    Vec ranks;
    ierr = DMPlexCreateRankField(dm, &ranks);CHKERRQ(ierr);
    ierr = VecView(ranks, viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&ranks);CHKERRQ(ierr);
  }
  /* Optionally view a label */
  ierr = PetscOptionsGetString(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_label_view", name, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (flg) {
    DMLabel label;
    Vec     val;

    ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
    if (!label) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Label %s provided to -dm_label_view does not exist in this DM", name);
    ierr = DMPlexCreateLabelField(dm, label, &val);CHKERRQ(ierr);
    ierr = VecView(val, viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&val);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMLoad_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,   &ishdf5);CHKERRQ(ierr);
  if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_HDF5_XDMF || format == PETSC_VIEWER_HDF5_VIZ) {
      ierr = DMPlexLoad_HDF5_Xdmf_Internal(dm, viewer);CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_HDF5_PETSC || format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_NATIVE) {
      ierr = DMPlexLoad_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 input.", PetscViewerFormats[format]);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlex loading", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Plex(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertBoundaryValues_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"DMCreateNeumannOverlap_C", NULL);CHKERRQ(ierr);
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  ierr = PetscSectionDestroy(&mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->cones);CHKERRQ(ierr);
  ierr = PetscFree(mesh->coneOrientations);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->subdomainSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  ierr = PetscFree(mesh->facesTmp);CHKERRQ(ierr);
  ierr = PetscFree(mesh->tetgenOpts);CHKERRQ(ierr);
  ierr = PetscFree(mesh->triangleOpts);CHKERRQ(ierr);
  ierr = PetscPartitionerDestroy(&mesh->partitioner);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&mesh->subpointMap);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->globalVertexNumbers);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->globalCellNumbers);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->anchorSection);CHKERRQ(ierr);
  ierr = ISDestroy(&mesh->anchorIS);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->parentSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->parents);CHKERRQ(ierr);
  ierr = PetscFree(mesh->childIDs);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->childSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->children);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh->referenceTree);CHKERRQ(ierr);
  ierr = PetscGridHashDestroy(&mesh->lbox);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_Plex(DM dm, Mat *J)
{
  PetscSection           sectionGlobal;
  PetscInt               bs = -1, mbs;
  PetscInt               localSize;
  PetscBool              isShell, isBlock, isSeqBlock, isMPIBlock, isSymBlock, isSymSeqBlock, isSymMPIBlock, isMatIS;
  PetscErrorCode         ierr;
  MatType                mtype;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  mtype = dm->mattype;
  ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  /* ierr = PetscSectionGetStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr); */
  ierr = PetscSectionGetConstrainedStorageSize(sectionGlobal, &localSize);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)dm), J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
  ierr = MatGetBlockSize(*J, &mbs);CHKERRQ(ierr);
  if (mbs > 1) bs = mbs;
  ierr = PetscStrcmp(mtype, MATSHELL, &isShell);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATBAIJ, &isBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSEQBAIJ, &isSeqBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATMPIBAIJ, &isMPIBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSBAIJ, &isSymBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATIS, &isMatIS);CHKERRQ(ierr);
  if (!isShell) {
    PetscSection subSection;
    PetscBool    fillMatrix = (PetscBool)(!dm->prealloc_only && !isMatIS);
    PetscInt    *dnz, *onz, *dnzu, *onzu, bsLocal[2], bsMinMax[2], *ltogidx, lsize;
    PetscInt     pStart, pEnd, p, dof, cdof;

    /* Set localtoglobalmapping on the matrix for MatSetValuesLocal() to work (it also creates the local matrices in case of MATIS) */
    if (isMatIS) { /* need a different l2g map than the one computed by DMGetLocalToGlobalMapping */
      PetscSection section;
      PetscInt     size;

      ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
      ierr = PetscMalloc1(size,&ltogidx);CHKERRQ(ierr);
      ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);
    } else {
      ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
    }
    ierr = PetscSectionGetChart(sectionGlobal, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart, lsize = 0; p < pEnd; ++p) {
      PetscInt bdof;

      ierr = PetscSectionGetDof(sectionGlobal, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(sectionGlobal, p, &cdof);CHKERRQ(ierr);
      dof  = dof < 0 ? -(dof+1) : dof;
      bdof = cdof && (dof-cdof) ? 1 : dof;
      if (dof) {
        if (bs < 0)          {bs = bdof;}
        else if (bs != bdof) {bs = 1; if (!isMatIS) break;}
      }
      if (isMatIS) {
        PetscInt loff,c,off;
        ierr = PetscSectionGetOffset(subSection, p, &loff);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, p, &off);CHKERRQ(ierr);
        for (c = 0; c < dof-cdof; ++c, ++lsize) ltogidx[loff+c] = off > -1 ? off+c : -(off+1)+c;
      }
    }
    /* Must have same blocksize on all procs (some might have no points) */
    bsLocal[0] = bs < 0 ? PETSC_MAX_INT : bs; bsLocal[1] = bs;
    ierr = PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax);CHKERRQ(ierr);
    if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
    else                            {bs = bsMinMax[0];}
    bs = PetscMax(1,bs);
    if (isMatIS) { /* Must reduce indices by blocksize */
      PetscInt l;

      lsize = lsize/bs;
      if (bs > 1) for (l = 0; l < lsize; ++l) ltogidx[l] = ltogidx[l*bs]/bs;
      ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)dm), bs, lsize, ltogidx, PETSC_OWN_POINTER, &ltog);CHKERRQ(ierr);
    }
    ierr = MatSetLocalToGlobalMapping(*J,ltog,ltog);CHKERRQ(ierr);
    if (isMatIS) {
      ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
    }
    ierr = PetscCalloc4(localSize/bs, &dnz, localSize/bs, &onz, localSize/bs, &dnzu, localSize/bs, &onzu);CHKERRQ(ierr);
    ierr = DMPlexPreallocateOperator(dm, bs, dnz, onz, dnzu, onzu, *J, fillMatrix);CHKERRQ(ierr);
    ierr = PetscFree4(dnz, onz, dnzu, onzu);CHKERRQ(ierr);
  }
  ierr = MatSetDM(*J, dm);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->subdomainSection) {
    PetscSection section;
    PetscSF      sf;

    ierr = PetscSFCreate(PETSC_COMM_SELF,&sf);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(section,sf,PETSC_FALSE,PETSC_TRUE,&mesh->subdomainSection);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetChart(mesh->coneSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(mesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  ierr = PetscSectionGetDof(mesh->coneSection, p, size);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetDof(mesh->coneSection, p, size);CHKERRQ(ierr);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionAddDof(mesh->coneSection, p, size);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &csize);CHKERRQ(ierr);

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

.seealso: DMPlexCreate(), DMPlexSetCone(), DMPlexGetConeTuple(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetCone(DM dm, PetscInt p, const PetscInt *cone[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cone, 3);
  ierr  = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  *cone = &mesh->cones[off];
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetConeTuple - Return the points on the in-edges of several points in the DAG

  Not collective

  Input Parameters:
+ dm - The DMPlex
- p - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &cs);CHKERRQ(ierr);
  ierr = PetscSectionExtractDofsFromArray(cs, MPIU_INT, cones, p, &newcs, pCones ? ((void**)&newarr) : NULL);CHKERRQ(ierr);
  if (pConesSection) *pConesSection = newcs;
  if (pCones) {
    ierr = PetscSectionGetStorageSize(newcs, &n);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)p), n, newarr, PETSC_OWN_POINTER, pCones);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(points, IS_CLASSID, 2);
  PetscValidPointer(expandedPoints, 3);
  ierr = DMPlexGetConeRecursive(dm, points, &depth, &expandedPointsAll, NULL);CHKERRQ(ierr);
  *expandedPoints = expandedPointsAll[0];
  ierr = PetscObjectReference((PetscObject)expandedPointsAll[0]);
  ierr = DMPlexRestoreConeRecursive(dm, points, &depth, &expandedPointsAll, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetConeRecursive - Expand each given point into its cone points and do that recursively until we end up just with vertices (DAG points of depth 0, i.e. without cones).

  Not collective

  Input Parameters:
+ dm - The DMPlex
- points - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(points, IS_CLASSID, 2);
  if (depth) PetscValidIntPointer(depth, 3);
  if (expandedPoints) PetscValidPointer(expandedPoints, 4);
  if (sections) PetscValidPointer(sections, 5);
  ierr = ISGetLocalSize(points, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &arr0);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth_);CHKERRQ(ierr);
  ierr = PetscCalloc1(depth_, &expandedPoints_);CHKERRQ(ierr);
  ierr = PetscCalloc1(depth_, &sections_);CHKERRQ(ierr);
  arr = (PetscInt*) arr0; /* this is ok because first generation of arr is not modified */
  for (d=depth_-1; d>=0; d--) {
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &sections_[d]);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(sections_[d], 0, n);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = DMPlexGetDepthStratum(dm, d+1, &start, &end);CHKERRQ(ierr);
      if (arr[i] >= start && arr[i] < end) {
        ierr = DMPlexGetConeSize(dm, arr[i], &cn);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(sections_[d], i, cn);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionSetDof(sections_[d], i, 1);CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionSetUp(sections_[d]);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(sections_[d], &newn);CHKERRQ(ierr);
    ierr = PetscMalloc1(newn, &newarr);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscSectionGetDof(sections_[d], i, &cn);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sections_[d], i, &co);CHKERRQ(ierr);
      if (cn > 1) {
        ierr = DMPlexGetCone(dm, arr[i], &cone);CHKERRQ(ierr);
        ierr = PetscMemcpy(&newarr[co], cone, cn*sizeof(PetscInt));CHKERRQ(ierr);
      } else {
        newarr[co] = arr[i];
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, newn, newarr, PETSC_OWN_POINTER, &expandedPoints_[d]);CHKERRQ(ierr);
    arr = newarr;
    n = newn;
  }
  ierr = ISRestoreIndices(points, &arr0);CHKERRQ(ierr);
  *depth = depth_;
  if (expandedPoints) *expandedPoints = expandedPoints_;
  else {
    for (d=0; d<depth_; d++) {ierr = ISDestroy(&expandedPoints_[d]);CHKERRQ(ierr);}
    ierr = PetscFree(expandedPoints_);CHKERRQ(ierr);
  }
  if (sections) *sections = sections_;
  else {
    for (d=0; d<depth_; d++) {ierr = PetscSectionDestroy(&sections_[d]);CHKERRQ(ierr);}
    ierr = PetscFree(sections_);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexRestoreConeRecursive - Deallocates arrays created by DMPlexGetConeRecursive

  Not collective

  Input Parameters:
+ dm - The DMPlex
- points - The IS of points, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth_);CHKERRQ(ierr);
  if (depth && *depth != depth_) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "depth changed since last call to DMPlexGetConeRecursive");
  if (depth) *depth = 0;
  if (expandedPoints) {
    for (d=0; d<depth_; d++) {ierr = ISDestroy(&((*expandedPoints)[d]));CHKERRQ(ierr);}
    ierr = PetscFree(*expandedPoints);CHKERRQ(ierr);
  }
  if (sections)  {
    for (d=0; d<depth_; d++) {ierr = PetscSectionDestroy(&((*sections)[d]));CHKERRQ(ierr);}
    ierr = PetscFree(*sections);CHKERRQ(ierr);
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

  Developer Note: Why not call this DMPlexSetCover()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp(), DMPlexSetSupport(), DMPlexSetSupportSize()
@*/
PetscErrorCode DMPlexSetCone(DM dm, PetscInt p, const PetscInt cone[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  if (dof) PetscValidPointer(cone, 3);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    if ((cone[c] < pStart) || (cone[c] >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", cone[c], pStart, pEnd);
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
                    integer giving the prescription for cone traversal. If it is negative, the cone is
                    traversed in the opposite direction. Its value 'o', or if negative '-(o+1)', gives
                    the index of the cone point on which to start.

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.
  You must also call DMPlexRestoreConeOrientation() after you finish using the returned array.
  DMPlexRestoreConeOrientation() is not needed/available in C.

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetCone(), DMPlexSetChart()
@*/
PetscErrorCode DMPlexGetConeOrientation(DM dm, PetscInt p, const PetscInt *coneOrientation[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt dof;
    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    if (dof) PetscValidPointer(coneOrientation, 3);
  }
#endif
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);

  *coneOrientation = &mesh->coneOrientations[off];
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetConeOrientation - Set the orientations on the in-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
- coneOrientation - An array of orientations which are on the in-edges for point p. An orientation is an
                    integer giving the prescription for cone traversal. If it is negative, the cone is
                    traversed in the opposite direction. Its value 'o', or if negative '-(o+1)', gives
                    the index of the cone point on which to start.

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetConeOrientation(), DMPlexSetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexSetConeOrientation(DM dm, PetscInt p, const PetscInt coneOrientation[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  if (dof) PetscValidPointer(coneOrientation, 3);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    PetscInt cdof, o = coneOrientation[c];

    ierr = PetscSectionGetDof(mesh->coneSection, mesh->cones[off+c], &cdof);CHKERRQ(ierr);
    if (o && ((o < -(cdof+1)) || (o >= cdof))) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone orientation %D is not in the valid range [%D. %D)", o, -(cdof+1), cdof);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  if ((conePoint < pStart) || (conePoint >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone point %D is not in the valid range [%D, %D)", conePoint, pStart, pEnd);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((conePos < 0) || (conePos >= dof)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
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

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
@*/
PetscErrorCode DMPlexInsertConeOrientation(DM dm, PetscInt p, PetscInt conePos, PetscInt coneOrientation)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       pStart, pEnd;
  PetscInt       dof, off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
  if ((conePos < 0) || (conePos >= dof)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone position %D of point %D is not in the valid range [0, %D)", conePos, p, dof);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(size, 3);
  ierr = PetscSectionGetDof(mesh->supportSection, p, size);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetDof(mesh->supportSection, p, size);CHKERRQ(ierr);

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

.seealso: DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetSupport(DM dm, PetscInt p, const PetscInt *support[])
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       off;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(support, 3);
  ierr     = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
  if (dof) PetscValidPointer(support, 3);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  for (c = 0; c < dof; ++c) {
    if ((support[c] < pStart) || (support[c] >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", support[c], pStart, pEnd);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetChart(mesh->supportSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
  if ((p < pStart) || (p >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D is not in the valid range [%D, %D)", p, pStart, pEnd);
  if ((supportPoint < pStart) || (supportPoint >= pEnd)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support point %D is not in the valid range [%D, %D)", supportPoint, pStart, pEnd);
  if (supportPos >= dof) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Support position %D of point %D is not in the valid range [0, %D)", supportPos, p, dof);
  mesh->supports[off+supportPos] = supportPoint;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetTransitiveClosure - Return the points on the transitive closure of the in-edges or out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
. useCone - PETSC_TRUE for in-edges,  otherwise use out-edges
- points - If points is NULL on input, internal storage will be returned, otherwise the provided array is used

  Output Parameters:
+ numPoints - The number of points in the closure, so points[] is of size 2*numPoints
- points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...]

  Note:
  If using internal storage (points is NULL on input), each call overwrites the last output.

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexRestoreTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  PetscInt       *closure, *fifo;
  const PetscInt *tmp = NULL, *tmpO = NULL;
  PetscInt        tmpSize, t;
  PetscInt        depth       = 0, maxSize;
  PetscInt        closureSize = 2, fifoSize = 0, fifoStart = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr    = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* This is only 1-level */
  if (useCone) {
    ierr = DMPlexGetConeSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &tmp);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &tmpO);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGetSupportSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &tmp);CHKERRQ(ierr);
  }
  if (depth == 1) {
    if (*points) {
      closure = *points;
    } else {
      maxSize = 2*(PetscMax(mesh->maxConeSize, mesh->maxSupportSize)+1);
      ierr = DMGetWorkArray(dm, maxSize, MPIU_INT, &closure);CHKERRQ(ierr);
    }
    closure[0] = p; closure[1] = 0;
    for (t = 0; t < tmpSize; ++t, closureSize += 2) {
      closure[closureSize]   = tmp[t];
      closure[closureSize+1] = tmpO ? tmpO[t] : 0;
    }
    if (numPoints) *numPoints = closureSize/2;
    if (points)    *points    = closure;
    PetscFunctionReturn(0);
  }
  {
    PetscInt c, coneSeries, s,supportSeries;

    c = mesh->maxConeSize;
    coneSeries = (c > 1) ? ((PetscPowInt(c,depth+1)-1)/(c-1)) : depth+1;
    s = mesh->maxSupportSize;
    supportSeries = (s > 1) ? ((PetscPowInt(s,depth+1)-1)/(s-1)) : depth+1;
    maxSize = 2*PetscMax(coneSeries,supportSeries);
  }
  ierr    = DMGetWorkArray(dm, maxSize, MPIU_INT, &fifo);CHKERRQ(ierr);
  if (*points) {
    closure = *points;
  } else {
    ierr = DMGetWorkArray(dm, maxSize, MPIU_INT, &closure);CHKERRQ(ierr);
  }
  closure[0] = p; closure[1] = 0;
  for (t = 0; t < tmpSize; ++t, closureSize += 2, fifoSize += 2) {
    const PetscInt cp = tmp[t];
    const PetscInt co = tmpO ? tmpO[t] : 0;

    closure[closureSize]   = cp;
    closure[closureSize+1] = co;
    fifo[fifoSize]         = cp;
    fifo[fifoSize+1]       = co;
  }
  /* Should kick out early when depth is reached, rather than checking all vertices for empty cones */
  while (fifoSize - fifoStart) {
    const PetscInt q   = fifo[fifoStart];
    const PetscInt o   = fifo[fifoStart+1];
    const PetscInt rev = o >= 0 ? 0 : 1;
    const PetscInt off = rev ? -(o+1) : o;

    if (useCone) {
      ierr = DMPlexGetConeSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, q, &tmp);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, q, &tmpO);CHKERRQ(ierr);
    } else {
      ierr = DMPlexGetSupportSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, q, &tmp);CHKERRQ(ierr);
      tmpO = NULL;
    }
    for (t = 0; t < tmpSize; ++t) {
      const PetscInt i  = ((rev ? tmpSize-t : t) + off)%tmpSize;
      const PetscInt cp = tmp[i];
      /* Must propogate orientation: When we reverse orientation, we both reverse the direction of iteration and start at the other end of the chain. */
      /* HACK: It is worse to get the size here, than to change the interpretation of -(*+1)
       const PetscInt co = tmpO ? (rev ? -(tmpO[i]+1) : tmpO[i]) : 0; */
      PetscInt       co = tmpO ? tmpO[i] : 0;
      PetscInt       c;

      if (rev) {
        PetscInt childSize, coff;
        ierr = DMPlexGetConeSize(dm, cp, &childSize);CHKERRQ(ierr);
        coff = tmpO[i] < 0 ? -(tmpO[i]+1) : tmpO[i];
        co   = childSize ? -(((coff+childSize-1)%childSize)+1) : 0;
      }
      /* Check for duplicate */
      for (c = 0; c < closureSize; c += 2) {
        if (closure[c] == cp) break;
      }
      if (c == closureSize) {
        closure[closureSize]   = cp;
        closure[closureSize+1] = co;
        fifo[fifoSize]         = cp;
        fifo[fifoSize+1]       = co;
        closureSize           += 2;
        fifoSize              += 2;
      }
    }
    fifoStart += 2;
  }
  if (numPoints) *numPoints = closureSize/2;
  if (points)    *points    = closure;
  ierr = DMRestoreWorkArray(dm, maxSize, MPIU_INT, &fifo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetTransitiveClosure_Internal - Return the points on the transitive closure of the in-edges or out-edges for this point in the DAG with a specified initial orientation

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
. orientation - The orientation of the point
. useCone - PETSC_TRUE for in-edges,  otherwise use out-edges
- points - If points is NULL on input, internal storage will be returned, otherwise the provided array is used

  Output Parameters:
+ numPoints - The number of points in the closure, so points[] is of size 2*numPoints
- points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...]

  Note:
  If using internal storage (points is NULL on input), each call overwrites the last output.

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexRestoreTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexGetTransitiveClosure_Internal(DM dm, PetscInt p, PetscInt ornt, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  PetscInt       *closure, *fifo;
  const PetscInt *tmp = NULL, *tmpO = NULL;
  PetscInt        tmpSize, t;
  PetscInt        depth       = 0, maxSize;
  PetscInt        closureSize = 2, fifoSize = 0, fifoStart = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr    = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* This is only 1-level */
  if (useCone) {
    ierr = DMPlexGetConeSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &tmp);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &tmpO);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGetSupportSize(dm, p, &tmpSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &tmp);CHKERRQ(ierr);
  }
  if (depth == 1) {
    if (*points) {
      closure = *points;
    } else {
      maxSize = 2*(PetscMax(mesh->maxConeSize, mesh->maxSupportSize)+1);
      ierr = DMGetWorkArray(dm, maxSize, MPIU_INT, &closure);CHKERRQ(ierr);
    }
    closure[0] = p; closure[1] = ornt;
    for (t = 0; t < tmpSize; ++t, closureSize += 2) {
      const PetscInt i = ornt >= 0 ? (t+ornt)%tmpSize : (-(ornt+1) + tmpSize-t)%tmpSize;
      closure[closureSize]   = tmp[i];
      closure[closureSize+1] = tmpO ? tmpO[i] : 0;
    }
    if (numPoints) *numPoints = closureSize/2;
    if (points)    *points    = closure;
    PetscFunctionReturn(0);
  }
  {
    PetscInt c, coneSeries, s,supportSeries;

    c = mesh->maxConeSize;
    coneSeries = (c > 1) ? ((PetscPowInt(c,depth+1)-1)/(c-1)) : depth+1;
    s = mesh->maxSupportSize;
    supportSeries = (s > 1) ? ((PetscPowInt(s,depth+1)-1)/(s-1)) : depth+1;
    maxSize = 2*PetscMax(coneSeries,supportSeries);
  }
  ierr    = DMGetWorkArray(dm, maxSize, MPIU_INT, &fifo);CHKERRQ(ierr);
  if (*points) {
    closure = *points;
  } else {
    ierr = DMGetWorkArray(dm, maxSize, MPIU_INT, &closure);CHKERRQ(ierr);
  }
  closure[0] = p; closure[1] = ornt;
  for (t = 0; t < tmpSize; ++t, closureSize += 2, fifoSize += 2) {
    const PetscInt i  = ornt >= 0 ? (t+ornt)%tmpSize : (-(ornt+1) + tmpSize-t)%tmpSize;
    const PetscInt cp = tmp[i];
    PetscInt       co = tmpO ? tmpO[i] : 0;

    if (ornt < 0) {
      PetscInt childSize, coff;
      ierr = DMPlexGetConeSize(dm, cp, &childSize);CHKERRQ(ierr);
      coff = co < 0 ? -(tmpO[i]+1) : tmpO[i];
      co   = childSize ? -(((coff+childSize-1)%childSize)+1) : 0;
    }
    closure[closureSize]   = cp;
    closure[closureSize+1] = co;
    fifo[fifoSize]         = cp;
    fifo[fifoSize+1]       = co;
  }
  /* Should kick out early when depth is reached, rather than checking all vertices for empty cones */
  while (fifoSize - fifoStart) {
    const PetscInt q   = fifo[fifoStart];
    const PetscInt o   = fifo[fifoStart+1];
    const PetscInt rev = o >= 0 ? 0 : 1;
    const PetscInt off = rev ? -(o+1) : o;

    if (useCone) {
      ierr = DMPlexGetConeSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, q, &tmp);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, q, &tmpO);CHKERRQ(ierr);
    } else {
      ierr = DMPlexGetSupportSize(dm, q, &tmpSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, q, &tmp);CHKERRQ(ierr);
      tmpO = NULL;
    }
    for (t = 0; t < tmpSize; ++t) {
      const PetscInt i  = ((rev ? tmpSize-t : t) + off)%tmpSize;
      const PetscInt cp = tmp[i];
      /* Must propogate orientation: When we reverse orientation, we both reverse the direction of iteration and start at the other end of the chain. */
      /* HACK: It is worse to get the size here, than to change the interpretation of -(*+1)
       const PetscInt co = tmpO ? (rev ? -(tmpO[i]+1) : tmpO[i]) : 0; */
      PetscInt       co = tmpO ? tmpO[i] : 0;
      PetscInt       c;

      if (rev) {
        PetscInt childSize, coff;
        ierr = DMPlexGetConeSize(dm, cp, &childSize);CHKERRQ(ierr);
        coff = tmpO[i] < 0 ? -(tmpO[i]+1) : tmpO[i];
        co   = childSize ? -(((coff+childSize-1)%childSize)+1) : 0;
      }
      /* Check for duplicate */
      for (c = 0; c < closureSize; c += 2) {
        if (closure[c] == cp) break;
      }
      if (c == closureSize) {
        closure[closureSize]   = cp;
        closure[closureSize+1] = co;
        fifo[fifoSize]         = cp;
        fifo[fifoSize+1]       = co;
        closureSize           += 2;
        fifoSize              += 2;
      }
    }
    fifoStart += 2;
  }
  if (numPoints) *numPoints = closureSize/2;
  if (points)    *points    = closure;
  ierr = DMRestoreWorkArray(dm, maxSize, MPIU_INT, &fifo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreTransitiveClosure - Restore the array of points on the transitive closure of the in-edges or out-edges for this point in the DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The point, which must lie in the chart set with DMPlexSetChart()
. useCone - PETSC_TRUE for in-edges,  otherwise use out-edges
. numPoints - The number of points in the closure, so points[] is of size 2*numPoints, zeroed on exit
- points - The points and point orientations, interleaved as pairs [p0, o0, p1, o1, ...], zeroed on exit

  Note:
  If not using internal storage (points is not NULL on input), this call is unnecessary

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  The numPoints argument is not present in the Fortran 90 binding since it is internal to the array.

  Level: beginner

.seealso: DMPlexGetTransitiveClosure(), DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart(), DMPlexGetCone()
@*/
PetscErrorCode DMPlexRestoreTransitiveClosure(DM dm, PetscInt p, PetscBool useCone, PetscInt *numPoints, PetscInt *points[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (numPoints) PetscValidIntPointer(numPoints,4);
  if (points) PetscValidPointer(points,5);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, points);CHKERRQ(ierr);
  if (numPoints) *numPoints = 0;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionSetUp(mesh->coneSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(mesh->coneSection, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &mesh->cones);CHKERRQ(ierr);
  ierr = PetscCalloc1(size, &mesh->coneOrientations);CHKERRQ(ierr);
  if (mesh->maxSupportSize) {
    ierr = PetscSectionSetUp(mesh->supportSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(mesh->supportSection, &size);CHKERRQ(ierr);
    ierr = PetscMalloc1(size, &mesh->supports);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_Plex(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {ierr = DMClone(dm, subdm);CHKERRQ(ierr);}
  ierr = DMCreateSectionSubDM(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  if (subdm) {(*subdm)->useNatural = dm->useNatural;}
  if (dm->useNatural && dm->sfMigration) {
    PetscSF        sfMigrationInv,sfNatural;
    PetscSection   section, sectionSeq;

    (*subdm)->sfMigration = dm->sfMigration;
    ierr = PetscObjectReference((PetscObject) dm->sfMigration);CHKERRQ(ierr);
    ierr = DMGetLocalSection((*subdm), &section);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF((*subdm)->sfMigration, &sfMigrationInv);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) (*subdm)), &sectionSeq);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMigrationInv, section, NULL, sectionSeq);CHKERRQ(ierr);

    ierr = DMPlexCreateGlobalToNaturalSF(*subdm, sectionSeq, (*subdm)->sfMigration, &sfNatural);CHKERRQ(ierr);
    (*subdm)->sfNatural = sfNatural;
    ierr = PetscSectionDestroy(&sectionSeq);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfMigrationInv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSuperDM_Plex(DM dms[], PetscInt len, IS **is, DM *superdm)
{
  PetscErrorCode ierr;
  PetscInt       i = 0;

  PetscFunctionBegin;
  ierr = DMClone(dms[0], superdm);CHKERRQ(ierr);
  ierr = DMCreateSectionSuperDM(dms, len, is, superdm);CHKERRQ(ierr);
  (*superdm)->useNatural = PETSC_FALSE;
  for (i = 0; i < len; i++){
    if (dms[i]->useNatural && dms[i]->sfMigration) {
      PetscSF        sfMigrationInv,sfNatural;
      PetscSection   section, sectionSeq;

      (*superdm)->sfMigration = dms[i]->sfMigration;
      ierr = PetscObjectReference((PetscObject) dms[i]->sfMigration);CHKERRQ(ierr);
      (*superdm)->useNatural = PETSC_TRUE;
      ierr = DMGetLocalSection((*superdm), &section);CHKERRQ(ierr);
      ierr = PetscSFCreateInverseSF((*superdm)->sfMigration, &sfMigrationInv);CHKERRQ(ierr);
      ierr = PetscSectionCreate(PetscObjectComm((PetscObject) (*superdm)), &sectionSeq);CHKERRQ(ierr);
      ierr = PetscSFDistributeSection(sfMigrationInv, section, NULL, sectionSeq);CHKERRQ(ierr);

      ierr = DMPlexCreateGlobalToNaturalSF(*superdm, sectionSeq, (*superdm)->sfMigration, &sfNatural);CHKERRQ(ierr);
      (*superdm)->sfNatural = sfNatural;
      ierr = PetscSectionDestroy(&sectionSeq);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&sfMigrationInv);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (mesh->supports) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Supports were already setup in this DMPlex");
  ierr = PetscLogEventBegin(DMPLEX_Symmetrize,dm,0,0,0);CHKERRQ(ierr);
  /* Calculate support sizes */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for (c = off; c < off+dof; ++c) {
      ierr = PetscSectionAddDof(mesh->supportSection, mesh->cones[c], 1);CHKERRQ(ierr);
    }
  }
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof;

    ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);

    mesh->maxSupportSize = PetscMax(mesh->maxSupportSize, dof);
  }
  ierr = PetscSectionSetUp(mesh->supportSection);CHKERRQ(ierr);
  /* Calculate supports */
  ierr = PetscSectionGetStorageSize(mesh->supportSection, &supportSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(supportSize, &mesh->supports);CHKERRQ(ierr);
  ierr = PetscCalloc1(pEnd - pStart, &offsets);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, off, c;

    ierr = PetscSectionGetDof(mesh->coneSection, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, p, &off);CHKERRQ(ierr);
    for (c = off; c < off+dof; ++c) {
      const PetscInt q = mesh->cones[c];
      PetscInt       offS;

      ierr = PetscSectionGetOffset(mesh->supportSection, q, &offS);CHKERRQ(ierr);

      mesh->supports[offS+offsets[q]] = p;
      ++offsets[q];
    }
  }
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Symmetrize,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateDepthStratum(DM dm, DMLabel label, PetscInt depth, PetscInt pStart, PetscInt pEnd)
{
  IS             stratumIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pStart >= pEnd) PetscFunctionReturn(0);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt  qStart, qEnd, numLevels, level;
    PetscBool overlap = PETSC_FALSE;
    ierr = DMLabelGetNumValues(label, &numLevels);CHKERRQ(ierr);
    for (level = 0; level < numLevels; level++) {
      ierr = DMLabelGetStratumBounds(label, level, &qStart, &qEnd);CHKERRQ(ierr);
      if ((pStart >= qStart && pStart < qEnd) || (pEnd > qStart && pEnd <= qEnd)) {overlap = PETSC_TRUE; break;}
    }
    if (overlap) SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_PLIB, "New depth %D range [%D,%D) overlaps with depth %D range [%D,%D)", depth, pStart, pEnd, level, qStart, qEnd);
  }
#endif
  ierr = ISCreateStride(PETSC_COMM_SELF, pEnd-pStart, pStart, 1, &stratumIS);CHKERRQ(ierr);
  ierr = DMLabelSetStratumIS(label, depth, stratumIS);CHKERRQ(ierr);
  ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateDimStratum(DM,DMLabel,DMLabel,PetscInt,PetscInt);

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
  PetscInt       cMax, fMax, eMax, vMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscLogEventBegin(DMPLEX_Stratify,dm,0,0,0);CHKERRQ(ierr);

  /* Create depth label */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "depth");CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);

  {
    /* Initialize roots and count leaves */
    PetscInt sMin = PETSC_MAX_INT;
    PetscInt sMax = PETSC_MIN_INT;
    PetscInt coneSize, supportSize;

    for (p = pStart; p < pEnd; ++p) {
      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
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
    ierr = DMPlexCreateDepthStratum(dm, label, 0, sMin, sMax+1);CHKERRQ(ierr);
  }

  if (numRoots + numLeaves == (pEnd - pStart)) {
    PetscInt sMin = PETSC_MAX_INT;
    PetscInt sMax = PETSC_MIN_INT;
    PetscInt coneSize, supportSize;

    for (p = pStart; p < pEnd; ++p) {
      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
      if (!supportSize && coneSize) {
        sMin = PetscMin(p, sMin);
        sMax = PetscMax(p, sMax);
      }
    }
    ierr = DMPlexCreateDepthStratum(dm, label, 1, sMin, sMax+1);CHKERRQ(ierr);
  } else {
    PetscInt level = 0;
    PetscInt qStart, qEnd, q;

    ierr = DMLabelGetStratumBounds(label, level, &qStart, &qEnd);CHKERRQ(ierr);
    while (qEnd > qStart) {
      PetscInt sMin = PETSC_MAX_INT;
      PetscInt sMax = PETSC_MIN_INT;

      for (q = qStart; q < qEnd; ++q) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        ierr = DMPlexGetSupportSize(dm, q, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, q, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          sMin = PetscMin(support[s], sMin);
          sMax = PetscMax(support[s], sMax);
        }
      }
      ierr = DMLabelGetNumValues(label, &level);CHKERRQ(ierr);
      ierr = DMPlexCreateDepthStratum(dm, label, level, sMin, sMax+1);CHKERRQ(ierr);
      ierr = DMLabelGetStratumBounds(label, level, &qStart, &qEnd);CHKERRQ(ierr);
    }
  }
  { /* just in case there is an empty process */
    PetscInt numValues, maxValues = 0, v;

    ierr = DMLabelGetNumValues(label, &numValues);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&numValues,&maxValues,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    for (v = numValues; v < maxValues; v++) {
      ierr = DMLabelAddStratum(label, v);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectStateGet((PetscObject) label, &mesh->depthState);CHKERRQ(ierr);

  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  if (cMax >= 0 || fMax >= 0 || eMax >= 0 || vMax >= 0) {
    PetscInt dim;
    DMLabel  dimLabel;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "dim");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "dim", &dimLabel);CHKERRQ(ierr);
    if (cMax >= 0) {ierr = DMPlexCreateDimStratum(dm, label, dimLabel, dim, cMax);CHKERRQ(ierr);}
    if (fMax >= 0) {ierr = DMPlexCreateDimStratum(dm, label, dimLabel, dim - 1, fMax);CHKERRQ(ierr);}
    if (eMax >= 0) {ierr = DMPlexCreateDimStratum(dm, label, dimLabel, 1, eMax);CHKERRQ(ierr);}
    if (vMax >= 0) {ierr = DMPlexCreateDimStratum(dm, label, dimLabel, 0, vMax);CHKERRQ(ierr);}
  }
  ierr = PetscLogEventEnd(DMPLEX_Stratify,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeCellTypes - Infer the polytope type of every cell using its dimension and cone size.

  Collective on dm

  Input Parameter:
. mesh - The DMPlex

  DMPlexComputeCellTypes() should be called after all calls to DMPlexSymmetrize() and DMPlexStratify()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSymmetrize(), DMPlexStratify()
@*/
PetscErrorCode DMPlexComputeCellTypes(DM dm)
{
  DM_Plex       *mesh;
  DMLabel        label;
  PetscInt       dim, depth, gcStart, gcEnd, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh = (DM_Plex *) dm->data;
  ierr = DMCreateLabel(dm, "celltype");CHKERRQ(ierr);
  ierr = DMPlexGetCellTypeLabel(dm, &label);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &gcStart, &gcEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct = DM_POLYTOPE_UNKNOWN;
    PetscInt       pdepth, pheight, coneSize;

    ierr = DMPlexGetPointDepth(dm, p, &pdepth);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
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
          case 6: ct = DM_POLYTOPE_TRI_PRISM_TENSOR;break;
          case 8: ct = DM_POLYTOPE_HEXAHEDRON;break;
          default: break;
        }
      }
    } else {
      if (pdepth == 0) {
        ct = DM_POLYTOPE_POINT;
      } else if (pheight == 0) {
        if ((p >= gcStart) && (p < gcEnd)) {
          if (coneSize == 1) ct = DM_POLYTOPE_FV_GHOST;
          else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %D should have a cone size of 1, not %D", p, coneSize);
        } else {
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
                case 5: ct = DM_POLYTOPE_TRI_PRISM_TENSOR;break;
                case 6: ct = DM_POLYTOPE_HEXAHEDRON;break;
                default: break;
              }
              break;
            default: break;
          }
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
    if (ct == DM_POLYTOPE_UNKNOWN) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Point %D is screwed up", p);
    ierr = DMLabelSetValue(label, p, ct);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateGet((PetscObject) label, &mesh->celltypeState);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) label, NULL, "-dm_plex_celltypes_view");CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(points, 3);
  PetscValidIntPointer(numCoveredPoints, 4);
  PetscValidPointer(coveredPoints, 5);
  ierr = DMGetWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[1]);CHKERRQ(ierr);
  /* Copy in support of first point */
  ierr = PetscSectionGetDof(mesh->supportSection, points[0], &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->supportSection, points[0], &off);CHKERRQ(ierr);
  for (joinSize = 0; joinSize < dof; ++joinSize) {
    join[i][joinSize] = mesh->supports[off+joinSize];
  }
  /* Check each successive support */
  for (p = 1; p < numPoints; ++p) {
    PetscInt newJoinSize = 0;

    ierr = PetscSectionGetDof(mesh->supportSection, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->supportSection, points[p], &off);CHKERRQ(ierr);
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
  ierr              = DMRestoreWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[1-i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (points) PetscValidIntPointer(points,3);
  if (numCoveredPoints) PetscValidIntPointer(numCoveredPoints,4);
  PetscValidPointer(coveredPoints, 5);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void*) coveredPoints);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(points, 3);
  PetscValidIntPointer(numCoveredPoints, 4);
  PetscValidPointer(coveredPoints, 5);

  ierr    = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr    = PetscCalloc1(numPoints, &closures);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, numPoints*(depth+2), MPIU_INT, &offsets);CHKERRQ(ierr);
  ms      = mesh->maxSupportSize;
  maxSize = (ms > 1) ? ((PetscPowInt(ms,depth+1)-1)/(ms-1)) : depth + 1;
  ierr    = DMGetWorkArray(dm, maxSize, MPIU_INT, &join[0]);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, maxSize, MPIU_INT, &join[1]);CHKERRQ(ierr);

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_FALSE, &closureSize, &closures[p]);CHKERRQ(ierr);

    offsets[p*(depth+2)+0] = 0;
    for (d = 0; d < depth+1; ++d) {
      PetscInt pStart, pEnd, i;

      ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
      for (i = offsets[p*(depth+2)+d]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(depth+2)+d+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(depth+2)+d+1] = i;
    }
    if (offsets[p*(depth+2)+depth+1] != closureSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(depth+2)+depth+1], closureSize);
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
    ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_FALSE, NULL, &closures[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(closures);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numPoints*(depth+2), MPIU_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxSupportSize, MPIU_INT, &join[1-i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveringPoints, 3);
  PetscValidPointer(coveringPoints, 4);
  ierr = DMGetWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[1]);CHKERRQ(ierr);
  /* Copy in cone of first point */
  ierr = PetscSectionGetDof(mesh->coneSection, points[0], &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(mesh->coneSection, points[0], &off);CHKERRQ(ierr);
  for (meetSize = 0; meetSize < dof; ++meetSize) {
    meet[i][meetSize] = mesh->cones[off+meetSize];
  }
  /* Check each successive cone */
  for (p = 1; p < numPoints; ++p) {
    PetscInt newMeetSize = 0;

    ierr = PetscSectionGetDof(mesh->coneSection, points[p], &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, points[p], &off);CHKERRQ(ierr);
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
  ierr               = DMRestoreWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[1-i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (points) PetscValidIntPointer(points,3);
  if (numCoveredPoints) PetscValidIntPointer(numCoveredPoints,4);
  PetscValidPointer(coveredPoints,5);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, (void*) coveredPoints);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);

  ierr    = DMPlexGetDepth(dm, &height);CHKERRQ(ierr);
  ierr    = PetscMalloc1(numPoints, &closures);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, numPoints*(height+2), MPIU_INT, &offsets);CHKERRQ(ierr);
  mc      = mesh->maxConeSize;
  maxSize = (mc > 1) ? ((PetscPowInt(mc,height+1)-1)/(mc-1)) : height + 1;
  ierr    = DMGetWorkArray(dm, maxSize, MPIU_INT, &meet[0]);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, maxSize, MPIU_INT, &meet[1]);CHKERRQ(ierr);

  for (p = 0; p < numPoints; ++p) {
    PetscInt closureSize;

    ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closures[p]);CHKERRQ(ierr);

    offsets[p*(height+2)+0] = 0;
    for (h = 0; h < height+1; ++h) {
      PetscInt pStart, pEnd, i;

      ierr = DMPlexGetHeightStratum(dm, h, &pStart, &pEnd);CHKERRQ(ierr);
      for (i = offsets[p*(height+2)+h]; i < closureSize; ++i) {
        if ((pStart > closures[p][i*2]) || (pEnd <= closures[p][i*2])) {
          offsets[p*(height+2)+h+1] = i;
          break;
        }
      }
      if (i == closureSize) offsets[p*(height+2)+h+1] = i;
    }
    if (offsets[p*(height+2)+height+1] != closureSize) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Total size of closure %D should be %D", offsets[p*(height+2)+height+1], closureSize);
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
    ierr = DMPlexRestoreTransitiveClosure(dm, points[p], PETSC_TRUE, NULL, &closures[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(closures);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numPoints*(height+2), MPIU_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxConeSize, MPIU_INT, &meet[1-i]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmA, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmB, DM_CLASSID, 2);
  PetscValidPointer(equal, 3);

  *equal = PETSC_FALSE;
  ierr = DMPlexGetDepth(dmA, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dmB, &depthB);CHKERRQ(ierr);
  if (depth != depthB) PetscFunctionReturn(0);
  ierr = DMPlexGetChart(dmA, &pStart,  &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dmB, &pStartB, &pEndB);CHKERRQ(ierr);
  if ((pStart != pStartB) || (pEnd != pEndB)) PetscFunctionReturn(0);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *coneB, *ornt, *orntB, *support, *supportB;
    PetscInt        coneSize, coneSizeB, c, supportSize, supportSizeB, s;

    ierr = DMPlexGetConeSize(dmA, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dmA, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmA, p, &ornt);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dmB, p, &coneSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dmB, p, &coneB);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dmB, p, &orntB);CHKERRQ(ierr);
    if (coneSize != coneSizeB) PetscFunctionReturn(0);
    for (c = 0; c < coneSize; ++c) {
      if (cone[c] != coneB[c]) PetscFunctionReturn(0);
      if (ornt[c] != orntB[c]) PetscFunctionReturn(0);
    }
    ierr = DMPlexGetSupportSize(dmA, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmA, p, &support);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmB, p, &supportSizeB);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmB, p, &supportB);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  PetscValidPointer(numFaceVertices,3);
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
      SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %D for dimension %D", numCorners, cellDim);
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
      SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of face corners %D for dimension %D", numCorners, cellDim);
    }
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid cell dimension %D", cellDim);
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

.seealso: DMPlexGetDepth(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum()
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
  The point depth is described more in detail in DMPlexSymmetrize().

.seealso: DMPlexGetDepthLabel(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum(), DMPlexGetPointDepth(), DMPlexGetPointHeight(), DMPlexSymmetrize()
@*/
PetscErrorCode DMPlexGetDepth(DM dm, PetscInt *depth)
{
  DMLabel        label;
  PetscInt       d = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depth, 2);
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (label) {ierr = DMLabelGetNumValues(label, &d);CHKERRQ(ierr);}
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

.seealso: DMPlexGetHeightStratum(), DMPlexGetDepth(), DMPlexGetPointDepth()
@*/
PetscErrorCode DMPlexGetDepthStratum(DM dm, PetscInt stratumValue, PetscInt *start, PetscInt *end)
{
  DMLabel        label;
  PetscInt       pStart, pEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) {PetscValidPointer(start, 3); *start = 0;}
  if (end)   {PetscValidPointer(end,   4); *end   = 0;}
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  if (pStart == pEnd) PetscFunctionReturn(0);
  if (stratumValue < 0) {
    if (start) *start = pStart;
    if (end)   *end   = pEnd;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (!label) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");
  ierr = DMLabelGetStratumBounds(label, stratumValue, start, end);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (start) {PetscValidPointer(start, 3); *start = 0;}
  if (end)   {PetscValidPointer(end,   4); *end   = 0;}
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  if (pStart == pEnd) PetscFunctionReturn(0);
  if (stratumValue < 0) {
    if (start) *start = pStart;
    if (end)   *end   = pEnd;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (!label) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");
  ierr = DMLabelGetNumValues(label, &depth);CHKERRQ(ierr);
  ierr = DMLabelGetStratumBounds(label, depth-1-stratumValue, start, end);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPointDepth - Get the depth of a given point

  Not Collective

  Input Parameter:
+ dm    - The DMPlex object
- point - The point

  Output Parameter:
. depth - The depth of the point

  Level: intermediate

.seealso: DMPlexGetCellType(), DMPlexGetDepthLabel(), DMPlexGetDepth(), DMPlexGetPointHeight()
@*/
PetscErrorCode DMPlexGetPointDepth(DM dm, PetscInt point, PetscInt *depth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(depth, 3);
  ierr = DMLabelGetValue(dm->depthLabel, point, depth);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPointHeight - Get the height of a given point

  Not Collective

  Input Parameter:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(height, 3);
  ierr = DMLabelGetNumValues(dm->depthLabel, &n);CHKERRQ(ierr);
  ierr = DMLabelGetValue(dm->depthLabel, point, &pDepth);CHKERRQ(ierr);
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

  Level: developer

.seealso: DMPlexGetCellType(), DMPlexGetDepthLabel(), DMPlexGetDepth()
@*/
PetscErrorCode DMPlexGetCellTypeLabel(DM dm, DMLabel *celltypeLabel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(celltypeLabel, 2);
  if (!dm->celltypeLabel) {ierr = DMPlexComputeCellTypes(dm);CHKERRQ(ierr);}
  *celltypeLabel = dm->celltypeLabel;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetCellType - Get the polytope type of a given cell

  Not Collective

  Input Parameter:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(celltype, 3);
  ierr = DMPlexGetCellTypeLabel(dm, &label);CHKERRQ(ierr);
  ierr = DMLabelGetValue(label, cell, &ct);CHKERRQ(ierr);
  *celltype = (DMPolytopeType) ct;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateCoordinateDM_Plex(DM dm, DM *cdm)
{
  PetscSection   section, s;
  Mat            m;
  PetscInt       maxHeight;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone(dm, cdm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxProjectionHeight(dm, &maxHeight);CHKERRQ(ierr);
  ierr = DMPlexSetMaxProjectionHeight(*cdm, maxHeight);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section);CHKERRQ(ierr);
  ierr = DMSetLocalSection(*cdm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &s);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &m);CHKERRQ(ierr);
  ierr = DMSetDefaultConstraints(*cdm, s, m);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = MatDestroy(&m);CHKERRQ(ierr);

  ierr = DMSetNumFields(*cdm, 1);CHKERRQ(ierr);
  ierr = DMCreateDS(*cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateCoordinateField_Plex(DM dm, DMField *field)
{
  Vec            coordsLocal;
  DM             coordsDM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *field = NULL;
  ierr = DMGetCoordinatesLocal(dm,&coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm,&coordsDM);CHKERRQ(ierr);
  if (coordsLocal && coordsDM) {
    ierr = DMFieldCreateDS(coordsDM, 0, coordsLocal, field);CHKERRQ(ierr);
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
. coneOrientations - The cone orientation for each point

  Level: developer

.seealso: DMPlexGetConeSection()
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
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscSectionGetFieldComponents(section, field, Nc);CHKERRQ(ierr);
  if (line < 0) {
    *k = 0;
    *Nc = 0;
  } else if (vertexchart) {            /* If we only have a vertex chart, we must have degree k=1 */
    *k = 1;
  } else {                      /* Assume the full interpolated mesh is in the chart; lines in particular */
    /* An order k SEM disc has k-1 dofs on an edge */
    ierr = PetscSectionGetFieldDof(section, line, field, k);CHKERRQ(ierr);
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
  PetscInt      *perm;
  PetscInt       dim, depth = -1, eStart = -1, k, Nf, f, Nc, c, i, j, size = 0, offset = 0, foffset = 0;
  PetscBool      vertexchart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 1) PetscFunctionReturn(0);
  if (point < 0) {
    PetscInt sStart,sEnd;

    ierr = DMPlexGetDepthStratum(dm, 1, &sStart, &sEnd);CHKERRQ(ierr);
    point = sEnd-sStart ? sStart : point;
  }
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  if (point >= 0) { ierr = DMLabelGetValue(label, point, &depth);CHKERRQ(ierr); }
  if (!section) {ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);}
  if (depth == 1) {eStart = point;}
  else if  (depth == dim) {
    const PetscInt *cone;

    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    if (dim == 2) eStart = cone[0];
    else if (dim == 3) {
      const PetscInt *cone2;
      ierr = DMPlexGetCone(dm, cone[0], &cone2);CHKERRQ(ierr);
      eStart = cone2[0];
    } else SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D of depth %D cannot be used to bootstrap spectral ordering for dim %D", point, depth, dim);
  } else if (depth >= 0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D of depth %D cannot be used to bootstrap spectral ordering for dim %D", point, depth, dim);
  {                             /* Determine whether the chart covers all points or just vertices. */
    PetscInt pStart,pEnd,cStart,cEnd;
    ierr = DMPlexGetDepthStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section,&cStart,&cEnd);CHKERRQ(ierr);
    if (pStart == cStart && pEnd == cEnd) vertexchart = PETSC_TRUE; /* Just vertices */
    else vertexchart = PETSC_FALSE;                                 /* Assume all interpolated points are in chart */
  }
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k);CHKERRQ(ierr);
    size += PetscPowInt(k+1, dim)*Nc;
  }
  ierr = PetscMalloc1(size, &perm);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    switch (dim) {
    case 1:
      ierr = PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k);CHKERRQ(ierr);
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
      ierr = PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k);CHKERRQ(ierr);
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
      ierr = PetscSectionFieldGetTensorDegree_Private(section,f,eStart,vertexchart,&Nc,&k);CHKERRQ(ierr);
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
    default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No spectral ordering for dimension %D", dim);
    }
  }
  if (offset != size) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Number of permutation entries %D != %D", offset, size);
  /* Check permutation */
  {
    PetscInt *check;

    ierr = PetscMalloc1(size, &check);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) {check[i] = -1; if (perm[i] < 0 || perm[i] >= size) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid permutation index p[%D] = %D", i, perm[i]);}
    for (i = 0; i < size; ++i) check[perm[i]] = i;
    for (i = 0; i < size; ++i) {if (check[i] < 0) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Missing permutation index %D", i);}
    ierr = PetscFree(check);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetClosurePermutation_Internal(section, (PetscObject) dm, size, PETSC_OWN_POINTER, perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetPointDualSpaceFEM(DM dm, PetscInt point, PetscInt field, PetscDualSpace *dspace)
{
  PetscDS        prob;
  PetscInt       depth, Nf, h;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  Nf      = prob->Nf;
  label   = dm->depthLabel;
  *dspace = NULL;
  if (field < Nf) {
    PetscObject disc = prob->disc[field];

    if (disc->classid == PETSCFE_CLASSID) {
      PetscDualSpace dsp;

      ierr = PetscFEGetDualSpace((PetscFE)disc,&dsp);CHKERRQ(ierr);
      ierr = DMLabelGetNumValues(label,&depth);CHKERRQ(ierr);
      ierr = DMLabelGetValue(label,point,&h);CHKERRQ(ierr);
      h    = depth - 1 - h;
      if (h) {
        ierr = PetscDualSpaceGetHeightSubspace(dsp,h,dspace);CHKERRQ(ierr);
      } else {
        *dspace = dsp;
      }
    }
  }
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode DMPlexVecGetClosure_Depth1_Static(DM dm, PetscSection section, Vec v, PetscInt point, PetscInt *csize, PetscScalar *values[])
{
  PetscScalar    *array, *vArray;
  const PetscInt *cone, *coneO;
  PetscInt        pStart, pEnd, p, numPoints, size = 0, offset = 0;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, point, &numPoints);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, point, &coneO);CHKERRQ(ierr);
  if (!values || !*values) {
    if ((point >= pStart) && (point < pEnd)) {
      PetscInt dof;

      ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
      size += dof;
    }
    for (p = 0; p < numPoints; ++p) {
      const PetscInt cp = cone[p];
      PetscInt       dof;

      if ((cp < pStart) || (cp >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, cp, &dof);CHKERRQ(ierr);
      size += dof;
    }
    if (!values) {
      if (csize) *csize = size;
      PetscFunctionReturn(0);
    }
    ierr = DMGetWorkArray(dm, size, MPIU_SCALAR, &array);CHKERRQ(ierr);
  } else {
    array = *values;
  }
  size = 0;
  ierr = VecGetArray(v, &vArray);CHKERRQ(ierr);
  if ((point >= pStart) && (point < pEnd)) {
    PetscInt     dof, off, d;
    PetscScalar *varr;

    ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
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
    ierr = PetscSectionGetDof(section, cp, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, cp, &off);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(v, &vArray);CHKERRQ(ierr);
  if (!*values) {
    if (csize) *csize = size;
    *values = array;
  } else {
    if (size > *csize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Size of input array %D < actual size %D", *csize, size);
    *csize = size;
  }
  PetscFunctionReturn(0);
}

/* Compressed closure does not apply closure permutation */
PetscErrorCode DMPlexGetCompressedClosure(DM dm, PetscSection section, PetscInt point, PetscInt *numPoints, PetscInt **points, PetscSection *clSec, IS *clPoints, const PetscInt **clp)
{
  const PetscInt *cla;
  PetscInt       np, *pts = NULL;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscSectionGetClosureIndex(section, (PetscObject) dm, clSec, clPoints);CHKERRQ(ierr);
  if (!*clPoints) {
    PetscInt pStart, pEnd, p, q;

    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &np, &pts);CHKERRQ(ierr);
    /* Compress out points not in the section */
    for (p = 0, q = 0; p < np; p++) {
      PetscInt r = pts[2*p];
      if ((r >= pStart) && (r < pEnd)) {
        pts[q*2]   = r;
        pts[q*2+1] = pts[2*p+1];
        ++q;
      }
    }
    np = q;
    cla = NULL;
  } else {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(*clSec, point, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(*clSec, point, &off);CHKERRQ(ierr);
    ierr = ISGetIndices(*clPoints, &cla);CHKERRQ(ierr);
    np   = dof/2;
    pts  = (PetscInt *) &cla[off];
  }
  *numPoints = np;
  *points    = pts;
  *clp       = cla;

  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexRestoreCompressedClosure(DM dm, PetscSection section, PetscInt point, PetscInt *numPoints, PetscInt **points, PetscSection *clSec, IS *clPoints, const PetscInt **clp)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (!*clPoints) {
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, numPoints, points);CHKERRQ(ierr);
  } else {
    ierr = ISRestoreIndices(*clPoints, clp);CHKERRQ(ierr);
  }
  *numPoints = 0;
  *points    = NULL;
  *clSec     = NULL;
  *clPoints  = NULL;
  *clp       = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexVecGetClosure_Static(DM dm, PetscSection section, PetscInt numPoints, const PetscInt points[], const PetscInt clperm[], const PetscScalar vArray[], PetscInt *size, PetscScalar array[])
{
  PetscInt          offset = 0, p;
  const PetscInt    **perms = NULL;
  const PetscScalar **flips = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBeginHot;
  *size = 0;
  ierr = PetscSectionGetPointSyms(section,numPoints,points,&perms,&flips);CHKERRQ(ierr);
  for (p = 0; p < numPoints; p++) {
    const PetscInt    point = points[2*p];
    const PetscInt    *perm = perms ? perms[p] : NULL;
    const PetscScalar *flip = flips ? flips[p] : NULL;
    PetscInt          dof, off, d;
    const PetscScalar *varr;

    ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
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
  ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms,&flips);CHKERRQ(ierr);
  *size = offset;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMPlexVecGetClosure_Fields_Static(DM dm, PetscSection section, PetscInt numPoints, const PetscInt points[], PetscInt numFields, const PetscInt clperm[], const PetscScalar vArray[], PetscInt *size, PetscScalar array[])
{
  PetscInt          offset = 0, f;
  PetscErrorCode    ierr;

  PetscFunctionBeginHot;
  *size = 0;
  for (f = 0; f < numFields; ++f) {
    PetscInt          p;
    const PetscInt    **perms = NULL;
    const PetscScalar **flips = NULL;

    ierr = PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms,&flips);CHKERRQ(ierr);
    for (p = 0; p < numPoints; p++) {
      const PetscInt    point = points[2*p];
      PetscInt          fdof, foff, b;
      const PetscScalar *varr;
      const PetscInt    *perm = perms ? perms[p] : NULL;
      const PetscScalar *flip = flips ? flips[p] : NULL;

      ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(section, point, f, &foff);CHKERRQ(ierr);
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
    ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms,&flips);CHKERRQ(ierr);
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
. point - The point in the DM
. csize - The size of the input values array, or NULL
- values - An array to use for the values, or NULL to have it allocated automatically

  Output Parameters:
+ csize - The number of values in the closure
- values - The array of values. If the user provided NULL, it is a borrowed array and should not be freed

$ Note that DMPlexVecGetClosure/DMPlexVecRestoreClosure only allocates the values array if it set to NULL in the
$ calling function. This is because DMPlexVecGetClosure() is typically called in the inner loop of a Vec or Mat
$ assembly function, and a user may already have allocated storage for this operation.
$
$ A typical use could be
$
$  values = NULL;
$  ierr = DMPlexVecGetClosure(dm, NULL, v, p, &clSize, &values);CHKERRQ(ierr);
$  for (cl = 0; cl < clSize; ++cl) {
$    <Compute on closure>
$  }
$  ierr = DMPlexVecRestoreClosure(dm, NULL, v, p, &clSize, &values);CHKERRQ(ierr);
$
$ or
$
$  PetscMalloc1(clMaxSize, &values);
$  for (p = pStart; p < pEnd; ++p) {
$    clSize = clMaxSize;
$    ierr = DMPlexVecGetClosure(dm, NULL, v, p, &clSize, &values);CHKERRQ(ierr);
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
  PetscScalar       *array;
  const PetscScalar *vArray;
  PetscInt          *points = NULL;
  const PetscInt    *clp, *perm;
  PetscInt           depth, numFields, numPoints, size;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (depth == 1 && numFields < 2) {
    ierr = DMPlexVecGetClosure_Depth1_Static(dm, section, v, point, csize, values);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Get points */
  ierr = DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  ierr = PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, NULL, &perm);CHKERRQ(ierr);
  /* Get array */
  if (!values || !*values) {
    PetscInt asize = 0, dof, p;

    for (p = 0; p < numPoints*2; p += 2) {
      ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
      asize += dof;
    }
    if (!values) {
      ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
      if (csize) *csize = asize;
      PetscFunctionReturn(0);
    }
    ierr = DMGetWorkArray(dm, asize, MPIU_SCALAR, &array);CHKERRQ(ierr);
  } else {
    array = *values;
  }
  ierr = VecGetArrayRead(v, &vArray);CHKERRQ(ierr);
  /* Get values */
  if (numFields > 0) {ierr = DMPlexVecGetClosure_Fields_Static(dm, section, numPoints, points, numFields, perm, vArray, &size, array);CHKERRQ(ierr);}
  else               {ierr = DMPlexVecGetClosure_Static(dm, section, numPoints, points, perm, vArray, &size, array);CHKERRQ(ierr);}
  /* Cleanup points */
  ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  /* Cleanup array */
  ierr = VecRestoreArrayRead(v, &vArray);CHKERRQ(ierr);
  if (!*values) {
    if (csize) *csize = size;
    *values = array;
  } else {
    if (size > *csize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Size of input array %D < actual size %D", *csize, size);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Should work without recalculating size */
  ierr = DMRestoreWorkArray(dm, size, MPIU_SCALAR, (void*) values);CHKERRQ(ierr);
  *values = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void add   (PetscScalar *x, PetscScalar y) {*x += y;}
PETSC_STATIC_INLINE void insert(PetscScalar *x, PetscScalar y) {*x  = y;}

PETSC_STATIC_INLINE PetscErrorCode updatePoint_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar*, PetscScalar), PetscBool setBC, const PetscInt perm[], const PetscScalar flip[], const PetscInt clperm[], const PetscScalar values[], PetscInt offset, PetscScalar array[])
{
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar    *a;
  PetscInt        off, cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
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
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode updatePointBC_private(PetscSection section, PetscInt point, PetscInt dof, void (*fuse)(PetscScalar*, PetscScalar), const PetscInt perm[], const PetscScalar flip[], const PetscInt clperm[], const PetscScalar values[], PetscInt offset, PetscScalar array[])
{
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscScalar    *a;
  PetscInt        off, cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
  a    = &array[off];
  if (cdof) {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode updatePointFields_private(PetscSection section, PetscInt point, const PetscInt *perm, const PetscScalar *flip, PetscInt f, void (*fuse)(PetscScalar*, PetscScalar), PetscBool setBC, const PetscInt clperm[], const PetscScalar values[], PetscInt *offset, PetscScalar array[])
{
  PetscScalar    *a;
  PetscInt        fdof, foff, fcdof, foffset = *offset;
  const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
  PetscInt        cind = 0, b;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldConstraintDof(section, point, f, &fcdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldOffset(section, point, f, &foff);CHKERRQ(ierr);
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
    ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode updatePointFieldsBC_private(PetscSection section, PetscInt point, const PetscInt perm[], const PetscScalar flip[], PetscInt f, PetscInt Ncc, const PetscInt comps[], void (*fuse)(PetscScalar*, PetscScalar), const PetscInt clperm[], const PetscScalar values[], PetscInt *offset, PetscScalar array[])
{
  PetscScalar    *a;
  PetscInt        fdof, foff, fcdof, foffset = *offset;
  const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
  PetscInt        cind = 0, ncind = 0, b;
  PetscBool       ncSet, fcSet;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldConstraintDof(section, point, f, &fcdof);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldOffset(section, point, f, &foff);CHKERRQ(ierr);
  a    = &array[foff];
  if (fcdof) {
    /* We just override fcdof and fcdofs with Ncc and comps */
    ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
    if (clperm) {
      if (perm) {
        if (comps) {
          for (b = 0; b < fdof; b++) {
            ncSet = fcSet = PETSC_FALSE;
            if ((ncind < Ncc)  && (b == comps[ncind])) {++ncind; ncSet = PETSC_TRUE;}
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
            if ((ncind < Ncc)  && (b == comps[ncind])) {++ncind; ncSet = PETSC_TRUE;}
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
            if ((ncind < Ncc)  && (b == comps[ncind])) {++ncind; ncSet = PETSC_TRUE;}
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
            if ((ncind < Ncc)  && (b == comps[ncind])) {++ncind; ncSet = PETSC_TRUE;}
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

PETSC_STATIC_INLINE PetscErrorCode DMPlexVecSetClosure_Depth1_Static(DM dm, PetscSection section, Vec v, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  PetscScalar    *array;
  const PetscInt *cone, *coneO;
  PetscInt        pStart, pEnd, p, numPoints, off, dof;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, point, &numPoints);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, point, &coneO);CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  for (p = 0, off = 0; p <= numPoints; ++p, off += dof) {
    const PetscInt cp = !p ? point : cone[p-1];
    const PetscInt o  = !p ? 0     : coneO[p-1];

    if ((cp < pStart) || (cp >= pEnd)) {dof = 0; continue;}
    ierr = PetscSectionGetDof(section, cp, &dof);CHKERRQ(ierr);
    /* ADD_VALUES */
    {
      const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
      PetscScalar    *a;
      PetscInt        cdof, coff, cind = 0, k;

      ierr = PetscSectionGetConstraintDof(section, cp, &cdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, cp, &coff);CHKERRQ(ierr);
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
        ierr = PetscSectionGetConstraintIndices(section, cp, &cdofs);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
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
  const PetscInt *clp, *clperm;
  PetscInt        depth, numFields, numPoints, p;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (depth == 1 && numFields < 2 && mode == ADD_VALUES) {
    ierr = DMPlexVecSetClosure_Depth1_Static(dm, section, v, point, values, mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Get points */
  ierr = PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, NULL, &clperm);CHKERRQ(ierr);
  ierr = DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  /* Get array */
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  /* Get values */
  if (numFields > 0) {
    PetscInt offset = 0, f;
    for (f = 0; f < numFields; ++f) {
      const PetscInt    **perms = NULL;
      const PetscScalar **flips = NULL;

      ierr = PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms,&flips);CHKERRQ(ierr);
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
        SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
      }
      ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms,&flips);CHKERRQ(ierr);
    }
  } else {
    PetscInt dof, off;
    const PetscInt    **perms = NULL;
    const PetscScalar **flips = NULL;

    ierr = PetscSectionGetPointSyms(section,numPoints,points,&perms,&flips);CHKERRQ(ierr);
    switch (mode) {
    case INSERT_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        updatePoint_private(section, point, dof, insert, PETSC_FALSE, perm, flip, clperm, values, off, array);
      } break;
    case INSERT_ALL_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        updatePoint_private(section, point, dof, insert, PETSC_TRUE,  perm, flip, clperm, values, off, array);
      } break;
    case INSERT_BC_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        updatePointBC_private(section, point, dof, insert,  perm, flip, clperm, values, off, array);
      } break;
    case ADD_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        updatePoint_private(section, point, dof, add,    PETSC_FALSE, perm, flip, clperm, values, off, array);
      } break;
    case ADD_ALL_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        updatePoint_private(section, point, dof, add,    PETSC_TRUE,  perm, flip, clperm, values, off, array);
      } break;
    case ADD_BC_VALUES:
      for (p = 0, off = 0; p < numPoints; p++, off += dof) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
        updatePointBC_private(section, point, dof, add,  perm, flip, clperm, values, off, array);
      } break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
    }
    ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms,&flips);CHKERRQ(ierr);
  }
  /* Cleanup points */
  ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  /* Cleanup array */
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Unlike DMPlexVecSetClosure(), this uses plex-native closure permutation, not a user-specified permutation such as DMPlexSetClosurePermutationTensor(). */
PetscErrorCode DMPlexVecSetFieldClosure_Internal(DM dm, PetscSection section, Vec v, PetscBool fieldActive[], PetscInt point, PetscInt Ncc, const PetscInt comps[], const PetscScalar values[], InsertMode mode)
{
  PetscSection      clSection;
  IS                clPoints;
  PetscScalar       *array;
  PetscInt          *points = NULL;
  const PetscInt    *clp;
  PetscInt          numFields, numPoints, p;
  PetscInt          offset = 0, f;
  PetscErrorCode    ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  /* Get points */
  ierr = DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  /* Get array */
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  /* Get values */
  for (f = 0; f < numFields; ++f) {
    const PetscInt    **perms = NULL;
    const PetscScalar **flips = NULL;

    if (!fieldActive[f]) {
      for (p = 0; p < numPoints*2; p += 2) {
        PetscInt fdof;
        ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
        offset += fdof;
      }
      continue;
    }
    ierr = PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms,&flips);CHKERRQ(ierr);
    switch (mode) {
    case INSERT_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        updatePointFields_private(section, point, perm, flip, f, insert, PETSC_FALSE, NULL, values, &offset, array);
      } break;
    case INSERT_ALL_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        updatePointFields_private(section, point, perm, flip, f, insert, PETSC_TRUE, NULL, values, &offset, array);
        } break;
    case INSERT_BC_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        updatePointFieldsBC_private(section, point, perm, flip, f, Ncc, comps, insert, NULL, values, &offset, array);
      } break;
    case ADD_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        updatePointFields_private(section, point, perm, flip, f, add, PETSC_FALSE, NULL, values, &offset, array);
      } break;
    case ADD_ALL_VALUES:
      for (p = 0; p < numPoints; p++) {
        const PetscInt    point = points[2*p];
        const PetscInt    *perm = perms ? perms[p] : NULL;
        const PetscScalar *flip = flips ? flips[p] : NULL;
        updatePointFields_private(section, point, perm, flip, f, add, PETSC_TRUE, NULL, values, &offset, array);
      } break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid insert mode %d", mode);
    }
    ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms,&flips);CHKERRQ(ierr);
  }
  /* Cleanup points */
  ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  /* Cleanup array */
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexPrintMatSetValues(PetscViewer viewer, Mat A, PetscInt point, PetscInt numRIndices, const PetscInt rindices[], PetscInt numCIndices, const PetscInt cindices[], const PetscScalar values[])
{
  PetscMPIInt    rank;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "[%d]mat for point %D\n", rank, point);CHKERRQ(ierr);
  for (i = 0; i < numRIndices; i++) {ierr = PetscViewerASCIIPrintf(viewer, "[%d]mat row indices[%D] = %D\n", rank, i, rindices[i]);CHKERRQ(ierr);}
  for (i = 0; i < numCIndices; i++) {ierr = PetscViewerASCIIPrintf(viewer, "[%d]mat col indices[%D] = %D\n", rank, i, cindices[i]);CHKERRQ(ierr);}
  numCIndices = numCIndices ? numCIndices : numRIndices;
  for (i = 0; i < numRIndices; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "[%d]", rank);CHKERRQ(ierr);
    for (j = 0; j < numCIndices; j++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer, " (%g,%g)", (double)PetscRealPart(values[i*numCIndices+j]), (double)PetscImaginaryPart(values[i*numCIndices+j]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer, " %g", (double)values[i*numCIndices+j]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
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
. setBC   - The flag determining whether to include indices of bounsary values
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!islocal && setBC) SETERRQ(PetscObjectComm((PetscObject)section),PETSC_ERR_ARG_INCOMP,"setBC incompatible with global indices; use a local section or disable setBC");
  ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  if (!cdof || setBC) {
    for (k = 0; k < dof; ++k) {
      const PetscInt preind = perm ? *loff+perm[k] : *loff+k;
      const PetscInt ind    = indperm ? indperm[preind] : preind;

      indices[ind] = off + k;
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!islocal && setBC) SETERRQ(PetscObjectComm((PetscObject)section),PETSC_ERR_ARG_INCOMP,"setBC incompatible with global indices; use a local section or disable setBC");
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (f = 0, foff = 0; f < numFields; ++f) {
    PetscInt        fdof, cfdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, b;
    const PetscInt  *perm = (perms && perms[f]) ? perms[f][permsoff] : NULL;

    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cfdof);CHKERRQ(ierr);
    if (!cfdof || setBC) {
      for (b = 0; b < fdof; ++b) {
        const PetscInt preind = perm ? foffs[f]+perm[b] : foffs[f]+b;
        const PetscInt ind    = indperm ? indperm[preind] : preind;

        indices[ind] = off+foff+b;
      }
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscInt        fdof, cfdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, b;
    const PetscInt  *perm = (perms && perms[f]) ? perms[f][permsoff] : NULL;

    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cfdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldOffset(globalSection, point, f, &foff);CHKERRQ(ierr);
    if (!cfdof) {
      for (b = 0; b < fdof; ++b) {
        const PetscInt preind = perm ? foffs[f]+perm[b] : foffs[f]+b;
        const PetscInt ind    = indperm ? indperm[preind] : preind;

        indices[ind] = foff+b;
      }
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);

  ierr = DMPlexGetAnchors(dm,&aSec,&aIS);CHKERRQ(ierr);
  /* if there are point-to-point constraints */
  if (aSec) {
    ierr = PetscArrayzero(newOffsets, 32);CHKERRQ(ierr);
    ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(aSec,&aStart,&aEnd);CHKERRQ(ierr);
    /* figure out how many points are going to be in the new element matrix
     * (we allow double counting, because it's all just going to be summed
     * into the global matrix anyway) */
    for (p = 0; p < 2*numPoints; p+=2) {
      PetscInt b    = points[p];
      PetscInt bDof = 0, bSecDof;

      ierr = PetscSectionGetDof(section,b,&bSecDof);CHKERRQ(ierr);
      if (!bSecDof) {
        continue;
      }
      if (b >= aStart && b < aEnd) {
        ierr = PetscSectionGetDof(aSec,b,&bDof);CHKERRQ(ierr);
      }
      if (bDof) {
        /* this point is constrained */
        /* it is going to be replaced by its anchors */
        PetscInt bOff, q;

        anyConstrained = PETSC_TRUE;
        newNumPoints  += bDof;
        ierr = PetscSectionGetOffset(aSec,b,&bOff);CHKERRQ(ierr);
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q];
          PetscInt aDof;

          ierr           = PetscSectionGetDof(section,a,&aDof);CHKERRQ(ierr);
          newNumIndices += aDof;
          for (f = 0; f < numFields; ++f) {
            PetscInt fDof;

            ierr             = PetscSectionGetFieldDof(section, a, f, &fDof);CHKERRQ(ierr);
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

          ierr = PetscSectionGetFieldDof(section, b, f, &fDof);CHKERRQ(ierr);
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
    if (aSec) {ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);}
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
    if (aSec) {ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }

  if (numFields && newOffsets[numFields] != newNumIndices) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", newOffsets[numFields], newNumIndices);

  ierr = DMGetDefaultConstraints(dm, &cSec, &cMat);CHKERRQ(ierr);

  /* workspaces */
  if (numFields) {
    for (f = 0; f < numFields; f++) {
      ierr = DMGetWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[f]);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm,numPoints+1,MPIU_INT,&newPointOffsets[f]);CHKERRQ(ierr);
    }
  }
  else {
    ierr = DMGetWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[0]);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm,numPoints,MPIU_INT,&newPointOffsets[0]);CHKERRQ(ierr);
  }

  /* get workspaces for the point-to-point matrices */
  if (numFields) {
    PetscInt totalOffset, totalMatOffset;

    for (p = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt bDof = 0, bSecDof;

      ierr = PetscSectionGetDof(section,b,&bSecDof);CHKERRQ(ierr);
      if (!bSecDof) {
        for (f = 0; f < numFields; f++) {
          newPointOffsets[f][p + 1] = 0;
          pointMatOffsets[f][p + 1] = 0;
        }
        continue;
      }
      if (b >= aStart && b < aEnd) {
        ierr = PetscSectionGetDof(aSec, b, &bDof);CHKERRQ(ierr);
      }
      if (bDof) {
        for (f = 0; f < numFields; f++) {
          PetscInt fDof, q, bOff, allFDof = 0;

          ierr = PetscSectionGetFieldDof(section, b, f, &fDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(aSec, b, &bOff);CHKERRQ(ierr);
          for (q = 0; q < bDof; q++) {
            PetscInt a = anchors[bOff + q];
            PetscInt aFDof;

            ierr     = PetscSectionGetFieldDof(section, a, f, &aFDof);CHKERRQ(ierr);
            allFDof += aFDof;
          }
          newPointOffsets[f][p+1] = allFDof;
          pointMatOffsets[f][p+1] = fDof * allFDof;
        }
      }
      else {
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          ierr = PetscSectionGetFieldDof(section, b, f, &fDof);CHKERRQ(ierr);
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
      ierr = DMGetWorkArray(dm,pointMatOffsets[f][numPoints],MPIU_SCALAR,&pointMat[f]);CHKERRQ(ierr);
    }
  }
  else {
    for (p = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt bDof = 0, bSecDof;

      ierr = PetscSectionGetDof(section,b,&bSecDof);CHKERRQ(ierr);
      if (!bSecDof) {
        newPointOffsets[0][p + 1] = 0;
        pointMatOffsets[0][p + 1] = 0;
        continue;
      }
      if (b >= aStart && b < aEnd) {
        ierr = PetscSectionGetDof(aSec, b, &bDof);CHKERRQ(ierr);
      }
      if (bDof) {
        PetscInt bOff, q, allDof = 0;

        ierr = PetscSectionGetOffset(aSec, b, &bOff);CHKERRQ(ierr);
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aDof;

          ierr    = PetscSectionGetDof(section, a, &aDof);CHKERRQ(ierr);
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
    ierr = DMGetWorkArray(dm,pointMatOffsets[0][numPoints],MPIU_SCALAR,&pointMat[0]);CHKERRQ(ierr);
  }

  /* output arrays */
  ierr = DMGetWorkArray(dm,2*newNumPoints,MPIU_INT,&newPoints);CHKERRQ(ierr);

  /* get the point-to-point matrices; construct newPoints */
  ierr = PetscSectionGetMaxDof(aSec, &maxAnchor);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,maxDof,MPIU_INT,&indices);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,maxAnchor*maxDof,MPIU_INT,&newIndices);CHKERRQ(ierr);
  if (numFields) {
    for (p = 0, newP = 0; p < numPoints; p++) {
      PetscInt b    = points[2*p];
      PetscInt o    = points[2*p+1];
      PetscInt bDof = 0, bSecDof;

      ierr = PetscSectionGetDof(section, b, &bSecDof);CHKERRQ(ierr);
      if (!bSecDof) {
        continue;
      }
      if (b >= aStart && b < aEnd) {
        ierr = PetscSectionGetDof(aSec, b, &bDof);CHKERRQ(ierr);
      }
      if (bDof) {
        PetscInt fStart[32], fEnd[32], fAnchorStart[32], fAnchorEnd[32], bOff, q;

        fStart[0] = 0;
        fEnd[0]   = 0;
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          ierr        = PetscSectionGetFieldDof(cSec, b, f, &fDof);CHKERRQ(ierr);
          fStart[f+1] = fStart[f] + fDof;
          fEnd[f+1]   = fStart[f+1];
        }
        ierr = PetscSectionGetOffset(cSec, b, &bOff);CHKERRQ(ierr);
        ierr = DMPlexGetIndicesPointFields_Internal(cSec, PETSC_TRUE, b, bOff, fEnd, PETSC_TRUE, perms, p, NULL, indices);CHKERRQ(ierr);

        fAnchorStart[0] = 0;
        fAnchorEnd[0]   = 0;
        for (f = 0; f < numFields; f++) {
          PetscInt fDof = newPointOffsets[f][p + 1] - newPointOffsets[f][p];

          fAnchorStart[f+1] = fAnchorStart[f] + fDof;
          fAnchorEnd[f+1]   = fAnchorStart[f + 1];
        }
        ierr = PetscSectionGetOffset(aSec, b, &bOff);CHKERRQ(ierr);
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aOff;

          /* we take the orientation of ap into account in the order that we constructed the indices above: the newly added points have no orientation */
          newPoints[2*(newP + q)]     = a;
          newPoints[2*(newP + q) + 1] = 0;
          ierr = PetscSectionGetOffset(section, a, &aOff);CHKERRQ(ierr);
          ierr = DMPlexGetIndicesPointFields_Internal(section, PETSC_TRUE, a, aOff, fAnchorEnd, PETSC_TRUE, NULL, -1, NULL, newIndices);CHKERRQ(ierr);
        }
        newP += bDof;

        if (outValues) {
          /* get the point-to-point submatrix */
          for (f = 0; f < numFields; f++) {
            ierr = MatGetValues(cMat,fEnd[f]-fStart[f],indices + fStart[f],fAnchorEnd[f] - fAnchorStart[f],newIndices + fAnchorStart[f],pointMat[f] + pointMatOffsets[f][p]);CHKERRQ(ierr);
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

      ierr = PetscSectionGetDof(section, b, &bSecDof);CHKERRQ(ierr);
      if (!bSecDof) {
        continue;
      }
      if (b >= aStart && b < aEnd) {
        ierr = PetscSectionGetDof(aSec, b, &bDof);CHKERRQ(ierr);
      }
      if (bDof) {
        PetscInt bEnd = 0, bAnchorEnd = 0, bOff;

        ierr = PetscSectionGetOffset(cSec, b, &bOff);CHKERRQ(ierr);
        ierr = DMPlexGetIndicesPoint_Internal(cSec, PETSC_TRUE, b, bOff, &bEnd, PETSC_TRUE, (perms && perms[0]) ? perms[0][p] : NULL, NULL, indices);CHKERRQ(ierr);

        ierr = PetscSectionGetOffset (aSec, b, &bOff);CHKERRQ(ierr);
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aOff;

          /* we take the orientation of ap into account in the order that we constructed the indices above: the newly added points have no orientation */

          newPoints[2*(newP + q)]     = a;
          newPoints[2*(newP + q) + 1] = 0;
          ierr = PetscSectionGetOffset(section, a, &aOff);CHKERRQ(ierr);
          ierr = DMPlexGetIndicesPoint_Internal(section, PETSC_TRUE, a, aOff, &bAnchorEnd, PETSC_TRUE, NULL, NULL, newIndices);CHKERRQ(ierr);
        }
        newP += bDof;

        /* get the point-to-point submatrix */
        if (outValues) {
          ierr = MatGetValues(cMat,bEnd,indices,bAnchorEnd,newIndices,pointMat[0] + pointMatOffsets[0][p]);CHKERRQ(ierr);
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
    ierr = DMGetWorkArray(dm,newNumIndices*numIndices,MPIU_SCALAR,&tmpValues);CHKERRQ(ierr);
    ierr = PetscArrayzero(tmpValues,newNumIndices*numIndices);CHKERRQ(ierr);
    /* multiply constraints on the right */
    if (numFields) {
      for (f = 0; f < numFields; f++) {
        PetscInt oldOff = offsets[f];

        for (p = 0; p < numPoints; p++) {
          PetscInt cStart = newPointOffsets[f][p];
          PetscInt b      = points[2 * p];
          PetscInt c, r, k;
          PetscInt dof;

          ierr = PetscSectionGetFieldDof(section,b,f,&dof);CHKERRQ(ierr);
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

        ierr = PetscSectionGetDof(section,b,&dof);CHKERRQ(ierr);
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
      ierr = DMGetWorkArray(dm,newNumIndices*newNumIndices,MPIU_SCALAR,&newValues);CHKERRQ(ierr);
      ierr = PetscArrayzero(newValues,newNumIndices*newNumIndices);CHKERRQ(ierr);
      /* multiply constraints transpose on the left */
      if (numFields) {
        for (f = 0; f < numFields; f++) {
          PetscInt oldOff = offsets[f];

          for (p = 0; p < numPoints; p++) {
            PetscInt rStart = newPointOffsets[f][p];
            PetscInt b      = points[2 * p];
            PetscInt c, r, k;
            PetscInt dof;

            ierr = PetscSectionGetFieldDof(section,b,f,&dof);CHKERRQ(ierr);
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

          ierr = PetscSectionGetDof(section,b,&dof);CHKERRQ(ierr);
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

      ierr = DMRestoreWorkArray(dm,newNumIndices*numIndices,MPIU_SCALAR,&tmpValues);CHKERRQ(ierr);
    }
    else {
      newValues = tmpValues;
    }
  }

  /* clean up */
  ierr = DMRestoreWorkArray(dm,maxDof,MPIU_INT,&indices);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,maxAnchor*maxDof,MPIU_INT,&newIndices);CHKERRQ(ierr);

  if (numFields) {
    for (f = 0; f < numFields; f++) {
      ierr = DMRestoreWorkArray(dm,pointMatOffsets[f][numPoints],MPIU_SCALAR,&pointMat[f]);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[f]);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&newPointOffsets[f]);CHKERRQ(ierr);
    }
  }
  else {
    ierr = DMRestoreWorkArray(dm,pointMatOffsets[0][numPoints],MPIU_SCALAR,&pointMat[0]);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&pointMatOffsets[0]);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,numPoints+1,MPIU_INT,&newPointOffsets[0]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);

  /* output */
  if (outPoints) {
    *outPoints = newPoints;
  }
  else {
    ierr = DMRestoreWorkArray(dm,2*newNumPoints,MPIU_INT,&newPoints);CHKERRQ(ierr);
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
  DMPlexGetClosureIndices - Get the global indices for all local points in the closure of the given point

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the points (a local section)
. idxSection - The section on which to obtain indices (may be local or global)
- point - The mesh point

  Output parameters:
+ numIndices - The number of indices
. indices - The indices
- outOffsets - Field offset if not NULL

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
PetscErrorCode DMPlexGetClosureIndices(DM dm, PetscSection section, PetscSection idxSection, PetscInt point, PetscInt *numIndices, PetscInt **indices, PetscInt *outOffsets)
{
  PetscBool       isLocal = (PetscBool)(section == idxSection);
  PetscSection    clSection;
  IS              clPoints;
  const PetscInt *clp, *clperm;
  const PetscInt  **perms[32] = {NULL};
  PetscInt       *points = NULL, *pointsNew;
  PetscInt        numPoints, numPointsNew;
  PetscInt        offsets[32];
  PetscInt        Nf, Nind, NindNew, off, idxOff, f, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(idxSection, PETSC_SECTION_CLASSID, 3);
  if (numIndices) PetscValidPointer(numIndices, 4);
  PetscValidPointer(indices, 5);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  if (Nf > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", Nf);
  ierr = PetscArrayzero(offsets, 32);CHKERRQ(ierr);
  /* Get points in closure */
  ierr = DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  ierr = PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, NULL, &clperm);CHKERRQ(ierr);
  /* Get number of indices and indices per field */
  for (p = 0, Nind = 0; p < numPoints*2; p += 2) {
    PetscInt dof, fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      ierr = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    Nind += dof;
  }
  for (f = 1; f < Nf; ++f) offsets[f+1] += offsets[f];
  if (Nf && offsets[Nf] != Nind) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", offsets[Nf], Nind);
  if (!Nf) offsets[1] = Nind;
  /* Get dual space symmetries */
  for (f = 0; f < PetscMax(1,Nf); f++) {
    if (Nf) {ierr = PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
    else    {ierr = PetscSectionGetPointSyms(section,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
  }
  /* Correct for hanging node constraints */
  {
    ierr = DMPlexAnchorsModifyMat(dm, section, numPoints, Nind, points, perms, NULL, &numPointsNew, &NindNew, &pointsNew, NULL, offsets, PETSC_TRUE);CHKERRQ(ierr);
    if (numPointsNew) {
      for (f = 0; f < PetscMax(1,Nf); f++) {
        if (Nf) {ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
        else    {ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
      }
      for (f = 0; f < PetscMax(1,Nf); f++) {
        if (Nf) {ierr = PetscSectionGetFieldPointSyms(section,f,numPointsNew,pointsNew,&perms[f],NULL);CHKERRQ(ierr);}
        else    {ierr = PetscSectionGetPointSyms(section,numPointsNew,pointsNew,&perms[f],NULL);CHKERRQ(ierr);}
      }
      ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
      numPoints = numPointsNew;
      Nind      = NindNew;
      points    = pointsNew;
    }
  }
  /* Calculate indices */
  ierr = DMGetWorkArray(dm, Nind, MPIU_INT, indices);CHKERRQ(ierr);
  if (Nf) {
    if (outOffsets) {
      PetscInt f;

      for (f = 0; f <= Nf; f++) {
        outOffsets[f] = offsets[f];
      }
    }
    for (p = 0; p < numPoints; p++) {
      ierr = PetscSectionGetOffset(idxSection, points[2*p], &idxOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPointFields_Internal(section, isLocal, points[2*p], idxOff < 0 ? -(idxOff+1) : idxOff, offsets, PETSC_FALSE, perms, p, clperm, *indices);CHKERRQ(ierr);
    }
  } else {
    for (p = 0, off = 0; p < numPoints; p++) {
      const PetscInt *perm = perms[0] ? perms[0][p] : NULL;

      ierr = PetscSectionGetOffset(idxSection, points[2*p], &idxOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPoint_Internal(section, isLocal, points[2*p], idxOff < 0 ? -(idxOff+1) : idxOff, &off, PETSC_FALSE, perm, clperm, *indices);CHKERRQ(ierr);
    }
  }
  /* Cleanup points */
  for (f = 0; f < PetscMax(1,Nf); f++) {
    if (Nf) {ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
    else    {ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
  }
  if (numPointsNew) {
    ierr = DMRestoreWorkArray(dm, 2*numPointsNew, MPIU_INT, &pointsNew);CHKERRQ(ierr);
  } else {
    ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  }
  if (numIndices) *numIndices = Nind;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreClosureIndices - Restore the indices in a vector v for all points in the closure of the given point

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. globalSection - The section describing the parallel layout in v, or NULL to use the default section
. point - The mesh point
. numIndices - The number of indices
. indices - The indices
- outOffsets - Field offset if not NULL

  Level: advanced

.seealso DMPlexGetClosureIndices(), DMPlexVecGetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexRestoreClosureIndices(DM dm, PetscSection section, PetscSection globalSection, PetscInt point, PetscInt *numIndices, PetscInt **indices,PetscInt *outOffsets)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(indices, 5);
  ierr = DMRestoreWorkArray(dm, 0, MPIU_INT, indices);CHKERRQ(ierr);
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

.seealso DMPlexVecGetClosure(), DMPlexVecSetClosure()
@*/
PetscErrorCode DMPlexMatSetClosure(DM dm, PetscSection section, PetscSection globalSection, Mat A, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  DM_Plex            *mesh   = (DM_Plex*) dm->data;
  PetscSection        clSection;
  IS                  clPoints;
  PetscInt           *points = NULL, *newPoints;
  const PetscInt     *clp, *clperm;
  PetscInt           *indices;
  PetscInt            offsets[32];
  const PetscInt    **perms[32] = {NULL};
  const PetscScalar **flips[32] = {NULL};
  PetscInt            numFields, numPoints, newNumPoints, numIndices, newNumIndices, dof, off, globalOff, p, f;
  PetscScalar        *valCopy = NULL;
  PetscScalar        *newValues;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  if (!globalSection) {ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 4);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscArrayzero(offsets, 32);CHKERRQ(ierr);
  ierr = PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, NULL, &clperm);CHKERRQ(ierr);
  ierr = DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  for (p = 0, numIndices = 0; p < numPoints*2; p += 2) {
    PetscInt fdof;

    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      ierr          = PetscSectionGetFieldDof(section, points[p], f, &fdof);CHKERRQ(ierr);
      offsets[f+1] += fdof;
    }
    numIndices += dof;
  }
  for (f = 1; f < numFields; ++f) offsets[f+1] += offsets[f];

  if (numFields && offsets[numFields] != numIndices) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", offsets[numFields], numIndices);
  /* Get symmetries */
  for (f = 0; f < PetscMax(1,numFields); f++) {
    if (numFields) {ierr = PetscSectionGetFieldPointSyms(section,f,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
    else           {ierr = PetscSectionGetPointSyms(section,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
    if (values && flips[f]) { /* may need to apply sign changes to the element matrix */
      PetscInt foffset = offsets[f];

      for (p = 0; p < numPoints; p++) {
        PetscInt point          = points[2*p], fdof;
        const PetscScalar *flip = flips[f] ? flips[f][p] : NULL;

        if (!numFields) {
          ierr = PetscSectionGetDof(section,point,&fdof);CHKERRQ(ierr);
        } else {
          ierr = PetscSectionGetFieldDof(section,point,f,&fdof);CHKERRQ(ierr);
        }
        if (flip) {
          PetscInt i, j, k;

          if (!valCopy) {
            ierr = DMGetWorkArray(dm,numIndices*numIndices,MPIU_SCALAR,&valCopy);CHKERRQ(ierr);
            for (j = 0; j < numIndices * numIndices; j++) valCopy[j] = values[j];
            values = valCopy;
          }
          for (i = 0; i < fdof; i++) {
            PetscScalar fval = flip[i];

            for (k = 0; k < numIndices; k++) {
              valCopy[numIndices * (foffset + i) + k] *= fval;
              valCopy[numIndices * k + (foffset + i)] *= fval;
            }
          }
        }
        foffset += fdof;
      }
    }
  }
  ierr = DMPlexAnchorsModifyMat(dm,section,numPoints,numIndices,points,perms,values,&newNumPoints,&newNumIndices,&newPoints,&newValues,offsets,PETSC_TRUE);CHKERRQ(ierr);
  if (newNumPoints) {
    if (valCopy) {
      ierr = DMRestoreWorkArray(dm,numIndices*numIndices,MPIU_SCALAR,&valCopy);CHKERRQ(ierr);
    }
    for (f = 0; f < PetscMax(1,numFields); f++) {
      if (numFields) {ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
      else           {ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
    }
    for (f = 0; f < PetscMax(1,numFields); f++) {
      if (numFields) {ierr = PetscSectionGetFieldPointSyms(section,f,newNumPoints,newPoints,&perms[f],&flips[f]);CHKERRQ(ierr);}
      else           {ierr = PetscSectionGetPointSyms(section,newNumPoints,newPoints,&perms[f],&flips[f]);CHKERRQ(ierr);}
    }
    ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
    numPoints  = newNumPoints;
    numIndices = newNumIndices;
    points     = newPoints;
    values     = newValues;
  }
  ierr = DMGetWorkArray(dm, numIndices, MPIU_INT, &indices);CHKERRQ(ierr);
  if (numFields) {
    PetscBool useFieldOffsets;

    ierr = PetscSectionGetUseFieldOffsets(globalSection, &useFieldOffsets);CHKERRQ(ierr);
    if (useFieldOffsets) {
      for (p = 0; p < numPoints; p++) {
        ierr = DMPlexGetIndicesPointFieldsSplit_Internal(section, globalSection, points[2*p], offsets, perms, p, clperm, indices);CHKERRQ(ierr);
      }
    } else {
      for (p = 0; p < numPoints; p++) {
        ierr = PetscSectionGetOffset(globalSection, points[2*p], &globalOff);CHKERRQ(ierr);
        /* Note that we pass a local section even though we're using global offsets.  This is because global sections do
         * not (at the time of this writing) have fields set. They probably should, in which case we would pass the
         * global section. */
        ierr = DMPlexGetIndicesPointFields_Internal(section, PETSC_FALSE, points[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, perms, p, clperm, indices);CHKERRQ(ierr);
      }
    }
  } else {
    for (p = 0, off = 0; p < numPoints; p++) {
      const PetscInt *perm = perms[0] ? perms[0][p] : NULL;
      ierr = PetscSectionGetOffset(globalSection, points[2*p], &globalOff);CHKERRQ(ierr);
      /* Note that we pass a local section even though we're using global offsets.  This is because global sections do
       * not (at the time of this writing) have fields set. They probably should, in which case we would pass the
       * global section. */
      ierr = DMPlexGetIndicesPoint_Internal(section, PETSC_FALSE, points[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, clperm, indices);CHKERRQ(ierr);
    }
  }
  if (mesh->printSetValues) {ierr = DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numIndices, indices, 0, NULL, values);CHKERRQ(ierr);}
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, values, mode);CHKERRQ(ierr);
  if (mesh->printFEM > 1) {
    PetscInt i;
    ierr = PetscPrintf(PETSC_COMM_SELF, "  Indices:");CHKERRQ(ierr);
    for (i = 0; i < numIndices; ++i) {ierr = PetscPrintf(PETSC_COMM_SELF, " %D", indices[i]);CHKERRQ(ierr);}
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
  }
  if (ierr) {
    PetscMPIInt    rank;
    PetscErrorCode ierr2;

    ierr2 = MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank);CHKERRQ(ierr2);
    ierr2 = (*PetscErrorPrintf)("[%d]ERROR in DMPlexMatSetClosure\n", rank);CHKERRQ(ierr2);
    ierr2 = DMPlexPrintMatSetValues(PETSC_VIEWER_STDERR_SELF, A, point, numIndices, indices, 0, NULL, values);CHKERRQ(ierr2);
    ierr2 = DMRestoreWorkArray(dm, numIndices, MPIU_INT, &indices);CHKERRQ(ierr2);
    CHKERRQ(ierr);
  }
  for (f = 0; f < PetscMax(1,numFields); f++) {
    if (numFields) {ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
    else           {ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
  }
  if (newNumPoints) {
    ierr = DMRestoreWorkArray(dm,newNumIndices*newNumIndices,MPIU_SCALAR,&newValues);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,2*newNumPoints,MPIU_INT,&newPoints);CHKERRQ(ierr);
  }
  else {
    if (valCopy) {
      ierr = DMRestoreWorkArray(dm,numIndices*numIndices,MPIU_SCALAR,&valCopy);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numIndices, MPIU_INT, &indices);CHKERRQ(ierr);
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
  CellRefiner     cellRefiner;
  PetscInt        numFields, numSubcells, maxFPoints, numFPoints, numCPoints, numFIndices, numCIndices, dof, off, globalOff, pStart, pEnd, p, q, r, s, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 4);
  if (!fsection) {ierr = DMGetLocalSection(dmf, &fsection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(fsection, PETSC_SECTION_CLASSID, 2);
  if (!csection) {ierr = DMGetLocalSection(dmc, &csection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(csection, PETSC_SECTION_CLASSID, 5);
  if (!globalFSection) {ierr = DMGetGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalFSection, PETSC_SECTION_CLASSID, 3);
  if (!globalCSection) {ierr = DMGetGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalCSection, PETSC_SECTION_CLASSID, 6);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 7);
  ierr = PetscSectionGetNumFields(fsection, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscArrayzero(foffsets, 32);CHKERRQ(ierr);
  ierr = PetscArrayzero(coffsets, 32);CHKERRQ(ierr);
  /* Column indices */
  ierr = DMPlexGetTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints);CHKERRQ(ierr);
  maxFPoints = numCPoints;
  /* Compress out points not in the section */
  /*   TODO: Squeeze out points with 0 dof as well */
  ierr = PetscSectionGetChart(csection, &pStart, &pEnd);CHKERRQ(ierr);
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

    ierr = PetscSectionGetDof(csection, cpoints[p], &dof);CHKERRQ(ierr);
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      ierr           = PetscSectionGetFieldDof(csection, cpoints[p], f, &fdof);CHKERRQ(ierr);
      coffsets[f+1] += fdof;
    }
    numCIndices += dof;
  }
  for (f = 1; f < numFields; ++f) coffsets[f+1] += coffsets[f];
  /* Row indices */
  ierr = DMPlexGetCellRefiner_Internal(dmc, &cellRefiner);CHKERRQ(ierr);
  ierr = CellRefinerGetAffineTransforms_Internal(cellRefiner, &numSubcells, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmf, maxFPoints*2*numSubcells, MPIU_INT, &ftotpoints);CHKERRQ(ierr);
  for (r = 0, q = 0; r < numSubcells; ++r) {
    /* TODO Map from coarse to fine cells */
    ierr = DMPlexGetTransitiveClosure(dmf, point*numSubcells + r, PETSC_TRUE, &numFPoints, &fpoints);CHKERRQ(ierr);
    /* Compress out points not in the section */
    ierr = PetscSectionGetChart(fsection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = 0; p < numFPoints*2; p += 2) {
      if ((fpoints[p] >= pStart) && (fpoints[p] < pEnd)) {
        ierr = PetscSectionGetDof(fsection, fpoints[p], &dof);CHKERRQ(ierr);
        if (!dof) continue;
        for (s = 0; s < q; ++s) if (fpoints[p] == ftotpoints[s*2]) break;
        if (s < q) continue;
        ftotpoints[q*2]   = fpoints[p];
        ftotpoints[q*2+1] = fpoints[p+1];
        ++q;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dmf, point, PETSC_TRUE, &numFPoints, &fpoints);CHKERRQ(ierr);
  }
  numFPoints = q;
  for (p = 0, numFIndices = 0; p < numFPoints*2; p += 2) {
    PetscInt fdof;

    ierr = PetscSectionGetDof(fsection, ftotpoints[p], &dof);CHKERRQ(ierr);
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      ierr           = PetscSectionGetFieldDof(fsection, ftotpoints[p], f, &fdof);CHKERRQ(ierr);
      foffsets[f+1] += fdof;
    }
    numFIndices += dof;
  }
  for (f = 1; f < numFields; ++f) foffsets[f+1] += foffsets[f];

  if (numFields && foffsets[numFields] != numFIndices) SETERRQ2(PetscObjectComm((PetscObject)dmf), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", foffsets[numFields], numFIndices);
  if (numFields && coffsets[numFields] != numCIndices) SETERRQ2(PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", coffsets[numFields], numCIndices);
  ierr = DMGetWorkArray(dmf, numFIndices, MPIU_INT, &findices);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmc, numCIndices, MPIU_INT, &cindices);CHKERRQ(ierr);
  if (numFields) {
    const PetscInt **permsF[32] = {NULL};
    const PetscInt **permsC[32] = {NULL};

    for (f = 0; f < numFields; f++) {
      ierr = PetscSectionGetFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL);CHKERRQ(ierr);
    }
    for (p = 0; p < numFPoints; p++) {
      ierr = PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPointFields_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, foffsets, PETSC_FALSE, permsF, p, fclperm, findices);CHKERRQ(ierr);
    }
    for (p = 0; p < numCPoints; p++) {
      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPointFields_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, coffsets, PETSC_FALSE, permsC, p, cclperm, cindices);CHKERRQ(ierr);
    }
    for (f = 0; f < numFields; f++) {
      ierr = PetscSectionRestoreFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL);CHKERRQ(ierr);
      ierr = PetscSectionRestoreFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL);CHKERRQ(ierr);
    }
  } else {
    const PetscInt **permsF = NULL;
    const PetscInt **permsC = NULL;

    ierr = PetscSectionGetPointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetPointSyms(csection,numCPoints,cpoints,&permsC,NULL);CHKERRQ(ierr);
    for (p = 0, off = 0; p < numFPoints; p++) {
      const PetscInt *perm = permsF ? permsF[p] : NULL;

      ierr = PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPoint_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, fclperm, findices);CHKERRQ(ierr);
    }
    for (p = 0, off = 0; p < numCPoints; p++) {
      const PetscInt *perm = permsC ? permsC[p] : NULL;

      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPoint_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, cclperm, cindices);CHKERRQ(ierr);
    }
    ierr = PetscSectionRestorePointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL);CHKERRQ(ierr);
    ierr = PetscSectionRestorePointSyms(csection,numCPoints,cpoints,&permsC,NULL);CHKERRQ(ierr);
  }
  if (mesh->printSetValues) {ierr = DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numFIndices, findices, numCIndices, cindices, values);CHKERRQ(ierr);}
  /* TODO: flips */
  ierr = MatSetValues(A, numFIndices, findices, numCIndices, cindices, values, mode);
  if (ierr) {
    PetscMPIInt    rank;
    PetscErrorCode ierr2;

    ierr2 = MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank);CHKERRQ(ierr2);
    ierr2 = (*PetscErrorPrintf)("[%d]ERROR in DMPlexMatSetClosure\n", rank);CHKERRQ(ierr2);
    ierr2 = DMPlexPrintMatSetValues(PETSC_VIEWER_STDERR_SELF, A, point, numFIndices, findices, numCIndices, cindices, values);CHKERRQ(ierr2);
    ierr2 = DMRestoreWorkArray(dmf, numFIndices, MPIU_INT, &findices);CHKERRQ(ierr2);
    ierr2 = DMRestoreWorkArray(dmc, numCIndices, MPIU_INT, &cindices);CHKERRQ(ierr2);
    CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dmf, numCPoints*2*4, MPIU_INT, &ftotpoints);CHKERRQ(ierr);
  ierr = DMPlexRestoreTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmf, numFIndices, MPIU_INT, &findices);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmc, numCIndices, MPIU_INT, &cindices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexMatGetClosureIndicesRefined(DM dmf, PetscSection fsection, PetscSection globalFSection, DM dmc, PetscSection csection, PetscSection globalCSection, PetscInt point, PetscInt cindices[], PetscInt findices[])
{
  PetscInt      *fpoints = NULL, *ftotpoints = NULL;
  PetscInt      *cpoints = NULL;
  PetscInt       foffsets[32], coffsets[32];
  const PetscInt *fclperm = NULL, *cclperm = NULL; /* Closure permutations cannot work here */
  CellRefiner    cellRefiner;
  PetscInt       numFields, numSubcells, maxFPoints, numFPoints, numCPoints, numFIndices, numCIndices, dof, off, globalOff, pStart, pEnd, p, q, r, s, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 4);
  if (!fsection) {ierr = DMGetLocalSection(dmf, &fsection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(fsection, PETSC_SECTION_CLASSID, 2);
  if (!csection) {ierr = DMGetLocalSection(dmc, &csection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(csection, PETSC_SECTION_CLASSID, 5);
  if (!globalFSection) {ierr = DMGetGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalFSection, PETSC_SECTION_CLASSID, 3);
  if (!globalCSection) {ierr = DMGetGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalCSection, PETSC_SECTION_CLASSID, 6);
  ierr = PetscSectionGetNumFields(fsection, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscArrayzero(foffsets, 32);CHKERRQ(ierr);
  ierr = PetscArrayzero(coffsets, 32);CHKERRQ(ierr);
  /* Column indices */
  ierr = DMPlexGetTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints);CHKERRQ(ierr);
  maxFPoints = numCPoints;
  /* Compress out points not in the section */
  /*   TODO: Squeeze out points with 0 dof as well */
  ierr = PetscSectionGetChart(csection, &pStart, &pEnd);CHKERRQ(ierr);
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

    ierr = PetscSectionGetDof(csection, cpoints[p], &dof);CHKERRQ(ierr);
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      ierr           = PetscSectionGetFieldDof(csection, cpoints[p], f, &fdof);CHKERRQ(ierr);
      coffsets[f+1] += fdof;
    }
    numCIndices += dof;
  }
  for (f = 1; f < numFields; ++f) coffsets[f+1] += coffsets[f];
  /* Row indices */
  ierr = DMPlexGetCellRefiner_Internal(dmc, &cellRefiner);CHKERRQ(ierr);
  ierr = CellRefinerGetAffineTransforms_Internal(cellRefiner, &numSubcells, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmf, maxFPoints*2*numSubcells, MPIU_INT, &ftotpoints);CHKERRQ(ierr);
  for (r = 0, q = 0; r < numSubcells; ++r) {
    /* TODO Map from coarse to fine cells */
    ierr = DMPlexGetTransitiveClosure(dmf, point*numSubcells + r, PETSC_TRUE, &numFPoints, &fpoints);CHKERRQ(ierr);
    /* Compress out points not in the section */
    ierr = PetscSectionGetChart(fsection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = 0; p < numFPoints*2; p += 2) {
      if ((fpoints[p] >= pStart) && (fpoints[p] < pEnd)) {
        ierr = PetscSectionGetDof(fsection, fpoints[p], &dof);CHKERRQ(ierr);
        if (!dof) continue;
        for (s = 0; s < q; ++s) if (fpoints[p] == ftotpoints[s*2]) break;
        if (s < q) continue;
        ftotpoints[q*2]   = fpoints[p];
        ftotpoints[q*2+1] = fpoints[p+1];
        ++q;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dmf, point, PETSC_TRUE, &numFPoints, &fpoints);CHKERRQ(ierr);
  }
  numFPoints = q;
  for (p = 0, numFIndices = 0; p < numFPoints*2; p += 2) {
    PetscInt fdof;

    ierr = PetscSectionGetDof(fsection, ftotpoints[p], &dof);CHKERRQ(ierr);
    if (!dof) continue;
    for (f = 0; f < numFields; ++f) {
      ierr           = PetscSectionGetFieldDof(fsection, ftotpoints[p], f, &fdof);CHKERRQ(ierr);
      foffsets[f+1] += fdof;
    }
    numFIndices += dof;
  }
  for (f = 1; f < numFields; ++f) foffsets[f+1] += foffsets[f];

  if (numFields && foffsets[numFields] != numFIndices) SETERRQ2(PetscObjectComm((PetscObject)dmf), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", foffsets[numFields], numFIndices);
  if (numFields && coffsets[numFields] != numCIndices) SETERRQ2(PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Invalid size for closure %D should be %D", coffsets[numFields], numCIndices);
  if (numFields) {
    const PetscInt **permsF[32] = {NULL};
    const PetscInt **permsC[32] = {NULL};

    for (f = 0; f < numFields; f++) {
      ierr = PetscSectionGetFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL);CHKERRQ(ierr);
    }
    for (p = 0; p < numFPoints; p++) {
      ierr = PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPointFields_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, foffsets, PETSC_FALSE, permsF, p, fclperm, findices);CHKERRQ(ierr);
    }
    for (p = 0; p < numCPoints; p++) {
      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPointFields_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, coffsets, PETSC_FALSE, permsC, p, cclperm, cindices);CHKERRQ(ierr);
    }
    for (f = 0; f < numFields; f++) {
      ierr = PetscSectionRestoreFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL);CHKERRQ(ierr);
      ierr = PetscSectionRestoreFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL);CHKERRQ(ierr);
    }
  } else {
    const PetscInt **permsF = NULL;
    const PetscInt **permsC = NULL;

    ierr = PetscSectionGetPointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetPointSyms(csection,numCPoints,cpoints,&permsC,NULL);CHKERRQ(ierr);
    for (p = 0, off = 0; p < numFPoints; p++) {
      const PetscInt *perm = permsF ? permsF[p] : NULL;

      ierr = PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPoint_Internal(fsection, PETSC_FALSE, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, fclperm, findices);CHKERRQ(ierr);
    }
    for (p = 0, off = 0; p < numCPoints; p++) {
      const PetscInt *perm = permsC ? permsC[p] : NULL;

      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPoint_Internal(csection, PETSC_FALSE, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, cclperm, cindices);CHKERRQ(ierr);
    }
    ierr = PetscSectionRestorePointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL);CHKERRQ(ierr);
    ierr = PetscSectionRestorePointSyms(csection,numCPoints,cpoints,&permsC,NULL);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dmf, numCPoints*2*4, MPIU_INT, &ftotpoints);CHKERRQ(ierr);
  ierr = DMPlexRestoreTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetHybridBounds - Get the first mesh point of each dimension which is a hybrid

  Input Parameter:
. dm - The DMPlex object

  Output Parameters:
+ cMax - The first hybrid cell
. fMax - The first hybrid face
. eMax - The first hybrid edge
- vMax - The first hybrid vertex

  Level: developer

.seealso DMPlexCreateHybridMesh(), DMPlexSetHybridBounds()
@*/
PetscErrorCode DMPlexGetHybridBounds(DM dm, PetscInt *cMax, PetscInt *fMax, PetscInt *eMax, PetscInt *vMax)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 0) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM dimension not yet set");
  if (cMax) *cMax = mesh->hybridPointMax[dim];
  if (fMax) *fMax = mesh->hybridPointMax[PetscMax(dim-1,0)];
  if (eMax) *eMax = mesh->hybridPointMax[1];
  if (vMax) *vMax = mesh->hybridPointMax[0];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateDimStratum(DM dm, DMLabel depthLabel, DMLabel dimLabel, PetscInt d, PetscInt dMax)
{
  IS             is, his;
  PetscInt       first = 0, stride;
  PetscBool      isStride;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetStratumIS(depthLabel, d, &is);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) is, ISSTRIDE, &isStride);CHKERRQ(ierr);
  if (isStride) {ierr = ISStrideGetInfo(is, &first, &stride);CHKERRQ(ierr);}
  if (is && (!isStride || stride != 1)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DM is not stratified: depth %D IS is not contiguous", d);
  ierr = ISCreateStride(PETSC_COMM_SELF, (dMax - first), first, 1, &his);CHKERRQ(ierr);
  ierr = DMLabelSetStratumIS(dimLabel, d, his);CHKERRQ(ierr);
  ierr = ISDestroy(&his);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetHybridBounds - Set the first mesh point of each dimension which is a hybrid

  Input Parameters:
+ dm   - The DMPlex object
. cMax - The first hybrid cell
. fMax - The first hybrid face
. eMax - The first hybrid edge
- vMax - The first hybrid vertex

  Level: developer

.seealso DMPlexCreateHybridMesh(), DMPlexGetHybridBounds()
@*/
PetscErrorCode DMPlexSetHybridBounds(DM dm, PetscInt cMax, PetscInt fMax, PetscInt eMax, PetscInt vMax)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 0) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM dimension not yet set");
  if (cMax >= 0) mesh->hybridPointMax[dim]               = cMax;
  if (fMax >= 0) mesh->hybridPointMax[PetscMax(dim-1,0)] = fMax;
  if (eMax >= 0) mesh->hybridPointMax[1]                 = eMax;
  if (vMax >= 0) mesh->hybridPointMax[0]                 = vMax;
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

.seealso DMPlexConstructGhostCells(), DMPlexSetGhostCellStratum(), DMPlexGetHybridBounds()
@*/
PetscErrorCode DMPlexGetGhostCellStratum(DM dm, PetscInt *gcStart, PetscInt *gcEnd)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 0) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM dimension not yet set");
  if (gcStart) {PetscValidIntPointer(gcStart, 2); *gcStart = mesh->ghostCellStart;}
  if (gcEnd)   {
    PetscValidIntPointer(gcEnd, 3);
    if (mesh->ghostCellStart >= 0) {ierr = DMPlexGetHeightStratum(dm, 0, NULL, gcEnd);CHKERRQ(ierr);}
    else                           {*gcEnd = -1;}
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetGhostCellStratum - Set the range of cells which are used to enforce FV boundary conditions

  Input Parameters:
+ dm      - The DMPlex object
. gcStart - The first ghost cell, or PETSC_DETERMINE
- gcEnd   - The upper bound on ghost cells, or PETSC_DETERMINE

  Level: advanced

  Note: This is not usually called directly by a user.

.seealso DMPlexConstructGhostCells(), DMPlexGetGhostCellStratum(), DMPlexSetHybridBounds()
@*/
PetscErrorCode DMPlexSetGhostCellStratum(DM dm, PetscInt gcStart, PetscInt gcEnd)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim < 0) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM dimension not yet set");
  mesh->ghostCellStart = gcStart;
  if (gcEnd >= 0) {
    PetscInt cEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
    if (gcEnd != cEnd) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Ghost cells must appear at the end of the cell range, but gcEnd %D is not equal to cEnd %D", gcEnd, cEnd);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetInteriorCellStratum - Get the range of cells which are neither hybrid nor ghost FV cells

  Input Parameter:
. dm - The DMPlex object

  Output Parameters:
+ cStartInterior - The first ghost cell
- cEndInterior   - The upper bound on ghost cells

  Level: developer

.seealso DMPlexConstructGhostCells(), DMPlexSetGhostCellStratum(), DMPlexGetHybridBounds()
@*/
PetscErrorCode DMPlexGetInteriorCellStratum(DM dm, PetscInt *cStartInterior, PetscInt *cEndInterior)
{
  PetscInt       gcEnd, cMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, cStartInterior, cEndInterior);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &gcEnd, NULL);CHKERRQ(ierr);
  *cEndInterior = gcEnd < 0 ? *cEndInterior : gcEnd;
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  *cEndInterior = cMax  < 0 ? *cEndInterior : cMax;
  PetscFunctionReturn(0);
}

/* We can easily have a form that takes an IS instead */
PetscErrorCode DMPlexCreateNumbering_Plex(DM dm, PetscInt pStart, PetscInt pEnd, PetscInt shift, PetscInt *globalSize, PetscSF sf, IS *numbering)
{
  PetscSection   section, globalSection;
  PetscInt      *numbers, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionSetDof(section, p, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionCreateGlobalSection(section, sf, PETSC_FALSE, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd - pStart, &numbers);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionGetOffset(globalSection, p, &numbers[p-pStart]);CHKERRQ(ierr);
    if (numbers[p-pStart] < 0) numbers[p-pStart] -= shift;
    else                       numbers[p-pStart] += shift;
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd - pStart, numbers, PETSC_OWN_POINTER, numbering);CHKERRQ(ierr);
  if (globalSize) {
    PetscLayout layout;
    ierr = PetscSectionGetPointLayout(PetscObjectComm((PetscObject) dm), globalSection, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutGetSize(layout, globalSize);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateCellNumbering_Internal(DM dm, PetscBool includeHybrid, IS *globalCellNumbers)
{
  PetscInt       cellHeight, cStart, cEnd, cMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  if (cMax >= 0 && !includeHybrid) cEnd = PetscMin(cEnd, cMax);
  ierr = DMPlexCreateNumbering_Plex(dm, cStart, cEnd, 0, NULL, dm->sf, globalCellNumbers);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalCellNumbers) {ierr = DMPlexCreateCellNumbering_Internal(dm, PETSC_FALSE, &mesh->globalCellNumbers);CHKERRQ(ierr);}
  *globalCellNumbers = mesh->globalCellNumbers;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateVertexNumbering_Internal(DM dm, PetscBool includeHybrid, IS *globalVertexNumbers)
{
  PetscInt       vStart, vEnd, vMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, NULL, NULL, NULL, &vMax);CHKERRQ(ierr);
  if (vMax >= 0 && !includeHybrid) vEnd = PetscMin(vEnd, vMax);
  ierr = DMPlexCreateNumbering_Plex(dm, vStart, vEnd, 0, NULL, dm->sf, globalVertexNumbers);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->globalVertexNumbers) {ierr = DMPlexCreateVertexNumbering_Internal(dm, PETSC_FALSE, &mesh->globalVertexNumbers);CHKERRQ(ierr);}
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* For unstratified meshes use dim instead of depth */
  if (depth < 0) {ierr = DMGetDimension(dm, &depth);CHKERRQ(ierr);}
  for (d = 0; d <= depth; ++d) {
    PetscInt end;

    depths[d] = depth-d;
    ierr = DMPlexGetDepthStratum(dm, depths[d], &starts[d], &end);CHKERRQ(ierr);
    if (!(starts[d]-end)) { starts[d] = depths[d] = -1; }
  }
  ierr = PetscSortIntWithArray(depth+1, starts, depths);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(depths, gdepths, depth+1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    if (starts[d] >= 0 && depths[d] != gdepths[d]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected depth %D, found %D",depths[d],gdepths[d]);
  }
  for (d = 0; d <= depth; ++d) {
    PetscInt pStart, pEnd, gsize;

    ierr = DMPlexGetDepthStratum(dm, gdepths[d], &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexCreateNumbering_Plex(dm, pStart, pEnd, shift, &gsize, dm->sf, &nums[d]);CHKERRQ(ierr);
    shift += gsize;
  }
  ierr = ISConcatenate(PetscObjectComm((PetscObject) dm), depth+1, nums, globalPointNumbers);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {ierr = ISDestroy(&nums[d]);CHKERRQ(ierr);}
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
  PetscInt       dim, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ranks, 2);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMClone(dm, &rdm);CHKERRQ(ierr);
  ierr = DMGetDimension(rdm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) rdm), dim, 1, PETSC_TRUE, "PETSc___rank_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "rank");CHKERRQ(ierr);
  ierr = DMSetField(rdm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(rdm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(rdm, ranks);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *ranks, "partition");CHKERRQ(ierr);
  ierr = VecGetArray(*ranks, &r);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *lr;

    ierr = DMPlexPointGlobalRef(rdm, c, r, &lr);CHKERRQ(ierr);
    if (lr) *lr = rank;
  }
  ierr = VecRestoreArray(*ranks, &r);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(label, 2);
  PetscValidPointer(val, 3);
  ierr = DMClone(dm, &rdm);CHKERRQ(ierr);
  ierr = DMGetDimension(rdm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) rdm), dim, 1, PETSC_TRUE, "PETSc___label_value_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "label_value");CHKERRQ(ierr);
  ierr = DMSetField(rdm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(rdm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(rdm, val);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *val, "label_value");CHKERRQ(ierr);
  ierr = VecGetArray(*val, &v);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *lv;
    PetscInt     cval;

    ierr = DMPlexPointGlobalRef(rdm, c, v, &lv);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label, c, &cval);CHKERRQ(ierr);
    *lv = cval;
  }
  ierr = VecRestoreArray(*val, &v);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetConeSection(dm, &coneSection);CHKERRQ(ierr);
  ierr = DMPlexGetSupportSection(dm, &supportSection);CHKERRQ(ierr);
  /* Check that point p is found in the support of its cone points, and vice versa */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      PetscBool dup = PETSC_FALSE;
      PetscInt  d;
      for (d = c-1; d >= 0; --d) {
        if (cone[c] == cone[d]) {dup = PETSC_TRUE; break;}
      }
      ierr = DMPlexGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {
        if (support[s] == p) break;
      }
      if ((s >= supportSize) || (dup && (support[s+1] != p))) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %D cone: ", p);CHKERRQ(ierr);
        for (s = 0; s < coneSize; ++s) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D, ", cone[s]);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %D support: ", cone[c]);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D, ", support[s]);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        if (dup) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not repeatedly found in support of repeated cone point %D", p, cone[c]);
        else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not found in support of cone point %D", p, cone[c]);
      }
    }
    ierr = DMPlexGetTreeParent(dm, p, &pp, NULL);CHKERRQ(ierr);
    if (p != pp) { storagecheck = PETSC_FALSE; continue; }
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        ierr = DMPlexGetTreeParent(dm, cone[c], &pp, NULL);CHKERRQ(ierr);
        if (cone[c] != pp) { c = 0; break; }
        if (cone[c] == p) break;
      }
      if (c >= coneSize) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %D support: ", p);CHKERRQ(ierr);
        for (c = 0; c < supportSize; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D, ", support[c]);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "p: %D cone: ", support[s]);CHKERRQ(ierr);
        for (c = 0; c < coneSize; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "%D, ", cone[c]);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D not found in cone of support point %D", p, support[s]);
      }
    }
  }
  if (storagecheck) {
    ierr = PetscSectionGetStorageSize(coneSection, &csize);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(supportSection, &ssize);CHKERRQ(ierr);
    if (csize != ssize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Total cone size %D != Total support size %D", csize, ssize);
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
  PetscInt       dim, numCorners, numHybridCorners, vStart, vEnd, cStart, cEnd, cMax, c;
  PetscBool      isSimplex = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  if (cStart < cEnd) {
    ierr = DMPlexGetConeSize(dm, cStart, &c);CHKERRQ(ierr);
    isSimplex = c == dim+1 ? PETSC_TRUE : PETSC_FALSE;
  }
  switch (dim) {
  case 1: numCorners = isSimplex ? 2 : 2; numHybridCorners = isSimplex ? 2 : 2; break;
  case 2: numCorners = isSimplex ? 3 : 4; numHybridCorners = isSimplex ? 4 : 4; break;
  case 3: numCorners = isSimplex ? 4 : 8; numHybridCorners = isSimplex ? 6 : 8; break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle meshes of dimension %D", dim);
  }
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  cMax = cMax >= 0 ? cMax : cEnd;
  for (c = cStart; c < cMax; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];
      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (coneSize != numCorners) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %D has  %D vertices != %D", c, coneSize, numCorners);
  }
  for (c = cMax; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];
      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (coneSize > numHybridCorners) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Hybrid cell %D has  %D vertices > %D", c, coneSize, numHybridCorners);
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
  PetscInt       pMax[4];
  PetscInt       dim, depth, vStart, vEnd, cStart, cEnd, c, h;
  PetscErrorCode ierr;
  DMPlexInterpolatedFlag interpEnum;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexIsInterpolated(dm, &interpEnum);CHKERRQ(ierr);
  if (interpEnum == DMPLEX_INTERPOLATED_NONE) PetscFunctionReturn(0);
  if (interpEnum == DMPLEX_INTERPOLATED_PARTIAL) {
    PetscMPIInt	rank;
    MPI_Comm	comm;

    ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mesh is only partially interpolated on rank %d, this is currently not supported", rank);
  }

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pMax[dim], &pMax[dim-1], &pMax[1], &pMax[0]);CHKERRQ(ierr);
  for (h = cellHeight; h < PetscMin(depth, dim); ++h) {
    ierr = DMPlexGetHeightStratum(dm, h, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone, *ornt, *faces;
      DMPolytopeType  ct;
      PetscInt        numFaces, faceSize, coneSize,f;
      PetscInt       *closure = NULL, closureSize, cl, numCorners = 0;

      if (pMax[dim-h] >= 0 && c >= pMax[dim-h]) continue;
      ierr = DMPlexGetCellType(dm, c, &ct);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt p = closure[cl];
        if ((p >= vStart) && (p < vEnd)) closure[numCorners++] = p;
      }
      ierr = DMPlexGetRawFaces_Internal(dm, ct, closure, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
      if (coneSize != numFaces) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %D has %D faces but should have %D", c, coneSize, numFaces);
      for (f = 0; f < numFaces; ++f) {
        PetscInt *fclosure = NULL, fclosureSize, cl, fnumCorners = 0, v;

        ierr = DMPlexGetTransitiveClosure_Internal(dm, cone[f], ornt[f], PETSC_TRUE, &fclosureSize, &fclosure);CHKERRQ(ierr);
        for (cl = 0; cl < fclosureSize*2; cl += 2) {
          const PetscInt p = fclosure[cl];
          if ((p >= vStart) && (p < vEnd)) fclosure[fnumCorners++] = p;
        }
        if (fnumCorners != faceSize) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D (%D) of cell %D has %D vertices but should have %D", cone[f], f, c, fnumCorners, faceSize);
        for (v = 0; v < fnumCorners; ++v) {
          if (fclosure[v] != faces[f*faceSize+v]) SETERRQ6(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D (%d) of cell %D vertex %D, %D != %D", cone[f], f, c, v, fclosure[v], faces[f*faceSize+v]);
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cone[f], PETSC_TRUE, &fclosureSize, &fclosure);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreFaces_Internal(dm, ct, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
  PetscReal      detJ, J[9], refVol = 1.0;
  PetscReal      vol;
  PetscInt       dim, depth, d, cStart, cEnd, c, cMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) refVol *= 2.0;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  cMax = cMax < 0 ? cEnd : cMax;
  for (c = cStart; c < cMax; ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted, |J| = %g", c, (double) detJ);
    ierr = PetscInfo2(dm, "Cell %D FEM Volume %g\n", c, (double) detJ*refVol);CHKERRQ(ierr);
    if (depth > 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      if (vol <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, vol = %g", c, (double) vol);
      ierr = PetscInfo2(dm, "Cell %D FVM Volume %g\n", c, (double) vol);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = DMPlexIsDistributed(dm, &distributed);CHKERRQ(ierr);
  if (!distributed) PetscFunctionReturn(0);
  ierr = DMPlexGetOverlap(dm, &overlap);CHKERRQ(ierr);
  if (overlap) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)dm), "Warning: DMPlexCheckPointSF() is currently not implemented for meshes with partition overlapping");
    PetscFunctionReturn(0);
  }
  if (!pointSF) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but does not have PointSF attached");
  ierr = PetscSFGetGraph(pointSF, &nroots, &nleaves, &locals, NULL);CHKERRQ(ierr);
  if (nroots < 0) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "This DMPlex is distributed but its PointSF has no graph set");
  ierr = PetscSFComputeDegreeBegin(pointSF, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(pointSF, &rootdegree);CHKERRQ(ierr);

  /* 1) check there are no faces in 2D, cells in 3D, in interface */
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  for (l = 0; l < nleaves; ++l) {
    const PetscInt point = locals[l];

    if (point >= cStart && point < cEnd) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D which is a cell", point);
  }

  /* 2) if some point is in interface, then all its cone points must be also in interface (either as leaves or roots) */
  for (l = 0; l < nleaves; ++l) {
    const PetscInt  point = locals[l];
    const PetscInt *cone;
    PetscInt        coneSize, c, idx;

    ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      if (!rootdegree[cone[c]]) {
        ierr = PetscFindInt(cone[c], nleaves, locals, &idx);CHKERRQ(ierr);
        if (idx < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point SF contains %D but not %D from its cone", point, cone[c]);
      }
    }
  }
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

.seealso: DMSetFromOptions()
@*/
PetscErrorCode DMPlexCheckCellShape(DM dm, PetscBool output, PetscReal condLimit)
{
  DM             dmCoarse;
  cell_stats_t   stats, globalStats;
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  PetscReal      *J, *invJ, min = 0, max = 0, mean = 0, stdev = 0;
  PetscReal      limit = condLimit > 0 ? condLimit : PETSC_MAX_REAL;
  PetscInt       cdim, cStart, cEnd, cMax, c, eStart, eEnd, count = 0;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  stats.min   = PETSC_MAX_REAL;
  stats.max   = PETSC_MIN_REAL;
  stats.sum   = stats.squaresum = 0.;
  stats.count = 0;

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm,&cdim);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscSqr(cdim), &J, PetscSqr(cdim), &invJ);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm,&cMax,NULL,NULL,NULL);CHKERRQ(ierr);
  cMax = cMax < 0 ? cEnd : cMax;
  for (c = cStart; c < cMax; c++) {
    PetscInt  i;
    PetscReal frobJ = 0., frobInvJ = 0., cond2, cond, detJ;

    ierr = DMPlexComputeCellGeometryAffineFEM(dm,c,NULL,J,invJ,&detJ);CHKERRQ(ierr);
    if (detJ < 0.0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted", c);
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

      ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dm, coordSection, coordsLocal, c, &Nv, &coords);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "[%d] Cell %D cond %g\n", rank, c, (double) cond);CHKERRQ(ierr);
      for (i = 0; i < Nv/cdim; ++i) {
        ierr = PetscSynchronizedPrintf(comm, "  Vertex %D: (", i);CHKERRQ(ierr);
        for (d = 0; d < cdim; ++d) {
          if (d > 0) {ierr = PetscSynchronizedPrintf(comm, ", ");CHKERRQ(ierr);}
          ierr = PetscSynchronizedPrintf(comm, "%g", (double) PetscRealPart(coords[i*cdim+d]));CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(comm, ")\n");CHKERRQ(ierr);
      }
      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < clSize*2; cl += 2) {
        const PetscInt edge = closure[cl];

        if ((edge >= eStart) && (edge < eEnd)) {
          PetscReal len;

          ierr = DMPlexComputeCellGeometryFVM(dm, edge, &len, NULL, NULL);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(comm, "  Edge %D: length %g\n", edge, (double) len);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordsLocal, c, &Nv, &coords);CHKERRQ(ierr);
    }
  }
  if (output) {ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);}

  if (size > 1) {
    PetscMPIInt   blockLengths[2] = {4,1};
    MPI_Aint      blockOffsets[2] = {offsetof(cell_stats_t,min),offsetof(cell_stats_t,count)};
    MPI_Datatype  blockTypes[2]   = {MPIU_REAL,MPIU_INT}, statType;
    MPI_Op        statReduce;

    ierr = MPI_Type_create_struct(2,blockLengths,blockOffsets,blockTypes,&statType);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&statType);CHKERRQ(ierr);
    ierr = MPI_Op_create(cell_stats_reduce, PETSC_TRUE, &statReduce);CHKERRQ(ierr);
    ierr = MPI_Reduce(&stats,&globalStats,1,statType,statReduce,0,comm);CHKERRQ(ierr);
    ierr = MPI_Op_free(&statReduce);CHKERRQ(ierr);
    ierr = MPI_Type_free(&statType);CHKERRQ(ierr);
  } else {
    ierr = PetscArraycpy(&globalStats,&stats,1);CHKERRQ(ierr);
  }
  if (!rank) {
    count = globalStats.count;
    min   = globalStats.min;
    max   = globalStats.max;
    mean  = globalStats.sum / globalStats.count;
    stdev = globalStats.count > 1 ? PetscSqrtReal(PetscMax((globalStats.squaresum - globalStats.count * mean * mean) / (globalStats.count - 1),0)) : 0.0;
  }

  if (output) {
    ierr = PetscPrintf(comm,"Mesh with %D cells, shape condition numbers: min = %g, max = %g, mean = %g, stddev = %g\n", count, (double) min, (double) max, (double) mean, (double) stdev);CHKERRQ(ierr);
  }
  ierr = PetscFree2(J,invJ);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dm,&dmCoarse);CHKERRQ(ierr);
  if (dmCoarse) {
    PetscBool isplex;

    ierr = PetscObjectTypeCompare((PetscObject)dmCoarse,DMPLEX,&isplex);CHKERRQ(ierr);
    if (isplex) {
      ierr = DMPlexCheckCellShape(dmCoarse,output,condLimit);CHKERRQ(ierr);
    }
  }
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
  PetscBool      regular, ismatis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalSection(dmFine, &gsf);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsf, &m);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmCoarse, &gsc);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsc, &n);CHKERRQ(ierr);

  ierr = PetscStrcmp(dmCoarse->mattype, MATIS, &ismatis);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject) dmCoarse), interpolation);CHKERRQ(ierr);
  ierr = MatSetSizes(*interpolation, m, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*interpolation, ismatis ? MATAIJ : dmCoarse->mattype);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dmFine, &ctx);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dmFine, &cdm);CHKERRQ(ierr);
  ierr = DMPlexGetRegularRefinement(dmFine, &regular);CHKERRQ(ierr);
  if (regular && cdm == dmCoarse) {ierr = DMPlexComputeInterpolatorNested(dmCoarse, dmFine, *interpolation, ctx);CHKERRQ(ierr);}
  else                            {ierr = DMPlexComputeInterpolatorGeneral(dmCoarse, dmFine, *interpolation, ctx);CHKERRQ(ierr);}
  ierr = MatViewFromOptions(*interpolation, NULL, "-interp_mat_view");CHKERRQ(ierr);
  if (scaling) {
    /* Use naive scaling */
    ierr = DMCreateInterpolationScale(dmCoarse, dmFine, *interpolation, scaling);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInjection_Plex(DM dmCoarse, DM dmFine, Mat *mat)
{
  PetscErrorCode ierr;
  VecScatter     ctx;

  PetscFunctionBegin;
  ierr = DMPlexComputeInjectorFEM(dmCoarse, dmFine, &ctx, NULL);CHKERRQ(ierr);
  ierr = MatCreateScatter(PetscObjectComm((PetscObject)ctx), ctx, mat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMassMatrix_Plex(DM dmCoarse, DM dmFine, Mat *mass)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  void          *ctx;
  DM             cdm;
  PetscBool      regular;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalSection(dmFine, &gsf);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsf, &m);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmCoarse, &gsc);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsc, &n);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject) dmCoarse), mass);CHKERRQ(ierr);
  ierr = MatSetSizes(*mass, m, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*mass, dmCoarse->mattype);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dmFine, &ctx);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dmFine, &cdm);CHKERRQ(ierr);
  ierr = DMPlexGetRegularRefinement(dmFine, &regular);CHKERRQ(ierr);
  if (regular && cdm == dmCoarse) {ierr = DMPlexComputeMassMatrixNested(dmCoarse, dmFine, *mass, ctx);CHKERRQ(ierr);}
  else                            {ierr = DMPlexComputeMassMatrixGeneral(dmCoarse, dmFine, *mass, ctx);CHKERRQ(ierr);}
  ierr = MatViewFromOptions(*mass, NULL, "-mass_mat_view");CHKERRQ(ierr);
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
  call DMPlexGetAnchors() directly: if there are anchors, then DMPlexGetAnchors() is called during DMGetConstraints().

  not collective

  Input Parameters:
. dm - The DMPlex object

  Output Parameters:
+ anchorSection - If not NULL, set to the section describing which points anchor the constrained points.
- anchorIS - If not NULL, set to the list of anchors indexed by anchorSection


  Level: intermediate

.seealso: DMPlexSetAnchors(), DMGetConstraints(), DMSetConstraints()
@*/
PetscErrorCode DMPlexGetAnchors(DM dm, PetscSection *anchorSection, IS *anchorIS)
{
  DM_Plex *plex = (DM_Plex *)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!plex->anchorSection && !plex->anchorIS && plex->createanchors) {ierr = (*plex->createanchors)(dm);CHKERRQ(ierr);}
  if (anchorSection) *anchorSection = plex->anchorSection;
  if (anchorIS) *anchorIS = plex->anchorIS;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetAnchors - Set the layout of the local anchor (point-to-point) constraints.  Unlike boundary conditions,
  when a point's degrees of freedom in a section are constrained to an outside value, the anchor constraints set a
  point's degrees of freedom to be a linear combination of other points' degrees of freedom.

  After specifying the layout of constraints with DMPlexSetAnchors(), one specifies the constraints by calling
  DMGetConstraints() and filling in the entries in the constraint matrix.

  collective on dm

  Input Parameters:
+ dm - The DMPlex object
. anchorSection - The section that describes the mapping from constrained points to the anchor points listed in anchorIS.  Must have a local communicator (PETSC_COMM_SELF or derivative).
- anchorIS - The list of all anchor points.  Must have a local communicator (PETSC_COMM_SELF or derivative).

  The reference counts of anchorSection and anchorIS are incremented.

  Level: intermediate

.seealso: DMPlexGetAnchors(), DMGetConstraints(), DMSetConstraints()
@*/
PetscErrorCode DMPlexSetAnchors(DM dm, PetscSection anchorSection, IS anchorIS)
{
  DM_Plex        *plex = (DM_Plex *)dm->data;
  PetscMPIInt    result;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (anchorSection) {
    PetscValidHeaderSpecific(anchorSection,PETSC_SECTION_CLASSID,2);
    ierr = MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)anchorSection),&result);CHKERRQ(ierr);
    if (result != MPI_CONGRUENT && result != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"anchor section must have local communicator");
  }
  if (anchorIS) {
    PetscValidHeaderSpecific(anchorIS,IS_CLASSID,3);
    ierr = MPI_Comm_compare(PETSC_COMM_SELF,PetscObjectComm((PetscObject)anchorIS),&result);CHKERRQ(ierr);
    if (result != MPI_CONGRUENT && result != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"anchor IS must have local communicator");
  }

  ierr = PetscObjectReference((PetscObject)anchorSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&plex->anchorSection);CHKERRQ(ierr);
  plex->anchorSection = anchorSection;

  ierr = PetscObjectReference((PetscObject)anchorIS);CHKERRQ(ierr);
  ierr = ISDestroy(&plex->anchorIS);CHKERRQ(ierr);
  plex->anchorIS = anchorIS;

#if defined(PETSC_USE_DEBUG)
  if (anchorIS && anchorSection) {
    PetscInt size, a, pStart, pEnd;
    const PetscInt *anchors;

    ierr = PetscSectionGetChart(anchorSection,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = ISGetLocalSize(anchorIS,&size);CHKERRQ(ierr);
    ierr = ISGetIndices(anchorIS,&anchors);CHKERRQ(ierr);
    for (a = 0; a < size; a++) {
      PetscInt p;

      p = anchors[a];
      if (p >= pStart && p < pEnd) {
        PetscInt dof;

        ierr = PetscSectionGetDof(anchorSection,p,&dof);CHKERRQ(ierr);
        if (dof) {
          PetscErrorCode ierr2;

          ierr2 = ISRestoreIndices(anchorIS,&anchors);CHKERRQ(ierr2);
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Point %D cannot be constrained and an anchor",p);
        }
      }
    }
    ierr = ISRestoreIndices(anchorIS,&anchors);CHKERRQ(ierr);
  }
#endif
  /* reset the generic constraints */
  ierr = DMSetDefaultConstraints(dm,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateConstraintSection_Anchors(DM dm, PetscSection section, PetscSection *cSec)
{
  PetscSection anchorSection;
  PetscInt pStart, pEnd, sStart, sEnd, p, dof, numFields, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetAnchors(dm,&anchorSection,NULL);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF,cSec);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section,&numFields);CHKERRQ(ierr);
  if (numFields) {
    PetscInt f;
    ierr = PetscSectionSetNumFields(*cSec,numFields);CHKERRQ(ierr);

    for (f = 0; f < numFields; f++) {
      PetscInt numComp;

      ierr = PetscSectionGetFieldComponents(section,f,&numComp);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldComponents(*cSec,f,numComp);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionGetChart(anchorSection,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section,&sStart,&sEnd);CHKERRQ(ierr);
  pStart = PetscMax(pStart,sStart);
  pEnd   = PetscMin(pEnd,sEnd);
  pEnd   = PetscMax(pStart,pEnd);
  ierr = PetscSectionSetChart(*cSec,pStart,pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    ierr = PetscSectionGetDof(anchorSection,p,&dof);CHKERRQ(ierr);
    if (dof) {
      ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(*cSec,p,dof);CHKERRQ(ierr);
      for (f = 0; f < numFields; f++) {
        ierr = PetscSectionGetFieldDof(section,p,f,&dof);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(*cSec,p,f,dof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(*cSec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateConstraintMatrix_Anchors(DM dm, PetscSection section, PetscSection cSec, Mat *cMat)
{
  PetscSection aSec;
  PetscInt pStart, pEnd, p, dof, aDof, aOff, off, nnz, annz, m, n, q, a, offset, *i, *j;
  const PetscInt *anchors;
  PetscInt numFields, f;
  IS aIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionGetStorageSize(cSec, &m);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,cMat);CHKERRQ(ierr);
  ierr = MatSetSizes(*cMat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*cMat,MATSEQAIJ);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(dm,&aSec,&aIS);CHKERRQ(ierr);
  ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
  /* cSec will be a subset of aSec and section */
  ierr = PetscSectionGetChart(cSec,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(m+1,&i);CHKERRQ(ierr);
  i[0] = 0;
  ierr = PetscSectionGetNumFields(section,&numFields);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt rDof, rOff, r;

    ierr = PetscSectionGetDof(aSec,p,&rDof);CHKERRQ(ierr);
    if (!rDof) continue;
    ierr = PetscSectionGetOffset(aSec,p,&rOff);CHKERRQ(ierr);
    if (numFields) {
      for (f = 0; f < numFields; f++) {
        annz = 0;
        for (r = 0; r < rDof; r++) {
          a = anchors[rOff + r];
          ierr = PetscSectionGetFieldDof(section,a,f,&aDof);CHKERRQ(ierr);
          annz += aDof;
        }
        ierr = PetscSectionGetFieldDof(cSec,p,f,&dof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(cSec,p,f,&off);CHKERRQ(ierr);
        for (q = 0; q < dof; q++) {
          i[off + q + 1] = i[off + q] + annz;
        }
      }
    }
    else {
      annz = 0;
      for (q = 0; q < dof; q++) {
        a = anchors[off + q];
        ierr = PetscSectionGetDof(section,a,&aDof);CHKERRQ(ierr);
        annz += aDof;
      }
      ierr = PetscSectionGetDof(cSec,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(cSec,p,&off);CHKERRQ(ierr);
      for (q = 0; q < dof; q++) {
        i[off + q + 1] = i[off + q] + annz;
      }
    }
  }
  nnz = i[m];
  ierr = PetscMalloc1(nnz,&j);CHKERRQ(ierr);
  offset = 0;
  for (p = pStart; p < pEnd; p++) {
    if (numFields) {
      for (f = 0; f < numFields; f++) {
        ierr = PetscSectionGetFieldDof(cSec,p,f,&dof);CHKERRQ(ierr);
        for (q = 0; q < dof; q++) {
          PetscInt rDof, rOff, r;
          ierr = PetscSectionGetDof(aSec,p,&rDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(aSec,p,&rOff);CHKERRQ(ierr);
          for (r = 0; r < rDof; r++) {
            PetscInt s;

            a = anchors[rOff + r];
            ierr = PetscSectionGetFieldDof(section,a,f,&aDof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(section,a,f,&aOff);CHKERRQ(ierr);
            for (s = 0; s < aDof; s++) {
              j[offset++] = aOff + s;
            }
          }
        }
      }
    }
    else {
      ierr = PetscSectionGetDof(cSec,p,&dof);CHKERRQ(ierr);
      for (q = 0; q < dof; q++) {
        PetscInt rDof, rOff, r;
        ierr = PetscSectionGetDof(aSec,p,&rDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(aSec,p,&rOff);CHKERRQ(ierr);
        for (r = 0; r < rDof; r++) {
          PetscInt s;

          a = anchors[rOff + r];
          ierr = PetscSectionGetDof(section,a,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(section,a,&aOff);CHKERRQ(ierr);
          for (s = 0; s < aDof; s++) {
            j[offset++] = aOff + s;
          }
        }
      }
    }
  }
  ierr = MatSeqAIJSetPreallocationCSR(*cMat,i,j,NULL);CHKERRQ(ierr);
  ierr = PetscFree(i);CHKERRQ(ierr);
  ierr = PetscFree(j);CHKERRQ(ierr);
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateDefaultConstraints_Plex(DM dm)
{
  DM_Plex        *plex = (DM_Plex *)dm->data;
  PetscSection   anchorSection, section, cSec;
  Mat            cMat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetAnchors(dm,&anchorSection,NULL);CHKERRQ(ierr);
  if (anchorSection) {
    PetscInt Nf;

    ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
    ierr = DMPlexCreateConstraintSection_Anchors(dm,section,&cSec);CHKERRQ(ierr);
    ierr = DMPlexCreateConstraintMatrix_Anchors(dm,section,cSec,&cMat);CHKERRQ(ierr);
    ierr = DMGetNumFields(dm,&Nf);CHKERRQ(ierr);
    if (Nf && plex->computeanchormatrix) {ierr = (*plex->computeanchormatrix)(dm,section,cSec,cMat);CHKERRQ(ierr);}
    ierr = DMSetDefaultConstraints(dm,cSec,cMat);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&cSec);CHKERRQ(ierr);
    ierr = MatDestroy(&cMat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDomainDM_Plex(DM dm, DMLabel label, PetscInt value, IS *is, DM *subdm)
{
  IS             subis;
  PetscSection   section, subsection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  if (!section) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Must set default section for DM before splitting subdomain");
  if (!subdm)   SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Must set output subDM for splitting subdomain");
  /* Create subdomain */
  ierr = DMPlexFilter(dm, label, value, subdm);CHKERRQ(ierr);
  /* Create submodel */
  ierr = DMPlexCreateSubpointIS(*subdm, &subis);CHKERRQ(ierr);
  ierr = PetscSectionCreateSubmeshSection(section, subis, &subsection);CHKERRQ(ierr);
  ierr = ISDestroy(&subis);CHKERRQ(ierr);
  ierr = DMSetLocalSection(*subdm, subsection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&subsection);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, *subdm);CHKERRQ(ierr);
  /* Create map from submodel to global model */
  if (is) {
    PetscSection    sectionGlobal, subsectionGlobal;
    IS              spIS;
    const PetscInt *spmap;
    PetscInt       *subIndices;
    PetscInt        subSize = 0, subOff = 0, pStart, pEnd, p;
    PetscInt        Nf, f, bs = -1, bsLocal[2], bsMinMax[2];

    ierr = DMPlexCreateSubpointIS(*subdm, &spIS);CHKERRQ(ierr);
    ierr = ISGetIndices(spIS, &spmap);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(*subdm, &subsectionGlobal);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(subsection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, pSubSize  = 0;

      ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        for (f = 0; f < Nf; ++f) {
          PetscInt fdof, fcdof;

          ierr     = PetscSectionGetFieldDof(subsection, p, f, &fdof);CHKERRQ(ierr);
          ierr     = PetscSectionGetFieldConstraintDof(subsection, p, f, &fcdof);CHKERRQ(ierr);
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
    ierr = PetscGlobalMinMaxInt(PetscObjectComm((PetscObject) dm), bsLocal, bsMinMax);CHKERRQ(ierr);
    if (bsMinMax[0] != bsMinMax[1]) {bs = 1;}
    else                            {bs = bsMinMax[0];}
    ierr = PetscMalloc1(subSize, &subIndices);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt gdof, goff;

      ierr = PetscSectionGetDof(subsectionGlobal, p, &gdof);CHKERRQ(ierr);
      if (gdof > 0) {
        const PetscInt point = spmap[p];

        ierr = PetscSectionGetOffset(sectionGlobal, point, &goff);CHKERRQ(ierr);
        for (f = 0; f < Nf; ++f) {
          PetscInt fdof, fcdof, fc, f2, poff = 0;

          /* Can get rid of this loop by storing field information in the global section */
          for (f2 = 0; f2 < f; ++f2) {
            ierr  = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
            ierr  = PetscSectionGetFieldConstraintDof(section, p, f2, &fcdof);CHKERRQ(ierr);
            poff += fdof-fcdof;
          }
          ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          for (fc = 0; fc < fdof-fcdof; ++fc, ++subOff) {
            subIndices[subOff] = goff+poff+fc;
          }
        }
      }
    }
    ierr = ISRestoreIndices(spIS, &spmap);CHKERRQ(ierr);
    ierr = ISDestroy(&spIS);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), subSize, subIndices, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
    if (bs > 1) {
      /* We need to check that the block size does not come from non-contiguous fields */
      PetscInt i, j, set = 1;
      for (i = 0; i < subSize; i += bs) {
        for (j = 0; j < bs; ++j) {
          if (subIndices[i+j] != subIndices[i]+j) {set = 0; break;}
        }
      }
      if (set) {ierr = ISSetBlockSize(*is, bs);CHKERRQ(ierr);}
    }
    /* Attach nullspace */
    for (f = 0; f < Nf; ++f) {
      (*subdm)->nullspaceConstructors[f] = dm->nullspaceConstructors[f];
      if ((*subdm)->nullspaceConstructors[f]) break;
    }
    if (f < Nf) {
      MatNullSpace nullSpace;

      ierr = (*(*subdm)->nullspaceConstructors[f])(*subdm, f, &nullSpace);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) *is, "nullspace", (PetscObject) nullSpace);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
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
  PetscStageLog      stageLog;
  PetscLogEvent      event;
  PetscLogStage      stage;
  PetscEventPerfInfo eventInfo;
  PetscReal          cellRate, flopRate;
  PetscInt           cStart, cEnd, Nf, N;
  const char        *name;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
#if defined(PETSC_USE_LOG)
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  ierr = PetscLogEventGetId("DMPlexResidualFE", &event);CHKERRQ(ierr);
  ierr = PetscLogEventGetPerfInfo(stage, event, &eventInfo);CHKERRQ(ierr);
  N        = (cEnd - cStart)*Nf*eventInfo.count;
  flopRate = eventInfo.flops/eventInfo.time;
  cellRate = N/eventInfo.time;
  ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "DM (%s) FE Residual Integration: %D integrals %D reps\n  Cell rate: %.2g/s flop rate: %.2g MF/s\n", name ? name : "unknown", N, eventInfo.count, (double) cellRate, (double) (flopRate/1.e6));CHKERRQ(ierr);
#else
  SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Plex Throughput Monitor is not supported if logging is turned off. Reconfigure using --with-log.");
#endif
  PetscFunctionReturn(0);
}
