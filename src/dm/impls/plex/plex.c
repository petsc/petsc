#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/glvisvecimpl.h>
#include <petscsf.h>
#include <petscds.h>
#include <petscdraw.h>

/* Logging support */
PetscLogEvent DMPLEX_Interpolate, PETSCPARTITIONER_Partition, DMPLEX_Distribute, DMPLEX_DistributeCones, DMPLEX_DistributeLabels, DMPLEX_DistributeSF, DMPLEX_DistributeOverlap, DMPLEX_DistributeField, DMPLEX_DistributeData, DMPLEX_Migrate, DMPLEX_InterpolateSF, DMPLEX_GlobalToNaturalBegin, DMPLEX_GlobalToNaturalEnd, DMPLEX_NaturalToGlobalBegin, DMPLEX_NaturalToGlobalEnd, DMPLEX_Stratify, DMPLEX_Preallocate, DMPLEX_ResidualFEM, DMPLEX_JacobianFEM, DMPLEX_InterpolatorFEM, DMPLEX_InjectorFEM, DMPLEX_IntegralFEM, DMPLEX_CreateGmsh;

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
  PetscInt         dim, cMax, fMax, cStart, cEnd, coneSize;
  CellRefiner      cellRefiner;
  PetscBool        lop, allnoop, localized;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm,&cMax,&fMax,NULL,NULL);CHKERRQ(ierr);
  if (cMax >= 0 || fMax >= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle hybrid meshes yet");
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  if (!(cEnd - cStart)) cellRefiner = REFINER_NOOP;
  else {
    ierr = DMPlexGetConeSize(dm,cStart,&coneSize);CHKERRQ(ierr);
    switch (dim) {
    case 1:
      cellRefiner = REFINER_NOOP;
    break;
    case 2:
      switch (coneSize) {
      case 3:
        cellRefiner = REFINER_SIMPLEX_TO_HEX_2D;
      break;
      case 4:
        cellRefiner = REFINER_NOOP;
      break;
      default: SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle coneSize %D with dimension %D",coneSize,dim);
      }
    break;
    case 3:
      switch (coneSize) {
      case 4:
        cellRefiner = REFINER_SIMPLEX_TO_HEX_3D;
      break;
      case 6:
        cellRefiner = REFINER_NOOP;
      break;
      default: SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle coneSize %D with dimension %D",coneSize,dim);
      }
    break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle dimension %D",dim);
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
  PetscInt       dim, pStart, pEnd, vStart, vEnd, cStart, cEnd, cEndInterior, vdof = 0, cdof = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ft  = PETSC_VTK_POINT_FIELD;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  if (field >= 0) {
    if ((vStart >= pStart) && (vStart < pEnd)) {ierr = PetscSectionGetFieldDof(section, vStart, field, &vdof);CHKERRQ(ierr);}
    if ((cStart >= pStart) && (cStart < pEnd)) {ierr = PetscSectionGetFieldDof(section, cStart, field, &cdof);CHKERRQ(ierr);}
  } else {
    if ((vStart >= pStart) && (vStart < pEnd)) {ierr = PetscSectionGetDof(section, vStart, &vdof);CHKERRQ(ierr);}
    if ((cStart >= pStart) && (cStart < pEnd)) {ierr = PetscSectionGetDof(section, cStart, &cdof);CHKERRQ(ierr);}
  }
  if (vdof) {
    *sStart = vStart;
    *sEnd   = vEnd;
    if (vdof == dim) *ft = PETSC_VTK_POINT_VECTOR_FIELD;
    else             *ft = PETSC_VTK_POINT_FIELD;
  } else if (cdof) {
    *sStart = cStart;
    *sEnd   = cEnd;
    if (cdof == dim) *ft = PETSC_VTK_CELL_VECTOR_FIELD;
    else             *ft = PETSC_VTK_CELL_FIELD;
  } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Could not classify input Vec for VTK");
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
  ierr = DMGetDefaultSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
  ierr = DMGetCoarsenLevel(dm, &level);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, &coordSection);CHKERRQ(ierr);
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

    if (v->hdr.prefix) {ierr = PetscStrcpy(prefix, v->hdr.prefix);CHKERRQ(ierr);}
    else               {prefix[0] = '\0';}
    if (Nf > 1) {
      ierr = DMCreateSubDM(dm, 1, &f, &fis, &fdm);CHKERRQ(ierr);
      ierr = VecGetSubVector(v, fis, &fv);CHKERRQ(ierr);
      ierr = PetscStrcat(prefix, fname);CHKERRQ(ierr);
      ierr = PetscStrcat(prefix, "_");CHKERRQ(ierr);
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

PetscErrorCode VecView_Plex_Local(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isseq, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  if (isvtk || ishdf5 || isglvis) {
    PetscInt  numFields;
    PetscBool fem = PETSC_FALSE;

    ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
    if (numFields) {
      PetscObject fe;

      ierr = DMGetField(dm, 0, &fe);CHKERRQ(ierr);
      if (fe->classid == PETSCFE_CLASSID) fem = PETSC_TRUE;
    }
    if (fem) {ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, v, 0.0, NULL, NULL, NULL);CHKERRQ(ierr);}
  }
  if (isvtk) {
    PetscSection            section;
    PetscViewerVTKFieldType ft;
    PetscInt                pStart, pEnd;

    ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
    ierr = DMPlexGetFieldType_Internal(dm, section, PETSC_DETERMINE, &pStart, &pEnd, &ft);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) v);CHKERRQ(ierr);  /* viewer drops reference */
    ierr = PetscViewerVTKAddField(viewer, (PetscObject) dm, DMPlexVTKWriteAll, ft, (PetscObject) v);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = VecView_Plex_Local_HDF5_Internal(v, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isglvis) {
    ierr = VecView_GLVis(v, viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = VecView_Plex_Local_Draw(v, viewer);CHKERRQ(ierr);
  } else {
    if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
    else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscReal      time = 0.0;
  PetscBool      isvtk, ishdf5, isdraw, isseq, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  if (isvtk || isglvis) {
    PetscInt    num;
    Vec         locv;
    const char *name;

    ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = DMGetOutputSequenceNumber(dm, &num, &time);CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscViewerGLVisSetSnapId(viewer, num);CHKERRQ(ierr);
    ierr = VecView_Plex_Local(locv, viewer);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = VecView_Plex_HDF5_Internal(v, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isdraw) {
    Vec         locv;
    const char *name;

    ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
    ierr = DMGetOutputSequenceNumber(dm, NULL, &time);CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = VecView_Plex_Local(locv, viewer);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  } else {
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
    const char *vecname;
    PetscInt    n, nroots;

    if (dm->sfNatural) {
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
  } else {
    /* we are viewing a natural DMPlex vec. */
    v = originalv;
  }
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
  if (format == PETSC_VIEWER_NATIVE) {ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);}
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
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexView_Ascii_Geometry(DM dm, PetscViewer viewer)
{
  PetscSection       coordSection;
  Vec                coordinates;
  DMLabel            depthLabel;
  const char        *name[4];
  const PetscScalar *a;
  PetscInt           dim, pStart, pEnd, cStart, cEnd, c;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(coordSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &a);CHKERRQ(ierr);
  name[0]     = "vertex";
  name[1]     = "edge";
  name[dim-1] = "face";
  name[dim]   = "cell";
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    ierr = PetscViewerASCIIPrintf(viewer, "Geometry for cell %D:\n", c);CHKERRQ(ierr);
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
          ierr = PetscViewerASCIIPrintf(viewer, "%g", PetscRealPart(a[off+p*dim+d]));CHKERRQ(ierr);
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
  ierr = DMGetDefaultSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    const char *name;
    PetscInt    maxConeSize, maxSupportSize;
    PetscInt    pStart, pEnd, p;
    PetscMPIInt rank, size;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh '%s':\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "orientation is missing\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "cap --> base:\n", name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Max sizes cone: %D support: %D\n", rank,maxConeSize, maxSupportSize);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off, s;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      for (s = off; s < off+dof; ++s) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]: %D ----> %D\n", rank, p, mesh->supports[s]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "base <-- cap:\n", name);CHKERRQ(ierr);
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
    ierr = PetscSectionGetChart(coordSection, &pStart, NULL);CHKERRQ(ierr);
    if (pStart >= 0) {ierr = PetscSectionVecView(coordSection, coordinates, viewer);CHKERRQ(ierr);}
    ierr = DMGetLabel(dm, "marker", &markers);CHKERRQ(ierr);
    ierr = DMLabelView(markers,viewer);CHKERRQ(ierr);
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
    PetscReal    scale         = 2.0;
    PetscBool    useNumbers    = PETSC_TRUE, useLabels, useColors;
    double       tcoords[3];
    PetscScalar *coords;
    PetscInt     numLabels, l, numColors, numLColors, dim, depth, cStart, cEnd, c, vStart, vEnd, v, eStart = 0, eEnd = 0, e, p;
    PetscMPIInt  rank, size;
    char         **names, **colors, **lcolors;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
    numLabels  = PetscMax(numLabels, 10);
    numColors  = 10;
    numLColors = 10;
    ierr = PetscCalloc3(numLabels, &names, numColors, &colors, numLColors, &lcolors);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_view_scale", &scale, NULL);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer, "\\begin{tikzpicture}[scale = %g,font=\\fontsize{8}{8}\\selectfont]\n", 1.0);CHKERRQ(ierr);
    /* Plot vertices */
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt  off, dof, d;
      PetscBool isLabeled = PETSC_FALSE;

      ierr = PetscSectionGetDof(coordSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "\\path (");CHKERRQ(ierr);
      if (PetscUnlikely(dof > 3)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"coordSection vertex %D has dof %D > 3",v,dof);
      for (d = 0; d < dof; ++d) {
        tcoords[d] = (double) (scale*PetscRealPart(coords[off+d]));
        tcoords[d] = PetscAbsReal(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
      }
      /* Rotate coordinates since PGF makes z point out of the page instead of up */
      if (dim == 3) {PetscReal tmp = tcoords[1]; tcoords[1] = tcoords[2]; tcoords[2] = -tmp;}
      for (d = 0; d < dof; ++d) {
        if (d > 0) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, ",");CHKERRQ(ierr);}
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%g", tcoords[d]);CHKERRQ(ierr);
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
    /* Plot edges */
    if (depth > 1) {ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);}
    if (dim < 3 && useNumbers) {
      ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "\\path\n");CHKERRQ(ierr);
      for (e = eStart; e < eEnd; ++e) {
        const PetscInt *cone;
        PetscInt        coneSize, offA, offB, dof, d;

        ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
        if (coneSize != 2) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Edge %D cone should have two vertices, not %D", e, coneSize);
        ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(coordSection, cone[0], &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cone[0], &offA);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(coordSection, cone[1], &offB);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "(");CHKERRQ(ierr);
        for (d = 0; d < dof; ++d) {
          tcoords[d] = (double) (scale*PetscRealPart(coords[offA+d]+coords[offB+d]));
          tcoords[d] = PetscAbsReal(tcoords[d]) < 1e-10 ? 0.0 : tcoords[d];
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
    /* Plot cells */
    if (dim == 3 || !useNumbers) {
      for (e = eStart; e < eEnd; ++e) {
        const PetscInt *cone;

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
      ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
      for (c = cStart; c < cEnd; ++c) {
        PetscInt *closure = NULL;
        PetscInt  closureSize, firstPoint = -1;

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
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{tikzpicture}\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\\end{document}\n", name);CHKERRQ(ierr);
    for (l = 0; l < numLabels;  ++l) {ierr = PetscFree(names[l]);CHKERRQ(ierr);}
    for (c = 0; c < numColors;  ++c) {ierr = PetscFree(colors[c]);CHKERRQ(ierr);}
    for (c = 0; c < numLColors; ++c) {ierr = PetscFree(lcolors[c]);CHKERRQ(ierr);}
    ierr = PetscFree3(names, colors, lcolors);CHKERRQ(ierr);
  } else {
    MPI_Comm    comm;
    PetscInt   *sizes, *hybsizes;
    PetscInt    locDepth, depth, dim, d, pMax[4];
    PetscInt    pStart, pEnd, p;
    PetscInt    numLabels, l;
    const char *name;
    PetscMPIInt size;

    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    if (name) {ierr = PetscViewerASCIIPrintf(viewer, "%s in %D dimensions:\n", name, dim);CHKERRQ(ierr);}
    else      {ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %D dimensions:\n", dim);CHKERRQ(ierr);}
    ierr = DMPlexGetDepth(dm, &locDepth);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&locDepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(dm, &pMax[depth], depth > 0 ? &pMax[depth-1] : NULL, &pMax[1], &pMax[0]);CHKERRQ(ierr);
    ierr = PetscMalloc2(size,&sizes,size,&hybsizes);CHKERRQ(ierr);
    if (depth == 1) {
      ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d-cells:", 0);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
      pEnd = pEnd - pStart;
      ierr = MPI_Gather(&pEnd, 1, MPIU_INT, sizes, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", dim);CHKERRQ(ierr);
      for (p = 0; p < size; ++p) {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
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
        ierr = PetscViewerASCIIPrintf(viewer, "  %D-cells:", d);CHKERRQ(ierr);
        for (p = 0; p < size; ++p) {
          if (!rank) {
            if (hybsizes[p] >= 0) {ierr = PetscViewerASCIIPrintf(viewer, " %D (%D)", sizes[p], sizes[p] - hybsizes[p]);CHKERRQ(ierr);}
            else                  {ierr = PetscViewerASCIIPrintf(viewer, " %D", sizes[p]);CHKERRQ(ierr);}
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(sizes,hybsizes);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D", dim);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, &coordSection);CHKERRQ(ierr);
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

  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *coords = NULL;
    PetscInt     numCoords,coneSize;

    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
    switch (coneSize) {
    case 3:
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      break;
    case 4:
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells with %D facets", coneSize);
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords);CHKERRQ(ierr);
  }
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, ishdf5, isvtk, isdraw, flg, isglvis;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
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
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
    ierr = DMPlexView_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else if (isvtk) {
    ierr = DMPlexVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = DMPlexView_Draw(dm, viewer);CHKERRQ(ierr);
  } else if (isglvis) {
    ierr = DMPlexView_GLVis(dm, viewer);CHKERRQ(ierr);
  }
  /* Optionally view the partition */
  ierr = PetscOptionsHasName(((PetscObject) dm)->options, ((PetscObject) dm)->prefix, "-dm_partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    Vec ranks;
    ierr = DMPlexCreateRankField(dm, &ranks);CHKERRQ(ierr);
    ierr = VecView(ranks, viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&ranks);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMLoad_Plex(DM dm, PetscViewer viewer)
{
  PetscBool      isbinary, ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,   &ishdf5);CHKERRQ(ierr);
  if (isbinary) {SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Do not yet support binary viewers");}
  else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = DMPlexLoad_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
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
  ierr = DMGetDefaultGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
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
    PetscInt    *dnz, *onz, *dnzu, *onzu, bsLocal, bsMax, bsMin, *ltogidx, lsize;
    PetscInt     pStart, pEnd, p, dof, cdof;

    /* Set localtoglobalmapping on the matrix for MatSetValuesLocal() to work (it also creates the local matrices in case of MATIS) */
    if (isMatIS) { /* need a different l2g map than the one computed by DMGetLocalToGlobalMapping */
      PetscSection section;
      PetscInt     size;

      ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
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
    bsLocal = bs;
    ierr = MPIU_Allreduce(&bsLocal, &bsMax, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    bsLocal = bs < 0 ? bsMax : bs;
    ierr = MPIU_Allreduce(&bsLocal, &bsMin, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    if (bsMin != bsMax) {bs = 1;}
    else                {bs = bsMax;}
    bs   = bs < 0 ? 1 : bs;
    if (isMatIS) {
      PetscInt l;
      /* Must reduce indices by blocksize */
      if (bs > 1) for (l = 0; l < lsize; ++l) ltogidx[l] /= bs;
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
    ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
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
  DMPlexGetConeSize - Return the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

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
  DMPlexSetConeSize - Set the number of in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexAddConeSize - Add the given number of in-edges to this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexGetCone - Return the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. cone - An array of points which are on the in-edges for point p

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  You must also call DMPlexRestoreCone() after you finish using the returned array.

.seealso: DMPlexCreate(), DMPlexSetCone(), DMPlexSetChart()
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

/*@
  DMPlexSetCone - Set the points on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- cone - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetConeSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetCone(), DMPlexSetChart(), DMPlexSetConeSize(), DMSetUp()
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
  DMPlexGetConeOrientation - Return the orientations on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

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
  DMPlexSetConeOrientation - Set the orientations on the in-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexInsertCone - Insert a point into the in-edges for the point p in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexInsertConeOrientation - Insert a point orientation for the in-edge for the point p in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexGetSupportSize - Return the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

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
  DMPlexSetSupportSize - Set the number of out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexGetSupport - Return the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
- p - The Sieve point, which must lie in the chart set with DMPlexSetChart()

  Output Parameter:
. support - An array of points which are on the out-edges for point p

  Level: beginner

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

  You must also call DMPlexRestoreSupport() after you finish using the returned array.

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
  DMPlexSetSupport - Set the points on the out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
- support - An array of points which are on the in-edges for point p

  Output Parameter:

  Note:
  This should be called after all calls to DMPlexSetSupportSize() and DMSetUp().

  Level: beginner

.seealso: DMPlexCreate(), DMPlexGetSupport(), DMPlexSetChart(), DMPlexSetSupportSize(), DMSetUp()
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
  DMPlexInsertSupport - Insert a point into the out-edges for the point p in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  DMPlexGetTransitiveClosure - Return the points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
      ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
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
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  if (*points) {
    closure = *points;
  } else {
    ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
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
  ierr = DMRestoreWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetTransitiveClosure_Internal - Return the points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG with a specified initial orientation

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
      ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
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
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  if (*points) {
    closure = *points;
  } else {
    ierr = DMGetWorkArray(dm, maxSize, PETSC_INT, &closure);CHKERRQ(ierr);
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
  ierr = DMRestoreWorkArray(dm, maxSize, PETSC_INT, &fifo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreTransitiveClosure - Restore the array of points on the transitive closure of the in-edges or out-edges for this point in the Sieve DAG

  Not collective

  Input Parameters:
+ mesh - The DMPlex
. p - The Sieve point, which must lie in the chart set with DMPlexSetChart()
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
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, points);CHKERRQ(ierr);
  if (numPoints) *numPoints = 0;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMaxSizes - Return the maximum number of in-edges (cone) and out-edges (support) for any point in the Sieve DAG

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

PetscErrorCode DMCreateSubDM_Plex(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {ierr = DMClone(dm, subdm);CHKERRQ(ierr);}
  ierr = DMCreateSubDM_Section_Private(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSymmetrize - Creates support (out-edge) information from cone (in-edge) inoformation

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
  PetscFunctionReturn(0);
}

/*@
  DMPlexStratify - The Sieve DAG for most topologies is a graded poset (http://en.wikipedia.org/wiki/Graded_poset), and
  can be illustrated by Hasse Diagram (a http://en.wikipedia.org/wiki/Hasse_diagram). The strata group all points of the
  same grade, and this function calculates the strata. This grade can be seen as the height (or depth) of the point in
  the DAG.

  Collective on dm

  Input Parameter:
. mesh - The DMPlex

  Output Parameter:

  Notes:
  Concretely, DMPlexStratify() creates a new label named "depth" containing the dimension of each element: 0 for vertices,
  1 for edges, and so on.  The depth label can be accessed through DMPlexGetDepthLabel() or DMPlexGetDepthStratum(), or
  manually via DMGetLabel().  The height is defined implicitly by height = maxDimension - depth, and can be accessed
  via DMPlexGetHeightStratum().  For example, cells have height 0 and faces have height 1.

  DMPlexStratify() should be called after all calls to DMPlexSymmetrize()

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSymmetrize()
@*/
PetscErrorCode DMPlexStratify(DM dm)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  DMLabel        label;
  PetscInt       pStart, pEnd, p;
  PetscInt       numRoots = 0, numLeaves = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscLogEventBegin(DMPLEX_Stratify,dm,0,0,0);CHKERRQ(ierr);
  /* Calculate depth */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "depth");CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  /* Initialize roots and count leaves */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt coneSize, supportSize;

    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    if (!coneSize && supportSize) {
      ++numRoots;
      ierr = DMLabelSetValue(label, p, 0);CHKERRQ(ierr);
    } else if (!supportSize && coneSize) {
      ++numLeaves;
    } else if (!supportSize && !coneSize) {
      /* Isolated points */
      ierr = DMLabelSetValue(label, p, 0);CHKERRQ(ierr);
    }
  }
  if (numRoots + numLeaves == (pEnd - pStart)) {
    for (p = pStart; p < pEnd; ++p) {
      PetscInt coneSize, supportSize;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
      if (!supportSize && coneSize) {
        ierr = DMLabelSetValue(label, p, 1);CHKERRQ(ierr);
      }
    }
  } else {
    IS       pointIS;
    PetscInt numPoints = 0, level = 0;

    ierr = DMLabelGetStratumIS(label, level, &pointIS);CHKERRQ(ierr);
    if (pointIS) {ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);}
    while (numPoints) {
      const PetscInt *points;
      const PetscInt  newLevel = level+1;

      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints; ++p) {
        const PetscInt  point = points[p];
        const PetscInt *support;
        PetscInt        supportSize, s;

        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMLabelSetValue(label, support[s], newLevel);CHKERRQ(ierr);
        }
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ++level;
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, level, &pointIS);CHKERRQ(ierr);
      if (pointIS) {ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);}
      else         {numPoints = 0;}
    }
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  { /* just in case there is an empty process */
    PetscInt numValues, maxValues = 0, v;

    ierr = DMLabelGetNumValues(label,&numValues);CHKERRQ(ierr);
    for (v = 0; v < numValues; v++) {
      IS pointIS;

      ierr = DMLabelGetStratumIS(label, v, &pointIS);CHKERRQ(ierr);
      if (pointIS) {
        PetscInt  min, max, numPoints;
        PetscInt  start;
        PetscBool contig;

        ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
        ierr = ISGetMinMax(pointIS, &min, &max);CHKERRQ(ierr);
        ierr = ISContiguousLocal(pointIS,min,max+1,&start,&contig);CHKERRQ(ierr);
        if (start == 0 && contig) {
          ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
          ierr = ISCreateStride(PETSC_COMM_SELF,numPoints,min,1,&pointIS);CHKERRQ(ierr);
          ierr = DMLabelSetStratumIS(label, v, pointIS);CHKERRQ(ierr);
        }
      }
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&numValues,&maxValues,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    for (v = numValues; v < maxValues; v++) {
      DMLabelAddStratum(label,v);CHKERRQ(ierr);
    }
  }

  ierr = DMLabelGetState(label, &mesh->depthState);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Stratify,dm,0,0,0);CHKERRQ(ierr);
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

.keywords: mesh
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
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);
  ierr = DMGetWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1]);CHKERRQ(ierr);
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
  ierr              = DMRestoreWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1-i]);CHKERRQ(ierr);
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

.keywords: mesh
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
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, (void*) coveredPoints);CHKERRQ(ierr);
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

.keywords: mesh
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
  PetscValidPointer(points, 2);
  PetscValidPointer(numCoveredPoints, 3);
  PetscValidPointer(coveredPoints, 4);

  ierr    = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr    = PetscCalloc1(numPoints, &closures);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, numPoints*(depth+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ms      = mesh->maxSupportSize;
  maxSize = (ms > 1) ? ((PetscPowInt(ms,depth+1)-1)/(ms-1)) : depth + 1;
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &join[0]);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &join[1]);CHKERRQ(ierr);

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
  ierr = DMRestoreWorkArray(dm, numPoints*(depth+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxSupportSize, PETSC_INT, &join[1-i]);CHKERRQ(ierr);
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

.keywords: mesh
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
  ierr = DMGetWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[0]);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1]);CHKERRQ(ierr);
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
  ierr               = DMRestoreWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1-i]);CHKERRQ(ierr);
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

.keywords: mesh
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
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, (void*) coveredPoints);CHKERRQ(ierr);
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

.keywords: mesh
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
  ierr    = DMGetWorkArray(dm, numPoints*(height+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  mc      = mesh->maxConeSize;
  maxSize = (mc > 1) ? ((PetscPowInt(mc,height+1)-1)/(mc-1)) : height + 1;
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &meet[0]);CHKERRQ(ierr);
  ierr    = DMGetWorkArray(dm, maxSize, PETSC_INT, &meet[1]);CHKERRQ(ierr);

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
  ierr = DMRestoreWorkArray(dm, numPoints*(height+2), PETSC_INT, &offsets);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, mesh->maxConeSize, PETSC_INT, &meet[1-i]);CHKERRQ(ierr);
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

.keywords: mesh
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

.keywords: mesh, points
.seealso: DMPlexGetDepth(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum()
@*/
PetscErrorCode DMPlexGetDepthLabel(DM dm, DMLabel *depthLabel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(depthLabel, 2);
  if (!dm->depthLabel) {ierr = DMGetLabel(dm, "depth", &dm->depthLabel);CHKERRQ(ierr);}
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

.keywords: mesh, points
.seealso: DMPlexGetDepthLabel(), DMPlexGetHeightStratum(), DMPlexGetDepthStratum()
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

  Level: developer

.keywords: mesh, points
.seealso: DMPlexGetHeightStratum(), DMPlexGetDepth()
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

  Level: developer

.keywords: mesh, points
.seealso: DMPlexGetDepthStratum(), DMPlexGetDepth()
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
  if (!label) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "No label named depth was found");CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(label, &depth);CHKERRQ(ierr);
  ierr = DMLabelGetStratumBounds(label, depth-1-stratumValue, start, end);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set the number of dof on each point and separate by fields */
static PetscErrorCode DMPlexCreateSectionInitial(DM dm, PetscInt dim, PetscInt numFields,const PetscInt numComp[],const PetscInt numDof[], PetscSection *section)
{
  PetscInt      *pMax;
  PetscInt       depth, pStart = 0, pEnd = 0;
  PetscInt       Nf, p, d, dep, f;
  PetscBool     *isFE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(numFields, &isFE);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject  obj;
    PetscClassId id;

    isFE[f] = PETSC_FALSE;
    if (f >= Nf) continue;
    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), section);CHKERRQ(ierr);
  if (numFields > 0) {
    ierr = PetscSectionSetNumFields(*section, numFields);CHKERRQ(ierr);
    if (numComp) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldComponents(*section, f, numComp[f]);CHKERRQ(ierr);
        if (isFE[f]) {
          PetscFE           fe;
          PetscDualSpace    dspace;
          const PetscInt    ***perms;
          const PetscScalar ***flips;
          const PetscInt    *numDof;

          ierr = DMGetField(dm,f,(PetscObject *) &fe);CHKERRQ(ierr);
          ierr = PetscFEGetDualSpace(fe,&dspace);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetSymmetries(dspace,&perms,&flips);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetNumDof(dspace,&numDof);CHKERRQ(ierr);
          if (perms || flips) {
            DM               K;
            DMLabel          depthLabel;
            PetscInt         depth, h;
            PetscSectionSym  sym;

            ierr = PetscDualSpaceGetDM(dspace,&K);CHKERRQ(ierr);
            ierr = DMPlexGetDepthLabel(dm,&depthLabel);CHKERRQ(ierr);
            ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
            ierr = PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)*section),depthLabel,&sym);CHKERRQ(ierr);
            for (h = 0; h <= depth; h++) {
              PetscDualSpace    hspace;
              PetscInt          kStart, kEnd;
              PetscInt          kConeSize;
              const PetscInt    **perms0 = NULL;
              const PetscScalar **flips0 = NULL;

              ierr = PetscDualSpaceGetHeightSubspace(dspace,h,&hspace);CHKERRQ(ierr);
              ierr = DMPlexGetHeightStratum(K,h,&kStart,&kEnd);CHKERRQ(ierr);
              if (!hspace) continue;
              ierr = PetscDualSpaceGetSymmetries(hspace,&perms,&flips);CHKERRQ(ierr);
              if (perms) perms0 = perms[0];
              if (flips) flips0 = flips[0];
              if (!(perms0 || flips0)) continue;
              ierr = DMPlexGetConeSize(K,kStart,&kConeSize);CHKERRQ(ierr);
              ierr = PetscSectionSymLabelSetStratum(sym,depth - h,numDof[depth - h],-kConeSize,kConeSize,PETSC_USE_POINTER,perms0 ? &perms0[-kConeSize] : NULL,flips0 ? &flips0[-kConeSize] : NULL);CHKERRQ(ierr);
            }
            ierr = PetscSectionSetFieldSym(*section,f,sym);CHKERRQ(ierr);
            ierr = PetscSectionSymDestroy(&sym);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(*section, pStart, pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc1(depth+1,&pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, depth >= 0 ? &pMax[depth] : NULL, depth>1 ? &pMax[depth-1] : NULL, depth>2 ? &pMax[1] : NULL, &pMax[0]);CHKERRQ(ierr);
  for (dep = 0; dep <= depth; ++dep) {
    d    = dim == depth ? dep : (!dep ? 0 : dim);
    ierr = DMPlexGetDepthStratum(dm, dep, &pStart, &pEnd);CHKERRQ(ierr);
    pMax[dep] = pMax[dep] < 0 ? pEnd : pMax[dep];
    for (p = pStart; p < pEnd; ++p) {
      PetscInt tot = 0;

      for (f = 0; f < numFields; ++f) {
        if (isFE[f] && p >= pMax[dep]) continue;
        ierr = PetscSectionSetFieldDof(*section, p, f, numDof[f*(dim+1)+d]);CHKERRQ(ierr);
        tot += numDof[f*(dim+1)+d];
      }
      ierr = PetscSectionSetDof(*section, p, tot);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(pMax);CHKERRQ(ierr);
  ierr = PetscFree(isFE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set the number of dof on each point and separate by fields
   If bcComps is NULL or the IS is NULL, constrain every dof on the point
*/
static PetscErrorCode DMPlexCreateSectionBCDof(DM dm, PetscInt numBC, const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], PetscSection section)
{
  PetscInt       numFields;
  PetscInt       bc;
  PetscSection   aSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {
    PetscInt        field = 0;
    const PetscInt *comp;
    const PetscInt *idx;
    PetscInt        Nc = -1, n, i;

    if (numFields) field = bcField[bc];
    if (bcComps && bcComps[bc]) {ierr = ISGetLocalSize(bcComps[bc], &Nc);CHKERRQ(ierr);}
    if (bcComps && bcComps[bc]) {ierr = ISGetIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt p = idx[i];
      PetscInt       numConst;

      if (numFields) {
        ierr = PetscSectionGetFieldDof(section, p, field, &numConst);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetDof(section, p, &numConst);CHKERRQ(ierr);
      }
      /* If Nc < 0, constrain every dof on the point */
      if (Nc > 0) numConst = PetscMin(numConst, Nc);
      if (numFields) {ierr = PetscSectionAddFieldConstraintDof(section, p, field, numConst);CHKERRQ(ierr);}
      ierr = PetscSectionAddConstraintDof(section, p, numConst);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    if (bcComps && bcComps[bc]) {ierr = ISRestoreIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
  }
  ierr = DMPlexGetAnchors(dm, &aSec, NULL);CHKERRQ(ierr);
  if (aSec) {
    PetscInt aStart, aEnd, a;

    ierr = PetscSectionGetChart(aSec, &aStart, &aEnd);CHKERRQ(ierr);
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      ierr = PetscSectionGetDof(aSec, a, &dof);CHKERRQ(ierr);
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        ierr = PetscSectionGetDof(section, a, &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetConstraintDof(section, a, dof);CHKERRQ(ierr);
        for (f = 0; f < numFields; f++) {
          ierr = PetscSectionGetFieldDof(section, a, f, &dof);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldConstraintDof(section, a, f, dof);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Set the constrained field indices on each point
   If bcComps is NULL or the IS is NULL, constrain every dof on the point
*/
static PetscErrorCode DMPlexCreateSectionBCIndicesField(DM dm, PetscInt numBC,const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], PetscSection section)
{
  PetscSection   aSec;
  PetscInt      *indices;
  PetscInt       numFields, cdof, maxDof = 0, pStart, pEnd, p, bc, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (!numFields) PetscFunctionReturn(0);
  /* Initialize all field indices to -1 */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr); maxDof = PetscMax(maxDof, cdof);}
  ierr = PetscMalloc1(maxDof, &indices);CHKERRQ(ierr);
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) for (f = 0; f < numFields; ++f) {ierr = PetscSectionSetFieldConstraintIndices(section, p, f, indices);CHKERRQ(ierr);}
  /* Handle BC constraints */
  for (bc = 0; bc < numBC; ++bc) {
    const PetscInt  field = bcField[bc];
    const PetscInt *comp, *idx;
    PetscInt        Nc = -1, n, i;

    if (bcComps && bcComps[bc]) {ierr = ISGetLocalSize(bcComps[bc], &Nc);CHKERRQ(ierr);}
    if (bcComps && bcComps[bc]) {ierr = ISGetIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISGetLocalSize(bcPoints[bc], &n);CHKERRQ(ierr);
    ierr = ISGetIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscInt  p = idx[i];
      const PetscInt *find;
      PetscInt        fdof, fcdof, c;

      ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);
      if (!fdof) continue;
      if (Nc < 0) {
        for (d = 0; d < fdof; ++d) indices[d] = d;
        fcdof = fdof;
      } else {
        ierr = PetscSectionGetFieldConstraintDof(section, p, field, &fcdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintIndices(section, p, field, &find);CHKERRQ(ierr);
        for (d = 0; d < fcdof; ++d) {if (find[d] < 0) break; indices[d] = find[d];}
        for (c = 0; c < Nc; ++c) indices[d++] = comp[c];
        ierr = PetscSortRemoveDupsInt(&d, indices);CHKERRQ(ierr);
        for (c = d; c < fcdof; ++c) indices[c] = -1;
        fcdof = d;
      }
      ierr = PetscSectionSetFieldConstraintDof(section, p, field, fcdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldConstraintIndices(section, p, field, indices);CHKERRQ(ierr);
    }
    if (bcComps && bcComps[bc]) {ierr = ISRestoreIndices(bcComps[bc], &comp);CHKERRQ(ierr);}
    ierr = ISRestoreIndices(bcPoints[bc], &idx);CHKERRQ(ierr);
  }
  /* Handle anchors */
  ierr = DMPlexGetAnchors(dm, &aSec, NULL);CHKERRQ(ierr);
  if (aSec) {
    PetscInt aStart, aEnd, a;

    for (d = 0; d < maxDof; ++d) indices[d] = d;
    ierr = PetscSectionGetChart(aSec, &aStart, &aEnd);CHKERRQ(ierr);
    for (a = aStart; a < aEnd; a++) {
      PetscInt dof, f;

      ierr = PetscSectionGetDof(aSec, a, &dof);CHKERRQ(ierr);
      if (dof) {
        /* if there are point-to-point constraints, then all dofs are constrained */
        for (f = 0; f < numFields; f++) {
          ierr = PetscSectionSetFieldConstraintIndices(section, a, f, indices);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Set the constrained indices on each point */
static PetscErrorCode DMPlexCreateSectionBCIndices(DM dm, PetscSection section)
{
  PetscInt      *indices;
  PetscInt       numFields, maxDof, pStart, pEnd, p, f, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof, &indices);CHKERRQ(ierr);
  for (d = 0; d < maxDof; ++d) indices[d] = -1;
  for (p = pStart; p < pEnd; ++p) {
    PetscInt cdof, d;

    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    if (cdof) {
      if (numFields) {
        PetscInt numConst = 0, foff = 0;

        for (f = 0; f < numFields; ++f) {
          const PetscInt *find;
          PetscInt        fcdof, fdof;

          ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, p, f, &fcdof);CHKERRQ(ierr);
          /* Change constraint numbering from field component to local dof number */
          ierr = PetscSectionGetFieldConstraintIndices(section, p, f, &find);CHKERRQ(ierr);
          for (d = 0; d < fcdof; ++d) indices[numConst+d] = find[d] + foff;
          numConst += fcdof;
          foff     += fdof;
        }
        if (cdof != numConst) {ierr = PetscSectionSetConstraintDof(section, p, numConst);CHKERRQ(ierr);}
      } else {
        for (d = 0; d < cdof; ++d) indices[d] = d;
      }
      ierr = PetscSectionSetConstraintIndices(section, p, indices);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateSection - Create a PetscSection based upon the dof layout specification provided.

  Not Collective

  Input Parameters:
+ dm        - The DMPlex object
. dim       - The spatial dimension of the problem
. numFields - The number of fields in the problem
. numComp   - An array of size numFields that holds the number of components for each field
. numDof    - An array of size numFields*(dim+1) which holds the number of dof for each field on a mesh piece of dimension d
. numBC     - The number of boundary conditions
. bcField   - An array of size numBC giving the field number for each boundry condition
. bcComps   - [Optional] An array of size numBC giving an IS holding the field components to which each boundary condition applies
. bcPoints  - An array of size numBC giving an IS holding the Plex points to which each boundary condition applies
- perm      - Optional permutation of the chart, or NULL

  Output Parameter:
. section - The PetscSection object

  Notes: numDof[f*(dim+1)+d] gives the number of dof for field f on sieve points of dimension d. For instance, numDof[1] is the
  number of dof for field 0 on each edge.

  The chart permutation is the same one set using PetscSectionSetPermutation()

  Level: developer

  Fortran Notes:
  A Fortran 90 version is available as DMPlexCreateSectionF90()

.keywords: mesh, elements
.seealso: DMPlexCreate(), PetscSectionCreate(), PetscSectionSetPermutation()
@*/
PetscErrorCode DMPlexCreateSection(DM dm, PetscInt dim, PetscInt numFields,const PetscInt numComp[],const PetscInt numDof[], PetscInt numBC,const PetscInt bcField[], const IS bcComps[], const IS bcPoints[], IS perm, PetscSection *section)
{
  PetscSection   aSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateSectionInitial(dm, dim, numFields, numComp, numDof, section);CHKERRQ(ierr);
  ierr = DMPlexCreateSectionBCDof(dm, numBC, bcField, bcComps, bcPoints, *section);CHKERRQ(ierr);
  if (perm) {ierr = PetscSectionSetPermutation(*section, perm);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(*section);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(dm,&aSec,NULL);CHKERRQ(ierr);
  if (numBC || aSec) {
    ierr = DMPlexCreateSectionBCIndicesField(dm, numBC, bcField, bcComps, bcPoints, *section);CHKERRQ(ierr);
    ierr = DMPlexCreateSectionBCIndices(dm, *section);CHKERRQ(ierr);
  }
  ierr = PetscSectionViewFromOptions(*section,NULL,"-section_view");CHKERRQ(ierr);
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
  ierr = DMSetDefaultSection(*cdm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &s);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &m);CHKERRQ(ierr);
  ierr = DMSetDefaultConstraints(*cdm, s, m);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = MatDestroy(&m);CHKERRQ(ierr);
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

PetscErrorCode DMPlexCreateSpectralClosurePermutation(DM dm, PetscSection section)
{
  PetscInt      *perm;
  PetscInt       dim, eStart, k, Nf, f, Nc, c, i, j, size = 0, offset = 0, foffset = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  if (dim <= 1) PetscFunctionReturn(0);
  for (f = 0; f < Nf; ++f) {
    /* An order k SEM disc has k-1 dofs on an edge */
    ierr = DMPlexGetDepthStratum(dm, 1, &eStart, NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldDof(section, eStart, f, &k);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(section, f, &Nc);CHKERRQ(ierr);
    k = k/Nc + 1;
    size += PetscPowInt(k+1, dim)*Nc;
  }
  ierr = PetscMalloc1(size, &perm);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    switch (dim) {
    case 2:
      /* The original quad closure is oriented clockwise, {f, e_b, e_r, e_t, e_l, v_lb, v_rb, v_tr, v_tl} */
      ierr = DMPlexGetDepthStratum(dm, 1, &eStart, NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldDof(section, eStart, f, &k);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldComponents(section, f, &Nc);CHKERRQ(ierr);
      k = k/Nc + 1;
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
      ierr = DMPlexGetDepthStratum(dm, 1, &eStart, NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldDof(section, eStart, f, &k);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldComponents(section, f, &Nc);CHKERRQ(ierr);
      k = k/Nc + 1;
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
  prob    = dm->prob;
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
    ierr = DMGetWorkArray(dm, size, PETSC_SCALAR, &array);CHKERRQ(ierr);
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

static PetscErrorCode DMPlexGetCompressedClosure(DM dm, PetscSection section, PetscInt point, PetscInt *numPoints, PetscInt **points, PetscSection *clSec, IS *clPoints, const PetscInt **clp)
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

static PetscErrorCode DMPlexRestoreCompressedClosure(DM dm, PetscSection section, PetscInt point, PetscInt *numPoints, PetscInt **points, PetscSection *clSec, IS *clPoints, const PetscInt **clp)
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
- point - The sieve point in the DM

  Output Parameters:
+ csize - The number of values in the closure, or NULL
- values - The array of values, which is a borrowed array and should not be freed

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
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
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
    ierr = DMGetWorkArray(dm, asize, PETSC_SCALAR, &array);CHKERRQ(ierr);
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
. point - The sieve point in the DM
. csize - The number of values in the closure, or NULL
- values - The array of values, which is a borrowed array and should not be freed

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
  ierr = DMRestoreWorkArray(dm, size, PETSC_SCALAR, (void*) values);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode updatePointFieldsBC_private(PetscSection section, PetscInt point, const PetscInt perm[], const PetscScalar flip[], PetscInt f, void (*fuse)(PetscScalar*, PetscScalar), const PetscInt clperm[], const PetscScalar values[], PetscInt *offset, PetscScalar array[])
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
  if (fcdof) {
    ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
    if (clperm) {
      if (perm) {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {
            fuse(&a[b], values[clperm[foffset+perm[b]]] * (flip ? flip[perm[b]] : 1.));
            ++cind;
          }
        }
      } else {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {
            fuse(&a[b], values[clperm[foffset+     b ]] * (flip ? flip[     b ] : 1.));
            ++cind;
          }
        }
      }
    } else {
      if (perm) {
        for (b = 0; b < fdof; b++) {
          if ((cind < fcdof) && (b == fcdofs[cind])) {
            fuse(&a[b], values[foffset+perm[b]] * (flip ? flip[perm[b]] : 1.));
            ++cind;
          }
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
. point - The sieve point in the DM
. values - The array of values
- mode - The insert mode, where INSERT_ALL_VALUES and ADD_ALL_VALUES also overwrite boundary conditions

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
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
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
          updatePointFieldsBC_private(section, point, perm, flip, f, insert, clperm, values, &offset, array);
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
          updatePointFieldsBC_private(section, point, perm, flip, f, add, clperm, values, &offset, array);
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

PetscErrorCode DMPlexVecSetFieldClosure_Internal(DM dm, PetscSection section, Vec v, PetscBool fieldActive[], PetscInt point, const PetscScalar values[], InsertMode mode)
{
  PetscSection      clSection;
  IS                clPoints;
  PetscScalar       *array;
  PetscInt          *points = NULL;
  const PetscInt    *clp, *clperm;
  PetscInt          numFields, numPoints, p;
  PetscInt          offset = 0, f;
  PetscErrorCode    ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 3);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  /* Get points */
  ierr = PetscSectionGetClosureInversePermutation_Internal(section, (PetscObject) dm, NULL, &clperm);CHKERRQ(ierr);
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
        updatePointFieldsBC_private(section, point, perm, flip, f, insert, clperm, values, &offset, array);
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
  ierr = PetscViewerASCIIPrintf(viewer, "[%d]mat for sieve point %D\n", rank, point);CHKERRQ(ierr);
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

/* . off - The global offset of this point */
PetscErrorCode DMPlexGetIndicesPoint_Internal(PetscSection section, PetscInt point, PetscInt off, PetscInt *loff, PetscBool setBC, const PetscInt perm[], PetscInt indices[])
{
  PetscInt        dof;    /* The number of unknowns on this point */
  PetscInt        cdof;   /* The number of constraints on this point */
  const PetscInt *cdofs; /* The indices of the constrained dofs on this point */
  PetscInt        cind = 0, k;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetConstraintDof(section, point, &cdof);CHKERRQ(ierr);
  if (!cdof || setBC) {
    if (perm) {
      for (k = 0; k < dof; k++) indices[*loff+perm[k]] = off + k;
    } else {
      for (k = 0; k < dof; k++) indices[*loff+k] = off + k;
    }
  } else {
    ierr = PetscSectionGetConstraintIndices(section, point, &cdofs);CHKERRQ(ierr);
    if (perm) {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[*loff+perm[k]] = -(off+k+1);
          ++cind;
        } else {
          indices[*loff+perm[k]] = off+k-cind;
        }
      }
    } else {
      for (k = 0; k < dof; ++k) {
        if ((cind < cdof) && (k == cdofs[cind])) {
          /* Insert check for returning constrained indices */
          indices[*loff+k] = -(off+k+1);
          ++cind;
        } else {
          indices[*loff+k] = off+k-cind;
        }
      }
    }
  }
  *loff += dof;
  PetscFunctionReturn(0);
}

/* . off - The global offset of this point */
PetscErrorCode DMPlexGetIndicesPointFields_Internal(PetscSection section, PetscInt point, PetscInt off, PetscInt foffs[], PetscBool setBC, const PetscInt ***perms, PetscInt permsoff, PetscInt indices[])
{
  PetscInt       numFields, foff, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (f = 0, foff = 0; f < numFields; ++f) {
    PetscInt        fdof, cfdof;
    const PetscInt *fcdofs; /* The indices of the constrained dofs for field f on this point */
    PetscInt        cind = 0, b;
    const PetscInt  *perm = (perms && perms[f]) ? perms[f][permsoff] : NULL;

    ierr = PetscSectionGetFieldDof(section, point, f, &fdof);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cfdof);CHKERRQ(ierr);
    if (!cfdof || setBC) {
      if (perm) {for (b = 0; b < fdof; b++) {indices[foffs[f]+perm[b]] = off+foff+b;}}
      else      {for (b = 0; b < fdof; b++) {indices[foffs[f]+     b ] = off+foff+b;}}
    } else {
      ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &fcdofs);CHKERRQ(ierr);
      if (perm) {
        for (b = 0; b < fdof; b++) {
          if ((cind < cfdof) && (b == fcdofs[cind])) {
            indices[foffs[f]+perm[b]] = -(off+foff+b+1);
            ++cind;
          } else {
            indices[foffs[f]+perm[b]] = off+foff+b-cind;
          }
        }
      } else {
        for (b = 0; b < fdof; b++) {
          if ((cind < cfdof) && (b == fcdofs[cind])) {
            indices[foffs[f]+b] = -(off+foff+b+1);
            ++cind;
          } else {
            indices[foffs[f]+b] = off+foff+b-cind;
          }
        }
      }
    }
    foff     += (setBC ? fdof : (fdof - cfdof));
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
    ierr = PetscMemzero(newOffsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
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
      ierr = DMGetWorkArray(dm,numPoints+1,PETSC_INT,&pointMatOffsets[f]);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm,numPoints+1,PETSC_INT,&newPointOffsets[f]);CHKERRQ(ierr);
    }
  }
  else {
    ierr = DMGetWorkArray(dm,numPoints+1,PETSC_INT,&pointMatOffsets[0]);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm,numPoints,PETSC_INT,&newPointOffsets[0]);CHKERRQ(ierr);
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
      ierr = DMGetWorkArray(dm,pointMatOffsets[f][numPoints],PETSC_SCALAR,&pointMat[f]);CHKERRQ(ierr);
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
    ierr = DMGetWorkArray(dm,pointMatOffsets[0][numPoints],PETSC_SCALAR,&pointMat[0]);CHKERRQ(ierr);
  }

  /* output arrays */
  ierr = DMGetWorkArray(dm,2*newNumPoints,PETSC_INT,&newPoints);CHKERRQ(ierr);

  /* get the point-to-point matrices; construct newPoints */
  ierr = PetscSectionGetMaxDof(aSec, &maxAnchor);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,maxDof,PETSC_INT,&indices);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,maxAnchor*maxDof,PETSC_INT,&newIndices);CHKERRQ(ierr);
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
        ierr = DMPlexGetIndicesPointFields_Internal(cSec, b, bOff, fEnd, PETSC_TRUE, perms, p, indices);CHKERRQ(ierr);

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
          ierr = DMPlexGetIndicesPointFields_Internal(section, a, aOff, fAnchorEnd, PETSC_TRUE, NULL, -1, newIndices);CHKERRQ(ierr);
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
        ierr = DMPlexGetIndicesPoint_Internal(cSec, b, bOff, &bEnd, PETSC_TRUE, (perms && perms[0]) ? perms[0][p] : NULL, indices);CHKERRQ(ierr);

        ierr = PetscSectionGetOffset (aSec, b, &bOff);CHKERRQ(ierr);
        for (q = 0; q < bDof; q++) {
          PetscInt a = anchors[bOff + q], aOff;

          /* we take the orientation of ap into account in the order that we constructed the indices above: the newly added points have no orientation */

          newPoints[2*(newP + q)]     = a;
          newPoints[2*(newP + q) + 1] = 0;
          ierr = PetscSectionGetOffset(section, a, &aOff);CHKERRQ(ierr);
          ierr = DMPlexGetIndicesPoint_Internal(section, a, aOff, &bAnchorEnd, PETSC_TRUE, NULL, newIndices);CHKERRQ(ierr);
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
    ierr = DMGetWorkArray(dm,newNumIndices*numIndices,PETSC_SCALAR,&tmpValues);CHKERRQ(ierr);
    ierr = PetscMemzero(tmpValues,newNumIndices*numIndices*sizeof(*tmpValues));CHKERRQ(ierr);
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
      ierr = DMGetWorkArray(dm,newNumIndices*newNumIndices,PETSC_SCALAR,&newValues);CHKERRQ(ierr);
      ierr = PetscMemzero(newValues,newNumIndices*newNumIndices*sizeof(*newValues));CHKERRQ(ierr);
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

      ierr = DMRestoreWorkArray(dm,newNumIndices*numIndices,PETSC_SCALAR,&tmpValues);CHKERRQ(ierr);
    }
    else {
      newValues = tmpValues;
    }
  }

  /* clean up */
  ierr = DMRestoreWorkArray(dm,maxDof,PETSC_INT,&indices);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,maxAnchor*maxDof,PETSC_INT,&newIndices);CHKERRQ(ierr);

  if (numFields) {
    for (f = 0; f < numFields; f++) {
      ierr = DMRestoreWorkArray(dm,pointMatOffsets[f][numPoints],PETSC_SCALAR,&pointMat[f]);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm,numPoints+1,PETSC_INT,&pointMatOffsets[f]);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm,numPoints+1,PETSC_INT,&newPointOffsets[f]);CHKERRQ(ierr);
    }
  }
  else {
    ierr = DMRestoreWorkArray(dm,pointMatOffsets[0][numPoints],PETSC_SCALAR,&pointMat[0]);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,numPoints+1,PETSC_INT,&pointMatOffsets[0]);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,numPoints+1,PETSC_INT,&newPointOffsets[0]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);

  /* output */
  if (outPoints) {
    *outPoints = newPoints;
  }
  else {
    ierr = DMRestoreWorkArray(dm,2*newNumPoints,PETSC_INT,&newPoints);CHKERRQ(ierr);
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
  DMPlexGetClosureIndices - Get the global indices in a vector v for all points in the closure of the given point

  Not collective

  Input Parameters:
+ dm - The DM
. section - The section describing the layout in v, or NULL to use the default section
. globalSection - The section describing the parallel layout in v, or NULL to use the default section
- point - The mesh point

  Output parameters:
+ numIndices - The number of indices
. indices - The indices
- outOffsets - Field offset if not NULL

  Note: Must call DMPlexRestoreClosureIndices() to free allocated memory

  Level: advanced

.seealso DMPlexRestoreClosureIndices(), DMPlexVecGetClosure(), DMPlexMatSetClosure()
@*/
PetscErrorCode DMPlexGetClosureIndices(DM dm, PetscSection section, PetscSection globalSection, PetscInt point, PetscInt *numIndices, PetscInt **indices, PetscInt *outOffsets)
{
  PetscSection    clSection;
  IS              clPoints;
  const PetscInt *clp;
  const PetscInt  **perms[32] = {NULL};
  PetscInt       *points = NULL, *pointsNew;
  PetscInt        numPoints, numPointsNew;
  PetscInt        offsets[32];
  PetscInt        Nf, Nind, NindNew, off, globalOff, f, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);
  if (numIndices) PetscValidPointer(numIndices, 4);
  PetscValidPointer(indices, 5);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  if (Nf > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", Nf);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  /* Get points in closure */
  ierr = DMPlexGetCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
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
  ierr = DMGetWorkArray(dm, Nind, PETSC_INT, indices);CHKERRQ(ierr);
  if (Nf) {
    if (outOffsets) {
      PetscInt f;

      for (f = 0; f <= Nf; f++) {
        outOffsets[f] = offsets[f];
      }
    }
    for (p = 0; p < numPoints; p++) {
      ierr = PetscSectionGetOffset(globalSection, points[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPointFields_Internal(section, points[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, perms, p, *indices);
    }
  } else {
    for (p = 0, off = 0; p < numPoints; p++) {
      const PetscInt *perm = perms[0] ? perms[0][p] : NULL;

      ierr = PetscSectionGetOffset(globalSection, points[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPoint_Internal(section, points[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, *indices);
    }
  }
  /* Cleanup points */
  for (f = 0; f < PetscMax(1,Nf); f++) {
    if (Nf) {ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
    else    {ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms[f],NULL);CHKERRQ(ierr);}
  }
  if (numPointsNew) {
    ierr = DMRestoreWorkArray(dm, 2*numPointsNew, PETSC_INT, &pointsNew);CHKERRQ(ierr);
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
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, indices);CHKERRQ(ierr);
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
. point - The sieve point in the DM
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
  const PetscInt     *clp;
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
  if (!section) {ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 2);
  if (!globalSection) {ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 4);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(offsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
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
            ierr = DMGetWorkArray(dm,numIndices*numIndices,PETSC_SCALAR,&valCopy);CHKERRQ(ierr);
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
      ierr = DMRestoreWorkArray(dm,numIndices*numIndices,PETSC_SCALAR,&valCopy);CHKERRQ(ierr);
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
  ierr = DMGetWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  if (numFields) {
    for (p = 0; p < numPoints; p++) {
      ierr = PetscSectionGetOffset(globalSection, points[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPointFields_Internal(section, points[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, offsets, PETSC_FALSE, perms, p, indices);
    }
  } else {
    for (p = 0, off = 0; p < numPoints; p++) {
      const PetscInt *perm = perms[0] ? perms[0][p] : NULL;
      ierr = PetscSectionGetOffset(globalSection, points[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPoint_Internal(section, points[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, indices);
    }
  }
  if (mesh->printSetValues) {ierr = DMPlexPrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, A, point, numIndices, indices, 0, NULL, values);CHKERRQ(ierr);}
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, values, mode);
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
    ierr2 = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr2);
    CHKERRQ(ierr);
  }
  for (f = 0; f < PetscMax(1,numFields); f++) {
    if (numFields) {ierr = PetscSectionRestoreFieldPointSyms(section,f,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
    else           {ierr = PetscSectionRestorePointSyms(section,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
  }
  if (newNumPoints) {
    ierr = DMRestoreWorkArray(dm,newNumIndices*newNumIndices,PETSC_SCALAR,&newValues);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm,2*newNumPoints,PETSC_INT,&newPoints);CHKERRQ(ierr);
  }
  else {
    if (valCopy) {
      ierr = DMRestoreWorkArray(dm,numIndices*numIndices,PETSC_SCALAR,&valCopy);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreCompressedClosure(dm,section,point,&numPoints,&points,&clSection,&clPoints,&clp);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, numIndices, PETSC_INT, &indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexMatSetClosureRefined(DM dmf, PetscSection fsection, PetscSection globalFSection, DM dmc, PetscSection csection, PetscSection globalCSection, Mat A, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  DM_Plex        *mesh   = (DM_Plex*) dmf->data;
  PetscInt       *fpoints = NULL, *ftotpoints = NULL;
  PetscInt       *cpoints = NULL;
  PetscInt       *findices, *cindices;
  PetscInt        foffsets[32], coffsets[32];
  CellRefiner     cellRefiner;
  PetscInt        numFields, numSubcells, maxFPoints, numFPoints, numCPoints, numFIndices, numCIndices, dof, off, globalOff, pStart, pEnd, p, q, r, s, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 4);
  if (!fsection) {ierr = DMGetDefaultSection(dmf, &fsection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(fsection, PETSC_SECTION_CLASSID, 2);
  if (!csection) {ierr = DMGetDefaultSection(dmc, &csection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(csection, PETSC_SECTION_CLASSID, 5);
  if (!globalFSection) {ierr = DMGetDefaultGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalFSection, PETSC_SECTION_CLASSID, 3);
  if (!globalCSection) {ierr = DMGetDefaultGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalCSection, PETSC_SECTION_CLASSID, 6);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 7);
  ierr = PetscSectionGetNumFields(fsection, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(foffsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(coffsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = DMGetWorkArray(dmf, maxFPoints*2*numSubcells, PETSC_INT, &ftotpoints);CHKERRQ(ierr);
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
  ierr = DMGetWorkArray(dmf, numFIndices, PETSC_INT, &findices);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dmc, numCIndices, PETSC_INT, &cindices);CHKERRQ(ierr);
  if (numFields) {
    const PetscInt **permsF[32] = {NULL};
    const PetscInt **permsC[32] = {NULL};

    for (f = 0; f < numFields; f++) {
      ierr = PetscSectionGetFieldPointSyms(fsection,f,numFPoints,ftotpoints,&permsF[f],NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldPointSyms(csection,f,numCPoints,cpoints,&permsC[f],NULL);CHKERRQ(ierr);
    }
    for (p = 0; p < numFPoints; p++) {
      ierr = PetscSectionGetOffset(globalFSection, ftotpoints[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPointFields_Internal(fsection, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, foffsets, PETSC_FALSE, permsF, p, findices);
    }
    for (p = 0; p < numCPoints; p++) {
      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPointFields_Internal(csection, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, coffsets, PETSC_FALSE, permsC, p, cindices);
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
      ierr = DMPlexGetIndicesPoint_Internal(fsection, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, findices);CHKERRQ(ierr);
    }
    for (p = 0, off = 0; p < numCPoints; p++) {
      const PetscInt *perm = permsC ? permsC[p] : NULL;

      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      ierr = DMPlexGetIndicesPoint_Internal(csection, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, cindices);CHKERRQ(ierr);
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
    ierr2 = DMRestoreWorkArray(dmf, numFIndices, PETSC_INT, &findices);CHKERRQ(ierr2);
    ierr2 = DMRestoreWorkArray(dmc, numCIndices, PETSC_INT, &cindices);CHKERRQ(ierr2);
    CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dmf, numCPoints*2*4, PETSC_INT, &ftotpoints);CHKERRQ(ierr);
  ierr = DMPlexRestoreTransitiveClosure(dmc, point, PETSC_TRUE, &numCPoints, &cpoints);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmf, numFIndices, PETSC_INT, &findices);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dmc, numCIndices, PETSC_INT, &cindices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexMatGetClosureIndicesRefined(DM dmf, PetscSection fsection, PetscSection globalFSection, DM dmc, PetscSection csection, PetscSection globalCSection, PetscInt point, PetscInt cindices[], PetscInt findices[])
{
  PetscInt      *fpoints = NULL, *ftotpoints = NULL;
  PetscInt      *cpoints = NULL;
  PetscInt       foffsets[32], coffsets[32];
  CellRefiner    cellRefiner;
  PetscInt       numFields, numSubcells, maxFPoints, numFPoints, numCPoints, numFIndices, numCIndices, dof, off, globalOff, pStart, pEnd, p, q, r, s, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 4);
  if (!fsection) {ierr = DMGetDefaultSection(dmf, &fsection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(fsection, PETSC_SECTION_CLASSID, 2);
  if (!csection) {ierr = DMGetDefaultSection(dmc, &csection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(csection, PETSC_SECTION_CLASSID, 5);
  if (!globalFSection) {ierr = DMGetDefaultGlobalSection(dmf, &globalFSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalFSection, PETSC_SECTION_CLASSID, 3);
  if (!globalCSection) {ierr = DMGetDefaultGlobalSection(dmc, &globalCSection);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(globalCSection, PETSC_SECTION_CLASSID, 6);
  ierr = PetscSectionGetNumFields(fsection, &numFields);CHKERRQ(ierr);
  if (numFields > 31) SETERRQ1(PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Number of fields %D limited to 31", numFields);
  ierr = PetscMemzero(foffsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(coffsets, 32 * sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = DMGetWorkArray(dmf, maxFPoints*2*numSubcells, PETSC_INT, &ftotpoints);CHKERRQ(ierr);
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
      DMPlexGetIndicesPointFields_Internal(fsection, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, foffsets, PETSC_FALSE, permsF, p, findices);
    }
    for (p = 0; p < numCPoints; p++) {
      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPointFields_Internal(csection, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, coffsets, PETSC_FALSE, permsC, p, cindices);
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
      DMPlexGetIndicesPoint_Internal(fsection, ftotpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, findices);
    }
    for (p = 0, off = 0; p < numCPoints; p++) {
      const PetscInt *perm = permsC ? permsC[p] : NULL;

      ierr = PetscSectionGetOffset(globalCSection, cpoints[2*p], &globalOff);CHKERRQ(ierr);
      DMPlexGetIndicesPoint_Internal(csection, cpoints[2*p], globalOff < 0 ? -(globalOff+1) : globalOff, &off, PETSC_FALSE, perm, cindices);
    }
    ierr = PetscSectionRestorePointSyms(fsection,numFPoints,ftotpoints,&permsF,NULL);CHKERRQ(ierr);
    ierr = PetscSectionRestorePointSyms(csection,numCPoints,cpoints,&permsC,NULL);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dmf, numCPoints*2*4, PETSC_INT, &ftotpoints);CHKERRQ(ierr);
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
  if (cMax) *cMax = mesh->hybridPointMax[dim];
  if (fMax) *fMax = mesh->hybridPointMax[dim-1];
  if (eMax) *eMax = mesh->hybridPointMax[1];
  if (vMax) *vMax = mesh->hybridPointMax[0];
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetHybridBounds - Set the first mesh point of each dimension which is a hybrid

  Input Parameters:
. dm   - The DMPlex object
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
  if (cMax >= 0) mesh->hybridPointMax[dim]   = cMax;
  if (fMax >= 0) mesh->hybridPointMax[dim-1] = fMax;
  if (eMax >= 0) mesh->hybridPointMax[1]     = eMax;
  if (vMax >= 0) mesh->hybridPointMax[0]     = vMax;
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

/* We can easily have a form that takes an IS instead */
static PetscErrorCode DMPlexCreateNumbering_Private(DM dm, PetscInt pStart, PetscInt pEnd, PetscInt shift, PetscInt *globalSize, PetscSF sf, IS *numbering)
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
  ierr = DMPlexCreateNumbering_Private(dm, cStart, cEnd, 0, NULL, dm->sf, globalCellNumbers);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
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
  ierr = DMPlexCreateNumbering_Private(dm, vStart, vEnd, 0, NULL, dm->sf, globalVertexNumbers);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetVertexNumbering - Get a global certex numbering for all vertices on this process

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

/*@C
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
  PetscInt       depths[4];
  PetscInt       depth, d, shift = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  /* For unstratified meshes use dim instead of depth */
  if (depth < 0) {ierr = DMGetDimension(dm, &depth);CHKERRQ(ierr);}
  depths[0] = depth; depths[1] = 0;
  for (d = 2; d <= depth; ++d) depths[d] = depth-d+1;
  for (d = 0; d <= depth; ++d) {
    PetscInt pStart, pEnd, gsize;

    ierr = DMPlexGetDepthStratum(dm, depths[d], &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexCreateNumbering_Private(dm, pStart, pEnd, shift, &gsize, dm->sf, &nums[d]);CHKERRQ(ierr);
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
  PetscDS        prob;
  PetscFE        fe;
  PetscScalar   *r;
  PetscMPIInt    rank;
  PetscInt       dim, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMClone(dm, &rdm);CHKERRQ(ierr);
  ierr = DMGetDimension(rdm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(rdm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "rank");CHKERRQ(ierr);
  ierr = DMGetDS(rdm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(rdm, ranks);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *ranks, "partition");CHKERRQ(ierr);
  ierr = VecGetArray(*ranks, &r);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *lr;

    ierr = DMPlexPointGlobalRef(rdm, c, r, &lr);CHKERRQ(ierr);
    *lr = rank;
  }
  ierr = VecRestoreArray(*ranks, &r);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckSymmetry - Check that the adjacency information in the mesh is symmetric.

  Input Parameter:
. dm - The DMPlex object

  Note: This is a useful diagnostic when creating meshes programmatically.

  Level: developer

.seealso: DMCreate(), DMCheckSkeleton(), DMCheckFaces()
@*/
PetscErrorCode DMPlexCheckSymmetry(DM dm)
{
  PetscSection    coneSection, supportSection;
  const PetscInt *cone, *support;
  PetscInt        coneSize, c, supportSize, s;
  PetscInt        pStart, pEnd, p, csize, ssize;
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
    ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
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
  ierr = PetscSectionGetStorageSize(coneSection, &csize);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(supportSection, &ssize);CHKERRQ(ierr);
  if (csize != ssize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Total cone size %D != Total support size %D", csize, ssize);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCheckSkeleton - Check that each cell has the correct number of vertices

  Input Parameters:
+ dm - The DMPlex object
. isSimplex - Are the cells simplices or tensor products
- cellHeight - Normally 0

  Note: This is a useful diagnostic when creating meshes programmatically.

  Level: developer

.seealso: DMCreate(), DMCheckSymmetry(), DMCheckFaces()
@*/
PetscErrorCode DMPlexCheckSkeleton(DM dm, PetscBool isSimplex, PetscInt cellHeight)
{
  PetscInt       dim, numCorners, numHybridCorners, vStart, vEnd, cStart, cEnd, cMax, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  switch (dim) {
  case 1: numCorners = isSimplex ? 2 : 2; numHybridCorners = isSimplex ? 2 : 2; break;
  case 2: numCorners = isSimplex ? 3 : 4; numHybridCorners = isSimplex ? 4 : 4; break;
  case 3: numCorners = isSimplex ? 4 : 8; numHybridCorners = isSimplex ? 6 : 8; break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle meshes of dimension %D", dim);
  }
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
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

  Input Parameters:
+ dm - The DMPlex object
. isSimplex - Are the cells simplices or tensor products
- cellHeight - Normally 0

  Note: This is a useful diagnostic when creating meshes programmatically.

  Level: developer

.seealso: DMCreate(), DMCheckSymmetry(), DMCheckSkeleton()
@*/
PetscErrorCode DMPlexCheckFaces(DM dm, PetscBool isSimplex, PetscInt cellHeight)
{
  PetscInt       pMax[4];
  PetscInt       dim, vStart, vEnd, cStart, cEnd, c, h;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pMax[dim], &pMax[dim-1], &pMax[1], &pMax[0]);CHKERRQ(ierr);
  for (h = cellHeight; h < dim; ++h) {
    ierr = DMPlexGetHeightStratum(dm, h, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone, *ornt, *faces;
      PetscInt        numFaces, faceSize, coneSize,f;
      PetscInt       *closure = NULL, closureSize, cl, numCorners = 0;

      if (pMax[dim-h] >= 0 && c >= pMax[dim-h]) continue;
      ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt p = closure[cl];
        if ((p >= vStart) && (p < vEnd)) closure[numCorners++] = p;
      }
      ierr = DMPlexGetRawFaces_Internal(dm, dim-h, numCorners, closure, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
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
      ierr = DMPlexRestoreFaces_Internal(dm, dim, c, &numFaces, &faceSize, &faces);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
  PetscBool      regular;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultGlobalSection(dmFine, &gsf);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsf, &m);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmCoarse, &gsc);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsc, &n);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject) dmCoarse), interpolation);CHKERRQ(ierr);
  ierr = MatSetSizes(*interpolation, m, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*interpolation, dmCoarse->mattype);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dmFine, &ctx);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dmFine, &cdm);CHKERRQ(ierr);
  ierr = DMPlexGetRegularRefinement(dmFine, &regular);CHKERRQ(ierr);
  if (regular && cdm == dmCoarse) {ierr = DMPlexComputeInterpolatorNested(dmCoarse, dmFine, *interpolation, ctx);CHKERRQ(ierr);}
  else                            {ierr = DMPlexComputeInterpolatorGeneral(dmCoarse, dmFine, *interpolation, ctx);CHKERRQ(ierr);}
  ierr = MatViewFromOptions(*interpolation, NULL, "-interp_mat_view");CHKERRQ(ierr);
  /* Use naive scaling */
  ierr = DMCreateInterpolationScale(dmCoarse, dmFine, *interpolation, scaling);CHKERRQ(ierr);
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

PetscErrorCode DMCreateDefaultSection_Plex(DM dm)
{
  PetscSection   section;
  IS            *bcPoints, *bcComps;
  PetscBool     *isFE;
  PetscInt      *bcFields, *numComp, *numDof;
  PetscInt       depth, dim, numBd, numBC = 0, numFields, bd, bc = 0, f;
  PetscInt       cStart, cEnd, cEndInterior;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  if (!numFields) PetscFunctionReturn(0);
  /* FE and FV boundary conditions are handled slightly differently */
  ierr = PetscMalloc1(numFields, &isFE);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
  }
  /* Allocate boundary point storage for FEM boundaries */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(dm->prob, &numBd);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    PetscInt                field;
    DMBoundaryConditionType type;
    const char             *labelName;
    DMLabel                 label;

    ierr = PetscDSGetBoundary(dm->prob, bd, &type, NULL, &labelName, &field, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMGetLabel(dm,labelName,&label);CHKERRQ(ierr);
    if (label && isFE[field] && (type & DM_BC_ESSENTIAL)) ++numBC;
  }
  /* Add ghost cell boundaries for FVM */
  for (f = 0; f < numFields; ++f) if (!isFE[f] && cEndInterior >= 0) ++numBC;
  ierr = PetscCalloc3(numBC,&bcFields,numBC,&bcPoints,numBC,&bcComps);CHKERRQ(ierr);
  /* Constrain ghost cells for FV */
  for (f = 0; f < numFields; ++f) {
    PetscInt *newidx, c;

    if (isFE[f] || cEndInterior < 0) continue;
    ierr = PetscMalloc1(cEnd-cEndInterior,&newidx);CHKERRQ(ierr);
    for (c = cEndInterior; c < cEnd; ++c) newidx[c-cEndInterior] = c;
    bcFields[bc] = f;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), cEnd-cEndInterior, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]);CHKERRQ(ierr);
  }
  /* Handle FEM Dirichlet boundaries */
  for (bd = 0; bd < numBd; ++bd) {
    const char             *bdLabel;
    DMLabel                 label;
    const PetscInt         *comps;
    const PetscInt         *values;
    PetscInt                bd2, field, numComps, numValues;
    DMBoundaryConditionType type;
    PetscBool               duplicate = PETSC_FALSE;

    ierr = PetscDSGetBoundary(dm->prob, bd, &type, NULL, &bdLabel, &field, &numComps, &comps, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    if (!isFE[field] || !label) continue;
    /* Only want to modify label once */
    for (bd2 = 0; bd2 < bd; ++bd2) {
      const char *bdname;
      ierr = PetscDSGetBoundary(dm->prob, bd2, NULL, NULL, &bdname, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscStrcmp(bdname, bdLabel, &duplicate);CHKERRQ(ierr);
      if (duplicate) break;
    }
    if (!duplicate && (isFE[field])) {
      /* don't complete cells, which are just present to give orientation to the boundary */
      ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
    }
    /* Filter out cells, if you actually want to constrain cells you need to do things by hand right now */
    if (type & DM_BC_ESSENTIAL) {
      PetscInt       *newidx;
      PetscInt        n, newn = 0, p, v;

      bcFields[bc] = field;
      if (numComps) {ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), numComps, comps, PETSC_COPY_VALUES, &bcComps[bc]);CHKERRQ(ierr);}
      for (v = 0; v < numValues; ++v) {
        IS              tmp;
        const PetscInt *idx;

        ierr = DMGetStratumIS(dm, bdLabel, values[v], &tmp);CHKERRQ(ierr);
        if (!tmp) continue;
        ierr = ISGetLocalSize(tmp, &n);CHKERRQ(ierr);
        ierr = ISGetIndices(tmp, &idx);CHKERRQ(ierr);
        if (isFE[field]) {
          for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) ++newn;
        } else {
          for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) ++newn;
        }
        ierr = ISRestoreIndices(tmp, &idx);CHKERRQ(ierr);
        ierr = ISDestroy(&tmp);CHKERRQ(ierr);
      }
      ierr = PetscMalloc1(newn,&newidx);CHKERRQ(ierr);
      newn = 0;
      for (v = 0; v < numValues; ++v) {
        IS              tmp;
        const PetscInt *idx;

        ierr = DMGetStratumIS(dm, bdLabel, values[v], &tmp);CHKERRQ(ierr);
        if (!tmp) continue;
        ierr = ISGetLocalSize(tmp, &n);CHKERRQ(ierr);
        ierr = ISGetIndices(tmp, &idx);CHKERRQ(ierr);
        if (isFE[field]) {
          for (p = 0; p < n; ++p) if ((idx[p] < cStart) || (idx[p] >= cEnd)) newidx[newn++] = idx[p];
        } else {
          for (p = 0; p < n; ++p) if ((idx[p] >= cStart) || (idx[p] < cEnd)) newidx[newn++] = idx[p];
        }
        ierr = ISRestoreIndices(tmp, &idx);CHKERRQ(ierr);
        ierr = ISDestroy(&tmp);CHKERRQ(ierr);
      }
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), newn, newidx, PETSC_OWN_POINTER, &bcPoints[bc++]);CHKERRQ(ierr);
    }
  }
  /* Handle discretization */
  ierr = PetscCalloc2(numFields,&numComp,numFields*(dim+1),&numDof);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscObject obj;

    ierr = DMGetField(dm, f, &obj);CHKERRQ(ierr);
    if (isFE[f]) {
      PetscFE         fe = (PetscFE) obj;
      const PetscInt *numFieldDof;
      PetscInt        d;

      ierr = PetscFEGetNumComponents(fe, &numComp[f]);CHKERRQ(ierr);
      ierr = PetscFEGetNumDof(fe, &numFieldDof);CHKERRQ(ierr);
      for (d = 0; d < dim+1; ++d) numDof[f*(dim+1)+d] = numFieldDof[d];
    } else {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetNumComponents(fv, &numComp[f]);CHKERRQ(ierr);
      numDof[f*(dim+1)+dim] = numComp[f];
    }
  }
  for (f = 0; f < numFields; ++f) {
    PetscInt d;
    for (d = 1; d < dim; ++d) {
      if ((numDof[f*(dim+1)+d] > 0) && (depth < dim)) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh must be interpolated when unknowns are specified on edges or faces.");
    }
  }
  ierr = DMPlexCreateSection(dm, dim, numFields, numComp, numDof, numBC, bcFields, bcComps, bcPoints, NULL, &section);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    PetscFE     fe;
    const char *name;

    ierr = DMGetField(dm, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) fe, &name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, f, name);CHKERRQ(ierr);
  }
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  for (bc = 0; bc < numBC; ++bc) {ierr = ISDestroy(&bcPoints[bc]);CHKERRQ(ierr);ierr = ISDestroy(&bcComps[bc]);CHKERRQ(ierr);}
  ierr = PetscFree3(bcFields,bcPoints,bcComps);CHKERRQ(ierr);
  ierr = PetscFree2(numComp,numDof);CHKERRQ(ierr);
  ierr = PetscFree(isFE);CHKERRQ(ierr);
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
    PetscDS  ds;
    PetscInt nf;

    ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
    ierr = DMPlexCreateConstraintSection_Anchors(dm,section,&cSec);CHKERRQ(ierr);
    ierr = DMPlexCreateConstraintMatrix_Anchors(dm,section,cSec,&cMat);CHKERRQ(ierr);
    ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds,&nf);CHKERRQ(ierr);
    if (nf && plex->computeanchormatrix) {ierr = (*plex->computeanchormatrix)(dm,section,cSec,cMat);CHKERRQ(ierr);}
    ierr = DMSetDefaultConstraints(dm,cSec,cMat);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&cSec);CHKERRQ(ierr);
    ierr = MatDestroy(&cMat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
