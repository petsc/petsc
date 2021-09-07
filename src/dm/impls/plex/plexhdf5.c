#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode DMSequenceView_HDF5(DM dm, const char *seqname, PetscInt seqnum, PetscScalar value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp);CHKERRQ(ierr);
  ierr = VecSetBlockSize(stamp, 1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) stamp, seqname);CHKERRQ(ierr);
  if (!rank) {
    PetscReal timeScale;
    PetscBool istime;

    ierr = PetscStrncmp(seqname, "time", 5, &istime);CHKERRQ(ierr);
    if (istime) {ierr = DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale);CHKERRQ(ierr); value *= timeScale;}
    ierr = VecSetValue(stamp, 0, value, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(stamp);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(stamp);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = VecView(stamp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&stamp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSequenceLoad_HDF5_Internal(DM dm, const char *seqname, PetscInt seqnum, PetscScalar *value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp);CHKERRQ(ierr);
  ierr = VecSetBlockSize(stamp, 1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) stamp, seqname);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = VecLoad(stamp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  if (!rank) {
    const PetscScalar *a;
    PetscReal timeScale;
    PetscBool istime;

    ierr = VecGetArrayRead(stamp, &a);CHKERRQ(ierr);
    *value = a[0];
    ierr = VecRestoreArrayRead(stamp, &a);CHKERRQ(ierr);
    ierr = PetscStrncmp(seqname, "time", 5, &istime);CHKERRQ(ierr);
    if (istime) {ierr = DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale);CHKERRQ(ierr); *value /= timeScale;}
  }
  ierr = VecDestroy(&stamp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCutVertexLabel_Private(DM dm, DMLabel cutLabel, DMLabel *cutVertexLabel)
{
  IS              cutcells = NULL;
  const PetscInt *cutc;
  PetscInt        cellHeight, vStart, vEnd, cStart, cEnd, c;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!cutLabel) PetscFunctionReturn(0);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  /* Label vertices that should be duplicated */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Cut Vertices", cutVertexLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(cutLabel, 2, &cutcells);CHKERRQ(ierr);
  if (cutcells) {
    PetscInt n;

    ierr = ISGetIndices(cutcells, &cutc);CHKERRQ(ierr);
    ierr = ISGetLocalSize(cutcells, &n);CHKERRQ(ierr);
    for (c = 0; c < n; ++c) {
      if ((cutc[c] >= cStart) && (cutc[c] < cEnd)) {
        PetscInt *closure = NULL;
        PetscInt  closureSize, cl, value;

        ierr = DMPlexGetTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (cl = 0; cl < closureSize*2; cl += 2) {
          if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) {
            ierr = DMLabelGetValue(cutLabel, closure[cl], &value);CHKERRQ(ierr);
            if (value == 1) {
              ierr = DMLabelSetValue(*cutVertexLabel, closure[cl], 1);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(cutcells, &cutc);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&cutcells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_Local_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM                      dm;
  DM                      dmBC;
  PetscSection            section, sectionGlobal;
  Vec                     gv;
  const char             *name;
  PetscViewerVTKFieldType ft;
  PetscViewerFormat       format;
  PetscInt                seqnum;
  PetscReal               seqval;
  PetscBool               isseq;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum, &seqval);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar) seqval, viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = DMGetOutputDM(dm, &dmBC);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmBC, &sectionGlobal);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmBC, &gv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) gv, name);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmBC, v, INSERT_VALUES, gv);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmBC, v, INSERT_VALUES, gv);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) gv, VECSEQ, &isseq);CHKERRQ(ierr);
  if (isseq) {ierr = VecView_Seq(gv, viewer);CHKERRQ(ierr);}
  else       {ierr = VecView_MPI(gv, viewer);CHKERRQ(ierr);}
  if (format == PETSC_VIEWER_HDF5_VIZ) {
    /* Output visualization representation */
    PetscInt numFields, f;
    DMLabel  cutLabel, cutVertexLabel = NULL;

    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      Vec         subv;
      IS          is;
      const char *fname, *fgroup;
      char        subname[PETSC_MAX_PATH_LEN];
      PetscInt    pStart, pEnd;

      ierr = DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft);CHKERRQ(ierr);
      fgroup = (ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_POINT_FIELD) ? "/vertex_fields" : "/cell_fields";
      ierr = PetscSectionGetFieldName(section, f, &fname);CHKERRQ(ierr);
      if (!fname) continue;
      ierr = PetscViewerHDF5PushGroup(viewer, fgroup);CHKERRQ(ierr);
      if (cutLabel) {
        const PetscScalar *ga;
        PetscScalar       *suba;
        PetscInt           Nc, gstart, subSize = 0, extSize = 0, subOff = 0, newOff = 0, p;

        ierr = DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldComponents(section, f, &Nc);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, fdof = 0, val;

          ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
          if (gdof > 0) {ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);}
          subSize += fdof;
          ierr = DMLabelGetValue(cutVertexLabel, p, &val);CHKERRQ(ierr);
          if (val == 1) extSize += fdof;
        }
        ierr = VecCreate(PetscObjectComm((PetscObject) gv), &subv);CHKERRQ(ierr);
        ierr = VecSetSizes(subv, subSize+extSize, PETSC_DETERMINE);CHKERRQ(ierr);
        ierr = VecSetBlockSize(subv, Nc);CHKERRQ(ierr);
        ierr = VecSetType(subv, VECSTANDARD);CHKERRQ(ierr);
        ierr = VecGetOwnershipRange(gv, &gstart, NULL);CHKERRQ(ierr);
        ierr = VecGetArrayRead(gv, &ga);CHKERRQ(ierr);
        ierr = VecGetArray(subv, &suba);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, goff, val;

          ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
          if (gdof > 0) {
            PetscInt fdof, fc, f2, poff = 0;

            ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
            /* Can get rid of this loop by storing field information in the global section */
            for (f2 = 0; f2 < f; ++f2) {
              ierr  = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
              poff += fdof;
            }
            ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
            for (fc = 0; fc < fdof; ++fc, ++subOff) suba[subOff] = ga[goff+poff+fc - gstart];
            ierr = DMLabelGetValue(cutVertexLabel, p, &val);CHKERRQ(ierr);
            if (val == 1) {
              for (fc = 0; fc < fdof; ++fc, ++newOff) suba[subSize+newOff] = ga[goff+poff+fc - gstart];
            }
          }
        }
        ierr = VecRestoreArrayRead(gv, &ga);CHKERRQ(ierr);
        ierr = VecRestoreArray(subv, &suba);CHKERRQ(ierr);
        ierr = DMLabelDestroy(&cutVertexLabel);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);
      }
      ierr = PetscStrncpy(subname, name,sizeof(subname));CHKERRQ(ierr);
      ierr = PetscStrlcat(subname, "_",sizeof(subname));CHKERRQ(ierr);
      ierr = PetscStrlcat(subname, fname,sizeof(subname));CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) subv, subname);CHKERRQ(ierr);
      if (isseq) {ierr = VecView_Seq(subv, viewer);CHKERRQ(ierr);}
      else       {ierr = VecView_MPI(subv, viewer);CHKERRQ(ierr);}
      if ((ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_CELL_VECTOR_FIELD)) {
        ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, "vector_field_type", PETSC_STRING, "vector");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, "vector_field_type", PETSC_STRING, "scalar");CHKERRQ(ierr);
      }
      if (cutLabel) {ierr = VecDestroy(&subv);CHKERRQ(ierr);}
      else          {ierr = PetscSectionRestoreField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);}
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreGlobalVector(dmBC, &gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  PetscObject    isZero;
  const char    *name;
  PetscReal      time;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) v, "__Vec_bc_zero__", &isZero);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", isZero);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, NULL, &time);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
  ierr = VecView_Plex_Local_HDF5_Internal(locv, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  PetscBool      isseq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
  else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  const char    *name;
  PetscInt       seqnum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  ierr = VecLoad_Plex_Local(locv, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, locv, INSERT_VALUES, v);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, locv, INSERT_VALUES, v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscInt       seqnum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  ierr = VecLoad_Default(v, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteTopology_HDF5_Static(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  IS              orderIS, conesIS, cellsIS, orntsIS;
  const PetscInt *gpoint;
  PetscInt       *order, *sizes, *cones, *ornts;
  PetscInt        dim, pStart, pEnd, p, conesSize = 0, cellsSize = 0, c = 0, s = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      PetscInt coneSize;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      conesSize += 1;
      cellsSize += coneSize;
    }
  }
  ierr = PetscMalloc1(conesSize, &order);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &sizes);CHKERRQ(ierr);
  ierr = PetscMalloc1(cellsSize, &cones);CHKERRQ(ierr);
  ierr = PetscMalloc1(cellsSize, &ornts);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, cp;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
      order[s]   = gpoint[p];
      sizes[s++] = coneSize;
      for (cp = 0; cp < coneSize; ++cp, ++c) {cones[c] = gpoint[cone[cp]] < 0 ? -(gpoint[cone[cp]]+1) : gpoint[cone[cp]]; ornts[c] = ornt[cp];}
    }
  }
  if (s != conesSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of points %D != %D", s, conesSize);
  if (c != cellsSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cone points %D != %D", c, cellsSize);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, order, PETSC_OWN_POINTER, &orderIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orderIS, "order");CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, sizes, PETSC_OWN_POINTER, &conesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) conesIS, "cones");CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), cellsSize, cones, PETSC_OWN_POINTER, &cellsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cellsIS, "cells");CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), cellsSize, ornts, PETSC_OWN_POINTER, &orntsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orntsIS, "orientation");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/topology");CHKERRQ(ierr);
  ierr = ISView(orderIS, viewer);CHKERRQ(ierr);
  ierr = ISView(conesIS, viewer);CHKERRQ(ierr);
  ierr = ISView(cellsIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellsIS, "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  ierr = ISView(orntsIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = ISDestroy(&orderIS);CHKERRQ(ierr);
  ierr = ISDestroy(&conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&cellsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&orntsIS);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateConesIS_Private(DM dm, PetscInt cStart, PetscInt cEnd, IS globalCellNumbers, PetscInt *numCorners, IS *cellIS)
{
  PetscSF         sfPoint;
  DMLabel         cutLabel, cutVertexLabel = NULL;
  IS              globalVertexNumbers, cutvertices = NULL;
  const PetscInt *gcell, *gvertex, *cutverts = NULL;
  PetscInt       *vertices;
  PetscInt        conesSize = 0;
  PetscInt        dim, numCornersLocal = 0, cell, vStart, vEnd, vExtra = 0, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *numCorners = 0;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(globalCellNumbers, &gcell);CHKERRQ(ierr);

  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, v, Nc = 0;

    if (gcell[cell] < 0) continue;
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++Nc;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    conesSize += Nc;
    if (!numCornersLocal)           numCornersLocal = Nc;
    else if (numCornersLocal != Nc) numCornersLocal = 1;
  }
  ierr = MPIU_Allreduce(&numCornersLocal, numCorners, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  if (numCornersLocal && (numCornersLocal != *numCorners || *numCorners == 1)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Visualization topology currently only supports identical cell shapes");
  /* Handle periodic cuts by identifying vertices which should be duplicated */
  ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);
  ierr = DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel);CHKERRQ(ierr);
  if (cutVertexLabel) {ierr = DMLabelGetStratumIS(cutVertexLabel, 1, &cutvertices);CHKERRQ(ierr);}
  if (cutvertices) {
    ierr = ISGetIndices(cutvertices, &cutverts);CHKERRQ(ierr);
    ierr = ISGetLocalSize(cutvertices, &vExtra);CHKERRQ(ierr);
  }
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  if (cutLabel) {
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscInt           nroots, nleaves;

    ierr = PetscSFGetGraph(sfPoint, &nroots, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
    if (nleaves < 0) {
      ierr = PetscObjectReference((PetscObject) sfPoint);CHKERRQ(ierr);
    } else {
      ierr = PetscSFCreate(PetscObjectComm((PetscObject) sfPoint), &sfPoint);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(sfPoint, nroots+vExtra, nleaves, ilocal, PETSC_USE_POINTER, iremote, PETSC_USE_POINTER);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject) sfPoint);CHKERRQ(ierr);
  }
  /* Number all vertices */
  ierr = DMPlexCreateNumbering_Plex(dm, vStart, vEnd+vExtra, 0, NULL, sfPoint, &globalVertexNumbers);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfPoint);CHKERRQ(ierr);
  /* Create cones */
  ierr = ISGetIndices(globalVertexNumbers, &gvertex);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &vertices);CHKERRQ(ierr);
  for (cell = cStart, v = 0; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, Nc = 0, p, value = -1;
    PetscBool replace;

    if (gcell[cell] < 0) continue;
    if (cutLabel) {ierr = DMLabelGetValue(cutLabel, cell, &value);CHKERRQ(ierr);}
    replace = (value == 2) ? PETSC_TRUE : PETSC_FALSE;
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (p = 0; p < closureSize*2; p += 2) {
      if ((closure[p] >= vStart) && (closure[p] < vEnd)) {
        closure[Nc++] = closure[p];
      }
    }
    ierr = DMPlexReorderCell(dm, cell, closure);CHKERRQ(ierr);
    for (p = 0; p < Nc; ++p) {
      PetscInt nv, gv = gvertex[closure[p] - vStart];

      if (replace) {
        ierr = PetscFindInt(closure[p], vExtra, cutverts, &nv);CHKERRQ(ierr);
        if (nv >= 0) gv = gvertex[vEnd - vStart + nv];
      }
      vertices[v++] = gv < 0 ? -(gv+1) : gv;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(globalVertexNumbers, &gvertex);CHKERRQ(ierr);
  ierr = ISDestroy(&globalVertexNumbers);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalCellNumbers, &gcell);CHKERRQ(ierr);
  if (cutvertices) {ierr = ISRestoreIndices(cutvertices, &cutverts);CHKERRQ(ierr);}
  ierr = ISDestroy(&cutvertices);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&cutVertexLabel);CHKERRQ(ierr);
  if (v != conesSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cell vertices %D != %D", v, conesSize);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, vertices, PETSC_OWN_POINTER, cellIS);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize((*cellIS)->map, *numCorners);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *cellIS, "cells");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteTopology_Vertices_HDF5_Static(DM dm, IS globalCellNumbers, PetscViewer viewer)
{
  DM              cdm;
  DMLabel         depthLabel, ctLabel;
  IS              cellIS;
  PetscInt        dim, depth, cellHeight, c;
  hid_t           fileId, groupId;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Gclose,(groupId));

  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMPlexGetCellTypeLabel(dm, &ctLabel);CHKERRQ(ierr);
  for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
    const DMPolytopeType ict = (DMPolytopeType) c;
    PetscInt             pStart, pEnd, dep, numCorners, n = 0;
    PetscBool            output = PETSC_FALSE, doOutput;

    if (ict == DM_POLYTOPE_FV_GHOST) continue;
    ierr = DMLabelGetStratumBounds(ctLabel, ict, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart >= 0) {
      ierr = DMLabelGetValue(depthLabel, pStart, &dep);CHKERRQ(ierr);
      if (dep == depth - cellHeight) output = PETSC_TRUE;
    }
    ierr = MPI_Allreduce(&output, &doOutput, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
    if (!doOutput) continue;
    ierr = CreateConesIS_Private(dm, pStart, pEnd, globalCellNumbers, &numCorners,  &cellIS);CHKERRQ(ierr);
    if (!n) {
      ierr = PetscViewerHDF5PushGroup(viewer, "/viz/topology");CHKERRQ(ierr);
    } else {
      char group[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/viz/topology_%D", n);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
    }
    ierr = ISView(cellIS, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellIS, "cell_corners", PETSC_INT, (void *) &numCorners);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellIS, "cell_dim",     PETSC_INT, (void *) &dim);CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ++n;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteCoordinates_HDF5_Static(DM dm, PetscViewer viewer)
{
  DM             cdm;
  Vec            coordinates, newcoords;
  PetscReal      lengthScale;
  PetscInt       m, M, bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) coordinates), &newcoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newcoords, "vertices");CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &m);CHKERRQ(ierr);
  ierr = VecSetSizes(newcoords, m, M);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);
  ierr = VecSetType(newcoords,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecCopy(coordinates, newcoords);CHKERRQ(ierr);
  ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
  /* Did not use DMGetGlobalVector() in order to bypass default group assignment */
  ierr = PetscViewerHDF5PushGroup(viewer, "/geometry");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = VecView(newcoords, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteCoordinates_Vertices_HDF5_Static(DM dm, PetscViewer viewer)
{
  DM               cdm;
  Vec              coordinatesLocal, newcoords;
  PetscSection     cSection, cGlobalSection;
  PetscScalar     *coords, *ncoords;
  DMLabel          cutLabel, cutVertexLabel = NULL;
  const PetscReal *L;
  const DMBoundaryType *bd;
  PetscReal        lengthScale;
  PetscInt         vStart, vEnd, v, bs, N, coordSize, dof, off, d;
  PetscBool        localized, embedded;
  hid_t            fileId, groupId;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinatesLocal);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinatesLocal, &bs);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  if (localized == PETSC_FALSE) PetscFunctionReturn(0);
  ierr = DMGetPeriodicity(dm, NULL, NULL, &L, &bd);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &cSection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(cdm, &cGlobalSection);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);
  N    = 0;

  ierr = DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &newcoords);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(cSection, vStart, &dof);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "DOF: %D\n", dof);CHKERRQ(ierr);
  embedded  = (PetscBool) (L && dof == 2 && !cutLabel);
  if (cutVertexLabel) {
    ierr = DMLabelGetStratumSize(cutVertexLabel, 1, &v);CHKERRQ(ierr);
    N   += dof*v;
  }
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(cGlobalSection, v, &dof);CHKERRQ(ierr);
    if (dof < 0) continue;
    if (embedded) N += dof+1;
    else          N += dof;
  }
  if (embedded) {ierr = VecSetBlockSize(newcoords, bs+1);CHKERRQ(ierr);}
  else          {ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);}
  ierr = VecSetSizes(newcoords, N, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newcoords, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesLocal, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(newcoords,        &ncoords);CHKERRQ(ierr);
  coordSize = 0;
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(cGlobalSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cSection, v, &off);CHKERRQ(ierr);
    if (dof < 0) continue;
    if (embedded) {
      if ((bd[0] == DM_BOUNDARY_PERIODIC) && (bd[1] == DM_BOUNDARY_PERIODIC)) {
        PetscReal theta, phi, r, R;
        /* XY-periodic */
        /* Suppose its an y-z circle, then
             \hat r = (0, cos(th), sin(th)) \hat x = (1, 0, 0)
           and the circle in that plane is
             \hat r cos(phi) + \hat x sin(phi) */
        theta = 2.0*PETSC_PI*PetscRealPart(coords[off+1])/L[1];
        phi   = 2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0];
        r     = L[0]/(2.0*PETSC_PI * 2.0*L[1]);
        R     = L[1]/(2.0*PETSC_PI);
        ncoords[coordSize++] =  PetscSinReal(phi) * r;
        ncoords[coordSize++] = -PetscCosReal(theta) * (R + r * PetscCosReal(phi));
        ncoords[coordSize++] =  PetscSinReal(theta) * (R + r * PetscCosReal(phi));
      } else if ((bd[0] == DM_BOUNDARY_PERIODIC)) {
        /* X-periodic */
        ncoords[coordSize++] = -PetscCosReal(2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0])*(L[0]/(2.0*PETSC_PI));
        ncoords[coordSize++] = coords[off+1];
        ncoords[coordSize++] = PetscSinReal(2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0])*(L[0]/(2.0*PETSC_PI));
      } else if ((bd[1] == DM_BOUNDARY_PERIODIC)) {
        /* Y-periodic */
        ncoords[coordSize++] = coords[off+0];
        ncoords[coordSize++] = PetscSinReal(2.0*PETSC_PI*PetscRealPart(coords[off+1])/L[1])*(L[1]/(2.0*PETSC_PI));
        ncoords[coordSize++] = -PetscCosReal(2.0*PETSC_PI*PetscRealPart(coords[off+1])/L[1])*(L[1]/(2.0*PETSC_PI));
      } else if ((bd[0] == DM_BOUNDARY_TWIST)) {
        PetscReal phi, r, R;
        /* Mobius strip */
        /* Suppose its an x-z circle, then
             \hat r = (-cos(phi), 0, sin(phi)) \hat y = (0, 1, 0)
           and in that plane we rotate by pi as we go around the circle
             \hat r cos(phi/2) + \hat y sin(phi/2) */
        phi   = 2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0];
        R     = L[0];
        r     = PetscRealPart(coords[off+1]) - L[1]/2.0;
        ncoords[coordSize++] = -PetscCosReal(phi) * (R + r * PetscCosReal(phi/2.0));
        ncoords[coordSize++] =  PetscSinReal(phi/2.0) * r;
        ncoords[coordSize++] =  PetscSinReal(phi) * (R + r * PetscCosReal(phi/2.0));
      } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot handle periodicity in this domain");
    } else {
      for (d = 0; d < dof; ++d, ++coordSize) ncoords[coordSize] = coords[off+d];
    }
  }
  if (cutVertexLabel) {
    IS              vertices;
    const PetscInt *verts;
    PetscInt        n;

    ierr = DMLabelGetStratumIS(cutVertexLabel, 1, &vertices);CHKERRQ(ierr);
    if (vertices) {
      ierr = ISGetIndices(vertices, &verts);CHKERRQ(ierr);
      ierr = ISGetLocalSize(vertices, &n);CHKERRQ(ierr);
      for (v = 0; v < n; ++v) {
        ierr = PetscSectionGetDof(cSection, verts[v], &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(cSection, verts[v], &off);CHKERRQ(ierr);
        for (d = 0; d < dof; ++d) ncoords[coordSize++] = coords[off+d] + ((bd[d] == DM_BOUNDARY_PERIODIC) ? L[d] : 0.0);
      }
      ierr = ISRestoreIndices(vertices, &verts);CHKERRQ(ierr);
      ierr = ISDestroy(&vertices);CHKERRQ(ierr);
    }
  }
  if (coordSize != N) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched sizes: %D != %D", coordSize, N);
  ierr = DMLabelDestroy(&cutVertexLabel);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesLocal, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(newcoords,        &ncoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newcoords, "vertices");CHKERRQ(ierr);
  ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Gclose,(groupId));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz/geometry");CHKERRQ(ierr);
  ierr = VecView(newcoords, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode DMPlexWriteLabels_HDF5_Static(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const PetscInt   *gpoint;
  PetscInt          numLabels, l;
  hid_t             fileId, groupId;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/labels");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  if (groupId != fileId) PetscStackCallHDF5(H5Gclose,(groupId));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label;
    const char     *name;
    IS              valueIS, pvalueIS, globalValueIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscBool       isDepth, output;
    char            group[PETSC_MAX_PATH_LEN];

    ierr = DMGetLabelName(dm, l, &name);CHKERRQ(ierr);
    ierr = DMGetLabelOutput(dm, name, &output);CHKERRQ(ierr);
    ierr = PetscStrncmp(name, "depth", 10, &isDepth);CHKERRQ(ierr);
    if (isDepth || !output) continue;
    ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
    ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s", name);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
    ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
    if (groupId != fileId) PetscStackCallHDF5(H5Gclose,(groupId));
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    /* Must copy to a new IS on the global comm */
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), numValues, values, PETSC_COPY_VALUES, &pvalueIS);CHKERRQ(ierr);
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISAllGather(pvalueIS, &globalValueIS);CHKERRQ(ierr);
    ierr = ISDestroy(&pvalueIS);CHKERRQ(ierr);
    ierr = ISSortRemoveDups(globalValueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(globalValueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(globalValueIS, &values);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      IS              stratumIS, globalStratumIS;
      const PetscInt *spoints = NULL;
      PetscInt       *gspoints, n = 0, gn, p;
      const char     *iname = "indices";

      ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s/%D", name, values[v]);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, values[v], &stratumIS);CHKERRQ(ierr);

      if (stratumIS) {ierr = ISGetLocalSize(stratumIS, &n);CHKERRQ(ierr);}
      if (stratumIS) {ierr = ISGetIndices(stratumIS, &spoints);CHKERRQ(ierr);}
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) ++gn;
      ierr = PetscMalloc1(gn,&gspoints);CHKERRQ(ierr);
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) gspoints[gn++] = gpoint[spoints[p]];
      if (stratumIS) {ierr = ISRestoreIndices(stratumIS, &spoints);CHKERRQ(ierr);}
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), gn, gspoints, PETSC_OWN_POINTER, &globalStratumIS);CHKERRQ(ierr);
      if (stratumIS) {ierr = PetscObjectGetName((PetscObject) stratumIS, &iname);CHKERRQ(ierr);}
      ierr = PetscObjectSetName((PetscObject) globalStratumIS, iname);CHKERRQ(ierr);

      ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
      ierr = ISView(globalStratumIS, viewer);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&globalStratumIS);CHKERRQ(ierr);
      ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(globalValueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&globalValueIS);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* We only write cells and vertices. Does this screw up parallel reading? */
PetscErrorCode DMPlexView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  IS                globalPointNumbers;
  PetscViewerFormat format;
  PetscBool         viz_geom=PETSC_FALSE, xdmf_topo=PETSC_FALSE, petsc_topo=PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreatePointNumbering(dm, &globalPointNumbers);CHKERRQ(ierr);
  ierr = DMPlexWriteCoordinates_HDF5_Static(dm, viewer);CHKERRQ(ierr);
  ierr = DMPlexWriteLabels_HDF5_Static(dm, globalPointNumbers, viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "petsc_version_git", PETSC_STRING, PETSC_VERSION_GIT);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  switch (format) {
    case PETSC_VIEWER_HDF5_VIZ:
      viz_geom    = PETSC_TRUE;
      xdmf_topo   = PETSC_TRUE;
      break;
    case PETSC_VIEWER_HDF5_XDMF:
      xdmf_topo   = PETSC_TRUE;
      break;
    case PETSC_VIEWER_HDF5_PETSC:
      petsc_topo  = PETSC_TRUE;
      break;
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_NATIVE:
      viz_geom    = PETSC_TRUE;
      xdmf_topo   = PETSC_TRUE;
      petsc_topo  = PETSC_TRUE;
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 output.", PetscViewerFormats[format]);
  }

  if (viz_geom)   {ierr = DMPlexWriteCoordinates_Vertices_HDF5_Static(dm, viewer);CHKERRQ(ierr);}
  if (xdmf_topo)  {ierr = DMPlexWriteTopology_Vertices_HDF5_Static(dm, globalPointNumbers, viewer);CHKERRQ(ierr);}
  if (petsc_topo) {ierr = DMPlexWriteTopology_HDF5_Static(dm, globalPointNumbers, viewer);CHKERRQ(ierr);}

  ierr = ISDestroy(&globalPointNumbers);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscMPIInt rank;
  DM          dm;
  PetscViewer viewer;
  DMLabel     label;
} LabelCtx;

static herr_t ReadLabelStratumHDF5_Static(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)
{
  PetscViewer     viewer = ((LabelCtx *) op_data)->viewer;
  DMLabel         label  = ((LabelCtx *) op_data)->label;
  IS              stratumIS;
  const PetscInt *ind;
  PetscInt        value, N, i;
  const char     *lname;
  char            group[PETSC_MAX_PATH_LEN];
  PetscErrorCode  ierr;

  ierr = PetscOptionsStringToInt(name, &value);
  ierr = ISCreate(PetscObjectComm((PetscObject) viewer), &stratumIS);
  ierr = PetscObjectSetName((PetscObject) stratumIS, "indices");
  ierr = PetscObjectGetName((PetscObject) label, &lname);
  ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s/%s", lname, name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, "indices", NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(stratumIS->map, !((LabelCtx *) op_data)->rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(stratumIS->map, N);CHKERRQ(ierr);
  }
  ierr = ISLoad(stratumIS, viewer);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = ISGetLocalSize(stratumIS, &N);
  ierr = ISGetIndices(stratumIS, &ind);
  for (i = 0; i < N; ++i) {ierr = DMLabelSetValue(label, ind[i], value);}
  ierr = ISRestoreIndices(stratumIS, &ind);
  ierr = ISDestroy(&stratumIS);
  return 0;
}

static herr_t ReadLabelHDF5_Static(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)
{
  DM             dm  = ((LabelCtx *) op_data)->dm;
  hsize_t        idx = 0;
  PetscErrorCode ierr;
  herr_t         err;

  ierr = DMCreateLabel(dm, name); if (ierr) return (herr_t) ierr;
  ierr = DMGetLabel(dm, name, &((LabelCtx *) op_data)->label); if (ierr) return (herr_t) ierr;
  PetscStackCall("H5Literate_by_name",err = H5Literate_by_name(g_id, name, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelStratumHDF5_Static, op_data, 0));
  return err;
}

PetscErrorCode DMPlexLoadLabels_HDF5_Internal(DM dm, PetscViewer viewer)
{
  LabelCtx        ctx;
  hid_t           fileId, groupId;
  hsize_t         idx = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &ctx.rank);CHKERRMPI(ierr);
  ctx.dm     = dm;
  ctx.viewer = viewer;
  ierr = PetscViewerHDF5PushGroup(viewer, "/labels");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Literate,(groupId, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelHDF5_Static, &ctx));
  PetscStackCallHDF5(H5Gclose,(groupId));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The first version will read everything onto proc 0, letting the user distribute
   The next will create a naive partition, and then rebalance after reading
*/
PetscErrorCode DMPlexLoad_HDF5_Internal(DM dm, PetscViewer viewer)
{
  PetscSection    coordSection;
  Vec             coordinates;
  IS              orderIS, conesIS, cellsIS, orntsIS;
  const PetscInt *order, *cones, *cells, *ornts;
  PetscReal       lengthScale;
  PetscInt       *cone, *ornt;
  PetscInt        dim, spatialDim, N, numVertices, vStart, vEnd, v, pEnd, p, q, maxConeSize = 0, c;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
  /* Read toplogy */
  ierr = PetscViewerHDF5PushGroup(viewer, "/topology");CHKERRQ(ierr);
  ierr = ISCreate(PetscObjectComm((PetscObject) dm), &orderIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orderIS, "order");CHKERRQ(ierr);
  ierr = ISCreate(PetscObjectComm((PetscObject) dm), &conesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) conesIS, "cones");CHKERRQ(ierr);
  ierr = ISCreate(PetscObjectComm((PetscObject) dm), &cellsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cellsIS, "cells");CHKERRQ(ierr);
  ierr = ISCreate(PetscObjectComm((PetscObject) dm), &orntsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orntsIS, "orientation");CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadObjectAttribute(viewer, (PetscObject) cellsIS, "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, "order", NULL, &pEnd);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(orderIS->map, !rank ? pEnd : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(orderIS->map, pEnd);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, "cones", NULL, &pEnd);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(conesIS->map, !rank ? pEnd : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(conesIS->map, pEnd);CHKERRQ(ierr);
    pEnd = !rank ? pEnd : 0;
    ierr = PetscViewerHDF5ReadSizes(viewer, "cells", NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(cellsIS->map, !rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(cellsIS->map, N);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, "orientation", NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(orntsIS->map, !rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(orntsIS->map, N);CHKERRQ(ierr);
  }
  ierr = ISLoad(orderIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(conesIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(cellsIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(orntsIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  /* Read geometry */
  ierr = PetscViewerHDF5PushGroup(viewer, "/geometry");CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "vertices");CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, "vertices", &spatialDim, &N);CHKERRQ(ierr);
    ierr = VecSetSizes(coordinates, !rank ? N : 0, N);CHKERRQ(ierr);
    ierr = VecSetBlockSize(coordinates, spatialDim);CHKERRQ(ierr);
  }
  ierr = VecLoad(coordinates, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = VecScale(coordinates, 1.0/lengthScale);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &numVertices);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &spatialDim);CHKERRQ(ierr);
  numVertices /= spatialDim;
  /* Create Plex */
  ierr = DMPlexSetChart(dm, 0, pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(orderIS, &order);CHKERRQ(ierr);
  ierr = ISGetIndices(conesIS, &cones);CHKERRQ(ierr);
  for (p = 0; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, order[p], cones[p]);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, cones[p]);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = ISGetIndices(cellsIS, &cells);CHKERRQ(ierr);
  ierr = ISGetIndices(orntsIS, &ornts);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxConeSize,&cone,maxConeSize,&ornt);CHKERRQ(ierr);
  for (p = 0, q = 0; p < pEnd; ++p) {
    for (c = 0; c < cones[p]; ++c, ++q) {cone[c] = cells[q]; ornt[c] = ornts[q];}
    ierr = DMPlexSetCone(dm, order[p], cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, order[p], ornt);CHKERRQ(ierr);
  }
  ierr = PetscFree2(cone,ornt);CHKERRQ(ierr);
  ierr = ISRestoreIndices(orderIS, &order);CHKERRQ(ierr);
  ierr = ISRestoreIndices(conesIS, &cones);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cellsIS, &cells);CHKERRQ(ierr);
  ierr = ISRestoreIndices(orntsIS, &ornts);CHKERRQ(ierr);
  ierr = ISDestroy(&orderIS);CHKERRQ(ierr);
  ierr = ISDestroy(&conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&cellsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&orntsIS);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Create coordinates */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  if (numVertices != vEnd - vStart) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of coordinates loaded %d does not match number of vertices %d", numVertices, vEnd - vStart);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spatialDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, spatialDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, spatialDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  /* Read Labels */
  ierr = DMPlexLoadLabels_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
