#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc-private/isimpl.h>
#include <petscviewerhdf5.h>

PETSC_EXTERN PetscErrorCode VecView_Seq(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__
#define __FUNCT__ "GetField_Static"
static PetscErrorCode GetField_Static(DM dm, PetscSection section, PetscSection sectionGlobal, Vec v, PetscInt field, PetscInt pStart, PetscInt pEnd, IS *is, Vec *subv)
{
  PetscInt      *subIndices;
  PetscInt       Nc, subSize = 0, subOff = 0, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetFieldComponents(section, field, &Nc);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, fdof = 0;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);}
    subSize += fdof;
  }
  ierr = PetscMalloc1(subSize, &subIndices);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gdof, goff;

    ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
    if (gdof > 0) {
      PetscInt fdof, fc, f2, poff = 0;

      ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
      /* Can get rid of this loop by storing field information in the global section */
      for (f2 = 0; f2 < field; ++f2) {
        ierr  = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
        poff += fdof;
      }
      ierr = PetscSectionGetFieldDof(section, p, field, &fdof);CHKERRQ(ierr);
      for (fc = 0; fc < fdof; ++fc, ++subOff) subIndices[subOff] = goff+poff+fc;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), subSize, subIndices, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
  ierr = VecGetSubVector(v, *is, subv);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*subv, Nc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RestoreField_Static"
static PetscErrorCode RestoreField_Static(DM dm, PetscSection section, PetscSection sectionGlobal, Vec v, PetscInt field, PetscInt pStart, PetscInt pEnd, IS *is, Vec *subv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecRestoreSubVector(v, *is, subv);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSequenceView_HDF5"
static PetscErrorCode DMSequenceView_HDF5(DM dm, const char *seqname, PetscInt seqnum, PetscScalar value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "VecView_Plex_Local_HDF5"
PetscErrorCode VecView_Plex_Local_HDF5(Vec v, PetscViewer viewer)
{
  DM                      dm;
  DM                      dmBC;
  PetscSection            section, sectionGlobal;
  Vec                     gv;
  const char             *name;
  PetscViewerVTKFieldType ft;
  PetscViewerFormat       format;
  PetscInt                seqnum;
  PetscBool               isseq;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar) seqnum, viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = DMGetOutputDM(dm, &dmBC);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmBC, &sectionGlobal);CHKERRQ(ierr);
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

    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      Vec         subv;
      IS          is;
      const char *fname, *fgroup;
      char        group[PETSC_MAX_PATH_LEN];
      PetscInt    pStart, pEnd;
      PetscBool   flag;

      ierr = DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft);CHKERRQ(ierr);
      fgroup = (ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_POINT_FIELD) ? "/vertex_fields" : "/cell_fields";
      ierr = PetscSectionGetFieldName(section, f, &fname);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer, fgroup);CHKERRQ(ierr);
      ierr = GetField_Static(dmBC, section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) subv, fname);CHKERRQ(ierr);
      if (isseq) {ierr = VecView_Seq(subv, viewer);CHKERRQ(ierr);}
      else       {ierr = VecView_MPI(subv, viewer);CHKERRQ(ierr);}
      ierr = RestoreField_Static(dmBC, section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
      ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "%s/%s", fgroup, fname);CHKERRQ(ierr);
      ierr = PetscViewerHDF5HasAttribute(viewer, group, "vector_field_type", &flag);CHKERRQ(ierr);
      if (!flag) {
        if ((ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_CELL_VECTOR_FIELD)) {
          ierr = PetscViewerHDF5WriteAttribute(viewer, group, "vector_field_type", PETSC_STRING, "vector");CHKERRQ(ierr);
        } else {
          ierr = PetscViewerHDF5WriteAttribute(viewer, group, "vector_field_type", PETSC_STRING, "scalar");CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMRestoreGlobalVector(dmBC, &gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView_Plex_HDF5"
PetscErrorCode VecView_Plex_HDF5(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
  ierr = VecView_Plex_Local(locv, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecLoad_Plex_HDF5"
PetscErrorCode VecLoad_Plex_HDF5(Vec v, PetscViewer viewer)
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
  ierr = DMGetOutputSequenceNumber(dm, &seqnum);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  ierr = VecLoad_Plex_Local(locv, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, locv, INSERT_VALUES, v);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, locv, INSERT_VALUES, v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexView_HDF5"
/* We only write cells and vertices. Does this screw up parallel reading? */
PetscErrorCode DMPlexView_HDF5(DM dm, PetscViewer viewer)
{
  Vec             coordinates, newcoords;
  Vec             coneVec, cellVec;
  IS              globalVertexNumbers, globalPointNumbers;
  const PetscInt *gvertex, *gpoint;
  PetscScalar    *sizes, *vertices;
  PetscReal       lengthScale;
  const char     *label   = NULL;
  PetscInt        labelId = 0, dim;
  char            group[PETSC_MAX_PATH_LEN];
  PetscInt        vStart, vEnd, v, cellHeight, cStart, cEnd, cMax, cell, conesSize = 0, numCornersLocal = 0, numCorners, bs, coordSize, numLabels, l;
  hid_t           fileId, groupId;
  herr_t          status;
  PetscViewerFormat format;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (cMax >= 0) cEnd = PetscMin(cEnd, cMax);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  /* Write coordinates */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecDuplicate(coordinates, &newcoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newcoords, "vertices");CHKERRQ(ierr);
  ierr = VecCopy(coordinates, newcoords);CHKERRQ(ierr);
  ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
  /* Use the local version to bypass the default group setting */
  ierr = PetscViewerHDF5PushGroup(viewer, "/geometry");CHKERRQ(ierr);
  ierr = VecView(newcoords, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &coordSize);CHKERRQ(ierr);
  if ((format == PETSC_VIEWER_HDF5_VIZ) && (coordSize != (vEnd - vStart)*bs)) {
    PetscSection     cSection;
    PetscScalar     *coords, *ncoords;
    const PetscReal *L;
    PetscInt         dof, off, d;

    ierr = DMGetPeriodicity(dm, NULL, &L);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &cSection);CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject) coordinates), &newcoords);CHKERRQ(ierr);
    coordSize = 0;
    for (v = vStart; v < vEnd; ++v) {
      ierr = PetscSectionGetDof(cSection, v, &dof);CHKERRQ(ierr);
      if (L && dof == 2) coordSize += dof+1;
      else               coordSize += dof;
    }
    if (L && bs == 2) {ierr = VecSetBlockSize(newcoords, bs+1);CHKERRQ(ierr);}
    else              {ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);}
    ierr = VecSetSizes(newcoords, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetType(newcoords,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecGetArray(newcoords,   &ncoords);CHKERRQ(ierr);
    coordSize = 0;
    for (v = vStart; v < vEnd; ++v) {
      ierr = PetscSectionGetDof(cSection, v, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(cSection, v, &off);CHKERRQ(ierr);
      if (L && dof == 2) {
        ncoords[coordSize++] = cos(2.0*PETSC_PI*coords[off+0]/L[0]);
        ncoords[coordSize++] = coords[off+1];
        ncoords[coordSize++] = sin(2.0*PETSC_PI*coords[off+0]/L[0]);
      } else {
        for (d = 0; d < dof; ++d, ++coordSize) ncoords[coordSize] = coords[off+d];
      }
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(newcoords,   &ncoords);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newcoords, "vertices");CHKERRQ(ierr);
    ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
    /* Use the local version to bypass the default group setting */
    ierr = PetscViewerHDF5PushGroup(viewer, "/viz");CHKERRQ(ierr);
    ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
    status = H5Gclose(groupId);CHKERRQ(status);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, "/viz/geometry");CHKERRQ(ierr);
    ierr = VecView(newcoords, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
  }
  /* Write toplogy */
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &coneVec);CHKERRQ(ierr);
  ierr = VecSetSizes(coneVec, cEnd-cStart, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coneVec, 1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coneVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coneVec, "coneSize");CHKERRQ(ierr);
  ierr = VecGetArray(coneVec, &sizes);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, v, Nc = 0;

    if (label) {
      PetscInt value;
      ierr = DMPlexGetLabelValue(dm, label, cell, &value);CHKERRQ(ierr);
      if (value == labelId) continue;
    }
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++Nc;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    conesSize += Nc;
    if (!numCornersLocal)           numCornersLocal = Nc;
    else if (numCornersLocal != Nc) numCornersLocal = 1;
  }
  ierr = VecRestoreArray(coneVec, &sizes);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&numCornersLocal, &numCorners, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  if (numCornersLocal && numCornersLocal != numCorners) numCorners = 1;

  ierr = DMPlexGetVertexNumbering(dm, &globalVertexNumbers);CHKERRQ(ierr);
  ierr = ISGetIndices(globalVertexNumbers, &gvertex);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &cellVec);CHKERRQ(ierr);
  ierr = VecSetSizes(cellVec, conesSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(cellVec, numCorners);CHKERRQ(ierr);
  ierr = VecSetFromOptions(cellVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cellVec, "cells");CHKERRQ(ierr);
  ierr = VecGetArray(cellVec, &vertices);CHKERRQ(ierr);
  for (cell = cStart, v = 0; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, Nc = 0, p;

    if (label) {
      PetscInt value;
      ierr = DMPlexGetLabelValue(dm, label, cell, &value);CHKERRQ(ierr);
      if (value == labelId) continue;
    }
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (p = 0; p < closureSize*2; p += 2) {
      if ((closure[p] >= vStart) && (closure[p] < vEnd)) {
        closure[Nc++] = closure[p];
        }
    }
    ierr = DMPlexInvertCell_Internal(dim, Nc, closure);CHKERRQ(ierr);
    for (p = 0; p < Nc; ++p) {
      const PetscInt gv = gvertex[closure[p] - vStart];
      vertices[v++] = gv < 0 ? -(gv+1) : gv;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  if (v != conesSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cell vertices %d != %d", v, conesSize);
  ierr = VecRestoreArray(cellVec, &vertices);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/topology");CHKERRQ(ierr);
  ierr = VecView(cellVec, viewer);CHKERRQ(ierr);
  if (numCorners == 1) {
    ierr = VecView(coneVec, viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerHDF5WriteAttribute(viewer, "/topology/cells", "cell_corners", PETSC_INT, (void *) &numCorners);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&cellVec);CHKERRQ(ierr);
  ierr = VecDestroy(&coneVec);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalVertexNumbers, &gvertex);CHKERRQ(ierr);

  ierr = PetscViewerHDF5WriteAttribute(viewer, "/topology/cells", "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  /* Write Labels*/
  ierr = DMPlexCreatePointNumbering(dm, &globalPointNumbers);CHKERRQ(ierr);
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/labels");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  if (groupId != fileId) {status = H5Gclose(groupId);CHKERRQ(status);}
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMPlexGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label;
    const char     *name;
    IS              valueIS, globalValueIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscBool       isDepth;

    ierr = DMPlexGetLabelName(dm, l, &name);CHKERRQ(ierr);
    ierr = PetscStrncmp(name, "depth", 10, &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = DMPlexGetLabel(dm, name, &label);CHKERRQ(ierr);
    ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s", name);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
    ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
    if (groupId != fileId) {status = H5Gclose(groupId);CHKERRQ(status);}
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISAllGather(valueIS, &globalValueIS);CHKERRQ(ierr);
    ierr = ISSortRemoveDups(globalValueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(globalValueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(globalValueIS, &values);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      IS              stratumIS, globalStratumIS;
      const PetscInt *spoints;
      PetscInt       *gspoints, n, gn, p;
      const char     *iname;

      ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s/%d", name, values[v]);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, values[v], &stratumIS);CHKERRQ(ierr);

      ierr = ISGetLocalSize(stratumIS, &n);CHKERRQ(ierr);
      ierr = ISGetIndices(stratumIS, &spoints);CHKERRQ(ierr);
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) ++gn;
      ierr = PetscMalloc1(gn,&gspoints);CHKERRQ(ierr);
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) gspoints[gn++] = gpoint[spoints[p]];
      ierr = ISRestoreIndices(stratumIS, &spoints);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), gn, gspoints, PETSC_OWN_POINTER, &globalStratumIS);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) stratumIS, &iname);CHKERRQ(ierr);
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
  ierr = DMLabelGetName(label, &lname);
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
  DM             dm = ((LabelCtx *) op_data)->dm;
  hsize_t        idx;
  herr_t         status;
  PetscErrorCode ierr;

  ierr = DMPlexCreateLabel(dm, name); if (ierr) return (herr_t) ierr;
  ierr = DMPlexGetLabel(dm, name, &((LabelCtx *) op_data)->label); if (ierr) return (herr_t) ierr;
  status = H5Literate_by_name(g_id, name, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelStratumHDF5_Static, op_data, 0);
  return status;
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexLoad_HDF5"
/* The first version will read everything onto proc 0, letting the user distribute
   The next will create a naive partition, and then rebalance after reading
*/
PetscErrorCode DMPlexLoad_HDF5(DM dm, PetscViewer viewer)
{
  LabelCtx       ctx;
  PetscSection   coordSection;
  Vec            coordinates;
  Vec            cellVec;
  PetscScalar   *cells;
  PetscReal      lengthScale;
  PetscInt      *cone;
  PetscInt       dim, spatialDim, N, numVertices, v, numCorners, numCells, cell, c;
  hid_t          fileId, groupId;
  hsize_t        idx = 0;
  herr_t         status;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  /* Read toplogy */
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/topology/cells", "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(dm, dim);CHKERRQ(ierr);
  /*   TODO Check for coneSize vector rather than this attribute */
  ierr = PetscViewerHDF5ReadAttribute(viewer, "/topology/cells", "cell_corners", PETSC_INT, (void *) &numCorners);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/topology");CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &cellVec);CHKERRQ(ierr);
  ierr = VecSetBlockSize(cellVec, numCorners);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cellVec, "cells");CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, "cells", NULL, &N);CHKERRQ(ierr);
    ierr = VecSetSizes(cellVec, !rank ? N : 0, N);CHKERRQ(ierr);
  }
  ierr = VecLoad(cellVec, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecGetLocalSize(cellVec, &numCells);CHKERRQ(ierr);
  numCells /= numCorners;
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
  ierr = DMPlexSetChart(dm, 0, numCells+numVertices);CHKERRQ(ierr);
  for (cell = 0; cell < numCells; ++cell) {ierr = DMPlexSetConeSize(dm, cell, numCorners);CHKERRQ(ierr);}
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCorners,&cone);CHKERRQ(ierr);
  ierr = VecGetArray(cellVec, &cells);CHKERRQ(ierr);
  for (cell = 0; cell < numCells; ++cell) {
    for (c = 0; c < numCorners; ++c) {cone[c] = numCells + cells[cell*numCorners+c];}
    ierr = DMPlexSetCone(dm, cell, cone);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(cellVec, &cells);CHKERRQ(ierr);
  ierr = PetscFree(cone);CHKERRQ(ierr);
  ierr = VecDestroy(&cellVec);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Create coordinates */
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, spatialDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells+numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, spatialDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, spatialDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  /* Read Labels*/
  ctx.rank   = rank;
  ctx.dm     = dm;
  ctx.viewer = viewer;
  ierr = PetscViewerHDF5PushGroup(viewer, "/labels");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  status = H5Literate(groupId, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelHDF5_Static, &ctx);CHKERRQ(status);
  status = H5Gclose(groupId);CHKERRQ(status);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
