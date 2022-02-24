#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)

typedef struct DMPlexStorageVersion {
  PetscInt major, minor, subminor;
} DMPlexStorageVersion;

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

static PetscErrorCode DMPlexStorageVersionParseString_Private(DM dm, const char str[], DMPlexStorageVersion *v)
{
  PetscToken      t;
  char           *ts;
  PetscInt        i;
  PetscInt        ti[3];

  PetscFunctionBegin;
  CHKERRQ(PetscTokenCreate(str, '.', &t));
  for (i=0; i<3; i++) {
    CHKERRQ(PetscTokenFind(t, &ts));
    PetscCheckFalse(!ts,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Malformed version string %s", str);
    CHKERRQ(PetscOptionsStringToInt(ts, &ti[i]));
  }
  CHKERRQ(PetscTokenFind(t, &ts));
  PetscCheckFalse(ts,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Malformed version string %s", str);
  CHKERRQ(PetscTokenDestroy(&t));
  v->major    = ti[0];
  v->minor    = ti[1];
  v->subminor = ti[2];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexStorageVersionSetUpWriting_Private(DM dm, PetscViewer viewer, DMPlexStorageVersion *version)
{
  const char      ATTR_NAME[] = "dmplex_storage_version";
  PetscBool       fileHasVersion;
  char            optVersion[16], fileVersion[16];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscStrcpy(fileVersion, DMPLEX_STORAGE_VERSION_STABLE));
  CHKERRQ(PetscViewerHDF5HasAttribute(viewer, NULL, ATTR_NAME, &fileHasVersion));
  if (fileHasVersion) {
    char *tmp;

    CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, NULL, &tmp));
    CHKERRQ(PetscStrcpy(fileVersion, tmp));
    CHKERRQ(PetscFree(tmp));
  }
  CHKERRQ(PetscStrcpy(optVersion, fileVersion));
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5 Viewer Options","PetscViewer");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-dm_plex_view_hdf5_storage_version","DMPlex HDF5 viewer storage version",NULL,optVersion,optVersion,sizeof(optVersion),NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!fileHasVersion) {
    CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, optVersion));
  } else {
    PetscBool flg;

    CHKERRQ(PetscStrcmp(fileVersion, optVersion, &flg));
    PetscCheckFalse(!flg,PetscObjectComm((PetscObject)dm), PETSC_ERR_FILE_UNEXPECTED, "User requested DMPlex storage version %s but file already has version %s - cannot mix versions", optVersion, fileVersion);
  }
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "petsc_version_git", PETSC_STRING, PETSC_VERSION_GIT));
  CHKERRQ(DMPlexStorageVersionParseString_Private(dm, optVersion, version));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexStorageVersionGet_Private(DM dm, PetscViewer viewer, DMPlexStorageVersion *version)
{
  const char      ATTR_NAME[]       = "dmplex_storage_version";
  char           *defaultVersion;
  char           *versionString;

  PetscFunctionBegin;
  //TODO string HDF5 attribute handling is terrible and should be redesigned
  CHKERRQ(PetscStrallocpy("1.0.0", &defaultVersion));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, &defaultVersion, &versionString));
  CHKERRQ(DMPlexStorageVersionParseString_Private(dm, versionString, version));
  CHKERRQ(PetscFree(versionString));
  CHKERRQ(PetscFree(defaultVersion));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetHDF5Name_Private(DM dm, const char *name[])
{
  PetscFunctionBegin;
  if (((PetscObject)dm)->name) {
    CHKERRQ(PetscObjectGetName((PetscObject)dm, name));
  } else {
    *name = "plex";
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSequenceView_HDF5(DM dm, const char *seqname, PetscInt seqnum, PetscScalar value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank));
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp));
  CHKERRQ(VecSetBlockSize(stamp, 1));
  CHKERRQ(PetscObjectSetName((PetscObject) stamp, seqname));
  if (rank == 0) {
    PetscReal timeScale;
    PetscBool istime;

    CHKERRQ(PetscStrncmp(seqname, "time", 5, &istime));
    if (istime) {CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale)); value *= timeScale;}
    CHKERRQ(VecSetValue(stamp, 0, value, INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(stamp));
  CHKERRQ(VecAssemblyEnd(stamp));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/"));
  CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));
  CHKERRQ(PetscViewerHDF5SetTimestep(viewer, seqnum)); /* seqnum < 0 jumps out above */
  CHKERRQ(VecView(stamp, viewer));
  CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(VecDestroy(&stamp));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSequenceLoad_HDF5_Internal(DM dm, const char *seqname, PetscInt seqnum, PetscScalar *value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank));
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp));
  CHKERRQ(VecSetBlockSize(stamp, 1));
  CHKERRQ(PetscObjectSetName((PetscObject) stamp, seqname));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/"));
  CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));
  CHKERRQ(PetscViewerHDF5SetTimestep(viewer, seqnum));  /* seqnum < 0 jumps out above */
  CHKERRQ(VecLoad(stamp, viewer));
  CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  if (rank == 0) {
    const PetscScalar *a;
    PetscReal timeScale;
    PetscBool istime;

    CHKERRQ(VecGetArrayRead(stamp, &a));
    *value = a[0];
    CHKERRQ(VecRestoreArrayRead(stamp, &a));
    CHKERRQ(PetscStrncmp(seqname, "time", 5, &istime));
    if (istime) {CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale)); *value /= timeScale;}
  }
  CHKERRQ(VecDestroy(&stamp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCutVertexLabel_Private(DM dm, DMLabel cutLabel, DMLabel *cutVertexLabel)
{
  IS              cutcells = NULL;
  const PetscInt *cutc;
  PetscInt        cellHeight, vStart, vEnd, cStart, cEnd, c;

  PetscFunctionBegin;
  if (!cutLabel) PetscFunctionReturn(0);
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  /* Label vertices that should be duplicated */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Cut Vertices", cutVertexLabel));
  CHKERRQ(DMLabelGetStratumIS(cutLabel, 2, &cutcells));
  if (cutcells) {
    PetscInt n;

    CHKERRQ(ISGetIndices(cutcells, &cutc));
    CHKERRQ(ISGetLocalSize(cutcells, &n));
    for (c = 0; c < n; ++c) {
      if ((cutc[c] >= cStart) && (cutc[c] < cEnd)) {
        PetscInt *closure = NULL;
        PetscInt  closureSize, cl, value;

        CHKERRQ(DMPlexGetTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure));
        for (cl = 0; cl < closureSize*2; cl += 2) {
          if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) {
            CHKERRQ(DMLabelGetValue(cutLabel, closure[cl], &value));
            if (value == 1) {
              CHKERRQ(DMLabelSetValue(*cutVertexLabel, closure[cl], 1));
            }
          }
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure));
      }
    }
    CHKERRQ(ISRestoreIndices(cutcells, &cutc));
  }
  CHKERRQ(ISDestroy(&cutcells));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq));
  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetOutputSequenceNumber(dm, &seqnum, &seqval));
  CHKERRQ(DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar) seqval, viewer));
  if (seqnum >= 0) {
    CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));
    CHKERRQ(PetscViewerHDF5SetTimestep(viewer, seqnum));
  }
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  CHKERRQ(DMGetOutputDM(dm, &dmBC));
  CHKERRQ(DMGetGlobalSection(dmBC, &sectionGlobal));
  CHKERRQ(DMGetGlobalVector(dmBC, &gv));
  CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
  CHKERRQ(PetscObjectSetName((PetscObject) gv, name));
  CHKERRQ(DMLocalToGlobalBegin(dmBC, v, INSERT_VALUES, gv));
  CHKERRQ(DMLocalToGlobalEnd(dmBC, v, INSERT_VALUES, gv));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) gv, VECSEQ, &isseq));
  if (isseq) CHKERRQ(VecView_Seq(gv, viewer));
  else       CHKERRQ(VecView_MPI(gv, viewer));
  if (format == PETSC_VIEWER_HDF5_VIZ) {
    /* Output visualization representation */
    PetscInt numFields, f;
    DMLabel  cutLabel, cutVertexLabel = NULL;

    CHKERRQ(PetscSectionGetNumFields(section, &numFields));
    CHKERRQ(DMGetLabel(dm, "periodic_cut", &cutLabel));
    for (f = 0; f < numFields; ++f) {
      Vec         subv;
      IS          is;
      const char *fname, *fgroup, *componentName;
      char        subname[PETSC_MAX_PATH_LEN];
      PetscInt    pStart, pEnd, Nc, c;

      CHKERRQ(DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft));
      fgroup = (ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_POINT_FIELD) ? "/vertex_fields" : "/cell_fields";
      CHKERRQ(PetscSectionGetFieldName(section, f, &fname));
      if (!fname) continue;
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, fgroup));
      if (cutLabel) {
        const PetscScalar *ga;
        PetscScalar       *suba;
        PetscInt          gstart, subSize = 0, extSize = 0, subOff = 0, newOff = 0, p;

        CHKERRQ(DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel));
        CHKERRQ(PetscSectionGetFieldComponents(section, f, &Nc));
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, fdof = 0, val;

          CHKERRQ(PetscSectionGetDof(sectionGlobal, p, &gdof));
          if (gdof > 0) CHKERRQ(PetscSectionGetFieldDof(section, p, f, &fdof));
          subSize += fdof;
          CHKERRQ(DMLabelGetValue(cutVertexLabel, p, &val));
          if (val == 1) extSize += fdof;
        }
        CHKERRQ(VecCreate(PetscObjectComm((PetscObject) gv), &subv));
        CHKERRQ(VecSetSizes(subv, subSize+extSize, PETSC_DETERMINE));
        CHKERRQ(VecSetBlockSize(subv, Nc));
        CHKERRQ(VecSetType(subv, VECSTANDARD));
        CHKERRQ(VecGetOwnershipRange(gv, &gstart, NULL));
        CHKERRQ(VecGetArrayRead(gv, &ga));
        CHKERRQ(VecGetArray(subv, &suba));
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, goff, val;

          CHKERRQ(PetscSectionGetDof(sectionGlobal, p, &gdof));
          if (gdof > 0) {
            PetscInt fdof, fc, f2, poff = 0;

            CHKERRQ(PetscSectionGetOffset(sectionGlobal, p, &goff));
            /* Can get rid of this loop by storing field information in the global section */
            for (f2 = 0; f2 < f; ++f2) {
              CHKERRQ(PetscSectionGetFieldDof(section, p, f2, &fdof));
              poff += fdof;
            }
            CHKERRQ(PetscSectionGetFieldDof(section, p, f, &fdof));
            for (fc = 0; fc < fdof; ++fc, ++subOff) suba[subOff] = ga[goff+poff+fc - gstart];
            CHKERRQ(DMLabelGetValue(cutVertexLabel, p, &val));
            if (val == 1) {
              for (fc = 0; fc < fdof; ++fc, ++newOff) suba[subSize+newOff] = ga[goff+poff+fc - gstart];
            }
          }
        }
        CHKERRQ(VecRestoreArrayRead(gv, &ga));
        CHKERRQ(VecRestoreArray(subv, &suba));
        CHKERRQ(DMLabelDestroy(&cutVertexLabel));
      } else {
        CHKERRQ(PetscSectionGetField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv));
      }
      CHKERRQ(PetscStrncpy(subname, name,sizeof(subname)));
      CHKERRQ(PetscStrlcat(subname, "_",sizeof(subname)));
      CHKERRQ(PetscStrlcat(subname, fname,sizeof(subname)));
      CHKERRQ(PetscObjectSetName((PetscObject) subv, subname));
      if (isseq) CHKERRQ(VecView_Seq(subv, viewer));
      else       CHKERRQ(VecView_MPI(subv, viewer));
      if ((ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_CELL_VECTOR_FIELD)) {
        CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, "vector_field_type", PETSC_STRING, "vector"));
      } else {
        CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, "vector_field_type", PETSC_STRING, "scalar"));
      }

      /* Output the component names in the field if available */
      CHKERRQ(PetscSectionGetFieldComponents(section, f, &Nc));
      for (c = 0; c < Nc; ++c){
        char componentNameLabel[PETSC_MAX_PATH_LEN];
        CHKERRQ(PetscSectionGetComponentName(section, f, c, &componentName));
        CHKERRQ(PetscSNPrintf(componentNameLabel, sizeof(componentNameLabel), "componentName%D", c));
        CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, componentNameLabel, PETSC_STRING, componentName));
      }

      if (cutLabel) CHKERRQ(VecDestroy(&subv));
      else          CHKERRQ(PetscSectionRestoreField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv));
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    }
  }
  if (seqnum >= 0) {
    CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  }
  CHKERRQ(DMRestoreGlobalVector(dmBC, &gv));
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  PetscObject    isZero;
  const char    *name;
  PetscReal      time;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMGetLocalVector(dm, &locv));
  CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
  CHKERRQ(PetscObjectSetName((PetscObject) locv, name));
  CHKERRQ(PetscObjectQuery((PetscObject) v, "__Vec_bc_zero__", &isZero));
  CHKERRQ(PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", isZero));
  CHKERRQ(DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv));
  CHKERRQ(DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv));
  CHKERRQ(DMGetOutputSequenceNumber(dm, NULL, &time));
  CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/fields"));
  CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ));
  CHKERRQ(VecView_Plex_Local_HDF5_Internal(locv, viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", NULL));
  CHKERRQ(DMRestoreLocalVector(dm, &locv));
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  PetscBool      isseq;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/fields"));
  if (isseq) CHKERRQ(VecView_Seq(v, viewer));
  else       CHKERRQ(VecView_MPI(v, viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  const char    *name;
  PetscInt       seqnum;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMGetLocalVector(dm, &locv));
  CHKERRQ(PetscObjectGetName((PetscObject) v, &name));
  CHKERRQ(PetscObjectSetName((PetscObject) locv, name));
  CHKERRQ(DMGetOutputSequenceNumber(dm, &seqnum, NULL));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/fields"));
  if (seqnum >= 0) {
    CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));
    CHKERRQ(PetscViewerHDF5SetTimestep(viewer, seqnum));
  }
  CHKERRQ(VecLoad_Plex_Local(locv, viewer));
  if (seqnum >= 0) {
    CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(DMLocalToGlobalBegin(dm, locv, INSERT_VALUES, v));
  CHKERRQ(DMLocalToGlobalEnd(dm, locv, INSERT_VALUES, v));
  CHKERRQ(DMRestoreLocalVector(dm, &locv));
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscInt       seqnum;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(v, &dm));
  CHKERRQ(DMGetOutputSequenceNumber(dm, &seqnum, NULL));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/fields"));
  if (seqnum >= 0) {
    CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));
    CHKERRQ(PetscViewerHDF5SetTimestep(viewer, seqnum));
  }
  CHKERRQ(VecLoad_Default(v, viewer));
  if (seqnum >= 0) {
    CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTopologyView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char           *topologydm_name;
  const char           *pointsName, *coneSizesName, *conesName, *orientationsName;
  IS                    pointsIS, coneSizesIS, conesIS, orientationsIS;
  PetscInt             *points, *coneSizes, *cones, *orientations;
  const PetscInt       *gpoint;
  PetscInt              pStart, pEnd, nPoints = 0, conesSize = 0;
  PetscInt              p, c, s;
  DMPlexStorageVersion  version;
  char                  group[PETSC_MAX_PATH_LEN];
  MPI_Comm              comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  pointsName        = "order";
  coneSizesName     = "cones";
  conesName         = "cells";
  orientationsName  = "orientation";
  CHKERRQ(DMPlexStorageVersionSetUpWriting_Private(dm, viewer, &version));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  if (version.major <= 1) {
    CHKERRQ(PetscStrcpy(group, "/topology"));
  } else {
    CHKERRQ(PetscSNPrintf(group, sizeof(group), "topologies/%s/topology", topologydm_name));
  }
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, group));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(ISGetIndices(globalPointNumbers, &gpoint));
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      PetscInt coneSize;

      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      nPoints += 1;
      conesSize += coneSize;
    }
  }
  CHKERRQ(PetscMalloc1(nPoints, &points));
  CHKERRQ(PetscMalloc1(nPoints, &coneSizes));
  CHKERRQ(PetscMalloc1(conesSize, &cones));
  CHKERRQ(PetscMalloc1(conesSize, &orientations));
  for (p = pStart, c = 0, s = 0; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, cp;

      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      CHKERRQ(DMPlexGetCone(dm, p, &cone));
      CHKERRQ(DMPlexGetConeOrientation(dm, p, &ornt));
      points[s]    = gpoint[p];
      coneSizes[s] = coneSize;
      for (cp = 0; cp < coneSize; ++cp, ++c) {
        cones[c] = gpoint[cone[cp]] < 0 ? -(gpoint[cone[cp]]+1) : gpoint[cone[cp]];
        orientations[c] = ornt[cp];
      }
      ++s;
    }
  }
  PetscCheckFalse(s != nPoints,PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of points %D != %D", s, nPoints);
  PetscCheckFalse(c != conesSize,PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cone points %D != %D", c, conesSize);
  CHKERRQ(ISCreateGeneral(comm, nPoints, points, PETSC_OWN_POINTER, &pointsIS));
  CHKERRQ(ISCreateGeneral(comm, nPoints, coneSizes, PETSC_OWN_POINTER, &coneSizesIS));
  CHKERRQ(ISCreateGeneral(comm, conesSize, cones, PETSC_OWN_POINTER, &conesIS));
  CHKERRQ(ISCreateGeneral(comm, conesSize, orientations, PETSC_OWN_POINTER, &orientationsIS));
  CHKERRQ(PetscObjectSetName((PetscObject) pointsIS, pointsName));
  CHKERRQ(PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName));
  CHKERRQ(PetscObjectSetName((PetscObject) conesIS, conesName));
  CHKERRQ(PetscObjectSetName((PetscObject) orientationsIS, orientationsName));
  CHKERRQ(ISView(pointsIS, viewer));
  CHKERRQ(ISView(coneSizesIS, viewer));
  CHKERRQ(ISView(conesIS, viewer));
  CHKERRQ(ISView(orientationsIS, viewer));
  CHKERRQ(ISDestroy(&pointsIS));
  CHKERRQ(ISDestroy(&coneSizesIS));
  CHKERRQ(ISDestroy(&conesIS));
  CHKERRQ(ISDestroy(&orientationsIS));
  CHKERRQ(ISRestoreIndices(globalPointNumbers, &gpoint));
  {
    PetscInt dim;

    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, conesName, "cell_dim", PETSC_INT, (void *) &dim));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
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

  PetscFunctionBegin;
  *numCorners = 0;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(ISGetIndices(globalCellNumbers, &gcell));

  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, v, Nc = 0;

    if (gcell[cell] < 0) continue;
    CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++Nc;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    conesSize += Nc;
    if (!numCornersLocal)           numCornersLocal = Nc;
    else if (numCornersLocal != Nc) numCornersLocal = 1;
  }
  CHKERRMPI(MPIU_Allreduce(&numCornersLocal, numCorners, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm)));
  PetscCheckFalse(numCornersLocal && (numCornersLocal != *numCorners || *numCorners == 1),PETSC_COMM_SELF, PETSC_ERR_SUP, "Visualization topology currently only supports identical cell shapes");
  /* Handle periodic cuts by identifying vertices which should be duplicated */
  CHKERRQ(DMGetLabel(dm, "periodic_cut", &cutLabel));
  CHKERRQ(DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel));
  if (cutVertexLabel) CHKERRQ(DMLabelGetStratumIS(cutVertexLabel, 1, &cutvertices));
  if (cutvertices) {
    CHKERRQ(ISGetIndices(cutvertices, &cutverts));
    CHKERRQ(ISGetLocalSize(cutvertices, &vExtra));
  }
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  if (cutLabel) {
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscInt           nroots, nleaves;

    CHKERRQ(PetscSFGetGraph(sfPoint, &nroots, &nleaves, &ilocal, &iremote));
    if (nleaves < 0) {
      CHKERRQ(PetscObjectReference((PetscObject) sfPoint));
    } else {
      CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject) sfPoint), &sfPoint));
      CHKERRQ(PetscSFSetGraph(sfPoint, nroots+vExtra, nleaves, (PetscInt*)ilocal, PETSC_COPY_VALUES, (PetscSFNode*)iremote, PETSC_COPY_VALUES));
    }
  } else {
    CHKERRQ(PetscObjectReference((PetscObject) sfPoint));
  }
  /* Number all vertices */
  CHKERRQ(DMPlexCreateNumbering_Plex(dm, vStart, vEnd+vExtra, 0, NULL, sfPoint, &globalVertexNumbers));
  CHKERRQ(PetscSFDestroy(&sfPoint));
  /* Create cones */
  CHKERRQ(ISGetIndices(globalVertexNumbers, &gvertex));
  CHKERRQ(PetscMalloc1(conesSize, &vertices));
  for (cell = cStart, v = 0; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, Nc = 0, p, value = -1;
    PetscBool replace;

    if (gcell[cell] < 0) continue;
    if (cutLabel) CHKERRQ(DMLabelGetValue(cutLabel, cell, &value));
    replace = (value == 2) ? PETSC_TRUE : PETSC_FALSE;
    CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (p = 0; p < closureSize*2; p += 2) {
      if ((closure[p] >= vStart) && (closure[p] < vEnd)) {
        closure[Nc++] = closure[p];
      }
    }
    CHKERRQ(DMPlexReorderCell(dm, cell, closure));
    for (p = 0; p < Nc; ++p) {
      PetscInt nv, gv = gvertex[closure[p] - vStart];

      if (replace) {
        CHKERRQ(PetscFindInt(closure[p], vExtra, cutverts, &nv));
        if (nv >= 0) gv = gvertex[vEnd - vStart + nv];
      }
      vertices[v++] = gv < 0 ? -(gv+1) : gv;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  }
  CHKERRQ(ISRestoreIndices(globalVertexNumbers, &gvertex));
  CHKERRQ(ISDestroy(&globalVertexNumbers));
  CHKERRQ(ISRestoreIndices(globalCellNumbers, &gcell));
  if (cutvertices) CHKERRQ(ISRestoreIndices(cutvertices, &cutverts));
  CHKERRQ(ISDestroy(&cutvertices));
  CHKERRQ(DMLabelDestroy(&cutVertexLabel));
  PetscCheckFalse(v != conesSize,PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cell vertices %D != %D", v, conesSize);
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, vertices, PETSC_OWN_POINTER, cellIS));
  CHKERRQ(PetscLayoutSetBlockSize((*cellIS)->map, *numCorners));
  CHKERRQ(PetscObjectSetName((PetscObject) *cellIS, "cells"));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteTopology_Vertices_HDF5_Static(DM dm, IS globalCellNumbers, PetscViewer viewer)
{
  DM              cdm;
  DMLabel         depthLabel, ctLabel;
  IS              cellIS;
  PetscInt        dim, depth, cellHeight, c;
  hid_t           fileId, groupId;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/viz"));
  CHKERRQ(PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId));
  PetscStackCallHDF5(H5Gclose,(groupId));

  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMPlexGetCellTypeLabel(dm, &ctLabel));
  for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
    const DMPolytopeType ict = (DMPolytopeType) c;
    PetscInt             pStart, pEnd, dep, numCorners, n = 0;
    PetscBool            output = PETSC_FALSE, doOutput;

    if (ict == DM_POLYTOPE_FV_GHOST) continue;
    CHKERRQ(DMLabelGetStratumBounds(ctLabel, ict, &pStart, &pEnd));
    if (pStart >= 0) {
      CHKERRQ(DMLabelGetValue(depthLabel, pStart, &dep));
      if (dep == depth - cellHeight) output = PETSC_TRUE;
    }
    CHKERRMPI(MPI_Allreduce(&output, &doOutput, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm)));
    if (!doOutput) continue;
    CHKERRQ(CreateConesIS_Private(dm, pStart, pEnd, globalCellNumbers, &numCorners,  &cellIS));
    if (!n) {
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/viz/topology"));
    } else {
      char group[PETSC_MAX_PATH_LEN];

      CHKERRQ(PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/viz/topology_%D", n));
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, group));
    }
    CHKERRQ(ISView(cellIS, viewer));
    CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellIS, "cell_corners", PETSC_INT, (void *) &numCorners));
    CHKERRQ(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellIS, "cell_dim",     PETSC_INT, (void *) &dim));
    CHKERRQ(ISDestroy(&cellIS));
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    ++n;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCoordinatesView_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
{
  DM             cdm;
  Vec            coordinates, newcoords;
  PetscReal      lengthScale;
  PetscInt       m, M, bs;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetCoordinates(dm, &coordinates));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject) coordinates), &newcoords));
  CHKERRQ(PetscObjectSetName((PetscObject) newcoords, "vertices"));
  CHKERRQ(VecGetSize(coordinates, &M));
  CHKERRQ(VecGetLocalSize(coordinates, &m));
  CHKERRQ(VecSetSizes(newcoords, m, M));
  CHKERRQ(VecGetBlockSize(coordinates, &bs));
  CHKERRQ(VecSetBlockSize(newcoords, bs));
  CHKERRQ(VecSetType(newcoords,VECSTANDARD));
  CHKERRQ(VecCopy(coordinates, newcoords));
  CHKERRQ(VecScale(newcoords, lengthScale));
  /* Did not use DMGetGlobalVector() in order to bypass default group assignment */
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/geometry"));
  CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  CHKERRQ(VecView(newcoords, viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(VecDestroy(&newcoords));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCoordinatesView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  DM              cdm;
  Vec             coords, newcoords;
  PetscInt        m, M, bs;
  PetscReal       lengthScale;
  const char     *topologydm_name, *coordinatedm_name, *coordinates_name;

  PetscFunctionBegin;
  {
    PetscViewerFormat     format;
    DMPlexStorageVersion  version;

    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    CHKERRQ(DMPlexStorageVersionSetUpWriting_Private(dm, viewer, &version));
    if (format == PETSC_VIEWER_HDF5_XDMF || format == PETSC_VIEWER_HDF5_VIZ || version.major <= 1) {
      CHKERRQ(DMPlexCoordinatesView_HDF5_Legacy_Private(dm, viewer));
      PetscFunctionReturn(0);
    }
  }
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetCoordinates(dm, &coords));
  CHKERRQ(PetscObjectGetName((PetscObject)cdm, &coordinatedm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)coords, &coordinates_name));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "coordinateDMName", PETSC_STRING, coordinatedm_name));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "coordinatesName", PETSC_STRING, coordinates_name));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(DMPlexSectionView(dm, viewer, cdm));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)coords), &newcoords));
  CHKERRQ(PetscObjectSetName((PetscObject)newcoords, coordinates_name));
  CHKERRQ(VecGetSize(coords, &M));
  CHKERRQ(VecGetLocalSize(coords, &m));
  CHKERRQ(VecSetSizes(newcoords, m, M));
  CHKERRQ(VecGetBlockSize(coords, &bs));
  CHKERRQ(VecSetBlockSize(newcoords, bs));
  CHKERRQ(VecSetType(newcoords,VECSTANDARD));
  CHKERRQ(VecCopy(coords, newcoords));
  CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  CHKERRQ(VecScale(newcoords, lengthScale));
  CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  CHKERRQ(DMPlexGlobalVectorView(dm, viewer, cdm, newcoords));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(VecDestroy(&newcoords));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinatesLocal));
  CHKERRQ(VecGetBlockSize(coordinatesLocal, &bs));
  CHKERRQ(DMGetCoordinatesLocalized(dm, &localized));
  if (localized == PETSC_FALSE) PetscFunctionReturn(0);
  CHKERRQ(DMGetPeriodicity(dm, NULL, NULL, &L, &bd));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &cSection));
  CHKERRQ(DMGetGlobalSection(cdm, &cGlobalSection));
  CHKERRQ(DMGetLabel(dm, "periodic_cut", &cutLabel));
  N    = 0;

  CHKERRQ(DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject) dm), &newcoords));
  CHKERRQ(PetscSectionGetDof(cSection, vStart, &dof));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "DOF: %D\n", dof));
  embedded  = (PetscBool) (L && dof == 2 && !cutLabel);
  if (cutVertexLabel) {
    CHKERRQ(DMLabelGetStratumSize(cutVertexLabel, 1, &v));
    N   += dof*v;
  }
  for (v = vStart; v < vEnd; ++v) {
    CHKERRQ(PetscSectionGetDof(cGlobalSection, v, &dof));
    if (dof < 0) continue;
    if (embedded) N += dof+1;
    else          N += dof;
  }
  if (embedded) CHKERRQ(VecSetBlockSize(newcoords, bs+1));
  else          CHKERRQ(VecSetBlockSize(newcoords, bs));
  CHKERRQ(VecSetSizes(newcoords, N, PETSC_DETERMINE));
  CHKERRQ(VecSetType(newcoords, VECSTANDARD));
  CHKERRQ(VecGetArray(coordinatesLocal, &coords));
  CHKERRQ(VecGetArray(newcoords,        &ncoords));
  coordSize = 0;
  for (v = vStart; v < vEnd; ++v) {
    CHKERRQ(PetscSectionGetDof(cGlobalSection, v, &dof));
    CHKERRQ(PetscSectionGetOffset(cSection, v, &off));
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

    CHKERRQ(DMLabelGetStratumIS(cutVertexLabel, 1, &vertices));
    if (vertices) {
      CHKERRQ(ISGetIndices(vertices, &verts));
      CHKERRQ(ISGetLocalSize(vertices, &n));
      for (v = 0; v < n; ++v) {
        CHKERRQ(PetscSectionGetDof(cSection, verts[v], &dof));
        CHKERRQ(PetscSectionGetOffset(cSection, verts[v], &off));
        for (d = 0; d < dof; ++d) ncoords[coordSize++] = coords[off+d] + ((bd[d] == DM_BOUNDARY_PERIODIC) ? L[d] : 0.0);
      }
      CHKERRQ(ISRestoreIndices(vertices, &verts));
      CHKERRQ(ISDestroy(&vertices));
    }
  }
  PetscCheckFalse(coordSize != N,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched sizes: %D != %D", coordSize, N);
  CHKERRQ(DMLabelDestroy(&cutVertexLabel));
  CHKERRQ(VecRestoreArray(coordinatesLocal, &coords));
  CHKERRQ(VecRestoreArray(newcoords,        &ncoords));
  CHKERRQ(PetscObjectSetName((PetscObject) newcoords, "vertices"));
  CHKERRQ(VecScale(newcoords, lengthScale));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/viz"));
  CHKERRQ(PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId));
  PetscStackCallHDF5(H5Gclose,(groupId));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/viz/geometry"));
  CHKERRQ(VecView(newcoords, viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(VecDestroy(&newcoords));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLabelsView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char           *topologydm_name;
  const PetscInt       *gpoint;
  PetscInt              numLabels, l;
  DMPlexStorageVersion  version;
  char                  group[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  CHKERRQ(DMPlexStorageVersionSetUpWriting_Private(dm, viewer, &version));
  CHKERRQ(ISGetIndices(globalPointNumbers, &gpoint));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  if (version.major <= 1) {
    CHKERRQ(PetscStrcpy(group, "/labels"));
  } else {
    CHKERRQ(PetscSNPrintf(group, sizeof(group), "topologies/%s/labels", topologydm_name));
  }
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, group));
  CHKERRQ(DMGetNumLabels(dm, &numLabels));
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label;
    const char     *name;
    IS              valueIS, pvalueIS, globalValueIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscBool       isDepth, output;

    CHKERRQ(DMGetLabelByNum(dm, l, &label));
    CHKERRQ(PetscObjectGetName((PetscObject)label, &name));
    CHKERRQ(DMGetLabelOutput(dm, name, &output));
    CHKERRQ(PetscStrncmp(name, "depth", 10, &isDepth));
    if (isDepth || !output) continue;
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, name));
    CHKERRQ(DMLabelGetValueIS(label, &valueIS));
    /* Must copy to a new IS on the global comm */
    CHKERRQ(ISGetLocalSize(valueIS, &numValues));
    CHKERRQ(ISGetIndices(valueIS, &values));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), numValues, values, PETSC_COPY_VALUES, &pvalueIS));
    CHKERRQ(ISRestoreIndices(valueIS, &values));
    CHKERRQ(ISAllGather(pvalueIS, &globalValueIS));
    CHKERRQ(ISDestroy(&pvalueIS));
    CHKERRQ(ISSortRemoveDups(globalValueIS));
    CHKERRQ(ISGetLocalSize(globalValueIS, &numValues));
    CHKERRQ(ISGetIndices(globalValueIS, &values));
    for (v = 0; v < numValues; ++v) {
      IS              stratumIS, globalStratumIS;
      const PetscInt *spoints = NULL;
      PetscInt       *gspoints, n = 0, gn, p;
      const char     *iname = "indices";
      char            group[PETSC_MAX_PATH_LEN];

      CHKERRQ(PetscSNPrintf(group, sizeof(group), "%D", values[v]));
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, group));
      CHKERRQ(DMLabelGetStratumIS(label, values[v], &stratumIS));

      if (stratumIS) CHKERRQ(ISGetLocalSize(stratumIS, &n));
      if (stratumIS) CHKERRQ(ISGetIndices(stratumIS, &spoints));
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) ++gn;
      CHKERRQ(PetscMalloc1(gn,&gspoints));
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) gspoints[gn++] = gpoint[spoints[p]];
      if (stratumIS) CHKERRQ(ISRestoreIndices(stratumIS, &spoints));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) dm), gn, gspoints, PETSC_OWN_POINTER, &globalStratumIS));
      CHKERRQ(PetscObjectSetName((PetscObject) globalStratumIS, iname));

      CHKERRQ(ISView(globalStratumIS, viewer));
      CHKERRQ(ISDestroy(&globalStratumIS));
      CHKERRQ(ISDestroy(&stratumIS));
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    }
    CHKERRQ(ISRestoreIndices(globalValueIS, &values));
    CHKERRQ(ISDestroy(&globalValueIS));
    CHKERRQ(ISDestroy(&valueIS));
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  }
  CHKERRQ(ISRestoreIndices(globalPointNumbers, &gpoint));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

/* We only write cells and vertices. Does this screw up parallel reading? */
PetscErrorCode DMPlexView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  IS                globalPointNumbers;
  PetscViewerFormat format;
  PetscBool         viz_geom=PETSC_FALSE, xdmf_topo=PETSC_FALSE, petsc_topo=PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(DMPlexCreatePointNumbering(dm, &globalPointNumbers));
  CHKERRQ(DMPlexCoordinatesView_HDF5_Internal(dm, viewer));

  CHKERRQ(PetscViewerGetFormat(viewer, &format));
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
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 output.", PetscViewerFormats[format]);
  }

  if (viz_geom)   CHKERRQ(DMPlexWriteCoordinates_Vertices_HDF5_Static(dm, viewer));
  if (xdmf_topo)  CHKERRQ(DMPlexWriteTopology_Vertices_HDF5_Static(dm, globalPointNumbers, viewer));
  if (petsc_topo) {
    CHKERRQ(DMPlexTopologyView_HDF5_Internal(dm, globalPointNumbers, viewer));
    CHKERRQ(DMPlexLabelsView_HDF5_Internal(dm, globalPointNumbers, viewer));
  }

  CHKERRQ(ISDestroy(&globalPointNumbers));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexSectionView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm)
{
  MPI_Comm       comm;
  const char    *topologydm_name;
  const char    *sectiondm_name;
  PetscSection   gsection;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)sectiondm, &comm));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "dms"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  CHKERRQ(DMGetGlobalSection(sectiondm, &gsection));
  /* Save raw section */
  CHKERRQ(PetscSectionView(gsection, viewer));
  /* Save plex wrapper */
  {
    PetscInt        pStart, pEnd, p, n;
    IS              globalPointNumbers;
    const PetscInt *gpoints;
    IS              orderIS;
    PetscInt       *order;

    CHKERRQ(PetscSectionGetChart(gsection, &pStart, &pEnd));
    CHKERRQ(DMPlexCreatePointNumbering(dm, &globalPointNumbers));
    CHKERRQ(ISGetIndices(globalPointNumbers, &gpoints));
    for (p = pStart, n = 0; p < pEnd; ++p) if (gpoints[p] >= 0) n++;
    /* "order" is an array of global point numbers.
       When loading, it is used with topology/order array
       to match section points with plex topology points. */
    CHKERRQ(PetscMalloc1(n, &order));
    for (p = pStart, n = 0; p < pEnd; ++p) if (gpoints[p] >= 0) order[n++] = gpoints[p];
    CHKERRQ(ISRestoreIndices(globalPointNumbers, &gpoints));
    CHKERRQ(ISDestroy(&globalPointNumbers));
    CHKERRQ(ISCreateGeneral(comm, n, order, PETSC_OWN_POINTER, &orderIS));
    CHKERRQ(PetscObjectSetName((PetscObject)orderIS, "order"));
    CHKERRQ(ISView(orderIS, viewer));
    CHKERRQ(ISDestroy(&orderIS));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGlobalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  const char     *topologydm_name;
  const char     *sectiondm_name;
  const char     *vec_name;
  PetscInt        bs;

  PetscFunctionBegin;
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;

    CHKERRQ(DMGetPointSF(dm, &pointsf));
    CHKERRQ(DMGetPointSF(sectiondm, &pointsf1));
    PetscCheckFalse(pointsf1 != pointsf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)vec, &vec_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "dms"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "vecs"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, vec_name));
  CHKERRQ(VecGetBlockSize(vec, &bs));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *) &bs));
  CHKERRQ(VecSetBlockSize(vec, 1));
  /* VecView(vec, viewer) would call (*vec->opt->view)(vec, viewer), but,    */
  /* if vec was created with DMGet{Global, Local}Vector(), vec->opt->view    */
  /* is set to VecView_Plex, which would save vec in a predefined location.  */
  /* To save vec in where we want, we create a new Vec (temp) with           */
  /* VecCreate(), wrap the vec data in temp, and call VecView(temp, viewer). */
  {
    Vec                temp;
    const PetscScalar *array;
    PetscLayout        map;

    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)vec), &temp));
    CHKERRQ(PetscObjectSetName((PetscObject)temp, vec_name));
    CHKERRQ(VecGetLayout(vec, &map));
    CHKERRQ(VecSetLayout(temp, map));
    CHKERRQ(VecSetUp(temp));
    CHKERRQ(VecGetArrayRead(vec, &array));
    CHKERRQ(VecPlaceArray(temp, array));
    CHKERRQ(VecView(temp, viewer));
    CHKERRQ(VecResetArray(temp));
    CHKERRQ(VecRestoreArrayRead(vec, &array));
    CHKERRQ(VecDestroy(&temp));
  }
  CHKERRQ(VecSetBlockSize(vec, bs));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLocalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  MPI_Comm        comm;
  const char     *topologydm_name;
  const char     *sectiondm_name;
  const char     *vec_name;
  PetscSection    section;
  PetscBool       includesConstraints;
  Vec             gvec;
  PetscInt        m, bs;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;

    CHKERRQ(DMGetPointSF(dm, &pointsf));
    CHKERRQ(DMGetPointSF(sectiondm, &pointsf1));
    PetscCheckFalse(pointsf1 != pointsf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)vec, &vec_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "dms"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "vecs"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, vec_name));
  CHKERRQ(VecGetBlockSize(vec, &bs));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *) &bs));
  CHKERRQ(VecCreate(comm, &gvec));
  CHKERRQ(PetscObjectSetName((PetscObject)gvec, vec_name));
  CHKERRQ(DMGetGlobalSection(sectiondm, &section));
  CHKERRQ(PetscSectionGetIncludesConstraints(section, &includesConstraints));
  if (includesConstraints) CHKERRQ(PetscSectionGetStorageSize(section, &m));
  else CHKERRQ(PetscSectionGetConstrainedStorageSize(section, &m));
  CHKERRQ(VecSetSizes(gvec, m, PETSC_DECIDE));
  CHKERRQ(VecSetUp(gvec));
  CHKERRQ(DMLocalToGlobalBegin(sectiondm, vec, INSERT_VALUES, gvec));
  CHKERRQ(DMLocalToGlobalEnd(sectiondm, vec, INSERT_VALUES, gvec));
  CHKERRQ(VecView(gvec, viewer));
  CHKERRQ(VecDestroy(&gvec));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

struct _n_LoadLabelsCtx {
  MPI_Comm    comm;
  PetscMPIInt rank;
  DM          dm;
  PetscViewer viewer;
  DMLabel     label;
  PetscSF     sfXC;
  PetscLayout layoutX;
};
typedef struct _n_LoadLabelsCtx *LoadLabelsCtx;

static PetscErrorCode LoadLabelsCtxCreate(DM dm, PetscViewer viewer, PetscSF sfXC, LoadLabelsCtx *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNew(ctx));
  CHKERRQ(PetscObjectReference((PetscObject) ((*ctx)->dm = dm)));
  CHKERRQ(PetscObjectReference((PetscObject) ((*ctx)->viewer = viewer)));
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &(*ctx)->comm));
  CHKERRMPI(MPI_Comm_rank((*ctx)->comm, &(*ctx)->rank));
  (*ctx)->sfXC = sfXC;
  if (sfXC) {
    PetscInt nX;

    CHKERRQ(PetscObjectReference((PetscObject) sfXC));
    CHKERRQ(PetscSFGetGraph(sfXC, &nX, NULL, NULL, NULL));
    CHKERRQ(PetscLayoutCreateFromSizes((*ctx)->comm, nX, PETSC_DECIDE, 1, &(*ctx)->layoutX));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadLabelsCtxDestroy(LoadLabelsCtx *ctx)
{
  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(0);
  CHKERRQ(DMDestroy(&(*ctx)->dm));
  CHKERRQ(PetscViewerDestroy(&(*ctx)->viewer));
  CHKERRQ(PetscSFDestroy(&(*ctx)->sfXC));
  CHKERRQ(PetscLayoutDestroy(&(*ctx)->layoutX));
  CHKERRQ(PetscFree(*ctx));
  PetscFunctionReturn(0);
}

/*
    A: on-disk points
    X: global points [0, NX)
    C: distributed plex points
*/
static herr_t ReadLabelStratumHDF5_Distribute_Private(IS stratumIS, LoadLabelsCtx ctx, IS *newStratumIS)
{
  MPI_Comm        comm    = ctx->comm;
  PetscSF         sfXC    = ctx->sfXC;
  PetscLayout     layoutX = ctx->layoutX;
  PetscSF         sfXA;
  const PetscInt *A_points;
  PetscInt        nX, nC;
  PetscInt        n;

  PetscFunctionBegin;
  CHKERRQ(PetscSFGetGraph(sfXC, &nX, &nC, NULL, NULL));
  CHKERRQ(ISGetLocalSize(stratumIS, &n));
  CHKERRQ(ISGetIndices(stratumIS, &A_points));
  CHKERRQ(PetscSFCreate(comm, &sfXA));
  CHKERRQ(PetscSFSetGraphLayout(sfXA, layoutX, n, NULL, PETSC_USE_POINTER, A_points));
  CHKERRQ(ISCreate(comm, newStratumIS));
  CHKERRQ(ISSetType(*newStratumIS,ISGENERAL));
  {
    PetscInt    i;
    PetscBool  *A_mask, *X_mask, *C_mask;

    CHKERRQ(PetscCalloc3(n, &A_mask, nX, &X_mask, nC, &C_mask));
    for (i=0; i<n; i++) A_mask[i] = PETSC_TRUE;
    CHKERRQ(PetscSFReduceBegin(sfXA, MPIU_BOOL, A_mask, X_mask, MPI_REPLACE));
    CHKERRQ(PetscSFReduceEnd(  sfXA, MPIU_BOOL, A_mask, X_mask, MPI_REPLACE));
    CHKERRQ(PetscSFBcastBegin( sfXC, MPIU_BOOL, X_mask, C_mask, MPI_LOR));
    CHKERRQ(PetscSFBcastEnd(   sfXC, MPIU_BOOL, X_mask, C_mask, MPI_LOR));
    CHKERRQ(ISGeneralSetIndicesFromMask(*newStratumIS, 0, nC, C_mask));
    CHKERRQ(PetscFree3(A_mask, X_mask, C_mask));
  }
  CHKERRQ(PetscSFDestroy(&sfXA));
  CHKERRQ(ISRestoreIndices(stratumIS, &A_points));
  PetscFunctionReturn(0);
}

static herr_t ReadLabelStratumHDF5_Static(hid_t g_id, const char *vname, const H5L_info_t *info, void *op_data)
{
  LoadLabelsCtx   ctx    = (LoadLabelsCtx) op_data;
  PetscViewer     viewer = ctx->viewer;
  DMLabel         label  = ctx->label;
  MPI_Comm        comm   = ctx->comm;
  IS              stratumIS;
  const PetscInt *ind;
  PetscInt        value, N, i;

  CHKERRQ(PetscOptionsStringToInt(vname, &value));
  CHKERRQ(ISCreate(comm, &stratumIS));
  CHKERRQ(PetscObjectSetName((PetscObject) stratumIS, "indices"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, vname)); /* labels/<lname>/<vname> */

  if (!ctx->sfXC) {
    /* Force serial load */
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "indices", NULL, &N));
    CHKERRQ(PetscLayoutSetLocalSize(stratumIS->map, !ctx->rank ? N : 0));
    CHKERRQ(PetscLayoutSetSize(stratumIS->map, N));
  }
  CHKERRQ(ISLoad(stratumIS, viewer));

  if (ctx->sfXC) {
    IS newStratumIS;

    CHKERRQ(ReadLabelStratumHDF5_Distribute_Private(stratumIS, ctx, &newStratumIS));
    CHKERRQ(ISDestroy(&stratumIS));
    stratumIS = newStratumIS;
  }

  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(ISGetLocalSize(stratumIS, &N));
  CHKERRQ(ISGetIndices(stratumIS, &ind));
  for (i = 0; i < N; ++i) CHKERRQ(DMLabelSetValue(label, ind[i], value));
  CHKERRQ(ISRestoreIndices(stratumIS, &ind));
  CHKERRQ(ISDestroy(&stratumIS));
  return 0;
}

static herr_t ReadLabelHDF5_Static(hid_t g_id, const char *lname, const H5L_info_t *info, void *op_data)
{
  LoadLabelsCtx  ctx = (LoadLabelsCtx) op_data;
  DM             dm  = ctx->dm;
  hsize_t        idx = 0;
  PetscErrorCode ierr;
  PetscBool      flg;
  herr_t         err;

  CHKERRQ(DMHasLabel(dm, lname, &flg));
  if (flg) CHKERRQ(DMRemoveLabel(dm, lname, NULL));
  ierr = DMCreateLabel(dm, lname); if (ierr) return (herr_t) ierr;
  ierr = DMGetLabel(dm, lname, &ctx->label); if (ierr) return (herr_t) ierr;
  CHKERRQ(PetscViewerHDF5PushGroup(ctx->viewer, lname)); /* labels/<lname> */
  /* Iterate over the label's strata */
  PetscStackCallHDF5Return(err, H5Literate_by_name, (g_id, lname, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelStratumHDF5_Static, op_data, 0));
  CHKERRQ(PetscViewerHDF5PopGroup(ctx->viewer));
  return err;
}

PetscErrorCode DMPlexLabelsLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF sfXC)
{
  const char           *topologydm_name;
  LoadLabelsCtx         ctx;
  hsize_t               idx = 0;
  char                  group[PETSC_MAX_PATH_LEN];
  DMPlexStorageVersion  version;
  PetscBool             distributed, hasGroup;

  PetscFunctionBegin;
  CHKERRQ(DMPlexIsDistributed(dm, &distributed));
  if (distributed) {
    PetscCheckFalse(!sfXC,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "PetscSF must be given for parallel load");
  }
  CHKERRQ(LoadLabelsCtxCreate(dm, viewer, sfXC, &ctx));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(DMPlexStorageVersionGet_Private(dm, viewer, &version));
  if (version.major <= 1) {
    CHKERRQ(PetscStrcpy(group, "labels"));
  } else {
    CHKERRQ(PetscSNPrintf(group, sizeof(group), "topologies/%s/labels", topologydm_name));
  }
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, group));
  CHKERRQ(PetscViewerHDF5HasGroup(viewer, NULL, &hasGroup));
  if (hasGroup) {
    hid_t fileId, groupId;

    CHKERRQ(PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId));
    /* Iterate over labels */
    PetscStackCallHDF5(H5Literate,(groupId, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelHDF5_Static, ctx));
    PetscStackCallHDF5(H5Gclose,(groupId));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(LoadLabelsCtxDestroy(&ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTopologyLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF *sf)
{
  MPI_Comm              comm;
  const char           *topologydm_name;
  const char           *pointsName, *coneSizesName, *conesName, *orientationsName;
  IS                    pointsIS, coneSizesIS, conesIS, orientationsIS;
  const PetscInt       *points, *coneSizes, *cones, *orientations;
  PetscInt             *cone, *ornt;
  PetscInt              dim, N, Np, pEnd, p, q, maxConeSize = 0, c;
  PetscMPIInt           size, rank;
  char                  group[PETSC_MAX_PATH_LEN];
  DMPlexStorageVersion  version;

  PetscFunctionBegin;
  pointsName        = "order";
  coneSizesName     = "cones";
  conesName         = "cells";
  orientationsName  = "orientation";
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(DMPlexStorageVersionGet_Private(dm, viewer, &version));
  if (version.major <= 1) {
    CHKERRQ(PetscStrcpy(group, "/topology"));
  } else {
    CHKERRQ(PetscSNPrintf(group, sizeof(group), "topologies/%s/topology", topologydm_name));
  }
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, group));
  CHKERRQ(ISCreate(comm, &pointsIS));
  CHKERRQ(PetscObjectSetName((PetscObject) pointsIS, pointsName));
  CHKERRQ(ISCreate(comm, &coneSizesIS));
  CHKERRQ(PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName));
  CHKERRQ(ISCreate(comm, &conesIS));
  CHKERRQ(PetscObjectSetName((PetscObject) conesIS, conesName));
  CHKERRQ(ISCreate(comm, &orientationsIS));
  CHKERRQ(PetscObjectSetName((PetscObject) orientationsIS, orientationsName));
  CHKERRQ(PetscViewerHDF5ReadObjectAttribute(viewer, (PetscObject) conesIS, "cell_dim", PETSC_INT, NULL, &dim));
  CHKERRQ(DMSetDimension(dm, dim));
  {
    /* Force serial load */
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, pointsName, NULL, &Np));
    CHKERRQ(PetscLayoutSetLocalSize(pointsIS->map, rank == 0 ? Np : 0));
    CHKERRQ(PetscLayoutSetSize(pointsIS->map, Np));
    pEnd = rank == 0 ? Np : 0;
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, coneSizesName, NULL, &Np));
    CHKERRQ(PetscLayoutSetLocalSize(coneSizesIS->map, rank == 0 ? Np : 0));
    CHKERRQ(PetscLayoutSetSize(coneSizesIS->map, Np));
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, conesName, NULL, &N));
    CHKERRQ(PetscLayoutSetLocalSize(conesIS->map, rank == 0 ? N : 0));
    CHKERRQ(PetscLayoutSetSize(conesIS->map, N));
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, orientationsName, NULL, &N));
    CHKERRQ(PetscLayoutSetLocalSize(orientationsIS->map, rank == 0 ? N : 0));
    CHKERRQ(PetscLayoutSetSize(orientationsIS->map, N));
  }
  CHKERRQ(ISLoad(pointsIS, viewer));
  CHKERRQ(ISLoad(coneSizesIS, viewer));
  CHKERRQ(ISLoad(conesIS, viewer));
  CHKERRQ(ISLoad(orientationsIS, viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  /* Create Plex */
  CHKERRQ(DMPlexSetChart(dm, 0, pEnd));
  CHKERRQ(ISGetIndices(pointsIS, &points));
  CHKERRQ(ISGetIndices(coneSizesIS, &coneSizes));
  for (p = 0; p < pEnd; ++p) {
    CHKERRQ(DMPlexSetConeSize(dm, points[p], coneSizes[p]));
    maxConeSize = PetscMax(maxConeSize, coneSizes[p]);
  }
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(ISGetIndices(conesIS, &cones));
  CHKERRQ(ISGetIndices(orientationsIS, &orientations));
  CHKERRQ(PetscMalloc2(maxConeSize,&cone,maxConeSize,&ornt));
  for (p = 0, q = 0; p < pEnd; ++p) {
    for (c = 0; c < coneSizes[p]; ++c, ++q) {
      cone[c] = cones[q];
      ornt[c] = orientations[q];
    }
    CHKERRQ(DMPlexSetCone(dm, points[p], cone));
    CHKERRQ(DMPlexSetConeOrientation(dm, points[p], ornt));
  }
  CHKERRQ(PetscFree2(cone,ornt));
  /* Create global section migration SF */
  if (sf) {
    PetscLayout  layout;
    PetscInt    *globalIndices;

    CHKERRQ(PetscMalloc1(pEnd, &globalIndices));
    /* plex point == globalPointNumber in this case */
    for (p = 0; p < pEnd; ++p) globalIndices[p] = p;
    CHKERRQ(PetscLayoutCreate(comm, &layout));
    CHKERRQ(PetscLayoutSetSize(layout, Np));
    CHKERRQ(PetscLayoutSetBlockSize(layout, 1));
    CHKERRQ(PetscLayoutSetUp(layout));
    CHKERRQ(PetscSFCreate(comm, sf));
    CHKERRQ(PetscSFSetFromOptions(*sf));
    CHKERRQ(PetscSFSetGraphLayout(*sf, layout, pEnd, NULL, PETSC_OWN_POINTER, globalIndices));
    CHKERRQ(PetscLayoutDestroy(&layout));
    CHKERRQ(PetscFree(globalIndices));
  }
  /* Clean-up */
  CHKERRQ(ISRestoreIndices(pointsIS, &points));
  CHKERRQ(ISRestoreIndices(coneSizesIS, &coneSizes));
  CHKERRQ(ISRestoreIndices(conesIS, &cones));
  CHKERRQ(ISRestoreIndices(orientationsIS, &orientations));
  CHKERRQ(ISDestroy(&pointsIS));
  CHKERRQ(ISDestroy(&coneSizesIS));
  CHKERRQ(ISDestroy(&conesIS));
  CHKERRQ(ISDestroy(&orientationsIS));
  /* Fill in the rest of the topology structure */
  CHKERRQ(DMPlexSymmetrize(dm));
  CHKERRQ(DMPlexStratify(dm));
  PetscFunctionReturn(0);
}

/* If the file is old, it not only has different path to the coordinates, but   */
/* does not contain coordinateDMs, so must fall back to the old implementation. */
static PetscErrorCode DMPlexCoordinatesLoad_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
{
  PetscSection    coordSection;
  Vec             coordinates;
  PetscReal       lengthScale;
  PetscInt        spatialDim, N, numVertices, vStart, vEnd, v;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  /* Read geometry */
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/geometry"));
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject) dm), &coordinates));
  CHKERRQ(PetscObjectSetName((PetscObject) coordinates, "vertices"));
  {
    /* Force serial load */
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "vertices", &spatialDim, &N));
    CHKERRQ(VecSetSizes(coordinates, !rank ? N : 0, N));
    CHKERRQ(VecSetBlockSize(coordinates, spatialDim));
  }
  CHKERRQ(VecLoad(coordinates, viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  CHKERRQ(VecScale(coordinates, 1.0/lengthScale));
  CHKERRQ(VecGetLocalSize(coordinates, &numVertices));
  CHKERRQ(VecGetBlockSize(coordinates, &spatialDim));
  numVertices /= spatialDim;
  /* Create coordinates */
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCheckFalse(numVertices != vEnd - vStart,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of coordinates loaded %d does not match number of vertices %d", numVertices, vEnd - vStart);
  CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
  CHKERRQ(PetscSectionSetNumFields(coordSection, 1));
  CHKERRQ(PetscSectionSetFieldComponents(coordSection, 0, spatialDim));
  CHKERRQ(PetscSectionSetChart(coordSection, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    CHKERRQ(PetscSectionSetDof(coordSection, v, spatialDim));
    CHKERRQ(PetscSectionSetFieldDof(coordSection, v, 0, spatialDim));
  }
  CHKERRQ(PetscSectionSetUp(coordSection));
  CHKERRQ(DMSetCoordinates(dm, coordinates));
  CHKERRQ(VecDestroy(&coordinates));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCoordinatesLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF sfXC)
{
  DM                    cdm;
  Vec                   coords;
  PetscInt              blockSize;
  PetscReal             lengthScale;
  PetscSF               lsf;
  const char           *topologydm_name;
  char                 *coordinatedm_name, *coordinates_name;

  PetscFunctionBegin;
  {
    DMPlexStorageVersion  version;

    CHKERRQ(DMPlexStorageVersionGet_Private(dm, viewer, &version));
    if (version.major <= 1) {
      CHKERRQ(DMPlexCoordinatesLoad_HDF5_Legacy_Private(dm, viewer));
      PetscFunctionReturn(0);
    }
  }
  PetscCheckFalse(!sfXC,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "PetscSF must be given for parallel load");
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "coordinateDMName", PETSC_STRING , NULL, &coordinatedm_name));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "coordinatesName", PETSC_STRING , NULL, &coordinates_name));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(PetscObjectSetName((PetscObject)cdm, coordinatedm_name));
  CHKERRQ(PetscFree(coordinatedm_name));
  /* lsf: on-disk data -> in-memory local vector associated with cdm's local section */
  CHKERRQ(DMPlexSectionLoad(dm, viewer, cdm, sfXC, NULL, &lsf));
  CHKERRQ(DMCreateLocalVector(cdm, &coords));
  CHKERRQ(PetscObjectSetName((PetscObject)coords, coordinates_name));
  CHKERRQ(PetscFree(coordinates_name));
  CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  CHKERRQ(DMPlexLocalVectorLoad(dm, viewer, cdm, lsf, coords));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  CHKERRQ(VecScale(coords, 1.0/lengthScale));
  CHKERRQ(DMSetCoordinatesLocal(dm, coords));
  CHKERRQ(VecGetBlockSize(coords, &blockSize));
  CHKERRQ(DMSetCoordinateDim(dm, blockSize));
  CHKERRQ(VecDestroy(&coords));
  CHKERRQ(PetscSFDestroy(&lsf));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLoad_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
{
  PetscFunctionBegin;
  CHKERRQ(DMPlexTopologyLoad_HDF5_Internal(dm, viewer, NULL));
  CHKERRQ(DMPlexLabelsLoad_HDF5_Internal(dm, viewer, NULL));
  CHKERRQ(DMPlexCoordinatesLoad_HDF5_Legacy_Private(dm, viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLoad_HDF5_Internal(DM dm, PetscViewer viewer)
{
  PetscSF               sfXC;

  PetscFunctionBegin;
  {
    DMPlexStorageVersion  version;

    CHKERRQ(DMPlexStorageVersionGet_Private(dm, viewer, &version));
    if (version.major <= 1) {
      CHKERRQ(DMPlexLoad_HDF5_Legacy_Private(dm, viewer));
      PetscFunctionReturn(0);
    }
  }
  CHKERRQ(DMPlexTopologyLoad_HDF5_Internal(dm, viewer, &sfXC));
  CHKERRQ(DMPlexLabelsLoad_HDF5_Internal(dm, viewer, sfXC));
  CHKERRQ(DMPlexCoordinatesLoad_HDF5_Internal(dm, viewer, sfXC));
  CHKERRQ(PetscSFDestroy(&sfXC));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSectionLoad_HDF5_Internal_CreateDataSF(PetscSection rootSection, PetscLayout layout, PetscInt globalOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm        comm;
  PetscInt        pStart, pEnd, p, m;
  PetscInt       *goffs, *ilocal;
  PetscBool       rootIncludeConstraints, leafIncludeConstraints;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)leafSection, &comm));
  CHKERRQ(PetscSectionGetChart(leafSection, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetIncludesConstraints(rootSection, &rootIncludeConstraints));
  CHKERRQ(PetscSectionGetIncludesConstraints(leafSection, &leafIncludeConstraints));
  if (rootIncludeConstraints && leafIncludeConstraints) CHKERRQ(PetscSectionGetStorageSize(leafSection, &m));
  else CHKERRQ(PetscSectionGetConstrainedStorageSize(leafSection, &m));
  CHKERRQ(PetscMalloc1(m, &ilocal));
  CHKERRQ(PetscMalloc1(m, &goffs));
  /* Currently, PetscSFDistributeSection() returns globalOffsets[] only */
  /* for the top-level section (not for each field), so one must have   */
  /* rootSection->pointMajor == PETSC_TRUE.                             */
  PetscCheckFalse(!rootSection->pointMajor,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for field major ordering");
  /* Currently, we also assume that leafSection->pointMajor == PETSC_TRUE. */
  PetscCheckFalse(!leafSection->pointMajor,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for field major ordering");
  for (p = pStart, m = 0; p < pEnd; ++p) {
    PetscInt        dof, cdof, i, j, off, goff;
    const PetscInt *cinds;

    CHKERRQ(PetscSectionGetDof(leafSection, p, &dof));
    if (dof < 0) continue;
    goff = globalOffsets[p-pStart];
    CHKERRQ(PetscSectionGetOffset(leafSection, p, &off));
    CHKERRQ(PetscSectionGetConstraintDof(leafSection, p, &cdof));
    CHKERRQ(PetscSectionGetConstraintIndices(leafSection, p, &cinds));
    for (i = 0, j = 0; i < dof; ++i) {
      PetscBool constrained = (PetscBool) (j < cdof && i == cinds[j]);

      if (!constrained || (leafIncludeConstraints && rootIncludeConstraints)) {ilocal[m] = off++; goffs[m++] = goff++;}
      else if (leafIncludeConstraints && !rootIncludeConstraints) ++off;
      else if (!leafIncludeConstraints &&  rootIncludeConstraints) ++goff;
      if (constrained) ++j;
    }
  }
  CHKERRQ(PetscSFCreate(comm, sectionSF));
  CHKERRQ(PetscSFSetFromOptions(*sectionSF));
  CHKERRQ(PetscSFSetGraphLayout(*sectionSF, layout, m, ilocal, PETSC_OWN_POINTER, goffs));
  CHKERRQ(PetscFree(goffs));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexSectionLoad_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sfXB, PetscSF *gsf, PetscSF *lsf)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  const char    *topologydm_name;
  const char    *sectiondm_name;
  PetscSection   sectionA, sectionB;
  PetscInt       nX, n, i;
  PetscSF        sfAB;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "dms"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  /* A: on-disk points                        */
  /* X: list of global point numbers, [0, NX) */
  /* B: plex points                           */
  /* Load raw section (sectionA)              */
  CHKERRQ(PetscSectionCreate(comm, &sectionA));
  CHKERRQ(PetscSectionLoad(sectionA, viewer));
  CHKERRQ(PetscSectionGetChart(sectionA, NULL, &n));
  /* Create sfAB: A -> B */
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt  N, N1;

    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "order", NULL, &N1));
    CHKERRMPI(MPI_Allreduce(&n, &N, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Mismatching sizes: on-disk order array size (%D) != number of loaded section points (%D)", N1, N);
  }
#endif
  {
    IS              orderIS;
    const PetscInt *gpoints;
    PetscSF         sfXA, sfAX;
    PetscLayout     layout;
    PetscSFNode    *owners, *buffer;
    PetscInt        nleaves;
    PetscInt       *ilocal;
    PetscSFNode    *iremote;

    /* Create sfAX: A -> X */
    CHKERRQ(ISCreate(comm, &orderIS));
    CHKERRQ(PetscObjectSetName((PetscObject)orderIS, "order"));
    CHKERRQ(PetscLayoutSetLocalSize(orderIS->map, n));
    CHKERRQ(ISLoad(orderIS, viewer));
    CHKERRQ(PetscLayoutCreate(comm, &layout));
    CHKERRQ(PetscSFGetGraph(sfXB, &nX, NULL, NULL, NULL));
    CHKERRQ(PetscLayoutSetLocalSize(layout, nX));
    CHKERRQ(PetscLayoutSetBlockSize(layout, 1));
    CHKERRQ(PetscLayoutSetUp(layout));
    CHKERRQ(PetscSFCreate(comm, &sfXA));
    CHKERRQ(ISGetIndices(orderIS, &gpoints));
    CHKERRQ(PetscSFSetGraphLayout(sfXA, layout, n, NULL, PETSC_OWN_POINTER, gpoints));
    CHKERRQ(ISRestoreIndices(orderIS, &gpoints));
    CHKERRQ(ISDestroy(&orderIS));
    CHKERRQ(PetscLayoutDestroy(&layout));
    CHKERRQ(PetscMalloc1(n, &owners));
    CHKERRQ(PetscMalloc1(nX, &buffer));
    for (i = 0; i < n; ++i) {owners[i].rank = rank; owners[i].index = i;}
    for (i = 0; i < nX; ++i) {buffer[i].rank = -1; buffer[i].index = -1;}
    CHKERRQ(PetscSFReduceBegin(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC));
    CHKERRQ(PetscSFReduceEnd(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC));
    CHKERRQ(PetscSFDestroy(&sfXA));
    CHKERRQ(PetscFree(owners));
    for (i = 0, nleaves = 0; i < nX; ++i) if (buffer[i].rank >= 0) nleaves++;
    CHKERRQ(PetscMalloc1(nleaves, &ilocal));
    CHKERRQ(PetscMalloc1(nleaves, &iremote));
    for (i = 0, nleaves = 0; i < nX; ++i) {
      if (buffer[i].rank >= 0) {
        ilocal[nleaves] = i;
        iremote[nleaves].rank = buffer[i].rank;
        iremote[nleaves].index = buffer[i].index;
        nleaves++;
      }
    }
    CHKERRQ(PetscSFCreate(comm, &sfAX));
    CHKERRQ(PetscSFSetFromOptions(sfAX));
    CHKERRQ(PetscSFSetGraph(sfAX, n, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    /* Fix PetscSFCompose() and replace the code-block below with:  */
    /* CHKERRQ(PetscSFCompose(sfAX, sfXB, &sfAB));      */
    /* which currently causes segmentation fault due to sparse map. */
    {
      PetscInt     npoints;
      PetscInt     mleaves;
      PetscInt    *jlocal;
      PetscSFNode *jremote;

      CHKERRQ(PetscSFGetGraph(sfXB, NULL, &npoints, NULL, NULL));
      CHKERRQ(PetscMalloc1(npoints, &owners));
      for (i = 0; i < npoints; ++i) {owners[i].rank = -1; owners[i].index = -1;}
      CHKERRQ(PetscSFBcastBegin(sfXB, MPIU_2INT, buffer, owners, MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(sfXB, MPIU_2INT, buffer, owners, MPI_REPLACE));
      for (i = 0, mleaves = 0; i < npoints; ++i) if (owners[i].rank >= 0) mleaves++;
      CHKERRQ(PetscMalloc1(mleaves, &jlocal));
      CHKERRQ(PetscMalloc1(mleaves, &jremote));
      for (i = 0, mleaves = 0; i < npoints; ++i) {
        if (owners[i].rank >= 0) {
          jlocal[mleaves] = i;
          jremote[mleaves].rank = owners[i].rank;
          jremote[mleaves].index = owners[i].index;
          mleaves++;
        }
      }
      CHKERRQ(PetscSFCreate(comm, &sfAB));
      CHKERRQ(PetscSFSetFromOptions(sfAB));
      CHKERRQ(PetscSFSetGraph(sfAB, n, mleaves, jlocal, PETSC_OWN_POINTER, jremote, PETSC_OWN_POINTER));
      CHKERRQ(PetscFree(owners));
    }
    CHKERRQ(PetscFree(buffer));
    CHKERRQ(PetscSFDestroy(&sfAX));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  /* Create plex section (sectionB) */
  CHKERRQ(DMGetLocalSection(sectiondm, &sectionB));
  if (lsf || gsf) {
    PetscLayout  layout;
    PetscInt     M, m;
    PetscInt    *offsetsA;
    PetscBool    includesConstraintsA;

    CHKERRQ(PetscSFDistributeSection(sfAB, sectionA, &offsetsA, sectionB));
    CHKERRQ(PetscSectionGetIncludesConstraints(sectionA, &includesConstraintsA));
    if (includesConstraintsA) CHKERRQ(PetscSectionGetStorageSize(sectionA, &m));
    else CHKERRQ(PetscSectionGetConstrainedStorageSize(sectionA, &m));
    CHKERRMPI(MPI_Allreduce(&m, &M, 1, MPIU_INT, MPI_SUM, comm));
    CHKERRQ(PetscLayoutCreate(comm, &layout));
    CHKERRQ(PetscLayoutSetSize(layout, M));
    CHKERRQ(PetscLayoutSetUp(layout));
    if (lsf) {
      PetscSF lsfABdata;

      CHKERRQ(DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, sectionB, &lsfABdata));
      *lsf = lsfABdata;
    }
    if (gsf) {
      PetscSection  gsectionB, gsectionB1;
      PetscBool     includesConstraintsB;
      PetscSF       gsfABdata, pointsf;

      CHKERRQ(DMGetGlobalSection(sectiondm, &gsectionB1));
      CHKERRQ(PetscSectionGetIncludesConstraints(gsectionB1, &includesConstraintsB));
      CHKERRQ(DMGetPointSF(sectiondm, &pointsf));
      CHKERRQ(PetscSectionCreateGlobalSection(sectionB, pointsf, includesConstraintsB, PETSC_TRUE, &gsectionB));
      CHKERRQ(DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, gsectionB, &gsfABdata));
      CHKERRQ(PetscSectionDestroy(&gsectionB));
      *gsf = gsfABdata;
    }
    CHKERRQ(PetscLayoutDestroy(&layout));
    CHKERRQ(PetscFree(offsetsA));
  } else {
    CHKERRQ(PetscSFDistributeSection(sfAB, sectionA, NULL, sectionB));
  }
  CHKERRQ(PetscSFDestroy(&sfAB));
  CHKERRQ(PetscSectionDestroy(&sectionA));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexVecLoad_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sf, Vec vec)
{
  MPI_Comm           comm;
  const char        *topologydm_name;
  const char        *sectiondm_name;
  const char        *vec_name;
  Vec                vecA;
  PetscInt           mA, m, bs;
  const PetscInt    *ilocal;
  const PetscScalar *src;
  PetscScalar       *dest;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRQ(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  CHKERRQ(PetscObjectGetName((PetscObject)vec, &vec_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "topologies"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "dms"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "vecs"));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, vec_name));
  CHKERRQ(VecCreate(comm, &vecA));
  CHKERRQ(PetscObjectSetName((PetscObject)vecA, vec_name));
  CHKERRQ(PetscSFGetGraph(sf, &mA, &m, &ilocal, NULL));
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;
    PetscInt  m1, i, j;

    CHKERRQ(DMGetPointSF(dm, &pointsf));
    CHKERRQ(DMGetPointSF(sectiondm, &pointsf1));
    PetscCheckFalse(pointsf1 != pointsf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt  MA, MA1;

      CHKERRMPI(MPIU_Allreduce(&mA, &MA, 1, MPIU_INT, MPI_SUM, comm));
      CHKERRQ(PetscViewerHDF5ReadSizes(viewer, vec_name, NULL, &MA1));
      PetscCheckFalse(MA1 != MA,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total SF root size (%D) != On-disk vector data size (%D)", MA, MA1);
    }
#endif
    CHKERRQ(VecGetLocalSize(vec, &m1));
    PetscCheckFalse(m1 < m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Target vector size (%D) < SF leaf size (%D)", m1, m);
    for (i = 0; i < m; ++i) {
      j = ilocal ? ilocal[i] : i;
      PetscCheckFalse(j < 0 || j >= m1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Leaf's %D-th index, %D, not in [%D, %D)", i, j, 0, m1);
    }
  }
  CHKERRQ(VecSetSizes(vecA, mA, PETSC_DECIDE));
  CHKERRQ(VecLoad(vecA, viewer));
  CHKERRQ(VecGetArrayRead(vecA, &src));
  CHKERRQ(VecGetArray(vec, &dest));
  CHKERRQ(PetscSFBcastBegin(sf, MPIU_SCALAR, src, dest, MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPIU_SCALAR, src, dest, MPI_REPLACE));
  CHKERRQ(VecRestoreArray(vec, &dest));
  CHKERRQ(VecRestoreArrayRead(vecA, &src));
  CHKERRQ(VecDestroy(&vecA));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "blockSize", PETSC_INT, NULL, (void *) &bs));
  CHKERRQ(VecSetBlockSize(vec, bs));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}
#endif
