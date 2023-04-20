#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>

/* Logging support */
PetscLogEvent DMPLEX_DistributionView, DMPLEX_DistributionLoad;

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode PetscViewerParseVersion_Private(PetscViewer, const char[], DMPlexStorageVersion *);
static PetscErrorCode PetscViewerCheckVersion_Private(PetscViewer, DMPlexStorageVersion);
static PetscErrorCode PetscViewerAttachVersion_Private(PetscViewer, const char[], DMPlexStorageVersion);
static PetscErrorCode PetscViewerGetAttachedVersion_Private(PetscViewer, const char[], DMPlexStorageVersion *);

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

static PetscErrorCode PetscViewerParseVersion_Private(PetscViewer viewer, const char str[], DMPlexStorageVersion *version)
{
  PetscToken           t;
  char                *ts;
  PetscInt             i;
  PetscInt             ti[3];
  DMPlexStorageVersion v;

  PetscFunctionBegin;
  PetscCall(PetscTokenCreate(str, '.', &t));
  for (i = 0; i < 3; i++) {
    PetscCall(PetscTokenFind(t, &ts));
    PetscCheck(ts, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Malformed version string %s", str);
    PetscCall(PetscOptionsStringToInt(ts, &ti[i]));
  }
  PetscCall(PetscTokenFind(t, &ts));
  PetscCheck(!ts, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Malformed version string %s", str);
  PetscCall(PetscTokenDestroy(&t));
  PetscCall(PetscNew(&v));
  v->major    = ti[0];
  v->minor    = ti[1];
  v->subminor = ti[2];
  PetscCall(PetscViewerCheckVersion_Private(viewer, v));
  *version = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerAttachVersion_Private(PetscViewer viewer, const char key[], DMPlexStorageVersion v)
{
  PetscContainer cont;

  PetscFunctionBegin;
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)viewer), &cont));
  PetscCall(PetscContainerSetPointer(cont, v));
  PetscCall(PetscContainerSetUserDestroy(cont, PetscContainerUserDestroyDefault));
  PetscCall(PetscObjectCompose((PetscObject)viewer, key, (PetscObject)cont));
  PetscCall(PetscContainerDestroy(&cont));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerGetAttachedVersion_Private(PetscViewer viewer, const char key[], DMPlexStorageVersion *v)
{
  PetscContainer cont;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)viewer, key, (PetscObject *)&cont));
  *v = NULL;
  if (cont) PetscCall(PetscContainerGetPointer(cont, (void **)v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Version log:
  1.0.0 legacy version (default if no "dmplex_storage_version" attribute found in file)
  2.0.0 introduce versioning and multiple topologies
  2.1.0 introduce distributions
*/
static PetscErrorCode PetscViewerCheckVersion_Private(PetscViewer viewer, DMPlexStorageVersion version)
{
  PetscBool valid = PETSC_FALSE;

  PetscFunctionBegin;
  switch (version->major) {
  case 1:
    switch (version->minor) {
    case 0:
      switch (version->subminor) {
      case 0:
        valid = PETSC_TRUE;
        break;
      };
      break;
    };
    break;
  case 2:
    switch (version->minor) {
    case 0:
      switch (version->subminor) {
      case 0:
        valid = PETSC_TRUE;
        break;
      };
      break;
    case 1:
      switch (version->subminor) {
      case 0:
        valid = PETSC_TRUE;
        break;
      };
      break;
    };
    break;
  case 3:
    switch (version->minor) {
    case 0:
      switch (version->subminor) {
      case 0:
        valid = PETSC_TRUE;
        break;
      };
      break;
    };
    break;
  }
  PetscCheck(valid, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "DMPlexStorageVersion %d.%d.%d not supported", version->major, version->minor, version->subminor);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscBool DMPlexStorageVersionGE(DMPlexStorageVersion version, int major, int minor, int subminor)
{
  return (PetscBool)((version->major == major && version->minor == minor && version->subminor >= subminor) || (version->major == major && version->minor > minor) || (version->major > major));
}

PetscErrorCode PetscViewerHDF5GetDMPlexStorageVersionWriting(PetscViewer viewer, DMPlexStorageVersion *version)
{
  const char ATTR_NAME[] = "dmplex_storage_version";
  PetscBool  fileHasVersion;
  char       optVersion[16], fileVersion[16];

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERHDF5);
  PetscValidPointer(version, 2);
  PetscCall(PetscViewerGetAttachedVersion_Private(viewer, DMPLEX_STORAGE_VERSION_WRITING_KEY, version));
  if (*version) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscStrncpy(fileVersion, DMPLEX_STORAGE_VERSION_STABLE, sizeof(fileVersion)));
  PetscCall(PetscViewerHDF5HasAttribute(viewer, NULL, ATTR_NAME, &fileHasVersion));
  if (fileHasVersion) {
    char *tmp;

    PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", ATTR_NAME, PETSC_STRING, NULL, &tmp));
    PetscCall(PetscStrncpy(fileVersion, tmp, sizeof(fileVersion)));
    PetscCall(PetscFree(tmp));
  }
  PetscCall(PetscStrncpy(optVersion, fileVersion, sizeof(optVersion)));
  PetscObjectOptionsBegin((PetscObject)viewer);
  PetscCall(PetscOptionsString("-dm_plex_view_hdf5_storage_version", "DMPlex HDF5 viewer storage version", NULL, optVersion, optVersion, sizeof(optVersion), NULL));
  PetscOptionsEnd();
  if (!fileHasVersion) {
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, optVersion));
  } else {
    PetscBool flg;

    PetscCall(PetscStrcmp(fileVersion, optVersion, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "User requested DMPlex storage version %s but file already has version %s - cannot mix versions", optVersion, fileVersion);
  }
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "petsc_version_git", PETSC_STRING, PETSC_VERSION_GIT));
  PetscCall(PetscViewerParseVersion_Private(viewer, optVersion, version));
  PetscCall(PetscViewerAttachVersion_Private(viewer, DMPLEX_STORAGE_VERSION_WRITING_KEY, *version));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerHDF5GetDMPlexStorageVersionReading(PetscViewer viewer, DMPlexStorageVersion *version)
{
  const char ATTR_NAME[] = "dmplex_storage_version";
  char      *defaultVersion;
  char      *versionString;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERHDF5);
  PetscValidPointer(version, 2);
  PetscCall(PetscViewerGetAttachedVersion_Private(viewer, DMPLEX_STORAGE_VERSION_READING_KEY, version));
  if (*version) PetscFunctionReturn(PETSC_SUCCESS);

  //TODO string HDF5 attribute handling is terrible and should be redesigned
  PetscCall(PetscStrallocpy(DMPLEX_STORAGE_VERSION_FIRST, &defaultVersion));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/", ATTR_NAME, PETSC_STRING, &defaultVersion, &versionString));
  PetscCall(PetscViewerParseVersion_Private(viewer, versionString, version));
  PetscCall(PetscViewerAttachVersion_Private(viewer, DMPLEX_STORAGE_VERSION_READING_KEY, *version));
  PetscCall(PetscFree(versionString));
  PetscCall(PetscFree(defaultVersion));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexGetHDF5Name_Private(DM dm, const char *name[])
{
  PetscFunctionBegin;
  if (((PetscObject)dm)->name) {
    PetscCall(PetscObjectGetName((PetscObject)dm, name));
  } else {
    *name = "plex";
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSequenceView_HDF5(DM dm, const char *seqname, PetscInt seqnum, PetscScalar value, PetscViewer viewer)
{
  Vec         stamp;
  PetscMPIInt rank;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)viewer), rank ? 0 : 1, 1, &stamp));
  PetscCall(VecSetBlockSize(stamp, 1));
  PetscCall(PetscObjectSetName((PetscObject)stamp, seqname));
  if (rank == 0) {
    PetscReal timeScale;
    PetscBool istime;

    PetscCall(PetscStrncmp(seqname, "time", 5, &istime));
    if (istime) {
      PetscCall(DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale));
      value *= timeScale;
    }
    PetscCall(VecSetValue(stamp, 0, value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(stamp));
  PetscCall(VecAssemblyEnd(stamp));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
  PetscCall(PetscViewerHDF5PushTimestepping(viewer));
  PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum)); /* seqnum < 0 jumps out above */
  PetscCall(VecView(stamp, viewer));
  PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(VecDestroy(&stamp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSequenceLoad_HDF5_Internal(DM dm, const char *seqname, PetscInt seqnum, PetscScalar *value, PetscViewer viewer)
{
  Vec         stamp;
  PetscMPIInt rank;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)viewer), rank ? 0 : 1, 1, &stamp));
  PetscCall(VecSetBlockSize(stamp, 1));
  PetscCall(PetscObjectSetName((PetscObject)stamp, seqname));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
  PetscCall(PetscViewerHDF5PushTimestepping(viewer));
  PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum)); /* seqnum < 0 jumps out above */
  PetscCall(VecLoad(stamp, viewer));
  PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  if (rank == 0) {
    const PetscScalar *a;
    PetscReal          timeScale;
    PetscBool          istime;

    PetscCall(VecGetArrayRead(stamp, &a));
    *value = a[0];
    PetscCall(VecRestoreArrayRead(stamp, &a));
    PetscCall(PetscStrncmp(seqname, "time", 5, &istime));
    if (istime) {
      PetscCall(DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale));
      *value /= timeScale;
    }
  }
  PetscCall(VecDestroy(&stamp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCreateCutVertexLabel_Private(DM dm, DMLabel cutLabel, DMLabel *cutVertexLabel)
{
  IS              cutcells = NULL;
  const PetscInt *cutc;
  PetscInt        cellHeight, vStart, vEnd, cStart, cEnd, c;

  PetscFunctionBegin;
  if (!cutLabel) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  /* Label vertices that should be duplicated */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Cut Vertices", cutVertexLabel));
  PetscCall(DMLabelGetStratumIS(cutLabel, 2, &cutcells));
  if (cutcells) {
    PetscInt n;

    PetscCall(ISGetIndices(cutcells, &cutc));
    PetscCall(ISGetLocalSize(cutcells, &n));
    for (c = 0; c < n; ++c) {
      if ((cutc[c] >= cStart) && (cutc[c] < cEnd)) {
        PetscInt *closure = NULL;
        PetscInt  closureSize, cl, value;

        PetscCall(DMPlexGetTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure));
        for (cl = 0; cl < closureSize * 2; cl += 2) {
          if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) {
            PetscCall(DMLabelGetValue(cutLabel, closure[cl], &value));
            if (value == 1) PetscCall(DMLabelSetValue(*cutVertexLabel, closure[cl], 1));
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure));
      }
    }
    PetscCall(ISRestoreIndices(cutcells, &cutc));
  }
  PetscCall(ISDestroy(&cutcells));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectTypeCompare((PetscObject)v, VECSEQ, &isseq));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetOutputSequenceNumber(dm, &seqnum, &seqval));
  PetscCall(DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar)seqval, viewer));
  if (seqnum >= 0) {
    PetscCall(PetscViewerHDF5PushTimestepping(viewer));
    PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum));
  }
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(DMGetOutputDM(dm, &dmBC));
  PetscCall(DMGetGlobalSection(dmBC, &sectionGlobal));
  PetscCall(DMGetGlobalVector(dmBC, &gv));
  PetscCall(PetscObjectGetName((PetscObject)v, &name));
  PetscCall(PetscObjectSetName((PetscObject)gv, name));
  PetscCall(DMLocalToGlobalBegin(dmBC, v, INSERT_VALUES, gv));
  PetscCall(DMLocalToGlobalEnd(dmBC, v, INSERT_VALUES, gv));
  PetscCall(PetscObjectTypeCompare((PetscObject)gv, VECSEQ, &isseq));
  if (isseq) PetscCall(VecView_Seq(gv, viewer));
  else PetscCall(VecView_MPI(gv, viewer));
  if (format == PETSC_VIEWER_HDF5_VIZ) {
    /* Output visualization representation */
    PetscInt numFields, f;
    DMLabel  cutLabel, cutVertexLabel = NULL;

    PetscCall(PetscSectionGetNumFields(section, &numFields));
    PetscCall(DMGetLabel(dm, "periodic_cut", &cutLabel));
    for (f = 0; f < numFields; ++f) {
      Vec         subv;
      IS          is;
      const char *fname, *fgroup, *componentName;
      char        subname[PETSC_MAX_PATH_LEN];
      PetscInt    pStart, pEnd, Nc, c;

      PetscCall(DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft));
      fgroup = (ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_POINT_FIELD) ? "/vertex_fields" : "/cell_fields";
      PetscCall(PetscSectionGetFieldName(section, f, &fname));
      if (!fname || ft == PETSC_VTK_INVALID) continue;
      PetscCall(PetscViewerHDF5PushGroup(viewer, fgroup));
      if (cutLabel) {
        const PetscScalar *ga;
        PetscScalar       *suba;
        PetscInt           gstart, subSize = 0, extSize = 0, subOff = 0, newOff = 0, p;

        PetscCall(DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel));
        PetscCall(PetscSectionGetFieldComponents(section, f, &Nc));
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, fdof = 0, val;

          PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
          if (gdof > 0) PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
          subSize += fdof;
          PetscCall(DMLabelGetValue(cutVertexLabel, p, &val));
          if (val == 1) extSize += fdof;
        }
        PetscCall(VecCreate(PetscObjectComm((PetscObject)gv), &subv));
        PetscCall(VecSetSizes(subv, subSize + extSize, PETSC_DETERMINE));
        PetscCall(VecSetBlockSize(subv, Nc));
        PetscCall(VecSetType(subv, VECSTANDARD));
        PetscCall(VecGetOwnershipRange(gv, &gstart, NULL));
        PetscCall(VecGetArrayRead(gv, &ga));
        PetscCall(VecGetArray(subv, &suba));
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, goff, val;

          PetscCall(PetscSectionGetDof(sectionGlobal, p, &gdof));
          if (gdof > 0) {
            PetscInt fdof, fc, f2, poff = 0;

            PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
            /* Can get rid of this loop by storing field information in the global section */
            for (f2 = 0; f2 < f; ++f2) {
              PetscCall(PetscSectionGetFieldDof(section, p, f2, &fdof));
              poff += fdof;
            }
            PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
            for (fc = 0; fc < fdof; ++fc, ++subOff) suba[subOff] = ga[goff + poff + fc - gstart];
            PetscCall(DMLabelGetValue(cutVertexLabel, p, &val));
            if (val == 1) {
              for (fc = 0; fc < fdof; ++fc, ++newOff) suba[subSize + newOff] = ga[goff + poff + fc - gstart];
            }
          }
        }
        PetscCall(VecRestoreArrayRead(gv, &ga));
        PetscCall(VecRestoreArray(subv, &suba));
        PetscCall(DMLabelDestroy(&cutVertexLabel));
      } else {
        PetscCall(PetscSectionGetField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv));
      }
      PetscCall(PetscStrncpy(subname, name, sizeof(subname)));
      PetscCall(PetscStrlcat(subname, "_", sizeof(subname)));
      PetscCall(PetscStrlcat(subname, fname, sizeof(subname)));
      PetscCall(PetscObjectSetName((PetscObject)subv, subname));
      if (isseq) PetscCall(VecView_Seq(subv, viewer));
      else PetscCall(VecView_MPI(subv, viewer));
      if ((ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_CELL_VECTOR_FIELD)) {
        PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)subv, "vector_field_type", PETSC_STRING, "vector"));
      } else {
        PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)subv, "vector_field_type", PETSC_STRING, "scalar"));
      }

      /* Output the component names in the field if available */
      PetscCall(PetscSectionGetFieldComponents(section, f, &Nc));
      for (c = 0; c < Nc; ++c) {
        char componentNameLabel[PETSC_MAX_PATH_LEN];
        PetscCall(PetscSectionGetComponentName(section, f, c, &componentName));
        PetscCall(PetscSNPrintf(componentNameLabel, sizeof(componentNameLabel), "componentName%" PetscInt_FMT, c));
        PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)subv, componentNameLabel, PETSC_STRING, componentName));
      }

      if (cutLabel) PetscCall(VecDestroy(&subv));
      else PetscCall(PetscSectionRestoreField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv));
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    }
  }
  if (seqnum >= 0) PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(DMRestoreGlobalVector(dmBC, &gv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM          dm;
  Vec         locv;
  PetscObject isZero;
  const char *name;
  PetscReal   time;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetLocalVector(dm, &locv));
  PetscCall(PetscObjectGetName((PetscObject)v, &name));
  PetscCall(PetscObjectSetName((PetscObject)locv, name));
  PetscCall(PetscObjectQuery((PetscObject)v, "__Vec_bc_zero__", &isZero));
  PetscCall(PetscObjectCompose((PetscObject)locv, "__Vec_bc_zero__", isZero));
  PetscCall(DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv));
  PetscCall(DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv));
  PetscCall(DMGetOutputSequenceNumber(dm, NULL, &time));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/fields"));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ));
  PetscCall(VecView_Plex_Local_HDF5_Internal(locv, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscObjectCompose((PetscObject)locv, "__Vec_bc_zero__", NULL));
  PetscCall(DMRestoreLocalVector(dm, &locv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  PetscBool isseq;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, VECSEQ, &isseq));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/fields"));
  if (isseq) PetscCall(VecView_Seq(v, viewer));
  else PetscCall(VecView_MPI(v, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecLoad_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM          dm;
  Vec         locv;
  const char *name;
  PetscInt    seqnum;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetLocalVector(dm, &locv));
  PetscCall(PetscObjectGetName((PetscObject)v, &name));
  PetscCall(PetscObjectSetName((PetscObject)locv, name));
  PetscCall(DMGetOutputSequenceNumber(dm, &seqnum, NULL));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/fields"));
  if (seqnum >= 0) {
    PetscCall(PetscViewerHDF5PushTimestepping(viewer));
    PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum));
  }
  PetscCall(VecLoad_Plex_Local(locv, viewer));
  if (seqnum >= 0) PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(DMLocalToGlobalBegin(dm, locv, INSERT_VALUES, v));
  PetscCall(DMLocalToGlobalEnd(dm, locv, INSERT_VALUES, v));
  PetscCall(DMRestoreLocalVector(dm, &locv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecLoad_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  DM       dm;
  PetscInt seqnum;

  PetscFunctionBegin;
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetOutputSequenceNumber(dm, &seqnum, NULL));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/fields"));
  if (seqnum >= 0) {
    PetscCall(PetscViewerHDF5PushTimestepping(viewer));
    PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum));
  }
  PetscCall(VecLoad_Default(v, viewer));
  if (seqnum >= 0) PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexDistributionView_HDF5_Private(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  MPI_Comm           comm;
  PetscMPIInt        size, rank;
  PetscInt           size_petsc_int;
  const char        *topologydm_name, *distribution_name;
  const PetscInt    *gpoint;
  PetscInt           pStart, pEnd, p;
  PetscSF            pointSF;
  PetscInt           nroots, nleaves;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  IS                 chartSizesIS, ownersIS, gpointsIS;
  PetscInt          *chartSize, *owners, *gpoints;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(DMPlexDistributionGetName(dm, &distribution_name));
  if (!distribution_name) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(DMPLEX_DistributionView, viewer, 0, 0, 0));
  size_petsc_int = (PetscInt)size;
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "comm_size", PETSC_INT, (void *)&size_petsc_int));
  PetscCall(ISGetIndices(globalPointNumbers, &gpoint));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscMalloc1(1, &chartSize));
  *chartSize = pEnd - pStart;
  PetscCall(PetscMalloc1(*chartSize, &owners));
  PetscCall(PetscMalloc1(*chartSize, &gpoints));
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(PetscSFGetGraph(pointSF, &nroots, &nleaves, &ilocal, &iremote));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt gp = gpoint[p - pStart];

    owners[p - pStart]  = rank;
    gpoints[p - pStart] = (gp < 0 ? -(gp + 1) : gp);
  }
  for (p = 0; p < nleaves; ++p) {
    PetscInt ilocalp = (ilocal ? ilocal[p] : p);

    owners[ilocalp] = iremote[p].rank;
  }
  PetscCall(ISCreateGeneral(comm, 1, chartSize, PETSC_OWN_POINTER, &chartSizesIS));
  PetscCall(ISCreateGeneral(comm, *chartSize, owners, PETSC_OWN_POINTER, &ownersIS));
  PetscCall(ISCreateGeneral(comm, *chartSize, gpoints, PETSC_OWN_POINTER, &gpointsIS));
  PetscCall(PetscObjectSetName((PetscObject)chartSizesIS, "chart_sizes"));
  PetscCall(PetscObjectSetName((PetscObject)ownersIS, "owners"));
  PetscCall(PetscObjectSetName((PetscObject)gpointsIS, "global_point_numbers"));
  PetscCall(ISView(chartSizesIS, viewer));
  PetscCall(ISView(ownersIS, viewer));
  PetscCall(ISView(gpointsIS, viewer));
  PetscCall(ISDestroy(&chartSizesIS));
  PetscCall(ISDestroy(&ownersIS));
  PetscCall(ISDestroy(&gpointsIS));
  PetscCall(ISRestoreIndices(globalPointNumbers, &gpoint));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributionView, viewer, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyView_HDF5_Inner_Private(DM dm, IS globalPointNumbers, PetscViewer viewer, PetscInt pStart, PetscInt pEnd, const char pointsName[], const char coneSizesName[], const char conesName[], const char orientationsName[])
{
  IS              coneSizesIS, conesIS, orientationsIS;
  PetscInt       *coneSizes, *cones, *orientations;
  const PetscInt *gpoint;
  PetscInt        nPoints = 0, conesSize = 0;
  PetscInt        p, c, s;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(ISGetIndices(globalPointNumbers, &gpoint));
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      PetscInt coneSize;

      PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
      nPoints += 1;
      conesSize += coneSize;
    }
  }
  PetscCall(PetscMalloc1(nPoints, &coneSizes));
  PetscCall(PetscMalloc1(conesSize, &cones));
  PetscCall(PetscMalloc1(conesSize, &orientations));
  for (p = pStart, c = 0, s = 0; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, cp;

      PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
      PetscCall(DMPlexGetCone(dm, p, &cone));
      PetscCall(DMPlexGetConeOrientation(dm, p, &ornt));
      coneSizes[s] = coneSize;
      for (cp = 0; cp < coneSize; ++cp, ++c) {
        cones[c]        = gpoint[cone[cp]] < 0 ? -(gpoint[cone[cp]] + 1) : gpoint[cone[cp]];
        orientations[c] = ornt[cp];
      }
      ++s;
    }
  }
  PetscCheck(s == nPoints, PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of points %" PetscInt_FMT " != %" PetscInt_FMT, s, nPoints);
  PetscCheck(c == conesSize, PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cone points %" PetscInt_FMT " != %" PetscInt_FMT, c, conesSize);
  PetscCall(ISCreateGeneral(comm, nPoints, coneSizes, PETSC_OWN_POINTER, &coneSizesIS));
  PetscCall(ISCreateGeneral(comm, conesSize, cones, PETSC_OWN_POINTER, &conesIS));
  PetscCall(ISCreateGeneral(comm, conesSize, orientations, PETSC_OWN_POINTER, &orientationsIS));
  PetscCall(PetscObjectSetName((PetscObject)coneSizesIS, coneSizesName));
  PetscCall(PetscObjectSetName((PetscObject)conesIS, conesName));
  PetscCall(PetscObjectSetName((PetscObject)orientationsIS, orientationsName));
  PetscCall(ISView(coneSizesIS, viewer));
  PetscCall(ISView(conesIS, viewer));
  PetscCall(ISView(orientationsIS, viewer));
  PetscCall(ISDestroy(&coneSizesIS));
  PetscCall(ISDestroy(&conesIS));
  PetscCall(ISDestroy(&orientationsIS));
  if (pointsName) {
    IS        pointsIS;
    PetscInt *points;

    PetscCall(PetscMalloc1(nPoints, &points));
    for (p = pStart, c = 0, s = 0; p < pEnd; ++p) {
      if (gpoint[p] >= 0) {
        points[s] = gpoint[p];
        ++s;
      }
    }
    PetscCall(ISCreateGeneral(comm, nPoints, points, PETSC_OWN_POINTER, &pointsIS));
    PetscCall(PetscObjectSetName((PetscObject)pointsIS, pointsName));
    PetscCall(ISView(pointsIS, viewer));
    PetscCall(ISDestroy(&pointsIS));
  }
  PetscCall(ISRestoreIndices(globalPointNumbers, &gpoint));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyView_HDF5_Legacy_Private(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char *pointsName, *coneSizesName, *conesName, *orientationsName;
  PetscInt    pStart, pEnd, dim;

  PetscFunctionBegin;
  pointsName       = "order";
  coneSizesName    = "cones";
  conesName        = "cells";
  orientationsName = "orientation";
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexTopologyView_HDF5_Inner_Private(dm, globalPointNumbers, viewer, pStart, pEnd, pointsName, coneSizesName, conesName, orientationsName));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, conesName, "cell_dim", PETSC_INT, (void *)&dim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO get this numbering right away without needing this function
/* Renumber global point numbers so that they are 0-based per stratum */
static PetscErrorCode RenumberGlobalPointNumbersPerStratum_Private(DM dm, IS globalPointNumbers, IS *newGlobalPointNumbers, IS *strataPermutation)
{
  PetscInt        d, depth, p, n;
  PetscInt       *offsets;
  const PetscInt *gpn;
  PetscInt       *ngpn;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(ISGetLocalSize(globalPointNumbers, &n));
  PetscCall(ISGetIndices(globalPointNumbers, &gpn));
  PetscCall(PetscMalloc1(n, &ngpn));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(PetscMalloc1(depth + 1, &offsets));
  for (d = 0; d <= depth; d++) {
    PetscInt pStart, pEnd;

    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    offsets[d] = PETSC_MAX_INT;
    for (p = pStart; p < pEnd; p++) {
      if (gpn[p] >= 0 && gpn[p] < offsets[d]) offsets[d] = gpn[p];
    }
  }
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, offsets, depth + 1, MPIU_INT, MPI_MIN, comm));
  for (d = 0; d <= depth; d++) {
    PetscInt pStart, pEnd;

    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    for (p = pStart; p < pEnd; p++) ngpn[p] = gpn[p] - PetscSign(gpn[p]) * offsets[d];
  }
  PetscCall(ISRestoreIndices(globalPointNumbers, &gpn));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)globalPointNumbers), n, ngpn, PETSC_OWN_POINTER, newGlobalPointNumbers));
  {
    PetscInt *perm;

    PetscCall(PetscMalloc1(depth + 1, &perm));
    for (d = 0; d <= depth; d++) perm[d] = d;
    PetscCall(PetscSortIntWithPermutation(depth + 1, offsets, perm));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, depth + 1, perm, PETSC_OWN_POINTER, strataPermutation));
  }
  PetscCall(PetscFree(offsets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyView_HDF5_Private(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  IS          globalPointNumbers0, strataPermutation;
  const char *coneSizesName, *conesName, *orientationsName;
  PetscInt    depth, d;
  MPI_Comm    comm;

  PetscFunctionBegin;
  coneSizesName    = "cone_sizes";
  conesName        = "cones";
  orientationsName = "orientations";
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetDepth(dm, &depth));
  {
    PetscInt dim;

    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "cell_dim", PETSC_INT, &dim));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "depth", PETSC_INT, &depth));
  }

  PetscCall(RenumberGlobalPointNumbersPerStratum_Private(dm, globalPointNumbers, &globalPointNumbers0, &strataPermutation));
  /* TODO dirty trick to save serial IS using the same parallel viewer */
  {
    IS              spOnComm;
    PetscInt        n   = 0, N;
    const PetscInt *idx = NULL;
    const PetscInt *old;
    PetscMPIInt     rank;

    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(ISGetLocalSize(strataPermutation, &N));
    PetscCall(ISGetIndices(strataPermutation, &old));
    if (!rank) {
      n   = N;
      idx = old;
    }
    PetscCall(ISCreateGeneral(comm, n, idx, PETSC_COPY_VALUES, &spOnComm));
    PetscCall(ISRestoreIndices(strataPermutation, &old));
    PetscCall(ISDestroy(&strataPermutation));
    strataPermutation = spOnComm;
  }
  PetscCall(PetscObjectSetName((PetscObject)strataPermutation, "permutation"));
  PetscCall(ISView(strataPermutation, viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "strata"));
  for (d = 0; d <= depth; d++) {
    PetscInt pStart, pEnd;
    char     group[128];

    PetscCall(PetscSNPrintf(group, sizeof(group), "%" PetscInt_FMT, d));
    PetscCall(PetscViewerHDF5PushGroup(viewer, group));
    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    PetscCall(DMPlexTopologyView_HDF5_Inner_Private(dm, globalPointNumbers0, viewer, pStart, pEnd, NULL, coneSizesName, conesName, orientationsName));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer)); /* strata */
  PetscCall(ISDestroy(&globalPointNumbers0));
  PetscCall(ISDestroy(&strataPermutation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexTopologyView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  DMPlexStorageVersion version;
  const char          *topologydm_name;
  char                 group[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetDMPlexStorageVersionWriting(viewer, &version));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  if (DMPlexStorageVersionGE(version, 2, 0, 0)) {
    PetscCall(PetscSNPrintf(group, sizeof(group), "topologies/%s", topologydm_name));
  } else {
    PetscCall(PetscStrncpy(group, "/", sizeof(group)));
  }
  PetscCall(PetscViewerHDF5PushGroup(viewer, group));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "topology"));
  if (version->major < 3) {
    PetscCall(DMPlexTopologyView_HDF5_Legacy_Private(dm, globalPointNumbers, viewer));
  } else {
    /* since DMPlexStorageVersion 3.0.0 */
    PetscCall(DMPlexTopologyView_HDF5_Private(dm, globalPointNumbers, viewer));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer)); /* "topology" */

  if (DMPlexStorageVersionGE(version, 2, 1, 0)) {
    const char *distribution_name;

    PetscCall(DMPlexDistributionGetName(dm, &distribution_name));
    PetscCall(PetscViewerHDF5PushGroup(viewer, "distributions"));
    PetscCall(PetscViewerHDF5WriteGroup(viewer, NULL));
    PetscCall(PetscViewerHDF5PushGroup(viewer, distribution_name));
    PetscCall(DMPlexDistributionView_HDF5_Private(dm, globalPointNumbers, viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer)); /* distribution_name */
    PetscCall(PetscViewerHDF5PopGroup(viewer)); /* "distributions" */
  }

  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConesIS_Private(DM dm, PetscInt cStart, PetscInt cEnd, IS globalCellNumbers, PetscInt *numCorners, IS *cellIS)
{
  PetscSF         sfPoint;
  DMLabel         cutLabel, cutVertexLabel         = NULL;
  IS              globalVertexNumbers, cutvertices = NULL;
  const PetscInt *gcell, *gvertex, *cutverts = NULL;
  PetscInt       *vertices;
  PetscInt        conesSize = 0;
  PetscInt        dim, numCornersLocal = 0, cell, vStart, vEnd, vExtra = 0, v;

  PetscFunctionBegin;
  *numCorners = 0;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(ISGetIndices(globalCellNumbers, &gcell));

  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, v, Nc = 0;

    if (gcell[cell] < 0) continue;
    PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (v = 0; v < closureSize * 2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++Nc;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    conesSize += Nc;
    if (!numCornersLocal) numCornersLocal = Nc;
    else if (numCornersLocal != Nc) numCornersLocal = 1;
  }
  PetscCall(MPIU_Allreduce(&numCornersLocal, numCorners, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  PetscCheck(!numCornersLocal || !(numCornersLocal != *numCorners || *numCorners == 0), PETSC_COMM_SELF, PETSC_ERR_SUP, "Visualization topology currently only supports identical cell shapes");
  /* Handle periodic cuts by identifying vertices which should be duplicated */
  PetscCall(DMGetLabel(dm, "periodic_cut", &cutLabel));
  PetscCall(DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel));
  if (cutVertexLabel) PetscCall(DMLabelGetStratumIS(cutVertexLabel, 1, &cutvertices));
  if (cutvertices) {
    PetscCall(ISGetIndices(cutvertices, &cutverts));
    PetscCall(ISGetLocalSize(cutvertices, &vExtra));
  }
  PetscCall(DMGetPointSF(dm, &sfPoint));
  if (cutLabel) {
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscInt           nroots, nleaves;

    PetscCall(PetscSFGetGraph(sfPoint, &nroots, &nleaves, &ilocal, &iremote));
    if (nleaves < 0) {
      PetscCall(PetscObjectReference((PetscObject)sfPoint));
    } else {
      PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)sfPoint), &sfPoint));
      PetscCall(PetscSFSetGraph(sfPoint, nroots + vExtra, nleaves, (PetscInt *)ilocal, PETSC_COPY_VALUES, (PetscSFNode *)iremote, PETSC_COPY_VALUES));
    }
  } else {
    PetscCall(PetscObjectReference((PetscObject)sfPoint));
  }
  /* Number all vertices */
  PetscCall(DMPlexCreateNumbering_Plex(dm, vStart, vEnd + vExtra, 0, NULL, sfPoint, &globalVertexNumbers));
  PetscCall(PetscSFDestroy(&sfPoint));
  /* Create cones */
  PetscCall(ISGetIndices(globalVertexNumbers, &gvertex));
  PetscCall(PetscMalloc1(conesSize, &vertices));
  for (cell = cStart, v = 0; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, Nc = 0, p, value = -1;
    PetscBool replace;

    if (gcell[cell] < 0) continue;
    if (cutLabel) PetscCall(DMLabelGetValue(cutLabel, cell, &value));
    replace = (value == 2) ? PETSC_TRUE : PETSC_FALSE;
    PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (p = 0; p < closureSize * 2; p += 2) {
      if ((closure[p] >= vStart) && (closure[p] < vEnd)) closure[Nc++] = closure[p];
    }
    PetscCall(DMPlexReorderCell(dm, cell, closure));
    for (p = 0; p < Nc; ++p) {
      PetscInt nv, gv = gvertex[closure[p] - vStart];

      if (replace) {
        PetscCall(PetscFindInt(closure[p], vExtra, cutverts, &nv));
        if (nv >= 0) gv = gvertex[vEnd - vStart + nv];
      }
      vertices[v++] = gv < 0 ? -(gv + 1) : gv;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  }
  PetscCall(ISRestoreIndices(globalVertexNumbers, &gvertex));
  PetscCall(ISDestroy(&globalVertexNumbers));
  PetscCall(ISRestoreIndices(globalCellNumbers, &gcell));
  if (cutvertices) PetscCall(ISRestoreIndices(cutvertices, &cutverts));
  PetscCall(ISDestroy(&cutvertices));
  PetscCall(DMLabelDestroy(&cutVertexLabel));
  PetscCheck(v == conesSize, PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cell vertices %" PetscInt_FMT " != %" PetscInt_FMT, v, conesSize);
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), conesSize, vertices, PETSC_OWN_POINTER, cellIS));
  PetscCall(PetscLayoutSetBlockSize((*cellIS)->map, *numCorners));
  PetscCall(PetscObjectSetName((PetscObject)*cellIS, "cells"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyView_HDF5_XDMF_Private(DM dm, IS globalCellNumbers, PetscViewer viewer)
{
  DM       cdm;
  DMLabel  depthLabel, ctLabel;
  IS       cellIS;
  PetscInt dim, depth, cellHeight, c, n = 0;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/viz"));
  PetscCall(PetscViewerHDF5WriteGroup(viewer, NULL));

  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMPlexGetCellTypeLabel(dm, &ctLabel));
  for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
    const DMPolytopeType ict = (DMPolytopeType)c;
    PetscInt             pStart, pEnd, dep, numCorners;
    PetscBool            output = PETSC_FALSE, doOutput;

    if (ict == DM_POLYTOPE_FV_GHOST) continue;
    PetscCall(DMLabelGetStratumBounds(ctLabel, ict, &pStart, &pEnd));
    if (pStart >= 0) {
      PetscCall(DMLabelGetValue(depthLabel, pStart, &dep));
      if (dep == depth - cellHeight) output = PETSC_TRUE;
    }
    PetscCall(MPIU_Allreduce(&output, &doOutput, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)dm)));
    if (!doOutput) continue;
    PetscCall(CreateConesIS_Private(dm, pStart, pEnd, globalCellNumbers, &numCorners, &cellIS));
    if (!n) {
      PetscCall(PetscViewerHDF5PushGroup(viewer, "/viz/topology"));
    } else {
      char group[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/viz/topology_%" PetscInt_FMT, n));
      PetscCall(PetscViewerHDF5PushGroup(viewer, group));
    }
    PetscCall(ISView(cellIS, viewer));
    PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)cellIS, "cell_corners", PETSC_INT, (void *)&numCorners));
    PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)cellIS, "cell_dim", PETSC_INT, (void *)&dim));
    PetscCall(ISDestroy(&cellIS));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    ++n;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCoordinatesView_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
{
  DM        cdm;
  Vec       coordinates, newcoords;
  PetscReal lengthScale;
  PetscInt  m, M, bs;

  PetscFunctionBegin;
  PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinates(dm, &coordinates));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)coordinates), &newcoords));
  PetscCall(PetscObjectSetName((PetscObject)newcoords, "vertices"));
  PetscCall(VecGetSize(coordinates, &M));
  PetscCall(VecGetLocalSize(coordinates, &m));
  PetscCall(VecSetSizes(newcoords, m, M));
  PetscCall(VecGetBlockSize(coordinates, &bs));
  PetscCall(VecSetBlockSize(newcoords, bs));
  PetscCall(VecSetType(newcoords, VECSTANDARD));
  PetscCall(VecCopy(coordinates, newcoords));
  PetscCall(VecScale(newcoords, lengthScale));
  /* Did not use DMGetGlobalVector() in order to bypass default group assignment */
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/geometry"));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(VecView(newcoords, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(VecDestroy(&newcoords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCoordinatesView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  DM          cdm;
  Vec         coords, newcoords;
  PetscInt    m, M, bs;
  PetscReal   lengthScale;
  const char *topologydm_name, *coordinatedm_name, *coordinates_name;

  PetscFunctionBegin;
  {
    PetscViewerFormat    format;
    DMPlexStorageVersion version;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscViewerHDF5GetDMPlexStorageVersionWriting(viewer, &version));
    if (!DMPlexStorageVersionGE(version, 2, 0, 0) || format == PETSC_VIEWER_HDF5_XDMF || format == PETSC_VIEWER_HDF5_VIZ) {
      PetscCall(DMPlexCoordinatesView_HDF5_Legacy_Private(dm, viewer));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  /* since 2.0.0 */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinates(dm, &coords));
  PetscCall(PetscObjectGetName((PetscObject)cdm, &coordinatedm_name));
  PetscCall(PetscObjectGetName((PetscObject)coords, &coordinates_name));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "coordinateDMName", PETSC_STRING, coordinatedm_name));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "coordinatesName", PETSC_STRING, coordinates_name));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(DMPlexSectionView(dm, viewer, cdm));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)coords), &newcoords));
  PetscCall(PetscObjectSetName((PetscObject)newcoords, coordinates_name));
  PetscCall(VecGetSize(coords, &M));
  PetscCall(VecGetLocalSize(coords, &m));
  PetscCall(VecSetSizes(newcoords, m, M));
  PetscCall(VecGetBlockSize(coords, &bs));
  PetscCall(VecSetBlockSize(newcoords, bs));
  PetscCall(VecSetType(newcoords, VECSTANDARD));
  PetscCall(VecCopy(coords, newcoords));
  PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  PetscCall(VecScale(newcoords, lengthScale));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(DMPlexGlobalVectorView(dm, viewer, cdm, newcoords));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(VecDestroy(&newcoords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCoordinatesView_HDF5_XDMF_Private(DM dm, PetscViewer viewer)
{
  DM               cdm;
  Vec              coordinatesLocal, newcoords;
  PetscSection     cSection, cGlobalSection;
  PetscScalar     *coords, *ncoords;
  DMLabel          cutLabel, cutVertexLabel = NULL;
  const PetscReal *L;
  PetscReal        lengthScale;
  PetscInt         vStart, vEnd, v, bs, N, coordSize, dof, off, d;
  PetscBool        localized, embedded;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));
  PetscCall(VecGetBlockSize(coordinatesLocal, &bs));
  PetscCall(DMGetCoordinatesLocalized(dm, &localized));
  if (localized == PETSC_FALSE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetPeriodicity(dm, NULL, NULL, &L));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &cSection));
  PetscCall(DMGetGlobalSection(cdm, &cGlobalSection));
  PetscCall(DMGetLabel(dm, "periodic_cut", &cutLabel));
  N = 0;

  PetscCall(DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), &newcoords));
  PetscCall(PetscSectionGetDof(cSection, vStart, &dof));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "DOF: %" PetscInt_FMT "\n", dof));
  embedded = (PetscBool)(L && dof == 2 && !cutLabel);
  if (cutVertexLabel) {
    PetscCall(DMLabelGetStratumSize(cutVertexLabel, 1, &v));
    N += dof * v;
  }
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionGetDof(cGlobalSection, v, &dof));
    if (dof < 0) continue;
    if (embedded) N += dof + 1;
    else N += dof;
  }
  if (embedded) PetscCall(VecSetBlockSize(newcoords, bs + 1));
  else PetscCall(VecSetBlockSize(newcoords, bs));
  PetscCall(VecSetSizes(newcoords, N, PETSC_DETERMINE));
  PetscCall(VecSetType(newcoords, VECSTANDARD));
  PetscCall(VecGetArray(coordinatesLocal, &coords));
  PetscCall(VecGetArray(newcoords, &ncoords));
  coordSize = 0;
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionGetDof(cGlobalSection, v, &dof));
    PetscCall(PetscSectionGetOffset(cSection, v, &off));
    if (dof < 0) continue;
    if (embedded) {
      if (L && (L[0] > 0.0) && (L[1] > 0.0)) {
        PetscReal theta, phi, r, R;
        /* XY-periodic */
        /* Suppose its an y-z circle, then
             \hat r = (0, cos(th), sin(th)) \hat x = (1, 0, 0)
           and the circle in that plane is
             \hat r cos(phi) + \hat x sin(phi) */
        theta                = 2.0 * PETSC_PI * PetscRealPart(coords[off + 1]) / L[1];
        phi                  = 2.0 * PETSC_PI * PetscRealPart(coords[off + 0]) / L[0];
        r                    = L[0] / (2.0 * PETSC_PI * 2.0 * L[1]);
        R                    = L[1] / (2.0 * PETSC_PI);
        ncoords[coordSize++] = PetscSinReal(phi) * r;
        ncoords[coordSize++] = -PetscCosReal(theta) * (R + r * PetscCosReal(phi));
        ncoords[coordSize++] = PetscSinReal(theta) * (R + r * PetscCosReal(phi));
      } else if (L && (L[0] > 0.0)) {
        /* X-periodic */
        ncoords[coordSize++] = -PetscCosReal(2.0 * PETSC_PI * PetscRealPart(coords[off + 0]) / L[0]) * (L[0] / (2.0 * PETSC_PI));
        ncoords[coordSize++] = coords[off + 1];
        ncoords[coordSize++] = PetscSinReal(2.0 * PETSC_PI * PetscRealPart(coords[off + 0]) / L[0]) * (L[0] / (2.0 * PETSC_PI));
      } else if (L && (L[1] > 0.0)) {
        /* Y-periodic */
        ncoords[coordSize++] = coords[off + 0];
        ncoords[coordSize++] = PetscSinReal(2.0 * PETSC_PI * PetscRealPart(coords[off + 1]) / L[1]) * (L[1] / (2.0 * PETSC_PI));
        ncoords[coordSize++] = -PetscCosReal(2.0 * PETSC_PI * PetscRealPart(coords[off + 1]) / L[1]) * (L[1] / (2.0 * PETSC_PI));
  #if 0
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
  #endif
      } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Cannot handle periodicity in this domain");
    } else {
      for (d = 0; d < dof; ++d, ++coordSize) ncoords[coordSize] = coords[off + d];
    }
  }
  if (cutVertexLabel) {
    IS              vertices;
    const PetscInt *verts;
    PetscInt        n;

    PetscCall(DMLabelGetStratumIS(cutVertexLabel, 1, &vertices));
    if (vertices) {
      PetscCall(ISGetIndices(vertices, &verts));
      PetscCall(ISGetLocalSize(vertices, &n));
      for (v = 0; v < n; ++v) {
        PetscCall(PetscSectionGetDof(cSection, verts[v], &dof));
        PetscCall(PetscSectionGetOffset(cSection, verts[v], &off));
        for (d = 0; d < dof; ++d) ncoords[coordSize++] = coords[off + d] + ((L[d] > 0.) ? L[d] : 0.0);
      }
      PetscCall(ISRestoreIndices(vertices, &verts));
      PetscCall(ISDestroy(&vertices));
    }
  }
  PetscCheck(coordSize == N, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched sizes: %" PetscInt_FMT " != %" PetscInt_FMT, coordSize, N);
  PetscCall(DMLabelDestroy(&cutVertexLabel));
  PetscCall(VecRestoreArray(coordinatesLocal, &coords));
  PetscCall(VecRestoreArray(newcoords, &ncoords));
  PetscCall(PetscObjectSetName((PetscObject)newcoords, "vertices"));
  PetscCall(VecScale(newcoords, lengthScale));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/viz"));
  PetscCall(PetscViewerHDF5WriteGroup(viewer, NULL));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/viz/geometry"));
  PetscCall(VecView(newcoords, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(VecDestroy(&newcoords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexLabelsView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char          *topologydm_name;
  const PetscInt      *gpoint;
  PetscInt             numLabels, l;
  DMPlexStorageVersion version;
  char                 group[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetDMPlexStorageVersionWriting(viewer, &version));
  PetscCall(ISGetIndices(globalPointNumbers, &gpoint));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  if (DMPlexStorageVersionGE(version, 2, 0, 0)) {
    PetscCall(PetscSNPrintf(group, sizeof(group), "topologies/%s/labels", topologydm_name));
  } else {
    PetscCall(PetscStrncpy(group, "/labels", sizeof(group)));
  }
  PetscCall(PetscViewerHDF5PushGroup(viewer, group));
  PetscCall(DMGetNumLabels(dm, &numLabels));
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label;
    const char     *name;
    IS              valueIS, pvalueIS, globalValueIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscBool       isDepth, output;

    PetscCall(DMGetLabelByNum(dm, l, &label));
    PetscCall(PetscObjectGetName((PetscObject)label, &name));
    PetscCall(DMGetLabelOutput(dm, name, &output));
    PetscCall(PetscStrncmp(name, "depth", 10, &isDepth));
    if (isDepth || !output) continue;
    PetscCall(PetscViewerHDF5PushGroup(viewer, name));
    PetscCall(DMLabelGetValueIS(label, &valueIS));
    /* Must copy to a new IS on the global comm */
    PetscCall(ISGetLocalSize(valueIS, &numValues));
    PetscCall(ISGetIndices(valueIS, &values));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), numValues, values, PETSC_COPY_VALUES, &pvalueIS));
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISAllGather(pvalueIS, &globalValueIS));
    PetscCall(ISDestroy(&pvalueIS));
    PetscCall(ISSortRemoveDups(globalValueIS));
    PetscCall(ISGetLocalSize(globalValueIS, &numValues));
    PetscCall(ISGetIndices(globalValueIS, &values));
    for (v = 0; v < numValues; ++v) {
      IS              stratumIS, globalStratumIS;
      const PetscInt *spoints = NULL;
      PetscInt       *gspoints, n = 0, gn, p;
      const char     *iname = "indices";
      char            group[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(group, sizeof(group), "%" PetscInt_FMT, values[v]));
      PetscCall(PetscViewerHDF5PushGroup(viewer, group));
      PetscCall(DMLabelGetStratumIS(label, values[v], &stratumIS));

      if (stratumIS) PetscCall(ISGetLocalSize(stratumIS, &n));
      if (stratumIS) PetscCall(ISGetIndices(stratumIS, &spoints));
      for (gn = 0, p = 0; p < n; ++p)
        if (gpoint[spoints[p]] >= 0) ++gn;
      PetscCall(PetscMalloc1(gn, &gspoints));
      for (gn = 0, p = 0; p < n; ++p)
        if (gpoint[spoints[p]] >= 0) gspoints[gn++] = gpoint[spoints[p]];
      if (stratumIS) PetscCall(ISRestoreIndices(stratumIS, &spoints));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), gn, gspoints, PETSC_OWN_POINTER, &globalStratumIS));
      PetscCall(PetscObjectSetName((PetscObject)globalStratumIS, iname));

      PetscCall(ISView(globalStratumIS, viewer));
      PetscCall(ISDestroy(&globalStratumIS));
      PetscCall(ISDestroy(&stratumIS));
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    }
    PetscCall(ISRestoreIndices(globalValueIS, &values));
    PetscCall(ISDestroy(&globalValueIS));
    PetscCall(ISDestroy(&valueIS));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
  }
  PetscCall(ISRestoreIndices(globalPointNumbers, &gpoint));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* We only write cells and vertices. Does this screw up parallel reading? */
PetscErrorCode DMPlexView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  IS                globalPointNumbers;
  PetscViewerFormat format;
  PetscBool         viz_geom = PETSC_FALSE, xdmf_topo = PETSC_FALSE, petsc_topo = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DMPlexCreatePointNumbering(dm, &globalPointNumbers));
  PetscCall(DMPlexCoordinatesView_HDF5_Internal(dm, viewer));

  PetscCall(PetscViewerGetFormat(viewer, &format));
  switch (format) {
  case PETSC_VIEWER_HDF5_VIZ:
    viz_geom  = PETSC_TRUE;
    xdmf_topo = PETSC_TRUE;
    break;
  case PETSC_VIEWER_HDF5_XDMF:
    xdmf_topo = PETSC_TRUE;
    break;
  case PETSC_VIEWER_HDF5_PETSC:
    petsc_topo = PETSC_TRUE;
    break;
  case PETSC_VIEWER_DEFAULT:
  case PETSC_VIEWER_NATIVE:
    viz_geom   = PETSC_TRUE;
    xdmf_topo  = PETSC_TRUE;
    petsc_topo = PETSC_TRUE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 output.", PetscViewerFormats[format]);
  }

  if (viz_geom) PetscCall(DMPlexCoordinatesView_HDF5_XDMF_Private(dm, viewer));
  if (xdmf_topo) PetscCall(DMPlexTopologyView_HDF5_XDMF_Private(dm, globalPointNumbers, viewer));
  if (petsc_topo) {
    PetscCall(DMPlexTopologyView_HDF5_Internal(dm, globalPointNumbers, viewer));
    PetscCall(DMPlexLabelsView_HDF5_Internal(dm, globalPointNumbers, viewer));
  }

  PetscCall(ISDestroy(&globalPointNumbers));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexSectionView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm)
{
  MPI_Comm     comm;
  const char  *topologydm_name;
  const char  *sectiondm_name;
  PetscSection gsection;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sectiondm, &comm));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "dms"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  PetscCall(DMGetGlobalSection(sectiondm, &gsection));
  /* Save raw section */
  PetscCall(PetscSectionView(gsection, viewer));
  /* Save plex wrapper */
  {
    PetscInt        pStart, pEnd, p, n;
    IS              globalPointNumbers;
    const PetscInt *gpoints;
    IS              orderIS;
    PetscInt       *order;

    PetscCall(PetscSectionGetChart(gsection, &pStart, &pEnd));
    PetscCall(DMPlexCreatePointNumbering(dm, &globalPointNumbers));
    PetscCall(ISGetIndices(globalPointNumbers, &gpoints));
    for (p = pStart, n = 0; p < pEnd; ++p)
      if (gpoints[p] >= 0) n++;
    /* "order" is an array of global point numbers.
       When loading, it is used with topology/order array
       to match section points with plex topology points. */
    PetscCall(PetscMalloc1(n, &order));
    for (p = pStart, n = 0; p < pEnd; ++p)
      if (gpoints[p] >= 0) order[n++] = gpoints[p];
    PetscCall(ISRestoreIndices(globalPointNumbers, &gpoints));
    PetscCall(ISDestroy(&globalPointNumbers));
    PetscCall(ISCreateGeneral(comm, n, order, PETSC_OWN_POINTER, &orderIS));
    PetscCall(PetscObjectSetName((PetscObject)orderIS, "order"));
    PetscCall(ISView(orderIS, viewer));
    PetscCall(ISDestroy(&orderIS));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexGlobalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  const char *topologydm_name;
  const char *sectiondm_name;
  const char *vec_name;
  PetscInt    bs;

  PetscFunctionBegin;
  /* Check consistency */
  {
    PetscSF pointsf, pointsf1;

    PetscCall(DMGetPointSF(dm, &pointsf));
    PetscCall(DMGetPointSF(sectiondm, &pointsf1));
    PetscCheck(pointsf1 == pointsf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  PetscCall(PetscObjectGetName((PetscObject)vec, &vec_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "dms"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "vecs"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, vec_name));
  PetscCall(VecGetBlockSize(vec, &bs));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *)&bs));
  PetscCall(VecSetBlockSize(vec, 1));
  /* VecView(vec, viewer) would call (*vec->opt->view)(vec, viewer), but,    */
  /* if vec was created with DMGet{Global, Local}Vector(), vec->opt->view    */
  /* is set to VecView_Plex, which would save vec in a predefined location.  */
  /* To save vec in where we want, we create a new Vec (temp) with           */
  /* VecCreate(), wrap the vec data in temp, and call VecView(temp, viewer). */
  {
    Vec                temp;
    const PetscScalar *array;
    PetscLayout        map;

    PetscCall(VecCreate(PetscObjectComm((PetscObject)vec), &temp));
    PetscCall(PetscObjectSetName((PetscObject)temp, vec_name));
    PetscCall(VecGetLayout(vec, &map));
    PetscCall(VecSetLayout(temp, map));
    PetscCall(VecSetUp(temp));
    PetscCall(VecGetArrayRead(vec, &array));
    PetscCall(VecPlaceArray(temp, array));
    PetscCall(VecView(temp, viewer));
    PetscCall(VecResetArray(temp));
    PetscCall(VecRestoreArrayRead(vec, &array));
    PetscCall(VecDestroy(&temp));
  }
  PetscCall(VecSetBlockSize(vec, bs));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexLocalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  MPI_Comm     comm;
  const char  *topologydm_name;
  const char  *sectiondm_name;
  const char  *vec_name;
  PetscSection section;
  PetscBool    includesConstraints;
  Vec          gvec;
  PetscInt     m, bs;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  /* Check consistency */
  {
    PetscSF pointsf, pointsf1;

    PetscCall(DMGetPointSF(dm, &pointsf));
    PetscCall(DMGetPointSF(sectiondm, &pointsf1));
    PetscCheck(pointsf1 == pointsf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  PetscCall(PetscObjectGetName((PetscObject)vec, &vec_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "dms"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "vecs"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, vec_name));
  PetscCall(VecGetBlockSize(vec, &bs));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *)&bs));
  PetscCall(VecCreate(comm, &gvec));
  PetscCall(PetscObjectSetName((PetscObject)gvec, vec_name));
  PetscCall(DMGetGlobalSection(sectiondm, &section));
  PetscCall(PetscSectionGetIncludesConstraints(section, &includesConstraints));
  if (includesConstraints) PetscCall(PetscSectionGetStorageSize(section, &m));
  else PetscCall(PetscSectionGetConstrainedStorageSize(section, &m));
  PetscCall(VecSetSizes(gvec, m, PETSC_DECIDE));
  PetscCall(VecSetUp(gvec));
  PetscCall(DMLocalToGlobalBegin(sectiondm, vec, INSERT_VALUES, gvec));
  PetscCall(DMLocalToGlobalEnd(sectiondm, vec, INSERT_VALUES, gvec));
  PetscCall(VecView(gvec, viewer));
  PetscCall(VecDestroy(&gvec));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscNew(ctx));
  PetscCall(PetscObjectReference((PetscObject)((*ctx)->dm = dm)));
  PetscCall(PetscObjectReference((PetscObject)((*ctx)->viewer = viewer)));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &(*ctx)->comm));
  PetscCallMPI(MPI_Comm_rank((*ctx)->comm, &(*ctx)->rank));
  (*ctx)->sfXC = sfXC;
  if (sfXC) {
    PetscInt nX;

    PetscCall(PetscObjectReference((PetscObject)sfXC));
    PetscCall(PetscSFGetGraph(sfXC, &nX, NULL, NULL, NULL));
    PetscCall(PetscLayoutCreateFromSizes((*ctx)->comm, nX, PETSC_DECIDE, 1, &(*ctx)->layoutX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LoadLabelsCtxDestroy(LoadLabelsCtx *ctx)
{
  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMDestroy(&(*ctx)->dm));
  PetscCall(PetscViewerDestroy(&(*ctx)->viewer));
  PetscCall(PetscSFDestroy(&(*ctx)->sfXC));
  PetscCall(PetscLayoutDestroy(&(*ctx)->layoutX));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscSFGetGraph(sfXC, &nX, &nC, NULL, NULL));
  PetscCall(ISGetLocalSize(stratumIS, &n));
  PetscCall(ISGetIndices(stratumIS, &A_points));
  PetscCall(PetscSFCreate(comm, &sfXA));
  PetscCall(PetscSFSetGraphLayout(sfXA, layoutX, n, NULL, PETSC_USE_POINTER, A_points));
  PetscCall(ISCreate(comm, newStratumIS));
  PetscCall(ISSetType(*newStratumIS, ISGENERAL));
  {
    PetscInt   i;
    PetscBool *A_mask, *X_mask, *C_mask;

    PetscCall(PetscCalloc3(n, &A_mask, nX, &X_mask, nC, &C_mask));
    for (i = 0; i < n; i++) A_mask[i] = PETSC_TRUE;
    PetscCall(PetscSFReduceBegin(sfXA, MPIU_BOOL, A_mask, X_mask, MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(sfXA, MPIU_BOOL, A_mask, X_mask, MPI_REPLACE));
    PetscCall(PetscSFBcastBegin(sfXC, MPIU_BOOL, X_mask, C_mask, MPI_LOR));
    PetscCall(PetscSFBcastEnd(sfXC, MPIU_BOOL, X_mask, C_mask, MPI_LOR));
    PetscCall(ISGeneralSetIndicesFromMask(*newStratumIS, 0, nC, C_mask));
    PetscCall(PetscFree3(A_mask, X_mask, C_mask));
  }
  PetscCall(PetscSFDestroy(&sfXA));
  PetscCall(ISRestoreIndices(stratumIS, &A_points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static herr_t ReadLabelStratumHDF5_Static(hid_t g_id, const char *vname, const H5L_info_t *info, void *op_data)
{
  LoadLabelsCtx   ctx    = (LoadLabelsCtx)op_data;
  PetscViewer     viewer = ctx->viewer;
  DMLabel         label  = ctx->label;
  MPI_Comm        comm   = ctx->comm;
  IS              stratumIS;
  const PetscInt *ind;
  PetscInt        value, N, i;

  PetscCall(PetscOptionsStringToInt(vname, &value));
  PetscCall(ISCreate(comm, &stratumIS));
  PetscCall(PetscObjectSetName((PetscObject)stratumIS, "indices"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, vname)); /* labels/<lname>/<vname> */

  if (!ctx->sfXC) {
    /* Force serial load */
    PetscCall(PetscViewerHDF5ReadSizes(viewer, "indices", NULL, &N));
    PetscCall(PetscLayoutSetLocalSize(stratumIS->map, !ctx->rank ? N : 0));
    PetscCall(PetscLayoutSetSize(stratumIS->map, N));
  }
  PetscCall(ISLoad(stratumIS, viewer));

  if (ctx->sfXC) {
    IS newStratumIS;

    PetscCallHDF5(ReadLabelStratumHDF5_Distribute_Private, (stratumIS, ctx, &newStratumIS));
    PetscCall(ISDestroy(&stratumIS));
    stratumIS = newStratumIS;
  }

  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(ISGetLocalSize(stratumIS, &N));
  PetscCall(ISGetIndices(stratumIS, &ind));
  for (i = 0; i < N; ++i) PetscCall(DMLabelSetValue(label, ind[i], value));
  PetscCall(ISRestoreIndices(stratumIS, &ind));
  PetscCall(ISDestroy(&stratumIS));
  return 0;
}

/* TODO: Fix this code, it is returning PETSc error codes when it should be translating them to herr_t codes */
static herr_t ReadLabelHDF5_Static(hid_t g_id, const char *lname, const H5L_info_t *info, void *op_data)
{
  LoadLabelsCtx  ctx = (LoadLabelsCtx)op_data;
  DM             dm  = ctx->dm;
  hsize_t        idx = 0;
  PetscErrorCode ierr;
  PetscBool      flg;
  herr_t         err;

  PetscCall(DMHasLabel(dm, lname, &flg));
  if (flg) PetscCall(DMRemoveLabel(dm, lname, NULL));
  ierr = DMCreateLabel(dm, lname);
  if (ierr) return (herr_t)ierr;
  ierr = DMGetLabel(dm, lname, &ctx->label);
  if (ierr) return (herr_t)ierr;
  ierr = PetscViewerHDF5PushGroup(ctx->viewer, lname);
  if (ierr) return (herr_t)ierr;
  /* Iterate over the label's strata */
  PetscCallHDF5Return(err, H5Literate_by_name, (g_id, lname, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelStratumHDF5_Static, op_data, 0));
  ierr = PetscViewerHDF5PopGroup(ctx->viewer);
  if (ierr) return (herr_t)ierr;
  return err;
}

PetscErrorCode DMPlexLabelsLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF sfXC)
{
  const char          *topologydm_name;
  LoadLabelsCtx        ctx;
  hsize_t              idx = 0;
  char                 group[PETSC_MAX_PATH_LEN];
  DMPlexStorageVersion version;
  PetscBool            distributed, hasGroup;

  PetscFunctionBegin;
  PetscCall(DMPlexIsDistributed(dm, &distributed));
  if (distributed) PetscCheck(sfXC, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "PetscSF must be given for parallel load");
  PetscCall(LoadLabelsCtxCreate(dm, viewer, sfXC, &ctx));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscViewerHDF5GetDMPlexStorageVersionReading(viewer, &version));
  if (DMPlexStorageVersionGE(version, 2, 0, 0)) {
    PetscCall(PetscSNPrintf(group, sizeof(group), "topologies/%s/labels", topologydm_name));
  } else {
    PetscCall(PetscStrncpy(group, "labels", sizeof(group)));
  }
  PetscCall(PetscViewerHDF5PushGroup(viewer, group));
  PetscCall(PetscViewerHDF5HasGroup(viewer, NULL, &hasGroup));
  if (hasGroup) {
    hid_t fileId, groupId;

    PetscCall(PetscViewerHDF5OpenGroup(viewer, NULL, &fileId, &groupId));
    /* Iterate over labels */
    PetscCallHDF5(H5Literate, (groupId, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelHDF5_Static, ctx));
    PetscCallHDF5(H5Gclose, (groupId));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(LoadLabelsCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexDistributionLoad_HDF5_Private(DM dm, PetscViewer viewer, PetscSF sf, PetscSF *distsf, DM *distdm)
{
  MPI_Comm        comm;
  PetscMPIInt     size, rank;
  PetscInt        dist_size;
  const char     *distribution_name;
  PetscInt        p, lsize;
  IS              chartSizesIS, ownersIS, gpointsIS;
  const PetscInt *chartSize, *owners, *gpoints;
  PetscLayout     layout;
  PetscBool       has;

  PetscFunctionBegin;
  *distsf = NULL;
  *distdm = NULL;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexDistributionGetName(dm, &distribution_name));
  if (!distribution_name) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(DMPLEX_DistributionLoad, viewer, 0, 0, 0));
  PetscCall(PetscViewerHDF5HasGroup(viewer, NULL, &has));
  if (!has) {
    char *full_group;

    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &full_group));
    PetscCheck(has, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Distribution %s cannot be found: HDF5 group %s not found in file", distribution_name, full_group);
  }
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "comm_size", PETSC_INT, NULL, (void *)&dist_size));
  PetscCheck(dist_size == (PetscInt)size, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Mismatching comm sizes: comm size of this session (%d) != comm size used for %s (%" PetscInt_FMT ")", size, distribution_name, dist_size);
  PetscCall(ISCreate(comm, &chartSizesIS));
  PetscCall(PetscObjectSetName((PetscObject)chartSizesIS, "chart_sizes"));
  PetscCall(ISCreate(comm, &ownersIS));
  PetscCall(PetscObjectSetName((PetscObject)ownersIS, "owners"));
  PetscCall(ISCreate(comm, &gpointsIS));
  PetscCall(PetscObjectSetName((PetscObject)gpointsIS, "global_point_numbers"));
  PetscCall(PetscLayoutSetLocalSize(chartSizesIS->map, 1));
  PetscCall(ISLoad(chartSizesIS, viewer));
  PetscCall(ISGetIndices(chartSizesIS, &chartSize));
  PetscCall(PetscLayoutSetLocalSize(ownersIS->map, *chartSize));
  PetscCall(PetscLayoutSetLocalSize(gpointsIS->map, *chartSize));
  PetscCall(ISLoad(ownersIS, viewer));
  PetscCall(ISLoad(gpointsIS, viewer));
  PetscCall(ISGetIndices(ownersIS, &owners));
  PetscCall(ISGetIndices(gpointsIS, &gpoints));
  PetscCall(PetscSFCreate(comm, distsf));
  PetscCall(PetscSFSetFromOptions(*distsf));
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscSFGetGraph(sf, &lsize, NULL, NULL, NULL));
  PetscCall(PetscLayoutSetLocalSize(layout, lsize));
  PetscCall(PetscLayoutSetBlockSize(layout, 1));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscSFSetGraphLayout(*distsf, layout, *chartSize, NULL, PETSC_OWN_POINTER, gpoints));
  PetscCall(PetscLayoutDestroy(&layout));
  /* Migrate DM */
  {
    PetscInt     pStart, pEnd;
    PetscSFNode *buffer0, *buffer1, *buffer2;

    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(PetscMalloc2(pEnd - pStart, &buffer0, lsize, &buffer1));
    PetscCall(PetscMalloc1(*chartSize, &buffer2));
    {
      PetscSF            workPointSF;
      PetscInt           workNroots, workNleaves;
      const PetscInt    *workIlocal;
      const PetscSFNode *workIremote;

      for (p = pStart; p < pEnd; ++p) {
        buffer0[p - pStart].rank  = rank;
        buffer0[p - pStart].index = p - pStart;
      }
      PetscCall(DMGetPointSF(dm, &workPointSF));
      PetscCall(PetscSFGetGraph(workPointSF, &workNroots, &workNleaves, &workIlocal, &workIremote));
      for (p = 0; p < workNleaves; ++p) {
        PetscInt workIlocalp = (workIlocal ? workIlocal[p] : p);

        buffer0[workIlocalp].rank = -1;
      }
    }
    for (p = 0; p < lsize; ++p) buffer1[p].rank = -1;
    for (p = 0; p < *chartSize; ++p) buffer2[p].rank = -1;
    PetscCall(PetscSFReduceBegin(sf, MPIU_2INT, buffer0, buffer1, MPI_MAXLOC));
    PetscCall(PetscSFReduceEnd(sf, MPIU_2INT, buffer0, buffer1, MPI_MAXLOC));
    PetscCall(PetscSFBcastBegin(*distsf, MPIU_2INT, buffer1, buffer2, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(*distsf, MPIU_2INT, buffer1, buffer2, MPI_REPLACE));
    if (PetscDefined(USE_DEBUG)) {
      for (p = 0; p < *chartSize; ++p) {
        PetscCheck(buffer2[p].rank >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Found negative root rank %" PetscInt_FMT " at local point %" PetscInt_FMT " on rank %d when making migrationSF", buffer2[p].rank, p, rank);
      }
    }
    PetscCall(PetscFree2(buffer0, buffer1));
    PetscCall(DMCreate(comm, distdm));
    PetscCall(DMSetType(*distdm, DMPLEX));
    PetscCall(PetscObjectSetName((PetscObject)(*distdm), ((PetscObject)dm)->name));
    PetscCall(DMPlexDistributionSetName(*distdm, distribution_name));
    {
      PetscSF migrationSF;

      PetscCall(PetscSFCreate(comm, &migrationSF));
      PetscCall(PetscSFSetFromOptions(migrationSF));
      PetscCall(PetscSFSetGraph(migrationSF, pEnd - pStart, *chartSize, NULL, PETSC_OWN_POINTER, buffer2, PETSC_OWN_POINTER));
      PetscCall(PetscSFSetUp(migrationSF));
      PetscCall(DMPlexMigrate(dm, migrationSF, *distdm));
      PetscCall(PetscSFDestroy(&migrationSF));
    }
  }
  /* Set pointSF */
  {
    PetscSF      pointSF;
    PetscInt    *ilocal, nleaves, q;
    PetscSFNode *iremote, *buffer0, *buffer1;

    PetscCall(PetscMalloc2(*chartSize, &buffer0, lsize, &buffer1));
    for (p = 0, nleaves = 0; p < *chartSize; ++p) {
      if (owners[p] == rank) {
        buffer0[p].rank = rank;
      } else {
        buffer0[p].rank = -1;
        nleaves++;
      }
      buffer0[p].index = p;
    }
    for (p = 0; p < lsize; ++p) buffer1[p].rank = -1;
    PetscCall(PetscSFReduceBegin(*distsf, MPIU_2INT, buffer0, buffer1, MPI_MAXLOC));
    PetscCall(PetscSFReduceEnd(*distsf, MPIU_2INT, buffer0, buffer1, MPI_MAXLOC));
    for (p = 0; p < *chartSize; ++p) buffer0[p].rank = -1;
    PetscCall(PetscSFBcastBegin(*distsf, MPIU_2INT, buffer1, buffer0, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(*distsf, MPIU_2INT, buffer1, buffer0, MPI_REPLACE));
    if (PetscDefined(USE_DEBUG)) {
      for (p = 0; p < *chartSize; ++p) PetscCheck(buffer0[p].rank >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Found negative root rank %" PetscInt_FMT " at local point %" PetscInt_FMT " on rank %d when making pointSF", buffer0[p].rank, p, rank);
    }
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (p = 0, q = 0; p < *chartSize; ++p) {
      if (buffer0[p].rank != rank) {
        ilocal[q]        = p;
        iremote[q].rank  = buffer0[p].rank;
        iremote[q].index = buffer0[p].index;
        q++;
      }
    }
    PetscCheck(q == nleaves, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching leaf sizes: %" PetscInt_FMT " != %" PetscInt_FMT, q, nleaves);
    PetscCall(PetscFree2(buffer0, buffer1));
    PetscCall(PetscSFCreate(comm, &pointSF));
    PetscCall(PetscSFSetFromOptions(pointSF));
    PetscCall(PetscSFSetGraph(pointSF, *chartSize, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(DMSetPointSF(*distdm, pointSF));
    {
      DM cdm;

      PetscCall(DMGetCoordinateDM(*distdm, &cdm));
      PetscCall(DMSetPointSF(cdm, pointSF));
    }
    PetscCall(PetscSFDestroy(&pointSF));
  }
  PetscCall(ISRestoreIndices(chartSizesIS, &chartSize));
  PetscCall(ISRestoreIndices(ownersIS, &owners));
  PetscCall(ISRestoreIndices(gpointsIS, &gpoints));
  PetscCall(ISDestroy(&chartSizesIS));
  PetscCall(ISDestroy(&ownersIS));
  PetscCall(ISDestroy(&gpointsIS));
  /* Record that overlap has been manually created.               */
  /* This is to pass `DMPlexCheckPointSF()`, which checks that    */
  /* pointSF does not contain cells in the leaves if overlap = 0. */
  PetscCall(DMPlexSetOverlap_Plex(*distdm, NULL, DMPLEX_OVERLAP_MANUAL));
  PetscCall(DMPlexDistributeSetDefault(*distdm, PETSC_FALSE));
  PetscCall(DMPlexReorderSetDefault(*distdm, DMPLEX_REORDER_DEFAULT_FALSE));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributionLoad, viewer, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyLoad_HDF5_Legacy_Private(DM dm, PetscViewer viewer, PetscSF *sf)
{
  MPI_Comm        comm;
  const char     *pointsName, *coneSizesName, *conesName, *orientationsName;
  IS              pointsIS, coneSizesIS, conesIS, orientationsIS;
  const PetscInt *points, *coneSizes, *cones, *orientations;
  PetscInt       *cone, *ornt;
  PetscInt        dim, N, Np, pEnd, p, q, maxConeSize = 0, c;
  PetscMPIInt     size, rank;

  PetscFunctionBegin;
  pointsName       = "order";
  coneSizesName    = "cones";
  conesName        = "cells";
  orientationsName = "orientation";
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(ISCreate(comm, &pointsIS));
  PetscCall(PetscObjectSetName((PetscObject)pointsIS, pointsName));
  PetscCall(ISCreate(comm, &coneSizesIS));
  PetscCall(PetscObjectSetName((PetscObject)coneSizesIS, coneSizesName));
  PetscCall(ISCreate(comm, &conesIS));
  PetscCall(PetscObjectSetName((PetscObject)conesIS, conesName));
  PetscCall(ISCreate(comm, &orientationsIS));
  PetscCall(PetscObjectSetName((PetscObject)orientationsIS, orientationsName));
  PetscCall(PetscViewerHDF5ReadObjectAttribute(viewer, (PetscObject)conesIS, "cell_dim", PETSC_INT, NULL, &dim));
  PetscCall(DMSetDimension(dm, dim));
  {
    /* Force serial load */
    PetscCall(PetscViewerHDF5ReadSizes(viewer, pointsName, NULL, &Np));
    PetscCall(PetscLayoutSetLocalSize(pointsIS->map, rank == 0 ? Np : 0));
    PetscCall(PetscLayoutSetSize(pointsIS->map, Np));
    pEnd = rank == 0 ? Np : 0;
    PetscCall(PetscViewerHDF5ReadSizes(viewer, coneSizesName, NULL, &Np));
    PetscCall(PetscLayoutSetLocalSize(coneSizesIS->map, rank == 0 ? Np : 0));
    PetscCall(PetscLayoutSetSize(coneSizesIS->map, Np));
    PetscCall(PetscViewerHDF5ReadSizes(viewer, conesName, NULL, &N));
    PetscCall(PetscLayoutSetLocalSize(conesIS->map, rank == 0 ? N : 0));
    PetscCall(PetscLayoutSetSize(conesIS->map, N));
    PetscCall(PetscViewerHDF5ReadSizes(viewer, orientationsName, NULL, &N));
    PetscCall(PetscLayoutSetLocalSize(orientationsIS->map, rank == 0 ? N : 0));
    PetscCall(PetscLayoutSetSize(orientationsIS->map, N));
  }
  PetscCall(ISLoad(pointsIS, viewer));
  PetscCall(ISLoad(coneSizesIS, viewer));
  PetscCall(ISLoad(conesIS, viewer));
  PetscCall(ISLoad(orientationsIS, viewer));
  /* Create Plex */
  PetscCall(DMPlexSetChart(dm, 0, pEnd));
  PetscCall(ISGetIndices(pointsIS, &points));
  PetscCall(ISGetIndices(coneSizesIS, &coneSizes));
  for (p = 0; p < pEnd; ++p) {
    PetscCall(DMPlexSetConeSize(dm, points[p], coneSizes[p]));
    maxConeSize = PetscMax(maxConeSize, coneSizes[p]);
  }
  PetscCall(DMSetUp(dm));
  PetscCall(ISGetIndices(conesIS, &cones));
  PetscCall(ISGetIndices(orientationsIS, &orientations));
  PetscCall(PetscMalloc2(maxConeSize, &cone, maxConeSize, &ornt));
  for (p = 0, q = 0; p < pEnd; ++p) {
    for (c = 0; c < coneSizes[p]; ++c, ++q) {
      cone[c] = cones[q];
      ornt[c] = orientations[q];
    }
    PetscCall(DMPlexSetCone(dm, points[p], cone));
    PetscCall(DMPlexSetConeOrientation(dm, points[p], ornt));
  }
  PetscCall(PetscFree2(cone, ornt));
  /* Create global section migration SF */
  if (sf) {
    PetscLayout layout;
    PetscInt   *globalIndices;

    PetscCall(PetscMalloc1(pEnd, &globalIndices));
    /* plex point == globalPointNumber in this case */
    for (p = 0; p < pEnd; ++p) globalIndices[p] = p;
    PetscCall(PetscLayoutCreate(comm, &layout));
    PetscCall(PetscLayoutSetSize(layout, Np));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    PetscCall(PetscLayoutSetUp(layout));
    PetscCall(PetscSFCreate(comm, sf));
    PetscCall(PetscSFSetFromOptions(*sf));
    PetscCall(PetscSFSetGraphLayout(*sf, layout, pEnd, NULL, PETSC_OWN_POINTER, globalIndices));
    PetscCall(PetscLayoutDestroy(&layout));
    PetscCall(PetscFree(globalIndices));
  }
  /* Clean-up */
  PetscCall(ISRestoreIndices(pointsIS, &points));
  PetscCall(ISRestoreIndices(coneSizesIS, &coneSizes));
  PetscCall(ISRestoreIndices(conesIS, &cones));
  PetscCall(ISRestoreIndices(orientationsIS, &orientations));
  PetscCall(ISDestroy(&pointsIS));
  PetscCall(ISDestroy(&coneSizesIS));
  PetscCall(ISDestroy(&conesIS));
  PetscCall(ISDestroy(&orientationsIS));
  /* Fill in the rest of the topology structure */
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Representation of two DMPlex strata in 0-based global numbering */
struct _n_PlexLayer {
  PetscInt     d;
  IS           conesIS, orientationsIS;
  PetscSection coneSizesSection;
  PetscLayout  vertexLayout;
  PetscSF      overlapSF, l2gSF; //TODO maybe confusing names (in DMPlex in general)
  PetscInt     offset, conesOffset, leafOffset;
};
typedef struct _n_PlexLayer *PlexLayer;

static PetscErrorCode PlexLayerDestroy(PlexLayer *layer)
{
  PetscFunctionBegin;
  if (!*layer) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSectionDestroy(&(*layer)->coneSizesSection));
  PetscCall(ISDestroy(&(*layer)->conesIS));
  PetscCall(ISDestroy(&(*layer)->orientationsIS));
  PetscCall(PetscSFDestroy(&(*layer)->overlapSF));
  PetscCall(PetscSFDestroy(&(*layer)->l2gSF));
  PetscCall(PetscLayoutDestroy(&(*layer)->vertexLayout));
  PetscCall(PetscFree(*layer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PlexLayerCreate_Private(PlexLayer *layer)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(layer));
  (*layer)->d           = -1;
  (*layer)->offset      = -1;
  (*layer)->conesOffset = -1;
  (*layer)->leafOffset  = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PlexLayerLoad_Private(PlexLayer layer, PetscViewer viewer, PetscInt d, PetscLayout pointsLayout)
{
  char         path[128];
  MPI_Comm     comm;
  const char  *coneSizesName, *conesName, *orientationsName;
  IS           coneSizesIS, conesIS, orientationsIS;
  PetscSection coneSizesSection;
  PetscLayout  vertexLayout = NULL;
  PetscInt     s;

  PetscFunctionBegin;
  coneSizesName    = "cone_sizes";
  conesName        = "cones";
  orientationsName = "orientations";
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  /* query size of next lower depth stratum (next lower dimension) */
  if (d > 0) {
    PetscInt NVertices;

    PetscCall(PetscSNPrintf(path, sizeof(path), "%" PetscInt_FMT "/%s", d - 1, coneSizesName));
    PetscCall(PetscViewerHDF5ReadSizes(viewer, path, NULL, &NVertices));
    PetscCall(PetscLayoutCreate(comm, &vertexLayout));
    PetscCall(PetscLayoutSetSize(vertexLayout, NVertices));
    PetscCall(PetscLayoutSetUp(vertexLayout));
  }

  PetscCall(PetscSNPrintf(path, sizeof(path), "%" PetscInt_FMT, d));
  PetscCall(PetscViewerHDF5PushGroup(viewer, path));

  /* create coneSizesSection from stored IS coneSizes */
  {
    const PetscInt *coneSizes;

    PetscCall(ISCreate(comm, &coneSizesIS));
    PetscCall(PetscObjectSetName((PetscObject)coneSizesIS, coneSizesName));
    if (pointsLayout) PetscCall(ISSetLayout(coneSizesIS, pointsLayout));
    PetscCall(ISLoad(coneSizesIS, viewer));
    if (!pointsLayout) PetscCall(ISGetLayout(coneSizesIS, &pointsLayout));
    PetscCall(ISGetIndices(coneSizesIS, &coneSizes));
    PetscCall(PetscSectionCreate(comm, &coneSizesSection));
    //TODO different start ?
    PetscCall(PetscSectionSetChart(coneSizesSection, 0, pointsLayout->n));
    for (s = 0; s < pointsLayout->n; ++s) PetscCall(PetscSectionSetDof(coneSizesSection, s, coneSizes[s]));
    PetscCall(PetscSectionSetUp(coneSizesSection));
    PetscCall(ISRestoreIndices(coneSizesIS, &coneSizes));
    {
      PetscLayout tmp = NULL;

      /* We need to keep the layout until the end of function */
      PetscCall(PetscLayoutReference((PetscLayout)pointsLayout, &tmp));
    }
    PetscCall(ISDestroy(&coneSizesIS));
  }

  /* use value layout of coneSizesSection as layout of cones and orientations */
  {
    PetscLayout conesLayout;

    PetscCall(PetscSectionGetValueLayout(comm, coneSizesSection, &conesLayout));
    PetscCall(ISCreate(comm, &conesIS));
    PetscCall(ISCreate(comm, &orientationsIS));
    PetscCall(PetscObjectSetName((PetscObject)conesIS, conesName));
    PetscCall(PetscObjectSetName((PetscObject)orientationsIS, orientationsName));
    PetscCall(PetscLayoutDuplicate(conesLayout, &conesIS->map));
    PetscCall(PetscLayoutDuplicate(conesLayout, &orientationsIS->map));
    PetscCall(ISLoad(conesIS, viewer));
    PetscCall(ISLoad(orientationsIS, viewer));
    PetscCall(PetscLayoutDestroy(&conesLayout));
  }

  /* check assertion that layout of points is the same as point layout of coneSizesSection */
  {
    PetscLayout pointsLayout0;
    PetscBool   flg;

    PetscCall(PetscSectionGetPointLayout(comm, coneSizesSection, &pointsLayout0));
    PetscCall(PetscLayoutCompare(pointsLayout, pointsLayout0, &flg));
    PetscCheck(flg, comm, PETSC_ERR_PLIB, "points layout != coneSizesSection point layout");
    PetscCall(PetscLayoutDestroy(&pointsLayout0));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscLayoutDestroy(&pointsLayout));

  layer->d                = d;
  layer->conesIS          = conesIS;
  layer->coneSizesSection = coneSizesSection;
  layer->orientationsIS   = orientationsIS;
  layer->vertexLayout     = vertexLayout;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PlexLayerDistribute_Private(PlexLayer layer, PetscSF cellLocalToGlobalSF)
{
  IS           newConesIS, newOrientationsIS;
  PetscSection newConeSizesSection;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)cellLocalToGlobalSF, &comm));
  PetscCall(PetscSectionCreate(comm, &newConeSizesSection));
  //TODO rename to something like ISDistribute() with optional PetscSection argument
  PetscCall(DMPlexDistributeFieldIS(NULL, cellLocalToGlobalSF, layer->coneSizesSection, layer->conesIS, newConeSizesSection, &newConesIS));
  PetscCall(DMPlexDistributeFieldIS(NULL, cellLocalToGlobalSF, layer->coneSizesSection, layer->orientationsIS, newConeSizesSection, &newOrientationsIS));

  PetscCall(PetscObjectSetName((PetscObject)newConeSizesSection, ((PetscObject)layer->coneSizesSection)->name));
  PetscCall(PetscObjectSetName((PetscObject)newConesIS, ((PetscObject)layer->conesIS)->name));
  PetscCall(PetscObjectSetName((PetscObject)newOrientationsIS, ((PetscObject)layer->orientationsIS)->name));
  PetscCall(PetscSectionDestroy(&layer->coneSizesSection));
  PetscCall(ISDestroy(&layer->conesIS));
  PetscCall(ISDestroy(&layer->orientationsIS));
  layer->coneSizesSection = newConeSizesSection;
  layer->conesIS          = newConesIS;
  layer->orientationsIS   = newOrientationsIS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  //TODO share code with DMPlexBuildFromCellListParallel()
  #include <petsc/private/hashseti.h>
static PetscErrorCode PlexLayerCreateSFs_Private(PlexLayer layer, PetscSF *vertexOverlapSF, PetscSF *sfXC)
{
  PetscLayout     vertexLayout     = layer->vertexLayout;
  PetscSection    coneSection      = layer->coneSizesSection;
  IS              cellVertexData   = layer->conesIS;
  IS              coneOrientations = layer->orientationsIS;
  PetscSF         vl2gSF, vOverlapSF;
  PetscInt       *verticesAdj;
  PetscInt        i, n, numVerticesAdj;
  const PetscInt *cvd, *co = NULL;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)coneSection, &comm));
  PetscCall(PetscSectionGetStorageSize(coneSection, &n));
  {
    PetscInt n0;

    PetscCall(ISGetLocalSize(cellVertexData, &n0));
    PetscCheck(n == n0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local size of IS cellVertexData = %" PetscInt_FMT " != %" PetscInt_FMT " = storage size of PetscSection coneSection", n0, n);
    PetscCall(ISGetIndices(cellVertexData, &cvd));
  }
  if (coneOrientations) {
    PetscInt n0;

    PetscCall(ISGetLocalSize(coneOrientations, &n0));
    PetscCheck(n == n0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local size of IS coneOrientations = %" PetscInt_FMT " != %" PetscInt_FMT " = storage size of PetscSection coneSection", n0, n);
    PetscCall(ISGetIndices(coneOrientations, &co));
  }
  /* Get/check global number of vertices */
  {
    PetscInt NVerticesInCells = PETSC_MIN_INT;

    /* NVerticesInCells = max(cellVertexData) + 1 */
    for (i = 0; i < n; i++)
      if (cvd[i] > NVerticesInCells) NVerticesInCells = cvd[i];
    ++NVerticesInCells;
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &NVerticesInCells, 1, MPIU_INT, MPI_MAX, comm));

    if (vertexLayout->n == PETSC_DECIDE && vertexLayout->N == PETSC_DECIDE) vertexLayout->N = NVerticesInCells;
    else
      PetscCheck(vertexLayout->N == PETSC_DECIDE || vertexLayout->N >= NVerticesInCells, comm, PETSC_ERR_ARG_SIZ, "Specified global number of vertices %" PetscInt_FMT " must be greater than or equal to the number of unique vertices in the cell-vertex dataset %" PetscInt_FMT,
                 vertexLayout->N, NVerticesInCells);
    PetscCall(PetscLayoutSetUp(vertexLayout));
  }
  /* Find locally unique vertices in cellVertexData */
  {
    PetscHSetI vhash;
    PetscInt   off = 0;

    PetscCall(PetscHSetICreate(&vhash));
    for (i = 0; i < n; ++i) PetscCall(PetscHSetIAdd(vhash, cvd[i]));
    PetscCall(PetscHSetIGetSize(vhash, &numVerticesAdj));
    PetscCall(PetscMalloc1(numVerticesAdj, &verticesAdj));
    PetscCall(PetscHSetIGetElems(vhash, &off, verticesAdj));
    PetscAssert(off == numVerticesAdj, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Assertion failed: off == numVerticesAdj (%" PetscInt_FMT " != %" PetscInt_FMT ")", off, numVerticesAdj);
    PetscCall(PetscHSetIDestroy(&vhash));
  }
  /* We must sort vertices to preserve numbering */
  PetscCall(PetscSortInt(numVerticesAdj, verticesAdj));
  /* Connect locally unique vertices with star forest */
  PetscCall(PetscSFCreateByMatchingIndices(vertexLayout, numVerticesAdj, verticesAdj, NULL, 0, numVerticesAdj, verticesAdj, NULL, 0, &vl2gSF, &vOverlapSF));
  PetscCall(PetscObjectSetName((PetscObject)vOverlapSF, "overlapSF"));
  PetscCall(PetscObjectSetName((PetscObject)vl2gSF, "localToGlobalSF"));

  PetscCall(PetscFree(verticesAdj));
  *vertexOverlapSF = vOverlapSF;
  *sfXC            = vl2gSF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PlexLayerCreateCellSFs_Private(PlexLayer layer, PetscSF *cellOverlapSF, PetscSF *cellLocalToGlobalSF)
{
  PetscSection coneSection = layer->coneSizesSection;
  PetscInt     nCells;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)coneSection, &comm));
  {
    PetscInt cStart;

    PetscCall(PetscSectionGetChart(coneSection, &cStart, &nCells));
    PetscCheck(cStart == 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "coneSection must start at 0");
  }
  /* Create overlapSF as empty SF with the right number of roots */
  PetscCall(PetscSFCreate(comm, cellOverlapSF));
  PetscCall(PetscSFSetGraph(*cellOverlapSF, nCells, 0, NULL, PETSC_USE_POINTER, NULL, PETSC_USE_POINTER));
  PetscCall(PetscSFSetUp(*cellOverlapSF));
  /* Create localToGlobalSF as identity mapping */
  {
    PetscLayout map;

    PetscCall(PetscLayoutCreateFromSizes(comm, nCells, PETSC_DECIDE, 1, &map));
    PetscCall(PetscSFCreateFromLayouts(map, map, cellLocalToGlobalSF));
    PetscCall(PetscSFSetUp(*cellLocalToGlobalSF));
    PetscCall(PetscLayoutDestroy(&map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyBuildFromLayers_Private(DM dm, PetscInt depth, PlexLayer *layers, IS strataPermutation)
{
  const PetscInt *permArr;
  PetscInt        d, nPoints;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(ISGetIndices(strataPermutation, &permArr));

  /* Count points, strata offsets and cones offsets (taking strataPermutation into account) */
  {
    PetscInt stratumOffset = 0;
    PetscInt conesOffset   = 0;

    for (d = 0; d <= depth; d++) {
      const PetscInt  e = permArr[d];
      const PlexLayer l = layers[e];
      PetscInt        lo, n, size;

      PetscCall(PetscSectionGetChart(l->coneSizesSection, &lo, &n));
      PetscCall(PetscSectionGetStorageSize(l->coneSizesSection, &size));
      PetscCheck(lo == 0, comm, PETSC_ERR_PLIB, "starting point should be 0 in coneSizesSection %" PetscInt_FMT, d);
      l->offset      = stratumOffset;
      l->conesOffset = conesOffset;
      stratumOffset += n;
      conesOffset += size;
    }
    nPoints = stratumOffset;
  }

  /* Set interval for all plex points */
  //TODO we should store starting point of plex
  PetscCall(DMPlexSetChart(dm, 0, nPoints));

  /* Set up plex coneSection from layer coneSections */
  {
    PetscSection coneSection;

    PetscCall(DMPlexGetConeSection(dm, &coneSection));
    for (d = 0; d <= depth; d++) {
      const PlexLayer l = layers[d];
      PetscInt        n, q;

      PetscCall(PetscSectionGetChart(l->coneSizesSection, NULL, &n));
      for (q = 0; q < n; q++) {
        const PetscInt p = l->offset + q;
        PetscInt       coneSize;

        PetscCall(PetscSectionGetDof(l->coneSizesSection, q, &coneSize));
        PetscCall(PetscSectionSetDof(coneSection, p, coneSize));
      }
    }
  }
  //TODO this is terrible, DMSetUp_Plex() should be DMPlexSetUpSections() or so
  PetscCall(DMSetUp(dm));

  /* Renumber cones points from layer-global numbering to plex-local numbering */
  {
    PetscInt *cones, *ornts;

    PetscCall(DMPlexGetCones(dm, &cones));
    PetscCall(DMPlexGetConeOrientations(dm, &ornts));
    for (d = 1; d <= depth; d++) {
      const PlexLayer l = layers[d];
      PetscInt        i, lConesSize;
      PetscInt       *lCones;
      const PetscInt *lOrnts;
      PetscInt       *pCones = &cones[l->conesOffset];
      PetscInt       *pOrnts = &ornts[l->conesOffset];

      PetscCall(PetscSectionGetStorageSize(l->coneSizesSection, &lConesSize));
      /* Get cones in local plex numbering */
      {
        ISLocalToGlobalMapping l2g;
        PetscLayout            vertexLayout = l->vertexLayout;
        PetscSF                vertexSF     = layers[d - 1]->l2gSF; /* vertices of this layer are cells of previous layer */
        const PetscInt        *gCones;
        PetscInt               lConesSize0;

        PetscCall(ISGetLocalSize(l->conesIS, &lConesSize0));
        PetscCheck(lConesSize0 == lConesSize, comm, PETSC_ERR_PLIB, "layer %" PetscInt_FMT " size(conesIS) = %" PetscInt_FMT " != %" PetscInt_FMT " = storageSize(coneSizesSection)", d, lConesSize0, lConesSize);
        PetscCall(ISGetLocalSize(l->orientationsIS, &lConesSize0));
        PetscCheck(lConesSize0 == lConesSize, comm, PETSC_ERR_PLIB, "layer %" PetscInt_FMT " size(orientationsIS) = %" PetscInt_FMT " != %" PetscInt_FMT " = storageSize(coneSizesSection)", d, lConesSize0, lConesSize);

        PetscCall(PetscMalloc1(lConesSize, &lCones));
        PetscCall(ISGetIndices(l->conesIS, &gCones));
        PetscCall(ISLocalToGlobalMappingCreateSF(vertexSF, vertexLayout->rstart, &l2g));
        PetscCall(ISGlobalToLocalMappingApply(l2g, IS_GTOLM_MASK, lConesSize, gCones, &lConesSize0, lCones));
        PetscCheck(lConesSize0 == lConesSize, comm, PETSC_ERR_PLIB, "global to local does not cover all indices (%" PetscInt_FMT " of %" PetscInt_FMT ")", lConesSize0, lConesSize);
        PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
        PetscCall(ISRestoreIndices(l->conesIS, &gCones));
      }
      PetscCall(ISGetIndices(l->orientationsIS, &lOrnts));
      /* Set cones, need to add stratum offset */
      for (i = 0; i < lConesSize; i++) {
        pCones[i] = lCones[i] + layers[d - 1]->offset; /* cone points of current layer are points of previous layer */
        pOrnts[i] = lOrnts[i];
      }
      PetscCall(PetscFree(lCones));
      PetscCall(ISRestoreIndices(l->orientationsIS, &lOrnts));
    }
  }
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  PetscCall(ISRestoreIndices(strataPermutation, &permArr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PlexLayerConcatenateSFs_Private(MPI_Comm comm, PetscInt depth, PlexLayer layers[], IS strataPermutation, PetscSF *overlapSF, PetscSF *l2gSF)
{
  PetscInt        d;
  PetscSF        *osfs, *lsfs;
  PetscInt       *leafOffsets;
  const PetscInt *permArr;

  PetscFunctionBegin;
  PetscCall(ISGetIndices(strataPermutation, &permArr));
  PetscCall(PetscCalloc3(depth + 1, &osfs, depth + 1, &lsfs, depth + 1, &leafOffsets));
  for (d = 0; d <= depth; d++) {
    const PetscInt e = permArr[d];

    PetscAssert(e == layers[e]->d, PETSC_COMM_SELF, PETSC_ERR_PLIB, "assertion: e == layers[e]->d");
    osfs[d]        = layers[e]->overlapSF;
    lsfs[d]        = layers[e]->l2gSF;
    leafOffsets[d] = layers[e]->offset;
  }
  PetscCall(PetscSFConcatenate(comm, depth + 1, osfs, PETSCSF_CONCATENATE_ROOTMODE_LOCAL, leafOffsets, overlapSF));
  PetscCall(PetscSFConcatenate(comm, depth + 1, lsfs, PETSCSF_CONCATENATE_ROOTMODE_GLOBAL, leafOffsets, l2gSF));
  PetscCall(PetscFree3(osfs, lsfs, leafOffsets));
  PetscCall(ISRestoreIndices(strataPermutation, &permArr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTopologyLoad_HDF5_Private(DM dm, PetscViewer viewer, PetscSF *sfXC)
{
  PlexLayer  *layers;
  IS          strataPermutation;
  PetscLayout pointsLayout;
  PetscInt    depth;
  PetscInt    d;
  MPI_Comm    comm;

  PetscFunctionBegin;
  {
    PetscInt dim;

    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "depth", PETSC_INT, NULL, &depth));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "cell_dim", PETSC_INT, NULL, &dim));
    PetscCall(DMSetDimension(dm, dim));
  }
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  {
    IS spOnComm;

    PetscCall(ISCreate(comm, &spOnComm));
    PetscCall(PetscObjectSetName((PetscObject)spOnComm, "permutation"));
    PetscCall(ISLoad(spOnComm, viewer));
    /* have the same serial IS on every rank */
    PetscCall(ISAllGather(spOnComm, &strataPermutation));
    PetscCall(PetscObjectSetName((PetscObject)strataPermutation, ((PetscObject)spOnComm)->name));
    PetscCall(ISDestroy(&spOnComm));
  }

  /* Create layers, load raw data for each layer */
  PetscCall(PetscViewerHDF5PushGroup(viewer, "strata"));
  PetscCall(PetscMalloc1(depth + 1, &layers));
  for (d = depth, pointsLayout = NULL; d >= 0; pointsLayout = layers[d]->vertexLayout, d--) {
    PetscCall(PlexLayerCreate_Private(&layers[d]));
    PetscCall(PlexLayerLoad_Private(layers[d], viewer, d, pointsLayout));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer)); /* strata */

  for (d = depth; d >= 0; d--) {
    /* Redistribute cells and vertices for each applicable layer */
    if (d < depth) PetscCall(PlexLayerDistribute_Private(layers[d], layers[d]->l2gSF));
    /* Create vertex overlap SF and vertex localToGlobal SF for each applicable layer */
    if (d > 0) PetscCall(PlexLayerCreateSFs_Private(layers[d], &layers[d - 1]->overlapSF, &layers[d - 1]->l2gSF));
  }
  /* Build trivial SFs for the cell layer as well */
  PetscCall(PlexLayerCreateCellSFs_Private(layers[depth], &layers[depth]->overlapSF, &layers[depth]->l2gSF));

  /* Build DMPlex topology from the layers */
  PetscCall(DMPlexTopologyBuildFromLayers_Private(dm, depth, layers, strataPermutation));

  /* Build overall point SF alias overlap SF */
  {
    PetscSF overlapSF;

    PetscCall(PlexLayerConcatenateSFs_Private(comm, depth, layers, strataPermutation, &overlapSF, sfXC));
    PetscCall(DMSetPointSF(dm, overlapSF));
    PetscCall(PetscSFDestroy(&overlapSF));
  }

  for (d = depth; d >= 0; d--) PetscCall(PlexLayerDestroy(&layers[d]));
  PetscCall(PetscFree(layers));
  PetscCall(ISDestroy(&strataPermutation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexTopologyLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF *sfXC)
{
  DMPlexStorageVersion version;
  const char          *topologydm_name;
  char                 group[PETSC_MAX_PATH_LEN];
  PetscSF              sfwork = NULL;

  PetscFunctionBegin;
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscViewerHDF5GetDMPlexStorageVersionReading(viewer, &version));
  if (DMPlexStorageVersionGE(version, 2, 0, 0)) {
    PetscCall(PetscSNPrintf(group, sizeof(group), "topologies/%s", topologydm_name));
  } else {
    PetscCall(PetscStrncpy(group, "/", sizeof(group)));
  }
  PetscCall(PetscViewerHDF5PushGroup(viewer, group));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "topology"));
  if (version->major < 3) {
    PetscCall(DMPlexTopologyLoad_HDF5_Legacy_Private(dm, viewer, &sfwork));
  } else {
    /* since DMPlexStorageVersion 3.0.0 */
    PetscCall(DMPlexTopologyLoad_HDF5_Private(dm, viewer, &sfwork));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer)); /* "topology" */

  if (DMPlexStorageVersionGE(version, 2, 0, 0)) {
    DM          distdm;
    PetscSF     distsf;
    const char *distribution_name;
    PetscBool   exists;

    PetscCall(DMPlexDistributionGetName(dm, &distribution_name));
    /* this group is guaranteed to be present since DMPlexStorageVersion 2.1.0 */
    PetscCall(PetscViewerHDF5PushGroup(viewer, "distributions"));
    PetscCall(PetscViewerHDF5HasGroup(viewer, NULL, &exists));
    if (exists) {
      PetscCall(PetscViewerHDF5PushGroup(viewer, distribution_name));
      PetscCall(DMPlexDistributionLoad_HDF5_Private(dm, viewer, sfwork, &distsf, &distdm));
      if (distdm) {
        PetscCall(DMPlexReplace_Internal(dm, &distdm));
        PetscCall(PetscSFDestroy(&sfwork));
        sfwork = distsf;
      }
      PetscCall(PetscViewerHDF5PopGroup(viewer)); /* distribution_name */
    }
    PetscCall(PetscViewerHDF5PopGroup(viewer)); /* "distributions" */
  }
  if (sfXC) {
    *sfXC = sfwork;
  } else {
    PetscCall(PetscSFDestroy(&sfwork));
  }

  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* If the file is old, it not only has different path to the coordinates, but   */
/* does not contain coordinateDMs, so must fall back to the old implementation. */
static PetscErrorCode DMPlexCoordinatesLoad_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
{
  PetscSection coordSection;
  Vec          coordinates;
  PetscReal    lengthScale;
  PetscInt     spatialDim, N, numVertices, vStart, vEnd, v;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  /* Read geometry */
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/geometry"));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), &coordinates));
  PetscCall(PetscObjectSetName((PetscObject)coordinates, "vertices"));
  {
    /* Force serial load */
    PetscCall(PetscViewerHDF5ReadSizes(viewer, "vertices", &spatialDim, &N));
    PetscCall(VecSetSizes(coordinates, rank == 0 ? N : 0, N));
    PetscCall(VecSetBlockSize(coordinates, spatialDim));
  }
  PetscCall(VecLoad(coordinates, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  PetscCall(VecScale(coordinates, 1.0 / lengthScale));
  PetscCall(VecGetLocalSize(coordinates, &numVertices));
  PetscCall(VecGetBlockSize(coordinates, &spatialDim));
  numVertices /= spatialDim;
  /* Create coordinates */
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCheck(numVertices == vEnd - vStart, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of coordinates loaded %" PetscInt_FMT " does not match number of vertices %" PetscInt_FMT, numVertices, vEnd - vStart);
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, spatialDim));
  PetscCall(PetscSectionSetChart(coordSection, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, spatialDim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, spatialDim));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(DMSetCoordinates(dm, coordinates));
  PetscCall(VecDestroy(&coordinates));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCoordinatesLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF sfXC)
{
  DMPlexStorageVersion version;
  DM                   cdm;
  Vec                  coords;
  PetscInt             blockSize;
  PetscReal            lengthScale;
  PetscSF              lsf;
  const char          *topologydm_name;
  char                *coordinatedm_name, *coordinates_name;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetDMPlexStorageVersionReading(viewer, &version));
  if (!DMPlexStorageVersionGE(version, 2, 0, 0)) {
    PetscCall(DMPlexCoordinatesLoad_HDF5_Legacy_Private(dm, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* else: since DMPlexStorageVersion 2.0.0 */
  PetscCheck(sfXC, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "PetscSF must be given for parallel load");
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "coordinateDMName", PETSC_STRING, NULL, &coordinatedm_name));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "coordinatesName", PETSC_STRING, NULL, &coordinates_name));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(PetscObjectSetName((PetscObject)cdm, coordinatedm_name));
  PetscCall(PetscFree(coordinatedm_name));
  /* lsf: on-disk data -> in-memory local vector associated with cdm's local section */
  PetscCall(DMPlexSectionLoad(dm, viewer, cdm, sfXC, NULL, &lsf));
  PetscCall(DMCreateLocalVector(cdm, &coords));
  PetscCall(PetscObjectSetName((PetscObject)coords, coordinates_name));
  PetscCall(PetscFree(coordinates_name));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(DMPlexLocalVectorLoad(dm, viewer, cdm, lsf, coords));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale));
  PetscCall(VecScale(coords, 1.0 / lengthScale));
  PetscCall(DMSetCoordinatesLocal(dm, coords));
  PetscCall(VecGetBlockSize(coords, &blockSize));
  PetscCall(DMSetCoordinateDim(dm, blockSize));
  PetscCall(VecDestroy(&coords));
  PetscCall(PetscSFDestroy(&lsf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexLoad_HDF5_Internal(DM dm, PetscViewer viewer)
{
  DMPlexStorageVersion version;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetDMPlexStorageVersionReading(viewer, &version));
  if (!DMPlexStorageVersionGE(version, 2, 0, 0)) {
    PetscCall(DMPlexTopologyLoad_HDF5_Internal(dm, viewer, NULL));
    PetscCall(DMPlexLabelsLoad_HDF5_Internal(dm, viewer, NULL));
    PetscCall(DMPlexCoordinatesLoad_HDF5_Legacy_Private(dm, viewer));
  } else {
    PetscSF sfXC;

    /* since DMPlexStorageVersion 2.0.0 */
    PetscCall(DMPlexTopologyLoad_HDF5_Internal(dm, viewer, &sfXC));
    PetscCall(DMPlexLabelsLoad_HDF5_Internal(dm, viewer, sfXC));
    PetscCall(DMPlexCoordinatesLoad_HDF5_Internal(dm, viewer, sfXC));
    PetscCall(PetscSFDestroy(&sfXC));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexSectionLoad_HDF5_Internal_CreateDataSF(PetscSection rootSection, PetscLayout layout, PetscInt globalOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm  comm;
  PetscInt  pStart, pEnd, p, m;
  PetscInt *goffs, *ilocal;
  PetscBool rootIncludeConstraints, leafIncludeConstraints;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)leafSection, &comm));
  PetscCall(PetscSectionGetChart(leafSection, &pStart, &pEnd));
  PetscCall(PetscSectionGetIncludesConstraints(rootSection, &rootIncludeConstraints));
  PetscCall(PetscSectionGetIncludesConstraints(leafSection, &leafIncludeConstraints));
  if (rootIncludeConstraints && leafIncludeConstraints) PetscCall(PetscSectionGetStorageSize(leafSection, &m));
  else PetscCall(PetscSectionGetConstrainedStorageSize(leafSection, &m));
  PetscCall(PetscMalloc1(m, &ilocal));
  PetscCall(PetscMalloc1(m, &goffs));
  /* Currently, PetscSFDistributeSection() returns globalOffsets[] only */
  /* for the top-level section (not for each field), so one must have   */
  /* rootSection->pointMajor == PETSC_TRUE.                             */
  PetscCheck(rootSection->pointMajor, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for field major ordering");
  /* Currently, we also assume that leafSection->pointMajor == PETSC_TRUE. */
  PetscCheck(leafSection->pointMajor, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for field major ordering");
  for (p = pStart, m = 0; p < pEnd; ++p) {
    PetscInt        dof, cdof, i, j, off, goff;
    const PetscInt *cinds;

    PetscCall(PetscSectionGetDof(leafSection, p, &dof));
    if (dof < 0) continue;
    goff = globalOffsets[p - pStart];
    PetscCall(PetscSectionGetOffset(leafSection, p, &off));
    PetscCall(PetscSectionGetConstraintDof(leafSection, p, &cdof));
    PetscCall(PetscSectionGetConstraintIndices(leafSection, p, &cinds));
    for (i = 0, j = 0; i < dof; ++i) {
      PetscBool constrained = (PetscBool)(j < cdof && i == cinds[j]);

      if (!constrained || (leafIncludeConstraints && rootIncludeConstraints)) {
        ilocal[m]  = off++;
        goffs[m++] = goff++;
      } else if (leafIncludeConstraints && !rootIncludeConstraints) ++off;
      else if (!leafIncludeConstraints && rootIncludeConstraints) ++goff;
      if (constrained) ++j;
    }
  }
  PetscCall(PetscSFCreate(comm, sectionSF));
  PetscCall(PetscSFSetFromOptions(*sectionSF));
  PetscCall(PetscSFSetGraphLayout(*sectionSF, layout, m, ilocal, PETSC_OWN_POINTER, goffs));
  PetscCall(PetscFree(goffs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexSectionLoad_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sfXB, PetscSF *gsf, PetscSF *lsf)
{
  MPI_Comm     comm;
  PetscMPIInt  size, rank;
  const char  *topologydm_name;
  const char  *sectiondm_name;
  PetscSection sectionA, sectionB;
  PetscInt     nX, n, i;
  PetscSF      sfAB;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "dms"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  /* A: on-disk points                        */
  /* X: list of global point numbers, [0, NX) */
  /* B: plex points                           */
  /* Load raw section (sectionA)              */
  PetscCall(PetscSectionCreate(comm, &sectionA));
  PetscCall(PetscSectionLoad(sectionA, viewer));
  PetscCall(PetscSectionGetChart(sectionA, NULL, &n));
  /* Create sfAB: A -> B */
  #if defined(PETSC_USE_DEBUG)
  {
    PetscInt N, N1;

    PetscCall(PetscViewerHDF5ReadSizes(viewer, "order", NULL, &N1));
    PetscCall(MPIU_Allreduce(&n, &N, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(N1 == N, comm, PETSC_ERR_ARG_SIZ, "Mismatching sizes: on-disk order array size (%" PetscInt_FMT ") != number of loaded section points (%" PetscInt_FMT ")", N1, N);
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
    PetscCall(ISCreate(comm, &orderIS));
    PetscCall(PetscObjectSetName((PetscObject)orderIS, "order"));
    PetscCall(PetscLayoutSetLocalSize(orderIS->map, n));
    PetscCall(ISLoad(orderIS, viewer));
    PetscCall(PetscLayoutCreate(comm, &layout));
    PetscCall(PetscSFGetGraph(sfXB, &nX, NULL, NULL, NULL));
    PetscCall(PetscLayoutSetLocalSize(layout, nX));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    PetscCall(PetscLayoutSetUp(layout));
    PetscCall(PetscSFCreate(comm, &sfXA));
    PetscCall(ISGetIndices(orderIS, &gpoints));
    PetscCall(PetscSFSetGraphLayout(sfXA, layout, n, NULL, PETSC_OWN_POINTER, gpoints));
    PetscCall(ISRestoreIndices(orderIS, &gpoints));
    PetscCall(ISDestroy(&orderIS));
    PetscCall(PetscLayoutDestroy(&layout));
    PetscCall(PetscMalloc1(n, &owners));
    PetscCall(PetscMalloc1(nX, &buffer));
    for (i = 0; i < n; ++i) {
      owners[i].rank  = rank;
      owners[i].index = i;
    }
    for (i = 0; i < nX; ++i) {
      buffer[i].rank  = -1;
      buffer[i].index = -1;
    }
    PetscCall(PetscSFReduceBegin(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC));
    PetscCall(PetscSFReduceEnd(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC));
    PetscCall(PetscSFDestroy(&sfXA));
    PetscCall(PetscFree(owners));
    for (i = 0, nleaves = 0; i < nX; ++i)
      if (buffer[i].rank >= 0) nleaves++;
    PetscCall(PetscMalloc1(nleaves, &ilocal));
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (i = 0, nleaves = 0; i < nX; ++i) {
      if (buffer[i].rank >= 0) {
        ilocal[nleaves]        = i;
        iremote[nleaves].rank  = buffer[i].rank;
        iremote[nleaves].index = buffer[i].index;
        nleaves++;
      }
    }
    PetscCall(PetscFree(buffer));
    PetscCall(PetscSFCreate(comm, &sfAX));
    PetscCall(PetscSFSetFromOptions(sfAX));
    PetscCall(PetscSFSetGraph(sfAX, n, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(PetscSFCompose(sfAX, sfXB, &sfAB));
    PetscCall(PetscSFDestroy(&sfAX));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  /* Create plex section (sectionB) */
  PetscCall(DMGetLocalSection(sectiondm, &sectionB));
  if (lsf || gsf) {
    PetscLayout layout;
    PetscInt    M, m;
    PetscInt   *offsetsA;
    PetscBool   includesConstraintsA;

    PetscCall(PetscSFDistributeSection(sfAB, sectionA, &offsetsA, sectionB));
    PetscCall(PetscSectionGetIncludesConstraints(sectionA, &includesConstraintsA));
    if (includesConstraintsA) PetscCall(PetscSectionGetStorageSize(sectionA, &m));
    else PetscCall(PetscSectionGetConstrainedStorageSize(sectionA, &m));
    PetscCall(MPIU_Allreduce(&m, &M, 1, MPIU_INT, MPI_SUM, comm));
    PetscCall(PetscLayoutCreate(comm, &layout));
    PetscCall(PetscLayoutSetSize(layout, M));
    PetscCall(PetscLayoutSetUp(layout));
    if (lsf) {
      PetscSF lsfABdata;

      PetscCall(DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, sectionB, &lsfABdata));
      *lsf = lsfABdata;
    }
    if (gsf) {
      PetscSection gsectionB, gsectionB1;
      PetscBool    includesConstraintsB;
      PetscSF      gsfABdata, pointsf;

      PetscCall(DMGetGlobalSection(sectiondm, &gsectionB1));
      PetscCall(PetscSectionGetIncludesConstraints(gsectionB1, &includesConstraintsB));
      PetscCall(DMGetPointSF(sectiondm, &pointsf));
      PetscCall(PetscSectionCreateGlobalSection(sectionB, pointsf, includesConstraintsB, PETSC_TRUE, &gsectionB));
      PetscCall(DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, gsectionB, &gsfABdata));
      PetscCall(PetscSectionDestroy(&gsectionB));
      *gsf = gsfABdata;
    }
    PetscCall(PetscLayoutDestroy(&layout));
    PetscCall(PetscFree(offsetsA));
  } else {
    PetscCall(PetscSFDistributeSection(sfAB, sectionA, NULL, sectionB));
  }
  PetscCall(PetscSFDestroy(&sfAB));
  PetscCall(PetscSectionDestroy(&sectionA));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetHDF5Name_Private(dm, &topologydm_name));
  PetscCall(PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name));
  PetscCall(PetscObjectGetName((PetscObject)vec, &vec_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "topologies"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, topologydm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "dms"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, sectiondm_name));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "vecs"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, vec_name));
  PetscCall(VecCreate(comm, &vecA));
  PetscCall(PetscObjectSetName((PetscObject)vecA, vec_name));
  PetscCall(PetscSFGetGraph(sf, &mA, &m, &ilocal, NULL));
  /* Check consistency */
  {
    PetscSF  pointsf, pointsf1;
    PetscInt m1, i, j;

    PetscCall(DMGetPointSF(dm, &pointsf));
    PetscCall(DMGetPointSF(sectiondm, &pointsf1));
    PetscCheck(pointsf1 == pointsf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  #if defined(PETSC_USE_DEBUG)
    {
      PetscInt MA, MA1;

      PetscCall(MPIU_Allreduce(&mA, &MA, 1, MPIU_INT, MPI_SUM, comm));
      PetscCall(PetscViewerHDF5ReadSizes(viewer, vec_name, NULL, &MA1));
      PetscCheck(MA1 == MA, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total SF root size (%" PetscInt_FMT ") != On-disk vector data size (%" PetscInt_FMT ")", MA, MA1);
    }
  #endif
    PetscCall(VecGetLocalSize(vec, &m1));
    PetscCheck(m1 >= m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Target vector size (%" PetscInt_FMT ") < SF leaf size (%" PetscInt_FMT ")", m1, m);
    for (i = 0; i < m; ++i) {
      j = ilocal ? ilocal[i] : i;
      PetscCheck(j >= 0 && j < m1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Leaf's %" PetscInt_FMT "-th index, %" PetscInt_FMT ", not in [%" PetscInt_FMT ", %" PetscInt_FMT ")", i, j, (PetscInt)0, m1);
    }
  }
  PetscCall(VecSetSizes(vecA, mA, PETSC_DECIDE));
  PetscCall(VecLoad(vecA, viewer));
  PetscCall(VecGetArrayRead(vecA, &src));
  PetscCall(VecGetArray(vec, &dest));
  PetscCall(PetscSFBcastBegin(sf, MPIU_SCALAR, src, dest, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, src, dest, MPI_REPLACE));
  PetscCall(VecRestoreArray(vec, &dest));
  PetscCall(VecRestoreArrayRead(vecA, &src));
  PetscCall(VecDestroy(&vecA));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "blockSize", PETSC_INT, NULL, (void *)&bs));
  PetscCall(VecSetBlockSize(vec, bs));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
