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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTokenCreate(str, '.', &t);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = PetscTokenFind(t, &ts);CHKERRQ(ierr);
    PetscCheckFalse(!ts,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Malformed version string %s", str);
    ierr = PetscOptionsStringToInt(ts, &ti[i]);CHKERRQ(ierr);
  }
  ierr = PetscTokenFind(t, &ts);CHKERRQ(ierr);
  PetscCheckFalse(ts,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Malformed version string %s", str);
  ierr = PetscTokenDestroy(&t);CHKERRQ(ierr);
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
  ierr = PetscStrcpy(fileVersion, DMPLEX_STORAGE_VERSION_STABLE);CHKERRQ(ierr);
  ierr = PetscViewerHDF5HasAttribute(viewer, NULL, ATTR_NAME, &fileHasVersion);CHKERRQ(ierr);
  if (fileHasVersion) {
    char *tmp;

    ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, NULL, &tmp);CHKERRQ(ierr);
    ierr = PetscStrcpy(fileVersion, tmp);CHKERRQ(ierr);
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  }
  ierr = PetscStrcpy(optVersion, fileVersion);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5 Viewer Options","PetscViewer");CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_view_hdf5_storage_version","DMPlex HDF5 viewer storage version",NULL,optVersion,optVersion,sizeof(optVersion),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!fileHasVersion) {
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, optVersion);CHKERRQ(ierr);
  } else {
    PetscBool flg;

    ierr = PetscStrcmp(fileVersion, optVersion, &flg);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PetscObjectComm((PetscObject)dm), PETSC_ERR_FILE_UNEXPECTED, "User requested DMPlex storage version %s but file already has version %s - cannot mix versions", optVersion, fileVersion);
  }
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "petsc_version_git", PETSC_STRING, PETSC_VERSION_GIT);CHKERRQ(ierr);
  ierr = DMPlexStorageVersionParseString_Private(dm, optVersion, version);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexStorageVersionGet_Private(DM dm, PetscViewer viewer, DMPlexStorageVersion *version)
{
  const char      ATTR_NAME[]       = "dmplex_storage_version";
  char           *defaultVersion;
  char           *versionString;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  //TODO string HDF5 attribute handling is terrible and should be redesigned
  ierr = PetscStrallocpy("1.0.0", &defaultVersion);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, ATTR_NAME, PETSC_STRING, &defaultVersion, &versionString);CHKERRQ(ierr);
  ierr = DMPlexStorageVersionParseString_Private(dm, versionString, version);CHKERRQ(ierr);
  ierr = PetscFree(versionString);CHKERRQ(ierr);
  ierr = PetscFree(defaultVersion);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetHDF5Name_Private(DM dm, const char *name[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (((PetscObject)dm)->name) {
    ierr = PetscObjectGetName((PetscObject)dm, name);CHKERRQ(ierr);
  } else {
    *name = "plex";
  }
  PetscFunctionReturn(0);
}

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
  if (rank == 0) {
    PetscReal timeScale;
    PetscBool istime;

    ierr = PetscStrncmp(seqname, "time", 5, &istime);CHKERRQ(ierr);
    if (istime) {ierr = DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale);CHKERRQ(ierr); value *= timeScale;}
    ierr = VecSetValue(stamp, 0, value, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(stamp);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(stamp);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr); /* seqnum < 0 jumps out above */
  ierr = VecView(stamp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
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
  ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);  /* seqnum < 0 jumps out above */
  ierr = VecLoad(stamp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  if (rank == 0) {
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
  ierr = DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar) seqval, viewer);CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  }
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
      const char *fname, *fgroup, *componentName;
      char        subname[PETSC_MAX_PATH_LEN];
      PetscInt    pStart, pEnd, Nc, c;

      ierr = DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft);CHKERRQ(ierr);
      fgroup = (ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_POINT_FIELD) ? "/vertex_fields" : "/cell_fields";
      ierr = PetscSectionGetFieldName(section, f, &fname);CHKERRQ(ierr);
      if (!fname) continue;
      ierr = PetscViewerHDF5PushGroup(viewer, fgroup);CHKERRQ(ierr);
      if (cutLabel) {
        const PetscScalar *ga;
        PetscScalar       *suba;
        PetscInt          gstart, subSize = 0, extSize = 0, subOff = 0, newOff = 0, p;

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

      /* Output the component names in the field if available */
      ierr = PetscSectionGetFieldComponents(section, f, &Nc);CHKERRQ(ierr);
      for (c = 0; c < Nc; ++c){
        char componentNameLabel[PETSC_MAX_PATH_LEN];
        ierr = PetscSectionGetComponentName(section, f, c, &componentName);CHKERRQ(ierr);
        ierr = PetscSNPrintf(componentNameLabel, sizeof(componentNameLabel), "componentName%D", c);CHKERRQ(ierr);
        ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, componentNameLabel, PETSC_STRING, componentName);CHKERRQ(ierr);
      }

      if (cutLabel) {ierr = VecDestroy(&subv);CHKERRQ(ierr);}
      else          {ierr = PetscSectionRestoreField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);}
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
  }
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
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
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  }
  ierr = VecLoad_Plex_Local(locv, viewer);CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  }
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
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  }
  ierr = VecLoad_Default(v, viewer);CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTopologyView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char           *topologydm_name;
  const char           *pointsName, *coneSizesName, *conesName, *orientationsName;
  IS                    pointsIS, coneSizesIS, conesIS, orientationsIS;
  PetscInt             *points, *coneSizes, *cones, *orientations;
  const PetscInt       *gpoint;
  PetscInt              dim, pStart, pEnd, p, nPoints = 0, conesSize = 0, c = 0, s = 0;
  DMPlexStorageVersion  version;
  char                  group[PETSC_MAX_PATH_LEN];
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  pointsName        = "order";
  coneSizesName     = "cones";
  conesName         = "cells";
  orientationsName  = "orientation";
  ierr = DMPlexStorageVersionSetUpWriting_Private(dm, viewer, &version);CHKERRQ(ierr);
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      PetscInt coneSize;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      nPoints += 1;
      conesSize += coneSize;
    }
  }
  ierr = PetscMalloc1(nPoints, &points);CHKERRQ(ierr);
  ierr = PetscMalloc1(nPoints, &coneSizes);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &cones);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &orientations);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, cp;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
      points[s]   = gpoint[p];
      coneSizes[s++] = coneSize;
      for (cp = 0; cp < coneSize; ++cp, ++c) {cones[c] = gpoint[cone[cp]] < 0 ? -(gpoint[cone[cp]]+1) : gpoint[cone[cp]]; orientations[c] = ornt[cp];}
    }
  }
  PetscCheckFalse(s != nPoints,PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of points %D != %D", s, nPoints);
  PetscCheckFalse(c != conesSize,PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cone points %D != %D", c, conesSize);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), nPoints, points, PETSC_OWN_POINTER, &pointsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) pointsIS, pointsName);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), nPoints, coneSizes, PETSC_OWN_POINTER, &coneSizesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, cones, PETSC_OWN_POINTER, &conesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) conesIS, conesName);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, orientations, PETSC_OWN_POINTER, &orientationsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orientationsIS, orientationsName);CHKERRQ(ierr);
  if (version.major <= 1) {
    ierr = PetscStrcpy(group, "/topology");CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(group, sizeof(group), "topologies/%s/topology", topologydm_name);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
  ierr = ISView(pointsIS, viewer);CHKERRQ(ierr);
  ierr = ISView(coneSizesIS, viewer);CHKERRQ(ierr);
  ierr = ISView(conesIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) conesIS, "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  ierr = ISView(orientationsIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = ISDestroy(&pointsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&coneSizesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&orientationsIS);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(&numCornersLocal, numCorners, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
  PetscCheckFalse(numCornersLocal && (numCornersLocal != *numCorners || *numCorners == 1),PETSC_COMM_SELF, PETSC_ERR_SUP, "Visualization topology currently only supports identical cell shapes");
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
  PetscCheckFalse(v != conesSize,PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cell vertices %D != %D", v, conesSize);
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

static PetscErrorCode DMPlexCoordinatesView_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
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

PetscErrorCode DMPlexCoordinatesView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  DM              cdm;
  Vec             coords, newcoords;
  PetscInt        m, M, bs;
  PetscReal       lengthScale;
  const char     *topologydm_name, *coordinatedm_name, *coordinates_name;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  {
    PetscViewerFormat     format;
    DMPlexStorageVersion  version;

    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    ierr = DMPlexStorageVersionSetUpWriting_Private(dm, viewer, &version);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_HDF5_XDMF || format == PETSC_VIEWER_HDF5_VIZ || version.major <= 1) {
      ierr = DMPlexCoordinatesView_HDF5_Legacy_Private(dm, viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)cdm, &coordinatedm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)coords, &coordinates_name);CHKERRQ(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "coordinateDMName", PETSC_STRING, coordinatedm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "coordinatesName", PETSC_STRING, coordinates_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMPlexSectionView(dm, viewer, cdm);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)coords), &newcoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)newcoords, coordinates_name);CHKERRQ(ierr);
  ierr = VecGetSize(coords, &M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coords, &m);CHKERRQ(ierr);
  ierr = VecSetSizes(newcoords, m, M);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coords, &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);
  ierr = VecSetType(newcoords,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecCopy(coords, newcoords);CHKERRQ(ierr);
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = DMPlexGlobalVectorView(dm, viewer, cdm, newcoords);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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
  PetscCheckFalse(coordSize != N,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched sizes: %D != %D", coordSize, N);
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

PetscErrorCode DMPlexLabelsView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char           *topologydm_name;
  const PetscInt       *gpoint;
  PetscInt              numLabels, l;
  DMPlexStorageVersion  version;
  char                  group[PETSC_MAX_PATH_LEN];
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMPlexStorageVersionSetUpWriting_Private(dm, viewer, &version);CHKERRQ(ierr);
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  if (version.major <= 1) {
    ierr = PetscStrcpy(group, "/labels");CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(group, sizeof(group), "topologies/%s/labels", topologydm_name);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label;
    const char     *name;
    IS              valueIS, pvalueIS, globalValueIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscBool       isDepth, output;

    ierr = DMGetLabelByNum(dm, l, &label);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)label, &name);CHKERRQ(ierr);
    ierr = DMGetLabelOutput(dm, name, &output);CHKERRQ(ierr);
    ierr = PetscStrncmp(name, "depth", 10, &isDepth);CHKERRQ(ierr);
    if (isDepth || !output) continue;
    ierr = PetscViewerHDF5PushGroup(viewer, name);CHKERRQ(ierr);
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
      char            group[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(group, sizeof(group), "%D", values[v]);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, values[v], &stratumIS);CHKERRQ(ierr);

      if (stratumIS) {ierr = ISGetLocalSize(stratumIS, &n);CHKERRQ(ierr);}
      if (stratumIS) {ierr = ISGetIndices(stratumIS, &spoints);CHKERRQ(ierr);}
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) ++gn;
      ierr = PetscMalloc1(gn,&gspoints);CHKERRQ(ierr);
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) gspoints[gn++] = gpoint[spoints[p]];
      if (stratumIS) {ierr = ISRestoreIndices(stratumIS, &spoints);CHKERRQ(ierr);}
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), gn, gspoints, PETSC_OWN_POINTER, &globalStratumIS);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) globalStratumIS, iname);CHKERRQ(ierr);

      ierr = ISView(globalStratumIS, viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&globalStratumIS);CHKERRQ(ierr);
      ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(globalValueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&globalValueIS);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
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
  ierr = DMPlexCoordinatesView_HDF5_Internal(dm, viewer);CHKERRQ(ierr);

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
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 output.", PetscViewerFormats[format]);
  }

  if (viz_geom)   {ierr = DMPlexWriteCoordinates_Vertices_HDF5_Static(dm, viewer);CHKERRQ(ierr);}
  if (xdmf_topo)  {ierr = DMPlexWriteTopology_Vertices_HDF5_Static(dm, globalPointNumbers, viewer);CHKERRQ(ierr);}
  if (petsc_topo) {
    ierr = DMPlexTopologyView_HDF5_Internal(dm, globalPointNumbers, viewer);CHKERRQ(ierr);
    ierr = DMPlexLabelsView_HDF5_Internal(dm, globalPointNumbers, viewer);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&globalPointNumbers);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexSectionView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm)
{
  MPI_Comm       comm;
  const char    *topologydm_name;
  const char    *sectiondm_name;
  PetscSection   gsection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sectiondm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(sectiondm, &gsection);CHKERRQ(ierr);
  /* Save raw section */
  ierr = PetscSectionView(gsection, viewer);CHKERRQ(ierr);
  /* Save plex wrapper */
  {
    PetscInt        pStart, pEnd, p, n;
    IS              globalPointNumbers;
    const PetscInt *gpoints;
    IS              orderIS;
    PetscInt       *order;

    ierr = PetscSectionGetChart(gsection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexCreatePointNumbering(dm, &globalPointNumbers);CHKERRQ(ierr);
    ierr = ISGetIndices(globalPointNumbers, &gpoints);CHKERRQ(ierr);
    for (p = pStart, n = 0; p < pEnd; ++p) if (gpoints[p] >= 0) n++;
    /* "order" is an array of global point numbers.
       When loading, it is used with topology/order array
       to match section points with plex topology points. */
    ierr = PetscMalloc1(n, &order);CHKERRQ(ierr);
    for (p = pStart, n = 0; p < pEnd; ++p) if (gpoints[p] >= 0) order[n++] = gpoints[p];
    ierr = ISRestoreIndices(globalPointNumbers, &gpoints);CHKERRQ(ierr);
    ierr = ISDestroy(&globalPointNumbers);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, n, order, PETSC_OWN_POINTER, &orderIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)orderIS, "order");CHKERRQ(ierr);
    ierr = ISView(orderIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&orderIS);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGlobalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  const char     *topologydm_name;
  const char     *sectiondm_name;
  const char     *vec_name;
  PetscInt        bs;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;

    ierr = DMGetPointSF(dm, &pointsf);CHKERRQ(ierr);
    ierr = DMGetPointSF(sectiondm, &pointsf1);CHKERRQ(ierr);
    PetscCheckFalse(pointsf1 != pointsf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)vec, &vec_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "vecs");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vec_name);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vec, &bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *) &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec, 1);CHKERRQ(ierr);
  /* VecView(vec, viewer) would call (*vec->opt->view)(vec, viewer), but,    */
  /* if vec was created with DMGet{Global, Local}Vector(), vec->opt->view    */
  /* is set to VecView_Plex, which would save vec in a predefined location.  */
  /* To save vec in where we want, we create a new Vec (temp) with           */
  /* VecCreate(), wrap the vec data in temp, and call VecView(temp, viewer). */
  {
    Vec                temp;
    const PetscScalar *array;
    PetscLayout        map;

    ierr = VecCreate(PetscObjectComm((PetscObject)vec), &temp);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)temp, vec_name);CHKERRQ(ierr);
    ierr = VecGetLayout(vec, &map);CHKERRQ(ierr);
    ierr = VecSetLayout(temp, map);CHKERRQ(ierr);
    ierr = VecSetUp(temp);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec, &array);CHKERRQ(ierr);
    ierr = VecPlaceArray(temp, array);CHKERRQ(ierr);
    ierr = VecView(temp, viewer);CHKERRQ(ierr);
    ierr = VecResetArray(temp);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec, &array);CHKERRQ(ierr);
    ierr = VecDestroy(&temp);CHKERRQ(ierr);
  }
  ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;

    ierr = DMGetPointSF(dm, &pointsf);CHKERRQ(ierr);
    ierr = DMGetPointSF(sectiondm, &pointsf1);CHKERRQ(ierr);
    PetscCheckFalse(pointsf1 != pointsf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)vec, &vec_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "vecs");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vec_name);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vec, &bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *) &bs);CHKERRQ(ierr);
  ierr = VecCreate(comm, &gvec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)gvec, vec_name);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(sectiondm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(section, &includesConstraints);CHKERRQ(ierr);
  if (includesConstraints) {ierr = PetscSectionGetStorageSize(section, &m);CHKERRQ(ierr);}
  else {ierr = PetscSectionGetConstrainedStorageSize(section, &m);CHKERRQ(ierr);}
  ierr = VecSetSizes(gvec, m, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetUp(gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(sectiondm, vec, INSERT_VALUES, gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(sectiondm, vec, INSERT_VALUES, gvec);CHKERRQ(ierr);
  ierr = VecView(gvec, viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) ((*ctx)->dm = dm));
  ierr = PetscObjectReference((PetscObject) ((*ctx)->viewer = viewer));
  ierr = PetscObjectGetComm((PetscObject)dm, &(*ctx)->comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank((*ctx)->comm, &(*ctx)->rank);CHKERRMPI(ierr);
  (*ctx)->sfXC = sfXC;
  if (sfXC) {
    PetscInt nX;

    ierr = PetscObjectReference((PetscObject) sfXC);
    ierr = PetscSFGetGraph(sfXC, &nX, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscLayoutCreateFromSizes((*ctx)->comm, nX, PETSC_DECIDE, 1, &(*ctx)->layoutX);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode LoadLabelsCtxDestroy(LoadLabelsCtx *ctx)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(0);
  ierr = DMDestroy(&(*ctx)->dm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ctx)->viewer);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*ctx)->sfXC);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&(*ctx)->layoutX);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(sfXC, &nX, &nC, NULL, NULL);CHKERRQ(ierr);
  ierr = ISGetLocalSize(stratumIS, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(stratumIS, &A_points);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &sfXA);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sfXA, layoutX, n, NULL, PETSC_USE_POINTER, A_points);CHKERRQ(ierr);
  ierr = ISCreate(comm, newStratumIS);CHKERRQ(ierr);
  ierr = ISSetType(*newStratumIS,ISGENERAL);CHKERRQ(ierr);
  {
    PetscInt    i;
    PetscBool  *A_mask, *X_mask, *C_mask;

    ierr = PetscCalloc3(n, &A_mask, nX, &X_mask, nC, &C_mask);CHKERRQ(ierr);
    for (i=0; i<n; i++) A_mask[i] = PETSC_TRUE;
    ierr = PetscSFReduceBegin(sfXA, MPIU_BOOL, A_mask, X_mask, MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(  sfXA, MPIU_BOOL, A_mask, X_mask, MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin( sfXC, MPIU_BOOL, X_mask, C_mask, MPI_LOR);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(   sfXC, MPIU_BOOL, X_mask, C_mask, MPI_LOR);CHKERRQ(ierr);
    ierr = ISGeneralSetIndicesFromMask(*newStratumIS, 0, nC, C_mask);CHKERRQ(ierr);
    ierr = PetscFree3(A_mask, X_mask, C_mask);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sfXA);CHKERRQ(ierr);
  ierr = ISRestoreIndices(stratumIS, &A_points);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  ierr = PetscOptionsStringToInt(vname, &value);CHKERRQ(ierr);
  ierr = ISCreate(comm, &stratumIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) stratumIS, "indices");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vname);CHKERRQ(ierr); /* labels/<lname>/<vname> */

  if (!ctx->sfXC) {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, "indices", NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(stratumIS->map, !ctx->rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(stratumIS->map, N);CHKERRQ(ierr);
  }
  ierr = ISLoad(stratumIS, viewer);CHKERRQ(ierr);

  if (ctx->sfXC) {
    IS newStratumIS;

    ierr = ReadLabelStratumHDF5_Distribute_Private(stratumIS, ctx, &newStratumIS);CHKERRQ(ierr);
    ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
    stratumIS = newStratumIS;
  }

  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = ISGetLocalSize(stratumIS, &N);CHKERRQ(ierr);
  ierr = ISGetIndices(stratumIS, &ind);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {ierr = DMLabelSetValue(label, ind[i], value);CHKERRQ(ierr);}
  ierr = ISRestoreIndices(stratumIS, &ind);CHKERRQ(ierr);
  ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
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

  ierr = DMHasLabel(dm, lname, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMRemoveLabel(dm, lname, NULL);CHKERRQ(ierr);
  }
  ierr = DMCreateLabel(dm, lname); if (ierr) return (herr_t) ierr;
  ierr = DMGetLabel(dm, lname, &ctx->label); if (ierr) return (herr_t) ierr;
  ierr = PetscViewerHDF5PushGroup(ctx->viewer, lname);CHKERRQ(ierr); /* labels/<lname> */
  /* Iterate over the label's strata */
  PetscStackCallHDF5Return(err, H5Literate_by_name, (g_id, lname, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelStratumHDF5_Static, op_data, 0));
  ierr = PetscViewerHDF5PopGroup(ctx->viewer);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = DMPlexIsDistributed(dm, &distributed);CHKERRQ(ierr);
  if (distributed) {
    PetscCheckFalse(!sfXC,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "PetscSF must be given for parallel load");
  }
  ierr = LoadLabelsCtxCreate(dm, viewer, sfXC, &ctx);CHKERRQ(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = DMPlexStorageVersionGet_Private(dm, viewer, &version);CHKERRQ(ierr);
  if (version.major <= 1) {
    ierr = PetscStrcpy(group, "labels");CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(group, sizeof(group), "topologies/%s/labels", topologydm_name);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5HasGroup(viewer, NULL, &hasGroup);CHKERRQ(ierr);
  if (hasGroup) {
    hid_t fileId, groupId;

    ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
    /* Iterate over labels */
    PetscStackCallHDF5(H5Literate,(groupId, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelHDF5_Static, ctx));
    PetscStackCallHDF5(H5Gclose,(groupId));
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = LoadLabelsCtxDestroy(&ctx);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  pointsName        = "order";
  coneSizesName     = "cones";
  conesName         = "cells";
  orientationsName  = "orientation";
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = DMPlexStorageVersionGet_Private(dm, viewer, &version);CHKERRQ(ierr);
  if (version.major <= 1) {
    ierr = PetscStrcpy(group, "/topology");CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(group, sizeof(group), "topologies/%s/topology", topologydm_name);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
  ierr = ISCreate(comm, &pointsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) pointsIS, pointsName);CHKERRQ(ierr);
  ierr = ISCreate(comm, &coneSizesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName);CHKERRQ(ierr);
  ierr = ISCreate(comm, &conesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) conesIS, conesName);CHKERRQ(ierr);
  ierr = ISCreate(comm, &orientationsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orientationsIS, orientationsName);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadObjectAttribute(viewer, (PetscObject) conesIS, "cell_dim", PETSC_INT, NULL, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, pointsName, NULL, &Np);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(pointsIS->map, rank == 0 ? Np : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(pointsIS->map, Np);CHKERRQ(ierr);
    pEnd = rank == 0 ? Np : 0;
    ierr = PetscViewerHDF5ReadSizes(viewer, coneSizesName, NULL, &Np);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(coneSizesIS->map, rank == 0 ? Np : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(coneSizesIS->map, Np);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, conesName, NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(conesIS->map, rank == 0 ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(conesIS->map, N);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, orientationsName, NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(orientationsIS->map, rank == 0 ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(orientationsIS->map, N);CHKERRQ(ierr);
  }
  ierr = ISLoad(pointsIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(coneSizesIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(conesIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(orientationsIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  /* Create Plex */
  ierr = DMPlexSetChart(dm, 0, pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = ISGetIndices(coneSizesIS, &coneSizes);CHKERRQ(ierr);
  for (p = 0; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, points[p], coneSizes[p]);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, coneSizes[p]);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = ISGetIndices(conesIS, &cones);CHKERRQ(ierr);
  ierr = ISGetIndices(orientationsIS, &orientations);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxConeSize,&cone,maxConeSize,&ornt);CHKERRQ(ierr);
  for (p = 0, q = 0; p < pEnd; ++p) {
    for (c = 0; c < coneSizes[p]; ++c, ++q) {cone[c] = cones[q]; ornt[c] = orientations[q];}
    ierr = DMPlexSetCone(dm, points[p], cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, points[p], ornt);CHKERRQ(ierr);
  }
  ierr = PetscFree2(cone,ornt);CHKERRQ(ierr);
  /* Create global section migration SF */
  if (sf) {
    PetscLayout  layout;
    PetscInt    *globalIndices;

    ierr = PetscMalloc1(pEnd, &globalIndices);CHKERRQ(ierr);
    /* plex point == globalPointNumber in this case */
    for (p = 0; p < pEnd; ++p) globalIndices[p] = p;
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(layout, Np);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm, sf);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(*sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(*sf, layout, pEnd, NULL, PETSC_OWN_POINTER, globalIndices);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    ierr = PetscFree(globalIndices);CHKERRQ(ierr);
  }
  /* Clean-up */
  ierr = ISRestoreIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = ISRestoreIndices(coneSizesIS, &coneSizes);CHKERRQ(ierr);
  ierr = ISRestoreIndices(conesIS, &cones);CHKERRQ(ierr);
  ierr = ISRestoreIndices(orientationsIS, &orientations);CHKERRQ(ierr);
  ierr = ISDestroy(&pointsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&coneSizesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&orientationsIS);CHKERRQ(ierr);
  /* Fill in the rest of the topology structure */
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
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
  /* Create coordinates */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  PetscCheckFalse(numVertices != vEnd - vStart,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of coordinates loaded %d does not match number of vertices %d", numVertices, vEnd - vStart);
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
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  {
    DMPlexStorageVersion  version;

    ierr = DMPlexStorageVersionGet_Private(dm, viewer, &version);CHKERRQ(ierr);
    if (version.major <= 1) {
      ierr = DMPlexCoordinatesLoad_HDF5_Legacy_Private(dm, viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscCheckFalse(!sfXC,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_NULL, "PetscSF must be given for parallel load");
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "coordinateDMName", PETSC_STRING , NULL, &coordinatedm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "coordinatesName", PETSC_STRING , NULL, &coordinates_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)cdm, coordinatedm_name);CHKERRQ(ierr);
  ierr = PetscFree(coordinatedm_name);CHKERRQ(ierr);
  /* lsf: on-disk data -> in-memory local vector associated with cdm's local section */
  ierr = DMPlexSectionLoad(dm, viewer, cdm, sfXC, NULL, &lsf);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(cdm, &coords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)coords, coordinates_name);CHKERRQ(ierr);
  ierr = PetscFree(coordinates_name);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = DMPlexLocalVectorLoad(dm, viewer, cdm, lsf, coords);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = VecScale(coords, 1.0/lengthScale);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coords);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coords, &blockSize);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm, blockSize);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&lsf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexLoad_HDF5_Legacy_Private(DM dm, PetscViewer viewer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTopologyLoad_HDF5_Internal(dm, viewer, NULL);CHKERRQ(ierr);
  ierr = DMPlexLabelsLoad_HDF5_Internal(dm, viewer, NULL);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesLoad_HDF5_Legacy_Private(dm, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLoad_HDF5_Internal(DM dm, PetscViewer viewer)
{
  PetscSF               sfXC;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  {
    DMPlexStorageVersion  version;

    ierr = DMPlexStorageVersionGet_Private(dm, viewer, &version);CHKERRQ(ierr);
    if (version.major <= 1) {
      ierr = DMPlexLoad_HDF5_Legacy_Private(dm, viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  ierr = DMPlexTopologyLoad_HDF5_Internal(dm, viewer, &sfXC);CHKERRQ(ierr);
  ierr = DMPlexLabelsLoad_HDF5_Internal(dm, viewer, sfXC);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesLoad_HDF5_Internal(dm, viewer, sfXC);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfXC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSectionLoad_HDF5_Internal_CreateDataSF(PetscSection rootSection, PetscLayout layout, PetscInt globalOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm        comm;
  PetscInt        pStart, pEnd, p, m;
  PetscInt       *goffs, *ilocal;
  PetscBool       rootIncludeConstraints, leafIncludeConstraints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)leafSection, &comm);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(rootSection, &rootIncludeConstraints);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(leafSection, &leafIncludeConstraints);CHKERRQ(ierr);
  if (rootIncludeConstraints && leafIncludeConstraints) {ierr = PetscSectionGetStorageSize(leafSection, &m);CHKERRQ(ierr);}
  else {ierr = PetscSectionGetConstrainedStorageSize(leafSection, &m);CHKERRQ(ierr);}
  ierr = PetscMalloc1(m, &ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &goffs);CHKERRQ(ierr);
  /* Currently, PetscSFDistributeSection() returns globalOffsets[] only */
  /* for the top-level section (not for each field), so one must have   */
  /* rootSection->pointMajor == PETSC_TRUE.                             */
  PetscCheckFalse(!rootSection->pointMajor,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for field major ordering");
  /* Currently, we also assume that leafSection->pointMajor == PETSC_TRUE. */
  PetscCheckFalse(!leafSection->pointMajor,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for field major ordering");
  for (p = pStart, m = 0; p < pEnd; ++p) {
    PetscInt        dof, cdof, i, j, off, goff;
    const PetscInt *cinds;

    ierr = PetscSectionGetDof(leafSection, p, &dof);CHKERRQ(ierr);
    if (dof < 0) continue;
    goff = globalOffsets[p-pStart];
    ierr = PetscSectionGetOffset(leafSection, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(leafSection, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintIndices(leafSection, p, &cinds);CHKERRQ(ierr);
    for (i = 0, j = 0; i < dof; ++i) {
      PetscBool constrained = (PetscBool) (j < cdof && i == cinds[j]);

      if (!constrained || (leafIncludeConstraints && rootIncludeConstraints)) {ilocal[m] = off++; goffs[m++] = goff++;}
      else if (leafIncludeConstraints && !rootIncludeConstraints) ++off;
      else if (!leafIncludeConstraints &&  rootIncludeConstraints) ++goff;
      if (constrained) ++j;
    }
  }
  ierr = PetscSFCreate(comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sectionSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(*sectionSF, layout, m, ilocal, PETSC_OWN_POINTER, goffs);CHKERRQ(ierr);
  ierr = PetscFree(goffs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  /* A: on-disk points                        */
  /* X: list of global point numbers, [0, NX) */
  /* B: plex points                           */
  /* Load raw section (sectionA)              */
  ierr = PetscSectionCreate(comm, &sectionA);CHKERRQ(ierr);
  ierr = PetscSectionLoad(sectionA, viewer);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(sectionA, NULL, &n);CHKERRQ(ierr);
  /* Create sfAB: A -> B */
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt  N, N1;

    ierr = PetscViewerHDF5ReadSizes(viewer, "order", NULL, &N1);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&n, &N, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
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
    ierr = ISCreate(comm, &orderIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)orderIS, "order");CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(orderIS->map, n);CHKERRQ(ierr);
    ierr = ISLoad(orderIS, viewer);CHKERRQ(ierr);
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfXB, &nX, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(layout, nX);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm, &sfXA);CHKERRQ(ierr);
    ierr = ISGetIndices(orderIS, &gpoints);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sfXA, layout, n, NULL, PETSC_OWN_POINTER, gpoints);CHKERRQ(ierr);
    ierr = ISRestoreIndices(orderIS, &gpoints);CHKERRQ(ierr);
    ierr = ISDestroy(&orderIS);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &owners);CHKERRQ(ierr);
    ierr = PetscMalloc1(nX, &buffer);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {owners[i].rank = rank; owners[i].index = i;}
    for (i = 0; i < nX; ++i) {buffer[i].rank = -1; buffer[i].index = -1;}
    ierr = PetscSFReduceBegin(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfXA);CHKERRQ(ierr);
    ierr = PetscFree(owners);CHKERRQ(ierr);
    for (i = 0, nleaves = 0; i < nX; ++i) if (buffer[i].rank >= 0) nleaves++;
    ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
    for (i = 0, nleaves = 0; i < nX; ++i) {
      if (buffer[i].rank >= 0) {
        ilocal[nleaves] = i;
        iremote[nleaves].rank = buffer[i].rank;
        iremote[nleaves].index = buffer[i].index;
        nleaves++;
      }
    }
    ierr = PetscSFCreate(comm, &sfAX);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(sfAX);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfAX, n, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    /* Fix PetscSFCompose() and replace the code-block below with:  */
    /* ierr = PetscSFCompose(sfAX, sfXB, &sfAB);CHKERRQ(ierr);      */
    /* which currently causes segmentation fault due to sparse map. */
    {
      PetscInt     npoints;
      PetscInt     mleaves;
      PetscInt    *jlocal;
      PetscSFNode *jremote;

      ierr = PetscSFGetGraph(sfXB, NULL, &npoints, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(npoints, &owners);CHKERRQ(ierr);
      for (i = 0; i < npoints; ++i) {owners[i].rank = -1; owners[i].index = -1;}
      ierr = PetscSFBcastBegin(sfXB, MPIU_2INT, buffer, owners, MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfXB, MPIU_2INT, buffer, owners, MPI_REPLACE);CHKERRQ(ierr);
      for (i = 0, mleaves = 0; i < npoints; ++i) if (owners[i].rank >= 0) mleaves++;
      ierr = PetscMalloc1(mleaves, &jlocal);CHKERRQ(ierr);
      ierr = PetscMalloc1(mleaves, &jremote);CHKERRQ(ierr);
      for (i = 0, mleaves = 0; i < npoints; ++i) {
        if (owners[i].rank >= 0) {
          jlocal[mleaves] = i;
          jremote[mleaves].rank = owners[i].rank;
          jremote[mleaves].index = owners[i].index;
          mleaves++;
        }
      }
      ierr = PetscSFCreate(comm, &sfAB);CHKERRQ(ierr);
      ierr = PetscSFSetFromOptions(sfAB);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(sfAB, n, mleaves, jlocal, PETSC_OWN_POINTER, jremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree(owners);CHKERRQ(ierr);
    }
    ierr = PetscFree(buffer);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfAX);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  /* Create plex section (sectionB) */
  ierr = DMGetLocalSection(sectiondm, &sectionB);CHKERRQ(ierr);
  if (lsf || gsf) {
    PetscLayout  layout;
    PetscInt     M, m;
    PetscInt    *offsetsA;
    PetscBool    includesConstraintsA;

    ierr = PetscSFDistributeSection(sfAB, sectionA, &offsetsA, sectionB);CHKERRQ(ierr);
    ierr = PetscSectionGetIncludesConstraints(sectionA, &includesConstraintsA);CHKERRQ(ierr);
    if (includesConstraintsA) {ierr = PetscSectionGetStorageSize(sectionA, &m);CHKERRQ(ierr);}
    else {ierr = PetscSectionGetConstrainedStorageSize(sectionA, &m);CHKERRQ(ierr);}
    ierr = MPI_Allreduce(&m, &M, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(layout, M);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    if (lsf) {
      PetscSF lsfABdata;

      ierr = DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, sectionB, &lsfABdata);CHKERRQ(ierr);
      *lsf = lsfABdata;
    }
    if (gsf) {
      PetscSection  gsectionB, gsectionB1;
      PetscBool     includesConstraintsB;
      PetscSF       gsfABdata, pointsf;

      ierr = DMGetGlobalSection(sectiondm, &gsectionB1);CHKERRQ(ierr);
      ierr = PetscSectionGetIncludesConstraints(gsectionB1, &includesConstraintsB);CHKERRQ(ierr);
      ierr = DMGetPointSF(sectiondm, &pointsf);CHKERRQ(ierr);
      ierr = PetscSectionCreateGlobalSection(sectionB, pointsf, includesConstraintsB, PETSC_TRUE, &gsectionB);CHKERRQ(ierr);
      ierr = DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, gsectionB, &gsfABdata);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&gsectionB);CHKERRQ(ierr);
      *gsf = gsfABdata;
    }
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    ierr = PetscFree(offsetsA);CHKERRQ(ierr);
  } else {
    ierr = PetscSFDistributeSection(sfAB, sectionA, NULL, sectionB);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sfAB);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionA);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetHDF5Name_Private(dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)vec, &vec_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "vecs");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vec_name);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vecA);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vecA, vec_name);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &mA, &m, &ilocal, NULL);CHKERRQ(ierr);
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;
    PetscInt  m1, i, j;

    ierr = DMGetPointSF(dm, &pointsf);CHKERRQ(ierr);
    ierr = DMGetPointSF(sectiondm, &pointsf1);CHKERRQ(ierr);
    PetscCheckFalse(pointsf1 != pointsf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt  MA, MA1;

      ierr = MPIU_Allreduce(&mA, &MA, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
      ierr = PetscViewerHDF5ReadSizes(viewer, vec_name, NULL, &MA1);CHKERRQ(ierr);
      PetscCheckFalse(MA1 != MA,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total SF root size (%D) != On-disk vector data size (%D)", MA, MA1);
    }
#endif
    ierr = VecGetLocalSize(vec, &m1);CHKERRQ(ierr);
    PetscCheckFalse(m1 < m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Target vector size (%D) < SF leaf size (%D)", m1, m);
    for (i = 0; i < m; ++i) {
      j = ilocal ? ilocal[i] : i;
      PetscCheckFalse(j < 0 || j >= m1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Leaf's %D-th index, %D, not in [%D, %D)", i, j, 0, m1);
    }
  }
  ierr = VecSetSizes(vecA, mA, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecLoad(vecA, viewer);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vecA, &src);CHKERRQ(ierr);
  ierr = VecGetArray(vec, &dest);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, src, dest, MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, src, dest, MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec, &dest);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vecA, &src);CHKERRQ(ierr);
  ierr = VecDestroy(&vecA);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "blockSize", PETSC_INT, NULL, (void *) &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
