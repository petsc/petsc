#include <petsc/private/sectionimpl.h>   /*I "petscsection.h" I*/
#include <petscsf.h>
#include <petscis.h>
#include <petscviewerhdf5.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode PetscSectionView_HDF5_SingleField(PetscSection s, PetscViewer viewer)
{
  MPI_Comm        comm;
  PetscInt        pStart, pEnd, p, n;
  PetscBool       hasConstraints, includesConstraints;
  IS              dofIS, offIS, cdofIS, coffIS, cindIS;
  PetscInt       *dofs, *offs, *cdofs, *coffs, *cinds, dof, cdof, m, moff, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)s, &comm);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  hasConstraints = (s->bc) ? PETSC_TRUE : PETSC_FALSE;
  ierr = MPIU_Allreduce(MPI_IN_PLACE, &hasConstraints, 1, MPIU_BOOL, MPI_LOR, comm);CHKERRMPI(ierr);
  for (p = pStart, n = 0, m = 0; p < pEnd; ++p) {
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    if (dof >= 0) {
      if (hasConstraints) {
        ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
        m += cdof;
      }
      n++;
    }
  }
  ierr = PetscMalloc1(n, &dofs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &offs);CHKERRQ(ierr);
  if (hasConstraints) {
    ierr = PetscMalloc1(n, &cdofs);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &coffs);CHKERRQ(ierr);
    ierr = PetscMalloc1(m, &cinds);CHKERRQ(ierr);
  }
  for (p = pStart, n = 0, m = 0; p < pEnd; ++p) {
    ierr = PetscSectionGetDof(s, p, &dof);CHKERRQ(ierr);
    if (dof >= 0) {
      dofs[n] = dof;
      ierr = PetscSectionGetOffset(s, p, &offs[n]);CHKERRQ(ierr);
      if (hasConstraints) {
        const PetscInt *cpinds;

        ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintIndices(s, p, &cpinds);CHKERRQ(ierr);
        cdofs[n] = cdof;
        coffs[n] = m;
        for (i = 0; i < cdof; ++i) cinds[m++] = cpinds[i];
      }
      n++;
    }
  }
  if (hasConstraints) {
    ierr = MPI_Scan(&m, &moff, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    moff -= m;
    for (p = 0; p < n; ++p) coffs[p] += moff;
  }
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "hasConstraints", PETSC_BOOL, (void *) &hasConstraints);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(s, &includesConstraints);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "includesConstraints", PETSC_BOOL, (void *)&includesConstraints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, n, dofs, PETSC_OWN_POINTER, &dofIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dofIS, "atlasDof");CHKERRQ(ierr);
  ierr = ISView(dofIS, viewer);CHKERRQ(ierr);
  ierr = ISDestroy(&dofIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, n, offs, PETSC_OWN_POINTER, &offIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)offIS, "atlasOff");CHKERRQ(ierr);
  ierr = ISView(offIS, viewer);CHKERRQ(ierr);
  ierr = ISDestroy(&offIS);CHKERRQ(ierr);
  if (hasConstraints) {
    ierr = PetscViewerHDF5PushGroup(viewer, "bc");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, n, cdofs, PETSC_OWN_POINTER, &cdofIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)cdofIS, "atlasDof");CHKERRQ(ierr);
    ierr = ISView(cdofIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&cdofIS);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, n, coffs, PETSC_OWN_POINTER, &coffIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)coffIS, "atlasOff");CHKERRQ(ierr);
    ierr = ISView(coffIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&coffIS);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, m, cinds, PETSC_OWN_POINTER, &cindIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)cindIS, "bcIndices");CHKERRQ(ierr);
    ierr = ISView(cindIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&cindIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionView_HDF5_Internal(PetscSection s, PetscViewer viewer)
{
  PetscInt        numFields, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5PushGroup(viewer, "section");CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &numFields);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "numFields", PETSC_INT, (void *) &numFields);CHKERRQ(ierr);
  ierr = PetscSectionView_HDF5_SingleField(s, viewer);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    char        fname[PETSC_MAX_PATH_LEN];
    const char *fieldName;
    PetscInt    fieldComponents, c;

    ierr = PetscSNPrintf(fname, sizeof(fname), "field%" PetscInt_FMT, f);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, fname);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldName(s, f, &fieldName);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldName", PETSC_STRING, fieldName);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, f, &fieldComponents);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldComponents", PETSC_INT, (void *) &fieldComponents);CHKERRQ(ierr);
    for (c = 0; c < fieldComponents; ++c) {
      char        cname[PETSC_MAX_PATH_LEN];
      const char *componentName;

      ierr = PetscSNPrintf(cname, sizeof(cname), "component%" PetscInt_FMT, c);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer, cname);CHKERRQ(ierr);
      ierr = PetscSectionGetComponentName(s, f, c, &componentName);CHKERRQ(ierr);
      ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "componentName", PETSC_STRING, componentName);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
    ierr = PetscSectionView_HDF5_SingleField(s->field[f], viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionLoad_HDF5_SingleField_SetConstraintIndices(PetscSection s, IS cindIS, IS coffIS)
{
  MPI_Comm        comm;
  PetscInt        pStart, pEnd, p, M, m, i, cdof;
  const PetscInt *data;
  PetscInt       *cinds;
  const PetscInt *coffs;
  PetscInt       *coffsets;
  PetscSF         sf;
  PetscLayout     layout;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)s, &comm);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISGetSize(cindIS, &M);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cindIS, &m);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &coffsets);CHKERRQ(ierr);
  ierr = ISGetIndices(coffIS, &coffs);CHKERRQ(ierr);
  for (p = pStart, m = 0; p < pEnd; ++p) {
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    for (i = 0; i < cdof; ++i) coffsets[m++] = coffs[p-pStart] + i;
  }
  ierr = ISRestoreIndices(coffIS, &coffs);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm, &sf);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(layout, M);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, m);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf, layout, m, NULL, PETSC_OWN_POINTER, coffsets);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscFree(coffsets);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &cinds);CHKERRQ(ierr);
  ierr = ISGetIndices(cindIS, &data);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_INT, data, cinds, MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, data, cinds, MPI_REPLACE);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cindIS, &data);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscSectionSetUpBC(s);CHKERRQ(ierr);
  for (p = pStart, m = 0; p < pEnd; ++p) {
    ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionSetConstraintIndices(s, p, &cinds[m]);CHKERRQ(ierr);
    m += cdof;
  }
  ierr = PetscFree(cinds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSectionLoad_HDF5_SingleField(PetscSection s, PetscViewer viewer)
{
  MPI_Comm        comm;
  PetscInt        pStart, pEnd, p, N, n, M, m;
#if defined(PETSC_USE_DEBUG)
  PetscInt        N1, M1;
#endif
  PetscBool       hasConstraints, includesConstraints;
  IS              dofIS, offIS, cdofIS, coffIS, cindIS;
  const PetscInt *dofs, *offs, *cdofs;
  PetscLayout     map;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)s, &comm);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "includesConstraints", PETSC_BOOL, NULL, (void *) &includesConstraints);CHKERRQ(ierr);
  ierr = PetscSectionSetIncludesConstraints(s, includesConstraints);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  n = pEnd - pStart;
#if defined(PETSC_USE_DEBUG)
  ierr = MPIU_Allreduce(&n, &N1, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
#endif
  ierr = ISCreate(comm, &dofIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dofIS, "atlasDof");CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->atlasDof: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
  ierr = ISGetLayout(dofIS, &map);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(map, n);CHKERRQ(ierr);
  ierr = ISLoad(dofIS, viewer);CHKERRQ(ierr);
  ierr = ISCreate(comm, &offIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)offIS, "atlasOff");CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes(viewer, "atlasOff", NULL, &N);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->atlasOff: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
  ierr = ISGetLayout(offIS, &map);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(map, n);CHKERRQ(ierr);
  ierr = ISLoad(offIS, viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(dofIS, &dofs);CHKERRQ(ierr);
  ierr = ISGetIndices(offIS, &offs);CHKERRQ(ierr);
  for (p = pStart, n = 0; p < pEnd; ++p, ++n) {
    ierr = PetscSectionSetDof(s, p, dofs[n]);CHKERRQ(ierr);
    ierr = PetscSectionSetOffset(s, p, offs[n]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(dofIS, &dofs);CHKERRQ(ierr);
  ierr = ISRestoreIndices(offIS, &offs);CHKERRQ(ierr);
  ierr = ISDestroy(&dofIS);CHKERRQ(ierr);
  ierr = ISDestroy(&offIS);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "hasConstraints", PETSC_BOOL, NULL, (void *) &hasConstraints);CHKERRQ(ierr);
  if (hasConstraints) {
    ierr = PetscViewerHDF5PushGroup(viewer, "bc");CHKERRQ(ierr);
    ierr = ISCreate(comm, &cdofIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)cdofIS, "atlasDof");CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bc->atlasDof: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
    ierr = ISGetLayout(cdofIS, &map);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(map, n);CHKERRQ(ierr);
    ierr = ISLoad(cdofIS, viewer);CHKERRQ(ierr);
    ierr = ISGetIndices(cdofIS, &cdofs);CHKERRQ(ierr);
    for (p = pStart, n = 0; p < pEnd; ++p, ++n) {
      ierr = PetscSectionSetConstraintDof(s, p, cdofs[n]);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(cdofIS, &cdofs);CHKERRQ(ierr);
    ierr = ISDestroy(&cdofIS);CHKERRQ(ierr);
    ierr = ISCreate(comm, &coffIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)coffIS, "atlasOff");CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, "atlasOff", NULL, &N);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bc->atlasOff: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
    ierr = ISGetLayout(coffIS, &map);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(map, n);CHKERRQ(ierr);
    ierr = ISLoad(coffIS, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = ISCreate(comm, &cindIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)cindIS, "bcIndices");CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, "bcIndices", NULL, &M);CHKERRQ(ierr);
    if (!s->bc) m = 0;
    else {ierr = PetscSectionGetStorageSize(s->bc, &m);CHKERRQ(ierr);}
#if defined(PETSC_USE_DEBUG)
    ierr = MPIU_Allreduce(&m, &M1, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    PetscCheckFalse(M1 != M,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bcIndices: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, M1, M, m);
#endif
    ierr = ISGetLayout(cindIS, &map);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(map, M);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(map, m);CHKERRQ(ierr);
    ierr = ISLoad(cindIS, viewer);CHKERRQ(ierr);
    ierr = PetscSectionLoad_HDF5_SingleField_SetConstraintIndices(s, cindIS, coffIS);CHKERRQ(ierr);
    ierr = ISDestroy(&coffIS);CHKERRQ(ierr);
    ierr = ISDestroy(&cindIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionLoad_HDF5_Internal(PetscSection s, PetscViewer viewer)
{
  MPI_Comm        comm;
  PetscInt        N, n, numFields, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)s, &comm);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "section");CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "numFields", PETSC_INT, NULL, (void *)&numFields);CHKERRQ(ierr);
  if (s->pStart < 0 && s->pEnd < 0) n = PETSC_DECIDE;
  else {
    PetscCheckFalse(s->pStart != 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "s->pStart must be 0 (got %" PetscInt_FMT ")", s->pStart);
    PetscCheckFalse(s->pEnd < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "s->pEnd must be >= 0, (got %" PetscInt_FMT ")", s->pEnd);
    n = s->pEnd;
  }
  if (numFields > 0) {ierr = PetscSectionSetNumFields(s, numFields);CHKERRQ(ierr);}
  ierr = PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N);CHKERRQ(ierr);
  if (n == PETSC_DECIDE) {ierr = PetscSplitOwnership(comm, &n, &N);CHKERRQ(ierr);}
  ierr = PetscSectionSetChart(s, 0, n);CHKERRQ(ierr);
  ierr = PetscSectionLoad_HDF5_SingleField(s, viewer);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    char      fname[PETSC_MAX_PATH_LEN];
    char     *fieldName;
    PetscInt  fieldComponents, c;

    ierr = PetscSNPrintf(fname, sizeof(fname), "field%" PetscInt_FMT, f);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, fname);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "fieldName", PETSC_STRING, NULL, &fieldName);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(s, f, fieldName);CHKERRQ(ierr);
    ierr = PetscFree(fieldName);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "fieldComponents", PETSC_INT, NULL, (void *) &fieldComponents);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(s, f, fieldComponents);CHKERRQ(ierr);
    for (c = 0; c < fieldComponents; ++c) {
      char  cname[PETSC_MAX_PATH_LEN];
      char *componentName;

      ierr = PetscSNPrintf(cname, sizeof(cname), "component%" PetscInt_FMT, c);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer, cname);CHKERRQ(ierr);
      ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "componentName", PETSC_STRING, NULL, &componentName);CHKERRQ(ierr);
      ierr = PetscSectionSetComponentName(s, f, c, componentName);CHKERRQ(ierr);
      ierr = PetscFree(componentName);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
    ierr = PetscSectionLoad_HDF5_SingleField(s->field[f], viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
