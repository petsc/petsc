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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)s, &comm));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  hasConstraints = (s->bc) ? PETSC_TRUE : PETSC_FALSE;
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE, &hasConstraints, 1, MPIU_BOOL, MPI_LOR, comm));
  for (p = pStart, n = 0, m = 0; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    if (dof >= 0) {
      if (hasConstraints) {
        CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
        m += cdof;
      }
      n++;
    }
  }
  CHKERRQ(PetscMalloc1(n, &dofs));
  CHKERRQ(PetscMalloc1(n, &offs));
  if (hasConstraints) {
    CHKERRQ(PetscMalloc1(n, &cdofs));
    CHKERRQ(PetscMalloc1(n, &coffs));
    CHKERRQ(PetscMalloc1(m, &cinds));
  }
  for (p = pStart, n = 0, m = 0; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetDof(s, p, &dof));
    if (dof >= 0) {
      dofs[n] = dof;
      CHKERRQ(PetscSectionGetOffset(s, p, &offs[n]));
      if (hasConstraints) {
        const PetscInt *cpinds;

        CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
        CHKERRQ(PetscSectionGetConstraintIndices(s, p, &cpinds));
        cdofs[n] = cdof;
        coffs[n] = m;
        for (i = 0; i < cdof; ++i) cinds[m++] = cpinds[i];
      }
      n++;
    }
  }
  if (hasConstraints) {
    CHKERRMPI(MPI_Scan(&m, &moff, 1, MPIU_INT, MPI_SUM, comm));
    moff -= m;
    for (p = 0; p < n; ++p) coffs[p] += moff;
  }
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "hasConstraints", PETSC_BOOL, (void *) &hasConstraints));
  CHKERRQ(PetscSectionGetIncludesConstraints(s, &includesConstraints));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "includesConstraints", PETSC_BOOL, (void *)&includesConstraints));
  CHKERRQ(ISCreateGeneral(comm, n, dofs, PETSC_OWN_POINTER, &dofIS));
  CHKERRQ(PetscObjectSetName((PetscObject)dofIS, "atlasDof"));
  CHKERRQ(ISView(dofIS, viewer));
  CHKERRQ(ISDestroy(&dofIS));
  CHKERRQ(ISCreateGeneral(comm, n, offs, PETSC_OWN_POINTER, &offIS));
  CHKERRQ(PetscObjectSetName((PetscObject)offIS, "atlasOff"));
  CHKERRQ(ISView(offIS, viewer));
  CHKERRQ(ISDestroy(&offIS));
  if (hasConstraints) {
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, "bc"));
    CHKERRQ(ISCreateGeneral(comm, n, cdofs, PETSC_OWN_POINTER, &cdofIS));
    CHKERRQ(PetscObjectSetName((PetscObject)cdofIS, "atlasDof"));
    CHKERRQ(ISView(cdofIS, viewer));
    CHKERRQ(ISDestroy(&cdofIS));
    CHKERRQ(ISCreateGeneral(comm, n, coffs, PETSC_OWN_POINTER, &coffIS));
    CHKERRQ(PetscObjectSetName((PetscObject)coffIS, "atlasOff"));
    CHKERRQ(ISView(coffIS, viewer));
    CHKERRQ(ISDestroy(&coffIS));
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    CHKERRQ(ISCreateGeneral(comm, m, cinds, PETSC_OWN_POINTER, &cindIS));
    CHKERRQ(PetscObjectSetName((PetscObject)cindIS, "bcIndices"));
    CHKERRQ(ISView(cindIS, viewer));
    CHKERRQ(ISDestroy(&cindIS));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionView_HDF5_Internal(PetscSection s, PetscViewer viewer)
{
  PetscInt        numFields, f;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "section"));
  CHKERRQ(PetscSectionGetNumFields(s, &numFields));
  CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "numFields", PETSC_INT, (void *) &numFields));
  CHKERRQ(PetscSectionView_HDF5_SingleField(s, viewer));
  for (f = 0; f < numFields; ++f) {
    char        fname[PETSC_MAX_PATH_LEN];
    const char *fieldName;
    PetscInt    fieldComponents, c;

    CHKERRQ(PetscSNPrintf(fname, sizeof(fname), "field%" PetscInt_FMT, f));
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, fname));
    CHKERRQ(PetscSectionGetFieldName(s, f, &fieldName));
    CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldName", PETSC_STRING, fieldName));
    CHKERRQ(PetscSectionGetFieldComponents(s, f, &fieldComponents));
    CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldComponents", PETSC_INT, (void *) &fieldComponents));
    for (c = 0; c < fieldComponents; ++c) {
      char        cname[PETSC_MAX_PATH_LEN];
      const char *componentName;

      CHKERRQ(PetscSNPrintf(cname, sizeof(cname), "component%" PetscInt_FMT, c));
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, cname));
      CHKERRQ(PetscSectionGetComponentName(s, f, c, &componentName));
      CHKERRQ(PetscViewerHDF5WriteAttribute(viewer, NULL, "componentName", PETSC_STRING, componentName));
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    }
    CHKERRQ(PetscSectionView_HDF5_SingleField(s->field[f], viewer));
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)s, &comm));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(ISGetSize(cindIS, &M));
  CHKERRQ(ISGetLocalSize(cindIS, &m));
  CHKERRQ(PetscMalloc1(m, &coffsets));
  CHKERRQ(ISGetIndices(coffIS, &coffs));
  for (p = pStart, m = 0; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    for (i = 0; i < cdof; ++i) coffsets[m++] = coffs[p-pStart] + i;
  }
  CHKERRQ(ISRestoreIndices(coffIS, &coffs));
  CHKERRQ(PetscSFCreate(comm, &sf));
  CHKERRQ(PetscLayoutCreate(comm, &layout));
  CHKERRQ(PetscLayoutSetSize(layout, M));
  CHKERRQ(PetscLayoutSetLocalSize(layout, m));
  CHKERRQ(PetscLayoutSetBlockSize(layout, 1));
  CHKERRQ(PetscLayoutSetUp(layout));
  CHKERRQ(PetscSFSetGraphLayout(sf, layout, m, NULL, PETSC_OWN_POINTER, coffsets));
  CHKERRQ(PetscLayoutDestroy(&layout));
  CHKERRQ(PetscFree(coffsets));
  CHKERRQ(PetscMalloc1(m, &cinds));
  CHKERRQ(ISGetIndices(cindIS, &data));
  CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, data, cinds, MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, data, cinds, MPI_REPLACE));
  CHKERRQ(ISRestoreIndices(cindIS, &data));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRQ(PetscSectionSetUpBC(s));
  for (p = pStart, m = 0; p < pEnd; ++p) {
    CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
    CHKERRQ(PetscSectionSetConstraintIndices(s, p, &cinds[m]));
    m += cdof;
  }
  CHKERRQ(PetscFree(cinds));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)s, &comm));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "includesConstraints", PETSC_BOOL, NULL, (void *) &includesConstraints));
  CHKERRQ(PetscSectionSetIncludesConstraints(s, includesConstraints));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  n = pEnd - pStart;
#if defined(PETSC_USE_DEBUG)
  CHKERRMPI(MPIU_Allreduce(&n, &N1, 1, MPIU_INT, MPI_SUM, comm));
#endif
  CHKERRQ(ISCreate(comm, &dofIS));
  CHKERRQ(PetscObjectSetName((PetscObject)dofIS, "atlasDof"));
  CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N));
#if defined(PETSC_USE_DEBUG)
  PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->atlasDof: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
  CHKERRQ(ISGetLayout(dofIS, &map));
  CHKERRQ(PetscLayoutSetSize(map, N));
  CHKERRQ(PetscLayoutSetLocalSize(map, n));
  CHKERRQ(ISLoad(dofIS, viewer));
  CHKERRQ(ISCreate(comm, &offIS));
  CHKERRQ(PetscObjectSetName((PetscObject)offIS, "atlasOff"));
  CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "atlasOff", NULL, &N));
#if defined(PETSC_USE_DEBUG)
  PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->atlasOff: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
  CHKERRQ(ISGetLayout(offIS, &map));
  CHKERRQ(PetscLayoutSetSize(map, N));
  CHKERRQ(PetscLayoutSetLocalSize(map, n));
  CHKERRQ(ISLoad(offIS, viewer));
  CHKERRQ(ISGetIndices(dofIS, &dofs));
  CHKERRQ(ISGetIndices(offIS, &offs));
  for (p = pStart, n = 0; p < pEnd; ++p, ++n) {
    CHKERRQ(PetscSectionSetDof(s, p, dofs[n]));
    CHKERRQ(PetscSectionSetOffset(s, p, offs[n]));
  }
  CHKERRQ(ISRestoreIndices(dofIS, &dofs));
  CHKERRQ(ISRestoreIndices(offIS, &offs));
  CHKERRQ(ISDestroy(&dofIS));
  CHKERRQ(ISDestroy(&offIS));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "hasConstraints", PETSC_BOOL, NULL, (void *) &hasConstraints));
  if (hasConstraints) {
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, "bc"));
    CHKERRQ(ISCreate(comm, &cdofIS));
    CHKERRQ(PetscObjectSetName((PetscObject)cdofIS, "atlasDof"));
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N));
#if defined(PETSC_USE_DEBUG)
    PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bc->atlasDof: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
    CHKERRQ(ISGetLayout(cdofIS, &map));
    CHKERRQ(PetscLayoutSetSize(map, N));
    CHKERRQ(PetscLayoutSetLocalSize(map, n));
    CHKERRQ(ISLoad(cdofIS, viewer));
    CHKERRQ(ISGetIndices(cdofIS, &cdofs));
    for (p = pStart, n = 0; p < pEnd; ++p, ++n) {
      CHKERRQ(PetscSectionSetConstraintDof(s, p, cdofs[n]));
    }
    CHKERRQ(ISRestoreIndices(cdofIS, &cdofs));
    CHKERRQ(ISDestroy(&cdofIS));
    CHKERRQ(ISCreate(comm, &coffIS));
    CHKERRQ(PetscObjectSetName((PetscObject)coffIS, "atlasOff"));
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "atlasOff", NULL, &N));
#if defined(PETSC_USE_DEBUG)
    PetscCheckFalse(N1 != N,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bc->atlasOff: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
#endif
    CHKERRQ(ISGetLayout(coffIS, &map));
    CHKERRQ(PetscLayoutSetSize(map, N));
    CHKERRQ(PetscLayoutSetLocalSize(map, n));
    CHKERRQ(ISLoad(coffIS, viewer));
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    CHKERRQ(ISCreate(comm, &cindIS));
    CHKERRQ(PetscObjectSetName((PetscObject)cindIS, "bcIndices"));
    CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "bcIndices", NULL, &M));
    if (!s->bc) m = 0;
    else CHKERRQ(PetscSectionGetStorageSize(s->bc, &m));
#if defined(PETSC_USE_DEBUG)
    CHKERRMPI(MPIU_Allreduce(&m, &M1, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheckFalse(M1 != M,comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bcIndices: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, M1, M, m);
#endif
    CHKERRQ(ISGetLayout(cindIS, &map));
    CHKERRQ(PetscLayoutSetSize(map, M));
    CHKERRQ(PetscLayoutSetLocalSize(map, m));
    CHKERRQ(ISLoad(cindIS, viewer));
    CHKERRQ(PetscSectionLoad_HDF5_SingleField_SetConstraintIndices(s, cindIS, coffIS));
    CHKERRQ(ISDestroy(&coffIS));
    CHKERRQ(ISDestroy(&cindIS));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSectionLoad_HDF5_Internal(PetscSection s, PetscViewer viewer)
{
  MPI_Comm        comm;
  PetscInt        N, n, numFields, f;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)s, &comm));
  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "section"));
  CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "numFields", PETSC_INT, NULL, (void *)&numFields));
  if (s->pStart < 0 && s->pEnd < 0) n = PETSC_DECIDE;
  else {
    PetscCheckFalse(s->pStart != 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "s->pStart must be 0 (got %" PetscInt_FMT ")", s->pStart);
    PetscCheckFalse(s->pEnd < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "s->pEnd must be >= 0, (got %" PetscInt_FMT ")", s->pEnd);
    n = s->pEnd;
  }
  if (numFields > 0) CHKERRQ(PetscSectionSetNumFields(s, numFields));
  CHKERRQ(PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N));
  if (n == PETSC_DECIDE) CHKERRQ(PetscSplitOwnership(comm, &n, &N));
  CHKERRQ(PetscSectionSetChart(s, 0, n));
  CHKERRQ(PetscSectionLoad_HDF5_SingleField(s, viewer));
  for (f = 0; f < numFields; ++f) {
    char      fname[PETSC_MAX_PATH_LEN];
    char     *fieldName;
    PetscInt  fieldComponents, c;

    CHKERRQ(PetscSNPrintf(fname, sizeof(fname), "field%" PetscInt_FMT, f));
    CHKERRQ(PetscViewerHDF5PushGroup(viewer, fname));
    CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "fieldName", PETSC_STRING, NULL, &fieldName));
    CHKERRQ(PetscSectionSetFieldName(s, f, fieldName));
    CHKERRQ(PetscFree(fieldName));
    CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "fieldComponents", PETSC_INT, NULL, (void *) &fieldComponents));
    CHKERRQ(PetscSectionSetFieldComponents(s, f, fieldComponents));
    for (c = 0; c < fieldComponents; ++c) {
      char  cname[PETSC_MAX_PATH_LEN];
      char *componentName;

      CHKERRQ(PetscSNPrintf(cname, sizeof(cname), "component%" PetscInt_FMT, c));
      CHKERRQ(PetscViewerHDF5PushGroup(viewer, cname));
      CHKERRQ(PetscViewerHDF5ReadAttribute(viewer, NULL, "componentName", PETSC_STRING, NULL, &componentName));
      CHKERRQ(PetscSectionSetComponentName(s, f, c, componentName));
      CHKERRQ(PetscFree(componentName));
      CHKERRQ(PetscViewerHDF5PopGroup(viewer));
    }
    CHKERRQ(PetscSectionLoad_HDF5_SingleField(s->field[f], viewer));
    CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(0);
}

#endif
