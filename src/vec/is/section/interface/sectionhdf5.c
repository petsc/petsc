#include <petsc/private/sectionimpl.h> /*I "petscsection.h" I*/
#include <petscsf.h>
#include <petscis.h>
#include <petscviewerhdf5.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode PetscSectionView_HDF5_SingleField(PetscSection s, PetscViewer viewer)
{
  MPI_Comm  comm;
  PetscInt  pStart, pEnd, p, n;
  PetscBool hasConstraints, includesConstraints;
  IS        dofIS, offIS, cdofIS, coffIS, cindIS;
  PetscInt *dofs, *offs, *cdofs, *coffs, *cinds, dof, cdof, m, moff, i;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)s, &comm));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  hasConstraints = (s->bc) ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &hasConstraints, 1, MPIU_BOOL, MPI_LOR, comm));
  for (p = pStart, n = 0, m = 0; p < pEnd; ++p) {
    PetscCall(PetscSectionGetDof(s, p, &dof));
    if (dof >= 0) {
      if (hasConstraints) {
        PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
        m += cdof;
      }
      n++;
    }
  }
  PetscCall(PetscMalloc1(n, &dofs));
  PetscCall(PetscMalloc1(n, &offs));
  if (hasConstraints) {
    PetscCall(PetscMalloc1(n, &cdofs));
    PetscCall(PetscMalloc1(n, &coffs));
    PetscCall(PetscMalloc1(m, &cinds));
  }
  for (p = pStart, n = 0, m = 0; p < pEnd; ++p) {
    PetscCall(PetscSectionGetDof(s, p, &dof));
    if (dof >= 0) {
      dofs[n] = dof;
      PetscCall(PetscSectionGetOffset(s, p, &offs[n]));
      if (hasConstraints) {
        const PetscInt *cpinds;

        PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
        PetscCall(PetscSectionGetConstraintIndices(s, p, &cpinds));
        cdofs[n] = cdof;
        coffs[n] = m;
        for (i = 0; i < cdof; ++i) cinds[m++] = cpinds[i];
      }
      n++;
    }
  }
  if (hasConstraints) {
    PetscCallMPI(MPI_Scan(&m, &moff, 1, MPIU_INT, MPI_SUM, comm));
    moff -= m;
    for (p = 0; p < n; ++p) coffs[p] += moff;
  }
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "hasConstraints", PETSC_BOOL, (void *)&hasConstraints));
  PetscCall(PetscSectionGetIncludesConstraints(s, &includesConstraints));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "includesConstraints", PETSC_BOOL, (void *)&includesConstraints));
  PetscCall(ISCreateGeneral(comm, n, dofs, PETSC_OWN_POINTER, &dofIS));
  PetscCall(PetscObjectSetName((PetscObject)dofIS, "atlasDof"));
  PetscCall(ISView(dofIS, viewer));
  PetscCall(ISDestroy(&dofIS));
  PetscCall(ISCreateGeneral(comm, n, offs, PETSC_OWN_POINTER, &offIS));
  PetscCall(PetscObjectSetName((PetscObject)offIS, "atlasOff"));
  PetscCall(ISView(offIS, viewer));
  PetscCall(ISDestroy(&offIS));
  if (hasConstraints) {
    PetscCall(PetscViewerHDF5PushGroup(viewer, "bc"));
    PetscCall(ISCreateGeneral(comm, n, cdofs, PETSC_OWN_POINTER, &cdofIS));
    PetscCall(PetscObjectSetName((PetscObject)cdofIS, "atlasDof"));
    PetscCall(ISView(cdofIS, viewer));
    PetscCall(ISDestroy(&cdofIS));
    PetscCall(ISCreateGeneral(comm, n, coffs, PETSC_OWN_POINTER, &coffIS));
    PetscCall(PetscObjectSetName((PetscObject)coffIS, "atlasOff"));
    PetscCall(ISView(coffIS, viewer));
    PetscCall(ISDestroy(&coffIS));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    PetscCall(ISCreateGeneral(comm, m, cinds, PETSC_OWN_POINTER, &cindIS));
    PetscCall(PetscObjectSetName((PetscObject)cindIS, "bcIndices"));
    PetscCall(ISView(cindIS, viewer));
    PetscCall(ISDestroy(&cindIS));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSectionView_HDF5_Internal(PetscSection s, PetscViewer viewer)
{
  PetscInt numFields, f;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5PushGroup(viewer, "section"));
  PetscCall(PetscSectionGetNumFields(s, &numFields));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "numFields", PETSC_INT, (void *)&numFields));
  PetscCall(PetscSectionView_HDF5_SingleField(s, viewer));
  for (f = 0; f < numFields; ++f) {
    char        fname[PETSC_MAX_PATH_LEN];
    const char *fieldName;
    PetscInt    fieldComponents, c;

    PetscCall(PetscSNPrintf(fname, sizeof(fname), "field%" PetscInt_FMT, f));
    PetscCall(PetscViewerHDF5PushGroup(viewer, fname));
    PetscCall(PetscSectionGetFieldName(s, f, &fieldName));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldName", PETSC_STRING, fieldName));
    PetscCall(PetscSectionGetFieldComponents(s, f, &fieldComponents));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldComponents", PETSC_INT, (void *)&fieldComponents));
    for (c = 0; c < fieldComponents; ++c) {
      char        cname[PETSC_MAX_PATH_LEN];
      const char *componentName;

      PetscCall(PetscSNPrintf(cname, sizeof(cname), "component%" PetscInt_FMT, c));
      PetscCall(PetscViewerHDF5PushGroup(viewer, cname));
      PetscCall(PetscSectionGetComponentName(s, f, c, &componentName));
      PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "componentName", PETSC_STRING, componentName));
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    }
    PetscCall(PetscSectionView_HDF5_SingleField(s->field[f], viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectGetComm((PetscObject)s, &comm));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(ISGetSize(cindIS, &M));
  PetscCall(ISGetLocalSize(cindIS, &m));
  PetscCall(PetscMalloc1(m, &coffsets));
  PetscCall(ISGetIndices(coffIS, &coffs));
  for (p = pStart, m = 0; p < pEnd; ++p) {
    PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
    for (i = 0; i < cdof; ++i) coffsets[m++] = coffs[p - pStart] + i;
  }
  PetscCall(ISRestoreIndices(coffIS, &coffs));
  PetscCall(PetscSFCreate(comm, &sf));
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetSize(layout, M));
  PetscCall(PetscLayoutSetLocalSize(layout, m));
  PetscCall(PetscLayoutSetBlockSize(layout, 1));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscSFSetGraphLayout(sf, layout, m, NULL, PETSC_OWN_POINTER, coffsets));
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCall(PetscFree(coffsets));
  PetscCall(PetscMalloc1(m, &cinds));
  PetscCall(ISGetIndices(cindIS, &data));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, data, cinds, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, data, cinds, MPI_REPLACE));
  PetscCall(ISRestoreIndices(cindIS, &data));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscSectionSetUpBC(s));
  for (p = pStart, m = 0; p < pEnd; ++p) {
    PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
    PetscCall(PetscSectionSetConstraintIndices(s, p, &cinds[m]));
    m += cdof;
  }
  PetscCall(PetscFree(cinds));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSectionLoad_HDF5_SingleField(PetscSection s, PetscViewer viewer)
{
  MPI_Comm comm;
  PetscInt pStart, pEnd, p, N, n, M, m;
  #if defined(PETSC_USE_DEBUG)
  PetscInt N1, M1;
  #endif
  PetscBool       hasConstraints, includesConstraints;
  IS              dofIS, offIS, cdofIS, coffIS, cindIS;
  const PetscInt *dofs, *offs, *cdofs;
  PetscLayout     map;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)s, &comm));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "includesConstraints", PETSC_BOOL, NULL, (void *)&includesConstraints));
  PetscCall(PetscSectionSetIncludesConstraints(s, includesConstraints));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  n = pEnd - pStart;
  #if defined(PETSC_USE_DEBUG)
  PetscCall(MPIU_Allreduce(&n, &N1, 1, MPIU_INT, MPI_SUM, comm));
  #endif
  PetscCall(ISCreate(comm, &dofIS));
  PetscCall(PetscObjectSetName((PetscObject)dofIS, "atlasDof"));
  PetscCall(PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N));
  #if defined(PETSC_USE_DEBUG)
  PetscCheck(N1 == N, comm, PETSC_ERR_ARG_SIZ, "Unable to load s->atlasDof: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
  #endif
  PetscCall(ISGetLayout(dofIS, &map));
  PetscCall(PetscLayoutSetSize(map, N));
  PetscCall(PetscLayoutSetLocalSize(map, n));
  PetscCall(ISLoad(dofIS, viewer));
  PetscCall(ISCreate(comm, &offIS));
  PetscCall(PetscObjectSetName((PetscObject)offIS, "atlasOff"));
  PetscCall(PetscViewerHDF5ReadSizes(viewer, "atlasOff", NULL, &N));
  #if defined(PETSC_USE_DEBUG)
  PetscCheck(N1 == N, comm, PETSC_ERR_ARG_SIZ, "Unable to load s->atlasOff: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
  #endif
  PetscCall(ISGetLayout(offIS, &map));
  PetscCall(PetscLayoutSetSize(map, N));
  PetscCall(PetscLayoutSetLocalSize(map, n));
  PetscCall(ISLoad(offIS, viewer));
  PetscCall(ISGetIndices(dofIS, &dofs));
  PetscCall(ISGetIndices(offIS, &offs));
  for (p = pStart, n = 0; p < pEnd; ++p, ++n) {
    PetscCall(PetscSectionSetDof(s, p, dofs[n]));
    PetscCall(PetscSectionSetOffset(s, p, offs[n]));
  }
  PetscCall(ISRestoreIndices(dofIS, &dofs));
  PetscCall(ISRestoreIndices(offIS, &offs));
  PetscCall(ISDestroy(&dofIS));
  PetscCall(ISDestroy(&offIS));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "hasConstraints", PETSC_BOOL, NULL, (void *)&hasConstraints));
  if (hasConstraints) {
    PetscCall(PetscViewerHDF5PushGroup(viewer, "bc"));
    PetscCall(ISCreate(comm, &cdofIS));
    PetscCall(PetscObjectSetName((PetscObject)cdofIS, "atlasDof"));
    PetscCall(PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N));
  #if defined(PETSC_USE_DEBUG)
    PetscCheck(N1 == N, comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bc->atlasDof: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
  #endif
    PetscCall(ISGetLayout(cdofIS, &map));
    PetscCall(PetscLayoutSetSize(map, N));
    PetscCall(PetscLayoutSetLocalSize(map, n));
    PetscCall(ISLoad(cdofIS, viewer));
    PetscCall(ISGetIndices(cdofIS, &cdofs));
    for (p = pStart, n = 0; p < pEnd; ++p, ++n) PetscCall(PetscSectionSetConstraintDof(s, p, cdofs[n]));
    PetscCall(ISRestoreIndices(cdofIS, &cdofs));
    PetscCall(ISDestroy(&cdofIS));
    PetscCall(ISCreate(comm, &coffIS));
    PetscCall(PetscObjectSetName((PetscObject)coffIS, "atlasOff"));
    PetscCall(PetscViewerHDF5ReadSizes(viewer, "atlasOff", NULL, &N));
  #if defined(PETSC_USE_DEBUG)
    PetscCheck(N1 == N, comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bc->atlasOff: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, N1, N, n);
  #endif
    PetscCall(ISGetLayout(coffIS, &map));
    PetscCall(PetscLayoutSetSize(map, N));
    PetscCall(PetscLayoutSetLocalSize(map, n));
    PetscCall(ISLoad(coffIS, viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    PetscCall(ISCreate(comm, &cindIS));
    PetscCall(PetscObjectSetName((PetscObject)cindIS, "bcIndices"));
    PetscCall(PetscViewerHDF5ReadSizes(viewer, "bcIndices", NULL, &M));
    if (!s->bc) m = 0;
    else PetscCall(PetscSectionGetStorageSize(s->bc, &m));
  #if defined(PETSC_USE_DEBUG)
    PetscCall(MPIU_Allreduce(&m, &M1, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(M1 == M, comm, PETSC_ERR_ARG_SIZ, "Unable to load s->bcIndices: sum of local sizes (%" PetscInt_FMT ") != global size (%" PetscInt_FMT "): local size on this process is %" PetscInt_FMT, M1, M, m);
  #endif
    PetscCall(ISGetLayout(cindIS, &map));
    PetscCall(PetscLayoutSetSize(map, M));
    PetscCall(PetscLayoutSetLocalSize(map, m));
    PetscCall(ISLoad(cindIS, viewer));
    PetscCall(PetscSectionLoad_HDF5_SingleField_SetConstraintIndices(s, cindIS, coffIS));
    PetscCall(ISDestroy(&coffIS));
    PetscCall(ISDestroy(&cindIS));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSectionLoad_HDF5_Internal(PetscSection s, PetscViewer viewer)
{
  MPI_Comm comm;
  PetscInt N, n, numFields, f;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)s, &comm));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "section"));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "numFields", PETSC_INT, NULL, (void *)&numFields));
  if (s->pStart < 0 && s->pEnd < 0) n = PETSC_DECIDE;
  else {
    PetscCheck(s->pStart == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "s->pStart must be 0 (got %" PetscInt_FMT ")", s->pStart);
    PetscCheck(s->pEnd >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "s->pEnd must be >= 0, (got %" PetscInt_FMT ")", s->pEnd);
    n = s->pEnd;
  }
  if (numFields > 0) PetscCall(PetscSectionSetNumFields(s, numFields));
  PetscCall(PetscViewerHDF5ReadSizes(viewer, "atlasDof", NULL, &N));
  if (n == PETSC_DECIDE) PetscCall(PetscSplitOwnership(comm, &n, &N));
  PetscCall(PetscSectionSetChart(s, 0, n));
  PetscCall(PetscSectionLoad_HDF5_SingleField(s, viewer));
  for (f = 0; f < numFields; ++f) {
    char     fname[PETSC_MAX_PATH_LEN];
    char    *fieldName;
    PetscInt fieldComponents, c;

    PetscCall(PetscSNPrintf(fname, sizeof(fname), "field%" PetscInt_FMT, f));
    PetscCall(PetscViewerHDF5PushGroup(viewer, fname));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "fieldName", PETSC_STRING, NULL, &fieldName));
    PetscCall(PetscSectionSetFieldName(s, f, fieldName));
    PetscCall(PetscFree(fieldName));
    PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "fieldComponents", PETSC_INT, NULL, (void *)&fieldComponents));
    PetscCall(PetscSectionSetFieldComponents(s, f, fieldComponents));
    for (c = 0; c < fieldComponents; ++c) {
      char  cname[PETSC_MAX_PATH_LEN];
      char *componentName;

      PetscCall(PetscSNPrintf(cname, sizeof(cname), "component%" PetscInt_FMT, c));
      PetscCall(PetscViewerHDF5PushGroup(viewer, cname));
      PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "componentName", PETSC_STRING, NULL, &componentName));
      PetscCall(PetscSectionSetComponentName(s, f, c, componentName));
      PetscCall(PetscFree(componentName));
      PetscCall(PetscViewerHDF5PopGroup(viewer));
    }
    PetscCall(PetscSectionLoad_HDF5_SingleField(s->field[f], viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
