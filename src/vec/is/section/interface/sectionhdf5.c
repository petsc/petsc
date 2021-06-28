#include <petsc/private/sectionimpl.h>   /*I "petscsection.h" I*/
//#include <petscsf.h>
//#include <petsc/private/isimpl.h>
#include <petscviewerhdf5.h>
//#include <petsclayouthdf5.h>

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

    ierr = PetscSNPrintf(fname, sizeof(fname), "field%D", f);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, fname);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldName(s, f, &fieldName);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldName", PETSC_STRING, fieldName);CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, f, &fieldComponents);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "fieldComponents", PETSC_INT, (void *) &fieldComponents);CHKERRQ(ierr);
    for (c = 0; c < fieldComponents; ++c) {
      char        cname[PETSC_MAX_PATH_LEN];
      const char *componentName;

      ierr = PetscSNPrintf(cname, sizeof(cname), "component%D", c);CHKERRQ(ierr);
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
#endif
