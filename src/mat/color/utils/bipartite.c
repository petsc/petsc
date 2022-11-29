#include <petsc/private/matimpl.h> /*I "petscmat.h"  I*/
#include <petscsf.h>

PETSC_EXTERN PetscErrorCode MatColoringCreateBipartiteGraph(MatColoring mc, PetscSF *etoc, PetscSF *etor)
{
  PetscInt           nentries, ncolentries, idx;
  PetscInt           i, j, rs, re, cs, ce, cn;
  PetscInt          *rowleaf, *colleaf, *rowdata;
  PetscInt           ncol;
  const PetscScalar *vcol;
  const PetscInt    *icol;
  const PetscInt    *coldegrees, *rowdegrees;
  Mat                m = mc->mat;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(m, &rs, &re));
  PetscCall(MatGetOwnershipRangeColumn(m, &cs, &ce));
  cn       = ce - cs;
  nentries = 0;
  for (i = rs; i < re; i++) {
    PetscCall(MatGetRow(m, i, &ncol, NULL, &vcol));
    for (j = 0; j < ncol; j++) nentries++;
    PetscCall(MatRestoreRow(m, i, &ncol, NULL, &vcol));
  }
  PetscCall(PetscMalloc1(nentries, &rowleaf));
  PetscCall(PetscMalloc1(nentries, &rowdata));
  idx = 0;
  for (i = rs; i < re; i++) {
    PetscCall(MatGetRow(m, i, &ncol, &icol, &vcol));
    for (j = 0; j < ncol; j++) {
      rowleaf[idx] = icol[j];
      rowdata[idx] = i;
      idx++;
    }
    PetscCall(MatRestoreRow(m, i, &ncol, &icol, &vcol));
  }
  PetscCheck(idx == nentries, PetscObjectComm((PetscObject)m), PETSC_ERR_NOT_CONVERGED, "Bad number of entries %" PetscInt_FMT " vs %" PetscInt_FMT, idx, nentries);
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)m), etoc));
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)m), etor));

  PetscCall(PetscSFSetGraphLayout(*etoc, m->cmap, nentries, NULL, PETSC_COPY_VALUES, rowleaf));
  PetscCall(PetscSFSetFromOptions(*etoc));

  /* determine the number of entries in the column matrix */
  PetscCall(PetscLogEventBegin(MATCOLORING_Comm, *etoc, 0, 0, 0));
  PetscCall(PetscSFComputeDegreeBegin(*etoc, &coldegrees));
  PetscCall(PetscSFComputeDegreeEnd(*etoc, &coldegrees));
  PetscCall(PetscLogEventEnd(MATCOLORING_Comm, *etoc, 0, 0, 0));
  ncolentries = 0;
  for (i = 0; i < cn; i++) ncolentries += coldegrees[i];
  PetscCall(PetscMalloc1(ncolentries, &colleaf));

  /* create the one going the other way by building the leaf set */
  PetscCall(PetscLogEventBegin(MATCOLORING_Comm, *etoc, 0, 0, 0));
  PetscCall(PetscSFGatherBegin(*etoc, MPIU_INT, rowdata, colleaf));
  PetscCall(PetscSFGatherEnd(*etoc, MPIU_INT, rowdata, colleaf));
  PetscCall(PetscLogEventEnd(MATCOLORING_Comm, *etoc, 0, 0, 0));

  /* this one takes mat entries in *columns* to rows -- you never have to actually be able to order the leaf entries. */
  PetscCall(PetscSFSetGraphLayout(*etor, m->rmap, ncolentries, NULL, PETSC_COPY_VALUES, colleaf));
  PetscCall(PetscSFSetFromOptions(*etor));

  PetscCall(PetscLogEventBegin(MATCOLORING_Comm, *etor, 0, 0, 0));
  PetscCall(PetscSFComputeDegreeBegin(*etor, &rowdegrees));
  PetscCall(PetscSFComputeDegreeEnd(*etor, &rowdegrees));
  PetscCall(PetscLogEventEnd(MATCOLORING_Comm, *etor, 0, 0, 0));

  PetscCall(PetscFree(rowdata));
  PetscCall(PetscFree(rowleaf));
  PetscCall(PetscFree(colleaf));
  PetscFunctionReturn(0);
}
