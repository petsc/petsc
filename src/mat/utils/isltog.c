#include <petsc/private/matimpl.h>

PetscErrorCode MatCreateFromISLocalToGlobalMapping(ISLocalToGlobalMapping lgmap, Mat A, PetscBool cols, PetscBool trans, MatType ptype, Mat *P)
{
  PetscBool       matfree = PETSC_FALSE;
  const PetscInt *idxs;
  PetscInt        msize, *pidxs, c = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lgmap, IS_LTOGM_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 2);
  PetscValidType(A, 2);
  PetscValidLogicalCollectiveBool(A, cols, 3);
  PetscValidLogicalCollectiveBool(A, trans, 4);
  PetscAssertPointer(P, 6);

  if (!ptype) PetscCall(MatGetType(A, &ptype));
  PetscCall(PetscStrcmpAny(ptype, &matfree, MATSHELL, MATSCATTER, ""));
  PetscCall(ISLocalToGlobalMappingGetIndices(lgmap, &idxs));
  PetscCall(ISLocalToGlobalMappingGetSize(lgmap, &msize));
  PetscCall(PetscMalloc1(msize, &pidxs));
  for (PetscInt i = 0; i < msize; i++)
    if (idxs[i] >= 0) pidxs[c++] = idxs[i];
  PetscCall(ISLocalToGlobalMappingRestoreIndices(lgmap, &idxs));
  msize = c;
  if (matfree) {
    Vec        v, lv;
    VecType    vtype;
    IS         is;
    VecScatter sct;
    PetscBool  matshell;

    if (cols) PetscCall(MatCreateVecs(A, &v, NULL));
    else PetscCall(MatCreateVecs(A, NULL, &v));
    PetscCall(VecGetType(v, &vtype));
    PetscCall(VecCreate(PETSC_COMM_SELF, &lv));
    PetscCall(VecSetSizes(lv, msize, PETSC_DECIDE));
    PetscCall(VecSetType(lv, vtype));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), msize, pidxs, PETSC_USE_POINTER, &is));
    if (trans) PetscCall(VecScatterCreate(lv, NULL, v, is, &sct));
    else PetscCall(VecScatterCreate(v, is, lv, NULL, &sct));
    PetscCall(MatCreateScatter(PetscObjectComm((PetscObject)A), sct, P));
    PetscCall(PetscStrcmp(ptype, MATSHELL, &matshell));
    if (matshell) {
      Mat tP;
      PetscCall(MatConvert(*P, ptype, MAT_INITIAL_MATRIX, &tP));
      PetscCall(MatDestroy(P));
      *P = tP;
    }
    PetscCall(ISDestroy(&is));
    PetscCall(VecScatterDestroy(&sct));
    PetscCall(VecDestroy(&lv));
    PetscCall(VecDestroy(&v));
  } else {
    PetscInt lar, lac, rst;

    PetscCall(MatGetLocalSize(A, &lar, &lac));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), P));
    PetscCall(MatSetType(*P, ptype));
    PetscCall(MatSetSizes(*P, msize, cols ? lac : lar, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSeqAIJSetPreallocation(*P, 1, NULL));
    PetscCall(MatMPIAIJSetPreallocation(*P, 1, NULL, 1, NULL));
#if defined(PETSC_HAVE_HYPRE)
    PetscCall(MatHYPRESetPreallocation(*P, 1, NULL, 1, NULL));
#endif
    PetscCall(MatGetOwnershipRange(*P, &rst, NULL));
    for (PetscInt i = 0; i < msize; i++) PetscCall(MatSetValue(*P, i + rst, pidxs[i], 1.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));
    if (trans) {
      Mat tP;
      PetscCall(MatTranspose(*P, MAT_INITIAL_MATRIX, &tP));
      PetscCall(MatDestroy(P));
      *P = tP;
    }
  }
  PetscCall(PetscFree(pidxs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
