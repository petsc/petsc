#include <petsc.h>
#include <petsc/private/hashmapi.h>
#include <../src/ml/da/impls/ensemble/letkf/letkf.h>

/*
  PetscDALETKFDestroyObsScatter - Release the IS, hash, scatter context, and local work vectors
  built by PetscDALETKFSetupObsScatter(). Idempotent; safe to call when nothing has been set up.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFDestroyObsScatter(PetscDA_LETKF *impl)
{
  PetscFunctionBegin;
  PetscCall(ISDestroy(&impl->obs_is_local));
  PetscCall(VecScatterDestroy(&impl->obs_scat));
  PetscCall(VecDestroy(&impl->obs_work));
  PetscCall(VecDestroy(&impl->y_mean_work));
  PetscCall(VecDestroy(&impl->r_inv_sqrt_work));
  PetscCall(MatDestroy(&impl->Z_work));
  PetscCall(PetscHMapIDestroy(&impl->obs_g2l));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFSetupObsScatter - Build the IS, global-to-local hash, scatter context, and local
  work vectors needed by the per-vertex CPU and Kokkos analysis paths.

  For each row of impl->Q owned by this rank, collect the unique global observation indices
  referenced by that row's column indices. Sort them, build an IS, and create a VecScatter from
  any global observation-space vector (template taken from H) into a sequential work vector of
  matching length. Also build a hash mapping global obs index -> local position so the per-vertex
  extractor can translate column indices on the fly.

  Backend-agnostic: contains no Kokkos calls. Both the CPU analysis path and the device-CSR setup
  in PetscDALETKFSetupLocalization_Kokkos() consume what this populates.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFSetupObsScatter(PetscDA_LETKF *impl, Mat H)
{
  PetscInt      rstart, rend, nrows, n_obs_global_Q, n_obs_global_H;
  PetscInt      n_obs_local_total = 0, off = 0;
  PetscInt     *obs_indices;
  PetscHashIter iter;
  PetscBool     missing;
  Vec           gvec;
  IS            is_to;

  PetscFunctionBegin;
  PetscCheck(impl->Q, PetscObjectComm((PetscObject)H), PETSC_ERR_ARG_WRONGSTATE, "impl->Q must be installed before PetscDALETKFSetupObsScatter()");

  /* Q's columns are global obs indices; H supplies the source layout for the scatter. They
     must agree on the global obs-space size, else the scatter would dereference out-of-range
     global indices. Catches structurally-mismatched H reaching analysis without invalidating Q. */
  PetscCall(MatGetSize(impl->Q, NULL, &n_obs_global_Q));
  PetscCall(MatGetSize(H, &n_obs_global_H, NULL));
  PetscCheck(n_obs_global_Q == n_obs_global_H, PetscObjectComm((PetscObject)H), PETSC_ERR_ARG_INCOMP, "Q columns (%" PetscInt_FMT ") and H rows (%" PetscInt_FMT ") must agree on global obs-space size; re-supply coordinates if H changed", n_obs_global_Q, n_obs_global_H);

  PetscCall(MatGetOwnershipRange(impl->Q, &rstart, &rend));
  nrows = rend - rstart;

  PetscCall(PetscHMapICreate(&impl->obs_g2l));

  /* First pass: insert every column index into the hashmap to deduplicate. Sizing the workspace
     by the unique count (rather than the sum-of-row-nnz upper bound used previously) keeps peak
     memory O(unique obs) instead of O(sum nnz). */
  for (PetscInt i = 0; i < nrows; i++) {
    const PetscInt *cols;
    PetscInt        nnz;

    PetscCall(MatGetRow(impl->Q, rstart + i, &nnz, &cols, NULL));
    for (PetscInt k = 0; k < nnz; k++) PetscCall(PetscHMapIPut(impl->obs_g2l, cols[k], &iter, &missing));
    PetscCall(MatRestoreRow(impl->Q, rstart + i, &nnz, &cols, NULL));
  }
  PetscCall(PetscHMapIGetSize(impl->obs_g2l, &n_obs_local_total));

  PetscCall(PetscMalloc1(n_obs_local_total, &obs_indices));
  PetscCall(PetscHMapIGetKeys(impl->obs_g2l, &off, obs_indices));
  PetscCall(PetscSortInt(n_obs_local_total, obs_indices));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n_obs_local_total, obs_indices, PETSC_COPY_VALUES, &impl->obs_is_local));

  /* Repopulate obs_g2l with sorted-position values: each global index maps to its slot in obs_work
     after the scatter. */
  PetscCall(PetscHMapIClear(impl->obs_g2l));
  for (PetscInt i = 0; i < n_obs_local_total; i++) PetscCall(PetscHMapISet(impl->obs_g2l, obs_indices[i], i));

  PetscCall(PetscFree(obs_indices));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_obs_local_total, &impl->obs_work));
  PetscCall(VecDuplicate(impl->obs_work, &impl->y_mean_work));
  PetscCall(VecDuplicate(impl->obs_work, &impl->r_inv_sqrt_work));

  PetscCall(MatCreateVecs(H, NULL, &gvec));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, n_obs_local_total, 0, 1, &is_to));
  PetscCall(VecScatterCreate(gvec, impl->obs_is_local, impl->obs_work, is_to, &impl->obs_scat));
  PetscCall(VecDestroy(&gvec));
  PetscCall(ISDestroy(&is_to));
  PetscFunctionReturn(PETSC_SUCCESS);
}
