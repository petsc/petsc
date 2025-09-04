/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include <petsc/private/matimpl.h>
#include <../src/ksp/pc/impls/gamg/gamg.h> /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>

// PetscClangLinter pragma disable: -fdoc-sowing-chars
/*
  PCGAMGGetDataWithGhosts - Get array of local + ghost data with local data
  hacks into Mat MPIAIJ so this must have size > 1

  Input Parameters:
+ Gmat    - MPIAIJ matrix for scatters
. data_sz - number of data terms per node (# cols in output)
- data_in - column-oriented local data of size nloc*data_sz

  Output Parameters:
+ a_stride - number of rows of output (locals+ghosts)
- a_data_out - output data with ghosts of size stride*data_sz

*/
PetscErrorCode PCGAMGGetDataWithGhosts(Mat Gmat, PetscInt data_sz, PetscReal data_in[], PetscInt *a_stride, PetscReal **a_data_out)
{
  Vec          tmp_crds;
  Mat_MPIAIJ  *mpimat;
  PetscInt     nnodes, num_ghosts, dir, kk, jj, my0, Iend, nloc;
  PetscScalar *data_arr;
  PetscReal   *datas;
  PetscBool    isMPIAIJ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Gmat, MAT_CLASSID, 1);
  mpimat = (Mat_MPIAIJ *)Gmat->data;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ));
  PetscCall(MatGetOwnershipRange(Gmat, &my0, &Iend));
  nloc = Iend - my0;
  PetscCall(VecGetLocalSize(mpimat->lvec, &num_ghosts));
  nnodes    = num_ghosts + nloc;
  *a_stride = nnodes;
  PetscCall(MatCreateVecs(Gmat, &tmp_crds, NULL));

  PetscCall(PetscMalloc1(data_sz * nnodes, &datas));
  for (dir = 0; dir < data_sz; dir++) {
    /* set local, and global */
    for (kk = 0; kk < nloc; kk++) {
      PetscInt    gid          = my0 + kk;
      PetscScalar crd          = data_in[dir * nloc + kk]; /* col oriented */
      datas[dir * nnodes + kk] = PetscRealPart(crd);       // get local part now

      PetscCall(VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(tmp_crds));
    PetscCall(VecAssemblyEnd(tmp_crds));
    /* scatter / gather ghost data and add to end of output data */
    PetscCall(VecScatterBegin(mpimat->Mvctx, tmp_crds, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat->Mvctx, tmp_crds, mpimat->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArray(mpimat->lvec, &data_arr));
    for (kk = nloc, jj = 0; jj < num_ghosts; kk++, jj++) datas[dir * nnodes + kk] = PetscRealPart(data_arr[jj]);
    PetscCall(VecRestoreArray(mpimat->lvec, &data_arr));
  }
  PetscCall(VecDestroy(&tmp_crds));
  *a_data_out = datas;
  PetscFunctionReturn(PETSC_SUCCESS);
}
