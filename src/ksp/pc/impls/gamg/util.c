/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include <petsc/private/matimpl.h>
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>

/*
   PCGAMGGetDataWithGhosts - Get array of local + ghost data with local data
    hacks into Mat MPIAIJ so this must have size > 1

   Input Parameter:
   . Gmat - MPIAIJ matrix for scatters
   . data_sz - number of data terms per node (# cols in output)
   . data_in[nloc*data_sz] - column oriented local data

   Output Parameter:
   . a_stride - number of rows of output (locals+ghosts)
   . a_data_out[stride*data_sz] - output data with ghosts

*/
PetscErrorCode PCGAMGGetDataWithGhosts(Mat Gmat,PetscInt data_sz,PetscReal data_in[],PetscInt *a_stride,PetscReal **a_data_out)
{
  Vec            tmp_crds;
  Mat_MPIAIJ     *mpimat = (Mat_MPIAIJ*)Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar    *data_arr;
  PetscReal      *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ));
  PetscCall(MatGetOwnershipRange(Gmat, &my0, &Iend));
  nloc      = Iend - my0;
  PetscCall(VecGetLocalSize(mpimat->lvec, &num_ghosts));
  nnodes    = num_ghosts + nloc;
  *a_stride = nnodes;
  PetscCall(MatCreateVecs(Gmat, &tmp_crds, NULL));

  PetscCall(PetscMalloc1(data_sz*nnodes, &datas));
  for (dir=0; dir<data_sz; dir++) {
    /* set local, and global */
    for (kk=0; kk<nloc; kk++) {
      PetscInt    gid = my0 + kk;
      PetscScalar crd = (PetscScalar)data_in[dir*nloc + kk]; /* col oriented */
      datas[dir*nnodes + kk] = PetscRealPart(crd); // get local part now

      PetscCall(VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(tmp_crds));
    PetscCall(VecAssemblyEnd(tmp_crds));
    /* scatter / gather ghost data and add to end of output data */
    PetscCall(VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGetArray(mpimat->lvec, &data_arr));
    for (kk=nloc,jj=0;jj<num_ghosts;kk++,jj++) datas[dir*nnodes + kk] = PetscRealPart(data_arr[jj]);
    PetscCall(VecRestoreArray(mpimat->lvec, &data_arr));
  }
  PetscCall(VecDestroy(&tmp_crds));
  *a_data_out = datas;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableCreate(PetscInt a_size, PCGAMGHashTable *a_tab)
{
  PetscInt       kk;

  PetscFunctionBegin;
  a_tab->size = a_size;
  PetscCall(PetscMalloc2(a_size, &a_tab->table,a_size, &a_tab->data));
  for (kk=0; kk<a_size; kk++) a_tab->table[kk] = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableDestroy(PCGAMGHashTable *a_tab)
{
  PetscFunctionBegin;
  PetscCall(PetscFree2(a_tab->table,a_tab->data));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableAdd(PCGAMGHashTable *a_tab, PetscInt a_key, PetscInt a_data)
{
  PetscInt kk,idx;

  PetscFunctionBegin;
  PetscCheck(a_key>=0,PETSC_COMM_SELF,PETSC_ERR_USER,"Negative key %" PetscInt_FMT ".",a_key);
  for (kk = 0, idx = GAMG_HASH(a_key); kk < a_tab->size; kk++, idx = (idx==(a_tab->size-1)) ? 0 : idx + 1) {
    if (a_tab->table[idx] == a_key) {
      /* exists */
      a_tab->data[idx] = a_data;
      break;
    } else if (a_tab->table[idx] == -1) {
      /* add */
      a_tab->table[idx] = a_key;
      a_tab->data[idx]  = a_data;
      break;
    }
  }
  if (kk==a_tab->size) {
    /* this is not to efficient, waiting until completely full */
    PetscInt       oldsize = a_tab->size, new_size = 2*a_tab->size + 5, *oldtable = a_tab->table, *olddata = a_tab->data;

    a_tab->size = new_size;
    PetscCall(PetscMalloc2(a_tab->size, &a_tab->table,a_tab->size, &a_tab->data));
    for (kk=0;kk<a_tab->size;kk++) a_tab->table[kk] = -1;
    for (kk=0;kk<oldsize;kk++) {
      if (oldtable[kk] != -1) {
        PetscCall(PCGAMGHashTableAdd(a_tab, oldtable[kk], olddata[kk]));
       }
    }
    PetscCall(PetscFree2(oldtable,olddata));
    PetscCall(PCGAMGHashTableAdd(a_tab, a_key, a_data));
  }
  PetscFunctionReturn(0);
}
