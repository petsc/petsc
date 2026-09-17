#include <../src/mat/impls/adj/mpi/mpiadj.h> /*I "petscmat.h" I*/
#include <petsc/private/matmetisimpl.h>
#include <metis.h>

PETSC_EXTERN PetscErrorCode MatMeshToCellGraph_METIS(Mat mesh, PetscInt ncommonnodes, Mat *dual)
{
  PetscInt   *newxadj, *newadjncy;
  PetscInt    numflag = 0;
  Mat_MPIAdj *adj     = (Mat_MPIAdj *)mesh->data, *newadj;
  PetscBool   flg;
  MPI_Comm    comm;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)mesh, MATMPIADJ, &flg));
  PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));
  PetscCheck(flg, comm, PETSC_ERR_SUP, "Must use MPIAdj matrix type");

  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "MatMeshToCellGraph_METIS() requires a sequential matrix (communicator size must be 1)");

  {
    idx_t ne = mesh->rmap->N;
    idx_t nn = mesh->cmap->N;

    PetscCallMETIS(METIS_MeshToDual, &ne, &nn, (idx_t *)adj->i, (idx_t *)adj->j, (idx_t *)&ncommonnodes, (idx_t *)&numflag, (idx_t **)&newxadj, (idx_t **)&newadjncy);
  }

  for (PetscInt i = 0; i < mesh->rmap->N; i++) PetscCall(PetscSortInt(newxadj[i + 1] - newxadj[i], newadjncy + newxadj[i]));

  PetscCall(MatCreateMPIAdj(PetscObjectComm((PetscObject)mesh), mesh->rmap->n, mesh->rmap->N, newxadj, newadjncy, NULL, dual));
  newadj = (Mat_MPIAdj *)(*dual)->data;

  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by METIS */
  PetscFunctionReturn(PETSC_SUCCESS);
}
