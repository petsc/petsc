#include <../src/mat/impls/adj/mpi/mpiadj.h> /*I "petscmat.h" I*/

#include <metis.h>

#define PetscCallMETIS(n, func) \
  do { \
    PetscCheck(n != METIS_ERROR_INPUT, PETSC_COMM_SELF, PETSC_ERR_LIB, "METIS error due to wrong inputs and/or options for %s", func); \
    PetscCheck(n != METIS_ERROR_MEMORY, PETSC_COMM_SELF, PETSC_ERR_LIB, "METIS error due to insufficient memory in %s", func); \
    PetscCheck(n != METIS_ERROR, PETSC_COMM_SELF, PETSC_ERR_LIB, "METIS general error in %s", func); \
  } while (0)

#define PetscCallMetis_(name, func, args) \
  do { \
    PetscStackPushExternal(name); \
    int status = func args; \
    PetscStackPop; \
    PetscCallMETIS(status, name); \
  } while (0)

#define PetscCallMetis(func, args) PetscCallMetis_(PetscStringize(func), func, args)

PETSC_EXTERN PetscErrorCode MatMeshToCellGraph_Metis(Mat mesh, PetscInt ncommonnodes, Mat *dual)
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
  PetscCheck(size == 1, comm, PETSC_ERR_SUP, "MatMeshToCellGraph_Metis() requires a sequential matrix (communicator size must be 1)");

  {
    idx_t ne = mesh->rmap->N;
    idx_t nn = mesh->cmap->N;

    PetscCallMetis(METIS_MeshToDual, (&ne, &nn, (idx_t *)adj->i, (idx_t *)adj->j, (idx_t *)&ncommonnodes, (idx_t *)&numflag, (idx_t **)&newxadj, (idx_t **)&newadjncy));
  }

  for (PetscInt i = 0; i < mesh->rmap->N; i++) PetscCall(PetscSortInt(newxadj[i + 1] - newxadj[i], newadjncy + newxadj[i]));

  PetscCall(MatCreateMPIAdj(PetscObjectComm((PetscObject)mesh), mesh->rmap->n, mesh->rmap->N, newxadj, newadjncy, NULL, dual));
  newadj = (Mat_MPIAdj *)(*dual)->data;

  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by METIS */
  PetscFunctionReturn(PETSC_SUCCESS);
}
