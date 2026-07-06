static char help[] = "Tests MatMeshToCellGraph()\n\n";

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             mesh, dual;
  PetscInt        Nvertices = 6; /* total number of vertices */
  PetscInt        ncells, *ii, *jj;
  PetscMPIInt     size, rank;
  MatPartitioning part;
  IS              is;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (size == 1) {
    ncells = 4;
    PetscCall(PetscMalloc1(ncells + 1, &ii));
    PetscCall(PetscMalloc1(3 * ncells, &jj));
    ii[0] = 0;
    ii[1] = 3;
    ii[2] = 6;
    ii[3] = 9;
    ii[4] = 12;
    /* cell 0 */
    jj[0] = 0;
    jj[1] = 1;
    jj[2] = 2;
    /* cell 1 */
    jj[3] = 1;
    jj[4] = 2;
    jj[5] = 3;
    /* cell 2 */
    jj[6] = 1;
    jj[7] = 4;
    jj[8] = 5;
    /* cell 3 */
    jj[9]  = 1;
    jj[10] = 3;
    jj[11] = 5;
  } else {
    PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example is for one or two processes");
    ncells = 2;
    PetscCall(PetscMalloc1(3, &ii));
    PetscCall(PetscMalloc1(6, &jj));
    ii[0] = 0;
    ii[1] = 3;
    ii[2] = 6;
    if (rank == 0) {
      jj[0] = 0;
      jj[1] = 1;
      jj[2] = 2;
      jj[3] = 1;
      jj[4] = 2;
      jj[5] = 3;
    } else {
      jj[0] = 1;
      jj[1] = 4;
      jj[2] = 5;
      jj[3] = 1;
      jj[4] = 3;
      jj[5] = 5;
    }
  }
  PetscCall(MatCreateMPIAdj(MPI_COMM_WORLD, ncells, Nvertices, ii, jj, NULL, &mesh));
  PetscCall(MatMeshToCellGraph(mesh, 2, &dual));
  PetscCall(MatView(dual, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatPartitioningCreate(MPI_COMM_WORLD, &part));
  PetscCall(MatPartitioningSetAdjacency(part, dual));
  PetscCall(MatPartitioningSetFromOptions(part));
  PetscCall(MatPartitioningApply(part, &is));
  PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&is));
  PetscCall(MatPartitioningDestroy(&part));

  PetscCall(MatDestroy(&mesh));
  PetscCall(MatDestroy(&dual));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      requires: parmetis

   test:
      suffix: metis
      nsize: 1
      requires: metis
      args: -mat_mesh_to_cell_graph_type metis -mat_partitioning_type current

TEST*/
