static char help[] = "Example of using graph partitioning to partition a graph\n\n";

/*T
   Concepts: Mat^mat partitioning
   Concepts: Mat^image segmentation
   Processors: n
T*/

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             A;
  MatPartitioning part;
  IS              is;
  PetscInt        r,N = 10, start, end, *vweights;
  PetscBool       set_vweights=PETSC_FALSE,use_edge_weights=PETSC_FALSE;
  PetscMPIInt     rank;
  MPI_Comm        comm;

  CHKERRQ(PetscInitialize(&argc, &args, (char*) 0, help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(MatCreate(comm, &A));
  CHKERRQ(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqAIJSetPreallocation(A, 3, NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_vertex_weights",&set_vweights,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_use_edge_weights",&use_edge_weights,NULL));
  /* Create a linear mesh */
  CHKERRQ(MatGetOwnershipRange(A, &start, &end));
  if (set_vweights) {
    CHKERRQ(PetscMalloc1(end-start,&vweights));
    for (r = start; r < end; ++r)
      vweights[r-start] = rank+1;
  }
  for (r = start; r < end; ++r) {
    if (r == 0) {
      PetscInt    cols[2];
      PetscScalar vals[2];

      cols[0] = r;   cols[1] = r+1;
      vals[0] = 1.0; vals[1] = use_edge_weights? 2.0: 1.0;

      CHKERRQ(MatSetValues(A, 1, &r, 2, cols, vals, INSERT_VALUES));
    } else if (r == N-1) {
      PetscInt    cols[2];
      PetscScalar vals[2];

      cols[0] = r-1; cols[1] = r;
      vals[0] = use_edge_weights? 3.0:1.0; vals[1] = 1.0;

      CHKERRQ(MatSetValues(A, 1, &r, 2, cols, vals, INSERT_VALUES));
    } else {
      PetscInt    cols[3];
      PetscScalar vals[3];

      cols[0] = r-1; cols[1] = r;   cols[2] = r+1;
      /* ADJ matrix needs to be symmetric */
      vals[0] = use_edge_weights? (cols[0]==0? 2.0:5.0):1.0;
      vals[1] = 1.0;
      vals[2] = use_edge_weights? (cols[2]==N-1? 3.0:5.0):1.0;

      CHKERRQ(MatSetValues(A, 1, &r, 3, cols, vals, INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatPartitioningCreate(comm, &part));
  CHKERRQ(MatPartitioningSetAdjacency(part, A));
  if (set_vweights) {
    CHKERRQ(MatPartitioningSetVertexWeights(part,vweights));
  }
  if (use_edge_weights) {
    CHKERRQ(MatPartitioningSetUseEdgeWeights(part,use_edge_weights));

    CHKERRQ(MatPartitioningGetUseEdgeWeights(part,&use_edge_weights));
    PetscCheck(use_edge_weights,comm,PETSC_ERR_ARG_INCOMP, "use_edge_weights flag does not setup correctly ");
  }
  CHKERRQ(MatPartitioningSetFromOptions(part));
  CHKERRQ(MatPartitioningApply(part, &is));
  CHKERRQ(ISView(is, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(MatPartitioningDestroy(&part));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      requires: parmetis
      args: -mat_partitioning_type parmetis

   test:
      suffix: 2
      nsize: 3
      requires: ptscotch
      args: -mat_partitioning_type ptscotch

   test:
      suffix: 3
      nsize: 4
      requires: party
      args: -mat_partitioning_type party

   test:
      suffix: 4
      nsize: 3
      requires: chaco
      args: -mat_partitioning_type chaco

   test:
      suffix: 5
      nsize: 3
      requires: parmetis
      args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 3 -mat_partitioning_nparts 10 -N 100

   test:
      suffix: 6
      nsize: 3
      requires: parmetis
      args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 3 -mat_partitioning_nparts 10 -N 100 -test_vertex_weights 1 -mat_partitioning_use_edge_weights 1

   test:
      suffix: 7
      nsize: 2
      requires: parmetis
      args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 2 -mat_partitioning_nparts 10  -mat_partitioning_hierarchical_fineparttype hierarch -malloc_dump -N 100 -mat_partitioning_improve 1

   test:
      suffix: 8
      nsize: 2
      requires: parmetis
      args: -mat_partitioning_type parmetis -mat_partitioning_nparts 3 -test_use_edge_weights 1

   test:
      suffix: 9
      nsize: 2
      requires: ptscotch
      args: -mat_partitioning_type ptscotch -mat_partitioning_nparts 3 -test_use_edge_weights 1 -mat_partitioning_ptscotch_proc_weight 0

TEST*/
