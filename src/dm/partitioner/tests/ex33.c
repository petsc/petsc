static char help[] = "Tests PetscPartitioner.\n\n";

#include <petscpartitioner.h>

int main(int argc, char **argv)
{
  PetscPartitioner p;
  PetscSection     partSection, vertexSection = NULL, targetSection = NULL;
  IS               partition, is;
  PetscMPIInt      size, rank;
  PetscInt         nparts, i;
  PetscInt         nv      = 4;
  PetscInt         vv[5]   = {0, 2, 4, 6, 8};
  PetscInt         vadj[8] = {3, 1, 0, 2, 1, 3, 2, 0};
  PetscBool        sequential;
  PetscBool        vwgts = PETSC_FALSE;
  PetscBool        pwgts = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  nparts = size;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nparts", &nparts, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-vwgts", &vwgts, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-pwgts", &pwgts, NULL));

  /* create PetscPartitioner */
  PetscCall(PetscPartitionerCreate(PETSC_COMM_WORLD, &p));
  PetscCall(PetscPartitionerSetType(p, PETSCPARTITIONERSIMPLE));
  PetscCall(PetscPartitionerSetFromOptions(p));

  /* create partition section */
  PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &partSection));

  if (vwgts) { /* create vertex weights section */
    PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &vertexSection));
    PetscCall(PetscSectionSetChart(vertexSection, 0, nv));
    for (i = 0; i < nv; i++) PetscCall(PetscSectionSetDof(vertexSection, i, 1));
    PetscCall(PetscSectionSetUp(vertexSection));
  }

  if (pwgts) { /* create partition weights section */
    PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &targetSection));
    PetscCall(PetscSectionSetChart(targetSection, 0, nparts));
    for (i = 0; i < nparts; i++) PetscCall(PetscSectionSetDof(targetSection, i, 1));
    PetscCall(PetscSectionSetUp(targetSection));
  }

#if defined(PETSC_USE_LOG)
  { /* Test logging */
    PetscLogEvent event;

    PetscCall(PetscLogEventRegister("MyPartitionerEvent", PETSCPARTITIONER_CLASSID, &event));
    { /* PetscLogEventExcludeClass is broken, new events are not deactivated */
      char      logList[256];
      PetscBool opt, pkg;

      PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
      if (opt) {
        PetscCall(PetscStrInList("partitioner", logList, ',', &pkg));
        if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCPARTITIONER_CLASSID));
      }
    }
    PetscCall(PetscLogEventBegin(event, p, NULL, NULL, NULL));
    PetscCall(PetscLogEventEnd(event, p, NULL, NULL, NULL));
  }
#endif

  /* test setup and reset */
  PetscCall(PetscPartitionerSetUp(p));
  PetscCall(PetscPartitionerReset(p));

  /* test partitioning an empty graph */
  PetscCall(PetscPartitionerPartition(p, nparts, 0, NULL, NULL, vertexSection, targetSection, partSection, &partition));
  PetscCall(PetscObjectSetName((PetscObject)partSection, "NULL SECTION"));
  PetscCall(PetscSectionView(partSection, NULL));
  PetscCall(ISOnComm(partition, PETSC_COMM_WORLD, PETSC_USE_POINTER, &is));
  PetscCall(PetscObjectSetName((PetscObject)is, "NULL PARTITION"));
  PetscCall(ISView(is, NULL));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&partition));

  /* test view from options */
  PetscCall(PetscPartitionerViewFromOptions(p, NULL, "-part_view"));

  /* test partitioning a graph on one process only (not main) */
  if (rank == size - 1) {
    PetscCall(PetscPartitionerPartition(p, nparts, nv, vv, vadj, vertexSection, targetSection, partSection, &partition));
  } else {
    PetscCall(PetscPartitionerPartition(p, nparts, 0, NULL, NULL, vertexSection, targetSection, partSection, &partition));
  }
  PetscCall(PetscObjectSetName((PetscObject)partSection, "SEQ SECTION"));
  PetscCall(PetscSectionView(partSection, NULL));
  PetscCall(ISOnComm(partition, PETSC_COMM_WORLD, PETSC_USE_POINTER, &is));
  PetscCall(PetscObjectSetName((PetscObject)is, "SEQ PARTITION"));
  PetscCall(ISView(is, NULL));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&partition));

  PetscCall(PetscObjectTypeCompareAny((PetscObject)p, &sequential, PETSCPARTITIONERCHACO, NULL));
  if (sequential) goto finally;

  /* test partitioning a graph on a subset of the processes only */
  if (rank % 2) {
    PetscCall(PetscPartitionerPartition(p, nparts, 0, NULL, NULL, NULL, targetSection, partSection, &partition));
  } else {
    PetscInt i, totv = nv * ((size + 1) / 2), *pvadj;

    PetscCall(PetscMalloc1(2 * nv, &pvadj));
    for (i = 0; i < nv; i++) {
      pvadj[2 * i]     = (nv * (rank / 2) + totv + i - 1) % totv;
      pvadj[2 * i + 1] = (nv * (rank / 2) + totv + i + 1) % totv;
    }
    PetscCall(PetscPartitionerPartition(p, nparts, nv, vv, pvadj, NULL, targetSection, partSection, &partition));
    PetscCall(PetscFree(pvadj));
  }
  PetscCall(PetscObjectSetName((PetscObject)partSection, "PARVOID SECTION"));
  PetscCall(PetscSectionView(partSection, NULL));
  PetscCall(ISOnComm(partition, PETSC_COMM_WORLD, PETSC_USE_POINTER, &is));
  PetscCall(PetscObjectSetName((PetscObject)is, "PARVOID PARTITION"));
  PetscCall(ISView(is, NULL));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&partition));

finally:
  PetscCall(PetscSectionDestroy(&partSection));
  PetscCall(PetscSectionDestroy(&vertexSection));
  PetscCall(PetscSectionDestroy(&targetSection));
  PetscCall(PetscPartitionerDestroy(&p));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: default

  testset:
    requires: defined(PETSC_USE_LOG)
    args: -petscpartitioner_type simple -log_summary
    filter: grep MyPartitionerEvent | cut -d " " -f 1
    test:
       suffix: log_include
    test:
      suffix: log_exclude
      args: -log_exclude partitioner

  test:
    suffix: simple
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -pwgts {{false true}separate output} -petscpartitioner_type simple -petscpartitioner_view

  test:
    suffix: shell
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -petscpartitioner_type shell -petscpartitioner_shell_random -petscpartitioner_view

  test:
    suffix: gather
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -petscpartitioner_type gather -petscpartitioner_view -petscpartitioner_view_graph

  test:
    requires: parmetis
    suffix: parmetis
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -pwgts {{false true}} -vwgts {{false true}}
    args: -petscpartitioner_type parmetis -petscpartitioner_view -petscpartitioner_view_graph

  test:
    requires: parmetis
    suffix: parmetis_type
    nsize: {{1 2}}
    args: -petscpartitioner_type parmetis -part_view
    args: -petscpartitioner_parmetis_type {{kway rb}separate output}
    filter: grep "ParMetis type"

  test:
    requires: ptscotch
    suffix: ptscotch
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -pwgts {{false true}separate output} -vwgts {{false true}}
    args: -petscpartitioner_type ptscotch -petscpartitioner_view -petscpartitioner_view_graph

  test:
    requires: ptscotch
    suffix: ptscotch_strategy
    nsize: {{1 2}}
    args: -petscpartitioner_type ptscotch -part_view
    args: -petscpartitioner_ptscotch_strategy {{DEFAULT QUALITY SPEED BALANCE SAFETY SCALABILITY RECURSIVE REMAP}separate output}
    filter: grep "partitioning strategy"

  test:
    requires: chaco
    suffix: chaco
    nsize: {{1 2 3}separate output}
    args: -nparts {{1}separate output} -petscpartitioner_type chaco -petscpartitioner_view -petscpartitioner_view_graph

  test:
    TODO: non reproducible (uses C stdlib rand())
    requires: chaco
    suffix: chaco
    nsize: {{1 2 3}separate output}
    args: -nparts {{2 3}separate output} -petscpartitioner_type chaco -petscpartitioner_view -petscpartitioner_view_graph

TEST*/
