static char help[] = "Tests PetscPartitioner.\n\n";

#include <petscpartitioner.h>

int main(int argc, char **argv)
{
  PetscErrorCode   ierr;
  PetscPartitioner p;
  PetscSection     partSection, vertexSection = NULL, targetSection = NULL;
  IS               partition,is;
  PetscMPIInt      size,rank;
  PetscInt         nparts,i;
  PetscInt         nv = 4;
  PetscInt         vv[5] = {0,2,4,6,8};
  PetscInt         vadj[8] = {3,1,0,2,1,3,2,0};
  PetscBool        sequential;
  PetscBool        vwgts = PETSC_FALSE;
  PetscBool        pwgts = PETSC_FALSE;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  nparts = size;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nparts",&nparts,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-vwgts",&vwgts,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-pwgts",&pwgts,NULL));

  /* create PetscPartitioner */
  CHKERRQ(PetscPartitionerCreate(PETSC_COMM_WORLD,&p));
  CHKERRQ(PetscPartitionerSetType(p,PETSCPARTITIONERSIMPLE));
  CHKERRQ(PetscPartitionerSetFromOptions(p));

  /* create partition section */
  CHKERRQ(PetscSectionCreate(PETSC_COMM_WORLD,&partSection));

  if (vwgts) { /* create vertex weights section */
    CHKERRQ(PetscSectionCreate(PETSC_COMM_WORLD,&vertexSection));
    CHKERRQ(PetscSectionSetChart(vertexSection,0,nv));
    for (i = 0; i< nv; i++) CHKERRQ(PetscSectionSetDof(vertexSection,i,1));
    CHKERRQ(PetscSectionSetUp(vertexSection));
  }

  if (pwgts) { /* create partition weights section */
    CHKERRQ(PetscSectionCreate(PETSC_COMM_WORLD,&targetSection));
    CHKERRQ(PetscSectionSetChart(targetSection,0,nparts));
    for (i = 0; i< nparts; i++) CHKERRQ(PetscSectionSetDof(targetSection,i,1));
    CHKERRQ(PetscSectionSetUp(targetSection));
  }

#if defined(PETSC_USE_LOG)
  { /* Test logging */
    PetscLogEvent event;

    CHKERRQ(PetscLogEventRegister("MyPartitionerEvent",PETSCPARTITIONER_CLASSID,&event));
    { /* PetscLogEventExcludeClass is broken, new events are not deactivated */
      char      logList[256];
      PetscBool opt,pkg;

      CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
      if (opt) {
        CHKERRQ(PetscStrInList("partitioner",logList,',',&pkg));
        if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSCPARTITIONER_CLASSID));
      }
    }
    CHKERRQ(PetscLogEventBegin(event,p,NULL,NULL,NULL));
    CHKERRQ(PetscLogEventEnd(event,p,NULL,NULL,NULL));
  }
#endif

  /* test setup and reset */
  CHKERRQ(PetscPartitionerSetUp(p));
  CHKERRQ(PetscPartitionerReset(p));

  /* test partitioning an empty graph */
  CHKERRQ(PetscPartitionerPartition(p,nparts,0,NULL,NULL,vertexSection,targetSection,partSection,&partition));
  CHKERRQ(PetscObjectSetName((PetscObject)partSection,"NULL SECTION"));
  CHKERRQ(PetscSectionView(partSection,NULL));
  CHKERRQ(ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is));
  CHKERRQ(PetscObjectSetName((PetscObject)is,"NULL PARTITION"));
  CHKERRQ(ISView(is,NULL));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&partition));

  /* test view from options */
  CHKERRQ(PetscPartitionerViewFromOptions(p,NULL,"-part_view"));

  /* test partitioning a graph on one process only (not main) */
  if (rank == size - 1) {
    CHKERRQ(PetscPartitionerPartition(p,nparts,nv,vv,vadj,vertexSection,targetSection,partSection,&partition));
  } else {
    CHKERRQ(PetscPartitionerPartition(p,nparts,0,NULL,NULL,vertexSection,targetSection,partSection,&partition));
  }
  CHKERRQ(PetscObjectSetName((PetscObject)partSection,"SEQ SECTION"));
  CHKERRQ(PetscSectionView(partSection,NULL));
  CHKERRQ(ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is));
  CHKERRQ(PetscObjectSetName((PetscObject)is,"SEQ PARTITION"));
  CHKERRQ(ISView(is,NULL));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&partition));

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)p,&sequential,PETSCPARTITIONERCHACO,NULL));
  if (sequential) goto finally;

  /* test partitioning a graph on a subset of the processess only */
  if (rank%2) {
    CHKERRQ(PetscPartitionerPartition(p,nparts,0,NULL,NULL,NULL,targetSection,partSection,&partition));
  } else {
    PetscInt i,totv = nv*((size+1)/2),*pvadj;

    CHKERRQ(PetscMalloc1(2*nv,&pvadj));
    for (i = 0; i < nv; i++) {
      pvadj[2*i]   = (nv*(rank/2) + totv + i - 1)%totv;
      pvadj[2*i+1] = (nv*(rank/2) + totv + i + 1)%totv;
    }
    CHKERRQ(PetscPartitionerPartition(p,nparts,nv,vv,pvadj,NULL,targetSection,partSection,&partition));
    CHKERRQ(PetscFree(pvadj));
  }
  CHKERRQ(PetscObjectSetName((PetscObject)partSection,"PARVOID SECTION"));
  CHKERRQ(PetscSectionView(partSection,NULL));
  CHKERRQ(ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is));
  CHKERRQ(PetscObjectSetName((PetscObject)is,"PARVOID PARTITION"));
  CHKERRQ(ISView(is,NULL));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&partition));

finally:
  CHKERRQ(PetscSectionDestroy(&partSection));
  CHKERRQ(PetscSectionDestroy(&vertexSection));
  CHKERRQ(PetscSectionDestroy(&targetSection));
  CHKERRQ(PetscPartitionerDestroy(&p));
  ierr = PetscFinalize();
  return ierr;
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
