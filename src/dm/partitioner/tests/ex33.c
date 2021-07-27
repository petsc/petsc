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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  nparts = size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nparts",&nparts,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-vwgts",&vwgts,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-pwgts",&pwgts,NULL);CHKERRQ(ierr);

  /* create PetscPartitioner */
  ierr = PetscPartitionerCreate(PETSC_COMM_WORLD,&p);CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(p,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(p);CHKERRQ(ierr);

  /* create partition section */
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&partSection);CHKERRQ(ierr);

  if (vwgts) { /* create vertex weights section */
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&vertexSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(vertexSection,0,nv);CHKERRQ(ierr);
    for (i = 0; i< nv; i++) {ierr = PetscSectionSetDof(vertexSection,i,1);CHKERRQ(ierr);}
    ierr = PetscSectionSetUp(vertexSection);CHKERRQ(ierr);
  }

  if (pwgts) { /* create partition weights section */
    ierr = PetscSectionCreate(PETSC_COMM_WORLD,&targetSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(targetSection,0,nparts);CHKERRQ(ierr);
    for (i = 0; i< nparts; i++) {ierr = PetscSectionSetDof(targetSection,i,1);CHKERRQ(ierr);}
    ierr = PetscSectionSetUp(targetSection);CHKERRQ(ierr);
  }

#if defined(PETSC_USE_LOG)
  { /* Test logging */
    PetscLogEvent event;

    ierr = PetscLogEventRegister("MyPartitionerEvent",PETSCPARTITIONER_CLASSID,&event);CHKERRQ(ierr);
    { /* PetscLogEventExcludeClass is broken, new events are not deactivated */
      char      logList[256];
      PetscBool opt,pkg;

      ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
      if (opt) {
        ierr = PetscStrInList("partitioner",logList,',',&pkg);CHKERRQ(ierr);
        if (pkg) {ierr = PetscLogEventExcludeClass(PETSCPARTITIONER_CLASSID);CHKERRQ(ierr);}
      }
    }
    ierr = PetscLogEventBegin(event,p,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(event,p,NULL,NULL,NULL);CHKERRQ(ierr);
  }
#endif

  /* test setup and reset */
  ierr = PetscPartitionerSetUp(p);CHKERRQ(ierr);
  ierr = PetscPartitionerReset(p);CHKERRQ(ierr);

  /* test partitioning an empty graph */
  ierr = PetscPartitionerPartition(p,nparts,0,NULL,NULL,vertexSection,targetSection,partSection,&partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)partSection,"NULL SECTION");CHKERRQ(ierr);
  ierr = PetscSectionView(partSection,NULL);CHKERRQ(ierr);
  ierr = ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is,"NULL PARTITION");CHKERRQ(ierr);
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&partition);CHKERRQ(ierr);

  /* test view from options */
  ierr = PetscPartitionerViewFromOptions(p,NULL,"-part_view");CHKERRQ(ierr);

  /* test partitioning a graph on one process only (not main) */
  if (rank == size - 1) {
    ierr = PetscPartitionerPartition(p,nparts,nv,vv,vadj,vertexSection,targetSection,partSection,&partition);CHKERRQ(ierr);
  } else {
    ierr = PetscPartitionerPartition(p,nparts,0,NULL,NULL,vertexSection,targetSection,partSection,&partition);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)partSection,"SEQ SECTION");CHKERRQ(ierr);
  ierr = PetscSectionView(partSection,NULL);CHKERRQ(ierr);
  ierr = ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is,"SEQ PARTITION");CHKERRQ(ierr);
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&partition);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompareAny((PetscObject)p,&sequential,PETSCPARTITIONERCHACO,NULL);CHKERRQ(ierr);
  if (sequential) goto finally;

  /* test partitioning a graph on a subset of the processess only */
  if (rank%2) {
    ierr = PetscPartitionerPartition(p,nparts,0,NULL,NULL,NULL,targetSection,partSection,&partition);CHKERRQ(ierr);
  } else {
    PetscInt i,totv = nv*((size+1)/2),*pvadj;

    ierr = PetscMalloc1(2*nv,&pvadj);CHKERRQ(ierr);
    for (i = 0; i < nv; i++) {
      pvadj[2*i]   = (nv*(rank/2) + totv + i - 1)%totv;
      pvadj[2*i+1] = (nv*(rank/2) + totv + i + 1)%totv;
    }
    ierr = PetscPartitionerPartition(p,nparts,nv,vv,pvadj,NULL,targetSection,partSection,&partition);CHKERRQ(ierr);
    ierr = PetscFree(pvadj);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)partSection,"PARVOID SECTION");CHKERRQ(ierr);
  ierr = PetscSectionView(partSection,NULL);CHKERRQ(ierr);
  ierr = ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is,"PARVOID PARTITION");CHKERRQ(ierr);
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&partition);CHKERRQ(ierr);

finally:
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&vertexSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&targetSection);CHKERRQ(ierr);
  ierr = PetscPartitionerDestroy(&p);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: default

  testset:
    requires: define(PETSC_USE_LOG)
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
