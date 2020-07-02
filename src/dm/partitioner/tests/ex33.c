static char help[] = "Tests PetscPartitioner.\n\n";

#include <petscpartitioner.h>

int main(int argc, char **argv)
{
  PetscErrorCode   ierr;
  PetscPartitioner p;
  PetscSection     partSection;
  IS               partition,is;
  PetscMPIInt      size,rank;
  PetscInt         npar;
  PetscInt         nv = 4;
  PetscInt         vv[5] = {0,2,4,6,8};
  PetscInt         vadj[8] = {3,1,0,2,1,3,2,0};
  PetscBool        sequential;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  npar = size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nparts",&npar,NULL);CHKERRQ(ierr);

  /* create PetscPartitioner */
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&partSection);CHKERRQ(ierr);
  ierr = PetscPartitionerCreate(PETSC_COMM_WORLD,&p);CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(p,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(p);CHKERRQ(ierr);

  /* test partitioning an empty graph */
  ierr = PetscPartitionerPartition(p,npar,0,NULL,NULL,NULL,NULL,partSection,&partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)partSection,"NULL SECTION");
  ierr = PetscSectionView(partSection,NULL);CHKERRQ(ierr);
  ierr = ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is,"NULL PARTITION");
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&partition);CHKERRQ(ierr);

  /* test partitioning a graph on one process only (not master) */
  if (rank == size - 1) {
    ierr = PetscPartitionerPartition(p,npar,nv,vv,vadj,NULL,NULL,partSection,&partition);CHKERRQ(ierr);
  } else {
    ierr = PetscPartitionerPartition(p,npar,0,NULL,NULL,NULL,NULL,partSection,&partition);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)partSection,"SEQ SECTION");
  ierr = PetscSectionView(partSection,NULL);CHKERRQ(ierr);
  ierr = ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is,"SEQ PARTITION");
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&partition);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompareAny((PetscObject)p,&sequential,PETSCPARTITIONERCHACO,NULL);CHKERRQ(ierr);
  if (sequential) goto finally;

  /* test partitioning a graph on a subset of the processess only */
  if (rank%2) {
    ierr = PetscPartitionerPartition(p,npar,0,NULL,NULL,NULL,NULL,partSection,&partition);CHKERRQ(ierr);
  } else {
    PetscInt i,totv = nv*((size+1)/2),*pvadj;

    ierr = PetscMalloc1(2*nv,&pvadj);CHKERRQ(ierr);
    for (i = 0; i < nv; i++) {
      pvadj[2*i]   = (nv*(rank/2) + totv + i - 1)%totv;
      pvadj[2*i+1] = (nv*(rank/2) + totv + i + 1)%totv;
    }
    ierr = PetscPartitionerPartition(p,npar,nv,vv,pvadj,NULL,NULL,partSection,&partition);CHKERRQ(ierr);
    ierr = PetscFree(pvadj);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)partSection,"PARVOID SECTION");
  ierr = PetscSectionView(partSection,NULL);CHKERRQ(ierr);
  ierr = ISOnComm(partition,PETSC_COMM_WORLD,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is,"PARVOID PARTITION");
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&partition);CHKERRQ(ierr);

finally:
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = PetscPartitionerDestroy(&p);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: simple
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -petscpartitioner_type simple -petscpartitioner_view -petscpartitioner_view_graph

  test:
    requires: parmetis
    suffix: parmetis
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -petscpartitioner_type parmetis -petscpartitioner_view -petscpartitioner_view_graph

  test:
    requires: ptscotch
    suffix: ptscotch
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -petscpartitioner_type ptscotch -petscpartitioner_view -petscpartitioner_view_graph

  test:
    TODO: broken
    requires: chaco
    suffix: chaco
    nsize: {{1 2 3}separate output}
    args: -nparts {{1 2 3}separate output} -petscpartitioner_type chaco -petscpartitioner_view -petscpartitioner_view_graph

TEST*/
