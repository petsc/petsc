static char help[] = "Demonstrates BuildTwoSided functions.\n";

#include <petscsys.h>

typedef struct {
  PetscInt    rank;
  PetscScalar value;
} Unit;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,*toranks,*fromranks,lengths[2],nto,nfrom;
  MPI_Aint       displs[2];
  PetscInt       i,n;
  Unit           *todata,*fromdata;
  MPI_Datatype   dtype,dtypes[2];

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  for (i=1,nto=0; i<size; i*=2) nto++;
  ierr = PetscMalloc2(nto,Unit,&todata,nto,PetscMPIInt,&toranks);CHKERRQ(ierr);
  for (n=0,i=1; i<size; n++,i*=2) {
    toranks[n] = (rank+i) % size;
    todata[n].rank  = (rank+i) % size;
    todata[n].value = (PetscScalar)rank;
  }

  dtypes[0] = MPIU_INT;
  dtypes[1] = MPIU_SCALAR;
  lengths[0] = offsetof(Unit,rank);
  lengths[1] = sizeof(Unit) - offsetof(Unit,rank);
  displs[0] = offsetof(Unit,rank);
  displs[1] = offsetof(Unit,value);
  ierr = MPI_Type_create_struct(2,lengths,displs,dtypes,&dtype);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&dtype);CHKERRQ(ierr);

  ierr = PetscCommBuildTwoSided(PETSC_COMM_WORLD,1,dtype,nto,toranks,todata,&nfrom,&fromranks,&fromdata);CHKERRQ(ierr);
  ierr = MPI_Type_free(&dtype);CHKERRQ(ierr);

  if (nto != nfrom) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] From ranks %d does not match To ranks %d",rank,nto,nfrom);
  for (i=1; i<size; i*=2) {
    PetscMPIInt expected_rank = (rank-i+size)%size;
    for (n=0; n<nfrom; n++) {
      if (expected_rank == fromranks[n]) goto found;
    }
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"[%d] Could not find expected from rank %d",rank,expected_rank);
    found:
    if (fromdata[n].value != expected_rank) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"[%d] Got data %g from rank %d",rank,fromdata[n].value,expected_rank);
  }
  ierr = PetscFree2(todata,toranks);CHKERRQ(ierr);
  ierr = PetscFree(fromdata);CHKERRQ(ierr);
  ierr = PetscFree(fromranks);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

