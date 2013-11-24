static char help[] = "Demonstrates BuildTwoSided functions.\n";

#include <petscsys.h>

typedef struct {
  PetscInt    rank;
  PetscScalar value;
  char        ok[3];
} Unit;

#undef __FUNCT__
#define __FUNCT__ "MakeDatatype"
static PetscErrorCode MakeDatatype(MPI_Datatype *dtype)
{
  PetscErrorCode ierr;
  MPI_Datatype dtypes[3];
  PetscMPIInt  lengths[3];
  MPI_Aint     displs[3];

  PetscFunctionBegin;
  dtypes[0] = MPIU_INT;
  dtypes[1] = MPIU_SCALAR;
  dtypes[2] = MPI_CHAR;
  lengths[0] = 1;
  lengths[1] = 1;
  lengths[2] = 3;
  displs[0] = offsetof(Unit,rank);
  displs[1] = offsetof(Unit,value);
  displs[2] = offsetof(Unit,ok);
  ierr = MPI_Type_create_struct(3,lengths,displs,dtypes,dtype);CHKERRQ(ierr);
  ierr = MPI_Type_commit(dtype);CHKERRQ(ierr);
  {
    MPI_Aint lb,extent;
    ierr = MPI_Type_get_extent(*dtype,&lb,&extent);CHKERRQ(ierr);
    if (extent != sizeof(Unit)) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_LIB,"New type has extent %d != sizeof(Unit) %d",extent,(int)sizeof(Unit));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size,*toranks,*fromranks,nto,nfrom;
  PetscInt       i,n;
  PetscBool      verbose;
  Unit           *todata,*fromdata;
  MPI_Datatype   dtype;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  verbose = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-verbose",&verbose,NULL);CHKERRQ(ierr);

  for (i=1,nto=0; i<size; i*=2) nto++;
  ierr = PetscMalloc2(nto,&todata,nto,&toranks);CHKERRQ(ierr);
  for (n=0,i=1; i<size; n++,i*=2) {
    toranks[n] = (rank+i) % size;
    todata[n].rank  = (rank+i) % size;
    todata[n].value = (PetscScalar)rank;
    todata[n].ok[0] = 'o';
    todata[n].ok[1] = 'k';
    todata[n].ok[2] = 0;
  }
  if (verbose) {
    for (i=0; i<nto; i++) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] TO %d: {%D, %g, \"%s\"}\n",rank,toranks[i],todata[i].rank,todata[i].value,todata[i].ok);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }

  ierr = MakeDatatype(&dtype);CHKERRQ(ierr);

  ierr = PetscCommBuildTwoSided(PETSC_COMM_WORLD,1,dtype,nto,toranks,todata,&nfrom,&fromranks,&fromdata);CHKERRQ(ierr);
  ierr = MPI_Type_free(&dtype);CHKERRQ(ierr);

  if (verbose) {
    PetscInt *iranks,*iperm;
    ierr = PetscMalloc2(nfrom,&iranks,nfrom,&iperm);CHKERRQ(ierr);
    for (i=0; i<nfrom; i++) {
      iranks[i] = fromranks[i];
      iperm[i] = i;
    }
    /* Receive ordering is non-deterministic in general, so sort to make verbose output deterministic. */
    ierr = PetscSortIntWithPermutation(nfrom,iranks,iperm);CHKERRQ(ierr);
    for (i=0; i<nfrom; i++) {
      PetscInt ip = iperm[i];
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] FROM %d: {%D, %g, \"%s\"}\n",rank,fromranks[ip],fromdata[ip].rank,fromdata[ip].value,fromdata[ip].ok);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscFree2(iranks,iperm);CHKERRQ(ierr);
  }

  if (nto != nfrom) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] From ranks %d does not match To ranks %d",rank,nto,nfrom);
  for (i=1; i<size; i*=2) {
    PetscMPIInt expected_rank = (rank-i+size)%size;
    PetscBool flg;
    for (n=0; n<nfrom; n++) {
      if (expected_rank == fromranks[n]) goto found;
    }
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"[%d] Could not find expected from rank %d",rank,expected_rank);
    found:
    if (fromdata[n].value != expected_rank) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] Got data %g from rank %d",rank,fromdata[n].value,expected_rank);
    ierr = PetscStrcmp(fromdata[n].ok,"ok",&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"[%d] Got string %s from rank %d",rank,fromdata[n].ok,expected_rank);
  }
  ierr = PetscFree2(todata,toranks);CHKERRQ(ierr);
  ierr = PetscFree(fromdata);CHKERRQ(ierr);
  ierr = PetscFree(fromranks);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

