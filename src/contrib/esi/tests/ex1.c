
#include "esi/petsc/indexspace.h"

int innerIndexSpace(esi::IndexSpace<int> *map)
{
  MPI_Comm        *comm;
  esi::ErrorCode  ierr;


  int end;
  map->getRunTimeModel("MPI",static_cast<void*>(comm));
  int rank;
  MPI_Comm_rank(*comm,&rank);

  int localsize,offset;
  esi::IndexSpace<int> *lmap;
  ierr = map->getInterface("esi::IndexSpace",static_cast<void*>(lmap));

  lmap->getLocalSize(localsize);
  lmap->getLocalPartitionOffset(offset);
  PetscSynchronizedPrintf(*comm,"[%d]My size %d my offset %d\n",rank,localsize,offset);
  PetscSynchronizedFlush(*comm);
  return 0;
}

extern int ESI_IndexSpace_test(esi::IndexSpace<int>*);

int main(int argc,char **args)
{
  esi::ErrorCode ierr;

  PetscInitialize(&argc,&args,0,0);
  esi::petsc::IndexSpace<int> *map = new esi::petsc::IndexSpace<int>(MPI_COMM_WORLD,5,PETSC_DECIDE);

  ierr = ESI_IndexSpace_test(map); if (ierr) return 1;

  MPI_Comm *comm;
  map->getRunTimeModel("MPI",static_cast<void*>(comm));
  int rank;
  MPI_Comm_rank(*comm,&rank);


  int localsize,offset;

  map->getLocalSize(localsize);
  map->getLocalPartitionOffset(offset);
 
  PetscSynchronizedPrintf(*comm,"[%d]My start %d end %d\n",rank,offset,offset+localsize);
  PetscSynchronizedFlush(*comm);

  esi::IndexSpace<int> *emap = map;

  innerIndexSpace(map);


  int globalsize;
  map->getGlobalSize(globalsize);
  int size; MPI_Comm_size(*comm,&size);
  int *globaloffsets = new int [size+1];
  map->getGlobalPartitionOffsets(globaloffsets);
  for (int i=0; i<size+1; i++ ) {
    PetscSynchronizedPrintf(*comm,"[%d]Global i [%d] size %d offset %d\n",rank,i,globalsize,globaloffsets[i]);
  }
  PetscSynchronizedFlush(*comm);


  delete emap;
  PetscMap pmap;

  ierr = PetscMapCreateMPI(MPI_COMM_WORLD,5,PETSC_DECIDE,&pmap);

  esi::petsc::IndexSpace<int> *Pmap = new esi::petsc::IndexSpace<int>(pmap);

  PetscMapDestroy(pmap);

  delete Pmap;

  PetscFinalize();

  return 0;
}







