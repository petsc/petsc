
#include "petsc/map.h"

int innerMap(esi::Map<int> *map)
{
  MPI_Comm *comm;
  int      ierr;
  esi_msg  msg;

  int end;
  map->getRunTimeModel("MPI",static_cast<void*>(comm),msg);
  int rank;
  MPI_Comm_rank(*comm,&rank);

  int localsize,offset;
  esi::MapPartition<int> *lmap;
  ierr = map->getInterface("esi::MapPartition",static_cast<void*>(lmap),msg);

  lmap->getLocalSize(localsize,msg);
  lmap->getLocalPartitionOffset(offset,msg);
  PetscSynchronizedPrintf(*comm,"[%d]My size %d my offset %d\n",rank,localsize,offset);
  PetscSynchronizedFlush(*comm);
  return 0;
}

extern int ESI_MapPartition_test(esi::MapPartition<int>*);

int main(int argc,char **args)
{
  int ierr;
  esi_msg msg;

  PetscInitialize(&argc,&args,0,0);
  PETSc_Map<int> *map = new PETSc_Map<int>(MPI_COMM_WORLD,5,PETSC_DECIDE);

  ierr = ESI_MapPartition_test(map); if (ierr) return 1;

  MPI_Comm *comm;
  map->getRunTimeModel("MPI",static_cast<void*>(comm),msg);
  int rank;
  MPI_Comm_rank(*comm,&rank);


  int localsize,offset;

  map->getLocalSize(localsize,msg);
  map->getLocalPartitionOffset(offset,msg);
 
  PetscSynchronizedPrintf(*comm,"[%d]My start %d end %d\n",rank,offset,offset+localsize);
  PetscSynchronizedFlush(*comm);

  esi::Map<int> *emap = (esi::Map<int> *)map;

  innerMap((esi::Map<int> *)map);


  int globalsize;
  map->getGlobalSize(globalsize,msg);
  int size; MPI_Comm_size(*comm,&size);
  int *globaloffsets = new int [size+1];
  map->getGlobalPartitionOffsets(globaloffsets,msg);
  for (int i=0; i<size+1; i++ ) {
    PetscSynchronizedPrintf(*comm,"[%d]Global i [%d] size %d offset %d\n",rank,i,globalsize,globaloffsets[i]);
  }
  PetscSynchronizedFlush(*comm);


  delete emap;

  PetscMap pmap;

  ierr = PetscMapCreateMPI(MPI_COMM_WORLD,5,PETSC_DECIDE,&pmap);

  PETSc_Map<int> *Pmap = new PETSc_Map<int>(pmap);

  PetscMapDestroy(pmap);

  delete Pmap;

  PetscFinalize();

  return 0;
}







