
#include "PETSc_Map.h"

int innerMap(ESI_Map *map)
{
  MPI_Comm comm;
  int      ierr;

  int end;
  map->getCommunicator(&comm);
  int rank;
  MPI_Comm_rank(comm,&rank);

  int localsize,offset;
  ESI_MapAlgebraic *lmap;
  ierr = map->queryInterface("ESI_MapAlgebraic",(void **)&lmap);

  lmap->getLocalInfo(localsize,offset);
  PetscSynchronizedPrintf(comm,"[%d]My size %d\n",rank,localsize);
  PetscSynchronizedFlush(comm);
  return 0;
}

extern int ESI_MapAlgebraic_test(ESI_MapAlgebraic*);

int main(int argc,char **args)
{
  int ierr;

  PetscInitialize(&argc,&args,0,0);
  PETSc_Map *map = new PETSc_Map(MPI_COMM_WORLD,5,PETSC_DECIDE);

  ierr = ESI_MapAlgebraic_test(map); if (ierr) return 1;

  MPI_Comm comm;
  map->getCommunicator(&comm);
  int rank;
  MPI_Comm_rank(comm,&rank);


  int localsize,offset;

  map->getLocalInfo(localsize,offset);
 
  PetscSynchronizedPrintf(comm,"[%d]My start %d end %d\n",rank,offset,offset+localsize);
  PetscSynchronizedFlush(comm);

  ESI_Map *emap = (ESI_Map *)map;

  innerMap((ESI_Map *)map);


  int globalsize,*globaloffsets;
  map->getGlobalInfo(globalsize,globaloffsets);
  int size; MPI_Comm_size(comm,&size);
  for (int i=0; i<size+1; i++ ) {
    PetscSynchronizedPrintf(comm,"[%d]Global size %d offset %d\n",rank,globalsize,globaloffsets[i]);
  }
  PetscSynchronizedFlush(comm);


  delete emap;

  Map pmap;

  ierr = MapCreateMPI(MPI_COMM_WORLD,5,PETSC_DECIDE,&pmap);

  PETSc_Map *Pmap = new PETSc_Map(pmap);

  MapDestroy(pmap);

  delete Pmap;

  PetscFinalize();

  return 0;
}







