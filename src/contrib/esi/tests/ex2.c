
#include "PETSc_Map.h"

int innerMap(ESI_Map *map)
{
  MPI_Comm comm;

  int end;
  map->getCommunicator(&comm);
  int rank;
  MPI_Comm_rank(comm,&rank);

  int localsize;
  map->getLocalSize(localsize);
  PetscSynchronizedPrintf(comm,"[%d]My size %d\n",rank,localsize);
  PetscSynchronizedFlush(comm);
  return 0;
}




int innerMain()
{
  int ierr;
  
  PETSc_Map *map = new PETSc_Map(MPI_COMM_WORLD,5,PETSC_DECIDE);

  MPI_Comm comm;
  map->getCommunicator(&comm);
  int rank;
  MPI_Comm_rank(comm,&rank);


  int start,end;

  map->getLocalStart(start);
  map->getLocalEnd(end);
 
  PetscSynchronizedPrintf(comm,"[%d]My start %d end %d\n",rank,start,end);
  PetscSynchronizedFlush(comm);

  ESI_Map *emap = (ESI_Map *)map;

  innerMap((ESI_Map *)map);

  int localsize,localoffset;
  map->getLocalInfo(localsize,localoffset);
  PetscSynchronizedPrintf(comm,"[%d]My size %d offset %d\n",rank,localsize,localoffset);

  int globalsize,*globaloffsets;
  map->getGlobalInfo(globalsize,&globaloffsets);
  int size; MPI_Comm_size(comm,&size);
  for (int i=0; i<size+1; i++ ) {
    PetscSynchronizedPrintf(comm,"[%d]Global size %d offset %d\n",rank,globalsize,globaloffsets[i]);
  }
  PetscSynchronizedFlush(comm);


  delete emap;
  return 0;
}

/*
    We don't put any objects in main() because they won't be
  destructed until the end of the routine; after PetscFinalize()
  is called. Thus PETSc memory monitor will detect lots of memory
  that has not yet been freed.
*/
int main(int argc,char **args)
{
  int ierr;

  PetscInitialize(&argc,&args,0,0);
  ierr = innerMain();CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
