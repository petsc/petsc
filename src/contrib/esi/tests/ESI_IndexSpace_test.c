


/*
       Tests the ESI_MapAlgebraic interface
*/
#include "esi/ESI.h"
#include "mpi.h"

extern int ESI_Object_test(esi::Object *);

int ESI_IndexSpace_test(esi::IndexSpace<int> *map)
{
  int ierr,length,offset,*offsets,*lengths,size;
  MPI_Comm *comm;

  ierr = map->getRunTimeModel("MPI",static_cast<void*>(comm));
  ierr = MPI_Comm_size(*comm,&size);

  ierr = ESI_Object_test((esi::Object*) map);
  if (ierr) {printf("error calling ESI_Object_test\n");return ierr;}

  ierr = map->getGlobalSize(length);
  if (ierr) {printf("error calling map->getGlobalSize\n");return ierr;}
  printf("ESI_IndexSpace_test: length %d\n",length);

  ierr = map->getLocalSize(length);
  if (ierr) {printf("error calling mapalgebraic->getLocalInfo\n");return ierr;}

  ierr = map->getLocalPartitionOffset(offset);
  if (ierr) {printf("error calling mapalgebraic->getLocalInfo\n");return ierr;}
  printf("ESI_IndexSpace_test: local length %d offset %d\n",length,offset);

  lengths = new int [size+1];
  ierr = map->getGlobalPartitionSizes(lengths);
  if (ierr) {printf("error calling mapalgebraic->getGlobalInfo\n");return ierr;}

  offsets = new int [size+1];
  ierr = map->getGlobalPartitionOffsets(offsets);
  if (ierr) {printf("error calling mapalgebraic->getGlobalInfo\n");return ierr;}

  printf("ESI_IndexSpace_test: total length %d first offset %d\n",length,offsets[0]);
  return 0;
}







