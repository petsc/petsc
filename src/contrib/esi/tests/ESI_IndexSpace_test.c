


/*
       Tests the ESI_MapAlgebraic interface
*/
#include "ESI.h"
#include "mpi.h"

extern int ESI_Map_test(esi::Map<int> *);

int ESI_MapPartition_test(esi::MapPartition<int> *map)
{
  int ierr,length,offset,*offsets,*lengths,size;
  esi_msg msg;
  MPI_Comm *comm;

  ierr = map->getRunTimeModel("MPI",static_cast<void*>(comm),msg);
  ierr = MPI_Comm_size(*comm,&size);

  ierr = ESI_Map_test((esi::Map<int>*) map);
  if (ierr) {printf("error calling ESI_Map_test\n");return ierr;}

  ierr = map->getLocalSize(length,msg);
  if (ierr) {printf("error calling mapalgebraic->getLocalInfo\n");return ierr;}

  ierr = map->getLocalPartitionOffset(offset,msg);
  if (ierr) {printf("error calling mapalgebraic->getLocalInfo\n");return ierr;}
  printf("ESI_MapPartition_test: local length %d offset %d\n",length,offset);

  lengths = new int [size+1];
  ierr = map->getGlobalPartitionSizes(lengths,msg);
  if (ierr) {printf("error calling mapalgebraic->getGlobalInfo\n");return ierr;}

  offsets = new int [size+1];
  ierr = map->getGlobalPartitionOffsets(offsets,msg);
  if (ierr) {printf("error calling mapalgebraic->getGlobalInfo\n");return ierr;}

  printf("ESI_MapPartition_test: total length %d first offset %d\n",length,offsets[0]);
  return 0;
}







