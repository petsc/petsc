
/*
       Tests the esi::Object interface
*/
#include "esi/ESI.h"
#include "mpi.h"

int ESI_Object_test(esi::Object *obj)
{
  MPI_Comm       *comm;
  int            rank;
  esi::ErrorCode ierr;
  void           **interface;

  /* test query interface */
  ierr = obj->getInterface("DummyInterface",static_cast<void *>(interface));
  if (ierr) {printf("error calling obj->getInterface\n");return ierr;}

  /* test getCommunicator() method */
  ierr = obj->getRunTimeModel("MPI",static_cast<void *>(comm));
  if (ierr) {printf("error calling obj->getRunTimeModel");return ierr;}
  MPI_Comm_rank(*comm,&rank);
  printf("ESI_Object_test: rank %d\n",rank);
  return 0;
}








