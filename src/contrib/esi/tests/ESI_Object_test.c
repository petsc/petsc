
/*
       Tests the ESI_Object interface
*/
#include "ESI.h"
#include "mpi.h"

int ESI_Object_test(esi::Object *obj)
{
  MPI_Comm *comm;
  int      rank;
  esi_int  ierr;
  void     **interface;
  esi_msg  msg;

  /* test query interface */
  ierr = obj->getInterface("DummyInterface",static_cast<void *>(interface),msg);
  if (ierr) {printf("error calling obj->getInterface\n");return ierr;}

  /* test getCommunicator() method */
  ierr = obj->getRunTimeModel("MPI",static_cast<void *>(comm),msg);
  if (ierr) {printf("error calling obj->getRunTimeModel");return ierr;}
  MPI_Comm_rank(*comm,&rank);
  printf("ESI_Object_test: rank %d\n",rank);
  return 0;
}








