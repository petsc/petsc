
/*
       Tests the ESI_Object interface
*/
#include "ESI.h"

int ESI_Object_test(ESI_Object *obj)
{
  MPI_Comm comm;
  int      rank,ierr;
  void     *interface;

  /* test query interface */
  ierr = obj->queryInterface("DummyInterface",(void **) &interface);
  if (ierr) {printf("error calling obj->queryInterface\n");return ierr;}

  /* test getCommunicator() method */
  ierr = obj->getCommunicator(&comm);
  if (ierr) {printf("error calling obj->getCommunicator\n");return ierr;}
  MPI_Comm_rank(comm,&rank);
  printf("ESI_Object_test: rank %d\n",rank);
  return 0;
}








