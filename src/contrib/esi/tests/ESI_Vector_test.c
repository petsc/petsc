

/*
       Tests the ESI_Vector interface
*/
#include "ESI.h"

extern int ESI_Map_test(esi::Map<int> *);

int ESI_Vector_test(esi::Vector<double,int> *vector)
{
  int            ierr;
  esi::MapPartition<int>        *map;
  esi::Vector<double,int>     *newvector;
  double         mdot[1],dot,norm1,norm2,norm2squared,norminfinity;
  esi_msg msg;

  ierr = vector->getMap(map,msg);
  if (ierr) {printf("error calling vector->getMap()\n");return ierr;}
  ierr = ESI_Map_test(map); 
  if (ierr) {printf("error calling ESI_Map_test\n");return ierr;}

  ierr = vector->clone(newvector,msg);
  if (ierr) {printf("error calling vector->clone() \n");return ierr;}

  ierr = vector->copy(*newvector,msg);
  if (ierr) {printf("error calling vector->copy() \n");return ierr;}

  ierr = vector->put(2.0,msg);
  ierr = newvector->put(1.0,msg);
  if (ierr) {printf("error calling vector->put() \n");return ierr;}

  ierr = vector->scale(2.0,msg);
  if (ierr) {printf("error calling vector->scale() \n");return ierr;}

  ierr = vector->scaleDiagonal(*newvector,msg);
  if (ierr) {printf("error calling vector->scalediagonal() \n");return ierr;}

  ierr = vector->axpy(*newvector,3.0,msg);
  if (ierr) {printf("error calling vector->axpy() \n");return ierr;}

  ierr = vector->aypx(5.0,*newvector,msg);
  if (ierr) {printf("error calling vector->aypx() \n");return ierr;}

  ierr = vector->dot(*newvector,dot,msg);
  if (ierr) {printf("error calling vector->dot() \n");return ierr;}
  printf("ESI_Vector_test: dot %g\n",dot);

  ierr = vector->norm1(norm1,msg);
  if (ierr) {printf("error calling vector->norm1() \n");return ierr;}
  printf("ESI_Vector_test: norm1 %g\n",norm1);

  ierr = vector->norm2(norm2,msg);
  if (ierr) {printf("error calling vector->norm2() \n");return ierr;}
  printf("ESI_Vector_test: norm2 %g\n",norm2);

  ierr = vector->norm2squared(norm2squared,msg);
  if (ierr) {printf("error calling vector->norm2squared() \n");return ierr;}
  printf("ESI_Vector_test: norm2squared %g\n",norm2squared);

  ierr = vector->normInfinity(norminfinity,msg);
  if (ierr) {printf("error calling vector->normInfinity() \n");return ierr;}
  printf("ESI_Vector_test: normInfinity %g\n",norminfinity);

  //  ierr = vector->mdot(1,(esi::Vector<double,int> *[]) &newvector,mdot,msg);
  //if (ierr) {printf("error calling vector->mdot() \n");return ierr;}
  //printf("ESI_Vector_test: normmdot %g\n",mdot[0]);

  int localsize,i;
  ierr = map->getLocalSize(localsize,msg);
  double *values = new double [localsize];
  ierr = vector->getCoefPtrReadWriteLock(values,msg);
  if (ierr) {printf("error calling ->getArrayPointer()\n");return ierr;}

  for (i=0; i<localsize; i++) {
    values[i] = (double) i;
  }

  delete newvector;

  return 0;
}







