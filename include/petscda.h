
/*
      Regular array object, for easy parallism of simple grid 
   problems on regular arrays.
*/
#if !defined(__RA_PACKAGE)
#define __RA_PACKAGE
#include "petsc.h"
#include "vec.h"

#define RA_COOKIE PETSC_COOKIE+14

typedef struct _RA* RA;

extern int   RACreate2d(MPI_Comm,int,int,int,int,int,RA *);
extern int   RADestroy(RA);
extern int   RAView(RA,Viewer);
extern int   RAGetDistributedVector(RA,Vec*);
extern int   RAGetLocalVector(RA,Vec*);
extern int   RAGetOwnershipRange(RA,int*,int*,int*,int*);

#endif
