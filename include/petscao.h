/* $Id: ao.h,v 1.1 1996/06/25 19:10:32 bsmith Exp bsmith $ */

/* 
   An application ordering is mapping between application-centric
  ordering (the ordering that is "natural" to the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__AO_PACKAGE)
#define __AO_PACKAGE
#include "is.h"

typedef enum {AO_DEBUG=0, AO_BASIC=1} AOType;

#define AO_COOKIE PETSC_COOKIE+20

typedef struct _AO* AO;

extern int AOCreateDebug(MPI_Comm,int,int*,int*,AO*);
extern int AOCreateDebugIS(MPI_Comm,IS,IS,AO*);

extern int AOCreateBasic(MPI_Comm,int,int*,int*,AO*);
extern int AOCreateBasicIS(MPI_Comm,IS,IS,AO*);

extern int AOPetscToApplication(AO,int,int*);
extern int AOApplicationToPetsc(AO,int,int*);
extern int AOPetscToApplicationIS(AO,IS);
extern int AOApplicationToPetscIS(AO,IS);

extern int AODestroy(AO);
extern int AOView(AO,Viewer);

#endif


