/* $Id: ao.h,v 1.6 1997/10/01 04:09:15 bsmith Exp bsmith $ */

/* 
   An application ordering is mapping between application-centric
  ordering (the ordering that is "natural" to the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__AO_PACKAGE)
#define __AO_PACKAGE
#include "is.h"

typedef enum {AO_BASIC=0, AO_ADVANCED=1} AOType;

#define AO_COOKIE PETSC_COOKIE+20

typedef struct _p_AO* AO;

extern int AOCreateBasic(MPI_Comm,int,int*,int*,AO*);
extern int AOCreateBasicIS(MPI_Comm,IS,IS,AO*);

extern int AOPetscToApplication(AO,int,int*);
extern int AOApplicationToPetsc(AO,int,int*);
extern int AOPetscToApplicationIS(AO,IS);
extern int AOApplicationToPetscIS(AO,IS);

extern int AODestroy(AO);
extern int AOView(AO,Viewer);

/* ----------------------------------------------------*/

typedef enum {AODATA_BASIC=0, AODATA_ADVANCED=1} AODataType;

#define AODATA_COOKIE PETSC_COOKIE+24

typedef struct _p_AOData* AOData;

extern int AODataCreateBasic(MPI_Comm,int,AOData *);
extern int AODataAdd(AOData,char*,int, int, int *,void *,PetscDataType);
extern int AODataAddIS(AOData,char*,int, IS,void *,PetscDataType);
extern int AODataGetInfo(AOData,char *,int *,int*,PetscDataType*);
extern int AODataGet(AOData,char *,int,int*,void **);
extern int AODataRestore(AOData,char *,int,int*,void **);
extern int AODataGetIS(AOData,char *,IS,void **);
extern int AODataRestoreIS(AOData,char *,IS,void **);
extern int AODataView(AOData,Viewer);
extern int AODataDestroy(AOData);

extern int AODataLoadBasic(Viewer,AOData *);

#endif


