/* $Id: ao.h,v 1.7 1997/10/01 22:47:58 bsmith Exp bsmith $ */

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

extern int AODataAddKey(AOData,char*,int,int,int);

extern int AODataAddSegment(AOData,char*,char *,int, int, int *,void *,PetscDataType);
extern int AODataAddSegmentIS(AOData,char*,char *,int, IS,void *,PetscDataType);

extern int AODataGetInfoKey(AOData,char *,int *,int*,int *);
extern int AODataGetInfoKeyOwnership(AOData,char *,int *,int*);
extern int AODataGetInfoSegment(AOData,char *,char *,int *,int *,int*,PetscDataType*);

extern int AODataGetSegment(AOData,char *,char *,int,int*,void **);
extern int AODataRestoreSegment(AOData,char *,char *,int,int*,void **);
extern int AODataGetSegmentIS(AOData,char *,char *,IS,void **);
extern int AODataRestoreSegmentIS(AOData,char *,char *,IS,void **);

extern int AODataView(AOData,Viewer);
extern int AODataDestroy(AOData);

extern int AODataLoadBasic(Viewer,AOData *);

#endif


