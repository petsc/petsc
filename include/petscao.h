/* $Id: ao.h,v 1.10 1997/10/20 16:59:05 bsmith Exp bsmith $ */

/* 
   An application ordering is mapping between application-centric
  ordering (the ordering that is "natural" to the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__AO_PACKAGE)
#define __AO_PACKAGE
#include "is.h"
#include "mat.h"

typedef enum {AO_BASIC=0, AO_ADVANCED=1} AOType;

#define AO_COOKIE PETSC_COOKIE+20

typedef struct _p_AO* AO;

extern int AOCreateBasic(MPI_Comm,int,int*,int*,AO*);
extern int AOCreateBasicIS(IS,IS,AO*);

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

extern int AODataKeyAdd(AOData,char*,int,int,int);
extern int AODataKeyAddLocalToGlobalMapping(AOData,char*,ISLocalToGlobalMapping);
extern int AODataKeyRemap(AOData,char *,AO);

extern int AODataSegmentAdd(AOData,char*,char *,int, int, int *,void *,PetscDataType);
extern int AODataSegmentAddIS(AOData,char*,char *,int, IS,void *,PetscDataType);

extern int AODataKeyGetInfo(AOData,char *,int *,int*,int *);
extern int AODataKeyGetInfoOwnership(AOData,char *,int *,int*);
extern int AODataSegmentGetInfo(AOData,char *,char *,int *,int *,int*,PetscDataType*);

extern int AODataSegmentGet(AOData,char *,char *,int,int*,void **);
extern int AODataSegmentRestore(AOData,char *,char *,int,int*,void **);
extern int AODataSegmentGetIS(AOData,char *,char *,IS,void **);
extern int AODataSegmentRestoreIS(AOData,char *,char *,IS,void **);

extern int AODataSegmentGetLocal(AOData,char *,char *,int,int*,void **);
extern int AODataSegmentRestoreLocal(AOData,char *,char *,int,int*,void **);
extern int AODataSegmentGetLocalIS(AOData,char *,char *,IS,void **);
extern int AODataSegmentRestoreLocalIS(AOData,char *,char *,IS,void **);

extern int AODataSegmentGetReduced(AOData,char *,char *,int,int*,IS *);
extern int AODataSegmentGetReducedIS(AOData,char *,char *,IS,IS *);

extern int AODataKeyGetNeighbors(AOData,char *,int,int*,IS *);
extern int AODataKeyGetNeighborsIS(AOData,char *,IS,IS *);
extern int AODataKeyGetAdjacency(AOData,char *,Mat*);

extern int AODataView(AOData,Viewer);
extern int AODataDestroy(AOData);

extern int AODataLoadBasic(Viewer,AOData *);

#endif


