/* $Id: ao.h,v 1.11 1997/10/28 14:26:00 bsmith Exp bsmith $ */

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
extern int AODataKeySetLocalToGlobalMapping(AOData,char*,ISLocalToGlobalMapping);
extern int AODataKeyGetLocalToGlobalMapping(AOData,char*,ISLocalToGlobalMapping*);
extern int AODataKeyRemap(AOData,char *,AO);

extern int AODataKeyExists(AOData,char*,PetscTruth*);
extern int AODataKeyGetInfo(AOData,char *,int *,int*,int *);
extern int AODataKeyGetOwnershipRange(AOData,char *,int *,int*);

extern int AODataKeyGetNeighbors(AOData,char *,int,int*,IS *);
extern int AODataKeyGetNeighborsIS(AOData,char *,IS,IS *);
extern int AODataKeyGetAdjacency(AOData,char *,Mat*);

extern int AODataKeyPartition(AOData,char *);

extern int AODataSegmentAdd(AOData,char*,char *,int, int, int *,void *,PetscDataType);
extern int AODataSegmentAddIS(AOData,char*,char *,int, IS,void *,PetscDataType);

extern int AODataSegmentExists(AOData,char*,char*,PetscTruth*);
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
extern int AODataSegmentGetExtrema(AOData,char*,char*,void *,void *);

extern int AODataSegmentPartition(AOData,char *,char *);

extern int AODataView(AOData,Viewer);
extern int AODataDestroy(AOData);

extern int AODataLoadBasic(Viewer,AOData *);

#endif


