/* $Id: ao.h,v 1.21 2000/01/19 21:42:33 bsmith Exp bsmith $ */

/* 
  An application ordering is mapping between an application-centric
  ordering (the ordering that is "natural" for the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__AO_H)
#define __AO_H
#include "is.h"
#include "mat.h"

typedef enum {AO_BASIC=0,AO_ADVANCED=1} AOType;

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

typedef enum {AODATA_BASIC=0,AODATA_ADVANCED=1} AODataType;

#define AODATA_COOKIE PETSC_COOKIE+24

typedef struct _p_AOData* AOData;

extern int AODataCreateBasic(MPI_Comm,AOData *);
extern int AODataView(AOData,Viewer);
extern int AODataDestroy(AOData);
extern int AODataLoadBasic(Viewer,AOData *);
extern int AODataGetInfo(AOData,int*,char ***);

extern int AODataKeyAdd(AOData,char*,int,int);
extern int AODataKeyRemove(AOData,char*);

extern int AODataKeySetLocalToGlobalMapping(AOData,char*,ISLocalToGlobalMapping);
extern int AODataKeyGetLocalToGlobalMapping(AOData,char*,ISLocalToGlobalMapping*);
extern int AODataKeyRemap(AOData,char *,AO);

extern int AODataKeyExists(AOData,char*,PetscTruth*);
extern int AODataKeyGetInfo(AOData,char *,int *,int*,int*,char***);
extern int AODataKeyGetOwnershipRange(AOData,char *,int *,int*);

extern int AODataKeyGetNeighbors(AOData,char *,int,int*,IS *);
extern int AODataKeyGetNeighborsIS(AOData,char *,IS,IS *);
extern int AODataKeyGetAdjacency(AOData,char *,Mat*);

extern int AODataKeyGetActive(AOData,char*,char*,int,int *,int,IS*);
extern int AODataKeyGetActiveIS(AOData,char*,char*,IS,int,IS*);
extern int AODataKeyGetActiveLocal(AOData,char*,char*,int,int *,int,IS*);
extern int AODataKeyGetActiveLocalIS(AOData,char*,char*,IS,int,IS*);

extern int AODataKeyPartition(AOData,char *);

extern int AODataSegmentAdd(AOData,char*,char *,int,int,int *,void *,PetscDataType);
extern int AODataSegmentRemove(AOData,char *,char *);
extern int AODataSegmentAddIS(AOData,char*,char *,int,IS,void *,PetscDataType);

extern int AODataSegmentExists(AOData,char*,char*,PetscTruth*);
extern int AODataSegmentGetInfo(AOData,char *,char *,int *,PetscDataType*);

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

extern int AODataPartitionAndSetupLocal(AOData,char*,char*,IS*,IS*,ISLocalToGlobalMapping*);
extern int AODataAliasAdd(AOData,char *,char *);

   
typedef struct _p_AOData2dGrid *AOData2dGrid;
extern int AOData2dGridAddNode(AOData2dGrid, double, double, int *);
extern int AOData2dGridInput(AOData2dGrid,Draw);
extern int AOData2dGridFlipCells(AOData2dGrid);
extern int AOData2dGridComputeNeighbors(AOData2dGrid);
extern int AOData2dGridComputeVertexBoundary(AOData2dGrid);
extern int AOData2dGridDraw(AOData2dGrid,Draw);
extern int AOData2dGridDestroy(AOData2dGrid);
extern int AOData2dGridCreate(AOData2dGrid*);
extern int AOData2dGridToAOData(AOData2dGrid,AOData*);

#endif


