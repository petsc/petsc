/* $Id: petscao.h,v 1.27 2001/08/07 21:31:33 bsmith Exp $ */

/* 
  An application ordering is mapping between an application-centric
  ordering (the ordering that is "natural" for the application) and 
  the parallel ordering that PETSc uses.
*/
#if !defined(__PETSCAO_H)
#define __PETSCAO_H
#include "petscis.h"
#include "petscmat.h"
PETSC_EXTERN_CXX_BEGIN

typedef enum {AO_BASIC=0, AO_ADVANCED, AO_MAPPING, AO_NEW} AOType;

/*S
     AO - Abstract PETSc object that manages mapping between different global numbering

   Level: intermediate

  Concepts: global numbering

.seealso:  AOCreateBasic(), AOCreateBasicIS(), AOPetscToApplication(), AOView()
S*/
typedef struct _p_AO* AO;

/* Logging support */
extern int AO_COOKIE;
extern int AODATA_COOKIE;
enum {AO_PetscToApplication, AO_ApplicationToPetsc, AO_MAX_EVENTS};
extern int AOEvents[AO_MAX_EVENTS];
#define AOLogEventBegin(e,o1,o2,o3,o4) PetscLogEventBegin(AOEvents[e],o1,o2,o3,o4)
#define AOLogEventEnd(e,o1,o2,o3,o4)   PetscLogEventEnd(AOEvents[e],o1,o2,o3,o4)

EXTERN int DMInitializePackage(const char[]);

EXTERN int AOCreateBasic(MPI_Comm,int,const int[],const int[],AO*);
EXTERN int AOCreateBasicIS(IS,IS,AO*);

EXTERN int AOCreateMapping(MPI_Comm,int,const int[],const int[],AO*);
EXTERN int AOCreateMappingIS(IS,IS,AO*);

EXTERN int AOView(AO,PetscViewer);
EXTERN int AODestroy(AO);

EXTERN int AORegister_Private(const char [], const char [], const char [], int (*)(AO));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define AORegister(a,b,c,d) AORegister_Private(a,b,c,0)
#else
#define AORegister(a,b,c,d) AORegister_Private(a,b,c,d)
#endif

EXTERN int AOPetscToApplication(AO,int,int[]);
EXTERN int AOApplicationToPetsc(AO,int,int[]);
EXTERN int AOPetscToApplicationIS(AO,IS);
EXTERN int AOApplicationToPetscIS(AO,IS);

EXTERN int AOPetscToApplicationPermuteInt(AO, int, int[]);
EXTERN int AOApplicationToPetscPermuteInt(AO, int, int[]);
EXTERN int AOPetscToApplicationPermuteReal(AO, int, double[]);
EXTERN int AOApplicationToPetscPermuteReal(AO, int, double[]);

EXTERN int AOMappingHasApplicationIndex(AO, int, PetscTruth *);
EXTERN int AOMappingHasPetscIndex(AO, int, PetscTruth *);

/* ----------------------------------------------------*/

typedef enum {AODATA_BASIC=0,AODATA_ADVANCED=1} AODataType;

/*S
     AOData - Abstract PETSc object that manages complex parallel data structures intended to 
         hold grid information, etc

   Level: advanced

.seealso:  AODataCreateBasic()
S*/
typedef struct _p_AOData* AOData;

EXTERN int AODataCreateBasic(MPI_Comm,AOData *);
EXTERN int AODataView(AOData,PetscViewer);
EXTERN int AODataDestroy(AOData);
EXTERN int AODataLoadBasic(PetscViewer,AOData *);
EXTERN int AODataGetInfo(AOData,int*,char ***);

EXTERN int AODataKeyAdd(AOData,const char[],int,int);
EXTERN int AODataKeyRemove(AOData,const char[]);

EXTERN int AODataKeySetLocalToGlobalMapping(AOData,const char[],ISLocalToGlobalMapping);
EXTERN int AODataKeyGetLocalToGlobalMapping(AOData,const char[],ISLocalToGlobalMapping*);
EXTERN int AODataKeyRemap(AOData,const char[],AO);

EXTERN int AODataKeyExists(AOData,const char[],PetscTruth*);
EXTERN int AODataKeyGetInfo(AOData,const char[],int *,int*,int*,char***);
EXTERN int AODataKeyGetOwnershipRange(AOData,const char[],int *,int*);

EXTERN int AODataKeyGetNeighbors(AOData,const char[],int,int*,IS *);
EXTERN int AODataKeyGetNeighborsIS(AOData,const char[],IS,IS *);
EXTERN int AODataKeyGetAdjacency(AOData,const char[],Mat*);

EXTERN int AODataKeyGetActive(AOData,const char[],const char[],int,int *,int,IS*);
EXTERN int AODataKeyGetActiveIS(AOData,const char[],const char[],IS,int,IS*);
EXTERN int AODataKeyGetActiveLocal(AOData,const char[],const char[],int,int *,int,IS*);
EXTERN int AODataKeyGetActiveLocalIS(AOData,const char[],const char[],IS,int,IS*);

EXTERN int AODataKeyPartition(AOData,const char[]);

EXTERN int AODataSegmentAdd(AOData,const char[],const char[],int,int,int *,void *,PetscDataType);
EXTERN int AODataSegmentRemove(AOData,const char[],const char[]);
EXTERN int AODataSegmentAddIS(AOData,const char[],const char[],int,IS,void *,PetscDataType);

EXTERN int AODataSegmentExists(AOData,const char[],const char[],PetscTruth*);
EXTERN int AODataSegmentGetInfo(AOData,const char[],const char[],int *,PetscDataType*);

EXTERN int AODataSegmentGet(AOData,const char[],const char[],int,int*,void **);
EXTERN int AODataSegmentRestore(AOData,const char[],const char[],int,int*,void **);
EXTERN int AODataSegmentGetIS(AOData,const char[],const char[],IS,void **);
EXTERN int AODataSegmentRestoreIS(AOData,const char[],const char[],IS,void **);

EXTERN int AODataSegmentGetLocal(AOData,const char[],const char[],int,int*,void **);
EXTERN int AODataSegmentRestoreLocal(AOData,const char[],const char[],int,int*,void **);
EXTERN int AODataSegmentGetLocalIS(AOData,const char[],const char[],IS,void **);
EXTERN int AODataSegmentRestoreLocalIS(AOData,const char[],const char[],IS,void **);

EXTERN int AODataSegmentGetReduced(AOData,const char[],const char[],int,int*,IS *);
EXTERN int AODataSegmentGetReducedIS(AOData,const char[],const char[],IS,IS *);
EXTERN int AODataSegmentGetExtrema(AOData,const char[],const char[],void *,void *);

EXTERN int AODataSegmentPartition(AOData,const char[],const char[]);

EXTERN int AODataPartitionAndSetupLocal(AOData,const char[],const char[],IS*,IS*,ISLocalToGlobalMapping*);
EXTERN int AODataAliasAdd(AOData,const char[],const char[]);

   
typedef struct _p_AOData2dGrid *AOData2dGrid;
EXTERN int AOData2dGridAddNode(AOData2dGrid, PetscReal, PetscReal, int *);
EXTERN int AOData2dGridInput(AOData2dGrid,PetscDraw);
EXTERN int AOData2dGridFlipCells(AOData2dGrid);
EXTERN int AOData2dGridComputeNeighbors(AOData2dGrid);
EXTERN int AOData2dGridComputeVertexBoundary(AOData2dGrid);
EXTERN int AOData2dGridDraw(AOData2dGrid,PetscDraw);
EXTERN int AOData2dGridDestroy(AOData2dGrid);
EXTERN int AOData2dGridCreate(AOData2dGrid*);
EXTERN int AOData2dGridToAOData(AOData2dGrid,AOData*);

PETSC_EXTERN_CXX_END
#endif
