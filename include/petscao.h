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
extern PetscCookie AO_COOKIE, AODATA_COOKIE;
extern PetscEvent  AO_PetscToApplication, AO_ApplicationToPetsc;

EXTERN PetscErrorCode DMInitializePackage(const char[]);

EXTERN PetscErrorCode AOCreateBasic(MPI_Comm,int,const int[],const int[],AO*);
EXTERN PetscErrorCode AOCreateBasicIS(IS,IS,AO*);

EXTERN PetscErrorCode AOCreateMapping(MPI_Comm,int,const int[],const int[],AO*);
EXTERN PetscErrorCode AOCreateMappingIS(IS,IS,AO*);

EXTERN PetscErrorCode AOView(AO,PetscViewer);
EXTERN PetscErrorCode AODestroy(AO);

EXTERN PetscErrorCode AORegister_Private(const char [], const char [], const char [], PetscErrorCode (*)(AO));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define AORegister(a,b,c,d) AORegister_Private(a,b,c,0)
#else
#define AORegister(a,b,c,d) AORegister_Private(a,b,c,d)
#endif

EXTERN PetscErrorCode AOPetscToApplication(AO,int,int[]);
EXTERN PetscErrorCode AOApplicationToPetsc(AO,int,int[]);
EXTERN PetscErrorCode AOPetscToApplicationIS(AO,IS);
EXTERN PetscErrorCode AOApplicationToPetscIS(AO,IS);

EXTERN PetscErrorCode AOPetscToApplicationPermuteInt(AO, int, int[]);
EXTERN PetscErrorCode AOApplicationToPetscPermuteInt(AO, int, int[]);
EXTERN PetscErrorCode AOPetscToApplicationPermuteReal(AO, int, double[]);
EXTERN PetscErrorCode AOApplicationToPetscPermuteReal(AO, int, double[]);

EXTERN PetscErrorCode AOMappingHasApplicationIndex(AO, int, PetscTruth *);
EXTERN PetscErrorCode AOMappingHasPetscIndex(AO, int, PetscTruth *);

/* ----------------------------------------------------*/

typedef enum {AODATA_BASIC=0,AODATA_ADVANCED=1} AODataType;

/*S
     AOData - Abstract PETSc object that manages complex parallel data structures intended to 
         hold grid information, etc

   Level: advanced

.seealso:  AODataCreateBasic()
S*/
typedef struct _p_AOData* AOData;

EXTERN PetscErrorCode AODataCreateBasic(MPI_Comm,AOData *);
EXTERN PetscErrorCode AODataView(AOData,PetscViewer);
EXTERN PetscErrorCode AODataDestroy(AOData);
EXTERN PetscErrorCode AODataLoadBasic(PetscViewer,AOData *);
EXTERN PetscErrorCode AODataGetInfo(AOData,int*,char ***);

EXTERN PetscErrorCode AODataKeyAdd(AOData,const char[],int,int);
EXTERN PetscErrorCode AODataKeyRemove(AOData,const char[]);

EXTERN PetscErrorCode AODataKeySetLocalToGlobalMapping(AOData,const char[],ISLocalToGlobalMapping);
EXTERN PetscErrorCode AODataKeyGetLocalToGlobalMapping(AOData,const char[],ISLocalToGlobalMapping*);
EXTERN PetscErrorCode AODataKeyRemap(AOData,const char[],AO);

EXTERN PetscErrorCode AODataKeyExists(AOData,const char[],PetscTruth*);
EXTERN PetscErrorCode AODataKeyGetInfo(AOData,const char[],int *,int*,int*,char***);
EXTERN PetscErrorCode AODataKeyGetOwnershipRange(AOData,const char[],int *,int*);

EXTERN PetscErrorCode AODataKeyGetNeighbors(AOData,const char[],int,int*,IS *);
EXTERN PetscErrorCode AODataKeyGetNeighborsIS(AOData,const char[],IS,IS *);
EXTERN PetscErrorCode AODataKeyGetAdjacency(AOData,const char[],Mat*);

EXTERN PetscErrorCode AODataKeyGetActive(AOData,const char[],const char[],int,int *,int,IS*);
EXTERN PetscErrorCode AODataKeyGetActiveIS(AOData,const char[],const char[],IS,int,IS*);
EXTERN PetscErrorCode AODataKeyGetActiveLocal(AOData,const char[],const char[],int,int *,int,IS*);
EXTERN PetscErrorCode AODataKeyGetActiveLocalIS(AOData,const char[],const char[],IS,int,IS*);

EXTERN PetscErrorCode AODataKeyPartition(AOData,const char[]);

EXTERN PetscErrorCode AODataSegmentAdd(AOData,const char[],const char[],int,int,int *,void *,PetscDataType);
EXTERN PetscErrorCode AODataSegmentRemove(AOData,const char[],const char[]);
EXTERN PetscErrorCode AODataSegmentAddIS(AOData,const char[],const char[],int,IS,void *,PetscDataType);

EXTERN PetscErrorCode AODataSegmentExists(AOData,const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode AODataSegmentGetInfo(AOData,const char[],const char[],int *,PetscDataType*);

EXTERN PetscErrorCode AODataSegmentGet(AOData,const char[],const char[],int,int*,void **);
EXTERN PetscErrorCode AODataSegmentRestore(AOData,const char[],const char[],int,int*,void **);
EXTERN PetscErrorCode AODataSegmentGetIS(AOData,const char[],const char[],IS,void **);
EXTERN PetscErrorCode AODataSegmentRestoreIS(AOData,const char[],const char[],IS,void **);

EXTERN PetscErrorCode AODataSegmentGetLocal(AOData,const char[],const char[],int,int*,void **);
EXTERN PetscErrorCode AODataSegmentRestoreLocal(AOData,const char[],const char[],int,int*,void **);
EXTERN PetscErrorCode AODataSegmentGetLocalIS(AOData,const char[],const char[],IS,void **);
EXTERN PetscErrorCode AODataSegmentRestoreLocalIS(AOData,const char[],const char[],IS,void **);

EXTERN PetscErrorCode AODataSegmentGetReduced(AOData,const char[],const char[],int,int*,IS *);
EXTERN PetscErrorCode AODataSegmentGetReducedIS(AOData,const char[],const char[],IS,IS *);
EXTERN PetscErrorCode AODataSegmentGetExtrema(AOData,const char[],const char[],void *,void *);

EXTERN PetscErrorCode AODataSegmentPartition(AOData,const char[],const char[]);

EXTERN PetscErrorCode AODataPartitionAndSetupLocal(AOData,const char[],const char[],IS*,IS*,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode AODataAliasAdd(AOData,const char[],const char[]);

   
typedef struct _p_AOData2dGrid *AOData2dGrid;
EXTERN PetscErrorCode AOData2dGridAddNode(AOData2dGrid, PetscReal, PetscReal, int *);
EXTERN PetscErrorCode AOData2dGridInput(AOData2dGrid,PetscDraw);
EXTERN PetscErrorCode AOData2dGridFlipCells(AOData2dGrid);
EXTERN PetscErrorCode AOData2dGridComputeNeighbors(AOData2dGrid);
EXTERN PetscErrorCode AOData2dGridComputeVertexBoundary(AOData2dGrid);
EXTERN PetscErrorCode AOData2dGridDraw(AOData2dGrid,PetscDraw);
EXTERN PetscErrorCode AOData2dGridDestroy(AOData2dGrid);
EXTERN PetscErrorCode AOData2dGridCreate(AOData2dGrid*);
EXTERN PetscErrorCode AOData2dGridToAOData(AOData2dGrid,AOData*);

PETSC_EXTERN_CXX_END
#endif
