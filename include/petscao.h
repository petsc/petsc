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
extern PetscCookie PETSCDM_DLLEXPORT AO_COOKIE, AODATA_COOKIE;
extern PetscEvent  AO_PetscToApplication, AO_ApplicationToPetsc;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT DMInitializePackage(const char[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOCreateBasic(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOCreateBasicIS(IS,IS,AO*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOCreateMapping(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],AO*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOCreateMappingIS(IS,IS,AO*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOView(AO,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODestroy(AO);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AORegister_Private(const char [], const char [], const char [], PetscErrorCode (*)(AO));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define AORegister(a,b,c,d) AORegister_Private(a,b,c,0)
#else
#define AORegister(a,b,c,d) AORegister_Private(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOPetscToApplication(AO,PetscInt,PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOApplicationToPetsc(AO,PetscInt,PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOPetscToApplicationIS(AO,IS);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOApplicationToPetscIS(AO,IS);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOPetscToApplicationPermuteInt(AO, PetscInt, PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOApplicationToPetscPermuteInt(AO, PetscInt, PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOPetscToApplicationPermuteReal(AO, PetscInt, PetscReal[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOApplicationToPetscPermuteReal(AO, PetscInt, PetscReal[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOMappingHasApplicationIndex(AO, PetscInt, PetscTruth *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOMappingHasPetscIndex(AO, PetscInt, PetscTruth *);

/* ----------------------------------------------------*/

typedef enum {AODATA_BASIC=0,AODATA_ADVANCED=1} AODataType;

/*S
     AOData - Abstract PETSc object that manages complex parallel data structures intended to 
         hold grid information, etc

   Level: advanced

.seealso:  AODataCreateBasic()
S*/
typedef struct _p_AOData* AOData;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataCreateBasic(MPI_Comm,AOData *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataView(AOData,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataDestroy(AOData);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataLoadBasic(PetscViewer,AOData *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataGetInfo(AOData,PetscInt*,char ***);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyAdd(AOData,const char[],PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyRemove(AOData,const char[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeySetLocalToGlobalMapping(AOData,const char[],ISLocalToGlobalMapping);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetLocalToGlobalMapping(AOData,const char[],ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyRemap(AOData,const char[],AO);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyExists(AOData,const char[],PetscTruth*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetInfo(AOData,const char[],PetscInt *,PetscInt*,PetscInt*,char***);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetOwnershipRange(AOData,const char[],PetscInt *,PetscInt*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetNeighbors(AOData,const char[],PetscInt,PetscInt*,IS *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetNeighborsIS(AOData,const char[],IS,IS *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetAdjacency(AOData,const char[],Mat*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActive(AOData,const char[],const char[],PetscInt,PetscInt *,PetscInt,IS*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActiveIS(AOData,const char[],const char[],IS,PetscInt,IS*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActiveLocal(AOData,const char[],const char[],PetscInt,PetscInt *,PetscInt,IS*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActiveLocalIS(AOData,const char[],const char[],IS,PetscInt,IS*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataKeyPartition(AOData,const char[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentAdd(AOData,const char[],const char[],PetscInt,PetscInt,PetscInt *,void *,PetscDataType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRemove(AOData,const char[],const char[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentAddIS(AOData,const char[],const char[],PetscInt,IS,void *,PetscDataType);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentExists(AOData,const char[],const char[],PetscTruth*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetInfo(AOData,const char[],const char[],PetscInt *,PetscDataType*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGet(AOData,const char[],const char[],PetscInt,PetscInt*,void **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestore(AOData,const char[],const char[],PetscInt,PetscInt*,void **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetIS(AOData,const char[],const char[],IS,void **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestoreIS(AOData,const char[],const char[],IS,void **);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetLocal(AOData,const char[],const char[],PetscInt,PetscInt*,void **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestoreLocal(AOData,const char[],const char[],PetscInt,PetscInt*,void **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetLocalIS(AOData,const char[],const char[],IS,void **);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestoreLocalIS(AOData,const char[],const char[],IS,void **);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetReduced(AOData,const char[],const char[],PetscInt,PetscInt*,IS *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetReducedIS(AOData,const char[],const char[],IS,IS *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetExtrema(AOData,const char[],const char[],void *,void *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentPartition(AOData,const char[],const char[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataPartitionAndSetupLocal(AOData,const char[],const char[],IS*,IS*,ISLocalToGlobalMapping*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AODataAliasAdd(AOData,const char[],const char[]);

   
typedef struct _p_AOData2dGrid *AOData2dGrid;
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridAddNode(AOData2dGrid, PetscReal, PetscReal, PetscInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridInput(AOData2dGrid,PetscDraw);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridFlipCells(AOData2dGrid);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridComputeNeighbors(AOData2dGrid);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridComputeVertexBoundary(AOData2dGrid);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridDraw(AOData2dGrid,PetscDraw);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridDestroy(AOData2dGrid);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridCreate(AOData2dGrid*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT AOData2dGridToAOData(AOData2dGrid,AOData*);

PETSC_EXTERN_CXX_END
#endif
