/* $Id: petscvec.h,v 1.112 2000/09/02 02:50:55 bsmith Exp bsmith $ */
/* 
    Defines the vector component of PETSc. Vectors generally represent 
  degrees of freedom for finite element/finite difference functions
  on a grid. They have more mathematical structure then simple arrays.
*/

#ifndef __PETSCVEC_H 
#define __PETSCVEC_H
#include "petscis.h"
#include "petscsys.h"

#define VEC_COOKIE         PETSC_COOKIE+3
#define MAP_COOKIE         PETSC_COOKIE+22
#define VEC_SCATTER_COOKIE PETSC_COOKIE+4

typedef struct _p_Map*         Map;
typedef struct _p_Vec*         Vec;
typedef struct _p_VecScatter*  VecScatter;
#define VEC_SEQ    "seq"
#define VEC_MPI    "mpi"
#define VEC_FETI   "feti"
#define VEC_SHARED "shared"
typedef char*                  VecType;

EXTERN int VecCreateSeq(MPI_Comm,int,Vec*);
EXTERN int MapCreateMPI(MPI_Comm,int,int,Map*);  
EXTERN int VecCreateMPI(MPI_Comm,int,int,Vec*);  
EXTERN int VecCreateSeqWithArray(MPI_Comm,int,const Scalar[],Vec*);  
EXTERN int VecCreateMPIWithArray(MPI_Comm,int,int,const Scalar[],Vec*);  
EXTERN int VecCreateShared(MPI_Comm,int,int,Vec*);  
EXTERN int VecCreate(MPI_Comm,int,int,Vec*); 
EXTERN int VecSetType(Vec,VecType); 
EXTERN int VecSetFromOptions(Vec);

EXTERN int VecDestroy(Vec);        

EXTERN int MapDestroy(Map);
EXTERN int MapGetLocalSize(Map,int *);
EXTERN int MapGetSize(Map,int *);
EXTERN int MapGetLocalRange(Map,int *,int *);
EXTERN int MapGetGlobalRange(Map,int *[]);

EXTERN int VecDot(Vec,Vec,Scalar*);
EXTERN int VecTDot(Vec,Vec,Scalar*);  
EXTERN int VecMDot(int,Vec,const Vec[],Scalar*);
EXTERN int VecMTDot(int,Vec,const Vec[],Scalar*); 

typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,NORM_1_AND_2=5} NormType;
#define NORM_MAX NORM_INFINITY
EXTERN int VecNorm(Vec,NormType,PetscReal *);
EXTERN int VecSum(Vec,Scalar*);
EXTERN int VecMax(Vec,int*,PetscReal *);
EXTERN int VecMin(Vec,int*,PetscReal *);
EXTERN int VecScale(const Scalar*,Vec);    
EXTERN int VecCopy(Vec,Vec);        
EXTERN int VecSetRandom(PetscRandom,Vec);
EXTERN int VecSet(const Scalar*,Vec);
EXTERN int VecSwap(Vec,Vec);
EXTERN int VecAXPY(const Scalar*,Vec,Vec);  
EXTERN int VecAXPBY(const Scalar*,const Scalar *,Vec,Vec);  
EXTERN int VecMAXPY(int,const Scalar*,Vec,Vec*);
EXTERN int VecAYPX(const Scalar*,Vec,Vec);
EXTERN int VecWAXPY(const Scalar*,Vec,Vec,Vec);
EXTERN int VecPointwiseMult(Vec,Vec,Vec);    
EXTERN int VecPointwiseDivide(Vec,Vec,Vec);    
EXTERN int VecShift(const Scalar*,Vec);
EXTERN int VecReciprocal(Vec);
EXTERN int VecAbs(Vec);
EXTERN int VecDuplicate(Vec,Vec*);          
EXTERN int VecDuplicateVecs(Vec,int,Vec*[]);         
EXTERN int VecDestroyVecs(const Vec[],int); 
EXTERN int VecGetMap(Vec,Map*);

typedef enum {NOT_SET_VALUES,INSERT_VALUES,ADD_VALUES,MAX_VALUES} InsertMode;

EXTERN int VecStrideNorm(Vec,int,NormType,double*);
EXTERN int VecStrideGather(Vec,int,Vec,InsertMode);
EXTERN int VecStrideScatter(Vec,int,Vec,InsertMode);
EXTERN int VecStrideMax(Vec,int,int *,double *);
EXTERN int VecStrideMin(Vec,int,int *,double *);
EXTERN int VecStrideGatherAll(Vec,Vec*,InsertMode);
EXTERN int VecStrideScatterAll(Vec*,Vec,InsertMode);

EXTERN int VecSetValues(Vec,int,const int[],const Scalar[],InsertMode);
EXTERN int VecAssemblyBegin(Vec);
EXTERN int VecAssemblyEnd(Vec);
EXTERN int VecSetStashInitialSize(Vec,int,int);
EXTERN int VecStashView(Vec,PetscViewer);

#define VecSetValue(v,i,va,mode) \
{int _ierr,_row = i; Scalar _va = va; \
  _ierr = VecSetValues(v,1,&_row,&_va,mode);CHKERRQ(_ierr); \
}
EXTERN int VecSetBlockSize(Vec,int);
EXTERN int VecGetBlockSize(Vec,int*);
EXTERN int VecSetValuesBlocked(Vec,int,const int[],const Scalar[],InsertMode);

extern PetscTruth VecRegisterAllCalled;
EXTERN int        VecRegisterAll(const char []);
EXTERN int        VecRegister(const char[],const char[],const char[],int(*)(Vec));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define VecRegisterDynamic(a,b,c,d) VecRegister(a,b,c,0)
#else
#define VecRegisterDynamic(a,b,c,d) VecRegister(a,b,c,d)
#endif

typedef enum {SCATTER_FORWARD=0,SCATTER_REVERSE=1,SCATTER_FORWARD_LOCAL=2,
              SCATTER_REVERSE_LOCAL=3,SCATTER_LOCAL=2} ScatterMode;
EXTERN int VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
EXTERN int VecScatterPostRecvs(Vec,Vec,InsertMode,ScatterMode,VecScatter);
EXTERN int VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatter);
EXTERN int VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatter); 
EXTERN int VecScatterDestroy(VecScatter);
EXTERN int VecScatterCopy(VecScatter,VecScatter *);
EXTERN int VecScatterView(VecScatter,PetscViewer);
EXTERN int VecScatterRemap(VecScatter,int *,int*);

typedef enum {PIPELINE_DOWN=0,PIPELINE_UP=1} PipelineDirection;
typedef enum {PIPELINE_NONE=1,PIPELINE_SEQUENTIAL=2,
	      PIPELINE_REDBLACK=3,PIPELINE_MULTICOLOR=4} PipelineType;

typedef struct _p_VecPipeline*  VecPipeline;

EXTERN int VecPipelineCreate(MPI_Comm,Vec,IS,Vec,IS,VecPipeline *);
EXTERN int VecPipelineSetType(VecPipeline,PipelineType,PetscObject);
EXTERN int VecPipelineSetup(VecPipeline);
EXTERN int VecPipelineBegin(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline);
EXTERN int VecPipelineEnd(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline); 
EXTERN int VecPipelineView(VecPipeline,PetscViewer);
EXTERN int VecPipelineDestroy(VecPipeline);

EXTERN int VecGetArray(Vec,Scalar*[]);
EXTERN int VecRestoreArray(Vec,Scalar*[]);
EXTERN int VecGetArray4d(Vec,int,int,int,int,int,int,int,int,Scalar**[]);
EXTERN int VecRestoreArray4d(Vec,int,int,int,int,int,int,int,int,Scalar**[]);
EXTERN int VecGetArray3d(Vec,int,int,int,int,int,int,Scalar**[]);
EXTERN int VecRestoreArray3d(Vec,int,int,int,int,int,int,Scalar**[]);
EXTERN int VecGetArray2d(Vec,int,int,int,int,Scalar**[]);
EXTERN int VecRestoreArray2d(Vec,int,int,int,int,Scalar**[]);
EXTERN int VecGetArray1d(Vec,int,int,Scalar *[]);
EXTERN int VecRestoreArray1d(Vec,int,int,Scalar *[]);

EXTERN int VecPlaceArray(Vec,const Scalar[]);
EXTERN int VecResetArray(Vec);
EXTERN int VecReplaceArray(Vec,const Scalar[]);
EXTERN int VecGetArrays(const Vec[],int,Scalar**[]);
EXTERN int VecRestoreArrays(const Vec[],int,Scalar**[]);

EXTERN int VecValid(Vec,PetscTruth*);
EXTERN int VecView(Vec,PetscViewer);
EXTERN int VecEqual(Vec,Vec,PetscTruth*);
EXTERN int VecLoad(PetscViewer,Vec*);
EXTERN int VecLoadIntoVector(PetscViewer,Vec);

EXTERN int VecGetSize(Vec,int*);
EXTERN int VecGetType(Vec,VecType*);
EXTERN int VecGetLocalSize(Vec,int*);
EXTERN int VecGetOwnershipRange(Vec,int*,int*);

EXTERN int VecSetLocalToGlobalMapping(Vec,ISLocalToGlobalMapping);
EXTERN int VecSetValuesLocal(Vec,int,const int[],const Scalar[],InsertMode);
EXTERN int VecSetLocalToGlobalMappingBlocked(Vec,ISLocalToGlobalMapping);
EXTERN int VecSetValuesBlockedLocal(Vec,int,const int[],const Scalar[],InsertMode);

EXTERN int VecDotBegin(Vec,Vec,Scalar *);
EXTERN int VecDotEnd(Vec,Vec,Scalar *);
EXTERN int VecTDotBegin(Vec,Vec,Scalar *);
EXTERN int VecTDotEnd(Vec,Vec,Scalar *);
EXTERN int VecNormBegin(Vec,NormType,double *);
EXTERN int VecNormEnd(Vec,NormType,double *);

typedef enum {VEC_IGNORE_OFF_PROC_ENTRIES} VecOption;
EXTERN int VecSetOption(Vec,VecOption);

EXTERN int VecContourScale(Vec,double,double);

/*
    These numbers need to match the entries in 
  the function table in src/vec/vecimpl.h
*/
typedef enum { VECOP_VIEW = 33,
               VECOP_LOADINTOVECTOR = 40
             } VecOperation;
EXTERN int VecSetOperation(Vec,VecOperation,void*); /*  */

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
EXTERN int VecCreateGhost(MPI_Comm,int,int,int,const int[],Vec*);  
EXTERN int VecCreateGhostWithArray(MPI_Comm,int,int,int,const int[],const Scalar[],Vec*);  
EXTERN int VecCreateGhostBlock(MPI_Comm,int,int,int,int,const int[],Vec*);  
EXTERN int VecCreateGhostBlockWithArray(MPI_Comm,int,int,int,int,const int[],const Scalar[],Vec*);  
EXTERN int VecGhostGetLocalForm(Vec,Vec*);
EXTERN int VecGhostRestoreLocalForm(Vec,Vec*);
EXTERN int VecGhostUpdateBegin(Vec,InsertMode,ScatterMode);
EXTERN int VecGhostUpdateEnd(Vec,InsertMode,ScatterMode);

EXTERN int VecConjugate(Vec);

#endif



