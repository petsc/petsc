/* $Id: vec.h,v 1.97 1999/06/30 22:52:56 bsmith Exp bsmith $ */
/* 
    Defines the vector component of PETSc. Vectors generally represent 
  degrees of freedom for finite element/finite difference functions
  on a grid. They have more mathematical structure then simple arrays.
*/

#ifndef __VEC_H 
#define __VEC_H
#include "is.h"
#include "sys.h"

#define VEC_COOKIE         PETSC_COOKIE+3
#define MAP_COOKIE         PETSC_COOKIE+22
#define VEC_SCATTER_COOKIE PETSC_COOKIE+4

typedef struct _p_Map*         Map;
typedef struct _p_Vec*         Vec;
typedef struct _p_VecScatter*  VecScatter;
#define VEC_SEQ    "seq"
#define VEC_MPI    "mpi"
#define VEC_SHARED "shared"
typedef char*                  VecType;

extern int VecCreateSeq(MPI_Comm,int,Vec*);
extern int MapCreateMPI(MPI_Comm,int,int,Map*);  
extern int VecCreateMPI(MPI_Comm,int,int,Vec*);  
extern int VecCreateSeqWithArray(MPI_Comm,int,const Scalar[],Vec*);  
extern int VecCreateMPIWithArray(MPI_Comm,int,int,const Scalar[],Vec*);  
extern int VecCreateShared(MPI_Comm,int,int,Vec*);  
extern int VecCreate(MPI_Comm,int,int,Vec*); 
extern int VecSetType(Vec,VecType); 
extern int VecSetFromOptions(Vec);

extern int VecDestroy(Vec);        

extern int MapDestroy(Map);
extern int MapGetLocalSize(Map,int *);
extern int MapGetSize(Map,int *);
extern int MapGetLocalRange(Map,int *,int *);
extern int MapGetGlobalRange(Map,int *[]);

extern int VecDot(Vec,Vec,Scalar*);
extern int VecTDot(Vec,Vec,Scalar*);  
extern int VecMDot(int,Vec,const Vec[],Scalar*);
extern int VecMTDot(int,Vec,const Vec[],Scalar*); 

typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,NORM_1_AND_2=5} NormType;
#define NORM_MAX NORM_INFINITY
extern int VecNorm(Vec,NormType,double *);
extern int VecSum(Vec,Scalar*);
extern int VecMax(Vec,int*,double*);
extern int VecMin(Vec,int*,double*);
extern int VecScale(const Scalar*,Vec);    
extern int VecCopy(Vec,Vec);        
extern int VecSetRandom(PetscRandom,Vec);
extern int VecSet(const Scalar*,Vec);
extern int VecSwap(Vec,Vec);
extern int VecAXPY(const Scalar*,Vec,Vec);  
extern int VecAXPBY(const Scalar*,const Scalar *,Vec,Vec);  
extern int VecMAXPY(int,const Scalar*,Vec,Vec*);
extern int VecAYPX(const Scalar*,Vec,Vec);
extern int VecWAXPY(const Scalar*,Vec,Vec,Vec);
extern int VecPointwiseMult(Vec,Vec,Vec);    
extern int VecPointwiseDivide(Vec,Vec,Vec);    
extern int VecShift(const Scalar*,Vec);
extern int VecReciprocal(Vec);
extern int VecAbs(Vec);
extern int VecDuplicate(Vec,Vec*);          
extern int VecDuplicateVecs(Vec,int,Vec*[]);         
extern int VecDestroyVecs(const Vec[],int); 
extern int VecGetMap(Vec,Map*);

extern int VecStrideNorm(Vec,int,NormType,double*);
extern int VecStrideGather(Vec,int,Vec,InsertMode);
extern int VecStrideScatter(Vec,int,Vec,InsertMode);
extern int VecStrideMax(Vec,int,int *,double *);
extern int VecStrideMin(Vec,int,int *,double *);

typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES} InsertMode;
extern int VecSetValues(Vec,int,const int[],const Scalar[],InsertMode);
extern int VecAssemblyBegin(Vec);
extern int VecAssemblyEnd(Vec);
extern int VecSetStashInitialSize(Vec,int, int);

#define VecSetValue(v,i,va,mode) \
{int _ierr,_row = i; Scalar _va = va; \
  _ierr = VecSetValues(v,1,&_row,&_va,mode);CHKERRQ(_ierr); \
}
extern int VecSetBlockSize(Vec,int);
extern int VecGetBlockSize(Vec,int*);
extern int VecSetValuesBlocked(Vec,int,const int[],const Scalar[],InsertMode);

extern int VecRegisterAllCalled;
extern int VecRegisterAll(const char []);
extern int VecRegister_Private(const char[],const char[],const char[],int(*)(Vec));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define VecRegister(a,b,c,d) VecRegister_Private(a,b,c,0)
#else
#define VecRegister(a,b,c,d) VecRegister_Private(a,b,c,d)
#endif

typedef enum {SCATTER_FORWARD=0,SCATTER_REVERSE=1,SCATTER_FORWARD_LOCAL=2,
              SCATTER_REVERSE_LOCAL=3,SCATTER_LOCAL=2} ScatterMode;
extern int VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
extern int VecScatterPostRecvs(Vec,Vec,InsertMode,ScatterMode,VecScatter);
extern int VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatter);
extern int VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatter); 
extern int VecScatterDestroy(VecScatter);
extern int VecScatterCopy(VecScatter,VecScatter *);
extern int VecScatterView(VecScatter,Viewer);
extern int VecScatterRemap(VecScatter,int *,int*);

typedef enum {PIPELINE_DOWN=0,PIPELINE_UP=1} PipelineDirection;
typedef enum {PIPELINE_NONE=1, PIPELINE_SEQUENTIAL=2,
	      PIPELINE_REDBLACK=3, PIPELINE_MULTICOLOR=4} PipelineType;

typedef struct _p_VecPipeline*  VecPipeline;

extern int VecPipelineCreate(MPI_Comm,Vec,IS,Vec,IS,VecPipeline *);
extern int VecPipelineSetType(VecPipeline,PipelineType,PetscObject);
extern int VecPipelineSetup(VecPipeline);
extern int VecPipelineBegin(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline);
extern int VecPipelineEnd(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline); 
extern int VecPipelineView(VecPipeline,Viewer);
extern int VecPipelineDestroy(VecPipeline);

extern int VecGetArray(Vec,Scalar*[]);
extern int VecRestoreArray(Vec,Scalar*[]);
extern int VecPlaceArray(Vec,const Scalar[]);
extern int VecReplaceArray(Vec,const Scalar[]);
extern int VecGetArrays(const Vec[],int,Scalar**[]);
extern int VecRestoreArrays(const Vec[],int,Scalar**[]);

extern int VecValid(Vec,PetscTruth*);
extern int VecView(Vec,Viewer);
extern int VecEqual(Vec,Vec,PetscTruth*);
extern int VecLoad(Viewer,Vec*);
extern int VecLoadIntoVector(Viewer,Vec);

extern int VecGetSize(Vec,int*);
extern int VecGetType(Vec,VecType*);
extern int VecGetLocalSize(Vec,int*);
extern int VecGetOwnershipRange(Vec,int*,int*);

extern int VecSetLocalToGlobalMapping(Vec, ISLocalToGlobalMapping);
extern int VecSetValuesLocal(Vec,int,const int[],const Scalar[],InsertMode);
extern int VecSetLocalToGlobalMappingBlocked(Vec, ISLocalToGlobalMapping);
extern int VecSetValuesBlockedLocal(Vec,int,const int[],const Scalar[],InsertMode);

extern int VecDotBegin(Vec,Vec,Scalar *);
extern int VecDotEnd(Vec,Vec,Scalar *);
extern int VecTDotBegin(Vec,Vec,Scalar *);
extern int VecTDotEnd(Vec,Vec,Scalar *);
extern int VecNormBegin(Vec,NormType,double *);
extern int VecNormEnd(Vec,NormType,double *);

typedef enum {VEC_IGNORE_OFF_PROC_ENTRIES} VecOption;
extern int VecSetOption(Vec,VecOption);

extern int VecContourScale(Vec,double,double);

/*
    These numbers need to match the entries in 
  the function table in src/vec/vecimpl.h
*/
typedef enum { VECOP_VIEW = 32,
               VECOP_LOADINTOVECTOR = 40
             } VecOperation;
extern int VecSetOperation(Vec,VecOperation,void*);

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
extern int VecCreateGhost(MPI_Comm,int,int,int,const int[],Vec*);  
extern int VecCreateGhostWithArray(MPI_Comm,int,int,int,const int[],const Scalar[],Vec*);  
extern int VecCreateGhostBlock(MPI_Comm,int,int,int,int,const int[],Vec*);  
extern int VecCreateGhostBlockWithArray(MPI_Comm,int,int,int,int,const int[],const Scalar[],Vec*);  
extern int VecGhostGetLocalForm(Vec,Vec*);
extern int VecGhostRestoreLocalForm(Vec,Vec*);
extern int VecGhostUpdateBegin(Vec,InsertMode,ScatterMode);
extern int VecGhostUpdateEnd(Vec,InsertMode,ScatterMode);



#endif



