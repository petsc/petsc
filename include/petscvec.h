/* $Id: vec.h,v 1.78 1998/09/25 03:16:46 bsmith Exp bsmith $ */
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
extern int VecCreateSeqWithArray(MPI_Comm,int,Scalar*,Vec*);  
extern int VecCreateMPIWithArray(MPI_Comm,int,int,Scalar*,Vec*);  
extern int VecCreateShared(MPI_Comm,int,int,Vec*);  
extern int VecCreate(MPI_Comm,int,int,Vec*); 
extern int VecCreateWithType(MPI_Comm,VecType,int,int,Vec*); 

extern int VecDestroy(Vec);        

extern int MapDestroy(Map);
extern int MapGetLocalSize(Map,int *);
extern int MapGetGlobalSize(Map,int *);
extern int MapGetLocalRange(Map,int *,int *);
extern int MapGetGlobalRange(Map,int **);

extern int VecDot(Vec,Vec,Scalar*);
extern int VecTDot(Vec,Vec,Scalar*);  
extern int VecMDot(int,Vec,Vec*,Scalar*);
extern int VecMTDot(int,Vec,Vec*,Scalar*); 

typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,NORM_1_AND_2=5} NormType;
#define NORM_MAX NORM_INFINITY
extern int VecNorm(Vec,NormType,double *);
extern int VecSum(Vec,Scalar*);
extern int VecMax(Vec,int*,double*);
extern int VecMin(Vec,int*,double*);
extern int VecScale(Scalar*,Vec);    
extern int VecCopy(Vec,Vec);        
extern int VecSetRandom(PetscRandom,Vec);
extern int VecSet(Scalar*,Vec);
extern int VecSwap(Vec,Vec);
extern int VecAXPY(Scalar*,Vec,Vec);  
extern int VecAXPBY(Scalar*,Scalar *,Vec,Vec);  
extern int VecMAXPY(int,Scalar*,Vec,Vec*);
extern int VecAYPX(Scalar*,Vec,Vec);
extern int VecWAXPY(Scalar*,Vec,Vec,Vec);
extern int VecPointwiseMult(Vec,Vec,Vec);    
extern int VecPointwiseDivide(Vec,Vec,Vec);    
extern int VecShift(Scalar*,Vec);
extern int VecReciprocal(Vec);
extern int VecAbs(Vec);
extern int VecDuplicate(Vec,Vec*);          
extern int VecDuplicateVecs(Vec,int,Vec**);         
extern int VecDestroyVecs(Vec*,int); 
extern int VecGetMap(Vec,Map*);

typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES} InsertMode;
extern int VecStrideNorm(Vec,int,NormType,double*);
extern int VecStrideGather(Vec,int,Vec,InsertMode);
extern int VecStrideScatter(Vec,int,Vec,InsertMode);

extern int VecSetValues(Vec,int,int*,Scalar*,InsertMode);
extern int VecAssemblyBegin(Vec);
extern int VecAssemblyEnd(Vec);
#define VecSetValue(v,i,va,mode) \
{int _ierr,_row = i; Scalar _va = va; \
  _ierr = VecSetValues(v,1,&_row,&_va,mode);CHKERRQ(_ierr); \
}
extern int VecSetBlockSize(Vec,int);
extern int VecSetValuesBlocked(Vec,int,int*,Scalar*,InsertMode);

extern int VecRegisterAllCalled;
extern int VecRegisterAll(char *);
extern int VecRegister_Private(char*,char*,char*,int(*)(MPI_Comm,int,int,Vec*));
#if defined(USE_DYNAMIC_LIBRARIES)
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
	      PIPELINE_REDBLACK=3, PIPELINE_MULTICOLOUR=4} PipelineType;

typedef struct _p_VecPipeline*  VecPipeline;

extern int VecPipelineCreate(MPI_Comm comm,Vec xin,IS ix,Vec yin,IS iy,VecPipeline *newctx);
extern int VecPipelineSetType(VecPipeline ctx,PipelineType typ,PetscObject x);
extern int VecPipelineSetup(VecPipeline ctx);
extern int VecPipelineBegin(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline);
extern int VecPipelineEnd(Vec,Vec,InsertMode,ScatterMode,PipelineDirection,VecPipeline); 
extern int VecPipelineView(VecPipeline pipe,Viewer viewer);
extern int VecPipelineDestroy(VecPipeline ctx);

extern int VecGetArray(Vec,Scalar**);
extern int VecRestoreArray(Vec,Scalar**);
extern int VecPlaceArray(Vec,Scalar*);
extern int VecGetArrays(Vec*,int,Scalar***);
extern int VecRestoreArrays(Vec*,int,Scalar***);

extern int VecValid(Vec,PetscTruth*);
extern int VecView(Vec,Viewer);
extern int VecEqual(Vec,Vec,PetscTruth*);
extern int VecLoad(Viewer,Vec*);

extern int VecGetSize(Vec,int*);
extern int VecGetType(Vec,VecType*);
extern int VecGetLocalSize(Vec,int*);
extern int VecGetOwnershipRange(Vec,int*,int*);

extern int VecSetLocalToGlobalMapping(Vec, ISLocalToGlobalMapping);
extern int VecSetValuesLocal(Vec,int,int*,Scalar*,InsertMode);
extern int VecSetLocalToGlobalMappingBlocked(Vec, ISLocalToGlobalMapping);
extern int VecSetValuesBlockedLocal(Vec,int,int*,Scalar*,InsertMode);

typedef enum {VEC_IGNORE_OFF_PROC_ENTRIES} VecOption;
extern int VecSetOption(Vec,VecOption);

extern int VecContourScale(Vec,double,double);

/*
     Routines for dealing with ghosted vectors:
  vectors with ghost elements at the end of the array.
*/
extern int VecCreateGhost(MPI_Comm,int,int,int,int*,Vec*);  
extern int VecCreateGhostWithArray(MPI_Comm,int,int,int,int*,Scalar*,Vec*);  
extern int VecGhostGetLocalForm(Vec,Vec*);
extern int VecGhostRestoreLocalForm(Vec,Vec*);
extern int VecGhostUpdateBegin(Vec,InsertMode,ScatterMode);
extern int VecGhostUpdateEnd(Vec,InsertMode,ScatterMode);


extern int DrawTensorContour(Draw,int,int,double *,double *,Vec);

#endif



