/* $Id: vec.h,v 1.37 1995/11/01 19:12:32 bsmith Exp bsmith $ */
/* 
   This defines the abstract vector component. These are patterned
   after the Level-1 Blas, but with some additions that have proved
   useful. These include routines to allocate and free vectors.

   Note that the routines that are normally thought of as returning a
   value (e.g., dot, norm) return their value through an argument.
   This allows these routines to be used with other datatype, such
   as float and dcomplex.

   All vectors should be declared as a Vec. All vector routines begin
   with Vec.
   

 */

#ifndef __VEC_PACKAGE 
#define __VEC_PACKAGE
#include "is.h"

#define VEC_COOKIE         PETSC_COOKIE+3
#define VEC_SCATTER_COOKIE PETSC_COOKIE+4

typedef enum { VECSAME=-1, VECSEQ, VECMPI } VecType;

typedef struct _Vec*         Vec;
typedef struct _VecScatter*  VecScatter;

extern int VecCreateSeq(MPI_Comm,int,Vec *);  
extern int VecCreateMPI(MPI_Comm,int,int,Vec *);  
extern int VecCreate(MPI_Comm,int,Vec *); 

extern int VecDot(Vec, Vec, Scalar*);
extern int VecTDot(Vec, Vec, Scalar*);  
extern int VecMDot(int,      Vec ,Vec*,Scalar*);
extern int VecMTDot(int,      Vec ,Vec*,Scalar*); 

typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4} NormType;
#define NORM_MAX NORM_INFINITY
extern int VecNorm(Vec,NormType,double *);
extern int VecSum(Vec,Scalar*);
extern int VecMax(Vec, int *,    double*);
extern int VecMin(Vec, int *,    double*);
extern int VecScale(Scalar*, Vec);    
extern int VecCopy(Vec, Vec);        
extern int VecSet(Scalar*, Vec);    
extern int VecSwap(Vec, Vec);
extern int VecAXPY(Scalar*, Vec, Vec);  
extern int VecMAXPY(int,      Scalar*, Vec ,Vec*);
extern int VecAYPX(Scalar*, Vec, Vec);
extern int VecWAXPY(Scalar*, Vec, Vec, Vec);
extern int VecPMult(Vec, Vec, Vec);    
extern int VecPDiv(Vec, Vec, Vec);    
extern int VecDuplicate(Vec,Vec *);          
extern int VecDestroy(Vec);        
extern int VecDuplicateVecs(Vec, int,Vec **);         
extern int VecFreeVecs(Vec*,int); 

typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES} InsertMode;

extern int VecSetValues(Vec, int, int *,Scalar*,InsertMode);
extern int VecAssemblyBegin(Vec);
extern int VecAssemblyEnd(Vec);

typedef enum {SCATTER_REVERSE=1,SCATTER_DOWN=2,SCATTER_UP=4,SCATTER_ALL=8,
              SCATTER_ALL_REVERSE=9} ScatterMode;

extern int VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatter);
extern int VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatter); 
extern int VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
extern int VecScatterDestroy(VecScatter);
extern int VecScatterCopy(VecScatter,VecScatter *);
extern int VecScatterView(VecScatter,Viewer);

typedef enum {PIPELINE_DOWN=0,PIPELINE_UP=1} PipelineMode;

extern int VecPipelineBegin(Vec,Vec,InsertMode,PipelineMode,VecScatter);
extern int VecPipelineEnd(Vec,Vec,InsertMode,PipelineMode,VecScatter); 

extern int VecShift(Scalar *,Vec);
extern int VecGetArray(Vec,Scalar**);
extern int VecRestoreArray(Vec,Scalar**);
extern int VecPlaceArray(Vec,Scalar*);
extern int VecGetArrays(Vec*,int,Scalar***);
extern int VecRestoreArrays(Vec*,int,Scalar***);
extern int VecValidVector(Vec);
extern int VecView(Vec,Viewer);
extern int VecLoad(Viewer,Vec*);

extern int VecGetSize(Vec,int *);
extern int VecGetLocalSize(Vec,int *);
extern int VecGetOwnershipRange(Vec,int*,int*);


/* utility routines */
extern int VecReciprocal(Vec);
extern int VecAbs(Vec);

#if defined(__DRAW_PACKAGE)
extern int DrawTensorContour(DrawCtx,int,int,double *,double *,Vec);
#endif

#endif


