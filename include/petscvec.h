/* $Id: vec.h,v 1.28 1995/08/21 18:14:36 bsmith Exp curfman $ */
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

typedef struct _Vec*            Vec;
typedef struct _VecScatterCtx*  VecScatterCtx;

extern int VecCreateSequential(MPI_Comm,int,Vec *);  
extern int VecCreateMPI(MPI_Comm,int,int,Vec *);  
extern int VecCreate(MPI_Comm,int,Vec *); 

extern int VecDot(Vec, Vec, Scalar*);
extern int VecTDot(Vec, Vec, Scalar*);  
extern int VecMDot(int,      Vec ,Vec*,Scalar*);
extern int VecMTDot(int,      Vec ,Vec*,Scalar*); 
extern int VecNorm(Vec, double*);
extern int VecASum(Vec, double*);
extern int VecSum(Vec,Scalar*);
extern int VecAMax(Vec, int *,   double*);
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
extern int VecGetVecs(Vec, int,Vec **);         
extern int VecFreeVecs(Vec*,int); 

typedef enum {NOTSETVALUES, INSERTVALUES, ADDVALUES} InsertMode;

extern int VecSetValues(Vec, int, int *,Scalar*,InsertMode);
extern int VecAssemblyBegin(Vec);
extern int VecAssemblyEnd(Vec);

typedef enum {SCATTERREVERSE=1,SCATTERDOWN=2,SCATTERUP=4,SCATTERALL=8,
              SCATTERALLREVERSE=9} ScatterMode;

extern int VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatterCtx);
extern int VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatterCtx); 
extern int VecScatterCtxCreate(Vec,IS,Vec,IS,VecScatterCtx *);
extern int VecScatterCtxDestroy(VecScatterCtx);
extern int VecScatterCtxCopy(VecScatterCtx,VecScatterCtx *);

typedef enum {PIPELINEDOWN=0,PIPELINEUP=1} PipelineMode;

extern int VecPipelineBegin(Vec,Vec,InsertMode,PipelineMode,VecScatterCtx);
extern int VecPipelineEnd(Vec,Vec,InsertMode,PipelineMode,VecScatterCtx); 

extern int VecShift(Scalar *,Vec);
extern int VecGetArray(Vec,Scalar**);
extern int VecRestoreArray(Vec,Scalar**);
extern int VecGetArrays(Vec*,int,Scalar***);
extern int VecRestoreArrays(Vec*,int,Scalar***);
extern int VecValidVector(Vec);
extern int VecView(Vec,Viewer);
extern int VecLoad(MPI_Comm,int,VecType,IS,Vec*);

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


