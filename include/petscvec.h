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


typedef struct _Vec*            Vec;
typedef struct _VecScatterCtx*  VecScatterCtx;

extern int VecCreateSequential(int,Vec *);  
extern int VecCreateSequentialBLAS(int,Vec *); 

extern int VecCreateMPI(MPI_Comm,int,int,Vec *);  
extern int VecCreateMPIBLAS(MPI_Comm,int,int,Vec *); 

extern int VecCreateInitialVector(int,Vec *); 

extern int VecDot(Vec, Vec, Scalar*);
extern int VecTDot(Vec, Vec, Scalar*);  
extern int VecMDot(int,      Vec ,Vec*,Scalar*);
extern int VecMTDot(int,      Vec ,Vec*,Scalar*); 
extern int VecNorm(Vec, double*);
extern int VecASum(Vec, double*);
extern int VecMax(Vec, int *,    Scalar*);
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
extern int VecCreate(Vec,Vec *);          
extern int VecDestroy(Vec);        
extern int VecGetVecs(Vec, int,Vec **);         
extern int VecFreeVecs(Vec*,int); 

typedef enum {NotSetValues, InsertValues, AddValues} InsertMode;

extern int VecSetValues(Vec, int, int *,Scalar*,InsertMode);
extern int VecBeginAssembly(Vec);
extern int VecEndAssembly(Vec);

#define ScatterReverse 1
#define ScatterDown    2
#define ScatterUp      4
#define ScatterAll     8
extern int VecScatterBegin(Vec,IS,Vec,IS,InsertMode,int,VecScatterCtx);
extern int VecScatterEnd(Vec,IS,Vec,IS,InsertMode,int,VecScatterCtx); 
extern int VecScatterCtxCreate(Vec,IS,Vec,IS,VecScatterCtx *);
extern int VecScatterCtxDestroy(VecScatterCtx);
extern int VecScatterCtxCopy(VecScatterCtx,VecScatterCtx *);

#define PipelineDown 0
#define PipelineUp   1
extern int VecPipelineBegin(Vec,IS,Vec,IS,InsertMode,int,VecScatterCtx);
extern int VecPipelineEnd(Vec,IS,Vec,IS,InsertMode,int,VecScatterCtx); 

extern int VecGetArray(Vec,Scalar**);
extern int VecValidVector(Vec);
extern int VecView(Vec, Viewer);

extern int VecGetSize(Vec,int *);
extern int VecGetLocalSize(Vec,int *);
extern int VecGetOwnershipRange(Vec,int*,int*);

/* utility routines */
extern int VecReciprocal(Vec);

#endif


