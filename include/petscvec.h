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

typedef struct _Vec*           Vec;

extern int VecCreateSequential(int,Vec *);  
extern int VecCreateSequentialBLAS(int,Vec *); 

#if defined(USING_MPI)
extern int VecCreateMPI(MPI_Comm,int,int,Vec *);  
extern int VecCreateMPIBLAS(MPI_Comm,int,int,Vec *); 
#endif

extern int VecCreateInitialVector(int,Vec *); 




extern int VecDot(Vec, Vec, Scalar*);
extern int VecTDot(Vec, Vec, Scalar*);  
extern int VecMDot(int,      Vec ,Vec*,Scalar*);
extern int VecMTDot(int,      Vec ,Vec*,Scalar*); 
extern int VecNorm(Vec, double*);
extern int VecASum(Vec, Scalar*);
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

extern int VecAddValues(Vec, int, int *,Scalar*);
extern int VecInsertValues(Vec, int, int *,Scalar*);
extern int VecBeginAssembly(Vec);
extern int VecEndAssembly(Vec);

extern int VecScatterBegin(Vec,IS,Vec,IS,ISScatterCtx *);
extern int VecScatterEnd(Vec,IS,Vec,IS,ISScatterCtx *); 

extern int VecScatterAddBegin(Vec,IS,Vec,IS,ISScatterCtx *);
extern int VecScatterAddEnd(Vec,IS,Vec,IS,ISScatterCtx *);  

extern int VecGetArray(Vec,Scalar**);
extern int VecValidVector(Vec);
extern int VecView(Vec, Viewer);

extern int VecGetSize(Vec,int *);
extern int VecGetLocalSize(Vec,int *);

/* utility routines */
extern int VecReciprocal(Vec);

#endif


