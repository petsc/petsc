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
typedef struct _VecScatterCtx* VecScatterCtx;
typedef double*                VecScalar;

extern int VecCreateSequential          ANSI_ARGS((int,Vec *));  
extern int VecCreateSequentialBLAS      ANSI_ARGS((int,Vec *)); 
extern int VecCreateComplexSequential   ANSI_ARGS((int,Vec *));

extern int VecCreateInitialVector   ANSI_ARGS((int,int,char **,Vec *)); 

#if defined(MPI_PACKAGE)
extern int VecCreateMPI            ANSI_ARGS((void *,int,int,Vec *));  
extern int VecCreateMPIBLAS        ANSI_ARGS((void *,int,int,Vec *));  
extern int VecCreateComplexMPI     ANSI_ARGS((void *,int,int,Vec *));  
#endif

extern int VecDot           ANSI_ARGS((Vec, Vec, VecScalar));
extern int VecTDot          ANSI_ARGS((Vec, Vec, VecScalar));  
extern int VecMDot          ANSI_ARGS((int,      Vec ,Vec*,VecScalar));
extern int VecMTDot         ANSI_ARGS((int,      Vec ,Vec*,VecScalar)); 
extern int VecNorm          ANSI_ARGS((Vec, VecScalar));
extern int VecASum          ANSI_ARGS((Vec, VecScalar));
extern int VecMax           ANSI_ARGS((Vec, int *,    VecScalar));
extern int VecScale         ANSI_ARGS((VecScalar, Vec));    
extern int VecCopy          ANSI_ARGS((Vec, Vec));        
extern int VecSet           ANSI_ARGS((VecScalar, Vec));    
extern int VecSwap          ANSI_ARGS((Vec, Vec));
extern int VecAXPY          ANSI_ARGS((VecScalar, Vec, Vec));  
extern int VecMAXPY         ANSI_ARGS((int,      VecScalar, Vec ,Vec*));
extern int VecAYPX          ANSI_ARGS((VecScalar, Vec, Vec));
extern int VecWAXPY         ANSI_ARGS((VecScalar, Vec, Vec, Vec));
extern int VecPMult         ANSI_ARGS((Vec, Vec, Vec));    
extern int VecPDiv          ANSI_ARGS((Vec, Vec, Vec));    
extern int VecCreate        ANSI_ARGS((Vec,Vec *));          
extern int VecDestroy       ANSI_ARGS((Vec));        
extern int VecGetVecs       ANSI_ARGS((Vec, int,Vec **));         
extern int VecFreeVecs      ANSI_ARGS((Vec*,int)); 

extern int VecAddValues     ANSI_ARGS((Vec, int, int *,VecScalar));
extern int VecInsertValues  ANSI_ARGS((Vec, int, int *,VecScalar));
extern int VecBeginAssembly ANSI_ARGS((Vec));
extern int VecEndAssembly   ANSI_ARGS((Vec));

extern int VecScatterBegin  ANSI_ARGS((Vec,IS,Vec,IS,VecScatterCtx *));
extern int VecScatterEnd    ANSI_ARGS((Vec,IS,Vec,IS,VecScatterCtx *)); 

extern int VecScatterAddBegin ANSI_ARGS((Vec,IS,Vec,IS,VecScatterCtx *));
extern int VecScatterAddEnd   ANSI_ARGS((Vec,IS,Vec,IS,VecScatterCtx *));  

extern int VecGetArray      ANSI_ARGS((Vec,VecScalar **));
extern int VecValidVector   ANSI_ARGS((Vec));
extern int VecView          ANSI_ARGS((Vec,void *));

extern int VecGetSize       ANSI_ARGS((Vec,int *));
extern int VecGetLocalSize  ANSI_ARGS((Vec,int *));

#endif


