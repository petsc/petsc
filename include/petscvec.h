/* $Id: vec.h,v 1.51 1996/08/04 23:14:42 bsmith Exp bsmith $ */
/* 
   This defines the abstract vector component of PETSc.
 */

#ifndef __VEC_PACKAGE 
#define __VEC_PACKAGE
#include "is.h"
#include "sys.h"

#define VEC_COOKIE         PETSC_COOKIE+3
#define VEC_SCATTER_COOKIE PETSC_COOKIE+4

typedef enum { VECSAME=-1, VECSEQ, VECMPI } VecType;

typedef struct _Vec*         Vec;
typedef struct _VecScatter*  VecScatter;


extern int VecCreateSeq(MPI_Comm,int,Vec*);  
extern int VecCreateMPI(MPI_Comm,int,int,Vec*);  
extern int VecCreate(MPI_Comm,int,Vec*); 

extern int VecDestroy(Vec);        

extern int VecDot(Vec,Vec,Scalar*);
extern int VecTDot(Vec,Vec,Scalar*);  
extern int VecMDot(int,Vec,Vec*,Scalar*);
extern int VecMTDot(int,Vec,Vec*,Scalar*); 

typedef enum {NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4} NormType;
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

typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES} InsertMode;
extern int VecSetValues(Vec,int,int*,Scalar*,InsertMode);
extern int VecAssemblyBegin(Vec);
extern int VecAssemblyEnd(Vec);

typedef enum {SCATTER_REVERSE=1,SCATTER_ALL=8} ScatterMode;
extern int VecScatterBegin(Vec,Vec,InsertMode,ScatterMode,VecScatter);
extern int VecScatterEnd(Vec,Vec,InsertMode,ScatterMode,VecScatter); 
extern int VecScatterCreate(Vec,IS,Vec,IS,VecScatter *);
extern int VecScatterDestroy(VecScatter);
extern int VecScatterCopy(VecScatter,VecScatter *);
extern int VecScatterView(VecScatter,Viewer);

extern int VecScatterRemap(VecScatter,int *,int*);

extern int VecGetArray(Vec,Scalar**);
extern int VecRestoreArray(Vec,Scalar**);
extern int VecPlaceArray(Vec,Scalar*);
extern int VecGetArrays(Vec*,int,Scalar***);
extern int VecRestoreArrays(Vec*,int,Scalar***);
extern int VecValid(Vec,PetscTruth*);
extern int VecView(Vec,Viewer);
extern int VecLoad(Viewer,Vec*);
extern int VecEqual( Vec,Vec,PetscTruth*);

extern int VecGetSize(Vec,int*);
extern int VecGetType(Vec,VecType*,char**);
extern int VecGetLocalSize(Vec,int*);
extern int VecGetOwnershipRange(Vec,int*,int*);

#if defined(__DRAW_PACKAGE)
extern int DrawTensorContour(Draw,int,int,double *,double *,Vec);
#endif

#endif



