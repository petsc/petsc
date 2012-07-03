#ifndef ADINTRINSICS_H
#define ADINTRINSICS_H 1

/* stdio.h: For adintr_ehsout(FILE*) prototype below. */
#include <stdio.h>

#undef ADINTR_EXTERN
#ifdef ADINTRINSICS_C
#define ADINTR_EXTERN
#else
#define ADINTR_EXTERN extern
#endif

#if defined(__cplusplus)
#undef ADINTR_EXTERN
#define ADINTR_EXTERN extern "C"
#endif 

enum ADIntr_Modes
{
	ADINTR_IGNORE = -1,
	ADINTR_PERFORMANCE = 1,
	ADINTR_REPORTONCE = 2
};

enum ADIntr_Partials
{
	ADINTR_FX = 0, 
	ADINTR_FY, 
	ADINTR_FXX, 
	ADINTR_FXY, 
	ADINTR_FYY,
	ADINTR_MAX_PARTIALS
};

/*** This information now in an automatically generated include file ***/
/*
enum ADIntr_Funcs
{
	ADINTR_FABS = x,
	ADINTR_MAX_FUNC
};
*/
#include "initenum.h"


/* Global Variables */
#ifndef ADINTRINSICS_C   /* initialized in adintrinsics.c */
ADINTR_EXTERN double ADIntr_Partials[ADINTR_MAX_FUNC][ADINTR_MAX_PARTIALS];
ADINTR_EXTERN enum ADIntr_Modes ADIntr_Mode;
#endif

/* Prototypes */
#include "knr-compat.h"

/* All ADIntrinsics function prototypes */
#include "adintr_proto.h"

#ifdef ADINTRINSICS_WITH_ADIC
#include "g_proto.h"
#endif

/* Prototypes for mode switching functions */

ADINTR_EXTERN enum ADIntr_Modes adintr_current_mode Proto((void));
ADINTR_EXTERN void adintr_mode_push Proto((enum ADIntr_Modes new_mode));
ADINTR_EXTERN void adintr_mode_pop Proto((void));

/* Prototypes for ADIntrinsics interface functions */

ADINTR_EXTERN void adintr_ehsup Proto((enum ADIntr_Funcs func,
				enum ADIntr_Partials partial,
				double value));

ADINTR_EXTERN double adintr_ehgup Proto((enum ADIntr_Funcs func,
				  enum ADIntr_Partials partial));

ADINTR_EXTERN void adintr_ehsout Proto((FILE *the_file));
ADINTR_EXTERN void adintr_ehrpt Proto((void));
ADINTR_EXTERN void adintr_ehrst Proto((void));

ADINTR_EXTERN void adintr_ehsfid Proto((int *g_ehfid, char *routine, char *filename));

#if 0
#ifdef ADINTRINSICS_INLINE
/* Support is easy, but who needs it? */
/* #include "inline defininitions for mode switching functions" */
#endif /* ADINTRINSICS_INLINE */
#endif


#endif /* ndef ADINTRINSICS_H */

#if ad_GRAD_MAX == 1
PETSC_EXTERN MPI_Op PetscADMax_Op;
PETSC_EXTERN MPI_Op PetscADMin_Op;
#  define admf_PetscGlobalMax(c,a,b) MPI_Allreduce((void*)a,b,2,MPIU_SCALAR,PetscADMax_Op,c)
#  define admf_PetscGlobalMin(c,a,b) MPI_Allreduce((void*)a,b,2,MPIU_SCALAR,PetscADMin_Op,c)
#  define admf_PetscGlobalSum(c,a,b) MPI_Allreduce((void*)a,b,2,MPIU_SCALAR,MPIU_SUM,c)
#else
#  define ad_PetscGlobalMax(a,b,c) 1   /* 1 generates error to indicate not implemented */
#  define ad_PetscGlobalMin(a,b,c) 1
#  define ad_PetscGlobalSum(a,b,c) 1
#endif
