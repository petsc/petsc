/*
  Defines the interface functions for the DMMG object.
*/
#ifndef __PETSCDMMG_H
#define __PETSCDMMG_H
#include "petscsnes.h"
#include "petscdmda.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     DMMGArray - Fortran only. This is used in the main program when doing DMMGCreate(), DMMGSetDM() etc.
        in the subroutines like FormFunction() one should use DMMG.

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 

        You can use DMMGArrayGetDMMG(DMMGArray,DMMG,ierr) to obtain the DMMG from a DMMG.

   Level: intermediate

  Concepts: multigrid, Newton-multigrid

.seealso:  DMCompositeCreate(),  DMComposite, DM, DMMGCreate(), DMMGSetKSP(), DMMGSetSNES(), DMMGSetInitialGuess(),
           DMMGSetNullSpace(), DMMGSetMatType()
S*/

/*S
     DMMG -  Data structure to easily manage multi-level non-linear solvers on grids managed by DM

    This is being deprecated. Use KSPSetDM() for linear problems and SNESSetDM() for nonlinear problems. 
    See src/ksp/ksp/examples/tutorials/ex45.c and src/snes/examples/tutorials/ex57.c 
          
   Level: intermediate

   Fortran Users: see also DMMGArray

  Concepts: multigrid, Newton-multigrid

.seealso:  DMCompositeCreate(), DMComposite, DM, DMMGCreate(), DMMGSetKSP(), DMMGSetSNES(), DMMGSetInitialGuess(),
           DMMGSetNullSpace(),  DMMGSetMatType()
S*/
typedef struct _n_DMMG* DMMG;
struct _n_DMMG {
  DM             dm;                   /* grid information for this level */
  Vec            x,b,r;                /* global vectors used in multigrid preconditioner for this level*/
  Mat            J;                    /* matrix on this level */
  Mat            B;
  Mat            R;                    /* restriction to next coarser level (not defined on level 0) */
  PetscInt       nlevels;              /* number of levels above this one (total number of levels on level 0)*/
  MPI_Comm       comm;
  PetscErrorCode (*solve)(DMMG*,PetscInt);
  void           *user;         
  PetscBool      galerkin;                  /* for A_c = R*A*R^T */
  MatType        mtype;                     /* create matrices of this type */
  char           *prefix;

  /* KSP only */
  KSP            ksp;             
  PetscErrorCode (*rhs)(DMMG,Vec);

  /* SNES only */
  Vec            Rscale;                 /* scaling to restriction before computing Jacobian */
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);  

  PetscBool      updatejacobian;         /* compute new Jacobian when DMMGComputeJacobian_Multigrid() is called */
  PetscInt       updatejacobianperiod;   /* how often, inside a SNES, the Jacobian is recomputed */

  PetscBool      getcoloringfrommat;     /* call a graph coloring algorithm on the matrix to get the coloring, instead of getting it from the DM */
  ISColoringType isctype;
  MatFDColoring  fdcoloring;             /* only used with FD coloring for Jacobian */  
  SNES           snes;                  
  PetscErrorCode (*initialguess)(DMMG,Vec);
  Vec            w,work1,work2;         /* global vectors */
  Vec            lwork1;

  PetscErrorCode (*lfj)(void);          /* function used when computing Jacobian via FD, usually da->lf */

  /* FAS only */
  NLF            nlf;                   /* FAS smoother object */
  VecScatter     inject;                /* inject from this level to the next coarsest */
  PetscBool      monitor,monitorall;
  PetscInt       presmooth,postsmooth,coarsesmooth;
  PetscReal      rtol,abstol,rrtol;       /* convergence tolerance */   
  
};

extern PetscErrorCode  DMMGCreate(MPI_Comm,PetscInt,void*,DMMG**);
extern PetscErrorCode  DMMGDestroy(DMMG*);
extern PetscErrorCode  DMMGSetUp(DMMG*);
extern PetscErrorCode  DMMGSetKSP(DMMG*,PetscErrorCode (*)(DMMG,Vec),PetscErrorCode (*)(DMMG,Mat,Mat));
extern PetscErrorCode  DMMGSetSNES(DMMG*,PetscErrorCode (*)(SNES,Vec,Vec,void*),PetscErrorCode (*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*));
extern PetscErrorCode  DMMGSetFromOptions(DMMG*);

extern PetscErrorCode  DMMGSetInitialGuess(DMMG*,PetscErrorCode (*)(DMMG,Vec));
extern PetscErrorCode  DMMGInitialGuessCurrent(DMMG,Vec);
extern PetscErrorCode  DMMGView(DMMG*,PetscViewer);
extern PetscErrorCode  DMMGSolve(DMMG*);
extern PetscErrorCode  DMMGSetUseMatrixFree(DMMG*);
extern PetscErrorCode  DMMGSetDM(DMMG*,DM);
extern PetscErrorCode  DMMGSetUpLevel(DMMG*,KSP,PetscInt);
extern PetscErrorCode  DMMGSetNullSpace(DMMG*,PetscBool ,PetscInt,PetscErrorCode (*)(DMMG,Vec[]));
extern PetscErrorCode  DMMGSetMatType(DMMG*,const MatType);
extern PetscErrorCode  DMMGSetISColoringType(DMMG*,ISColoringType);
extern PetscErrorCode  DMMGSetOptionsPrefix(DMMG*,const char[]);
extern PetscErrorCode  DMMGFormFunction(SNES,Vec,Vec,void *);

extern PetscErrorCode  DMMGGetSNESLocal(DMMG*,DMDALocalFunction1*,DMDALocalFunction1*);
extern PetscErrorCode  DMMGSetSNESLocal_Private(DMMG*,DMDALocalFunction1,DMDALocalFunction1,DMDALocalFunction1,DMDALocalFunction1);
#if defined(PETSC_HAVE_ADIC)
#  define DMMGSetSNESLocal(dmmg,function,jacobian,ad_function,admf_function) \
  DMMGSetSNESLocal_Private(dmmg,(DMDALocalFunction1)function,(DMDALocalFunction1)jacobian,(DMDALocalFunction1)(ad_function),(DMDALocalFunction1)(admf_function))
#else
#  define DMMGSetSNESLocal(dmmg,function,jacobian,ad_function,admf_function) DMMGSetSNESLocal_Private(dmmg,(DMDALocalFunction1)function,(DMDALocalFunction1)jacobian,(DMDALocalFunction1)0,(DMDALocalFunction1)0)
#endif

extern PetscErrorCode  DMMGSetSNESLocali_Private(DMMG*,PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMMGSetSNESLocali(dmmg,function,ad_function,admf_function) DMMGSetSNESLocali_Private(dmmg,(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))(ad_function),(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))(admf_function))
#else
#  define DMMGSetSNESLocali(dmmg,function,ad_function,admf_function) DMMGSetSNESLocali_Private(dmmg,(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,0,0)
#endif

extern PetscErrorCode  DMMGSetSNESLocalib_Private(DMMG*,PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*),PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*),PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*));
#if defined(PETSC_HAVE_ADIC)
#  define DMMGSetSNESLocalib(dmmg,function,ad_function,admf_function) DMMGSetSNESLocalib_Private(dmmg,(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))(ad_function),(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,void*,void*))(admf_function))
#else
#  define DMMGSetSNESLocalib(dmmg,function,ad_function,admf_function) DMMGSetSNESLocalib_Private(dmmg,(PetscErrorCode(*)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*))function,0,0)
#endif

extern PetscErrorCode DMMGSetSNESLocalFD(DMMG*,DMDALocalFunction1);

/*MC
   DMMGGetRHS - Returns the right hand side vector from a DMMG solve on the finest grid

   Synopsis:
   Vec DMMGGetRHS(DMMG *dmmg)

   Not Collective, but resulting vector is parallel

   Input Parameters:
.   dmmg - DMMG solve context

   Level: intermediate

   Fortran Usage:
.     DMMGGetRHS(DMMG dmmg,Vec b,PetscErrorCode ierr)

.seealso: DMMGCreate(), DMMGSetSNES(), DMMGSetKSP(), DMMGSetSNESLocal(), DMMGGetRHS()

M*/
#define DMMGGetRHS(ctx)              (ctx)[(ctx)[0]->nlevels-1]->b

#define DMMGGetr(ctx)              (ctx)[(ctx)[0]->nlevels-1]->r

/*MC
   DMMGGetx - Returns the solution vector from a DMMG solve on the finest grid

   Synopsis:
   Vec DMMGGetx(DMMG *dmmg)

   Not Collective, but resulting vector is parallel

   Input Parameters:
.   dmmg - DMMG solve context

   Level: intermediate

   Fortran Usage:
.     DMMGGetx(DMMG dmmg,Vec x,PetscErrorCode ierr)

.seealso: DMMGCreate(), DMMGSetSNES(), DMMGSetKSP(), DMMGSetSNESLocal()

M*/
#define DMMGGetx(ctx)              (ctx)[(ctx)[0]->nlevels-1]->x

/*MC
   DMMGGetJ - Returns the Jacobian (matrix) for the finest level

   Synopsis:
   Mat DMMGGetJ(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetB(), DMMGGetRHS()

M*/
#define DMMGGetJ(ctx)              (ctx)[(ctx)[0]->nlevels-1]->J

/*MC
   DMMGGetComm - Returns the MPI_Comm for the finest level

   Synopsis:
   MPI_Comm DMMGGetJ(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ()

M*/
#define DMMGGetComm(ctx)           (ctx)[(ctx)[0]->nlevels-1]->comm

/*MC
   DMMGGetB - Returns the matrix for the finest level used to construct the preconditioner; usually 
              the same as the Jacobian

   Synopsis:
   Mat DMMGGetJ(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ()

M*/
#define DMMGGetB(ctx)              (ctx)[(ctx)[0]->nlevels-1]->B

/*MC
   DMMGGetFine - Returns the DMMG associated with the finest level

   Synopsis:
   DMMG DMMGGetFine(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ()

M*/
#define DMMGGetFine(ctx)           (ctx)[(ctx)[0]->nlevels-1]


/*MC
   DMMGGetKSP - Gets the KSP object (linear solver object) for the finest level

   Synopsis:
   KSP DMMGGetKSP(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

   Notes: If this is a linear problem (i.e. DMMGSetKSP() was used) then this is the 
     master linear solver. If this is a nonlinear problem (i.e. DMMGSetSNES() was used) this
     returns the KSP (linear solver) that is associated with the SNES (nonlinear solver)

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetSNES()

M*/
#define DMMGGetKSP(ctx)            (ctx)[(ctx)[0]->nlevels-1]->ksp

/*MC
   DMMGGetSNES - Gets the SNES object (nonlinear solver) for the finest level

   Synopsis:
   SNES DMMGGetSNES(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

   Notes: If this is a linear problem (i.e. DMMGSetKSP() was used) then this returns PETSC_NULL

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetKSP()

M*/
#define DMMGGetSNES(ctx)           (ctx)[(ctx)[0]->nlevels-1]->snes

/*MC
   DMMGGetDM - Gets the DM object on the finest level

   Synopsis:
   DM DMMGGetDM(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser(), DMMGGetJ(), KSPGetKSP()

M*/
#define DMMGGetDM(ctx)             ((ctx)[(ctx)[0]->nlevels-1]->dm)

/*MC
   DMMGGetUser - Returns the user context for a particular level

   Synopsis:
   void* DMMGGetUser(DMMG *dmmg,PetscInt level)

   Not Collective

   Input Parameters:
+   dmmg - DMMG solve context
-   level - the number of the level you want the context for

   Level: intermediate

.seealso: DMMGCreate(), DMMGSetUser()

M*/
#define DMMGGetUser(ctx,level)     ((ctx)[level]->user)

/*MC
   DMMGSetUser - Sets the user context for a particular level

   Synopsis:
   PetscErrorCode DMMGSetUser(DMMG *dmmg,PetscInt level,void *ctx)

   Not Collective

   Input Parameters:
+   dmmg - DMMG solve context
.   level - the number of the level you want the context for
-   ctx - the context

   Level: intermediate

   Note: if the context is the same for each level just pass it in with 
         DMMGCreate() and don't call this macro

.seealso: DMMGCreate(), DMMGGetUser()

M*/
#define DMMGSetUser(ctx,level,usr) ((ctx)[level]->user = usr,0)

/*MC
   DMMGGetLevels - Gets the number of levels in a DMMG object

   Synopsis:
   PetscInt DMMGGetLevels(DMMG *dmmg)

   Not Collective

   Input Parameter:
.   dmmg - DMMG solve context

   Level: intermediate

.seealso: DMMGCreate(), DMMGGetUser()

M*/
#define DMMGGetLevels(ctx)         (ctx)[0]->nlevels

PETSC_EXTERN_CXX_END
#endif
