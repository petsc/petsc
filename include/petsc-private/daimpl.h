/*
   Distributed arrays - communication tools for parallel, rectangular grids.
*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H

#include <petscdmda.h>
#include "petsc-private/dmimpl.h"

typedef struct {
  PetscInt            M,N,P;                 /* array dimensions */
  PetscInt            m,n,p;                 /* processor layout */
  PetscInt            w;                     /* degrees of freedom per node */
  PetscInt            s;                     /* stencil width */
  PetscInt            xs,xe,ys,ye,zs,ze;     /* range of local values */
  PetscInt            Xs,Xe,Ys,Ye,Zs,Ze;     /* range including ghost values
                                                   values above already scaled by w */
  PetscInt            *idx,Nl;               /* local to global map */
  PetscInt            base;                  /* global number of 1st local node */
  DMDABoundaryType    bx,by,bz;              /* indicates type of ghost nodes at boundary */
  VecScatter          gtol,ltog,ltol;        /* scatters, see below for details */
  DMDAStencilType       stencil_type;          /* stencil, either box or star */
  PetscInt            dim;                   /* DMDA dimension (1,2, or 3) */
  DMDAInterpolationType interptype;

  PetscInt            nlocal,Nlocal;         /* local size of local vector and global vector */

  AO                  ao;                    /* application ordering context */

  Vec                    coordinates;        /* coordinates (x,y,z) of local nodes, not including ghosts*/
  DM                     da_coordinates;     /* da for getting ghost values of coordinates */
  Vec                    ghosted_coordinates;/* coordinates with ghost nodes */
  char                   **fieldname;        /* names of individual components in vectors */

  PetscInt               *lx,*ly,*lz;        /* number of nodes in each partition block along 3 axis */
  Vec                    natural;            /* global vector for storing items in natural order */
  VecScatter             gton;               /* vector scatter from global to natural */
  PetscMPIInt            *neighbors;         /* ranks of all neighbors and self */

  ISColoring             localcoloring;       /* set by DMCreateColoring() */
  ISColoring             ghostedcoloring;  

  DMDAElementType          elementtype;
  PetscInt               ne;                  /* number of elements */
  PetscInt               *e;                  /* the elements */

  PetscInt               refine_x,refine_y,refine_z; /* ratio used in refining */

#define DMDA_MAX_AD_ARRAYS 2 /* work arrays for holding derivative type data, via DMDAGetAdicArray() */
  void                   *adarrayin[DMDA_MAX_AD_ARRAYS],*adarrayout[DMDA_MAX_AD_ARRAYS]; 
  void                   *adarrayghostedin[DMDA_MAX_AD_ARRAYS],*adarrayghostedout[DMDA_MAX_AD_ARRAYS];
  void                   *adstartin[DMDA_MAX_AD_ARRAYS],*adstartout[DMDA_MAX_AD_ARRAYS]; 
  void                   *adstartghostedin[DMDA_MAX_AD_ARRAYS],*adstartghostedout[DMDA_MAX_AD_ARRAYS];
  PetscInt                    tdof,ghostedtdof;

                            /* work arrays for holding derivative type data, via DMDAGetAdicMFArray() */
  void                   *admfarrayin[DMDA_MAX_AD_ARRAYS],*admfarrayout[DMDA_MAX_AD_ARRAYS]; 
  void                   *admfarrayghostedin[DMDA_MAX_AD_ARRAYS],*admfarrayghostedout[DMDA_MAX_AD_ARRAYS];
  void                   *admfstartin[DMDA_MAX_AD_ARRAYS],*admfstartout[DMDA_MAX_AD_ARRAYS]; 
  void                   *admfstartghostedin[DMDA_MAX_AD_ARRAYS],*admfstartghostedout[DMDA_MAX_AD_ARRAYS];

#define DMDA_MAX_WORK_ARRAYS 2 /* work arrays for holding work via DMDAGetArray() */
  void                   *arrayin[DMDA_MAX_WORK_ARRAYS],*arrayout[DMDA_MAX_WORK_ARRAYS]; 
  void                   *arrayghostedin[DMDA_MAX_WORK_ARRAYS],*arrayghostedout[DMDA_MAX_WORK_ARRAYS];
  void                   *startin[DMDA_MAX_WORK_ARRAYS],*startout[DMDA_MAX_WORK_ARRAYS]; 
  void                   *startghostedin[DMDA_MAX_WORK_ARRAYS],*startghostedout[DMDA_MAX_WORK_ARRAYS];

  DMDALocalFunction1       lf;
  DMDALocalFunction1       lj;
  DMDALocalFunction1       adic_lf;
  DMDALocalFunction1       adicmf_lf;
  DMDALocalFunction1       adifor_lf;
  DMDALocalFunction1       adiformf_lf;

  PetscErrorCode (*lfi)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*);
  PetscErrorCode (*adic_lfi)(DMDALocalInfo*,MatStencil*,void*,void*,void*);
  PetscErrorCode (*adicmf_lfi)(DMDALocalInfo*,MatStencil*,void*,void*,void*);
  PetscErrorCode (*lfib)(DMDALocalInfo*,MatStencil*,void*,PetscScalar*,void*);
  PetscErrorCode (*adic_lfib)(DMDALocalInfo*,MatStencil*,void*,void*,void*);
  PetscErrorCode (*adicmf_lfib)(DMDALocalInfo*,MatStencil*,void*,void*,void*);

  /* used by DMDASetBlockFills() */
  PetscInt               *ofill,*dfill;

  /* used by DMDASetMatPreallocateOnly() */
  PetscBool              prealloc_only;

  /* Allows a non-standard data layout */
  PetscSection           defaultSection;       /* Layout for local vectors */
  PetscSection           defaultGlobalSection; /* Layout for global vectors */
} DM_DA;

/*
  Vectors:
     Global has on each processor the interior degrees of freedom and
         no ghost points. This vector is what the solvers usually see.
     Local has on each processor the ghost points as well. This is 
          what code to calculate Jacobians, etc. usually sees.
  Vector scatters:
     gtol - Global representation to local
     ltog - Local representation to global (involves no communication)
     ltol - Local representation to local representation, updates the
            ghostpoint values in the second vector from (correct) interior
            values in the first vector.  This is good for explicit
            nearest neighbor timestepping.
*/

EXTERN_C_BEGIN
extern PetscErrorCode  VecView_MPI_DA(Vec,PetscViewer);
extern PetscErrorCode  VecLoad_Default_DA(Vec, PetscViewer);
EXTERN_C_END
extern PetscErrorCode DMView_DA_Private(DM);
extern PetscErrorCode DMView_DA_Matlab(DM,PetscViewer);
extern PetscErrorCode DMView_DA_Binary(DM,PetscViewer);
extern PetscErrorCode DMView_DA_VTK(DM,PetscViewer);
extern PetscErrorCode DMDAVTKWriteAll(PetscObject,PetscViewer);

extern PetscLogEvent  DMDA_LocalADFunction;

#endif
