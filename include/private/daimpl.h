/*
   Distributed arrays - communication tools for parallel, rectangular grids.
*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H

#include "private/dmimpl.h"

typedef struct _DAOps *DAOps;
struct _DAOps {
  DMOPS(DA)
};

struct _p_DA {
  PETSCHEADER(struct _DAOps);
  DMHEADER
  PetscInt            M,N,P;                 /* array dimensions */
  PetscInt            m,n,p;                 /* processor layout */
  PetscInt            w;                     /* degrees of freedom per node */
  PetscInt            s;                     /* stencil width */
  PetscInt            xs,xe,ys,ye,zs,ze;     /* range of local values */
  PetscInt            Xs,Xe,Ys,Ye,Zs,Ze;     /* range including ghost values
                                                   values above already scaled by w */
  PetscInt            *idx,Nl;               /* local to global map */
  PetscInt            base;                  /* global number of 1st local node */
  DAPeriodicType      wrap;                  /* indicates type of periodic boundaries */
  VecScatter          gtol,ltog,ltol;        /* scatters, see below for details */
  DAStencilType       stencil_type;          /* stencil, either box or star */
  PetscInt            dim;                   /* DA dimension (1,2, or 3) */
  DAInterpolationType interptype;

  PetscInt            nlocal,Nlocal;         /* local size of local vector and global vector */

  AO                  ao;                    /* application ordering context */

  ISLocalToGlobalMapping ltogmap,ltogmapb;   /* local to global mapping for associated vectors */
  Vec                    coordinates;        /* coordinates (x,y,z) of local nodes, not including ghosts*/
  DA                     da_coordinates;     /* da for getting ghost values of coordinates */
  Vec                    ghosted_coordinates;/* coordinates with ghost nodes */
  char                   **fieldname;        /* names of individual components in vectors */

  PetscInt               *lx,*ly,*lz;        /* number of nodes in each partition block along 3 axis */
  Vec                    natural;            /* global vector for storing items in natural order */
  VecScatter             gton;               /* vector scatter from global to natural */
  PetscMPIInt            *neighbors;         /* ranks of all neighbors and self */

  ISColoring             localcoloring;       /* set by DAGetColoring() */
  ISColoring             ghostedcoloring;  

  DAElementType          elementtype;
  PetscInt               ne;                  /* number of elements */
  PetscInt               *e;                  /* the elements */

  PetscInt               refine_x,refine_y,refine_z; /* ratio used in refining */

#define DA_MAX_AD_ARRAYS 2 /* work arrays for holding derivative type data, via DAGetAdicArray() */
  void                   *adarrayin[DA_MAX_AD_ARRAYS],*adarrayout[DA_MAX_AD_ARRAYS]; 
  void                   *adarrayghostedin[DA_MAX_AD_ARRAYS],*adarrayghostedout[DA_MAX_AD_ARRAYS];
  void                   *adstartin[DA_MAX_AD_ARRAYS],*adstartout[DA_MAX_AD_ARRAYS]; 
  void                   *adstartghostedin[DA_MAX_AD_ARRAYS],*adstartghostedout[DA_MAX_AD_ARRAYS];
  PetscInt                    tdof,ghostedtdof;

                            /* work arrays for holding derivative type data, via DAGetAdicMFArray() */
  void                   *admfarrayin[DA_MAX_AD_ARRAYS],*admfarrayout[DA_MAX_AD_ARRAYS]; 
  void                   *admfarrayghostedin[DA_MAX_AD_ARRAYS],*admfarrayghostedout[DA_MAX_AD_ARRAYS];
  void                   *admfstartin[DA_MAX_AD_ARRAYS],*admfstartout[DA_MAX_AD_ARRAYS]; 
  void                   *admfstartghostedin[DA_MAX_AD_ARRAYS],*admfstartghostedout[DA_MAX_AD_ARRAYS];

#define DA_MAX_WORK_ARRAYS 2 /* work arrays for holding work via DAGetArray() */
  void                   *arrayin[DA_MAX_WORK_ARRAYS],*arrayout[DA_MAX_WORK_ARRAYS]; 
  void                   *arrayghostedin[DA_MAX_WORK_ARRAYS],*arrayghostedout[DA_MAX_WORK_ARRAYS];
  void                   *startin[DA_MAX_WORK_ARRAYS],*startout[DA_MAX_WORK_ARRAYS]; 
  void                   *startghostedin[DA_MAX_WORK_ARRAYS],*startghostedout[DA_MAX_WORK_ARRAYS];

  DALocalFunction1       lf;
  DALocalFunction1       lj;
  DALocalFunction1       adic_lf;
  DALocalFunction1       adicmf_lf;
  DALocalFunction1       adifor_lf;
  DALocalFunction1       adiformf_lf;

  PetscErrorCode (*lfi)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*);
  PetscErrorCode (*adic_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*);
  PetscErrorCode (*adicmf_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*);
  PetscErrorCode (*lfib)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*);
  PetscErrorCode (*adic_lfib)(DALocalInfo*,MatStencil*,void*,void*,void*);
  PetscErrorCode (*adicmf_lfib)(DALocalInfo*,MatStencil*,void*,void*,void*);

  /* used by DASetBlockFills() */
  PetscInt               *ofill,*dfill;

  /* used by DASetMatPreallocateOnly() */
  PetscTruth             prealloc_only;
};

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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT VecView_MPI_DA(Vec,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT VecLoadIntoVector_Binary_DA(PetscViewer,Vec);
EXTERN_C_END
EXTERN PetscErrorCode DAView_Private(DA);

extern PetscLogEvent  DA_GlobalToLocal, DA_LocalToGlobal, DA_LocalADFunction;

#endif
