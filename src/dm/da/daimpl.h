/* $Id: daimpl.h,v 1.44 2001/08/06 21:18:31 bsmith Exp $ */

/*
   Distributed arrays - communication tools for parallel, rectangular grids.
*/

#if !defined(_DAIMPL_H)
#define _DAIMPL_H

#include "petscconfig.h"
#include "petscda.h"

/*
   The DM interface is shared by DA, VecPack, and any other object that may 
  be used with the DMMG class. If you change this MAKE SURE you change
  struct _DAOps and struct _VecPackOps!
*/
typedef struct _DMOps *DMOps;
struct _DMOps {
  int  (*view)(DM,PetscViewer);
  int  (*createglobalvector)(DM,Vec*);
  int  (*getcoloring)(DM,ISColoringType,ISColoring*);
  int  (*getmatrix)(DM,MatType,Mat*);
  int  (*getinterpolation)(DM,DM,Mat*,Vec*);
  int  (*refine)(DM,MPI_Comm,DM*);
};

struct _p_DM {
  PETSCHEADER(struct _DMOps)
};

typedef struct _DAOps *DAOps;
struct _DAOps {
  int  (*view)(DA,PetscViewer);
  int  (*createglobalvector)(DA,Vec*);
  int  (*getcoloring)(DA,ISColoringType,ISColoring*);
  int  (*getmatrix)(DA,MatType,Mat*);
  int  (*getinterpolation)(DA,DA,Mat*,Vec*);
  int  (*refine)(DA,MPI_Comm,DA*);
};

struct _p_DA {
  PETSCHEADER(struct _DAOps)
  int                 M,N,P;                 /* array dimensions */
  int                 m,n,p;                 /* processor layout */
  int                 w;                     /* degrees of freedom per node */
  int                 s;                     /* stencil width */
  int                 xs,xe,ys,ye,zs,ze;     /* range of local values */
  int                 Xs,Xe,Ys,Ye,Zs,Ze;     /* range including ghost values
                                                   values above already scaled by w */
  int                 *idx,Nl;               /* local to global map */
  int                 base;                  /* global number of 1st local node */
  DAPeriodicType      wrap;                  /* indicates type of periodic boundaries */
  VecScatter          gtol,ltog,ltol;        /* scatters, see below for details */
  Vec                 global,local;          /* vectors that are discrete functions */
  DAStencilType       stencil_type;          /* stencil, either box or star */
  int                 dim;                   /* DA dimension (1,2, or 3) */
  DAInterpolationType interptype;

  int                 *gtog1;                /* mapping from global ordering to
                                                  ordering that would be used for 1
                                                  proc; intended for internal use only */
  AO                  ao;                    /* application ordering context */

  ISLocalToGlobalMapping ltogmap,ltogmapb;   /* local to global mapping for associated vectors */
  Vec                    coordinates;        /* coordinates (x,y,x) of local nodes, not including ghosts*/
  char                   **fieldname;        /* names of individual components in vectors */

  int                    *lx,*ly,*lz;        /* number of nodes in each partition block along 3 axis */
  Vec                    natural;            /* global vector for storing items in natural order */
  VecScatter             gton;               /* vector scatter from global to natural */

  ISColoring            localcoloring;       /* set by DAGetColoring() */
  ISColoring            ghostedcoloring;  

#define DA_MAX_WORK_VECTORS 10 /* work vectors available to users  via DAVecGetArray() */
  Vec                    localin[DA_MAX_WORK_VECTORS],localout[DA_MAX_WORK_VECTORS];   
  Vec                    globalin[DA_MAX_WORK_VECTORS],globalout[DA_MAX_WORK_VECTORS]; 

  int                    refine_x,refine_y,refine_z; /* ratio used in refining */

#define DA_MAX_AD_ARRAYS 2 /* work arrays for holding derivative type data, via DAGetAdicArray() */
  void                   *adarrayin[DA_MAX_AD_ARRAYS],*adarrayout[DA_MAX_AD_ARRAYS]; 
  void                   *adarrayghostedin[DA_MAX_AD_ARRAYS],*adarrayghostedout[DA_MAX_AD_ARRAYS];
  void                   *adstartin[DA_MAX_AD_ARRAYS],*adstartout[DA_MAX_AD_ARRAYS]; 
  void                   *adstartghostedin[DA_MAX_AD_ARRAYS],*adstartghostedout[DA_MAX_AD_ARRAYS];
  int                    tdof,ghostedtdof;

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

  int                    (*lfi)(DALocalInfo*,MatStencil*,void*,PetscScalar*,void*);
  int                    (*adic_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*);
  int                    (*adicmf_lfi)(DALocalInfo*,MatStencil*,void*,void*,void*);
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

EXTERN int DAView_Binary(DA,PetscViewer);
EXTERN_C_BEGIN
EXTERN int VecView_MPI_DA(Vec,PetscViewer);
EXTERN int VecLoadIntoVector_Binary_DA(PetscViewer,Vec);
EXTERN_C_END

EXTERN int DAGetGlobalToGlobal1_Private(DA,int**);

#endif
