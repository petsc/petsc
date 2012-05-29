

#if !defined(_DMIMPL_H)
#define _DMIMPL_H

#include <petscdm.h>

typedef struct _DMOps *DMOps;
struct _DMOps {
  PetscErrorCode (*view)(DM,PetscViewer); 
  PetscErrorCode (*load)(DM,PetscViewer); 
  PetscErrorCode (*setfromoptions)(DM); 
  PetscErrorCode (*setup)(DM); 
  PetscErrorCode (*createglobalvector)(DM,Vec*);
  PetscErrorCode (*createlocalvector)(DM,Vec*);
  PetscErrorCode (*createlocaltoglobalmapping)(DM);
  PetscErrorCode (*createlocaltoglobalmappingblock)(DM);
  PetscErrorCode (*createfieldis)(DM,PetscInt*,char***,IS**);

  PetscErrorCode (*getcoloring)(DM,ISColoringType,const MatType,ISColoring*);	
  PetscErrorCode (*creatematrix)(DM, const MatType,Mat*);
  PetscErrorCode (*createinterpolation)(DM,DM,Mat*,Vec*);
  PetscErrorCode (*getaggregates)(DM,DM,Mat*);
  PetscErrorCode (*getinjection)(DM,DM,VecScatter*);

  PetscErrorCode (*refine)(DM,MPI_Comm,DM*);
  PetscErrorCode (*coarsen)(DM,MPI_Comm,DM*);
  PetscErrorCode (*refinehierarchy)(DM,PetscInt,DM*);
  PetscErrorCode (*coarsenhierarchy)(DM,PetscInt,DM*);

  PetscErrorCode (*globaltolocalbegin)(DM,Vec,InsertMode,Vec);		
  PetscErrorCode (*globaltolocalend)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*localtoglobalbegin)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*localtoglobalend)(DM,Vec,InsertMode,Vec);

  PetscErrorCode (*initialguess)(DM,Vec);
  PetscErrorCode (*function)(DM,Vec,Vec);
  PetscErrorCode (*functionj)(DM,Vec,Vec);
  PetscErrorCode (*jacobian)(DM,Vec,Mat,Mat,MatStructure*);

  PetscErrorCode (*destroy)(DM);

  PetscErrorCode (*computevariablebounds)(DM,Vec,Vec);

  PetscErrorCode (*createfielddecompositiondm)(DM,const char*,DM*);
  PetscErrorCode (*createfielddecomposition)(DM,PetscInt*,char***,IS**,DM**);
  PetscErrorCode (*createdomaindecompositiondm)(DM,const char*,DM*);
  PetscErrorCode (*createdomaindecomposition)(DM,PetscInt*,char***,IS**,IS**,DM**);

};

typedef struct _DMCoarsenHookLink *DMCoarsenHookLink;
struct _DMCoarsenHookLink {
  PetscErrorCode (*coarsenhook)(DM,DM,void*);              /* Run once, when coarse DM is created */
  PetscErrorCode (*restricthook)(DM,Mat,Vec,Mat,DM,void*); /* Run each time a new problem is restricted to a coarse grid */
  void *ctx;
  DMCoarsenHookLink next;
};

typedef struct _DMRefineHookLink *DMRefineHookLink;
struct _DMRefineHookLink {
  PetscErrorCode (*refinehook)(DM,DM,void*);     /* Run once, when a fine DM is created */
  PetscErrorCode (*interphook)(DM,Mat,DM,void*); /* Run each time a new problem is interpolated to a fine grid */
  void *ctx;
  DMRefineHookLink next;
};

typedef enum {DMVEC_STATUS_IN,DMVEC_STATUS_OUT} DMVecStatus;
typedef struct _DMNamedVecLink *DMNamedVecLink;
struct _DMNamedVecLink {
  Vec X;
  char *name;
  DMVecStatus status;
  DMNamedVecLink next;
};

#define DM_MAX_WORK_VECTORS 100 /* work vectors available to users  via DMGetGlobalVector(), DMGetLocalVector() */

struct _p_DM {
  PETSCHEADER(struct _DMOps);
  Vec                    localin[DM_MAX_WORK_VECTORS],localout[DM_MAX_WORK_VECTORS];
  Vec                    globalin[DM_MAX_WORK_VECTORS],globalout[DM_MAX_WORK_VECTORS];
  DMNamedVecLink         namedglobal;
  PetscInt               workSize;
  PetscScalar            *workArray;
  void                   *ctx;    /* a user context */
  PetscErrorCode         (*ctxdestroy)(void**);
  Vec                    x;       /* location at which the functions/Jacobian are computed */
  ISColoringType         coloringtype;
  MatFDColoring          fd;      /* used by DMComputeJacobianDefault() */
  VecType                vectype;  /* type of vector created with DMCreateLocalVector() and DMCreateGlobalVector() */
  MatType                mattype;  /* type of matrix created with DMCreateMatrix() */
  PetscInt               bs;
  ISLocalToGlobalMapping ltogmap,ltogmapb;
  PetscBool              prealloc_only; /* Flag indicating the DMCreateMatrix() should only preallocate, not fill the matrix */
  PetscInt               levelup,leveldown;  /* if the DM has been obtained by refining (or coarsening) this indicates how many times that process has been used to generate this DM */
  PetscBool              setupcalled;        /* Indicates that the DM has been set up, methods that modify a DM such that a fresh setup is required should reset this flag */
  void                   *data;
  DMCoarsenHookLink      coarsenhook; /* For transfering auxiliary problem data to coarser grids */
  DMRefineHookLink       refinehook;
  DMLocalFunction1       lf;
  DMLocalJacobian1       lj;
  /* Flexible communication */
  PetscSF                sf;                   /* SF for parallel point overlap */
  PetscSF                defaultSF;            /* SF for parallel dof overlap using default section */
  /* Allows a non-standard data layout */
  PetscSection           defaultSection;       /* Layout for local vectors */
  PetscSection           defaultGlobalSection; /* Layout for global vectors */
};

PETSC_EXTERN PetscLogEvent DM_Convert, DM_GlobalToLocal, DM_LocalToGlobal;

/*

          Composite Vectors 

      Single global representation
      Individual global representations
      Single local representation
      Individual local representations

      Subsets of individual as a single????? Do we handle this by having DMComposite inside composite??????

       DM da_u, da_v, da_p

       DM dm_velocities

       DM dm

       DMDACreate(,&da_u);
       DMDACreate(,&da_v);
       DMCompositeCreate(,&dm_velocities);
       DMCompositeAddDM(dm_velocities,(DM)du);
       DMCompositeAddDM(dm_velocities,(DM)dv);

       DMDACreate(,&da_p);
       DMCompositeCreate(,&dm_velocities);
       DMCompositeAddDM(dm,(DM)dm_velocities);     
       DMCompositeAddDM(dm,(DM)dm_p);     


    Access parts of composite vectors (Composite only)
    ---------------------------------
      DMCompositeGetAccess  - access the global vector as subvectors and array (for redundant arrays)
      ADD for local vector - 

    Element access
    --------------
      From global vectors 
         -DAVecGetArray   - for DMDA
         -VecGetArray - for DMSliced
         ADD for DMComposite???  maybe

      From individual vector
          -DAVecGetArray - for DMDA
          -VecGetArray -for sliced  
         ADD for DMComposite??? maybe

      From single local vector
          ADD         * single local vector as arrays?

   Communication 
   -------------
      DMGlobalToLocal - global vector to single local vector

      DMCompositeScatter/Gather - direct to individual local vectors and arrays   CHANGE name closer to GlobalToLocal?

   Obtaining vectors
   ----------------- 
      DMCreateGlobal/Local 
      DMGetGlobal/Local 
      DMCompositeGetLocalVectors   - gives individual local work vectors and arrays
         

?????   individual global vectors   ????

*/

#endif
