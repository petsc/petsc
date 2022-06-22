
#if !defined(_DMIMPL_H)
#define _DMIMPL_H

#include <petscdm.h>
#ifdef PETSC_HAVE_LIBCEED
#include <petscdmceed.h>
#endif
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscdsimpl.h>
#include <petsc/private/sectionimpl.h>     /* for inline access to atlasOff */

PETSC_EXTERN PetscBool DMRegisterAllCalled;
PETSC_EXTERN PetscErrorCode DMRegisterAll(void);
typedef PetscErrorCode (*NullSpaceFunc)(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullSpace);

typedef struct _PetscHashAuxKey
{
  DMLabel  label;
  PetscInt value;
  PetscInt part;
} PetscHashAuxKey;

#define PetscHashAuxKeyHash(key) PetscHashCombine(PetscHashCombine(PetscHashPointer((key).label),PetscHashInt((key).value)),PetscHashInt((key).part))

#define PetscHashAuxKeyEqual(k1,k2) (((k1).label == (k2).label) ? (((k1).value == (k2).value) ? ((k1).part == (k2).part) : 0) : 0)

PETSC_HASH_MAP(HMapAux, PetscHashAuxKey, Vec, PetscHashAuxKeyHash, PetscHashAuxKeyEqual, NULL)

struct _n_DMGeneratorFunctionList {
  PetscErrorCode (*generate)(DM, PetscBool, DM *);
  PetscErrorCode (*refine)(DM, PetscReal *, DM *);
  PetscErrorCode (*adapt)(DM, Vec, DMLabel, DMLabel, DM *);
  char            *name;
  PetscInt         dim;
  DMGeneratorFunctionList next;
};

typedef struct _DMOps *DMOps;
struct _DMOps {
  PetscErrorCode (*view)(DM,PetscViewer);
  PetscErrorCode (*load)(DM,PetscViewer);
  PetscErrorCode (*clone)(DM,DM*);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,DM);
  PetscErrorCode (*setup)(DM);
  PetscErrorCode (*createlocalsection)(DM);
  PetscErrorCode (*createdefaultconstraints)(DM);
  PetscErrorCode (*createglobalvector)(DM,Vec*);
  PetscErrorCode (*createlocalvector)(DM,Vec*);
  PetscErrorCode (*getlocaltoglobalmapping)(DM);
  PetscErrorCode (*createfieldis)(DM,PetscInt*,char***,IS**);
  PetscErrorCode (*createcoordinatedm)(DM,DM*);
  PetscErrorCode (*createcoordinatefield)(DM,DMField*);

  PetscErrorCode (*getcoloring)(DM,ISColoringType,ISColoring*);
  PetscErrorCode (*creatematrix)(DM, Mat*);
  PetscErrorCode (*createinterpolation)(DM,DM,Mat*,Vec*);
  PetscErrorCode (*createrestriction)(DM,DM,Mat*);
  PetscErrorCode (*createmassmatrix)(DM,DM,Mat*);
  PetscErrorCode (*createmassmatrixlumped)(DM,Vec*);
  PetscErrorCode (*hascreateinjection)(DM,PetscBool*);
  PetscErrorCode (*createinjection)(DM,DM,Mat*);

  PetscErrorCode (*refine)(DM,MPI_Comm,DM*);
  PetscErrorCode (*coarsen)(DM,MPI_Comm,DM*);
  PetscErrorCode (*refinehierarchy)(DM,PetscInt,DM*);
  PetscErrorCode (*coarsenhierarchy)(DM,PetscInt,DM*);
  PetscErrorCode (*extrude)(DM,PetscInt,DM*);

  PetscErrorCode (*globaltolocalbegin)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*globaltolocalend)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*localtoglobalbegin)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*localtoglobalend)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*localtolocalbegin)(DM,Vec,InsertMode,Vec);
  PetscErrorCode (*localtolocalend)(DM,Vec,InsertMode,Vec);

  PetscErrorCode (*destroy)(DM);

  PetscErrorCode (*computevariablebounds)(DM,Vec,Vec);

  PetscErrorCode (*createsubdm)(DM,PetscInt,const PetscInt*,IS*,DM*);
  PetscErrorCode (*createsuperdm)(DM*,PetscInt,IS**,DM*);
  PetscErrorCode (*createfielddecomposition)(DM,PetscInt*,char***,IS**,DM**);
  PetscErrorCode (*createdomaindecomposition)(DM,PetscInt*,char***,IS**,IS**,DM**);
  PetscErrorCode (*createddscatters)(DM,PetscInt,DM*,VecScatter**,VecScatter**,VecScatter**);

  PetscErrorCode (*getdimpoints)(DM,PetscInt,PetscInt*,PetscInt*);
  PetscErrorCode (*locatepoints)(DM,Vec,DMPointLocationType,PetscSF);
  PetscErrorCode (*getneighbors)(DM,PetscInt*,const PetscMPIInt**);
  PetscErrorCode (*getboundingbox)(DM,PetscReal*,PetscReal*);
  PetscErrorCode (*getlocalboundingbox)(DM,PetscReal*,PetscReal*);
  PetscErrorCode (*locatepointssubdomain)(DM,Vec,PetscMPIInt**);

  PetscErrorCode (*projectfunctionlocal)(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,InsertMode,Vec);
  PetscErrorCode (*projectfunctionlabellocal)(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,InsertMode,Vec);
  PetscErrorCode (*projectfieldlocal)(DM,PetscReal,Vec,void(**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
  PetscErrorCode (*projectfieldlabellocal)(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Vec,void(**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
  PetscErrorCode (*projectbdfieldlabellocal)(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Vec,void(**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
  PetscErrorCode (*computel2diff)(DM,PetscReal,PetscErrorCode(**)(PetscInt, PetscReal,const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, PetscReal *);
  PetscErrorCode (*computel2gradientdiff)(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal [],const PetscReal[],PetscInt, PetscScalar *,void *),void **,Vec,const PetscReal[],PetscReal *);
  PetscErrorCode (*computel2fielddiff)(DM,PetscReal,PetscErrorCode(**)(PetscInt, PetscReal,const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, PetscReal *);

  PetscErrorCode (*getcompatibility)(DM,DM,PetscBool*,PetscBool*);
};

PETSC_EXTERN PetscErrorCode DMLocalizeCoordinate_Internal(DM, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMLocalizeCoordinateReal_Internal(DM, PetscInt, const PetscReal[], const PetscReal[], PetscReal[]);
PETSC_EXTERN PetscErrorCode DMLocalizeAddCoordinate_Internal(DM, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]);

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

typedef struct _DMSubDomainHookLink *DMSubDomainHookLink;
struct _DMSubDomainHookLink {
  PetscErrorCode (*ddhook)(DM,DM,void*);
  PetscErrorCode (*restricthook)(DM,VecScatter,VecScatter,DM,void*);
  void *ctx;
  DMSubDomainHookLink next;
};

typedef struct _DMGlobalToLocalHookLink *DMGlobalToLocalHookLink;
struct _DMGlobalToLocalHookLink {
  PetscErrorCode (*beginhook)(DM,Vec,InsertMode,Vec,void*);
  PetscErrorCode (*endhook)(DM,Vec,InsertMode,Vec,void*);
  void *ctx;
  DMGlobalToLocalHookLink next;
};

typedef struct _DMLocalToGlobalHookLink *DMLocalToGlobalHookLink;
struct _DMLocalToGlobalHookLink {
  PetscErrorCode (*beginhook)(DM,Vec,InsertMode,Vec,void*);
  PetscErrorCode (*endhook)(DM,Vec,InsertMode,Vec,void*);
  void *ctx;
  DMLocalToGlobalHookLink next;
};

typedef enum {DMVEC_STATUS_IN,DMVEC_STATUS_OUT} DMVecStatus;
typedef struct _DMNamedVecLink *DMNamedVecLink;
struct _DMNamedVecLink {
  Vec X;
  char *name;
  DMVecStatus status;
  DMNamedVecLink next;
};

typedef struct _DMWorkLink *DMWorkLink;
struct _DMWorkLink {
  size_t     bytes;
  void       *mem;
  DMWorkLink next;
};

#define DM_MAX_WORK_VECTORS 100 /* work vectors available to users  via DMGetGlobalVector(), DMGetLocalVector() */

struct _n_DMLabelLink {
  DMLabel              label;
  PetscBool            output;
  struct _n_DMLabelLink *next;
};
typedef struct _n_DMLabelLink *DMLabelLink;

typedef struct _n_Boundary *DMBoundary;

struct _n_Boundary {
  DSBoundary  dsboundary;
  DMLabel     label;
  DMBoundary  next;
};

typedef struct _n_Field {
  PetscObject disc;         /* Field discretization, or a PetscContainer with the field name */
  DMLabel     label;        /* Label defining the domain of definition of the field */
  PetscBool   adjacency[2]; /* Flags for defining variable influence (adjacency) for each field [use cone() or support() first, use the transitive closure] */
  PetscBool   avoidTensor;  /* Flag to avoid defining field over tensor cells */
} RegionField;

typedef struct _n_Space {
  PetscDS ds;     /* Approximation space in this domain */
  DMLabel label;  /* Label defining the domain of definition of the discretization */
  IS      fields; /* Map from DS field numbers to original field numbers in the DM */
} DMSpace;

struct _p_UniversalLabel {
  DMLabel    label;   /* The universal label */
  PetscInt   Nl;      /* Number of labels encoded */
  char     **names;   /* The label names */
  PetscInt  *indices; /* The original indices in the input DM */
  PetscInt   Nv;      /* Total number of values in all the labels */
  PetscInt  *bits;    /* Starting bit for values of each label */
  PetscInt  *masks;   /* Masks to pull out label value bits for each label */
  PetscInt  *offsets; /* Starting offset for label values for each label */
  PetscInt  *values;  /* Original label values before renumbering */
};

PETSC_INTERN PetscErrorCode DMDestroyLabelLinkList_Internal(DM);

#define MAXDMMONITORS 5

struct _p_DM {
  PETSCHEADER(struct _DMOps);
  Vec                     localin[DM_MAX_WORK_VECTORS],localout[DM_MAX_WORK_VECTORS];
  Vec                     globalin[DM_MAX_WORK_VECTORS],globalout[DM_MAX_WORK_VECTORS];
  DMNamedVecLink          namedglobal;
  DMNamedVecLink          namedlocal;
  DMWorkLink              workin,workout;
  DMLabelLink             labels;            /* Linked list of labels */
  DMLabel                 depthLabel;        /* Optimized access to depth label */
  DMLabel                 celltypeLabel;     /* Optimized access to celltype label */
  void                    *ctx;    /* a user context */
  PetscErrorCode          (*ctxdestroy)(void**);
  ISColoringType          coloringtype;
  MatFDColoring           fd;
  VecType                 vectype;  /* type of vector created with DMCreateLocalVector() and DMCreateGlobalVector() */
  MatType                 mattype;  /* type of matrix created with DMCreateMatrix() */
  PetscInt                bind_below; /* Local size threshold (in entries/rows) below which Vec/Mat objects are bound to CPU */
  PetscInt                bs;
  ISLocalToGlobalMapping  ltogmap;
  PetscBool               prealloc_skip; // Flag indicating the DMCreateMatrix() should not preallocate (only set sizes and local-to-global)
  PetscBool               prealloc_only; /* Flag indicating the DMCreateMatrix() should only preallocate, not fill the matrix */
  PetscBool               structure_only; /* Flag indicating the DMCreateMatrix() create matrix structure without values */
  PetscInt                levelup,leveldown;  /* if the DM has been obtained by refining (or coarsening) this indicates how many times that process has been used to generate this DM */
  PetscBool               setupcalled;        /* Indicates that the DM has been set up, methods that modify a DM such that a fresh setup is required should reset this flag */
  PetscBool               setfromoptionscalled;
  void                    *data;
  /* Hierarchy / Submeshes */
  DM                      coarseMesh;
  DM                      fineMesh;
  DMCoarsenHookLink       coarsenhook; /* For transfering auxiliary problem data to coarser grids */
  DMRefineHookLink        refinehook;
  DMSubDomainHookLink     subdomainhook;
  DMGlobalToLocalHookLink gtolhook;
  DMLocalToGlobalHookLink ltoghook;
  /* Topology */
  PetscInt                dim;                  /* The topological dimension */
  /* Auxiliary data */
  PetscHMapAux            auxData;              /* Auxiliary DM and Vec for region denoted by the key */
  /* Flexible communication */
  PetscSF                 sfMigration;          /* SF for point distribution created during distribution */
  PetscSF                 sf;                   /* SF for parallel point overlap */
  PetscSF                 sectionSF;            /* SF for parallel dof overlap using section */
  PetscSF                 sfNatural;            /* SF mapping to the "natural" ordering */
  PetscBool               useNatural;           /* Create the natural SF */
  /* Allows a non-standard data layout */
  PetscBool               adjacency[2];         /* [use cone() or support() first, use the transitive closure] */
  PetscSection            localSection;         /* Layout for local vectors */
  PetscSection            globalSection;        /* Layout for global vectors */
  PetscLayout             map;
  /* Constraints */
  struct {
    PetscSection section;
    Mat          mat;
    Vec          bias;
  } defaultConstraint;
  /* Basis transformation */
  DM                      transformDM;          /* Layout for basis transformation */
  Vec                     transform;            /* Basis transformation matrices */
  void                   *transformCtx;         /* Basis transformation context */
  PetscErrorCode        (*transformSetUp)(DM, void *);
  PetscErrorCode        (*transformDestroy)(DM, void *);
  PetscErrorCode        (*transformGetMatrix)(DM, const PetscReal[], PetscBool, const PetscScalar **, void *);
  /* Coordinates */
  PetscInt                dimEmbed;             /* The dimension of the embedding space */
  DM                      coordinateDM;         /* Layout for coordinates */
  Vec                     coordinates;          /* Coordinate values in global vector */
  Vec                     coordinatesLocal;     /* Coordinate values in local  vector */
  PetscBool               periodic;             /* Is the DM periodic? */
  DMField                 coordinateField;      /* Coordinates as an abstract field */
  PetscReal              *L, *maxCell;          /* Size of periodic box and max cell size for determining periodicity */
  DMBoundaryType         *bdtype;               /* Indicates type of topological boundary */
  /* Null spaces -- of course I should make this have a variable number of fields */
  NullSpaceFunc           nullspaceConstructors[10];
  NullSpaceFunc           nearnullspaceConstructors[10];
  /* Fields are represented by objects */
  PetscInt                Nf;                   /* Number of fields defined on the total domain */
  RegionField            *fields;               /* Array of discretization fields with regions of validity */
  DMBoundary              boundary;             /* List of boundary conditions */
  PetscInt                Nds;                  /* Number of discrete systems defined on the total domain */
  DMSpace                *probs;                /* Array of discrete systems */
  /* Output structures */
  DM                      dmBC;                 /* The DM with boundary conditions in the global DM */
  PetscInt                outputSequenceNum;    /* The current sequence number for output */
  PetscReal               outputSequenceVal;    /* The current sequence value for output */
  PetscErrorCode        (*monitor[MAXDMMONITORS])(DM, void *);
  PetscErrorCode        (*monitordestroy[MAXDMMONITORS])(void **);
  void                   *monitorcontext[MAXDMMONITORS];
  PetscInt                numbermonitors;

  PetscObject             dmksp,dmsnes,dmts;
#ifdef PETSC_HAVE_LIBCEED
  Ceed                    ceed;                 /* LibCEED context */
  CeedElemRestriction     ceedERestrict;        /* Map from the local vector (Lvector) to the cells (Evector) */
#endif
};

PETSC_EXTERN PetscLogEvent DM_Convert;
PETSC_EXTERN PetscLogEvent DM_GlobalToLocal;
PETSC_EXTERN PetscLogEvent DM_LocalToGlobal;
PETSC_EXTERN PetscLogEvent DM_LocatePoints;
PETSC_EXTERN PetscLogEvent DM_Coarsen;
PETSC_EXTERN PetscLogEvent DM_Refine;
PETSC_EXTERN PetscLogEvent DM_CreateInterpolation;
PETSC_EXTERN PetscLogEvent DM_CreateRestriction;
PETSC_EXTERN PetscLogEvent DM_CreateInjection;
PETSC_EXTERN PetscLogEvent DM_CreateMatrix;
PETSC_EXTERN PetscLogEvent DM_CreateMassMatrix;
PETSC_EXTERN PetscLogEvent DM_Load;
PETSC_EXTERN PetscLogEvent DM_AdaptInterpolator;

PETSC_EXTERN PetscErrorCode DMCreateGlobalVector_Section_Private(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMCreateLocalVector_Section_Private(DM,Vec*);

PETSC_EXTERN PetscErrorCode DMView_GLVis(DM,PetscViewer,PetscErrorCode(*)(DM,PetscViewer));

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

#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscErrorCode DMSequenceLoad_HDF5_Internal(DM, const char *, PetscInt, PetscScalar *, PetscViewer);
#endif

static inline PetscErrorCode DMGetLocalOffset_Private(DM dm, PetscInt point, PetscInt *start, PetscInt *end)
{
  PetscFunctionBeginHot;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt dof;

    *start = *end = 0; /* Silence overzealous compiler warning */
    PetscCheck(dm->localSection,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a local section, see DMSetLocalSection()");
    PetscCall(PetscSectionGetOffset(dm->localSection, point, start));
    PetscCall(PetscSectionGetDof(dm->localSection, point, &dof));
    *end = *start + dof;
  }
#else
  {
    const PetscSection s = dm->localSection;
    *start = s->atlasOff[point - s->pStart];
    *end   = *start + s->atlasDof[point - s->pStart];
  }
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMGetLocalFieldOffset_Private(DM dm, PetscInt point, PetscInt field, PetscInt *start, PetscInt *end)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt       dof;
    *start = *end = 0; /* Silence overzealous compiler warning */
    PetscCheck(dm->localSection,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a local section, see DMSetLocalSection()");
    PetscCall(PetscSectionGetFieldOffset(dm->localSection, point, field, start));
    PetscCall(PetscSectionGetFieldDof(dm->localSection, point, field, &dof));
    *end = *start + dof;
  }
#else
  {
    const PetscSection s = dm->localSection->field[field];
    *start = s->atlasOff[point - s->pStart];
    *end   = *start + s->atlasDof[point - s->pStart];
  }
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMGetGlobalOffset_Private(DM dm, PetscInt point, PetscInt *start, PetscInt *end)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt       dof,cdof;
    *start = *end = 0; /* Silence overzealous compiler warning */
    PetscCheck(dm->localSection,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a local section, see DMSetLocalSection()");
    PetscCheck(dm->globalSection,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a global section. It will be created automatically by DMGetGlobalSection()");
    PetscCall(PetscSectionGetOffset(dm->globalSection, point, start));
    PetscCall(PetscSectionGetDof(dm->globalSection, point, &dof));
    PetscCall(PetscSectionGetConstraintDof(dm->globalSection, point, &cdof));
    *end = *start + dof - cdof + (dof < 0 ? 1 : 0);
  }
#else
  {
    const PetscSection s    = dm->globalSection;
    const PetscInt     dof  = s->atlasDof[point - s->pStart];
    const PetscInt     cdof = s->bc ? s->bc->atlasDof[point - s->bc->pStart] : 0;
    *start = s->atlasOff[point - s->pStart];
    *end   = *start + dof - cdof + (dof < 0 ? 1 : 0);
  }
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMGetGlobalFieldOffset_Private(DM dm, PetscInt point, PetscInt field, PetscInt *start, PetscInt *end)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt loff, lfoff, fdof, fcdof, ffcdof, f;
    *start = *end = 0; /* Silence overzealous compiler warning */
    PetscCheck(dm->localSection,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a local section, see DMSetLocalSection()");
    PetscCheck(dm->globalSection,PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "DM must have a global section. It will be crated automatically by DMGetGlobalSection()");
    PetscCall(PetscSectionGetOffset(dm->globalSection, point, start));
    PetscCall(PetscSectionGetOffset(dm->localSection, point, &loff));
    PetscCall(PetscSectionGetFieldOffset(dm->localSection, point, field, &lfoff));
    PetscCall(PetscSectionGetFieldDof(dm->localSection, point, field, &fdof));
    PetscCall(PetscSectionGetFieldConstraintDof(dm->localSection, point, field, &fcdof));
    *start = *start < 0 ? *start - (lfoff-loff) : *start + lfoff-loff;
    for (f = 0; f < field; ++f) {
      PetscCall(PetscSectionGetFieldConstraintDof(dm->localSection, point, f, &ffcdof));
      *start = *start < 0 ? *start + ffcdof : *start - ffcdof;
    }
    *end   = *start < 0 ? *start - (fdof-fcdof) : *start + fdof-fcdof;
  }
#else
  {
    const PetscSection s     = dm->localSection;
    const PetscSection fs    = dm->localSection->field[field];
    const PetscSection gs    = dm->globalSection;
    const PetscInt     loff  = s->atlasOff[point - s->pStart];
    const PetscInt     goff  = gs->atlasOff[point - s->pStart];
    const PetscInt     lfoff = fs->atlasOff[point - s->pStart];
    const PetscInt     fdof  = fs->atlasDof[point - s->pStart];
    const PetscInt     fcdof = fs->bc ? fs->bc->atlasDof[point - fs->bc->pStart] : 0;
    PetscInt           ffcdof = 0, f;

    for (f = 0; f < field; ++f) {
      const PetscSection ffs = dm->localSection->field[f];
      ffcdof += ffs->bc ? ffs->bc->atlasDof[point - ffs->bc->pStart] : 0;
    }
    *start = goff + (goff < 0 ? loff-lfoff + ffcdof : lfoff-loff - ffcdof);
    *end   = *start < 0 ? *start - (fdof-fcdof) : *start + fdof-fcdof;
  }
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMGetBasisTransformDM_Internal(DM, DM *);
PETSC_EXTERN PetscErrorCode DMGetBasisTransformVec_Internal(DM, Vec *);
PETSC_INTERN PetscErrorCode DMConstructBasisTransform_Internal(DM);

PETSC_INTERN PetscErrorCode DMGetLocalBoundingIndices_DMDA(DM, PetscReal[], PetscReal[]);
PETSC_INTERN PetscErrorCode DMSetField_Internal(DM, PetscInt, DMLabel, PetscObject);

PETSC_INTERN PetscErrorCode DMSetLabelValue_Fast(DM, DMLabel*, const char[], PetscInt, PetscInt);

PETSC_INTERN PetscErrorCode DMCompleteBCLabels_Internal(DM dm);
PETSC_EXTERN PetscErrorCode DMUniversalLabelCreate(DM, DMUniversalLabel *);
PETSC_EXTERN PetscErrorCode DMUniversalLabelDestroy(DMUniversalLabel *);
PETSC_EXTERN PetscErrorCode DMUniversalLabelGetLabel(DMUniversalLabel, DMLabel *);
PETSC_EXTERN PetscErrorCode DMUniversalLabelCreateLabels(DMUniversalLabel, PetscBool, DM);
PETSC_EXTERN PetscErrorCode DMUniversalLabelSetLabelValue(DMUniversalLabel, DM, PetscBool, PetscInt, PetscInt);
PETSC_INTERN PetscInt PetscGCD(PetscInt a, PetscInt b);

#endif
