#if !defined(PETSCSFIMPL_H)
#define PETSCSFIMPL_H

#include <petscsf.h>
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_EXTERN PetscLogEvent PETSCSF_SetGraph;
PETSC_EXTERN PetscLogEvent PETSCSF_SetUp;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastAndOpBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastAndOpEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_ReduceBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_ReduceEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_FetchAndOpBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_FetchAndOpEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_EmbedSF;
PETSC_EXTERN PetscLogEvent PETSCSF_DistSect;
PETSC_EXTERN PetscLogEvent PETSCSF_SectSF;
PETSC_EXTERN PetscLogEvent PETSCSF_RemoteOff;

struct _PetscSFOps {
  PetscErrorCode (*Reset)(PetscSF);
  PetscErrorCode (*Destroy)(PetscSF);
  PetscErrorCode (*SetUp)(PetscSF);
  PetscErrorCode (*SetFromOptions)(PetscOptionItems*,PetscSF);
  PetscErrorCode (*View)(PetscSF,PetscViewer);
  PetscErrorCode (*Duplicate)(PetscSF,PetscSFDuplicateOption,PetscSF);
  PetscErrorCode (*BcastAndOpBegin)(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
  PetscErrorCode (*BcastAndOpEnd)(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
  PetscErrorCode (*BcastToZero)(PetscSF,MPI_Datatype,const void*,void*); /* For interal use only */
  PetscErrorCode (*ReduceBegin)(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
  PetscErrorCode (*ReduceEnd)(PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
  PetscErrorCode (*FetchAndOpBegin)(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op);
  PetscErrorCode (*FetchAndOpEnd)(PetscSF,MPI_Datatype,void*,const void *,void *,MPI_Op);
  PetscErrorCode (*GetRootRanks)(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
  PetscErrorCode (*GetLeafRanks)(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
  PetscErrorCode (*CreateLocalSF)(PetscSF,PetscSF*);
  PetscErrorCode (*GetGraph)(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**);
  PetscErrorCode (*CreateEmbeddedSF)(PetscSF,PetscInt,const PetscInt*,PetscSF*);
  PetscErrorCode (*CreateEmbeddedLeafSF)(PetscSF,PetscInt,const PetscInt*,PetscSF*);
};

typedef struct _n_PetscSFPackOpt *PetscSFPackOpt;

struct _p_PetscSF {
  PETSCHEADER(struct _PetscSFOps);
  PetscInt        nroots;          /* Number of root vertices on current process (candidates for incoming edges) */
  PetscInt        nleaves;         /* Number of leaf vertices on current process (this process specifies a root for each leaf) */
  PetscInt        *mine;           /* Location of leaves in leafdata arrays provided to the communication routines */
  PetscInt        *mine_alloc;
  PetscInt        minleaf,maxleaf;
  PetscSFNode     *remote;         /* Remote references to roots for each local leaf */
  PetscSFNode     *remote_alloc;
  PetscInt        nranks;          /* Number of ranks owning roots connected to my leaves */
  PetscInt        ndranks;         /* Number of ranks in distinguished group holding roots connected to my leaves */
  PetscMPIInt     *ranks;          /* List of ranks referenced by "remote" */
  PetscInt        *roffset;        /* Array of length nranks+1, offset in rmine/rremote for each rank */
  PetscInt        *rmine;          /* Concatenated array holding local indices referencing each remote rank */
  PetscInt        *rremote;        /* Concatenated array holding remote indices referenced for each remote rank */
  PetscBool       degreeknown;     /* The degree is currently known, do not have to recompute */
  PetscInt        *degree;         /* Degree of each of my root vertices */
  PetscInt        *degreetmp;      /* Temporary local array for computing degree */
  PetscBool       rankorder;       /* Sort ranks for gather and scatter operations */
  MPI_Group       ingroup;         /* Group of processes connected to my roots */
  MPI_Group       outgroup;        /* Group of processes connected to my leaves */
  PetscSF         multi;           /* Internal graph used to implement gather and scatter operations */
  PetscBool       graphset;        /* Flag indicating that the graph has been set, required before calling communication routines */
  PetscBool       setupcalled;     /* Type and communication structures have been set up */
  PetscSFPackOpt  leafpackopt;     /* Optimization plans to (un)pack leaves based on patterns in rmine[]. NULL for no optimization */
  PetscSFPackOpt  selfleafpackopt; /* Optimization plans to (un)pack leaves connected to local roots */

  PetscSFPattern  pattern;         /* Pattern of the graph */
  PetscLayout     map;             /* Layout of leaves over all processes when building a patterned graph */

  void *data;                      /* Pointer to implementation */
};

PETSC_EXTERN PetscBool PetscSFRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscSFRegisterAll(void);

PETSC_INTERN PetscErrorCode PetscSFCreateLocalSF_Private(PetscSF,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFBcastToZero_Private(PetscSF,MPI_Datatype,const void*,void*);

PETSC_EXTERN PetscErrorCode MPIPetsc_Type_unwrap(MPI_Datatype,MPI_Datatype*,PetscBool*);
PETSC_EXTERN PetscErrorCode MPIPetsc_Type_compare(MPI_Datatype,MPI_Datatype,PetscBool*);
PETSC_EXTERN PetscErrorCode MPIPetsc_Type_compare_contig(MPI_Datatype,MPI_Datatype,PetscInt*);

#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
#define MPIU_Iscatter(a,b,c,d,e,f,g,h,req)     MPI_Iscatter(a,b,c,d,e,f,g,h,req)
#define MPIU_Iscatterv(a,b,c,d,e,f,g,h,i,req)  MPI_Iscatterv(a,b,c,d,e,f,g,h,i,req)
#define MPIU_Igather(a,b,c,d,e,f,g,h,req)      MPI_Igather(a,b,c,d,e,f,g,h,req)
#define MPIU_Igatherv(a,b,c,d,e,f,g,h,i,req)   MPI_Igatherv(a,b,c,d,e,f,g,h,i,req)
#define MPIU_Iallgather(a,b,c,d,e,f,g,req)     MPI_Iallgather(a,b,c,d,e,f,g,req)
#define MPIU_Iallgatherv(a,b,c,d,e,f,g,h,req)  MPI_Iallgatherv(a,b,c,d,e,f,g,h,req)
#define MPIU_Ialltoall(a,b,c,d,e,f,g,req)      MPI_Ialltoall(a,b,c,d,e,f,g,req)
#else
/* Ignore req, the MPI_Request argument, and use MPI blocking collectives. One should initialize req
   to MPI_REQUEST_NULL so that one can do MPI_Wait(req,status) no matter the call is blocking or not.
 */
#define MPIU_Iscatter(a,b,c,d,e,f,g,h,req)     MPI_Scatter(a,b,c,d,e,f,g,h)
#define MPIU_Iscatterv(a,b,c,d,e,f,g,h,i,req)  MPI_Scatterv(a,b,c,d,e,f,g,h,i)
#define MPIU_Igather(a,b,c,d,e,f,g,h,req)      MPI_Gather(a,b,c,d,e,f,g,h)
#define MPIU_Igatherv(a,b,c,d,e,f,g,h,i,req)   MPI_Gatherv(a,b,c,d,e,f,g,h,i)
#define MPIU_Iallgather(a,b,c,d,e,f,g,req)     MPI_Allgather(a,b,c,d,e,f,g)
#define MPIU_Iallgatherv(a,b,c,d,e,f,g,h,req)  MPI_Allgatherv(a,b,c,d,e,f,g,h)
#define MPIU_Ialltoall(a,b,c,d,e,f,g,req)      MPI_Alltoall(a,b,c,d,e,f,g)
#endif

#endif
