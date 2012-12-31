#define PETSC_DESIRE_COMPLEX
#include <petsc-private/sfimpl.h> /*I "petscsf.h" I*/

typedef struct _n_PetscSFBasicPack *PetscSFBasicPack;
struct _n_PetscSFBasicPack {
  void (*Pack)(PetscInt,const PetscInt*,const void*,void*);
  void (*UnpackInsert)(PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackAdd)(PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMin)(PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMax)(PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMinloc)(PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMaxloc)(PetscInt,const PetscInt*,void*,const void*);
  void (*FetchAndInsert)(PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndAdd)(PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMin)(PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMax)(PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMinloc)(PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMaxloc)(PetscInt,const PetscInt*,void*,void*);
  MPI_Datatype unit;
  size_t unitbytes;             /* Number of bytes in a unit */
  const void *key;              /* Array used as key for operation */
  char *root;                   /* Packed root data, contiguous by leaf rank */
  char *leaf;                   /* Packed leaf data, contiguous by root rank */
  MPI_Request *requests;        /* Array of root requests followed by leaf requests */
  PetscSFBasicPack next;
};

typedef struct {
  PetscMPIInt tag;
  PetscInt niranks;             /* Number of incoming ranks (ranks accessing my roots) */
  PetscMPIInt *iranks;          /* Array of ranks that reference my roots */
  PetscInt itotal;              /* Total number of graph edges referencing my roots */
  PetscInt *ioffset;            /* Array of length niranks+1 holding offset in irootloc[] for each rank */
  PetscInt *irootloc;           /* Incoming roots referenced by ranks starting at ioffset[rank] */
  PetscSFBasicPack avail;       /* One or more entries per MPI Datatype, lazily constructed */
  PetscSFBasicPack inuse;       /* Buffers being used for transactions that have not yet completed */
} PetscSF_Basic;

#if !defined(PETSC_HAVE_MPI_TYPE_DUP) /* Danger: type is not reference counted; subject to ABA problem */
PETSC_STATIC_INLINE PetscErrorCode MPI_Type_dup(MPI_Datatype datatype,MPI_Datatype *newtype) { *newtype = datatype; return 0; }
#endif

/*
 * MPI_Reduce_local is not really useful because it can't handle sparse data and it vectorizes "in the wrong direction",
 * therefore we pack data types manually. This section defines packing routines for the standard data types.
 */

/* Basic types defining addition */
#define DEF_PackNoInit(type)                                            \
  static void Pack_ ## type(PetscInt n,const PetscInt *idx,const void *unpacked,void *packed) { \
    const type *u = (const type*)unpacked;                              \
    type *p = (type *)packed;                                           \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) p[i] = u[idx[i]];                               \
  }                                                                     \
  static void UnpackInsert_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type *)unpacked;                                         \
    const type *p = (const type *)packed;                               \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) u[idx[i]] = p[i];                               \
  }                                                                     \
  static void UnpackAdd_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type *)unpacked;                                         \
    const type *p = (const type *)packed;                               \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) u[idx[i]] += p[i];                              \
  }                                                                     \
  static void FetchAndInsert_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type *)unpacked;                                         \
    type *p = (type *)packed;                                           \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type t = u[j];                                                    \
      u[j] = p[i];                                                      \
      p[i] = t;                                                         \
    }                                                                   \
  }                                                                     \
  static void FetchAndAdd_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type *)unpacked;                                         \
    type *p = (type *)packed;                                           \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type t = u[j];                                                    \
      u[j] = t + p[i];                                                  \
      p[i] = t;                                                         \
    }                                                                   \
  }
#define DEF_Pack(type) \
  DEF_PackNoInit(type)                                                  \
  static void PackInit_ ## type(PetscSFBasicPack link) {                \
    link->Pack = Pack_ ## type;                                         \
    link->UnpackInsert = UnpackInsert_ ## type;                         \
    link->UnpackAdd = UnpackAdd_ ## type;                               \
    link->FetchAndInsert = FetchAndInsert_ ## type;                     \
    link->FetchAndAdd = FetchAndAdd_ ## type;                           \
    link->unitbytes = sizeof(type);                                     \
  }
/* Comparable types */
#define DEF_PackCmp(type)                                               \
  DEF_PackNoInit(type)                                                  \
  static void UnpackMax_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type *)unpacked;                                         \
    const type *p = (const type *)packed;                               \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = PetscMax(v,p[i]);                                     \
    }                                                                   \
  }                                                                     \
  static void UnpackMin_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type *)unpacked;                                         \
    const type *p = (const type *)packed;                               \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = PetscMin(v,p[i]);                                     \
    }                                                                   \
  }                                                                     \
  static void FetchAndMax_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type *)unpacked;                                         \
    type *p = (type *)packed;                                           \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = PetscMax(v,p[i]);                                          \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void FetchAndMin_ ## type(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type *)unpacked;                                         \
    type *p = (type *)packed;                                           \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = PetscMin(v,p[i]);                                          \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void PackInit_ ## type(PetscSFBasicPack link) {                \
    link->Pack = Pack_ ## type;                                         \
    link->UnpackInsert = UnpackInsert_ ## type;                         \
    link->UnpackAdd = UnpackAdd_ ## type;                               \
    link->UnpackMax = UnpackMax_ ## type;                               \
    link->UnpackMin = UnpackMin_ ## type;                               \
    link->FetchAndInsert = FetchAndInsert_ ## type;                     \
    link->FetchAndAdd = FetchAndAdd_ ## type;                           \
    link->FetchAndMax = FetchAndMax_ ## type;                           \
    link->FetchAndMin = FetchAndMin_ ## type;                           \
    link->unitbytes = sizeof(type);                                     \
  }

/* Pair types */
#define PairType(type1,type2) _pairtype_ ## type1 ## _ ## type2
#define DEF_UnpackXloc(type1,type2,locname,op)                              \
  static void Unpack##locname##loc_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2) *)unpacked;       \
    const PairType(type1,type2) *p = (const PairType(type1,type2) *)packed; \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      if (u[j].a op p[i].a) {                                           \
        u[j].a = p[i].a;                                                \
        u[j].b = p[i].b;                                                \
      } else if (u[j].a == p[i].a) {                                    \
        u[j].b = PetscMin(u[j].b,p[i].b);                               \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  static void FetchAnd##locname##loc_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2) *)unpacked;       \
    PairType(type1,type2) *p = (PairType(type1,type2) *)packed;         \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      PairType(type1,type2) v;                                          \
      v.a = u[j].a;                                                     \
      v.b = u[j].b;                                                     \
      if (u[j].a op p[i].a) {                                           \
        u[j].a = p[i].a;                                                \
        u[j].b = p[i].b;                                                \
      } else if (u[j].a == p[i].a) {                                    \
        u[j].b = PetscMin(u[j].b,p[i].b);                               \
      }                                                                 \
      p[i].a = v.a;                                                     \
      p[i].b = v.b;                                                     \
    }                                                                   \
  }
#define DEF_PackPair(type1,type2)                                       \
  typedef struct {type1 a; type2 b;} PairType(type1,type2);             \
  static void Pack_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,const void *unpacked,void *packed) { \
    const PairType(type1,type2) *u = (const PairType(type1,type2) *)unpacked; \
    PairType(type1,type2) *p = (PairType(type1,type2) *)packed;         \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      p[i].a = u[idx[i]].a;                                             \
      p[i].b = u[idx[i]].b;                                             \
    }                                                                   \
  }                                                                     \
  static void UnpackInsert_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2) *)unpacked;       \
    const PairType(type1,type2) *p = (const PairType(type1,type2) *)packed; \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      u[idx[i]].a = p[i].a;                                             \
      u[idx[i]].b = p[i].b;                                             \
    }                                                                   \
  }                                                                     \
  static void UnpackAdd_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,void *unpacked,const void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2) *)unpacked;       \
    const PairType(type1,type2) *p = (const PairType(type1,type2) *)packed; \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      u[idx[i]].a += p[i].a;                                            \
      u[idx[i]].b += p[i].b;                                            \
    }                                                                   \
  }                                                                     \
  static void FetchAndInsert_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2) *)unpacked;       \
    PairType(type1,type2) *p = (PairType(type1,type2) *)packed;         \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      PairType(type1,type2) v;                                          \
      v.a = u[j].a;                                                     \
      v.b = u[j].b;                                                     \
      u[j].a = p[i].a;                                                  \
      u[j].b = p[i].b;                                                  \
      p[i].a = v.a;                                                     \
      p[i].b = v.b;                                                     \
    }                                                                   \
  }                                                                     \
  static void FetchAndAdd_ ## type1 ## _ ## type2(PetscInt n,const PetscInt *idx,void *unpacked,void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2) *)unpacked;       \
    PairType(type1,type2) *p = (PairType(type1,type2) *)packed;         \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      PairType(type1,type2) v;                                          \
      v.a = u[j].a;                                                     \
      v.b = u[j].b;                                                     \
      u[j].a = v.a + p[i].a;                                            \
      u[j].b = v.b + p[i].b;                                            \
      p[i].a = v.a;                                                     \
      p[i].b = v.b;                                                     \
    }                                                                   \
  }                                                                     \
  DEF_UnpackXloc(type1,type2,Max,>)                                     \
  DEF_UnpackXloc(type1,type2,Min,<)                                     \
  static void PackInit_ ## type1 ## _ ## type2(PetscSFBasicPack link) { \
    link->Pack = Pack_ ## type1 ## _ ## type2;                          \
    link->UnpackInsert = UnpackInsert_ ## type1 ## _ ## type2;          \
    link->UnpackAdd = UnpackAdd_ ## type1 ## _ ## type2;                \
    link->UnpackMaxloc = UnpackMaxloc_ ## type1 ## _ ## type2;          \
    link->UnpackMinloc = UnpackMinloc_ ## type1 ## _ ## type2;          \
    link->FetchAndInsert = FetchAndInsert_ ## type1 ## _ ## type2;      \
    link->FetchAndAdd = FetchAndAdd_ ## type1 ## _ ## type2;            \
    link->FetchAndMaxloc = FetchAndMaxloc_ ## type1 ## _ ## type2;      \
    link->FetchAndMinloc = FetchAndMinloc_ ## type1 ## _ ## type2;      \
    link->unitbytes = sizeof(PairType(type1,type2));                    \
  }

DEF_PackCmp(int)
DEF_PackCmp(PetscInt)
DEF_PackCmp(PetscReal)
#if defined(PETSC_HAVE_COMPLEX)
DEF_Pack(PetscComplex)
#endif
DEF_PackPair(int,int)
DEF_PackPair(PetscInt,PetscInt)

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetUp_Basic"
static PetscErrorCode PetscSFSetUp_Basic(PetscSF sf)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;
  PetscMPIInt i,size,nsends,nrecvs,*rflags,*rlengths,*ilengths;
  PetscInt *pilengths;
  MPI_Comm comm;
  MPI_Request *rootreqs,*leafreqs;

  PetscFunctionBegin;
  comm = ((PetscObject)sf)->comm;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)sf,&bas->tag);CHKERRQ(ierr);
  /*
   * Inform roots about their leaves
   */
  ierr = PetscMalloc2(size,PetscMPIInt,&rflags,size,PetscMPIInt,&rlengths);CHKERRQ(ierr);
  ierr = PetscMemzero(rflags,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(rlengths,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  /* Determine number, sending ranks, and length of incoming  */
  nsends = PetscMPIIntCast(sf->nranks);
  for (i=0; i<nsends; i++) {
    rflags[sf->ranks[i]] = 1;                        /* Ranks that my leaves reference */
    rlengths[sf->ranks[i]] = sf->roffset[i+1] - sf->roffset[i]; /* Number of roots referenced by my leaves; for rank sf->ranks[i] */
  }
  ierr = PetscGatherNumberOfMessages(comm,rflags,PETSC_NULL,&nrecvs);CHKERRQ(ierr);
  bas->niranks = nrecvs;
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,rlengths,&bas->iranks,&ilengths);CHKERRQ(ierr);
  ierr = PetscFree2(rflags,rlengths);CHKERRQ(ierr);
#if defined(PETSC_USE_64BIT_INDICES)
  ierr = PetscMalloc(nrecvs*sizeof(PetscInt),&pilengths);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    pilengths[i] = PetscMPIIntCast(ilengths[i]);
  }
  ierr = PetscFree(ilengths);CHKERRQ(ierr);
#else
  pilengths = ilengths;
#endif

  for (i=0,bas->itotal=0; i<bas->niranks; i++) bas->itotal += pilengths[i];
  ierr = PetscMalloc2(bas->niranks+1,PetscInt,&bas->ioffset,bas->itotal,PetscInt,&bas->irootloc);CHKERRQ(ierr);
  ierr = PetscMalloc((bas->niranks+sf->nranks)*sizeof(MPI_Request),&rootreqs);CHKERRQ(ierr);
  leafreqs = rootreqs + bas->niranks;
  bas->ioffset[0] = 0;
  for (i=0; i<bas->niranks; i++) {
    bas->ioffset[i+1] = bas->ioffset[i] + pilengths[i];
    ierr = MPI_Irecv(bas->irootloc+bas->ioffset[i],pilengths[i],MPIU_INT,bas->iranks[i],bas->tag,comm,&rootreqs[i]);CHKERRQ(ierr);
  }
  for (i=0; i<sf->nranks; i++) {
    PetscMPIInt npoints = PetscMPIIntCast(sf->roffset[i+1] - sf->roffset[i]);
    ierr = MPI_Isend(sf->rremote+sf->roffset[i],npoints,MPIU_INT,sf->ranks[i],bas->tag,comm,&leafreqs[i]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(sf->nranks+bas->niranks,rootreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree(pilengths);CHKERRQ(ierr);
  ierr = PetscFree(rootreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicPackTypeSetup"
static PetscErrorCode PetscSFBasicPackTypeSetup(PetscSFBasicPack link,MPI_Datatype unit)
{
  PetscErrorCode ierr;
  PetscBool isInt,isPetscInt,isPetscReal,is2Int,is2PetscInt;
#if defined(PETSC_HAVE_COMPLEX)
  PetscBool isPetscComplex;
#endif

  PetscFunctionBegin;
  ierr = MPIPetsc_Type_compare(unit,MPI_INT,&isInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_INT,&isPetscInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_REAL,&isPetscReal);CHKERRQ(ierr);
#if defined(PETSC_HAVE_COMPLEX)
  ierr = MPIPetsc_Type_compare(unit,MPIU_COMPLEX,&isPetscComplex);CHKERRQ(ierr);
#endif
  ierr = MPIPetsc_Type_compare(unit,MPI_2INT,&is2Int);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_2INT,&is2PetscInt);CHKERRQ(ierr);
  if (isInt) PackInit_int(link);
  else if (isPetscInt) PackInit_PetscInt(link);
  else if (isPetscReal) PackInit_PetscReal(link);
#if defined(PETSC_HAVE_COMPLEX)
  else if (isPetscComplex) PackInit_PetscComplex(link);
#endif
  else if (is2Int) PackInit_int_int(link);
  else if (is2PetscInt) PackInit_PetscInt_PetscInt(link);
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for type");
  ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicPackGetUnpackOp"
static PetscErrorCode PetscSFBasicPackGetUnpackOp(PetscSF sf,PetscSFBasicPack link,MPI_Op op,void (**UnpackOp)(PetscInt,const PetscInt*,void*,const void*))
{
  PetscFunctionBegin;
  *UnpackOp = PETSC_NULL;
  if (op == MPI_REPLACE) *UnpackOp = link->UnpackInsert;
  else if (op == MPI_SUM || op == MPIU_SUM) *UnpackOp = link->UnpackAdd;
  else if (op == MPI_MAX || op == MPIU_MAX) *UnpackOp = link->UnpackMax;
  else if (op == MPI_MIN || op == MPIU_MIN) *UnpackOp = link->UnpackMin;
  else if (op == MPI_MAXLOC) *UnpackOp = link->UnpackMaxloc;
  else if (op == MPI_MINLOC) *UnpackOp = link->UnpackMinloc;
  else SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_SUP,"No support for MPI_Op");
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicPackGetFetchAndOp"
static PetscErrorCode PetscSFBasicPackGetFetchAndOp(PetscSF sf,PetscSFBasicPack link,MPI_Op op,void (**FetchAndOp)(PetscInt,const PetscInt*,void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOp = PETSC_NULL;
  if (op == MPI_REPLACE) *FetchAndOp = link->FetchAndInsert;
  else if (op == MPI_SUM || op == MPIU_SUM) *FetchAndOp = link->FetchAndAdd;
  else if (op == MPI_MAX || op == MPIU_MAX) *FetchAndOp = link->FetchAndMax;
  else if (op == MPI_MIN || op == MPIU_MIN) *FetchAndOp = link->FetchAndMin;
  else if (op == MPI_MAXLOC) *FetchAndOp = link->FetchAndMaxloc;
  else if (op == MPI_MINLOC) *FetchAndOp = link->FetchAndMinloc;
  else SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_SUP,"No support for MPI_Op");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicPackGetReqs"
static PetscErrorCode PetscSFBasicPackGetReqs(PetscSF sf,PetscSFBasicPack link,MPI_Request **rootreqs,MPI_Request **leafreqs)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (rootreqs) *rootreqs = link->requests;
  if (leafreqs) *leafreqs = link->requests + bas->niranks;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicPackWaitall"
static PetscErrorCode PetscSFBasicPackWaitall(PetscSF sf,PetscSFBasicPack link)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Waitall(bas->niranks+sf->nranks,link->requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicGetRootInfo"
static PetscErrorCode PetscSFBasicGetRootInfo(PetscSF sf,PetscInt *nrootranks,const PetscMPIInt **rootranks,const PetscInt **rootoffset,const PetscInt **rootloc)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (nrootranks) *nrootranks = bas->niranks;
  if (rootranks)  *rootranks = bas->iranks;
  if (rootoffset) *rootoffset = bas->ioffset;
  if (rootloc)    *rootloc = bas->irootloc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicGetLeafInfo"
static PetscErrorCode PetscSFBasicGetLeafInfo(PetscSF sf,PetscInt *nleafranks,const PetscMPIInt **leafranks,const PetscInt **leafoffset,const PetscInt **leafloc)
{
  PetscFunctionBegin;
  if (nleafranks) *nleafranks = sf->nranks;
  if (leafranks)  *leafranks = sf->ranks;
  if (leafoffset) *leafoffset = sf->roffset;
  if (leafloc)    *leafloc = sf->rmine;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicGetPack"
static PetscErrorCode PetscSFBasicGetPack(PetscSF sf,MPI_Datatype unit,const void *key,PetscSFBasicPack *mylink)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;
  PetscSFBasicPack link,*p;
  PetscInt nrootranks,nleafranks;
  const PetscInt *rootoffset,*leafoffset;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (p=&bas->avail; (link=*p); p=&link->next) {
    PetscBool match;
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      *p = link->next;          /* Remove from available list */
      goto found;
    }
  }

  /* Create new composite types for each send rank */
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,PETSC_NULL,&rootoffset,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,PETSC_NULL,&leafoffset,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNew(struct _n_PetscSFBasicPack,&link);CHKERRQ(ierr);
  ierr = PetscSFBasicPackTypeSetup(link,unit);CHKERRQ(ierr);
  ierr = PetscMalloc2(rootoffset[nrootranks]*link->unitbytes,char,&link->root,leafoffset[nleafranks]*link->unitbytes,char,&link->leaf);CHKERRQ(ierr);
  ierr = PetscMalloc((nrootranks+nleafranks)*sizeof(MPI_Request),&link->requests);CHKERRQ(ierr);

  found:
  link->key = key;
  link->next = bas->inuse;
  bas->inuse = link;

  *mylink = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicGetPackInUse"
static PetscErrorCode PetscSFBasicGetPackInUse(PetscSF sf,MPI_Datatype unit,const void *key,PetscCopyMode cmode,PetscSFBasicPack *mylink)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;
  PetscSFBasicPack link,*p;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (p=&bas->inuse; (link=*p); p=&link->next) {
    PetscBool match;
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      switch (cmode) {
      case PETSC_OWN_POINTER: *p = link->next; break; /* Remove from inuse list */
      case PETSC_USE_POINTER: break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"invalid cmode");
      }
      *mylink = link;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_ARG_WRONGSTATE,"Could not find pack");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBasicReclaimPack"
static PetscErrorCode PetscSFBasicReclaimPack(PetscSF sf,PetscSFBasicPack *link)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  (*link)->key = PETSC_NULL;
  (*link)->next = bas->avail;
  bas->avail = *link;
  *link = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFSetFromOptions_Basic"
static PetscErrorCode PetscSFSetFromOptions_Basic(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("PetscSF Basic options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReset_Basic"
static PetscErrorCode PetscSFReset_Basic(PetscSF sf)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;
  PetscSFBasicPack link,next;

  PetscFunctionBegin;
  ierr = PetscFree(bas->iranks);CHKERRQ(ierr);
  ierr = PetscFree2(bas->ioffset,bas->irootloc);CHKERRQ(ierr);
  if (bas->inuse) SETERRQ(((PetscObject)sf)->comm,PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  for (link=bas->avail; link; link=next) {
    next = link->next;
    ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);
    ierr = PetscFree2(link->root,link->leaf);CHKERRQ(ierr);
    ierr = PetscFree(link->requests);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  bas->avail = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFDestroy_Basic"
static PetscErrorCode PetscSFDestroy_Basic(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReset_Basic(sf);CHKERRQ(ierr);
  ierr = PetscFree(sf->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFView_Basic"
static PetscErrorCode PetscSFView_Basic(PetscSF sf,PetscViewer viewer)
{
  /* PetscSF_Basic *bas = (PetscSF_Basic*)sf->data; */
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  sort=%s\n",sf->rankorder?"rank-order":"unordered");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBcastBegin_Basic"
/* Send from roots to leaves */
static PetscErrorCode PetscSFBcastBegin_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscSF_Basic      *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode     ierr;
  PetscSFBasicPack   link;
  PetscInt           i,nrootranks,nleafranks;
  const PetscInt     *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt  *rootranks,*leafranks;
  MPI_Request        *rootreqs,*leafreqs;
  size_t             unitbytes;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&leafranks,&leafoffset,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetPack(sf,unit,rootdata,&link);CHKERRQ(ierr);
  unitbytes = link->unitbytes;
  ierr = PetscSFBasicPackGetReqs(sf,link,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Eagerly post leaf receives */
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    ierr = MPI_Irecv(link->leaf+leafoffset[i]*unitbytes,n,unit,leafranks[i],bas->tag,((PetscObject)sf)->comm,&leafreqs[i]);CHKERRQ(ierr);
  }
  /* Pack and send root data */
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n = rootoffset[i+1] - rootoffset[i];
    void *packstart = link->root+rootoffset[i]*unitbytes;
    (*link->Pack)(n,rootloc+rootoffset[i],rootdata,packstart);
    ierr = MPI_Isend(packstart,n,unit,rootranks[i],bas->tag,((PetscObject)sf)->comm,&rootreqs[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFBcastEnd_Basic"
PetscErrorCode PetscSFBcastEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscErrorCode ierr;
  PetscSFBasicPack link;
  PetscInt i,nleafranks;
  const PetscInt *leafoffset,*leafloc;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetPackInUse(sf,unit,rootdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,PETSC_NULL,&leafoffset,&leafloc);CHKERRQ(ierr);
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    const void *packstart = link->leaf+leafoffset[i]*link->unitbytes;
    (*link->UnpackInsert)(n,leafloc+leafoffset[i],leafdata,packstart);
  }
  ierr = PetscSFBasicReclaimPack(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReduceBegin_Basic"
/* leaf -> root with reduction */
PetscErrorCode PetscSFReduceBegin_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscSF_Basic      *bas = (PetscSF_Basic*)sf->data;
  PetscSFBasicPack   link;
  PetscErrorCode     ierr;
  PetscInt           i,nrootranks,nleafranks;
  const PetscInt     *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt  *rootranks,*leafranks;
  MPI_Request        *rootreqs,*leafreqs;
  size_t             unitbytes;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&leafranks,&leafoffset,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetPack(sf,unit,rootdata,&link);CHKERRQ(ierr);
  unitbytes = link->unitbytes;
  ierr = PetscSFBasicPackGetReqs(sf,link,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Eagerly post root receives */
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n = rootoffset[i+1] - rootoffset[i];
    ierr = MPI_Irecv(link->root+rootoffset[i]*unitbytes,n,unit,rootranks[i],bas->tag,((PetscObject)sf)->comm,&rootreqs[i]);CHKERRQ(ierr);
  }
  /* Pack and send leaf data */
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    void *packstart = link->leaf+leafoffset[i]*unitbytes;
    (*link->Pack)(n,leafloc+leafoffset[i],leafdata,packstart);
    ierr = MPI_Isend(packstart,n,unit,leafranks[i],bas->tag,((PetscObject)sf)->comm,&leafreqs[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFReduceEnd_Basic"
static PetscErrorCode PetscSFReduceEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  void (*UnpackOp)(PetscInt,const PetscInt*,void*,const void*);
  PetscErrorCode ierr;
  PetscSFBasicPack link;
  PetscInt i,nrootranks;
  const PetscInt *rootoffset,*rootloc;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetPackInUse(sf,unit,rootdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,PETSC_NULL,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicPackGetUnpackOp(sf,link,op,&UnpackOp);CHKERRQ(ierr);
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n = rootoffset[i+1] - rootoffset[i];
    const void *packstart = link->root+rootoffset[i]*link->unitbytes;
    (*UnpackOp)(n,rootloc+rootoffset[i],rootdata,packstart);
  }
  ierr = PetscSFBasicReclaimPack(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFetchAndOpBegin_Basic"
static PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReduceBegin_Basic(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFetchAndOpEnd_Basic"
static PetscErrorCode PetscSFFetchAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  void              (*FetchAndOp)(PetscInt,const PetscInt*,void*,void*);
  PetscErrorCode    ierr;
  PetscSFBasicPack  link;
  PetscInt          i,nrootranks,nleafranks;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;
  size_t            unitbytes;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetPackInUse(sf,unit,rootdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  unitbytes = link->unitbytes;
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&leafranks,&leafoffset,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFBasicPackGetReqs(sf,link,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Post leaf receives */
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    ierr = MPI_Irecv(link->leaf+leafoffset[i]*unitbytes,n,unit,leafranks[i],bas->tag,((PetscObject)sf)->comm,&leafreqs[i]);CHKERRQ(ierr);
  }
  /* Process local fetch-and-op, post root sends */
  ierr = PetscSFBasicPackGetFetchAndOp(sf,link,op,&FetchAndOp);CHKERRQ(ierr);
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n = rootoffset[i+1] - rootoffset[i];
    void *packstart = link->root+rootoffset[i]*unitbytes;
    (*FetchAndOp)(n,rootloc+rootoffset[i],rootdata,packstart);
    ierr = MPI_Isend(packstart,n,unit,rootranks[i],bas->tag,((PetscObject)sf)->comm,&rootreqs[i]);CHKERRQ(ierr);
  }
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    const void *packstart = link->leaf+leafoffset[i]*unitbytes;
    (*link->UnpackInsert)(n,leafloc+leafoffset[i],leafupdate,packstart);
  }
  ierr = PetscSFBasicReclaimPack(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFCreate_Basic"
PETSC_EXTERN_C PetscErrorCode PetscSFCreate_Basic(PetscSF sf)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sf->ops->SetUp           = PetscSFSetUp_Basic;
  sf->ops->SetFromOptions  = PetscSFSetFromOptions_Basic;
  sf->ops->Reset           = PetscSFReset_Basic;
  sf->ops->Destroy         = PetscSFDestroy_Basic;
  sf->ops->View            = PetscSFView_Basic;
  sf->ops->BcastBegin      = PetscSFBcastBegin_Basic;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Basic;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Basic;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Basic;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Basic;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Basic;

  ierr = PetscNewLog(sf,PetscSF_Basic,&bas);CHKERRQ(ierr);
  sf->data = (void*)bas;
  PetscFunctionReturn(0);
}
