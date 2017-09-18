
#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

typedef struct _n_PetscSFBasicPack *PetscSFBasicPack;
struct _n_PetscSFBasicPack {
  void (*Pack)(PetscInt,PetscInt,const PetscInt*,const void*,void*);
  void (*UnpackInsert)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackAdd)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMin)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMax)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMinloc)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMaxloc)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  void (*UnpackMult)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*UnpackLAND)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*UnpackBAND)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*UnpackLOR)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*UnpackBOR)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*UnpackLXOR)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*UnpackBXOR)(PetscInt,PetscInt,const PetscInt*,void*,const void *);
  void (*FetchAndInsert)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndAdd)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMin)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMax)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMinloc)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMaxloc)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndMult)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndLAND)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndBAND)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndLOR)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndBOR)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndLXOR)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  void (*FetchAndBXOR)(PetscInt,PetscInt,const PetscInt*,void*,void*);

  MPI_Datatype     unit;
  size_t           unitbytes;   /* Number of bytes in a unit */
  PetscInt         bs;          /* Number of basic units in a unit */
  const void       *key;        /* Array used as key for operation */
  char             **root;      /* Packed root data, indexed by leaf rank */
  char             **leaf;      /* Packed leaf data, indexed by root rank */
  MPI_Request      *requests;   /* Array of root requests followed by leaf requests */
  PetscSFBasicPack next;
};

typedef struct {
  PetscMPIInt      tag;
  PetscMPIInt      niranks;     /* Number of incoming ranks (ranks accessing my roots) */
  PetscMPIInt      ndiranks;    /* Number of incoming ranks (ranks accessing my roots) in distinguished set */
  PetscMPIInt      *iranks;     /* Array of ranks that reference my roots */
  PetscInt         itotal;      /* Total number of graph edges referencing my roots */
  PetscInt         *ioffset;    /* Array of length niranks+1 holding offset in irootloc[] for each rank */
  PetscInt         *irootloc;   /* Incoming roots referenced by ranks starting at ioffset[rank] */
  PetscSFBasicPack avail;       /* One or more entries per MPI Datatype, lazily constructed */
  PetscSFBasicPack inuse;       /* Buffers being used for transactions that have not yet completed */
} PetscSF_Basic;

#if !defined(PETSC_HAVE_MPI_TYPE_DUP)
PETSC_STATIC_INLINE int MPI_Type_dup(MPI_Datatype datatype,MPI_Datatype *newtype)
{
  int ierr;
  ierr = MPI_Type_contiguous(1,datatype,newtype); if (ierr) return ierr;
  ierr = MPI_Type_commit(newtype); if (ierr) return ierr;
  return MPI_SUCCESS;
}
#endif

/*
 * MPI_Reduce_local is not really useful because it can't handle sparse data and it vectorizes "in the wrong direction",
 * therefore we pack data types manually. This section defines packing routines for the standard data types.
 */

#define CPPJoin2_exp(a,b) a ## b
#define CPPJoin2(a,b) CPPJoin2_exp(a,b)
#define CPPJoin3_exp_(a,b,c) a ## b ## _ ## c
#define CPPJoin3_(a,b,c) CPPJoin3_exp_(a,b,c)

/* Basic types without addition */
#define DEF_PackNoInit(type,BS)                                         \
  static void CPPJoin3_(Pack_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,const void *unpacked,void *packed) { \
    const type *u = (const type*)unpacked;                              \
    type *p = (type*)packed;                                            \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++)                                                 \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++)                                          \
          p[i*bs+k] = u[idx[i]*bs+k];                                   \
  }                                                                     \
  static void CPPJoin3_(UnpackInsert_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++)                                                 \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++)                                          \
          u[idx[i]*bs+k] = p[i*bs+k];                                   \
  }                                                                     \
  static void CPPJoin3_(FetchAndInsert_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++) {                                               \
      PetscInt ii = idx[i];                                             \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++) {                                        \
          type t = u[ii*bs+k];                                          \
          u[ii*bs+k] = p[i*bs+k];                                       \
          p[i*bs+k] = t;                                                \
        }                                                               \
    }                                                                   \
  }

/* Basic types defining addition */
#define DEF_PackAddNoInit(type,BS)                                      \
  DEF_PackNoInit(type,BS)                                               \
  static void CPPJoin3_(UnpackAdd_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++)                                                 \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++)                                          \
          u[idx[i]*bs+k] += p[i*bs+k];                                  \
  }                                                                     \
  static void CPPJoin3_(FetchAndAdd_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++) {                                               \
      PetscInt ii = idx[i];                                             \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++) {                                        \
          type t = u[ii*bs+k];                                          \
          u[ii*bs+k] = t + p[i*bs+k];                                   \
          p[i*bs+k] = t;                                                \
        }                                                               \
    }                                                                   \
  }                                                                     \
  static void CPPJoin3_(UnpackMult_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++)                                                 \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++)                                          \
          u[idx[i]*bs+k] *= p[i*bs+k];                                  \
  }                                                                     \
  static void CPPJoin3_(FetchAndMult_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i,j,k;                                                     \
    for (i=0; i<n; i++) {                                               \
      PetscInt ii = idx[i];                                             \
      for (j=0; j<bs; j+=BS)                                            \
        for (k=j; k<j+BS; k++) {                                        \
          type t = u[ii*bs+k];                                          \
          u[ii*bs+k] = t * p[i*bs+k];                                   \
          p[i*bs+k] = t;                                                \
        }                                                               \
    }                                                                   \
  }
#define DEF_Pack(type,BS)                                               \
  DEF_PackAddNoInit(type,BS)                                            \
  static void CPPJoin3_(PackInit_,type,BS)(PetscSFBasicPack link) {     \
    link->Pack = CPPJoin3_(Pack_,type,BS);                              \
    link->UnpackInsert = CPPJoin3_(UnpackInsert_,type,BS);              \
    link->UnpackAdd = CPPJoin3_(UnpackAdd_,type,BS);                    \
    link->UnpackMult = CPPJoin3_(UnpackMult_,type,BS);                  \
    link->FetchAndInsert = CPPJoin3_(FetchAndInsert_,type,BS);          \
    link->FetchAndAdd = CPPJoin3_(FetchAndAdd_,type,BS);                \
    link->FetchAndMult = CPPJoin3_(FetchAndMult_,type,BS);              \
    link->unitbytes = sizeof(type);                                     \
  }
/* Comparable types */
#define DEF_PackCmp(type)                                               \
  DEF_PackAddNoInit(type,1)                                             \
  static void CPPJoin2(UnpackMax_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = PetscMax(v,p[i]);                                     \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(UnpackMin_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = PetscMin(v,p[i]);                                     \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndMax_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = PetscMax(v,p[i]);                                          \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndMin_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = PetscMin(v,p[i]);                                          \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(PackInit_,type)(PetscSFBasicPack link) {         \
    link->Pack = CPPJoin3_(Pack_,type,1);                               \
    link->UnpackInsert = CPPJoin3_(UnpackInsert_,type,1);               \
    link->UnpackAdd  = CPPJoin3_(UnpackAdd_,type,1);                    \
    link->UnpackMax  = CPPJoin2(UnpackMax_,type);                       \
    link->UnpackMin  = CPPJoin2(UnpackMin_,type);                       \
    link->UnpackMult = CPPJoin3_(UnpackMult_,type,1);                   \
    link->FetchAndInsert = CPPJoin3_(FetchAndInsert_,type,1);           \
    link->FetchAndAdd = CPPJoin3_(FetchAndAdd_ ,type,1);                \
    link->FetchAndMax = CPPJoin2(FetchAndMax_ ,type);                   \
    link->FetchAndMin = CPPJoin2(FetchAndMin_ ,type);                   \
    link->FetchAndMult = CPPJoin3_(FetchAndMult_,type,1);               \
    link->unitbytes = sizeof(type);                                     \
  }

/* Logical Types */
#define DEF_PackLog(type)                                               \
  static void CPPJoin2(UnpackLAND_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = v && p[i];                                            \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(UnpackLOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = v || p[i];                                            \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(UnpackLXOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = (!v)!=(!p[i]);                                        \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndLAND_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = v && p[i];                                                 \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndLOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = v || p[i];                                                 \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndLXOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = (!v)!=(!p[i]);                                             \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(PackInit_Logical_,type)(PetscSFBasicPack link) { \
    link->UnpackLAND = CPPJoin2(UnpackLAND_,type);                      \
    link->UnpackLOR  = CPPJoin2(UnpackLOR_,type);                       \
    link->UnpackLXOR = CPPJoin2(UnpackLXOR_,type);                      \
    link->FetchAndLAND = CPPJoin2(FetchAndLAND_,type);                  \
    link->FetchAndLOR  = CPPJoin2(FetchAndLOR_,type);                   \
    link->FetchAndLXOR = CPPJoin2(FetchAndLXOR_,type);                  \
  }


/* Bitwise Types */
#define DEF_PackBit(type)                                               \
  static void CPPJoin2(UnpackBAND_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = v & p[i];                                             \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(UnpackBOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = v | p[i];                                             \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(UnpackBXOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    type *u = (type*)unpacked;                                          \
    const type *p = (const type*)packed;                                \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      type v = u[idx[i]];                                               \
      u[idx[i]] = v^p[i];                                               \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndBAND_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = v & p[i];                                                  \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndBOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = v | p[i];                                                  \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(FetchAndBXOR_,type)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    type *u = (type*)unpacked;                                          \
    type *p = (type*)packed;                                            \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      type v = u[j];                                                    \
      u[j] = v^p[i];                                                    \
      p[i] = v;                                                         \
    }                                                                   \
  }                                                                     \
  static void CPPJoin2(PackInit_Bitwise_,type)(PetscSFBasicPack link) { \
    link->UnpackBAND = CPPJoin2(UnpackBAND_,type);                      \
    link->UnpackBOR  = CPPJoin2(UnpackBOR_,type);                       \
    link->UnpackBXOR = CPPJoin2(UnpackBXOR_,type);                      \
    link->FetchAndBAND = CPPJoin2(FetchAndBAND_,type);                  \
    link->FetchAndBOR  = CPPJoin2(FetchAndBOR_,type);                   \
    link->FetchAndBXOR = CPPJoin2(FetchAndBXOR_,type);                  \
  }

/* Pair types */
#define CPPJoinloc_exp(base,op,t1,t2) base ## op ## loc_ ## t1 ## _ ## t2
#define CPPJoinloc(base,op,t1,t2) CPPJoinloc_exp(base,op,t1,t2)
#define PairType(type1,type2) CPPJoin3_(_pairtype_,type1,type2)
#define DEF_UnpackXloc(type1,type2,locname,op)                              \
  static void CPPJoinloc(Unpack,locname,type1,type2)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;        \
    const PairType(type1,type2) *p = (const PairType(type1,type2)*)packed; \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      if (p[i].a op u[j].a) {                                           \
        u[j].a = p[i].a;                                                \
        u[j].b = p[i].b;                                                \
      } else if (u[j].a == p[i].a) {                                    \
        u[j].b = PetscMin(u[j].b,p[i].b);                               \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  static void CPPJoinloc(FetchAnd,locname,type1,type2)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;        \
    PairType(type1,type2) *p = (PairType(type1,type2)*)packed;          \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      PetscInt j = idx[i];                                              \
      PairType(type1,type2) v;                                          \
      v.a = u[j].a;                                                     \
      v.b = u[j].b;                                                     \
      if (p[i].a op u[j].a) {                                           \
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
  static void CPPJoin3_(Pack_,type1,type2)(PetscInt n,PetscInt bs,const PetscInt *idx,const void *unpacked,void *packed) { \
    const PairType(type1,type2) *u = (const PairType(type1,type2)*)unpacked; \
    PairType(type1,type2) *p = (PairType(type1,type2)*)packed;          \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      p[i].a = u[idx[i]].a;                                             \
      p[i].b = u[idx[i]].b;                                             \
    }                                                                   \
  }                                                                     \
  static void CPPJoin3_(UnpackInsert_,type1,type2)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;       \
    const PairType(type1,type2) *p = (const PairType(type1,type2)*)packed; \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      u[idx[i]].a = p[i].a;                                             \
      u[idx[i]].b = p[i].b;                                             \
    }                                                                   \
  }                                                                     \
  static void CPPJoin3_(UnpackAdd_,type1,type2)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,const void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;       \
    const PairType(type1,type2) *p = (const PairType(type1,type2)*)packed; \
    PetscInt i;                                                         \
    for (i=0; i<n; i++) {                                               \
      u[idx[i]].a += p[i].a;                                            \
      u[idx[i]].b += p[i].b;                                            \
    }                                                                   \
  }                                                                     \
  static void CPPJoin3_(FetchAndInsert_,type1,type2)(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;        \
    PairType(type1,type2) *p = (PairType(type1,type2)*)packed;          \
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
  static void FetchAndAdd_ ## type1 ## _ ## type2(PetscInt n,PetscInt bs,const PetscInt *idx,void *unpacked,void *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;       \
    PairType(type1,type2) *p = (PairType(type1,type2)*)packed;         \
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
  static void CPPJoin3_(PackInit_,type1,type2)(PetscSFBasicPack link) { \
    link->Pack = CPPJoin3_(Pack_,type1,type2);                          \
    link->UnpackInsert = CPPJoin3_(UnpackInsert_,type1,type2);          \
    link->UnpackAdd = CPPJoin3_(UnpackAdd_,type1,type2);                \
    link->UnpackMaxloc = CPPJoin3_(UnpackMaxloc_,type1,type2);          \
    link->UnpackMinloc = CPPJoin3_(UnpackMinloc_,type1,type2);          \
    link->FetchAndInsert = CPPJoin3_(FetchAndInsert_,type1,type2);      \
    link->FetchAndAdd = CPPJoin3_(FetchAndAdd_,type1,type2);            \
    link->FetchAndMaxloc = CPPJoin3_(FetchAndMaxloc_,type1,type2);      \
    link->FetchAndMinloc = CPPJoin3_(FetchAndMinloc_,type1,type2);      \
    link->unitbytes = sizeof(PairType(type1,type2));                    \
  }

/* Currently only dumb blocks of data */
#define BlockType(unit,count) CPPJoin3_(_blocktype_,unit,count)
#define DEF_Block(unit,count)                                           \
  typedef struct {unit v[count];} BlockType(unit,count);                \
  DEF_PackNoInit(BlockType(unit,count),1)                               \
  static void CPPJoin3_(PackInit_block_,unit,count)(PetscSFBasicPack link) { \
    link->Pack = CPPJoin3_(Pack_,BlockType(unit,count),1);               \
    link->UnpackInsert = CPPJoin3_(UnpackInsert_,BlockType(unit,count),1); \
    link->FetchAndInsert = CPPJoin3_(FetchAndInsert_,BlockType(unit,count),1); \
    link->unitbytes = sizeof(BlockType(unit,count));                    \
  }

DEF_PackCmp(int)
DEF_PackBit(int)
DEF_PackLog(int)
DEF_PackCmp(PetscInt)
DEF_PackBit(PetscInt)
DEF_PackLog(PetscInt)
DEF_Pack(PetscInt,2)
DEF_Pack(PetscInt,3)
DEF_Pack(PetscInt,4)
DEF_Pack(PetscInt,5)
DEF_Pack(PetscInt,7)
DEF_PackCmp(PetscReal)
DEF_PackLog(PetscReal)
DEF_Pack(PetscReal,2)
DEF_Pack(PetscReal,3)
DEF_Pack(PetscReal,4)
DEF_Pack(PetscReal,5)
DEF_Pack(PetscReal,7)
#if defined(PETSC_HAVE_COMPLEX)
DEF_Pack(PetscComplex,1)
DEF_Pack(PetscComplex,2)
DEF_Pack(PetscComplex,3)
DEF_Pack(PetscComplex,4)
DEF_Pack(PetscComplex,5)
DEF_Pack(PetscComplex,7)
#endif
DEF_PackPair(int,int)
DEF_PackPair(PetscInt,PetscInt)
DEF_Block(int,1)
DEF_Block(int,2)
DEF_Block(int,3)
DEF_Block(int,4)
DEF_Block(int,5)
DEF_Block(int,6)
DEF_Block(int,7)
DEF_Block(int,8)

static PetscErrorCode PetscSFSetUp_Basic(PetscSF sf)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;
  PetscInt *rlengths,*ilengths,i;
  PetscMPIInt rank,niranks,*iranks;
  MPI_Comm comm;
  MPI_Group group;
  MPI_Request *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = MPI_Comm_group(PETSC_COMM_SELF,&group);CHKERRQ(ierr);
  ierr = PetscSFSetUpRanks(sf,group);CHKERRQ(ierr);
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)sf,&bas->tag);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  /*
   * Inform roots about how many leaves and from which ranks
   */
  ierr = PetscMalloc1(sf->nranks,&rlengths);CHKERRQ(ierr);
  /* Determine number, sending ranks, and length of incoming */
  for (i=0; i<sf->nranks; i++) {
    rlengths[i] = sf->roffset[i+1] - sf->roffset[i]; /* Number of roots referenced by my leaves; for rank sf->ranks[i] */
  }
  ierr = PetscCommBuildTwoSided(comm,1,MPIU_INT,sf->nranks-sf->ndranks,sf->ranks+sf->ndranks,rlengths+sf->ndranks,&niranks,&iranks,(void**)&ilengths);CHKERRQ(ierr);

  /* Partition into distinguished and non-distinguished incoming ranks */
  bas->ndiranks = sf->ndranks;
  bas->niranks = bas->ndiranks + niranks;
  ierr = PetscMalloc2(bas->niranks,&bas->iranks,bas->niranks+1,&bas->ioffset);CHKERRQ(ierr);
  bas->ioffset[0] = 0;
  for (i=0; i<bas->ndiranks; i++) {
    bas->iranks[i] = sf->ranks[i];
    bas->ioffset[i+1] = bas->ioffset[i] + rlengths[i];
  }
  if (bas->ndiranks > 1 || (bas->ndiranks == 1 && bas->iranks[0] != rank)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Broken setup for shared ranks");
  for ( ; i<bas->niranks; i++) {
    bas->iranks[i] = iranks[i-bas->ndiranks];
    bas->ioffset[i+1] = bas->ioffset[i] + ilengths[i-bas->ndiranks];
  }
  bas->itotal = bas->ioffset[i];
  ierr = PetscFree(rlengths);CHKERRQ(ierr);
  ierr = PetscFree(iranks);CHKERRQ(ierr);
  ierr = PetscFree(ilengths);CHKERRQ(ierr);

  /* Send leaf identities to roots */
  ierr = PetscMalloc1(bas->itotal,&bas->irootloc);CHKERRQ(ierr);
  ierr = PetscMalloc2(bas->niranks-bas->ndiranks,&rootreqs,sf->nranks-sf->ndranks,&leafreqs);CHKERRQ(ierr);
  for (i=bas->ndiranks; i<bas->niranks; i++) {
    ierr = MPI_Irecv(bas->irootloc+bas->ioffset[i],bas->ioffset[i+1]-bas->ioffset[i],MPIU_INT,bas->iranks[i],bas->tag,comm,&rootreqs[i-bas->ndiranks]);CHKERRQ(ierr);
  }
  for (i=0; i<sf->nranks; i++) {
    PetscMPIInt npoints;
    ierr = PetscMPIIntCast(sf->roffset[i+1] - sf->roffset[i],&npoints);CHKERRQ(ierr);
    if (i < sf->ndranks) {
      if (sf->ranks[i] != rank) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot interpret distinguished leaf rank");
      if (bas->iranks[0] != rank) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot interpret distinguished root rank");
      if (npoints != bas->ioffset[1]-bas->ioffset[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Distinguished rank exchange has mismatched lengths");
      ierr = PetscMemcpy(bas->irootloc+bas->ioffset[0],sf->rremote+sf->roffset[i],npoints*sizeof(bas->irootloc[0]));CHKERRQ(ierr);
      continue;
    }
    ierr = MPI_Isend(sf->rremote+sf->roffset[i],npoints,MPIU_INT,sf->ranks[i],bas->tag,comm,&leafreqs[i-sf->ndranks]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(bas->niranks-bas->ndiranks,rootreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(sf->nranks-sf->ndranks,leafreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree2(rootreqs,leafreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicPackTypeSetup(PetscSFBasicPack link,MPI_Datatype unit)
{
  PetscErrorCode ierr;
  PetscBool      isInt,isPetscInt,isPetscReal,is2Int,is2PetscInt;
  PetscInt       nPetscIntContig,nPetscRealContig;
#if defined(PETSC_HAVE_COMPLEX)
  PetscBool isPetscComplex;
  PetscInt nPetscComplexContig;
#endif

  PetscFunctionBegin;
  ierr = MPIPetsc_Type_compare(unit,MPI_INT,&isInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_INT,&isPetscInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_INT,&nPetscIntContig);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_REAL,&isPetscReal);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_REAL,&nPetscRealContig);CHKERRQ(ierr);
#if defined(PETSC_HAVE_COMPLEX)
  ierr = MPIPetsc_Type_compare(unit,MPIU_COMPLEX,&isPetscComplex);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_COMPLEX,&nPetscComplexContig);CHKERRQ(ierr);
#endif
  ierr = MPIPetsc_Type_compare(unit,MPI_2INT,&is2Int);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_2INT,&is2PetscInt);CHKERRQ(ierr);
  link->bs = 1;
  if (isInt) {PackInit_int(link); PackInit_Logical_int(link); PackInit_Bitwise_int(link);}
  else if (isPetscInt) {PackInit_PetscInt(link); PackInit_Logical_PetscInt(link); PackInit_Bitwise_PetscInt(link);}
  else if (isPetscReal) {PackInit_PetscReal(link); PackInit_Logical_PetscReal(link);}
#if defined(PETSC_HAVE_COMPLEX)
  else if (isPetscComplex) PackInit_PetscComplex_1(link);
#endif
  else if (is2Int) PackInit_int_int(link);
  else if (is2PetscInt) PackInit_PetscInt_PetscInt(link);
  else if (nPetscIntContig) {
    if (nPetscIntContig%7 == 0) PackInit_PetscInt_7(link);
    else if (nPetscIntContig%5 == 0) PackInit_PetscInt_5(link);
    else if (nPetscIntContig%4 == 0) PackInit_PetscInt_4(link);
    else if (nPetscIntContig%3 == 0) PackInit_PetscInt_3(link);
    else if (nPetscIntContig%2 == 0) PackInit_PetscInt_2(link);
    else PackInit_PetscInt(link);
    link->bs = nPetscIntContig;
    link->unitbytes *= nPetscIntContig;
  } else if (nPetscRealContig) {
    if (nPetscRealContig%7 == 0) PackInit_PetscReal_7(link);
    else if (nPetscRealContig%5 == 0) PackInit_PetscReal_5(link);
    else if (nPetscRealContig%4 == 0) PackInit_PetscReal_4(link);
    else if (nPetscRealContig%3 == 0) PackInit_PetscReal_3(link);
    else if (nPetscRealContig%2 == 0) PackInit_PetscReal_2(link);
    else PackInit_PetscReal(link);
    link->bs = nPetscRealContig;
    link->unitbytes *= nPetscRealContig;
#if defined(PETSC_HAVE_COMPLEX)
  } else if (nPetscComplexContig) {
    if (nPetscComplexContig%7 == 0) PackInit_PetscComplex_7(link);
    else if (nPetscComplexContig%5 == 0) PackInit_PetscComplex_5(link);
    else if (nPetscComplexContig%4 == 0) PackInit_PetscComplex_4(link);
    else if (nPetscComplexContig%3 == 0) PackInit_PetscComplex_3(link);
    else if (nPetscComplexContig%2 == 0) PackInit_PetscComplex_2(link);
    else PackInit_PetscComplex_1(link);
    link->bs = nPetscComplexContig;
    link->unitbytes *= nPetscComplexContig;
#endif
  } else {
    MPI_Aint lb,bytes;
    ierr = MPI_Type_get_extent(unit,&lb,&bytes);CHKERRQ(ierr);
    if (lb != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
    if (bytes % sizeof(int)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for type size not divisible by %D",sizeof(int));
    switch (bytes / sizeof(int)) {
    case 1: PackInit_block_int_1(link); break;
    case 2: PackInit_block_int_2(link); break;
    case 3: PackInit_block_int_3(link); break;
    case 4: PackInit_block_int_4(link); break;
    case 5: PackInit_block_int_5(link); break;
    case 6: PackInit_block_int_6(link); break;
    case 7: PackInit_block_int_7(link); break;
    case 8: PackInit_block_int_8(link); break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for arbitrary block sizes");
    }
  }
  ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicPackGetUnpackOp(PetscSF sf,PetscSFBasicPack link,MPI_Op op,void (**UnpackOp)(PetscInt,PetscInt,const PetscInt*,void*,const void*))
{
  PetscFunctionBegin;
  *UnpackOp = NULL;
  if (op == MPIU_REPLACE) *UnpackOp = link->UnpackInsert;
  else if (op == MPI_SUM || op == MPIU_SUM) *UnpackOp = link->UnpackAdd;
  else if (op == MPI_PROD) *UnpackOp = link->UnpackMult;
  else if (op == MPI_MAX || op == MPIU_MAX) *UnpackOp = link->UnpackMax;
  else if (op == MPI_MIN || op == MPIU_MIN) *UnpackOp = link->UnpackMin;
  else if (op == MPI_LAND) *UnpackOp = link->UnpackLAND;
  else if (op == MPI_BAND) *UnpackOp = link->UnpackBAND;
  else if (op == MPI_LOR) *UnpackOp = link->UnpackLOR;
  else if (op == MPI_BOR) *UnpackOp = link->UnpackBOR;
  else if (op == MPI_LXOR) *UnpackOp = link->UnpackLXOR;
  else if (op == MPI_BXOR) *UnpackOp = link->UnpackBXOR;
  else if (op == MPI_MAXLOC) *UnpackOp = link->UnpackMaxloc;
  else if (op == MPI_MINLOC) *UnpackOp = link->UnpackMinloc;
  else *UnpackOp = NULL;
  PetscFunctionReturn(0);
}
static PetscErrorCode PetscSFBasicPackGetFetchAndOp(PetscSF sf,PetscSFBasicPack link,MPI_Op op,void (**FetchAndOp)(PetscInt,PetscInt,const PetscInt*,void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOp = NULL;
  if (op == MPIU_REPLACE) *FetchAndOp = link->FetchAndInsert;
  else if (op == MPI_SUM || op == MPIU_SUM) *FetchAndOp = link->FetchAndAdd;
  else if (op == MPI_MAX || op == MPIU_MAX) *FetchAndOp = link->FetchAndMax;
  else if (op == MPI_MIN || op == MPIU_MIN) *FetchAndOp = link->FetchAndMin;
  else if (op == MPI_MAXLOC) *FetchAndOp = link->FetchAndMaxloc;
  else if (op == MPI_MINLOC) *FetchAndOp = link->FetchAndMinloc;
  else if (op == MPI_PROD)   *FetchAndOp = link->FetchAndMult;
  else if (op == MPI_LAND)   *FetchAndOp = link->FetchAndLAND;
  else if (op == MPI_BAND)   *FetchAndOp = link->FetchAndBAND;
  else if (op == MPI_LOR)    *FetchAndOp = link->FetchAndLOR;
  else if (op == MPI_BOR)    *FetchAndOp = link->FetchAndBOR;
  else if (op == MPI_LXOR)   *FetchAndOp = link->FetchAndLXOR;
  else if (op == MPI_BXOR)   *FetchAndOp = link->FetchAndBXOR;
  else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for MPI_Op");
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicPackGetReqs(PetscSF sf,PetscSFBasicPack link,MPI_Request **rootreqs,MPI_Request **leafreqs)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (rootreqs) *rootreqs = link->requests;
  if (leafreqs) *leafreqs = link->requests + (bas->niranks - bas->ndiranks);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicPackWaitall(PetscSF sf,PetscSFBasicPack link)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Waitall(bas->niranks+sf->nranks-(bas->ndiranks+sf->ndranks),link->requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicGetRootInfo(PetscSF sf,PetscInt *nrootranks,PetscInt *ndrootranks,const PetscMPIInt **rootranks,const PetscInt **rootoffset,const PetscInt **rootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (nrootranks)  *nrootranks  = bas->niranks;
  if (ndrootranks) *ndrootranks = bas->ndiranks;
  if (rootranks)   *rootranks   = bas->iranks;
  if (rootoffset)  *rootoffset  = bas->ioffset;
  if (rootloc)     *rootloc     = bas->irootloc;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicGetLeafInfo(PetscSF sf,PetscInt *nleafranks,PetscInt *ndleafranks,const PetscMPIInt **leafranks,const PetscInt **leafoffset,const PetscInt **leafloc)
{
  PetscFunctionBegin;
  if (nleafranks)  *nleafranks  = sf->nranks;
  if (ndleafranks) *ndleafranks = sf->ndranks;
  if (leafranks)   *leafranks   = sf->ranks;
  if (leafoffset)  *leafoffset  = sf->roffset;
  if (leafloc)     *leafloc     = sf->rmine;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicGetPack(PetscSF sf,MPI_Datatype unit,const void *key,PetscSFBasicPack *mylink)
{
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode   ierr;
  PetscSFBasicPack link,*p;
  PetscInt         nrootranks,ndrootranks,nleafranks,ndleafranks,i;
  const PetscInt   *rootoffset,*leafoffset;

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
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL);CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = PetscSFBasicPackTypeSetup(link,unit);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrootranks,&link->root,nleafranks,&link->leaf);CHKERRQ(ierr);
  for (i=0; i<nrootranks; i++) {
    ierr = PetscMalloc((rootoffset[i+1]-rootoffset[i])*link->unitbytes,&link->root[i]);CHKERRQ(ierr);
  }
  for (i=0; i<nleafranks; i++) {
    if (i < ndleafranks) {      /* Leaf buffers for distinguished ranks are pointers directly into root buffers */
      if (ndrootranks != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot match distinguished ranks");
      link->leaf[i] = link->root[0];
      continue;
    }
    ierr = PetscMalloc((leafoffset[i+1]-leafoffset[i])*link->unitbytes,&link->leaf[i]);CHKERRQ(ierr);
  }
  ierr = PetscCalloc1(nrootranks+nleafranks,&link->requests);CHKERRQ(ierr);

found:
  link->key  = key;
  link->next = bas->inuse;
  bas->inuse = link;

  *mylink = link;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicGetPackInUse(PetscSF sf,MPI_Datatype unit,const void *key,PetscCopyMode cmode,PetscSFBasicPack *mylink)
{
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode   ierr;
  PetscSFBasicPack link,*p;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (p=&bas->inuse; (link=*p); p=&link->next) {
    PetscBool match;
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match && (key == link->key)) {
      switch (cmode) {
      case PETSC_OWN_POINTER: *p = link->next; break; /* Remove from inuse list */
      case PETSC_USE_POINTER: break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"invalid cmode");
      }
      *mylink = link;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Could not find pack");
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBasicReclaimPack(PetscSF sf,PetscSFBasicPack *link)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  (*link)->key  = NULL;
  (*link)->next = bas->avail;
  bas->avail    = *link;
  *link         = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFSetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSF Basic options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReset_Basic(PetscSF sf)
{
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode   ierr;
  PetscSFBasicPack link,next;

  PetscFunctionBegin;
  ierr = PetscFree2(bas->iranks,bas->ioffset);CHKERRQ(ierr);
  ierr = PetscFree(bas->irootloc);CHKERRQ(ierr);
  if (bas->inuse) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Outstanding operation has not been completed");
  for (link=bas->avail; link; link=next) {
    PetscInt i;
    next = link->next;
    ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);
    for (i=0; i<bas->niranks; i++) {ierr = PetscFree(link->root[i]);CHKERRQ(ierr);}
    for (i=sf->ndranks; i<sf->nranks; i++) {ierr = PetscFree(link->leaf[i]);CHKERRQ(ierr);} /* Free only non-distinguished leaf buffers */
    ierr = PetscFree2(link->root,link->leaf);CHKERRQ(ierr);
    ierr = PetscFree(link->requests);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  bas->avail = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFDestroy_Basic(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sf->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFView_Basic(PetscSF sf,PetscViewer viewer)
{
  /* PetscSF_Basic *bas = (PetscSF_Basic*)sf->data; */
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  sort=%s\n",sf->rankorder ? "rank-order" : "unordered");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Send from roots to leaves */
static PetscErrorCode PetscSFBcastBegin_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode    ierr;
  PetscSFBasicPack  link;
  PetscInt          i,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetPack(sf,unit,rootdata,&link);CHKERRQ(ierr);

  ierr = PetscSFBasicPackGetReqs(sf,link,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Eagerly post leaf receives, but only from non-distinguished ranks -- distinguished ranks will receive via shared memory */
  for (i=ndleafranks; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    ierr = MPI_Irecv(link->leaf[i],n,unit,leafranks[i],bas->tag,PetscObjectComm((PetscObject)sf),&leafreqs[i-ndleafranks]);CHKERRQ(ierr);
  }
  /* Pack and send root data */
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n          = rootoffset[i+1] - rootoffset[i];
    void        *packstart = link->root[i];
    (*link->Pack)(n,link->bs,rootloc+rootoffset[i],rootdata,packstart);
    if (i < ndrootranks) continue; /* shared memory */
    ierr = MPI_Isend(packstart,n,unit,rootranks[i],bas->tag,PetscObjectComm((PetscObject)sf),&rootreqs[i-ndrootranks]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFBcastEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscErrorCode   ierr;
  PetscSFBasicPack link;
  PetscInt         i,nleafranks,ndleafranks;
  const PetscInt   *leafoffset,*leafloc;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetPackInUse(sf,unit,rootdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,&leafloc);CHKERRQ(ierr);
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n          = leafoffset[i+1] - leafoffset[i];
    const void  *packstart = link->leaf[i];
    (*link->UnpackInsert)(n,link->bs,leafloc+leafoffset[i],leafdata,packstart);
  }
  ierr = PetscSFBasicReclaimPack(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* leaf -> root with reduction */
PetscErrorCode PetscSFReduceBegin_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFBasicPack  link;
  PetscErrorCode    ierr;
  PetscInt          i,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetLeafInfo(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,&leafloc);CHKERRQ(ierr);
  ierr = PetscSFBasicGetPack(sf,unit,rootdata,&link);CHKERRQ(ierr);

  ierr = PetscSFBasicPackGetReqs(sf,link,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Eagerly post root receives for non-distinguished ranks */
  for (i=ndrootranks; i<nrootranks; i++) {
    PetscMPIInt n = rootoffset[i+1] - rootoffset[i];
    ierr = MPI_Irecv(link->root[i],n,unit,rootranks[i],bas->tag,PetscObjectComm((PetscObject)sf),&rootreqs[i-ndrootranks]);CHKERRQ(ierr);
  }
  /* Pack and send leaf data */
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n          = leafoffset[i+1] - leafoffset[i];
    void        *packstart = link->leaf[i];
    (*link->Pack)(n,link->bs,leafloc+leafoffset[i],leafdata,packstart);
    if (i < ndleafranks) continue; /* shared memory */
    ierr = MPI_Isend(packstart,n,unit,leafranks[i],bas->tag,PetscObjectComm((PetscObject)sf),&leafreqs[i-ndleafranks]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceEnd_Basic(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  void             (*UnpackOp)(PetscInt,PetscInt,const PetscInt*,void*,const void*);
  PetscErrorCode   ierr;
  PetscSFBasicPack link;
  PetscInt         i,nrootranks;
  PetscMPIInt      typesize = -1;
  const PetscInt   *rootoffset,*rootloc;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetPackInUse(sf,unit,rootdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  ierr = PetscSFBasicGetRootInfo(sf,&nrootranks,NULL,NULL,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr = PetscSFBasicPackGetUnpackOp(sf,link,op,&UnpackOp);CHKERRQ(ierr);
  if (UnpackOp) {
    typesize = link->unitbytes;
  }
  else {
    ierr = MPI_Type_size(unit,&typesize);CHKERRQ(ierr);
  }
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n   = rootoffset[i+1] - rootoffset[i];
    char *packstart = (char *) link->root[i];

    if (UnpackOp) {
      (*UnpackOp)(n,link->bs,rootloc+rootoffset[i],rootdata,(const void *)packstart);
    }
#if PETSC_HAVE_MPI_REDUCE_LOCAL
    else if (n) { /* the op should be defined to operate on the whole datatype, so we ignore link->bs */
      PetscInt j;

      for (j = 0; j < n; j++) {
        ierr = MPI_Reduce_local(packstart+j*typesize,((char *) rootdata)+(rootloc[rootoffset[i]+j])*typesize,1,unit,op);CHKERRQ(ierr);
      }
    }
#else
    else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
    }
#endif
  }
  ierr = PetscSFBasicReclaimPack(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReduceBegin_Basic(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Basic(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  void              (*FetchAndOp)(PetscInt,PetscInt,const PetscInt*,void*,void*);
  PetscErrorCode    ierr;
  PetscSFBasicPack  link;
  PetscInt          i,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt    *rootoffset,*leafoffset,*rootloc,*leafloc;
  const PetscMPIInt *rootranks,*leafranks;
  MPI_Request       *rootreqs,*leafreqs;

  PetscFunctionBegin;
  ierr = PetscSFBasicGetPackInUse(sf,unit,rootdata,PETSC_OWN_POINTER,&link);CHKERRQ(ierr);
  /* This implementation could be changed to unpack as receives arrive, at the cost of non-determinism */
  ierr      = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  ierr      = PetscSFBasicGetRootInfo(sf,&nrootranks,&ndrootranks,&rootranks,&rootoffset,&rootloc);CHKERRQ(ierr);
  ierr      = PetscSFBasicGetLeafInfo(sf,&nleafranks,&ndleafranks,&leafranks,&leafoffset,&leafloc);CHKERRQ(ierr);
  ierr      = PetscSFBasicPackGetReqs(sf,link,&rootreqs,&leafreqs);CHKERRQ(ierr);
  /* Post leaf receives */
  for (i=ndleafranks; i<nleafranks; i++) {
    PetscMPIInt n = leafoffset[i+1] - leafoffset[i];
    ierr = MPI_Irecv(link->leaf[i],n,unit,leafranks[i],bas->tag,PetscObjectComm((PetscObject)sf),&leafreqs[i-ndleafranks]);CHKERRQ(ierr);
  }
  /* Process local fetch-and-op, post root sends */
  ierr = PetscSFBasicPackGetFetchAndOp(sf,link,op,&FetchAndOp);CHKERRQ(ierr);
  for (i=0; i<nrootranks; i++) {
    PetscMPIInt n          = rootoffset[i+1] - rootoffset[i];
    void        *packstart = link->root[i];

    (*FetchAndOp)(n,link->bs,rootloc+rootoffset[i],rootdata,packstart);
    if (i < ndrootranks) continue; /* shared memory */
    ierr = MPI_Isend(packstart,n,unit,rootranks[i],bas->tag,PetscObjectComm((PetscObject)sf),&rootreqs[i-ndrootranks]);CHKERRQ(ierr);
  }
  ierr = PetscSFBasicPackWaitall(sf,link);CHKERRQ(ierr);
  for (i=0; i<nleafranks; i++) {
    PetscMPIInt n          = leafoffset[i+1] - leafoffset[i];
    const void  *packstart = link->leaf[i];
    (*link->UnpackInsert)(n,link->bs,leafloc+leafoffset[i],leafupdate,packstart);
  }
  ierr = PetscSFBasicReclaimPack(sf,&link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSFCreate_Basic(PetscSF sf)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
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

  ierr     = PetscNewLog(sf,&bas);CHKERRQ(ierr);
  sf->data = (void*)bas;
  PetscFunctionReturn(0);
}
