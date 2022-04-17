#include "petsc/private/sfimpl.h"
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

/* This is a C file that contains packing facilities, with dispatches to device if enabled. */

/*
 * MPI_Reduce_local is not really useful because it can't handle sparse data and it vectorizes "in the wrong direction",
 * therefore we pack data types manually. This file defines packing routines for the standard data types.
 */

#define CPPJoin4(a,b,c,d)  a##_##b##_##c##_##d

/* Operations working like s += t */
#define OP_BINARY(op,s,t)   do {(s) = (s) op (t);  } while (0)      /* binary ops in the middle such as +, *, && etc. */
#define OP_FUNCTION(op,s,t) do {(s) = op((s),(t)); } while (0)      /* ops like a function, such as PetscMax, PetscMin */
#define OP_LXOR(op,s,t)     do {(s) = (!(s)) != (!(t));} while (0)  /* logical exclusive OR */
#define OP_ASSIGN(op,s,t)   do {(s) = (t);} while (0)
/* Ref MPI MAXLOC */
#define OP_XLOC(op,s,t) \
  do {                                       \
    if ((s).u == (t).u) (s).i = PetscMin((s).i,(t).i); \
    else if (!((s).u op (t).u)) s = t;           \
  } while (0)

/* DEF_PackFunc - macro defining a Pack routine

   Arguments of the macro:
   +Type      Type of the basic data in an entry, i.e., int, PetscInt, PetscReal etc. It is not the type of an entry.
   .BS        Block size for vectorization. It is a factor of bsz.
   -EQ        (bs == BS) ? 1 : 0. EQ is a compile-time const to help compiler optimizations. See below.

   Arguments of the Pack routine:
   +count     Number of indices in idx[].
   .start     When opt and idx are NULL, it means indices are contiguous & start is the first index; otherwise, not used.
   .opt       Per-pack optimization plan. NULL means no such plan.
   .idx       Indices of entries to packed.
   .link      Provide a context for the current call, such as link->bs, number of basic types in an entry. Ex. if unit is MPI_2INT, then bs=2 and the basic type is int.
   .unpacked  Address of the unpacked data. The entries will be packed are unpacked[idx[i]],for i in [0,count).
   -packed    Address of the packed data.
 */
#define DEF_PackFunc(Type,BS,EQ) \
  static PetscErrorCode CPPJoin4(Pack,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,const void *unpacked,void *packed) \
  {                                                                                                          \
    const Type     *u = (const Type*)unpacked,*u2;                                                           \
    Type           *p = (Type*)packed,*p2;                                                                   \
    PetscInt       i,j,k,X,Y,r,bs=link->bs;                                                                  \
    const PetscInt M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */          \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) PetscCall(PetscArraycpy(p,u+start*MBS,MBS*count));/* idx[] are contiguous */                     \
    else if (opt) { /* has optimizations available */                                                        \
      p2 = p;                                                                                                \
      for (r=0; r<opt->n; r++) {                                                                             \
        u2 = u + opt->start[r]*MBS;                                                                          \
        X  = opt->X[r];                                                                                      \
        Y  = opt->Y[r];                                                                                      \
        for (k=0; k<opt->dz[r]; k++)                                                                         \
          for (j=0; j<opt->dy[r]; j++) {                                                                     \
            PetscCall(PetscArraycpy(p2,u2+(X*Y*k+X*j)*MBS,opt->dx[r]*MBS));                                    \
            p2  += opt->dx[r]*MBS;                                                                           \
          }                                                                                                  \
      }                                                                                                      \
    } else {                                                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)     /* Decent compilers should eliminate this loop when M = const 1 */           \
          for (k=0; k<BS; k++)  /* Compiler either unrolls (BS=1) or vectorizes (BS=2,4,8,etc) this loop */  \
            p[i*MBS+j*BS+k] = u[idx[i]*MBS+j*BS+k];                                                          \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

/* DEF_Action - macro defining a UnpackAndInsert routine that unpacks data from a contiguous buffer
                and inserts into a sparse array.

   Arguments:
  .Type       Type of the data
  .BS         Block size for vectorization
  .EQ        (bs == BS) ? 1 : 0. EQ is a compile-time const.

  Notes:
   This macro is not combined with DEF_ActionAndOp because we want to use memcpy in this macro.
 */
#define DEF_UnpackFunc(Type,BS,EQ)               \
  static PetscErrorCode CPPJoin4(UnpackAndInsert,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,void *unpacked,const void *packed) \
  {                                                                                                          \
    Type           *u = (Type*)unpacked,*u2;                                                                 \
    const Type     *p = (const Type*)packed;                                                                 \
    PetscInt       i,j,k,X,Y,r,bs=link->bs;                                                                  \
    const PetscInt M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */          \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      u += start*MBS;                                                                                        \
      if (u != p) PetscCall(PetscArraycpy(u,p,count*MBS));                                                     \
    } else if (opt) { /* has optimizations available */                                                      \
      for (r=0; r<opt->n; r++) {                                                                             \
        u2 = u + opt->start[r]*MBS;                                                                          \
        X  = opt->X[r];                                                                                      \
        Y  = opt->Y[r];                                                                                      \
        for (k=0; k<opt->dz[r]; k++)                                                                         \
          for (j=0; j<opt->dy[r]; j++) {                                                                     \
            PetscCall(PetscArraycpy(u2+(X*Y*k+X*j)*MBS,p,opt->dx[r]*MBS));                                     \
            p   += opt->dx[r]*MBS;                                                                           \
          }                                                                                                  \
      }                                                                                                      \
    } else {                                                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) u[idx[i]*MBS+j*BS+k] = p[i*MBS+j*BS+k];                                       \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

/* DEF_UnpackAndOp - macro defining a UnpackAndOp routine where Op should not be Insert

   Arguments:
  +Opname     Name of the Op, such as Add, Mult, LAND, etc.
  .Type       Type of the data
  .BS         Block size for vectorization
  .EQ         (bs == BS) ? 1 : 0. EQ is a compile-time const.
  .Op         Operator for the op, such as +, *, &&, ||, PetscMax, PetscMin, etc.
  .OpApply    Macro defining application of the op. Could be OP_BINARY, OP_FUNCTION, OP_LXOR
 */
#define DEF_UnpackAndOp(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(UnpackAnd##Opname,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,void *unpacked,const void *packed) \
  {                                                                                                          \
    Type           *u = (Type*)unpacked,*u2;                                                                 \
    const Type     *p = (const Type*)packed;                                                                 \
    PetscInt       i,j,k,X,Y,r,bs=link->bs;                                                                  \
    const PetscInt M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */          \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      u += start*MBS;                                                                                        \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++)                                                                               \
            OpApply(Op,u[i*MBS+j*BS+k],p[i*MBS+j*BS+k]);                                                     \
    } else if (opt) { /* idx[] has patterns */                                                               \
      for (r=0; r<opt->n; r++) {                                                                             \
        u2 = u + opt->start[r]*MBS;                                                                          \
        X  = opt->X[r];                                                                                      \
        Y  = opt->Y[r];                                                                                      \
        for (k=0; k<opt->dz[r]; k++)                                                                         \
          for (j=0; j<opt->dy[r]; j++) {                                                                     \
            for (i=0; i<opt->dx[r]*MBS; i++) OpApply(Op,u2[(X*Y*k+X*j)*MBS+i],p[i]);                         \
            p += opt->dx[r]*MBS;                                                                             \
          }                                                                                                  \
      }                                                                                                      \
    } else {                                                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++)                                                                               \
            OpApply(Op,u[idx[i]*MBS+j*BS+k],p[i*MBS+j*BS+k]);                                                \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

#define DEF_FetchAndOp(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(FetchAnd##Opname,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,PetscSFPackOpt opt,const PetscInt *idx,void *unpacked,void *packed) \
  {                                                                                                          \
    Type           *u = (Type*)unpacked,*p = (Type*)packed,tmp;                                              \
    PetscInt       i,j,k,r,l,bs=link->bs;                                                                    \
    const PetscInt M = (EQ) ? 1 : bs/BS;                                                                     \
    const PetscInt MBS = M*BS;                                                                               \
    PetscFunctionBegin;                                                                                      \
    for (i=0; i<count; i++) {                                                                                \
      r = (!idx ? start+i : idx[i])*MBS;                                                                     \
      l = i*MBS;                                                                                             \
      for (j=0; j<M; j++)                                                                                    \
        for (k=0; k<BS; k++) {                                                                               \
          tmp = u[r+j*BS+k];                                                                                 \
          OpApply(Op,u[r+j*BS+k],p[l+j*BS+k]);                                                               \
          p[l+j*BS+k] = tmp;                                                                                 \
        }                                                                                                    \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

#define DEF_ScatterAndOp(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(ScatterAnd##Opname,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt srcStart,PetscSFPackOpt srcOpt,const PetscInt *srcIdx,const void *src,PetscInt dstStart,PetscSFPackOpt dstOpt,const PetscInt *dstIdx,void *dst) \
  {                                                                                                          \
    const Type     *u = (const Type*)src;                                                                    \
    Type           *v = (Type*)dst;                                                                          \
    PetscInt       i,j,k,s,t,X,Y,bs = link->bs;                                                              \
    const PetscInt M = (EQ) ? 1 : bs/BS;                                                                     \
    const PetscInt MBS = M*BS;                                                                               \
    PetscFunctionBegin;                                                                                      \
    if (!srcIdx) { /* src is contiguous */                                                                   \
      u += srcStart*MBS;                                                                                     \
      PetscCall(CPPJoin4(UnpackAnd##Opname,Type,BS,EQ)(link,count,dstStart,dstOpt,dstIdx,dst,u));              \
    } else if (srcOpt && !dstIdx) { /* src is 3D, dst is contiguous */                                       \
      u += srcOpt->start[0]*MBS;                                                                             \
      v += dstStart*MBS;                                                                                     \
      X  = srcOpt->X[0]; Y = srcOpt->Y[0];                                                                   \
      for (k=0; k<srcOpt->dz[0]; k++)                                                                        \
        for (j=0; j<srcOpt->dy[0]; j++) {                                                                    \
          for (i=0; i<srcOpt->dx[0]*MBS; i++) OpApply(Op,v[i],u[(X*Y*k+X*j)*MBS+i]);                         \
          v += srcOpt->dx[0]*MBS;                                                                            \
        }                                                                                                    \
    } else { /* all other cases */                                                                           \
      for (i=0; i<count; i++) {                                                                              \
        s = (!srcIdx ? srcStart+i : srcIdx[i])*MBS;                                                          \
        t = (!dstIdx ? dstStart+i : dstIdx[i])*MBS;                                                          \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) OpApply(Op,v[t+j*BS+k],u[s+j*BS+k]);                                          \
      }                                                                                                      \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

#define DEF_FetchAndOpLocal(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(FetchAnd##Opname##Local,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt rootstart,PetscSFPackOpt rootopt,const PetscInt *rootidx,void *rootdata,PetscInt leafstart,PetscSFPackOpt leafopt,const PetscInt *leafidx,const void *leafdata,void *leafupdate) \
  {                                                                                                          \
    Type           *rdata = (Type*)rootdata,*lupdate = (Type*)leafupdate;                                    \
    const Type     *ldata = (const Type*)leafdata;                                                           \
    PetscInt       i,j,k,r,l,bs = link->bs;                                                                  \
    const PetscInt M = (EQ) ? 1 : bs/BS;                                                                     \
    const PetscInt MBS = M*BS;                                                                               \
    PetscFunctionBegin;                                                                                      \
    for (i=0; i<count; i++) {                                                                                \
      r = (rootidx ? rootidx[i] : rootstart+i)*MBS;                                                          \
      l = (leafidx ? leafidx[i] : leafstart+i)*MBS;                                                          \
      for (j=0; j<M; j++)                                                                                    \
        for (k=0; k<BS; k++) {                                                                               \
          lupdate[l+j*BS+k] = rdata[r+j*BS+k];                                                               \
          OpApply(Op,rdata[r+j*BS+k],ldata[l+j*BS+k]);                                                       \
        }                                                                                                    \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

/* Pack, Unpack/Fetch ops */
#define DEF_Pack(Type,BS,EQ)                                                      \
  DEF_PackFunc(Type,BS,EQ)                                                        \
  DEF_UnpackFunc(Type,BS,EQ)                                                      \
  DEF_ScatterAndOp(Type,BS,EQ,Insert,=,OP_ASSIGN)                                 \
  static void CPPJoin4(PackInit_Pack,Type,BS,EQ)(PetscSFLink link) {              \
    link->h_Pack            = CPPJoin4(Pack,           Type,BS,EQ);               \
    link->h_UnpackAndInsert = CPPJoin4(UnpackAndInsert,Type,BS,EQ);               \
    link->h_ScatterAndInsert= CPPJoin4(ScatterAndInsert,Type,BS,EQ);              \
  }

/* Add, Mult ops */
#define DEF_Add(Type,BS,EQ)                                                       \
  DEF_UnpackAndOp    (Type,BS,EQ,Add, +,OP_BINARY)                                \
  DEF_UnpackAndOp    (Type,BS,EQ,Mult,*,OP_BINARY)                                \
  DEF_FetchAndOp     (Type,BS,EQ,Add, +,OP_BINARY)                                \
  DEF_ScatterAndOp   (Type,BS,EQ,Add, +,OP_BINARY)                                \
  DEF_ScatterAndOp   (Type,BS,EQ,Mult,*,OP_BINARY)                                \
  DEF_FetchAndOpLocal(Type,BS,EQ,Add, +,OP_BINARY)                                \
  static void CPPJoin4(PackInit_Add,Type,BS,EQ)(PetscSFLink link) {               \
    link->h_UnpackAndAdd     = CPPJoin4(UnpackAndAdd,    Type,BS,EQ);             \
    link->h_UnpackAndMult    = CPPJoin4(UnpackAndMult,   Type,BS,EQ);             \
    link->h_FetchAndAdd      = CPPJoin4(FetchAndAdd,     Type,BS,EQ);             \
    link->h_ScatterAndAdd    = CPPJoin4(ScatterAndAdd,   Type,BS,EQ);             \
    link->h_ScatterAndMult   = CPPJoin4(ScatterAndMult,  Type,BS,EQ);             \
    link->h_FetchAndAddLocal = CPPJoin4(FetchAndAddLocal,Type,BS,EQ);             \
  }

/* Max, Min ops */
#define DEF_Cmp(Type,BS,EQ)                                                       \
  DEF_UnpackAndOp (Type,BS,EQ,Max,PetscMax,OP_FUNCTION)                           \
  DEF_UnpackAndOp (Type,BS,EQ,Min,PetscMin,OP_FUNCTION)                           \
  DEF_ScatterAndOp(Type,BS,EQ,Max,PetscMax,OP_FUNCTION)                           \
  DEF_ScatterAndOp(Type,BS,EQ,Min,PetscMin,OP_FUNCTION)                           \
  static void CPPJoin4(PackInit_Compare,Type,BS,EQ)(PetscSFLink link) {           \
    link->h_UnpackAndMax    = CPPJoin4(UnpackAndMax,   Type,BS,EQ);               \
    link->h_UnpackAndMin    = CPPJoin4(UnpackAndMin,   Type,BS,EQ);               \
    link->h_ScatterAndMax   = CPPJoin4(ScatterAndMax,  Type,BS,EQ);               \
    link->h_ScatterAndMin   = CPPJoin4(ScatterAndMin,  Type,BS,EQ);               \
  }

/* Logical ops.
  The operator in OP_LXOR should be empty but is ||. It is not used. Put here to avoid
  the compilation warning "empty macro arguments are undefined in ISO C90"
 */
#define DEF_Log(Type,BS,EQ)                                                       \
  DEF_UnpackAndOp (Type,BS,EQ,LAND,&&,OP_BINARY)                                  \
  DEF_UnpackAndOp (Type,BS,EQ,LOR, ||,OP_BINARY)                                  \
  DEF_UnpackAndOp (Type,BS,EQ,LXOR,||, OP_LXOR)                                   \
  DEF_ScatterAndOp(Type,BS,EQ,LAND,&&,OP_BINARY)                                  \
  DEF_ScatterAndOp(Type,BS,EQ,LOR, ||,OP_BINARY)                                  \
  DEF_ScatterAndOp(Type,BS,EQ,LXOR,||, OP_LXOR)                                   \
  static void CPPJoin4(PackInit_Logical,Type,BS,EQ)(PetscSFLink link) {           \
    link->h_UnpackAndLAND   = CPPJoin4(UnpackAndLAND, Type,BS,EQ);                \
    link->h_UnpackAndLOR    = CPPJoin4(UnpackAndLOR,  Type,BS,EQ);                \
    link->h_UnpackAndLXOR   = CPPJoin4(UnpackAndLXOR, Type,BS,EQ);                \
    link->h_ScatterAndLAND  = CPPJoin4(ScatterAndLAND,Type,BS,EQ);                \
    link->h_ScatterAndLOR   = CPPJoin4(ScatterAndLOR, Type,BS,EQ);                \
    link->h_ScatterAndLXOR  = CPPJoin4(ScatterAndLXOR,Type,BS,EQ);                \
  }

/* Bitwise ops */
#define DEF_Bit(Type,BS,EQ)                                                       \
  DEF_UnpackAndOp (Type,BS,EQ,BAND,&,OP_BINARY)                                   \
  DEF_UnpackAndOp (Type,BS,EQ,BOR, |,OP_BINARY)                                   \
  DEF_UnpackAndOp (Type,BS,EQ,BXOR,^,OP_BINARY)                                   \
  DEF_ScatterAndOp(Type,BS,EQ,BAND,&,OP_BINARY)                                   \
  DEF_ScatterAndOp(Type,BS,EQ,BOR, |,OP_BINARY)                                   \
  DEF_ScatterAndOp(Type,BS,EQ,BXOR,^,OP_BINARY)                                   \
  static void CPPJoin4(PackInit_Bitwise,Type,BS,EQ)(PetscSFLink link) {           \
    link->h_UnpackAndBAND   = CPPJoin4(UnpackAndBAND, Type,BS,EQ);                \
    link->h_UnpackAndBOR    = CPPJoin4(UnpackAndBOR,  Type,BS,EQ);                \
    link->h_UnpackAndBXOR   = CPPJoin4(UnpackAndBXOR, Type,BS,EQ);                \
    link->h_ScatterAndBAND  = CPPJoin4(ScatterAndBAND,Type,BS,EQ);                \
    link->h_ScatterAndBOR   = CPPJoin4(ScatterAndBOR, Type,BS,EQ);                \
    link->h_ScatterAndBXOR  = CPPJoin4(ScatterAndBXOR,Type,BS,EQ);                \
  }

/* Maxloc, Minloc ops */
#define DEF_Xloc(Type,BS,EQ)                                                      \
  DEF_UnpackAndOp (Type,BS,EQ,Max,>,OP_XLOC)                                      \
  DEF_UnpackAndOp (Type,BS,EQ,Min,<,OP_XLOC)                                      \
  DEF_ScatterAndOp(Type,BS,EQ,Max,>,OP_XLOC)                                      \
  DEF_ScatterAndOp(Type,BS,EQ,Min,<,OP_XLOC)                                      \
  static void CPPJoin4(PackInit_Xloc,Type,BS,EQ)(PetscSFLink link) {              \
    link->h_UnpackAndMaxloc  = CPPJoin4(UnpackAndMax, Type,BS,EQ);                \
    link->h_UnpackAndMinloc  = CPPJoin4(UnpackAndMin, Type,BS,EQ);                \
    link->h_ScatterAndMaxloc = CPPJoin4(ScatterAndMax,Type,BS,EQ);                \
    link->h_ScatterAndMinloc = CPPJoin4(ScatterAndMin,Type,BS,EQ);                \
  }

#define DEF_IntegerType(Type,BS,EQ)                                               \
  DEF_Pack(Type,BS,EQ)                                                            \
  DEF_Add(Type,BS,EQ)                                                             \
  DEF_Cmp(Type,BS,EQ)                                                             \
  DEF_Log(Type,BS,EQ)                                                             \
  DEF_Bit(Type,BS,EQ)                                                             \
  static void CPPJoin4(PackInit_IntegerType,Type,BS,EQ)(PetscSFLink link) {       \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                     \
    CPPJoin4(PackInit_Add,Type,BS,EQ)(link);                                      \
    CPPJoin4(PackInit_Compare,Type,BS,EQ)(link);                                  \
    CPPJoin4(PackInit_Logical,Type,BS,EQ)(link);                                  \
    CPPJoin4(PackInit_Bitwise,Type,BS,EQ)(link);                                  \
  }

#define DEF_RealType(Type,BS,EQ)                                                  \
  DEF_Pack(Type,BS,EQ)                                                            \
  DEF_Add(Type,BS,EQ)                                                             \
  DEF_Cmp(Type,BS,EQ)                                                             \
  static void CPPJoin4(PackInit_RealType,Type,BS,EQ)(PetscSFLink link) {          \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                     \
    CPPJoin4(PackInit_Add,Type,BS,EQ)(link);                                      \
    CPPJoin4(PackInit_Compare,Type,BS,EQ)(link);                                  \
  }

#if defined(PETSC_HAVE_COMPLEX)
#define DEF_ComplexType(Type,BS,EQ)                                               \
  DEF_Pack(Type,BS,EQ)                                                            \
  DEF_Add(Type,BS,EQ)                                                             \
  static void CPPJoin4(PackInit_ComplexType,Type,BS,EQ)(PetscSFLink link) {       \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                     \
    CPPJoin4(PackInit_Add,Type,BS,EQ)(link);                                      \
  }
#endif

#define DEF_DumbType(Type,BS,EQ)                                                  \
  DEF_Pack(Type,BS,EQ)                                                            \
  static void CPPJoin4(PackInit_DumbType,Type,BS,EQ)(PetscSFLink link) {          \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                     \
  }

/* Maxloc, Minloc */
#define DEF_PairType(Type,BS,EQ)                                                  \
  DEF_Pack(Type,BS,EQ)                                                            \
  DEF_Xloc(Type,BS,EQ)                                                            \
  static void CPPJoin4(PackInit_PairType,Type,BS,EQ)(PetscSFLink link) {          \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                     \
    CPPJoin4(PackInit_Xloc,Type,BS,EQ)(link);                                     \
  }

DEF_IntegerType(PetscInt,1,1) /* unit = 1 MPIU_INT  */
DEF_IntegerType(PetscInt,2,1) /* unit = 2 MPIU_INTs */
DEF_IntegerType(PetscInt,4,1) /* unit = 4 MPIU_INTs */
DEF_IntegerType(PetscInt,8,1) /* unit = 8 MPIU_INTs */
DEF_IntegerType(PetscInt,1,0) /* unit = 1*n MPIU_INTs, n>1 */
DEF_IntegerType(PetscInt,2,0) /* unit = 2*n MPIU_INTs, n>1 */
DEF_IntegerType(PetscInt,4,0) /* unit = 4*n MPIU_INTs, n>1 */
DEF_IntegerType(PetscInt,8,0) /* unit = 8*n MPIU_INTs, n>1. Routines with bigger BS are tried first. */

#if defined(PETSC_USE_64BIT_INDICES) /* Do not need (though it is OK) to generate redundant functions if PetscInt is int */
DEF_IntegerType(int,1,1)
DEF_IntegerType(int,2,1)
DEF_IntegerType(int,4,1)
DEF_IntegerType(int,8,1)
DEF_IntegerType(int,1,0)
DEF_IntegerType(int,2,0)
DEF_IntegerType(int,4,0)
DEF_IntegerType(int,8,0)
#endif

/* The typedefs are used to get a typename without space that CPPJoin can handle */
typedef signed char SignedChar;
DEF_IntegerType(SignedChar,1,1)
DEF_IntegerType(SignedChar,2,1)
DEF_IntegerType(SignedChar,4,1)
DEF_IntegerType(SignedChar,8,1)
DEF_IntegerType(SignedChar,1,0)
DEF_IntegerType(SignedChar,2,0)
DEF_IntegerType(SignedChar,4,0)
DEF_IntegerType(SignedChar,8,0)

typedef unsigned char UnsignedChar;
DEF_IntegerType(UnsignedChar,1,1)
DEF_IntegerType(UnsignedChar,2,1)
DEF_IntegerType(UnsignedChar,4,1)
DEF_IntegerType(UnsignedChar,8,1)
DEF_IntegerType(UnsignedChar,1,0)
DEF_IntegerType(UnsignedChar,2,0)
DEF_IntegerType(UnsignedChar,4,0)
DEF_IntegerType(UnsignedChar,8,0)

DEF_RealType(PetscReal,1,1)
DEF_RealType(PetscReal,2,1)
DEF_RealType(PetscReal,4,1)
DEF_RealType(PetscReal,8,1)
DEF_RealType(PetscReal,1,0)
DEF_RealType(PetscReal,2,0)
DEF_RealType(PetscReal,4,0)
DEF_RealType(PetscReal,8,0)

#if defined(PETSC_HAVE_COMPLEX)
DEF_ComplexType(PetscComplex,1,1)
DEF_ComplexType(PetscComplex,2,1)
DEF_ComplexType(PetscComplex,4,1)
DEF_ComplexType(PetscComplex,8,1)
DEF_ComplexType(PetscComplex,1,0)
DEF_ComplexType(PetscComplex,2,0)
DEF_ComplexType(PetscComplex,4,0)
DEF_ComplexType(PetscComplex,8,0)
#endif

#define PairType(Type1,Type2) Type1##_##Type2
typedef struct {int u; int i;}           PairType(int,int);
typedef struct {PetscInt u; PetscInt i;} PairType(PetscInt,PetscInt);
DEF_PairType(PairType(int,int),1,1)
DEF_PairType(PairType(PetscInt,PetscInt),1,1)

/* If we don't know the basic type, we treat it as a stream of chars or ints */
DEF_DumbType(char,1,1)
DEF_DumbType(char,2,1)
DEF_DumbType(char,4,1)
DEF_DumbType(char,1,0)
DEF_DumbType(char,2,0)
DEF_DumbType(char,4,0)

typedef int DumbInt; /* To have a different name than 'int' used above. The name is used to make routine names. */
DEF_DumbType(DumbInt,1,1)
DEF_DumbType(DumbInt,2,1)
DEF_DumbType(DumbInt,4,1)
DEF_DumbType(DumbInt,8,1)
DEF_DumbType(DumbInt,1,0)
DEF_DumbType(DumbInt,2,0)
DEF_DumbType(DumbInt,4,0)
DEF_DumbType(DumbInt,8,0)

PetscErrorCode PetscSFLinkDestroy(PetscSF sf,PetscSFLink link)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          i,nreqs = (bas->nrootreqs+sf->nleafreqs)*8;

  PetscFunctionBegin;
  /* Destroy device-specific fields */
  if (link->deviceinited) PetscCall((*link->Destroy)(sf,link));

  /* Destroy host related fields */
  if (!link->isbuiltin) PetscCallMPI(MPI_Type_free(&link->unit));
  if (!link->use_nvshmem) {
    for (i=0; i<nreqs; i++) { /* Persistent reqs must be freed. */
      if (link->reqs[i] != MPI_REQUEST_NULL) PetscCallMPI(MPI_Request_free(&link->reqs[i]));
    }
    PetscCall(PetscFree(link->reqs));
    for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
      PetscCall(PetscFree(link->rootbuf_alloc[i][PETSC_MEMTYPE_HOST]));
      PetscCall(PetscFree(link->leafbuf_alloc[i][PETSC_MEMTYPE_HOST]));
    }
  }
  PetscCall(PetscFree(link));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkCreate(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  PetscFunctionBegin;
  PetscCall(PetscSFSetErrorOnUnsupportedOverlap(sf,unit,rootdata,leafdata));
 #if defined(PETSC_HAVE_NVSHMEM)
  {
    PetscBool use_nvshmem;
    PetscCall(PetscSFLinkNvshmemCheck(sf,rootmtype,rootdata,leafmtype,leafdata,&use_nvshmem));
    if (use_nvshmem) {
      PetscCall(PetscSFLinkCreate_NVSHMEM(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,sfop,mylink));
      PetscFunctionReturn(0);
    }
  }
 #endif
  PetscCall(PetscSFLinkCreate_MPI(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op,sfop,mylink));
  PetscFunctionReturn(0);
}

/* Return root/leaf buffers and MPI requests attached to the link for MPI communication in the given direction.
   If the sf uses persistent requests and the requests have not been initialized, then initialize them.
*/
PetscErrorCode PetscSFLinkGetMPIBuffersAndRequests(PetscSF sf,PetscSFLink link,PetscSFDirection direction,void **rootbuf, void **leafbuf,MPI_Request **rootreqs,MPI_Request **leafreqs)
{
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscInt             i,j,cnt,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt       *rootoffset,*leafoffset;
  MPI_Aint             disp;
  MPI_Comm             comm = PetscObjectComm((PetscObject)sf);
  MPI_Datatype         unit = link->unit;
  const PetscMemType   rootmtype_mpi = link->rootmtype_mpi,leafmtype_mpi = link->leafmtype_mpi; /* Used to select buffers passed to MPI */
  const PetscInt       rootdirect_mpi = link->rootdirect_mpi,leafdirect_mpi = link->leafdirect_mpi;

  PetscFunctionBegin;
  /* Init persistent MPI requests if not yet. Currently only SFBasic uses persistent MPI */
  if (sf->persistent) {
    if (rootreqs && bas->rootbuflen[PETSCSF_REMOTE] && !link->rootreqsinited[direction][rootmtype_mpi][rootdirect_mpi]) {
      PetscCall(PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL));
      if (direction == PETSCSF_LEAF2ROOT) {
        for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
          disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
          cnt  = rootoffset[i+1]-rootoffset[i];
          PetscCallMPI(MPIU_Recv_init(link->rootbuf[PETSCSF_REMOTE][rootmtype_mpi]+disp,cnt,unit,bas->iranks[i],link->tag,comm,link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi]+j));
        }
      } else { /* PETSCSF_ROOT2LEAF */
        for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
          disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
          cnt  = rootoffset[i+1]-rootoffset[i];
          PetscCallMPI(MPIU_Send_init(link->rootbuf[PETSCSF_REMOTE][rootmtype_mpi]+disp,cnt,unit,bas->iranks[i],link->tag,comm,link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi]+j));
        }
      }
      link->rootreqsinited[direction][rootmtype_mpi][rootdirect_mpi] = PETSC_TRUE;
    }

    if (leafreqs && sf->leafbuflen[PETSCSF_REMOTE] && !link->leafreqsinited[direction][leafmtype_mpi][leafdirect_mpi]) {
      PetscCall(PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL));
      if (direction == PETSCSF_LEAF2ROOT) {
        for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
          disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
          cnt  = leafoffset[i+1]-leafoffset[i];
          PetscCallMPI(MPIU_Send_init(link->leafbuf[PETSCSF_REMOTE][leafmtype_mpi]+disp,cnt,unit,sf->ranks[i],link->tag,comm,link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi]+j));
        }
      } else { /* PETSCSF_ROOT2LEAF */
        for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
          disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
          cnt  = leafoffset[i+1]-leafoffset[i];
          PetscCallMPI(MPIU_Recv_init(link->leafbuf[PETSCSF_REMOTE][leafmtype_mpi]+disp,cnt,unit,sf->ranks[i],link->tag,comm,link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi]+j));
        }
      }
      link->leafreqsinited[direction][leafmtype_mpi][leafdirect_mpi] = PETSC_TRUE;
    }
  }
  if (rootbuf)  *rootbuf  = link->rootbuf[PETSCSF_REMOTE][rootmtype_mpi];
  if (leafbuf)  *leafbuf  = link->leafbuf[PETSCSF_REMOTE][leafmtype_mpi];
  if (rootreqs) *rootreqs = link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi];
  if (leafreqs) *leafreqs = link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetInUse(PetscSF sf,MPI_Datatype unit,const void *rootdata,const void *leafdata,PetscCopyMode cmode,PetscSFLink *mylink)
{
  PetscSFLink       link,*p;
  PetscSF_Basic     *bas=(PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (p=&bas->inuse; (link=*p); p=&link->next) {
    PetscBool match;
    PetscCall(MPIPetsc_Type_compare(unit,link->unit,&match));
    if (match && (rootdata == link->rootdata) && (leafdata == link->leafdata)) {
      switch (cmode) {
      case PETSC_OWN_POINTER: *p = link->next; break; /* Remove from inuse list */
      case PETSC_USE_POINTER: break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"invalid cmode");
      }
      *mylink = link;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Could not find pack");
}

PetscErrorCode PetscSFLinkReclaim(PetscSF sf,PetscSFLink *mylink)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFLink       link = *mylink;

  PetscFunctionBegin;
  link->rootdata = NULL;
  link->leafdata = NULL;
  link->next     = bas->avail;
  bas->avail     = link;
  *mylink        = NULL;
  PetscFunctionReturn(0);
}

/* Error out on unsupported overlapped communications */
PetscErrorCode PetscSFSetErrorOnUnsupportedOverlap(PetscSF sf,MPI_Datatype unit,const void *rootdata,const void *leafdata)
{
  PetscSFLink       link,*p;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscBool         match;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    /* Look up links in use and error out if there is a match. When both rootdata and leafdata are NULL, ignore
       the potential overlapping since this process does not participate in communication. Overlapping is harmless.
    */
    if (rootdata || leafdata) {
      for (p=&bas->inuse; (link=*p); p=&link->next) {
        PetscCall(MPIPetsc_Type_compare(unit,link->unit,&match));
        PetscCheck(!match || rootdata != link->rootdata || leafdata != link->leafdata,PETSC_COMM_SELF,PETSC_ERR_SUP,"Overlapped PetscSF with the same rootdata(%p), leafdata(%p) and data type. Undo the overlapping to avoid the error.",rootdata,leafdata);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFLinkMemcpy_Host(PetscSFLink link,PetscMemType dstmtype,void* dst,PetscMemType srcmtype,const void*src,size_t n)
{
  PetscFunctionBegin;
  if (n) PetscCall(PetscMemcpy(dst,src,n));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkSetUp_Host(PetscSF sf,PetscSFLink link,MPI_Datatype unit)
{
  PetscInt       nSignedChar=0,nUnsignedChar=0,nInt=0,nPetscInt=0,nPetscReal=0;
  PetscBool      is2Int,is2PetscInt;
  PetscMPIInt    ni,na,nd,combiner;
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt       nPetscComplex=0;
#endif

  PetscFunctionBegin;
  PetscCall(MPIPetsc_Type_compare_contig(unit,MPI_SIGNED_CHAR,  &nSignedChar));
  PetscCall(MPIPetsc_Type_compare_contig(unit,MPI_UNSIGNED_CHAR,&nUnsignedChar));
  /* MPI_CHAR is treated below as a dumb type that does not support reduction according to MPI standard */
  PetscCall(MPIPetsc_Type_compare_contig(unit,MPI_INT,  &nInt));
  PetscCall(MPIPetsc_Type_compare_contig(unit,MPIU_INT, &nPetscInt));
  PetscCall(MPIPetsc_Type_compare_contig(unit,MPIU_REAL,&nPetscReal));
#if defined(PETSC_HAVE_COMPLEX)
  PetscCall(MPIPetsc_Type_compare_contig(unit,MPIU_COMPLEX,&nPetscComplex));
#endif
  PetscCall(MPIPetsc_Type_compare(unit,MPI_2INT,&is2Int));
  PetscCall(MPIPetsc_Type_compare(unit,MPIU_2INT,&is2PetscInt));
  /* TODO: shaell we also handle Fortran MPI_2REAL? */
  PetscCallMPI(MPI_Type_get_envelope(unit,&ni,&na,&nd,&combiner));
  link->isbuiltin = (combiner == MPI_COMBINER_NAMED) ? PETSC_TRUE : PETSC_FALSE; /* unit is MPI builtin */
  link->bs = 1; /* default */

  if (is2Int) {
    PackInit_PairType_int_int_1_1(link);
    link->bs        = 1;
    link->unitbytes = 2*sizeof(int);
    link->isbuiltin = PETSC_TRUE; /* unit is PETSc builtin */
    link->basicunit = MPI_2INT;
    link->unit      = MPI_2INT;
  } else if (is2PetscInt) { /* TODO: when is2PetscInt and nPetscInt=2, we don't know which path to take. The two paths support different ops. */
    PackInit_PairType_PetscInt_PetscInt_1_1(link);
    link->bs        = 1;
    link->unitbytes = 2*sizeof(PetscInt);
    link->basicunit = MPIU_2INT;
    link->isbuiltin = PETSC_TRUE; /* unit is PETSc builtin */
    link->unit      = MPIU_2INT;
  } else if (nPetscReal) {
    if      (nPetscReal == 8) PackInit_RealType_PetscReal_8_1(link); else if (nPetscReal%8 == 0) PackInit_RealType_PetscReal_8_0(link);
    else if (nPetscReal == 4) PackInit_RealType_PetscReal_4_1(link); else if (nPetscReal%4 == 0) PackInit_RealType_PetscReal_4_0(link);
    else if (nPetscReal == 2) PackInit_RealType_PetscReal_2_1(link); else if (nPetscReal%2 == 0) PackInit_RealType_PetscReal_2_0(link);
    else if (nPetscReal == 1) PackInit_RealType_PetscReal_1_1(link); else if (nPetscReal%1 == 0) PackInit_RealType_PetscReal_1_0(link);
    link->bs        = nPetscReal;
    link->unitbytes = nPetscReal*sizeof(PetscReal);
    link->basicunit = MPIU_REAL;
    if (link->bs == 1) {link->isbuiltin = PETSC_TRUE; link->unit = MPIU_REAL;}
  } else if (nPetscInt) {
    if      (nPetscInt == 8) PackInit_IntegerType_PetscInt_8_1(link); else if (nPetscInt%8 == 0) PackInit_IntegerType_PetscInt_8_0(link);
    else if (nPetscInt == 4) PackInit_IntegerType_PetscInt_4_1(link); else if (nPetscInt%4 == 0) PackInit_IntegerType_PetscInt_4_0(link);
    else if (nPetscInt == 2) PackInit_IntegerType_PetscInt_2_1(link); else if (nPetscInt%2 == 0) PackInit_IntegerType_PetscInt_2_0(link);
    else if (nPetscInt == 1) PackInit_IntegerType_PetscInt_1_1(link); else if (nPetscInt%1 == 0) PackInit_IntegerType_PetscInt_1_0(link);
    link->bs        = nPetscInt;
    link->unitbytes = nPetscInt*sizeof(PetscInt);
    link->basicunit = MPIU_INT;
    if (link->bs == 1) {link->isbuiltin = PETSC_TRUE; link->unit = MPIU_INT;}
#if defined(PETSC_USE_64BIT_INDICES)
  } else if (nInt) {
    if      (nInt == 8) PackInit_IntegerType_int_8_1(link); else if (nInt%8 == 0) PackInit_IntegerType_int_8_0(link);
    else if (nInt == 4) PackInit_IntegerType_int_4_1(link); else if (nInt%4 == 0) PackInit_IntegerType_int_4_0(link);
    else if (nInt == 2) PackInit_IntegerType_int_2_1(link); else if (nInt%2 == 0) PackInit_IntegerType_int_2_0(link);
    else if (nInt == 1) PackInit_IntegerType_int_1_1(link); else if (nInt%1 == 0) PackInit_IntegerType_int_1_0(link);
    link->bs        = nInt;
    link->unitbytes = nInt*sizeof(int);
    link->basicunit = MPI_INT;
    if (link->bs == 1) {link->isbuiltin = PETSC_TRUE; link->unit = MPI_INT;}
#endif
  } else if (nSignedChar) {
    if      (nSignedChar == 8) PackInit_IntegerType_SignedChar_8_1(link); else if (nSignedChar%8 == 0) PackInit_IntegerType_SignedChar_8_0(link);
    else if (nSignedChar == 4) PackInit_IntegerType_SignedChar_4_1(link); else if (nSignedChar%4 == 0) PackInit_IntegerType_SignedChar_4_0(link);
    else if (nSignedChar == 2) PackInit_IntegerType_SignedChar_2_1(link); else if (nSignedChar%2 == 0) PackInit_IntegerType_SignedChar_2_0(link);
    else if (nSignedChar == 1) PackInit_IntegerType_SignedChar_1_1(link); else if (nSignedChar%1 == 0) PackInit_IntegerType_SignedChar_1_0(link);
    link->bs        = nSignedChar;
    link->unitbytes = nSignedChar*sizeof(SignedChar);
    link->basicunit = MPI_SIGNED_CHAR;
    if (link->bs == 1) {link->isbuiltin = PETSC_TRUE; link->unit = MPI_SIGNED_CHAR;}
  }  else if (nUnsignedChar) {
    if      (nUnsignedChar == 8) PackInit_IntegerType_UnsignedChar_8_1(link); else if (nUnsignedChar%8 == 0) PackInit_IntegerType_UnsignedChar_8_0(link);
    else if (nUnsignedChar == 4) PackInit_IntegerType_UnsignedChar_4_1(link); else if (nUnsignedChar%4 == 0) PackInit_IntegerType_UnsignedChar_4_0(link);
    else if (nUnsignedChar == 2) PackInit_IntegerType_UnsignedChar_2_1(link); else if (nUnsignedChar%2 == 0) PackInit_IntegerType_UnsignedChar_2_0(link);
    else if (nUnsignedChar == 1) PackInit_IntegerType_UnsignedChar_1_1(link); else if (nUnsignedChar%1 == 0) PackInit_IntegerType_UnsignedChar_1_0(link);
    link->bs        = nUnsignedChar;
    link->unitbytes = nUnsignedChar*sizeof(UnsignedChar);
    link->basicunit = MPI_UNSIGNED_CHAR;
    if (link->bs == 1) {link->isbuiltin = PETSC_TRUE; link->unit = MPI_UNSIGNED_CHAR;}
#if defined(PETSC_HAVE_COMPLEX)
  } else if (nPetscComplex) {
    if      (nPetscComplex == 8) PackInit_ComplexType_PetscComplex_8_1(link); else if (nPetscComplex%8 == 0) PackInit_ComplexType_PetscComplex_8_0(link);
    else if (nPetscComplex == 4) PackInit_ComplexType_PetscComplex_4_1(link); else if (nPetscComplex%4 == 0) PackInit_ComplexType_PetscComplex_4_0(link);
    else if (nPetscComplex == 2) PackInit_ComplexType_PetscComplex_2_1(link); else if (nPetscComplex%2 == 0) PackInit_ComplexType_PetscComplex_2_0(link);
    else if (nPetscComplex == 1) PackInit_ComplexType_PetscComplex_1_1(link); else if (nPetscComplex%1 == 0) PackInit_ComplexType_PetscComplex_1_0(link);
    link->bs        = nPetscComplex;
    link->unitbytes = nPetscComplex*sizeof(PetscComplex);
    link->basicunit = MPIU_COMPLEX;
    if (link->bs == 1) {link->isbuiltin = PETSC_TRUE; link->unit = MPIU_COMPLEX;}
#endif
  } else {
    MPI_Aint lb,nbyte;
    PetscCallMPI(MPI_Type_get_extent(unit,&lb,&nbyte));
    PetscCheck(lb == 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld",(long)lb);
    if (nbyte % sizeof(int)) { /* If the type size is not multiple of int */
      if      (nbyte == 4) PackInit_DumbType_char_4_1(link); else if (nbyte%4 == 0) PackInit_DumbType_char_4_0(link);
      else if (nbyte == 2) PackInit_DumbType_char_2_1(link); else if (nbyte%2 == 0) PackInit_DumbType_char_2_0(link);
      else if (nbyte == 1) PackInit_DumbType_char_1_1(link); else if (nbyte%1 == 0) PackInit_DumbType_char_1_0(link);
      link->bs        = nbyte;
      link->unitbytes = nbyte;
      link->basicunit = MPI_BYTE;
    } else {
      nInt = nbyte / sizeof(int);
      if      (nInt == 8) PackInit_DumbType_DumbInt_8_1(link); else if (nInt%8 == 0) PackInit_DumbType_DumbInt_8_0(link);
      else if (nInt == 4) PackInit_DumbType_DumbInt_4_1(link); else if (nInt%4 == 0) PackInit_DumbType_DumbInt_4_0(link);
      else if (nInt == 2) PackInit_DumbType_DumbInt_2_1(link); else if (nInt%2 == 0) PackInit_DumbType_DumbInt_2_0(link);
      else if (nInt == 1) PackInit_DumbType_DumbInt_1_1(link); else if (nInt%1 == 0) PackInit_DumbType_DumbInt_1_0(link);
      link->bs        = nInt;
      link->unitbytes = nbyte;
      link->basicunit = MPI_INT;
    }
    if (link->isbuiltin) link->unit = unit;
  }

  if (!link->isbuiltin) PetscCallMPI(MPI_Type_dup(unit,&link->unit));

  link->Memcpy = PetscSFLinkMemcpy_Host;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetUnpackAndOp(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*))
{
  PetscFunctionBegin;
  *UnpackAndOp = NULL;
  if (PetscMemTypeHost(mtype)) {
    if      (op == MPI_REPLACE)               *UnpackAndOp = link->h_UnpackAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *UnpackAndOp = link->h_UnpackAndAdd;
    else if (op == MPI_PROD)                  *UnpackAndOp = link->h_UnpackAndMult;
    else if (op == MPI_MAX || op == MPIU_MAX) *UnpackAndOp = link->h_UnpackAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *UnpackAndOp = link->h_UnpackAndMin;
    else if (op == MPI_LAND)                  *UnpackAndOp = link->h_UnpackAndLAND;
    else if (op == MPI_BAND)                  *UnpackAndOp = link->h_UnpackAndBAND;
    else if (op == MPI_LOR)                   *UnpackAndOp = link->h_UnpackAndLOR;
    else if (op == MPI_BOR)                   *UnpackAndOp = link->h_UnpackAndBOR;
    else if (op == MPI_LXOR)                  *UnpackAndOp = link->h_UnpackAndLXOR;
    else if (op == MPI_BXOR)                  *UnpackAndOp = link->h_UnpackAndBXOR;
    else if (op == MPI_MAXLOC)                *UnpackAndOp = link->h_UnpackAndMaxloc;
    else if (op == MPI_MINLOC)                *UnpackAndOp = link->h_UnpackAndMinloc;
  }
#if defined(PETSC_HAVE_DEVICE)
  else if (PetscMemTypeDevice(mtype) && !atomic) {
    if      (op == MPI_REPLACE)               *UnpackAndOp = link->d_UnpackAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *UnpackAndOp = link->d_UnpackAndAdd;
    else if (op == MPI_PROD)                  *UnpackAndOp = link->d_UnpackAndMult;
    else if (op == MPI_MAX || op == MPIU_MAX) *UnpackAndOp = link->d_UnpackAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *UnpackAndOp = link->d_UnpackAndMin;
    else if (op == MPI_LAND)                  *UnpackAndOp = link->d_UnpackAndLAND;
    else if (op == MPI_BAND)                  *UnpackAndOp = link->d_UnpackAndBAND;
    else if (op == MPI_LOR)                   *UnpackAndOp = link->d_UnpackAndLOR;
    else if (op == MPI_BOR)                   *UnpackAndOp = link->d_UnpackAndBOR;
    else if (op == MPI_LXOR)                  *UnpackAndOp = link->d_UnpackAndLXOR;
    else if (op == MPI_BXOR)                  *UnpackAndOp = link->d_UnpackAndBXOR;
    else if (op == MPI_MAXLOC)                *UnpackAndOp = link->d_UnpackAndMaxloc;
    else if (op == MPI_MINLOC)                *UnpackAndOp = link->d_UnpackAndMinloc;
  } else if (PetscMemTypeDevice(mtype) && atomic) {
    if      (op == MPI_REPLACE)               *UnpackAndOp = link->da_UnpackAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *UnpackAndOp = link->da_UnpackAndAdd;
    else if (op == MPI_PROD)                  *UnpackAndOp = link->da_UnpackAndMult;
    else if (op == MPI_MAX || op == MPIU_MAX) *UnpackAndOp = link->da_UnpackAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *UnpackAndOp = link->da_UnpackAndMin;
    else if (op == MPI_LAND)                  *UnpackAndOp = link->da_UnpackAndLAND;
    else if (op == MPI_BAND)                  *UnpackAndOp = link->da_UnpackAndBAND;
    else if (op == MPI_LOR)                   *UnpackAndOp = link->da_UnpackAndLOR;
    else if (op == MPI_BOR)                   *UnpackAndOp = link->da_UnpackAndBOR;
    else if (op == MPI_LXOR)                  *UnpackAndOp = link->da_UnpackAndLXOR;
    else if (op == MPI_BXOR)                  *UnpackAndOp = link->da_UnpackAndBXOR;
    else if (op == MPI_MAXLOC)                *UnpackAndOp = link->da_UnpackAndMaxloc;
    else if (op == MPI_MINLOC)                *UnpackAndOp = link->da_UnpackAndMinloc;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetScatterAndOp(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*))
{
  PetscFunctionBegin;
  *ScatterAndOp = NULL;
  if (PetscMemTypeHost(mtype)) {
    if      (op == MPI_REPLACE)               *ScatterAndOp = link->h_ScatterAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *ScatterAndOp = link->h_ScatterAndAdd;
    else if (op == MPI_PROD)                  *ScatterAndOp = link->h_ScatterAndMult;
    else if (op == MPI_MAX || op == MPIU_MAX) *ScatterAndOp = link->h_ScatterAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *ScatterAndOp = link->h_ScatterAndMin;
    else if (op == MPI_LAND)                  *ScatterAndOp = link->h_ScatterAndLAND;
    else if (op == MPI_BAND)                  *ScatterAndOp = link->h_ScatterAndBAND;
    else if (op == MPI_LOR)                   *ScatterAndOp = link->h_ScatterAndLOR;
    else if (op == MPI_BOR)                   *ScatterAndOp = link->h_ScatterAndBOR;
    else if (op == MPI_LXOR)                  *ScatterAndOp = link->h_ScatterAndLXOR;
    else if (op == MPI_BXOR)                  *ScatterAndOp = link->h_ScatterAndBXOR;
    else if (op == MPI_MAXLOC)                *ScatterAndOp = link->h_ScatterAndMaxloc;
    else if (op == MPI_MINLOC)                *ScatterAndOp = link->h_ScatterAndMinloc;
  }
#if defined(PETSC_HAVE_DEVICE)
  else if (PetscMemTypeDevice(mtype) && !atomic) {
    if      (op == MPI_REPLACE)               *ScatterAndOp = link->d_ScatterAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *ScatterAndOp = link->d_ScatterAndAdd;
    else if (op == MPI_PROD)                  *ScatterAndOp = link->d_ScatterAndMult;
    else if (op == MPI_MAX || op == MPIU_MAX) *ScatterAndOp = link->d_ScatterAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *ScatterAndOp = link->d_ScatterAndMin;
    else if (op == MPI_LAND)                  *ScatterAndOp = link->d_ScatterAndLAND;
    else if (op == MPI_BAND)                  *ScatterAndOp = link->d_ScatterAndBAND;
    else if (op == MPI_LOR)                   *ScatterAndOp = link->d_ScatterAndLOR;
    else if (op == MPI_BOR)                   *ScatterAndOp = link->d_ScatterAndBOR;
    else if (op == MPI_LXOR)                  *ScatterAndOp = link->d_ScatterAndLXOR;
    else if (op == MPI_BXOR)                  *ScatterAndOp = link->d_ScatterAndBXOR;
    else if (op == MPI_MAXLOC)                *ScatterAndOp = link->d_ScatterAndMaxloc;
    else if (op == MPI_MINLOC)                *ScatterAndOp = link->d_ScatterAndMinloc;
  } else if (PetscMemTypeDevice(mtype) && atomic) {
    if      (op == MPI_REPLACE)               *ScatterAndOp = link->da_ScatterAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *ScatterAndOp = link->da_ScatterAndAdd;
    else if (op == MPI_PROD)                  *ScatterAndOp = link->da_ScatterAndMult;
    else if (op == MPI_MAX || op == MPIU_MAX) *ScatterAndOp = link->da_ScatterAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *ScatterAndOp = link->da_ScatterAndMin;
    else if (op == MPI_LAND)                  *ScatterAndOp = link->da_ScatterAndLAND;
    else if (op == MPI_BAND)                  *ScatterAndOp = link->da_ScatterAndBAND;
    else if (op == MPI_LOR)                   *ScatterAndOp = link->da_ScatterAndLOR;
    else if (op == MPI_BOR)                   *ScatterAndOp = link->da_ScatterAndBOR;
    else if (op == MPI_LXOR)                  *ScatterAndOp = link->da_ScatterAndLXOR;
    else if (op == MPI_BXOR)                  *ScatterAndOp = link->da_ScatterAndBXOR;
    else if (op == MPI_MAXLOC)                *ScatterAndOp = link->da_ScatterAndMaxloc;
    else if (op == MPI_MINLOC)                *ScatterAndOp = link->da_ScatterAndMinloc;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetFetchAndOp(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**FetchAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOp = NULL;
  PetscCheck(op == MPI_SUM || op == MPIU_SUM,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op in FetchAndOp");
  if (PetscMemTypeHost(mtype)) *FetchAndOp = link->h_FetchAndAdd;
#if defined(PETSC_HAVE_DEVICE)
  else if (PetscMemTypeDevice(mtype) && !atomic) *FetchAndOp = link->d_FetchAndAdd;
  else if (PetscMemTypeDevice(mtype) && atomic)  *FetchAndOp = link->da_FetchAndAdd;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetFetchAndOpLocal(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**FetchAndOpLocal)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOpLocal = NULL;
  PetscCheck(op == MPI_SUM || op == MPIU_SUM,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op in FetchAndOp");
  if (PetscMemTypeHost(mtype)) *FetchAndOpLocal = link->h_FetchAndAddLocal;
#if defined(PETSC_HAVE_DEVICE)
  else if (PetscMemTypeDevice(mtype) && !atomic) *FetchAndOpLocal = link->d_FetchAndAddLocal;
  else if (PetscMemTypeDevice(mtype) && atomic)  *FetchAndOpLocal = link->da_FetchAndAddLocal;
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscSFLinkLogFlopsAfterUnpackRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,MPI_Op op)
{
  PetscLogDouble flops;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (op != MPI_REPLACE && link->basicunit == MPIU_SCALAR) { /* op is a reduction on PetscScalars */
    flops = bas->rootbuflen[scope]*link->bs; /* # of roots in buffer x # of scalars in unit */
#if defined(PETSC_HAVE_DEVICE)
    if (PetscMemTypeDevice(link->rootmtype)) PetscCall(PetscLogGpuFlops(flops)); else
#endif
    PetscCall(PetscLogFlops(flops));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscSFLinkLogFlopsAfterUnpackLeafData(PetscSF sf,PetscSFLink link,PetscSFScope scope,MPI_Op op)
{
  PetscLogDouble flops;

  PetscFunctionBegin;
  if (op != MPI_REPLACE && link->basicunit == MPIU_SCALAR) { /* op is a reduction on PetscScalars */
    flops = sf->leafbuflen[scope]*link->bs; /* # of roots in buffer x # of scalars in unit */
#if defined(PETSC_HAVE_DEVICE)
    if (PetscMemTypeDevice(link->leafmtype)) PetscCall(PetscLogGpuFlops(flops)); else
#endif
    PetscCall(PetscLogFlops(flops));
  }
  PetscFunctionReturn(0);
}

/* When SF could not find a proper UnpackAndOp() from link, it falls back to MPI_Reduce_local.
  Input Parameters:
  +sf      - The StarForest
  .link    - The link
  .count   - Number of entries to unpack
  .start   - The first index, significent when indices=NULL
  .indices - Indices of entries in <data>. If NULL, it means indices are contiguous and the first is given in <start>
  .buf     - A contiguous buffer to unpack from
  -op      - Operation after unpack

  Output Parameters:
  .data    - The data to unpack to
*/
static inline PetscErrorCode PetscSFLinkUnpackDataWithMPIReduceLocal(PetscSF sf,PetscSFLink link,PetscInt count,PetscInt start,const PetscInt *indices,void *data,const void *buf,MPI_Op op)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  {
    PetscInt       i;
    if (indices) {
      /* Note we use link->unit instead of link->basicunit. When op can be mapped to MPI_SUM etc, it operates on
         basic units of a root/leaf element-wisely. Otherwise, it is meant to operate on a whole root/leaf.
      */
      for (i=0; i<count; i++) PetscCallMPI(MPI_Reduce_local((const char*)buf+i*link->unitbytes,(char*)data+indices[i]*link->unitbytes,1,link->unit,op));
    } else {
      PetscCallMPI(MPIU_Reduce_local(buf,(char*)data+start*link->unitbytes,count,link->unit,op));
    }
  }
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
#endif
}

static inline PetscErrorCode PetscSFLinkScatterDataWithMPIReduceLocal(PetscSF sf,PetscSFLink link,PetscInt count,PetscInt srcStart,const PetscInt *srcIdx,const void *src,PetscInt dstStart,const PetscInt *dstIdx,void *dst,MPI_Op op)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  {
    PetscInt       i,disp;
    if (!srcIdx) {
      PetscCall(PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,dstStart,dstIdx,dst,(const char*)src+srcStart*link->unitbytes,op));
    } else {
      for (i=0; i<count; i++) {
        disp = dstIdx? dstIdx[i] : dstStart + i;
        PetscCallMPI(MPIU_Reduce_local((const char*)src+srcIdx[i]*link->unitbytes,(char*)dst+disp*link->unitbytes,1,link->unit,op));
      }
    }
  }
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
#endif
}

/*=============================================================================
              Pack/Unpack/Fetch/Scatter routines
 ============================================================================*/

/* Pack rootdata to rootbuf
  Input Parameters:
  + sf       - The SF this packing works on.
  . link     - It gives the memtype of the roots and also provides root buffer.
  . scope    - PETSCSF_LOCAL or PETSCSF_REMOTE. Note SF has the ability to do local and remote communications separately.
  - rootdata - Where to read the roots.

  Notes:
  When rootdata can be directly used as root buffer, the routine is almost a no-op. After the call, root data is
  in a place where the underlying MPI is ready to access (use_gpu_aware_mpi or not)
 */
PetscErrorCode PetscSFLinkPackRootData_Private(PetscSF sf,PetscSFLink link,PetscSFScope scope,const void *rootdata)
{
  const PetscInt   *rootindices = NULL;
  PetscInt         count,start;
  PetscErrorCode   (*Pack)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*) = NULL;
  PetscMemType     rootmtype = link->rootmtype;
  PetscSFPackOpt   opt = NULL;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0));
  if (!link->rootdirect[scope]) { /* If rootdata works directly as rootbuf, skip packing */
    PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,rootmtype,scope,&count,&start,&opt,&rootindices));
    PetscCall(PetscSFLinkGetPack(link,rootmtype,&Pack));
    PetscCall((*Pack)(link,count,start,opt,rootindices,rootdata,link->rootbuf[scope][rootmtype]));
  }
  PetscCall(PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0));
  PetscFunctionReturn(0);
}

/* Pack leafdata to leafbuf */
PetscErrorCode PetscSFLinkPackLeafData_Private(PetscSF sf,PetscSFLink link,PetscSFScope scope,const void *leafdata)
{
  const PetscInt   *leafindices = NULL;
  PetscInt         count,start;
  PetscErrorCode   (*Pack)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*) = NULL;
  PetscMemType     leafmtype = link->leafmtype;
  PetscSFPackOpt   opt = NULL;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0));
  if (!link->leafdirect[scope]) { /* If leafdata works directly as rootbuf, skip packing */
    PetscCall(PetscSFLinkGetLeafPackOptAndIndices(sf,link,leafmtype,scope,&count,&start,&opt,&leafindices));
    PetscCall(PetscSFLinkGetPack(link,leafmtype,&Pack));
    PetscCall((*Pack)(link,count,start,opt,leafindices,leafdata,link->leafbuf[scope][leafmtype]));
  }
  PetscCall(PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0));
  PetscFunctionReturn(0);
}

/* Pack rootdata to rootbuf, which are in the same memory space */
PetscErrorCode PetscSFLinkPackRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,const void *rootdata)
{
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (scope == PETSCSF_REMOTE) { /* Sync the device if rootdata is not on petsc default stream */
    if (PetscMemTypeDevice(link->rootmtype) && link->SyncDevice && sf->unknown_input_stream) PetscCall((*link->SyncDevice)(link));
    if (link->PrePack) PetscCall((*link->PrePack)(sf,link,PETSCSF_ROOT2LEAF)); /* Used by SF nvshmem */
  }
  PetscCall(PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0));
  if (bas->rootbuflen[scope]) PetscCall(PetscSFLinkPackRootData_Private(sf,link,scope,rootdata));
  PetscCall(PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0));
  PetscFunctionReturn(0);
}
/* Pack leafdata to leafbuf, which are in the same memory space */
PetscErrorCode PetscSFLinkPackLeafData(PetscSF sf,PetscSFLink link,PetscSFScope scope,const void *leafdata)
{
  PetscFunctionBegin;
  if (scope == PETSCSF_REMOTE) {
    if (PetscMemTypeDevice(link->leafmtype) && link->SyncDevice && sf->unknown_input_stream) PetscCall((*link->SyncDevice)(link));
    if (link->PrePack) PetscCall((*link->PrePack)(sf,link,PETSCSF_LEAF2ROOT));  /* Used by SF nvshmem */
  }
  PetscCall(PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0));
  if (sf->leafbuflen[scope]) PetscCall(PetscSFLinkPackLeafData_Private(sf,link,scope,leafdata));
  PetscCall(PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkUnpackRootData_Private(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *rootdata,MPI_Op op)
{
  const PetscInt   *rootindices = NULL;
  PetscInt         count,start;
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode   (*UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*) = NULL;
  PetscMemType     rootmtype = link->rootmtype;
  PetscSFPackOpt   opt = NULL;

  PetscFunctionBegin;
  if (!link->rootdirect[scope]) { /* If rootdata works directly as rootbuf, skip unpacking */
    PetscCall(PetscSFLinkGetUnpackAndOp(link,rootmtype,op,bas->rootdups[scope],&UnpackAndOp));
    if (UnpackAndOp) {
      PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,rootmtype,scope,&count,&start,&opt,&rootindices));
      PetscCall((*UnpackAndOp)(link,count,start,opt,rootindices,rootdata,link->rootbuf[scope][rootmtype]));
    } else {
      PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,PETSC_MEMTYPE_HOST,scope,&count,&start,&opt,&rootindices));
      PetscCall(PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,start,rootindices,rootdata,link->rootbuf[scope][rootmtype],op));
    }
  }
  PetscCall(PetscSFLinkLogFlopsAfterUnpackRootData(sf,link,scope,op));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkUnpackLeafData_Private(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *leafdata,MPI_Op op)
{
  const PetscInt   *leafindices = NULL;
  PetscInt         count,start;
  PetscErrorCode   (*UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*) = NULL;
  PetscMemType     leafmtype = link->leafmtype;
  PetscSFPackOpt   opt = NULL;

  PetscFunctionBegin;
  if (!link->leafdirect[scope]) { /* If leafdata works directly as rootbuf, skip unpacking */
    PetscCall(PetscSFLinkGetUnpackAndOp(link,leafmtype,op,sf->leafdups[scope],&UnpackAndOp));
    if (UnpackAndOp) {
      PetscCall(PetscSFLinkGetLeafPackOptAndIndices(sf,link,leafmtype,scope,&count,&start,&opt,&leafindices));
      PetscCall((*UnpackAndOp)(link,count,start,opt,leafindices,leafdata,link->leafbuf[scope][leafmtype]));
    } else {
      PetscCall(PetscSFLinkGetLeafPackOptAndIndices(sf,link,PETSC_MEMTYPE_HOST,scope,&count,&start,&opt,&leafindices));
      PetscCall(PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,start,leafindices,leafdata,link->leafbuf[scope][leafmtype],op));
    }
  }
  PetscCall(PetscSFLinkLogFlopsAfterUnpackLeafData(sf,link,scope,op));
  PetscFunctionReturn(0);
}
/* Unpack rootbuf to rootdata, which are in the same memory space */
PetscErrorCode PetscSFLinkUnpackRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *rootdata,MPI_Op op)
{
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0));
  if (bas->rootbuflen[scope]) PetscCall(PetscSFLinkUnpackRootData_Private(sf,link,scope,rootdata,op));
  PetscCall(PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0));
  if (scope == PETSCSF_REMOTE) {
    if (link->PostUnpack) PetscCall((*link->PostUnpack)(sf,link,PETSCSF_LEAF2ROOT));  /* Used by SF nvshmem */
    if (PetscMemTypeDevice(link->rootmtype) && link->SyncDevice && sf->unknown_input_stream) PetscCall((*link->SyncDevice)(link));
  }
  PetscFunctionReturn(0);
}

/* Unpack leafbuf to leafdata for remote (common case) or local (rare case when rootmtype != leafmtype) */
PetscErrorCode PetscSFLinkUnpackLeafData(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *leafdata,MPI_Op op)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0));
  if (sf->leafbuflen[scope]) PetscCall(PetscSFLinkUnpackLeafData_Private(sf,link,scope,leafdata,op));
  PetscCall(PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0));
  if (scope == PETSCSF_REMOTE) {
    if (link->PostUnpack) PetscCall((*link->PostUnpack)(sf,link,PETSCSF_ROOT2LEAF));  /* Used by SF nvshmem */
    if (PetscMemTypeDevice(link->leafmtype) && link->SyncDevice && sf->unknown_input_stream) PetscCall((*link->SyncDevice)(link));
  }
  PetscFunctionReturn(0);
}

/* FetchAndOp rootdata with rootbuf, it is a kind of Unpack on rootdata, except it also updates rootbuf */
PetscErrorCode PetscSFLinkFetchAndOpRemote(PetscSF sf,PetscSFLink link,void *rootdata,MPI_Op op)
{
  const PetscInt     *rootindices = NULL;
  PetscInt           count,start;
  PetscSF_Basic      *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode     (*FetchAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,void*) = NULL;
  PetscMemType       rootmtype = link->rootmtype;
  PetscSFPackOpt     opt = NULL;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0));
  if (bas->rootbuflen[PETSCSF_REMOTE]) {
    /* Do FetchAndOp on rootdata with rootbuf */
    PetscCall(PetscSFLinkGetFetchAndOp(link,rootmtype,op,bas->rootdups[PETSCSF_REMOTE],&FetchAndOp));
    PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,rootmtype,PETSCSF_REMOTE,&count,&start,&opt,&rootindices));
    PetscCall((*FetchAndOp)(link,count,start,opt,rootindices,rootdata,link->rootbuf[PETSCSF_REMOTE][rootmtype]));
  }
  PetscCall(PetscSFLinkLogFlopsAfterUnpackRootData(sf,link,PETSCSF_REMOTE,op));
  PetscCall(PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkScatterLocal(PetscSF sf,PetscSFLink link,PetscSFDirection direction,void *rootdata,void *leafdata,MPI_Op op)
{
  const PetscInt       *rootindices = NULL,*leafindices = NULL;
  PetscInt             count,rootstart,leafstart;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode       (*ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*) = NULL;
  PetscMemType         rootmtype = link->rootmtype,leafmtype = link->leafmtype,srcmtype,dstmtype;
  PetscSFPackOpt       leafopt = NULL,rootopt = NULL;
  PetscInt             buflen = sf->leafbuflen[PETSCSF_LOCAL];
  char                 *srcbuf = NULL,*dstbuf = NULL;
  PetscBool            dstdups;

  PetscFunctionBegin;
  if (!buflen) PetscFunctionReturn(0);
  if (rootmtype != leafmtype) { /* The cross memory space local scatter is done by pack, copy and unpack */
    if (direction == PETSCSF_ROOT2LEAF) {
      PetscCall(PetscSFLinkPackRootData(sf,link,PETSCSF_LOCAL,rootdata));
      srcmtype = rootmtype;
      srcbuf   = link->rootbuf[PETSCSF_LOCAL][rootmtype];
      dstmtype = leafmtype;
      dstbuf   = link->leafbuf[PETSCSF_LOCAL][leafmtype];
    } else {
      PetscCall(PetscSFLinkPackLeafData(sf,link,PETSCSF_LOCAL,leafdata));
      srcmtype = leafmtype;
      srcbuf   = link->leafbuf[PETSCSF_LOCAL][leafmtype];
      dstmtype = rootmtype;
      dstbuf   = link->rootbuf[PETSCSF_LOCAL][rootmtype];
    }
    PetscCall((*link->Memcpy)(link,dstmtype,dstbuf,srcmtype,srcbuf,buflen*link->unitbytes));
    /* If above is a device to host copy, we have to sync the stream before accessing the buffer on host */
    if (PetscMemTypeHost(dstmtype)) PetscCall((*link->SyncStream)(link));
    if (direction == PETSCSF_ROOT2LEAF) {
      PetscCall(PetscSFLinkUnpackLeafData(sf,link,PETSCSF_LOCAL,leafdata,op));
    } else {
      PetscCall(PetscSFLinkUnpackRootData(sf,link,PETSCSF_LOCAL,rootdata,op));
    }
  } else {
    dstdups  = (direction == PETSCSF_ROOT2LEAF) ? sf->leafdups[PETSCSF_LOCAL] : bas->rootdups[PETSCSF_LOCAL];
    dstmtype = (direction == PETSCSF_ROOT2LEAF) ? link->leafmtype : link->rootmtype;
    PetscCall(PetscSFLinkGetScatterAndOp(link,dstmtype,op,dstdups,&ScatterAndOp));
    if (ScatterAndOp) {
      PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,rootmtype,PETSCSF_LOCAL,&count,&rootstart,&rootopt,&rootindices));
      PetscCall(PetscSFLinkGetLeafPackOptAndIndices(sf,link,leafmtype,PETSCSF_LOCAL,&count,&leafstart,&leafopt,&leafindices));
      if (direction == PETSCSF_ROOT2LEAF) {
        PetscCall((*ScatterAndOp)(link,count,rootstart,rootopt,rootindices,rootdata,leafstart,leafopt,leafindices,leafdata));
      } else {
        PetscCall((*ScatterAndOp)(link,count,leafstart,leafopt,leafindices,leafdata,rootstart,rootopt,rootindices,rootdata));
      }
    } else {
      PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,&count,&rootstart,&rootopt,&rootindices));
      PetscCall(PetscSFLinkGetLeafPackOptAndIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,&count,&leafstart,&leafopt,&leafindices));
      if (direction == PETSCSF_ROOT2LEAF) {
        PetscCall(PetscSFLinkScatterDataWithMPIReduceLocal(sf,link,count,rootstart,rootindices,rootdata,leafstart,leafindices,leafdata,op));
      } else {
        PetscCall(PetscSFLinkScatterDataWithMPIReduceLocal(sf,link,count,leafstart,leafindices,leafdata,rootstart,rootindices,rootdata,op));
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Fetch rootdata to leafdata and leafupdate locally */
PetscErrorCode PetscSFLinkFetchAndOpLocal(PetscSF sf,PetscSFLink link,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  const PetscInt       *rootindices = NULL,*leafindices = NULL;
  PetscInt             count,rootstart,leafstart;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode       (*FetchAndOpLocal)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*) = NULL;
  const PetscMemType   rootmtype = link->rootmtype,leafmtype = link->leafmtype;
  PetscSFPackOpt       leafopt = NULL,rootopt = NULL;

  PetscFunctionBegin;
  if (!bas->rootbuflen[PETSCSF_LOCAL]) PetscFunctionReturn(0);
  if (rootmtype != leafmtype) {
    /* The local communication has to go through pack and unpack */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Doing PetscSFFetchAndOp with rootdata and leafdata on opposite side of CPU and GPU");
  } else {
    PetscCall(PetscSFLinkGetRootPackOptAndIndices(sf,link,rootmtype,PETSCSF_LOCAL,&count,&rootstart,&rootopt,&rootindices));
    PetscCall(PetscSFLinkGetLeafPackOptAndIndices(sf,link,leafmtype,PETSCSF_LOCAL,&count,&leafstart,&leafopt,&leafindices));
    PetscCall(PetscSFLinkGetFetchAndOpLocal(link,rootmtype,op,bas->rootdups[PETSCSF_LOCAL],&FetchAndOpLocal));
    PetscCall((*FetchAndOpLocal)(link,count,rootstart,rootopt,rootindices,rootdata,leafstart,leafopt,leafindices,leafdata,leafupdate));
  }
  PetscFunctionReturn(0);
}

/*
  Create per-rank pack/unpack optimizations based on indice patterns

   Input Parameters:
  +  n       - Number of destination ranks
  .  offset  - [n+1] For the i-th rank, its associated indices are idx[offset[i], offset[i+1]). offset[0] needs not to be 0.
  -  idx     - [*]   Array storing indices

   Output Parameters:
  +  opt     - Pack optimizations. NULL if no optimizations.
*/
PetscErrorCode PetscSFCreatePackOpt(PetscInt n,const PetscInt *offset,const PetscInt *idx,PetscSFPackOpt *out)
{
  PetscInt       r,p,start,i,j,k,dx,dy,dz,dydz,m,X,Y;
  PetscBool      optimizable = PETSC_TRUE;
  PetscSFPackOpt opt;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(1,&opt));
  PetscCall(PetscMalloc1(7*n+2,&opt->array));
  opt->n      = opt->array[0] = n;
  opt->offset = opt->array + 1;
  opt->start  = opt->array + n   + 2;
  opt->dx     = opt->array + 2*n + 2;
  opt->dy     = opt->array + 3*n + 2;
  opt->dz     = opt->array + 4*n + 2;
  opt->X      = opt->array + 5*n + 2;
  opt->Y      = opt->array + 6*n + 2;

  for (r=0; r<n; r++) { /* For each destination rank */
    m     = offset[r+1] - offset[r]; /* Total number of indices for this rank. We want to see if m can be factored into dx*dy*dz */
    p     = offset[r];
    start = idx[p]; /* First index for this rank */
    p++;

    /* Search in X dimension */
    for (dx=1; dx<m; dx++,p++) {
      if (start+dx != idx[p]) break;
    }

    dydz = m/dx;
    X    = dydz > 1 ? (idx[p]-start) : dx;
    /* Not optimizable if m is not a multiple of dx, or some unrecognized pattern is found */
    if (m%dx || X <= 0) {optimizable = PETSC_FALSE; goto finish;}
    for (dy=1; dy<dydz; dy++) { /* Search in Y dimension */
      for (i=0; i<dx; i++,p++) {
        if (start+X*dy+i != idx[p]) {
          if (i) {optimizable = PETSC_FALSE; goto finish;} /* The pattern is violated in the middle of an x-walk */
          else goto Z_dimension;
        }
      }
    }

Z_dimension:
    dz = m/(dx*dy);
    Y  = dz > 1 ? (idx[p]-start)/X : dy;
    /* Not optimizable if m is not a multiple of dx*dy, or some unrecognized pattern is found */
    if (m%(dx*dy) || Y <= 0) {optimizable = PETSC_FALSE; goto finish;}
    for (k=1; k<dz; k++) { /* Go through Z dimension to see if remaining indices follow the pattern */
      for (j=0; j<dy; j++) {
        for (i=0; i<dx; i++,p++) {
          if (start+X*Y*k+X*j+i != idx[p]) {optimizable = PETSC_FALSE; goto finish;}
        }
      }
    }
    opt->start[r] = start;
    opt->dx[r]    = dx;
    opt->dy[r]    = dy;
    opt->dz[r]    = dz;
    opt->X[r]     = X;
    opt->Y[r]     = Y;
  }

finish:
  /* If not optimizable, free arrays to save memory */
  if (!n || !optimizable) {
    PetscCall(PetscFree(opt->array));
    PetscCall(PetscFree(opt));
    *out = NULL;
  } else {
    opt->offset[0] = 0;
    for (r=0; r<n; r++) opt->offset[r+1] = opt->offset[r] + opt->dx[r]*opt->dy[r]*opt->dz[r];
    *out = opt;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscSFDestroyPackOpt(PetscSF sf,PetscMemType mtype,PetscSFPackOpt *out)
{
  PetscSFPackOpt opt = *out;

  PetscFunctionBegin;
  if (opt) {
    PetscCall(PetscSFFree(sf,mtype,opt->array));
    PetscCall(PetscFree(opt));
    *out = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFSetUpPackFields(PetscSF sf)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  /* [0] for PETSCSF_LOCAL and [1] for PETSCSF_REMOTE in the following */
  for (i=0; i<2; i++) { /* Set defaults */
    sf->leafstart[i]   = 0;
    sf->leafcontig[i]  = PETSC_TRUE;
    sf->leafdups[i]    = PETSC_FALSE;
    bas->rootstart[i]  = 0;
    bas->rootcontig[i] = PETSC_TRUE;
    bas->rootdups[i]   = PETSC_FALSE;
  }

  sf->leafbuflen[0] = sf->roffset[sf->ndranks];
  sf->leafbuflen[1] = sf->roffset[sf->nranks] - sf->roffset[sf->ndranks];

  if (sf->leafbuflen[0]) sf->leafstart[0] = sf->rmine[0];
  if (sf->leafbuflen[1]) sf->leafstart[1] = sf->rmine[sf->roffset[sf->ndranks]];

  /* Are leaf indices for self and remote contiguous? If yes, it is best for pack/unpack */
  for (i=0; i<sf->roffset[sf->ndranks]; i++) { /* self */
    if (sf->rmine[i] != sf->leafstart[0]+i) {sf->leafcontig[0] = PETSC_FALSE; break;}
  }
  for (i=sf->roffset[sf->ndranks],j=0; i<sf->roffset[sf->nranks]; i++,j++) { /* remote */
    if (sf->rmine[i] != sf->leafstart[1]+j) {sf->leafcontig[1] = PETSC_FALSE; break;}
  }

  /* If not, see if we can have per-rank optimizations by doing index analysis */
  if (!sf->leafcontig[0]) PetscCall(PetscSFCreatePackOpt(sf->ndranks,            sf->roffset,             sf->rmine, &sf->leafpackopt[0]));
  if (!sf->leafcontig[1]) PetscCall(PetscSFCreatePackOpt(sf->nranks-sf->ndranks, sf->roffset+sf->ndranks, sf->rmine, &sf->leafpackopt[1]));

  /* Are root indices for self and remote contiguous? */
  bas->rootbuflen[0] = bas->ioffset[bas->ndiranks];
  bas->rootbuflen[1] = bas->ioffset[bas->niranks] - bas->ioffset[bas->ndiranks];

  if (bas->rootbuflen[0]) bas->rootstart[0] = bas->irootloc[0];
  if (bas->rootbuflen[1]) bas->rootstart[1] = bas->irootloc[bas->ioffset[bas->ndiranks]];

  for (i=0; i<bas->ioffset[bas->ndiranks]; i++) {
    if (bas->irootloc[i] != bas->rootstart[0]+i) {bas->rootcontig[0] = PETSC_FALSE; break;}
  }
  for (i=bas->ioffset[bas->ndiranks],j=0; i<bas->ioffset[bas->niranks]; i++,j++) {
    if (bas->irootloc[i] != bas->rootstart[1]+j) {bas->rootcontig[1] = PETSC_FALSE; break;}
  }

  if (!bas->rootcontig[0]) PetscCall(PetscSFCreatePackOpt(bas->ndiranks,              bas->ioffset,               bas->irootloc, &bas->rootpackopt[0]));
  if (!bas->rootcontig[1]) PetscCall(PetscSFCreatePackOpt(bas->niranks-bas->ndiranks, bas->ioffset+bas->ndiranks, bas->irootloc, &bas->rootpackopt[1]));

    /* Check dups in indices so that CUDA unpacking kernels can use cheaper regular instructions instead of atomics when they know there are no data race chances */
  if (PetscDefined(HAVE_DEVICE)) {
    PetscBool ismulti = (sf->multi == sf) ? PETSC_TRUE : PETSC_FALSE;
    if (!sf->leafcontig[0]  && !ismulti) PetscCall(PetscCheckDupsInt(sf->leafbuflen[0],  sf->rmine,                                 &sf->leafdups[0]));
    if (!sf->leafcontig[1]  && !ismulti) PetscCall(PetscCheckDupsInt(sf->leafbuflen[1],  sf->rmine+sf->roffset[sf->ndranks],        &sf->leafdups[1]));
    if (!bas->rootcontig[0] && !ismulti) PetscCall(PetscCheckDupsInt(bas->rootbuflen[0], bas->irootloc,                             &bas->rootdups[0]));
    if (!bas->rootcontig[1] && !ismulti) PetscCall(PetscCheckDupsInt(bas->rootbuflen[1], bas->irootloc+bas->ioffset[bas->ndiranks], &bas->rootdups[1]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFResetPackFields(PetscSF sf)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    PetscCall(PetscSFDestroyPackOpt(sf,PETSC_MEMTYPE_HOST,&sf->leafpackopt[i]));
    PetscCall(PetscSFDestroyPackOpt(sf,PETSC_MEMTYPE_HOST,&bas->rootpackopt[i]));
   #if defined(PETSC_HAVE_DEVICE)
    PetscCall(PetscSFDestroyPackOpt(sf,PETSC_MEMTYPE_DEVICE,&sf->leafpackopt_d[i]));
    PetscCall(PetscSFDestroyPackOpt(sf,PETSC_MEMTYPE_DEVICE,&bas->rootpackopt_d[i]));
   #endif
  }
  PetscFunctionReturn(0);
}
