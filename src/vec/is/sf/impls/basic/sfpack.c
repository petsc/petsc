
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

#if defined(PETSC_HAVE_CUDA)
#include <cuda_runtime.h>
#endif
/*
 * MPI_Reduce_local is not really useful because it can't handle sparse data and it vectorizes "in the wrong direction",
 * therefore we pack data types manually. This file defines packing routines for the standard data types.
 */

#define CPPJoin4(a,b,c,d)  a##_##b##_##c##_##d

/* Operations working like s += t */
#define OP_BINARY(op,s,t)   do {(s) = (s) op (t);  } while(0)      /* binary ops in the middle such as +, *, && etc. */
#define OP_FUNCTION(op,s,t) do {(s) = op((s),(t)); } while(0)      /* ops like a function, such as PetscMax, PetscMin */
#define OP_LXOR(op,s,t)     do {(s) = (!(s)) != (!(t));} while(0)  /* logical exclusive OR */
#define OP_ASSIGN(op,s,t)   do {(s) = (t);} while(0)
/* Ref MPI MAXLOC */
#define OP_XLOC(op,s,t) \
  do {                                       \
    if ((s).u == (t).u) (s).i = PetscMin((s).i,(t).i); \
    else if (!((s).u op (t).u)) s = t;           \
  } while(0)

/* DEF_PackFunc - macro defining a Pack routine

   Arguments of the macro:
   +Type      Type of the basic data in an entry, i.e., int, PetscInt, PetscReal etc. It is not the type of an entry.
   .BS        Block size for vectorization. It is a factor of bs.
   -EQ        (bs == BS) ? 1 : 0. EQ is a compile-time const to help compiler optimizations. See below.

   Arguments of the Pack routine:
   +count     Number of indices in idx[].
   .start     If indices are contiguous, it is the first index; otherwise, not used.
   .idx       Indices of entries to packed. NULL means contiguous indices, that is [start,start+count).
   .link      Provide a context for the current call, such as link->bs, number of basic types in an entry. Ex. if unit is MPI_2INT, then bs=2 and the basic type is int.
   .opt       Per-remote pack optimization plan. NULL means no such plan.
   .unpacked  Address of the unpacked data. The entries will be packed are unpacked[idx[i]],for i in [0,count).
   -packed    Address of the packed data.
 */
#define DEF_PackFunc(Type,BS,EQ) \
  static PetscErrorCode CPPJoin4(Pack,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,const PetscInt *idx,PetscSFPackOpt opt,const void *unpacked,void *packed) \
  {                                                                                                          \
    PetscErrorCode ierr;                                                                                     \
    const Type     *u = (const Type*)unpacked,*u2;                                                           \
    Type           *p = (Type*)packed,*p2;                                                                   \
    PetscInt       i,j,k,l,r,step,bs=link->bs;                                                               \
    const PetscInt *idx2,M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */    \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {ierr = PetscArraycpy(p,u+start*bs,MBS*count);CHKERRQ(ierr);}  /* Indices are contiguous */    \
    else if (!opt) { /* No optimizations available */                                                        \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)     /* Decent compilers should eliminate this loop when M = const 1 */           \
          for (k=0; k<BS; k++)  /* Compiler either unrolls (BS=1) or vectorizes (BS=2,4,8,etc) this loop */  \
            p[i*MBS+j*BS+k] = u[idx[i]*MBS+j*BS+k];                                                          \
    } else {                                                                                                 \
      for (r=0; r<opt->n; r++) {                                                                             \
        p2  = p + opt->offset[r]*MBS;                                                                        \
        if (opt->type[r] == PETSCSF_PACKOPT_NONE) {                                                          \
          idx2 = idx + opt->offset[r];                                                                       \
          for (i=0; i<opt->offset[r+1]-opt->offset[r]; i++)                                                  \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++)                                                                           \
                p2[i*MBS+j*BS+k] = u[idx2[i]*MBS+j*BS+k];                                                    \
        } else if (opt->type[r] == PETSCSF_PACKOPT_MULTICOPY) {                                              \
          for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) {                                        \
            u2   = u + idx[opt->copy_start[i]]*MBS;                                                          \
            l    = opt->copy_length[i]*MBS; /* length in basic type such as MPI_INT */                       \
            ierr = PetscArraycpy(p2,u2,l);CHKERRQ(ierr);                                                     \
            p2  += l;                                                                                        \
          }                                                                                                  \
        } else if (opt->type[r] == PETSCSF_PACKOPT_STRIDE) {                                                 \
          u2   = u + idx[opt->offset[r]]*MBS;                                                                \
          step = opt->stride_step[r];                                                                        \
          for (i=0; i<opt->stride_n[r]; i++)                                                                 \
            for (j=0; j<MBS; j++) p2[i*MBS+j] = u2[i*step*MBS+j];                                            \
        } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unknown SFPack optimzation type %D",opt->type[r]);   \
      }                                                                                                      \
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
  static PetscErrorCode CPPJoin4(UnpackAndInsert,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,const PetscInt *idx,PetscSFPackOpt opt,void *unpacked,const void *packed) \
  {                                                                                                          \
    PetscErrorCode ierr;                                                                                     \
    Type           *u = (Type*)unpacked,*u2;                                                                 \
    const Type     *p = (const Type*)packed,*p2;                                                             \
    PetscInt       i,j,k,l,r,step,bs=link->bs;                                                               \
    const PetscInt *idx2,M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */    \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      u2 = u + start*bs;                                                                                     \
      if (u2 != p) {ierr = PetscArraycpy(u2,p,count*MBS);CHKERRQ(ierr);}                                     \
    } else if (!opt) { /* No optimizations available */                                                      \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) u[idx[i]*MBS+j*BS+k] = p[i*MBS+j*BS+k];                                       \
    } else {                                                                                                 \
      for (r=0; r<opt->n; r++) {                                                                             \
        p2 = p + opt->offset[r]*MBS;                                                                         \
        if (opt->type[r] == PETSCSF_PACKOPT_NONE) {                                                          \
          idx2 = idx + opt->offset[r];                                                                       \
          for (i=0; i<opt->offset[r+1]-opt->offset[r]; i++)                                                  \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++) u[idx2[i]*MBS+j*BS+k] = p2[i*MBS+j*BS+k];                                 \
        } else if (opt->type[r] == PETSCSF_PACKOPT_MULTICOPY) {                                              \
          for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) { /* i-th piece */                       \
            u2 = u + idx[opt->copy_start[i]]*MBS;                                                            \
            l  = opt->copy_length[i]*MBS;                                                                    \
            ierr = PetscArraycpy(u2,p2,l);CHKERRQ(ierr);                                                     \
            p2 += l;                                                                                         \
          }                                                                                                  \
        } else if (opt->type[r] == PETSCSF_PACKOPT_STRIDE) {                                                 \
          u2   = u + idx[opt->offset[r]]*MBS;                                                                \
          step = opt->stride_step[r];                                                                        \
          for (i=0; i<opt->stride_n[r]; i++)                                                                 \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++) u2[i*step*MBS+j*BS+k] = p2[i*MBS+j*BS+k];                                 \
        } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unknown SFPack optimzation type %D",opt->type[r]);   \
      }                                                                                                      \
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
  static PetscErrorCode CPPJoin4(UnpackAnd##Opname,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,const PetscInt *idx,PetscSFPackOpt opt,void *unpacked,const void *packed) \
  {                                                                                                          \
    Type           *u = (Type*)unpacked,*u2;                                                                 \
    const Type     *p = (const Type*)packed,*p2;                                                             \
    PetscInt       i,j,k,l,r,step,bs=link->bs;                                                               \
    const PetscInt *idx2,M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */    \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      u += start*bs;                                                                                         \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++)                                                                               \
            OpApply(Op,u[i*MBS+j*BS+k],p[i*MBS+j*BS+k]);                                                     \
    } else if (!opt) { /* No optimizations available */                                                      \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++)                                                                               \
              OpApply(Op,u[idx[i]*MBS+j*BS+k],p[i*MBS+j*BS+k]);                                              \
    } else {                                                                                                 \
      for (r=0; r<opt->n; r++) {                                                                             \
        p2 = p + opt->offset[r]*MBS;                                                                         \
        if (opt->type[r] == PETSCSF_PACKOPT_NONE) {                                                          \
          idx2 = idx + opt->offset[r];                                                                       \
          for (i=0; i<opt->offset[r+1]-opt->offset[r]; i++)                                                  \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++)                                                                           \
                OpApply(Op,u[idx2[i]*MBS+j*BS+k],p2[i*MBS+j*BS+k]);                                          \
        } else if (opt->type[r] == PETSCSF_PACKOPT_MULTICOPY) {                                              \
          for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) { /* i-th piece */                       \
            u2 = u + idx[opt->copy_start[i]]*MBS;                                                            \
            l  = opt->copy_length[i]*MBS;                                                                    \
            for (j=0; j<l; j++) OpApply(Op,u2[j],p2[j]);                                                     \
            p2 += l;                                                                                         \
          }                                                                                                  \
        } else if (opt->type[r] == PETSCSF_PACKOPT_STRIDE) {                                                 \
          u2   = u + idx[opt->offset[r]]*MBS;                                                                \
          step = opt->stride_step[r];                                                                        \
          for (i=0; i<opt->stride_n[r]; i++)                                                                 \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++)                                                                           \
                OpApply(Op,u2[i*step*MBS+j*BS+k],p2[i*MBS+j*BS+k]);                                          \
        } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unknown SFPack optimzation type %D",opt->type[r]);   \
      }                                                                                                      \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

#define DEF_FetchAndOp(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(FetchAnd##Opname,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt start,const PetscInt *idx,PetscSFPackOpt opt,void *unpacked,void *packed) \
  {                                                                                                          \
    Type           *u = (Type*)unpacked,*p = (Type*)packed,t;                                                \
    PetscInt       i,j,k,bs=link->bs;                                                                        \
    const PetscInt M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */          \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      u += start*bs;                                                                                         \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
            t = u[i*MBS+j*BS+k];                                                                             \
            OpApply(Op,u[i*MBS+j*BS+k],p[i*MBS+j*BS+k]);                                                     \
            p[i*MBS+j*BS+k] = t;                                                                             \
          }                                                                                                  \
    } else {                                                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
              t = u[idx[i]*MBS+j*BS+k];                                                                      \
              OpApply(Op,u[idx[i]*MBS+j*BS+k],p[i*MBS+j*BS+k]);                                              \
              p[i*MBS+j*BS+k] = t;                                                                           \
          }                                                                                                  \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

#define DEF_ScatterAndOp(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(ScatterAnd##Opname,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt startx,const PetscInt *idx,const void *xdata,PetscInt starty,const PetscInt *idy,void *ydata) \
  {                                                                                                          \
    const Type     *x = (const Type*)xdata;                                                                  \
    Type           *y = (Type*)ydata;                                                                        \
    PetscInt       i,j,k,bs = link->bs;                                                                      \
    const PetscInt M = (EQ) ? 1 : bs/BS;                                                                     \
    const PetscInt MBS = M*BS;                                                                               \
    PetscFunctionBegin;                                                                                      \
    if (!idx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Petsc should use UnpackAndOp instead of ScatterAndOp");\
    if (!idy) {                                                                                              \
      y += starty*bs;                                                                                        \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++)                                                                               \
            OpApply(Op,y[i*MBS+j*BS+k],x[idx[i]*MBS+j*BS+k]);                                                \
    } else {                                                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++)                                                                               \
            OpApply(Op,y[idy[i]*MBS+j*BS+k],x[idx[i]*MBS+j*BS+k]);                                           \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

#define DEF_FetchAndOpLocal(Type,BS,EQ,Opname,Op,OpApply) \
  static PetscErrorCode CPPJoin4(FetchAnd##Opname##Local,Type,BS,EQ)(PetscSFLink link,PetscInt count,PetscInt rootstart,const PetscInt *rootindices,void *rootdata,PetscInt leafstart,const PetscInt *leafindices,const void *leafdata,void *leafupdate) \
  {                                                                                                          \
    Type           *x = (Type*)rootdata,*y2 = (Type*)leafupdate;                                             \
    const Type     *y = (const Type*)leafdata;                                                               \
    PetscInt       i,j,k,bs = link->bs;                                                                      \
    const PetscInt M = (EQ) ? 1 : bs/BS;                                                                     \
    const PetscInt MBS = M*BS;                                                                               \
    PetscFunctionBegin;                                                                                      \
    if (!rootindices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Petsc should use FetchAndOp instead of FetchAndOpLocal");\
    if (!leafindices) {                                                                                      \
      y += leafstart*bs; y2 += leafstart*bs;                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
            y2[i*MBS+j*BS+k] = x[rootindices[i]*MBS+j*BS+k];                                                 \
            OpApply(Op,x[rootindices[i]*MBS+j*BS+k],y[i*MBS+j*BS+k]);                                        \
          }                                                                                                  \
    } else {                                                                                                 \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
            y2[leafindices[i]*MBS+j*BS+k] = x[rootindices[i]*MBS+j*BS+k];                                    \
            OpApply(Op,x[rootindices[i]*MBS+j*BS+k],y[leafindices[i]*MBS+j*BS+k]);                           \
          }                                                                                                  \
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
   The routine Creates a communication link for the given operation. It first looks up its link cache. If
   there is a free & suitable one, it uses it. Otherwise it creates a new one.

   A link contains buffers and MPI requests for send/recv. It also contains pack/unpack routines to pack/unpack
   root/leafdata to/from these buffers. Buffers are allocated at our discretion. When we find root/leafata
   can be directly passed to MPI, we won't allocate them. Even we allocate buffers, we only allocate
   those that are needed by the given `sfop` and `op`, in other words, we do lazy memory-allocation.

   The routine also allocates buffers on CPU when one does not use gpu-aware MPI but data is on GPU.

   In SFBasic, MPI requests are persistent. They are init'ed until we try to get requests from a link.

   The routine is shared by SFBasic and SFNeighbor based on the fact they all deal with sparse graphs and
   need pack/unpack data.
*/
PetscErrorCode PetscSFLinkCreate(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          i,j,k,nrootreqs,nleafreqs,nreqs;
  PetscSFLink       *p,link;
  PetscSFDirection  direction;
  MPI_Request       *reqs = NULL;
  PetscBool         match,rootdirect[2],leafdirect[2];
  PetscMemType      rootmtype_mpi,leafmtype_mpi;   /* mtypes seen by MPI */
  PetscInt          rootdirect_mpi,leafdirect_mpi; /* root/leafdirect seen by MPI*/

  PetscFunctionBegin;
  ierr = PetscSFSetErrorOnUnsupportedOverlap(sf,unit,rootdata,leafdata);CHKERRQ(ierr);

  /* Can we directly use root/leafdirect with the given sf, sfop and op? */
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    if (sfop == PETSCSF_BCAST) {
      rootdirect[i] = bas->rootcontig[i]; /* Pack roots */
      leafdirect[i] = (sf->leafcontig[i] && op == MPIU_REPLACE) ? PETSC_TRUE : PETSC_FALSE;  /* Unpack leaves */
    } else if (sfop == PETSCSF_REDUCE) {
      leafdirect[i] = sf->leafcontig[i];  /* Pack leaves */
      rootdirect[i] = (bas->rootcontig[i] && op == MPIU_REPLACE) ? PETSC_TRUE : PETSC_FALSE; /* Unpack roots */
    } else { /* PETSCSF_FETCH */
      rootdirect[i] = PETSC_FALSE; /* FETCH always need a separate rootbuf */
      leafdirect[i] = PETSC_FALSE; /* We also force allocating a separate leafbuf so that leafdata and leafupdate can share mpi requests */
    }
  }

  if (use_gpu_aware_mpi) {
    rootmtype_mpi = rootmtype;
    leafmtype_mpi = leafmtype;
  } else {
    rootmtype_mpi = leafmtype_mpi = PETSC_MEMTYPE_HOST;
  }
  /* Will root/leafdata be directly accessed by MPI?  Without use_gpu_aware_mpi, device data is bufferred on host and then passed to MPI */
  rootdirect_mpi = rootdirect[PETSCSF_REMOTE] && (rootmtype_mpi == rootmtype)? 1 : 0;
  leafdirect_mpi = leafdirect[PETSCSF_REMOTE] && (leafmtype_mpi == leafmtype)? 1 : 0;

  direction = (sfop == PETSCSF_BCAST)? PETSCSF_ROOT2LEAF : PETSCSF_LEAF2ROOT;
  nrootreqs = bas->nrootreqs;
  nleafreqs = sf->nleafreqs;

  /* Look for free links in cache */
  for (p=&bas->avail; (link=*p); p=&link->next) {
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      /* If root/leafdata will be directly passed to MPI, test if the data used to initialized the MPI requests matches with current.
         If not, free old requests. New requests will be lazily init'ed until one calls PetscSFLinkGetMPIBuffersAndRequests().
      */
      if (rootdirect_mpi && sf->persistent && link->rootreqsinited[direction][rootmtype][1] && link->rootdatadirect[direction][rootmtype] != rootdata) {
        reqs = link->rootreqs[direction][rootmtype][1]; /* Here, rootmtype = rootmtype_mpi */
        for (i=0; i<nrootreqs; i++) {if (reqs[i] != MPI_REQUEST_NULL) {ierr = MPI_Request_free(&reqs[i]);CHKERRQ(ierr);}}
        link->rootreqsinited[direction][rootmtype][1] = PETSC_FALSE;
      }
      if (leafdirect_mpi && sf->persistent && link->leafreqsinited[direction][leafmtype][1] && link->leafdatadirect[direction][leafmtype] != leafdata) {
        reqs = link->leafreqs[direction][leafmtype][1];
        for (i=0; i<nleafreqs; i++) {if (reqs[i] != MPI_REQUEST_NULL) {ierr = MPI_Request_free(&reqs[i]);CHKERRQ(ierr);}}
        link->leafreqsinited[direction][leafmtype][1] = PETSC_FALSE;
      }
      *p = link->next; /* Remove from available list */
      goto found;
    }
  }

  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = PetscSFLinkSetUp_Host(sf,link,unit);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(PetscObjectComm((PetscObject)sf),&link->tag);CHKERRQ(ierr); /* One tag per link */

  nreqs = (nrootreqs+nleafreqs)*8;
  ierr  = PetscMalloc1(nreqs,&link->reqs);CHKERRQ(ierr);
  for (i=0; i<nreqs; i++) link->reqs[i] = MPI_REQUEST_NULL; /* Initialized to NULL so that we know which need to be freed in Destroy */

  for (i=0; i<2; i++) { /* Two communication directions */
    for (j=0; j<2; j++) { /* Two memory types */
      for (k=0; k<2; k++) { /* root/leafdirect 0 or 1 */
        link->rootreqs[i][j][k] = link->reqs + nrootreqs*(4*i+2*j+k);
        link->leafreqs[i][j][k] = link->reqs + nrootreqs*8 + nleafreqs*(4*i+2*j+k);
      }
    }
  }

found:
  if ((rootmtype == PETSC_MEMTYPE_DEVICE || leafmtype == PETSC_MEMTYPE_DEVICE) && !link->deviceinited) {ierr = PetscSFLinkSetUp_Device(sf,link,unit);CHKERRQ(ierr);}

  /* Allocate buffers along root/leafdata */
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    /* For local communication, buffers are only needed when roots and leaves have different mtypes */
    if (i == PETSCSF_LOCAL && rootmtype == leafmtype) continue;
    if (bas->rootbuflen[i]) {
      if (rootdirect[i]) { /* Aha, we disguise rootdata as rootbuf */
        link->rootbuf[i][rootmtype] = (char*)rootdata + bas->rootstart[i]*link->unitbytes;
      } else { /* Have to have a separate rootbuf */
        if (!link->rootbuf_alloc[i][rootmtype]) {
          ierr = PetscMallocWithMemType(rootmtype,bas->rootbuflen[i]*link->unitbytes,(void**)&link->rootbuf_alloc[i][rootmtype]);CHKERRQ(ierr);
        }
        link->rootbuf[i][rootmtype] = link->rootbuf_alloc[i][rootmtype];
      }
    }

    if (sf->leafbuflen[i]) {
      if (leafdirect[i]) {
        link->leafbuf[i][leafmtype] = (char*)leafdata + sf->leafstart[i]*link->unitbytes;
      } else {
        if (!link->leafbuf_alloc[i][leafmtype]) {
          ierr = PetscMallocWithMemType(leafmtype,sf->leafbuflen[i]*link->unitbytes,(void**)&link->leafbuf_alloc[i][leafmtype]);CHKERRQ(ierr);
        }
        link->leafbuf[i][leafmtype] = link->leafbuf_alloc[i][leafmtype];
      }
    }
  }

  /* Allocate buffers on host for buffering data on device in cast not use_gpu_aware_mpi */
  if (rootmtype == PETSC_MEMTYPE_DEVICE && rootmtype_mpi == PETSC_MEMTYPE_HOST) {
    if(!link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]) {
      ierr = PetscMalloc(bas->rootbuflen[PETSCSF_REMOTE]*link->unitbytes,&link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
    }
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST] = link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
  }
  if (leafmtype == PETSC_MEMTYPE_DEVICE && leafmtype_mpi == PETSC_MEMTYPE_HOST) {
    if (!link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]) {
      ierr = PetscMalloc(sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes,&link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
    }
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST] = link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
  }

  /* Set `current` state of the link. They may change between different SF invocations with the same link */
  if (sf->persistent) { /* If data is directly passed to MPI and inits MPI requests, record the data for comparison on future invocations */
    if (rootdirect_mpi) link->rootdatadirect[direction][rootmtype] = rootdata;
    if (leafdirect_mpi) link->leafdatadirect[direction][leafmtype] = leafdata;
  }

  link->rootdata = rootdata; /* root/leafdata are keys to look up links in PetscSFXxxEnd */
  link->leafdata = leafdata;
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    link->rootdirect[i] = rootdirect[i];
    link->leafdirect[i] = leafdirect[i];
  }
  link->rootdirect_mpi  = rootdirect_mpi;
  link->leafdirect_mpi  = leafdirect_mpi;
  link->rootmtype       = rootmtype;
  link->leafmtype       = leafmtype;
  link->rootmtype_mpi   = rootmtype_mpi;
  link->leafmtype_mpi   = leafmtype_mpi;

  link->next            = bas->inuse;
  bas->inuse            = link;
  *mylink               = link;
  PetscFunctionReturn(0);
}

/* Return root/leaf buffers and MPI requests attached to the link for MPI communication in the given direction.
   If the sf uses persistent requests and the requests have not been initialized, then initialize them.
*/
PetscErrorCode PetscSFLinkGetMPIBuffersAndRequests(PetscSF sf,PetscSFLink link,PetscSFDirection direction,void **rootbuf, void **leafbuf,MPI_Request **rootreqs,MPI_Request **leafreqs)
{
  PetscErrorCode       ierr;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscInt             i,j,nrootranks,ndrootranks,nleafranks,ndleafranks;
  const PetscInt       *rootoffset,*leafoffset;
  PetscMPIInt          n;
  MPI_Aint             disp;
  MPI_Comm             comm = PetscObjectComm((PetscObject)sf);
  MPI_Datatype         unit = link->unit;
  const PetscMemType   rootmtype_mpi = link->rootmtype_mpi,leafmtype_mpi = link->leafmtype_mpi; /* Used to select buffers passed to MPI */
  const PetscInt       rootdirect_mpi = link->rootdirect_mpi,leafdirect_mpi = link->leafdirect_mpi;

  PetscFunctionBegin;
  /* Init persistent MPI requests if not yet. Currently only SFBasic uses persistent MPI */
  if (sf->persistent) {
    if (rootreqs && bas->rootbuflen[PETSCSF_REMOTE] && !link->rootreqsinited[direction][rootmtype_mpi][rootdirect_mpi]) {
      ierr = PetscSFGetRootInfo_Basic(sf,&nrootranks,&ndrootranks,NULL,&rootoffset,NULL);CHKERRQ(ierr);
      if (direction == PETSCSF_LEAF2ROOT) {
        for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
          disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
          ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],&n);CHKERRQ(ierr);
          ierr = MPI_Recv_init(link->rootbuf[PETSCSF_REMOTE][rootmtype_mpi]+disp,n,unit,bas->iranks[i],link->tag,comm,link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi]+j);CHKERRQ(ierr);
        }
      } else { /* PETSCSF_ROOT2LEAF */
        for (i=ndrootranks,j=0; i<nrootranks; i++,j++) {
          disp = (rootoffset[i] - rootoffset[ndrootranks])*link->unitbytes;
          ierr = PetscMPIIntCast(rootoffset[i+1]-rootoffset[i],&n);CHKERRQ(ierr);
          ierr = MPI_Send_init(link->rootbuf[PETSCSF_REMOTE][rootmtype_mpi]+disp,n,unit,bas->iranks[i],link->tag,comm,link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi]+j);CHKERRQ(ierr);
        }
      }
      link->rootreqsinited[direction][rootmtype_mpi][rootdirect_mpi] = PETSC_TRUE;
    }

    if (leafreqs && sf->leafbuflen[PETSCSF_REMOTE] && !link->leafreqsinited[direction][leafmtype_mpi][leafdirect_mpi]) {
      ierr = PetscSFGetLeafInfo_Basic(sf,&nleafranks,&ndleafranks,NULL,&leafoffset,NULL,NULL);CHKERRQ(ierr);
      if (direction == PETSCSF_LEAF2ROOT) {
        for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
          disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
          ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],&n);CHKERRQ(ierr);
          ierr = MPI_Send_init(link->leafbuf[PETSCSF_REMOTE][leafmtype_mpi]+disp,n,unit,sf->ranks[i],link->tag,comm,link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi]+j);CHKERRQ(ierr);
        }
      } else { /* PETSCSF_ROOT2LEAF */
        for (i=ndleafranks,j=0; i<nleafranks; i++,j++) {
          disp = (leafoffset[i] - leafoffset[ndleafranks])*link->unitbytes;
          ierr = PetscMPIIntCast(leafoffset[i+1]-leafoffset[i],&n);CHKERRQ(ierr);
          ierr = MPI_Recv_init(link->leafbuf[PETSCSF_REMOTE][leafmtype_mpi]+disp,n,unit,sf->ranks[i],link->tag,comm,link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi]+j);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscSFLink       link,*p;
  PetscSF_Basic     *bas=(PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (p=&bas->inuse; (link=*p); p=&link->next) {
    PetscBool match;
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkReclaim(PetscSF sf,PetscSFLink *link)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  (*link)->rootdata = NULL;
  (*link)->leafdata = NULL;
  (*link)->next     = bas->avail;
  bas->avail        = *link;
  *link             = NULL;
  PetscFunctionReturn(0);
}

/* Destroy all links chained in 'avail' */
PetscErrorCode PetscSFLinkDestroy(PetscSF sf,PetscSFLink *avail)
{
  PetscErrorCode    ierr;
  PetscSFLink       link = *avail,next;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscInt          i,nreqs = (bas->nrootreqs+sf->nleafreqs)*8;

  PetscFunctionBegin;
  for (; link; link=next) {
    next = link->next;
    if (!link->isbuiltin) {ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);}
    for (i=0; i<nreqs; i++) { /* Persistent reqs must be freed. */
      if (link->reqs[i] != MPI_REQUEST_NULL) {ierr = MPI_Request_free(&link->reqs[i]);CHKERRQ(ierr);}
    }
    ierr = PetscFree(link->reqs);CHKERRQ(ierr);
    for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
#if defined(PETSC_HAVE_CUDA)
      ierr = PetscFreeWithMemType(PETSC_MEMTYPE_DEVICE,link->rootbuf_alloc[i][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
      ierr = PetscFreeWithMemType(PETSC_MEMTYPE_DEVICE,link->leafbuf_alloc[i][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
      if (link->stream) {cudaError_t err =  cudaStreamDestroy(link->stream);CHKERRCUDA(err); link->stream = NULL;}
#endif
      ierr = PetscFree(link->rootbuf_alloc[i][PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
      ierr = PetscFree(link->leafbuf_alloc[i][PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
    }
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  *avail = NULL;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
/* Error out on unsupported overlapped communications */
PetscErrorCode PetscSFSetErrorOnUnsupportedOverlap(PetscSF sf,MPI_Datatype unit,const void *rootdata,const void *leafdata)
{
  PetscErrorCode    ierr;
  PetscSFLink       link,*p;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscBool         match;

  PetscFunctionBegin;
  /* Look up links in use and error out if there is a match. When both rootdata and leafdata are NULL, ignore
     the potential overlapping since this process does not participate in communication. Overlapping is harmless.
  */
  if (rootdata || leafdata) {
    for (p=&bas->inuse; (link=*p); p=&link->next) {
      ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
      if (match && (rootdata == link->rootdata) && (leafdata == link->leafdata)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Overlapped PetscSF with the same rootdata(%p), leafdata(%p) and data type. Undo the overlapping to avoid the error.",rootdata,leafdata);
    }
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode PetscSFLinkSetUp_Host(PetscSF sf,PetscSFLink link,MPI_Datatype unit)
{
  PetscErrorCode ierr;
  PetscInt       nSignedChar=0,nUnsignedChar=0,nInt=0,nPetscInt=0,nPetscReal=0;
  PetscBool      is2Int,is2PetscInt;
  PetscMPIInt    ni,na,nd,combiner;
#if defined(PETSC_HAVE_COMPLEX)
  PetscInt       nPetscComplex=0;
#endif

  PetscFunctionBegin;
  ierr = MPIPetsc_Type_compare_contig(unit,MPI_SIGNED_CHAR,  &nSignedChar);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPI_UNSIGNED_CHAR,&nUnsignedChar);CHKERRQ(ierr);
  /* MPI_CHAR is treated below as a dumb type that does not support reduction according to MPI standard */
  ierr = MPIPetsc_Type_compare_contig(unit,MPI_INT,  &nInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_INT, &nPetscInt);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_REAL,&nPetscReal);CHKERRQ(ierr);
#if defined(PETSC_HAVE_COMPLEX)
  ierr = MPIPetsc_Type_compare_contig(unit,MPIU_COMPLEX,&nPetscComplex);CHKERRQ(ierr);
#endif
  ierr = MPIPetsc_Type_compare(unit,MPI_2INT,&is2Int);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPIU_2INT,&is2PetscInt);CHKERRQ(ierr);
  /* TODO: shaell we also handle Fortran MPI_2REAL? */
  ierr = MPI_Type_get_envelope(unit,&ni,&na,&nd,&combiner);CHKERRQ(ierr);
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
    ierr = MPI_Type_get_extent(unit,&lb,&nbyte);CHKERRQ(ierr);
    if (lb != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
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

  if (!link->isbuiltin) {ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetUnpackAndOp(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*))
{
  PetscFunctionBegin;
  *UnpackAndOp = NULL;
  if (mtype == PETSC_MEMTYPE_HOST) {
    if      (op == MPIU_REPLACE)              *UnpackAndOp = link->h_UnpackAndInsert;
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
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE && !atomic) {
    if      (op == MPIU_REPLACE)              *UnpackAndOp = link->d_UnpackAndInsert;
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
  } else if (mtype == PETSC_MEMTYPE_DEVICE && atomic) {
    if      (op == MPIU_REPLACE)              *UnpackAndOp = link->da_UnpackAndInsert;
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

PetscErrorCode PetscSFLinkGetScatterAndOp(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*))
{
  PetscFunctionBegin;
  *ScatterAndOp = NULL;
  if (mtype == PETSC_MEMTYPE_HOST) {
    if      (op == MPIU_REPLACE)              *ScatterAndOp = link->h_ScatterAndInsert;
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
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE && !atomic) {
    if      (op == MPIU_REPLACE)              *ScatterAndOp = link->d_ScatterAndInsert;
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
  } else if (mtype == PETSC_MEMTYPE_DEVICE && atomic) {
    if      (op == MPIU_REPLACE)              *ScatterAndOp = link->da_ScatterAndInsert;
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

PetscErrorCode PetscSFLinkGetFetchAndOp(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**FetchAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOp = NULL;
  if (op != MPI_SUM && op != MPIU_SUM) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op in FetchAndOp");
  if (mtype == PETSC_MEMTYPE_HOST) *FetchAndOp = link->h_FetchAndAdd;
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE && !atomic) *FetchAndOp = link->d_FetchAndAdd;
  else if (mtype == PETSC_MEMTYPE_DEVICE && atomic)  *FetchAndOp = link->da_FetchAndAdd;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkGetFetchAndOpLocal(PetscSFLink link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**FetchAndOpLocal)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,void*,PetscInt,const PetscInt*,const void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOpLocal = NULL;
  if (op != MPI_SUM && op != MPIU_SUM) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op in FetchAndOp");
  if (mtype == PETSC_MEMTYPE_HOST) *FetchAndOpLocal = link->h_FetchAndAddLocal;
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE && !atomic) *FetchAndOpLocal = link->d_FetchAndAddLocal;
  else if (mtype == PETSC_MEMTYPE_DEVICE && atomic)  *FetchAndOpLocal = link->da_FetchAndAddLocal;
#endif
  PetscFunctionReturn(0);
}

/*=============================================================================
              A set of helper routines for Pack/Unpack/Scatter on GPUs
 ============================================================================*/
#if defined(PETSC_HAVE_CUDA)
/* If SF does not know which stream root/leafdata is being computed on, it has to sync the device to
   make sure the data is ready for packing.
 */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncDeviceBeforePackData(PetscSF sf,PetscSFLink link)
{
  PetscFunctionBegin;
  if (sf->use_default_stream) PetscFunctionReturn(0);
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE || link->leafmtype == PETSC_MEMTYPE_DEVICE) {
    cudaError_t cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

/* PetscSFLinkSyncStreamAfterPackXxxData routines make sure root/leafbuf for the remote is ready for MPI */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterPackRootData(PetscSF sf,PetscSFLink link)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* Do nothing if we use stream aware mpi || has nothing for remote */
  if (sf->use_stream_aware_mpi || link->rootmtype != PETSC_MEMTYPE_DEVICE || !bas->rootbuflen[PETSCSF_REMOTE]) PetscFunctionReturn(0);
  /* If we called a packing kernel || we async-copied rootdata from device to host || No cudaDeviceSynchronize was called (since default stream is assumed) */
  if (!link->rootdirect[PETSCSF_REMOTE] || !use_gpu_aware_mpi || sf->use_default_stream) {
    cudaError_t cerr = cudaStreamSynchronize(link->stream);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterPackLeafData(PetscSF sf,PetscSFLink link)
{
  PetscFunctionBegin;
  /* See comments above */
  if (sf->use_stream_aware_mpi || link->leafmtype != PETSC_MEMTYPE_DEVICE || !sf->leafbuflen[PETSCSF_REMOTE]) PetscFunctionReturn(0);
  if (!link->leafdirect[PETSCSF_REMOTE] || !use_gpu_aware_mpi || sf->use_default_stream) {
    cudaError_t cerr = cudaStreamSynchronize(link->stream);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

/* PetscSFLinkSyncStreamAfterUnpackXxx routines make sure root/leafdata (local & remote) is ready to use for SF callers, when SF
   does not know which stream the callers will use.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterUnpackRootData(PetscSF sf,PetscSFLink link)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* Do nothing if we are expected to put rootdata on default stream */
  if (sf->use_default_stream || link->rootmtype != PETSC_MEMTYPE_DEVICE) PetscFunctionReturn(0);
  /* If we have something from local, then we called a scatter kernel (on link->stream), then we must sync it;
     If we have something from remote and we called unpack kernel, then we must also sycn it.
   */
  if (bas->rootbuflen[PETSCSF_LOCAL] || (bas->rootbuflen[PETSCSF_REMOTE] && !link->rootdirect[PETSCSF_REMOTE])) {
    cudaError_t cerr = cudaStreamSynchronize(link->stream);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterUnpackLeafData(PetscSF sf,PetscSFLink link)
{
  PetscFunctionBegin;
  /* See comments above */
  if (sf->use_default_stream || link->leafmtype != PETSC_MEMTYPE_DEVICE) PetscFunctionReturn(0);
  if (sf->leafbuflen[PETSCSF_LOCAL] || (sf->leafbuflen[PETSCSF_REMOTE] && !link->leafdirect[PETSCSF_REMOTE])) {
    cudaError_t cerr = cudaStreamSynchronize(link->stream);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

/* PetscSFLinkCopyXxxxBufferInCaseNotUseGpuAwareMPI routines are simple: if not use_gpu_aware_mpi, we need
   to copy the buffer from GPU to CPU before MPI calls, and from CPU to GPU after MPI calls.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(PetscSF sf,PetscSFLink link,PetscBool device2host)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && (link->rootmtype_mpi != link->rootmtype) && bas->rootbuflen[PETSCSF_REMOTE]) {
    void  *h_buf = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
    void  *d_buf = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
    size_t count = bas->rootbuflen[PETSCSF_REMOTE]*link->unitbytes;
    if (device2host) {
      cerr = cudaMemcpyAsync(h_buf,d_buf,count,cudaMemcpyDeviceToHost,link->stream);CHKERRCUDA(cerr);
      ierr = PetscLogGpuToCpu(count);CHKERRQ(ierr);
    } else {
      cerr = cudaMemcpyAsync(d_buf,h_buf,count,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(cerr);
      ierr = PetscLogCpuToGpu(count);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(PetscSF sf,PetscSFLink link,PetscBool device2host)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;

  PetscFunctionBegin;
  if (link->leafmtype == PETSC_MEMTYPE_DEVICE && (link->leafmtype_mpi != link->leafmtype) && sf->leafbuflen[PETSCSF_REMOTE]) {
    void  *h_buf = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
    void  *d_buf = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
    size_t count = sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes;
    if (device2host) {
      cerr = cudaMemcpyAsync(h_buf,d_buf,count,cudaMemcpyDeviceToHost,link->stream);CHKERRCUDA(cerr);
      ierr = PetscLogGpuToCpu(count);CHKERRQ(ierr);
    } else {
      cerr = cudaMemcpyAsync(d_buf,h_buf,count,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(cerr);
      ierr = PetscLogCpuToGpu(count);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
#else

#define PetscSFLinkSyncDeviceBeforePackData(a,b)                0
#define PetscSFLinkSyncStreamAfterPackRootData(a,b)             0
#define PetscSFLinkSyncStreamAfterPackLeafData(a,b)             0
#define PetscSFLinkSyncStreamAfterUnpackRootData(a,b)           0
#define PetscSFLinkSyncStreamAfterUnpackLeafData(a,b)           0
#define PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(a,b,c) 0
#define PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(a,b,c) 0

#endif

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkLogFlopsAfterUnpackRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscLogDouble flops;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;


  PetscFunctionBegin;
  if (op != MPIU_REPLACE && link->basicunit == MPIU_SCALAR) { /* op is a reduction on PetscScalars */
    flops = bas->rootbuflen[scope]*link->bs; /* # of roots in buffer x # of scalars in unit */
#if defined(PETSC_HAVE_CUDA)
    if (link->rootmtype == PETSC_MEMTYPE_DEVICE) {ierr = PetscLogGpuFlops(flops);CHKERRQ(ierr);} else
#endif
    {ierr = PetscLogFlops(flops);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkLogFlopsAfterUnpackLeafData(PetscSF sf,PetscSFLink link,PetscSFScope scope,MPI_Op op)
{
  PetscLogDouble flops;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (op != MPIU_REPLACE && link->basicunit == MPIU_SCALAR) { /* op is a reduction on PetscScalars */
    flops = sf->leafbuflen[scope]*link->bs; /* # of roots in buffer x # of scalars in unit */
#if defined(PETSC_HAVE_CUDA)
    if (link->leafmtype == PETSC_MEMTYPE_DEVICE) {ierr = PetscLogGpuFlops(flops);CHKERRQ(ierr);} else
#endif
    {ierr = PetscLogFlops(flops);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/* When SF could not find a proper UnpackAndOp() from link, it falls back to MPI_Reduce_local.
  Input Arguments:
  +sf      - The StarForest
  .link    - The link
  .count   - Number of entries to unpack
  .start   - The first index, significent when indices=NULL
  .indices - Indices of entries in <data>. If NULL, it means indices are contiguous and the first is given in <start>
  .buf     - A contiguous buffer to unpack from
  -op      - Operation after unpack

  Output Arguments:
  .data    - The data to unpack to
*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkUnpackDataWithMPIReduceLocal(PetscSF sf,PetscSFLink link,PetscInt count,PetscInt start,const PetscInt *indices,void *data,const void *buf,MPI_Op op)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  {
    PetscErrorCode ierr;
    PetscInt       i;
    PetscMPIInt    n;
    if (indices) {
      /* Note we use link->unit instead of link->basicunit. When op can be mapped to MPI_SUM etc, it operates on
         basic units of a root/leaf element-wisely. Otherwise, it is meant to operate on a whole root/leaf.
      */
      for (i=0; i<count; i++) {ierr = MPI_Reduce_local((const char*)buf+i*link->unitbytes,(char*)data+indices[i]*link->unitbytes,1,link->unit,op);CHKERRQ(ierr);}
    } else {
      ierr = PetscMPIIntCast(count,&n);CHKERRQ(ierr);
      ierr = MPI_Reduce_local(buf,(char*)data+start*link->unitbytes,n,link->unit,op);CHKERRQ(ierr);
    }
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkScatterDataWithMPIReduceLocal(PetscSF sf,PetscSFLink link,PetscInt count,const PetscInt *idx,const void *xdata,PetscInt starty,const PetscInt *idy,void *ydata,MPI_Op op)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
  {
    PetscErrorCode ierr;
    PetscInt       i,disp;
    for (i=0; i<count; i++) {
      disp = idy? idy[i] : starty + i;
      ierr = MPI_Reduce_local((const char*)xdata+idx[i]*link->unitbytes,(char*)ydata+disp*link->unitbytes,1,link->unit,op);CHKERRQ(ierr);
    }
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
#endif
  PetscFunctionReturn(0);
}

/*=============================================================================
              Pack/Unpack/Fetch/Scatter routines
 ============================================================================*/

/* Pack rootdata to rootbuf
  Input Arguments:
  + sf       - The SF this packing works on.
  . link     - It gives the memtype of the roots and also provides root buffer.
  . scope    - PETSCSF_LOCAL or PETSCSF_REMOTE. Note SF has the ability to do local and remote communications separately.
  - rootdata - Where to read the roots.

  Notes:
  When rootdata can be directly used as root buffer, the routine is almost a no-op. After the call, root data is
  in a place where the underlying MPI is ready can access (use_gpu_aware_mpi or not)
 */
PetscErrorCode PetscSFLinkPackRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,const void *rootdata)
{
  PetscErrorCode   ierr;
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;
  const PetscInt   *rootindices = NULL;
  PetscInt         count,start;
  PetscErrorCode   (*Pack)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,const void*,void*) = NULL;
  PetscMemType     rootmtype = link->rootmtype;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkSyncDeviceBeforePackData(sf,link);CHKERRQ(ierr);}
  if (!link->rootdirect[scope] && bas->rootbuflen[scope]) { /* If rootdata works directly as rootbuf, skip packing */
    ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,scope,&count,&start,&rootindices);CHKERRQ(ierr);
    ierr = PetscSFLinkGetPack(link,rootmtype,&Pack);CHKERRQ(ierr);
    ierr = (*Pack)(link,count,start,rootindices,bas->rootpackopt[scope],rootdata,link->rootbuf[scope][rootmtype]);CHKERRQ(ierr);
  }
  if (scope == PETSCSF_REMOTE) {
    ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/*device2host*/);CHKERRQ(ierr);
    ierr = PetscSFLinkSyncStreamAfterPackRootData(sf,link);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Pack leafdata to leafbuf */
PetscErrorCode PetscSFLinkPackLeafData(PetscSF sf,PetscSFLink link,PetscSFScope scope,const void *leafdata)
{
  PetscErrorCode   ierr;
  const PetscInt   *leafindices = NULL;
  PetscInt         count,start;
  PetscErrorCode   (*Pack)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,const void*,void*) = NULL;
  PetscMemType     leafmtype = link->leafmtype;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkSyncDeviceBeforePackData(sf,link);CHKERRQ(ierr);}
  if (!link->leafdirect[scope] && sf->leafbuflen[scope]) { /* If leafdata works directly as rootbuf, skip packing */
    ierr = PetscSFLinkGetLeafIndices(sf,link,leafmtype,scope,&count,&start,&leafindices);CHKERRQ(ierr);
    ierr = PetscSFLinkGetPack(link,leafmtype,&Pack);CHKERRQ(ierr);
    ierr = (*Pack)(link,count,start,leafindices,sf->leafpackopt[scope],leafdata,link->leafbuf[scope][leafmtype]);CHKERRQ(ierr);
  }
  if (scope == PETSCSF_REMOTE) {
    ierr = PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE/*device2host*/);CHKERRQ(ierr);
    ierr = PetscSFLinkSyncStreamAfterPackLeafData(sf,link);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Unpack rootbuf to rootdata */
PetscErrorCode PetscSFLinkUnpackRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *rootdata,MPI_Op op)
{
  PetscErrorCode   ierr;
  const PetscInt   *rootindices = NULL;
  PetscInt         count,start;
  PetscSF_Basic    *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode   (*UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*) = NULL;
  PetscMemType     rootmtype = link->rootmtype;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_FALSE);CHKERRQ(ierr);}
  if (!link->rootdirect[scope] && bas->rootbuflen[scope]) { /* If rootdata works directly as rootbuf, skip unpacking */
    ierr = PetscSFLinkGetUnpackAndOp(link,rootmtype,op,bas->rootdups[scope],&UnpackAndOp);CHKERRQ(ierr);
    if (UnpackAndOp) {
      ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,scope,&count,&start,&rootindices);CHKERRQ(ierr);
      ierr = (*UnpackAndOp)(link,count,start,rootindices,bas->rootpackopt[scope],rootdata,link->rootbuf[scope][rootmtype]);CHKERRQ(ierr);
    } else {
      ierr = PetscSFLinkGetRootIndices(sf,link,PETSC_MEMTYPE_HOST,scope,&count,&start,&rootindices);CHKERRQ(ierr);
      ierr = PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,start,rootindices,rootdata,link->rootbuf[scope][rootmtype],op);CHKERRQ(ierr);
    }
  }
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkSyncStreamAfterUnpackRootData(sf,link);CHKERRQ(ierr);}
  ierr = PetscSFLinkLogFlopsAfterUnpackRootData(sf,link,scope,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Unpack leafbuf to leafdata */
PetscErrorCode PetscSFLinkUnpackLeafData(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *leafdata,MPI_Op op)
{
  PetscErrorCode   ierr;
  const PetscInt   *leafindices = NULL;
  PetscInt         count,start;
  PetscErrorCode   (*UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*) = NULL;
  PetscMemType     leafmtype = link->leafmtype;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_FALSE);CHKERRQ(ierr);}
  if (!link->leafdirect[scope] && sf->leafbuflen[scope]) { /* If leafdata works directly as rootbuf, skip unpacking */
    ierr = PetscSFLinkGetUnpackAndOp(link,leafmtype,op,sf->leafdups[scope],&UnpackAndOp);CHKERRQ(ierr);
    if (UnpackAndOp) {
      ierr = PetscSFLinkGetLeafIndices(sf,link,leafmtype,scope,&count,&start,&leafindices);CHKERRQ(ierr);
      ierr = (*UnpackAndOp)(link,count,start,leafindices,sf->leafpackopt[scope],leafdata,link->leafbuf[scope][leafmtype]);CHKERRQ(ierr);
    } else {
      ierr = PetscSFLinkGetLeafIndices(sf,link,PETSC_MEMTYPE_HOST,scope,&count,&start,&leafindices);CHKERRQ(ierr);
      ierr = PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,start,leafindices,leafdata,link->leafbuf[scope][leafmtype],op);CHKERRQ(ierr);
    }
  }
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkSyncStreamAfterUnpackLeafData(sf,link);CHKERRQ(ierr);}
  ierr = PetscSFLinkLogFlopsAfterUnpackLeafData(sf,link,scope,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* FetchAndOp rootdata with rootbuf */
PetscErrorCode PetscSFLinkFetchRootData(PetscSF sf,PetscSFLink link,PetscSFScope scope,void *rootdata,MPI_Op op)
{
  PetscErrorCode     ierr;
  const PetscInt     *rootindices = NULL;
  PetscInt           count,start;
  PetscSF_Basic      *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode     (*FetchAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,void*) = NULL;
  PetscMemType       rootmtype = link->rootmtype;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  if (scope == PETSCSF_REMOTE) {ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_FALSE);CHKERRQ(ierr);}
  if (bas->rootbuflen[scope]) {
    /* Do FetchAndOp on rootdata with rootbuf */
    ierr = PetscSFLinkGetFetchAndOp(link,rootmtype,op,bas->rootdups[scope],&FetchAndOp);CHKERRQ(ierr);
    ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,scope,&count,&start,&rootindices);CHKERRQ(ierr);
    ierr = (*FetchAndOp)(link,count,start,rootindices,bas->rootpackopt[scope],rootdata,link->rootbuf[scope][rootmtype]);CHKERRQ(ierr);
  }
  if (scope == PETSCSF_REMOTE) {
    ierr = PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(sf,link,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscSFLinkSyncStreamAfterUnpackRootData(sf,link);CHKERRQ(ierr);
  }
  ierr = PetscSFLinkLogFlopsAfterUnpackRootData(sf,link,scope,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Bcast rootdata to leafdata locally (i.e., only for local communication - PETSCSF_LOCAL) */
PetscErrorCode PetscSFLinkBcastAndOpLocal(PetscSF sf,PetscSFLink link,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  const PetscInt       *rootindices = NULL,*leafindices = NULL;
  PetscInt             count,rootstart,leafstart;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode       (*UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*) = NULL;
  PetscErrorCode       (*ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*) = NULL;
  const PetscMemType   rootmtype = link->rootmtype,leafmtype = link->leafmtype;

  PetscFunctionBegin;
  if (!bas->rootbuflen[PETSCSF_LOCAL]) PetscFunctionReturn(0);
  if (rootmtype != leafmtype) { /* Uncommon case */
     /* The local communication has to go through pack and unpack */
    ierr = PetscSFLinkPackRootData(sf,link,PETSCSF_LOCAL,rootdata);CHKERRQ(ierr);
    ierr = PetscMemcpyWithMemType(leafmtype,rootmtype,link->leafbuf[PETSCSF_LOCAL][leafmtype],link->rootbuf[PETSCSF_LOCAL][rootmtype],sf->leafbuflen[PETSCSF_LOCAL]*link->unitbytes);CHKERRQ(ierr);
    ierr = PetscSFLinkUnpackLeafData(sf,link,PETSCSF_LOCAL,leafdata,op);CHKERRQ(ierr);
  } else {
    if (bas->rootcontig[PETSCSF_LOCAL]) { /* If root indices are contiguous, Scatter becomes Unpack */
      ierr = PetscSFLinkGetUnpackAndOp(link,leafmtype,op,sf->leafdups[PETSCSF_LOCAL],&UnpackAndOp);CHKERRQ(ierr);
      rootdata = (const char*)rootdata + bas->rootstart[PETSCSF_LOCAL]*link->unitbytes; /* Make rootdata point to start of the buffer */
      if (UnpackAndOp) {
        ierr = PetscSFLinkGetLeafIndices(sf,link,leafmtype,PETSCSF_LOCAL,&count,&leafstart,&leafindices);CHKERRQ(ierr);
        ierr = (*UnpackAndOp)(link,count,leafstart,leafindices,sf->leafpackopt[PETSCSF_LOCAL],leafdata,rootdata);CHKERRQ(ierr);
      } else {
        ierr = PetscSFLinkGetLeafIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,&count,&leafstart,&leafindices);CHKERRQ(ierr);
        ierr = PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,leafstart,leafindices,leafdata,rootdata,op);CHKERRQ(ierr);
      }
    } else { /* ScatterAndOp */
      ierr = PetscSFLinkGetScatterAndOp(link,leafmtype,op,sf->leafdups[PETSCSF_LOCAL],&ScatterAndOp);CHKERRQ(ierr);
      if (ScatterAndOp) {
        ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,PETSCSF_LOCAL,&count,&rootstart,&rootindices);CHKERRQ(ierr);
        ierr = PetscSFLinkGetLeafIndices(sf,link,leafmtype,PETSCSF_LOCAL,&count,&leafstart,&leafindices);CHKERRQ(ierr);
        ierr = (*ScatterAndOp)(link,count,rootstart,rootindices,rootdata,leafstart,leafindices,leafdata);CHKERRQ(ierr);
      } else {
        ierr = PetscSFLinkGetRootIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,&count,NULL,&rootindices);CHKERRQ(ierr);
        ierr = PetscSFLinkGetLeafIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,NULL,&leafstart,&leafindices);CHKERRQ(ierr);
        ierr = PetscSFLinkScatterDataWithMPIReduceLocal(sf,link,count,rootindices,rootdata,leafstart,leafindices,leafdata,op);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Reduce leafdata to rootdata locally */
PetscErrorCode PetscSFLinkReduceLocal(PetscSF sf,PetscSFLink link,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode       ierr;
  const PetscInt       *rootindices = NULL,*leafindices = NULL;
  PetscInt             count,rootstart,leafstart;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode       (*UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*) = NULL;
  PetscErrorCode       (*ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*) = NULL;
  const PetscMemType   rootmtype = link->rootmtype,leafmtype = link->leafmtype;

  PetscFunctionBegin;
  if (!sf->leafbuflen[PETSCSF_LOCAL]) PetscFunctionReturn(0);
  if (rootmtype != leafmtype) {
    /* The local communication has to go through pack and unpack */
    ierr = PetscSFLinkPackLeafData(sf,link,PETSCSF_LOCAL,leafdata);CHKERRQ(ierr);
    ierr = PetscMemcpyWithMemType(rootmtype,leafmtype,link->rootbuf[PETSCSF_LOCAL][rootmtype],link->leafbuf[PETSCSF_LOCAL][leafmtype],bas->rootbuflen[PETSCSF_LOCAL]*link->unitbytes);CHKERRQ(ierr);
    ierr = PetscSFLinkUnpackRootData(sf,link,PETSCSF_LOCAL,rootdata,op);CHKERRQ(ierr);
  } else {
    if (sf->leafcontig[PETSCSF_LOCAL]) {
      /* If leaf indices are contiguous, Scatter becomes Unpack */
      ierr = PetscSFLinkGetUnpackAndOp(link,rootmtype,op,bas->rootdups[PETSCSF_LOCAL],&UnpackAndOp);CHKERRQ(ierr);
      leafdata = (const char*)leafdata + sf->leafstart[PETSCSF_LOCAL]*link->unitbytes; /* Make leafdata point to start of the buffer */
      if (UnpackAndOp) {
        ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,PETSCSF_LOCAL,&count,&rootstart,&rootindices);CHKERRQ(ierr);
        ierr = (*UnpackAndOp)(link,count,rootstart,rootindices,bas->rootpackopt[PETSCSF_LOCAL],rootdata,leafdata);CHKERRQ(ierr);
      } else {
        ierr = PetscSFLinkGetRootIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,&count,&rootstart,&rootindices);CHKERRQ(ierr);
        ierr = PetscSFLinkUnpackDataWithMPIReduceLocal(sf,link,count,rootstart,rootindices,rootdata,leafdata,op);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscSFLinkGetScatterAndOp(link,rootmtype,op,bas->rootdups[PETSCSF_LOCAL],&ScatterAndOp);CHKERRQ(ierr);
      if (ScatterAndOp) {
        ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,PETSCSF_LOCAL,&count,&rootstart,&rootindices);CHKERRQ(ierr);
        ierr = PetscSFLinkGetLeafIndices(sf,link,leafmtype,PETSCSF_LOCAL,NULL,&leafstart,&leafindices);CHKERRQ(ierr);
        ierr = (*ScatterAndOp)(link,count,leafstart,leafindices,leafdata,rootstart,rootindices,rootdata);CHKERRQ(ierr);
      } else {
        ierr = PetscSFLinkGetRootIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,&count,&rootstart,&rootindices);CHKERRQ(ierr);
        ierr = PetscSFLinkGetLeafIndices(sf,link,PETSC_MEMTYPE_HOST,PETSCSF_LOCAL,NULL,NULL,&leafindices);CHKERRQ(ierr);
        ierr = PetscSFLinkScatterDataWithMPIReduceLocal(sf,link,count,leafindices,leafdata,rootstart,rootindices,rootdata,op);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Fetch rootdata to leafdata and leafupdate locally */
PetscErrorCode PetscSFLinkFetchAndOpLocal(PetscSF sf,PetscSFLink link,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode       ierr;
  const PetscInt       *rootindices = NULL,*leafindices = NULL;
  PetscInt             count,rootstart,leafstart;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode       (*FetchAndOpLocal)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,void*,PetscInt,const PetscInt*,const void*,void*) = NULL;
  const PetscMemType   rootmtype = link->rootmtype,leafmtype = link->leafmtype;

  PetscFunctionBegin;
  if (!bas->rootbuflen[PETSCSF_LOCAL]) PetscFunctionReturn(0);
  if (rootmtype != leafmtype) {
   /* The local communication has to go through pack and unpack */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Doing PetscSFFetchAndOp with rootdata and leafdata on opposite side of CPU and GPU");
  } else {
    ierr = PetscSFLinkGetRootIndices(sf,link,rootmtype,PETSCSF_LOCAL,&count,&rootstart,&rootindices);CHKERRQ(ierr);
    ierr = PetscSFLinkGetLeafIndices(sf,link,leafmtype,PETSCSF_LOCAL,NULL,&leafstart,&leafindices);CHKERRQ(ierr);
    ierr = PetscSFLinkGetFetchAndOpLocal(link,rootmtype,op,bas->rootdups[PETSCSF_LOCAL],&FetchAndOpLocal);CHKERRQ(ierr);
    ierr = (*FetchAndOpLocal)(link,count,rootstart,rootindices,rootdata,leafstart,leafindices,leafdata,leafupdate);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Create per-rank pack/unpack optimizations based on indice patterns

   Input Parameters:
  +  n       - Number of target ranks
  .  offset  - [n+1] For the i-th rank, its associated indices are idx[offset[i], offset[i+1]). offset[0] needs not to be 0.
  -  idx     - [*]   Array storing indices

   Output Parameters:
  +  opt     - Pack optimizations. NULL if no optimizations.
*/
PetscErrorCode PetscSFCreatePackOpt(PetscInt n,const PetscInt *offset,const PetscInt *idx,PetscSFPackOpt *out)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,n_copies,tot_copies = 0,step;
  PetscBool      strided,optimized = PETSC_FALSE;
  PetscSFPackOpt opt;

  PetscFunctionBegin;
  if (!n) {
    *out = NULL;
    PetscFunctionReturn(0);
  }

  ierr = PetscCalloc1(1,&opt);CHKERRQ(ierr);
  ierr = PetscCalloc3(n,&opt->type,n+1,&opt->offset,n+1,&opt->copy_offset);CHKERRQ(ierr);
  ierr = PetscArraycpy(opt->offset,offset,n+1);CHKERRQ(ierr);
  /* Make opt->offset[] zero-based. If one calls this routine with non-zero offset[0], one should use packing routine in this way, Pack(count,idx+offset[0],packopt,...) */
  if (offset[0]) {for (i=0; i<n+1; i++) opt->offset[i] -= offset[0];}

  opt->n = n;

  /* Check if the indices are piece-wise contiguous (if yes, we can optimize a packing with multiple memcpy's ) */
  for (i=0; i<n; i++) { /* for each target processor */
    /* Scan indices to count n_copies -- the number of contiguous pieces for i-th target */
    n_copies = 1;
    for (j=offset[i]; j<offset[i+1]-1; j++) {
      if (idx[j]+1 != idx[j+1]) n_copies++;
    }
    /* If the average length (in no. of indices) of contiguous pieces is long enough, say >=32,
       then it is worth using memcpy for this target. 32 is an arbitrarily chosen number.
     */
    if ((offset[i+1]-offset[i])/n_copies >= 32) {
      opt->type[i] = PETSCSF_PACKOPT_MULTICOPY;
      optimized    = PETSC_TRUE;
      tot_copies  += n_copies;
    }
  }

  /* Setup memcpy plan for each contiguous piece */
  k    = 0; /* k-th copy */
  ierr = PetscMalloc4(tot_copies,&opt->copy_start,tot_copies,&opt->copy_length,n,&opt->stride_step,n,&opt->stride_n);CHKERRQ(ierr);
  for (i=0; i<n; i++) { /* for each target processor */
    if (opt->type[i] == PETSCSF_PACKOPT_MULTICOPY) {
      n_copies           = 1;
      opt->copy_start[k] = offset[i] - offset[0];
      for (j=offset[i]; j<offset[i+1]-1; j++) {
        if (idx[j]+1 != idx[j+1]) { /* meet end of a copy (and next copy must exist) */
          n_copies++;
          opt->copy_start[k+1] = j-offset[0]+1;
          opt->copy_length[k]  = opt->copy_start[k+1] - opt->copy_start[k];
          k++;
        }
      }
      /* Set copy length of the last copy for this target */
      opt->copy_length[k] = j-offset[0]+1 - opt->copy_start[k];
      k++;
    }
    /* Set offset for next target. When opt->type[i]=PETSCSF_PACKOPT_NONE, copy_offsets[i]=copy_offsets[i+1] */
    opt->copy_offset[i+1] = k;
  }

  /* Last chance! If the indices do not have long contiguous pieces, are they strided? */
  for (i=0; i<n; i++) { /* for each remote */
    if (opt->type[i]==PETSCSF_PACKOPT_NONE && (offset[i+1] - offset[i]) >= 16) { /* few indices (<16) are not worth striding */
      strided = PETSC_TRUE;
      step    = idx[offset[i]+1] - idx[offset[i]];
      for (j=offset[i]; j<offset[i+1]-1; j++) {
        if (idx[j]+step != idx[j+1]) { strided = PETSC_FALSE; break; }
      }
      if (strided) {
        opt->type[i]         = PETSCSF_PACKOPT_STRIDE;
        opt->stride_step[i]  = step;
        opt->stride_n[i]     = offset[i+1] - offset[i];
        optimized            = PETSC_TRUE;
      }
    }
  }
  /* If no rank gets optimized, free arrays to save memory */
  if (!optimized) {
    ierr = PetscFree3(opt->type,opt->offset,opt->copy_offset);CHKERRQ(ierr);
    ierr = PetscFree4(opt->copy_start,opt->copy_length,opt->stride_step,opt->stride_n);CHKERRQ(ierr);
    ierr = PetscFree(opt);CHKERRQ(ierr);
    *out = NULL;
  } else *out = opt;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFDestroyPackOpt(PetscSFPackOpt *out)
{
  PetscErrorCode ierr;
  PetscSFPackOpt opt = *out;

  PetscFunctionBegin;
  if (opt) {
    ierr = PetscFree3(opt->type,opt->offset,opt->copy_offset);CHKERRQ(ierr);
    ierr = PetscFree4(opt->copy_start,opt->copy_length,opt->stride_step,opt->stride_n);CHKERRQ(ierr);
    ierr = PetscFree(opt);CHKERRQ(ierr);
    *out = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFSetUpPackFields(PetscSF sf)
{
  PetscErrorCode ierr;
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
  if (!sf->leafcontig[0]) {ierr = PetscSFCreatePackOpt(sf->ndranks,            sf->roffset,             sf->rmine, &sf->leafpackopt[0]);CHKERRQ(ierr);}
  if (!sf->leafcontig[1]) {ierr = PetscSFCreatePackOpt(sf->nranks-sf->ndranks, sf->roffset+sf->ndranks, sf->rmine, &sf->leafpackopt[1]);CHKERRQ(ierr);}

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

  if (!bas->rootcontig[0]) {ierr = PetscSFCreatePackOpt(bas->ndiranks,              bas->ioffset,               bas->irootloc, &bas->rootpackopt[0]);CHKERRQ(ierr);}
  if (!bas->rootcontig[1]) {ierr = PetscSFCreatePackOpt(bas->niranks-bas->ndiranks, bas->ioffset+bas->ndiranks, bas->irootloc, &bas->rootpackopt[1]);CHKERRQ(ierr);}

#if defined(PETSC_HAVE_CUDA)
    /* Check dups in indices so that CUDA unpacking kernels can use cheaper regular instructions instead of atomics when they know there are no data race chances */
  if (!sf->leafcontig[0])  {ierr = PetscCheckDupsInt(sf->leafbuflen[0],  sf->rmine,                                 &sf->leafdups[0]);CHKERRQ(ierr);}
  if (!sf->leafcontig[1])  {ierr = PetscCheckDupsInt(sf->leafbuflen[1],  sf->rmine+sf->roffset[sf->ndranks],        &sf->leafdups[1]);CHKERRQ(ierr);}
  if (!bas->rootcontig[0]) {ierr = PetscCheckDupsInt(bas->rootbuflen[0], bas->irootloc,                             &bas->rootdups[0]);CHKERRQ(ierr);}
  if (!bas->rootcontig[1]) {ierr = PetscCheckDupsInt(bas->rootbuflen[1], bas->irootloc+bas->ioffset[bas->ndiranks], &bas->rootdups[1]);CHKERRQ(ierr);}
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFResetPackFields(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=PETSCSF_LOCAL; i<=PETSCSF_REMOTE; i++) {
    ierr = PetscSFDestroyPackOpt(&sf->leafpackopt[i]);CHKERRQ(ierr);
    ierr = PetscSFDestroyPackOpt(&bas->rootpackopt[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
