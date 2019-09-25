
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

#if defined(PETSC_HAVE_CUDA)
#include <cuda_runtime.h>
#endif
/*
 * MPI_Reduce_local is not really useful because it can't handle sparse data and it vectorizes "in the wrong direction",
 * therefore we pack data types manually. This file defines packing routines for the standard data types.
 */

#define CPPJoin2(a,b)         a ##_## b
#define CPPJoin3(a,b,c)       a ##_## b ##_## c
#define CPPJoin4_(a,b,c,d)    a##b##_##c##_##d
#define CPPJoin4(a,b,c,d)     CPPJoin4_(a##_,b,c,d)

#define EXECUTE(statement)    statement /* no braces since the statement might declare a variable; braces impose an unwanted scope */
#define IGNORE(statement)     do {} while(0)

#define BINARY_OP(r,s,op,t)   do {(r) = (s) op (t);  } while(0)      /* binary ops in the middle such as +, *, && etc. */
#define FUNCTION_OP(r,s,op,t) do {(r) = op((s),(t)); } while(0)      /* ops like a function, such as PetscMax, PetscMin */
#define LXOR_OP(r,s,op,t)     do {(r) = (!(s)) != (!(t));} while(0)  /* logical exclusive OR */
#define PAIRTYPE_OP(r,s,op,t) do {(r).a = (s).a op (t).a; (r).b = (s).b op (t).b;} while(0)

#define PairType(Type1,Type2) Type1##_##Type2 /* typename for struct {Type1 a; Type2 b;} */

/* DEF_PackFunc - macro defining a Pack routine

   Arguments of the macro:
   +Type      Type of the basic data in an entry, i.e., int, PetscInt, PetscReal etc. It is not the type of an entry.
   .BS        Block size for vectorization. It is a factor of bs.
   -EQ        (bs == BS) ? 1 : 0. EQ is a compile-time const to help compiler optimizations. See below.

   Arguments of the Pack routine:
   +count     Number of indices in idx[]
   .idx       Indices of entries to packed. NULL means contiguous indices, that is [0,count)
   .link      Provide a context for the current call, such as link->bs, number of basic types in an entry. Ex. if unit is MPI_2INT, then bs=2 and the basic type is int.
   .opt       Pack optimization plans. NULL means no plan at all.
   .unpacked  Address of the unpacked data. The entries will be packed are unpacked[idx[i]],for i in [0,count)
   -packed    Address of the packed data for each rank
 */
#define DEF_PackFunc(Type,BS,EQ) \
  static PetscErrorCode CPPJoin4(Pack,Type,BS,EQ)(PetscInt count,const PetscInt *idx,PetscSFPack link,PetscSFPackOpt opt,const void *unpacked,void *packed) \
  {                                                                                                          \
    PetscErrorCode ierr;                                                                                     \
    const Type     *u = (const Type*)unpacked,*u2;                                                           \
    Type           *p = (Type*)packed,*p2;                                                                   \
    PetscInt       i,j,k,l,r,step,bs=link->bs;                                                               \
    const PetscInt *idx2,M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */    \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {ierr = PetscArraycpy(p,u,MBS*count);CHKERRQ(ierr);}  /* Indices are contiguous */             \
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

/* DEF_Action - macro defining a Unpack(Fetch)AndInsert routine

   Arguments:
  +action     Unpack or Fetch
  .Type       Type of the data
  .BS         Block size for vectorization
  .EQ        (bs == BS) ? 1 : 0. EQ is a compile-time const.
  .FILTER     Macro defining what to do with a statement, either EXECUTE or IGNORE
  .CType      Type with or without the const qualifier, i.e., const Type or Type
  .Cvoid      void with or without the const qualifier, i.e., const void or void

  Notes:
   This macro is not combined with DEF_ActionAndOp because we want to use memcpy in this macro.
   The two arguments CType and Cvoid are used (instead of one constness argument), because we want to
   get rid of compilation warning "empty macro arguments are undefined in ISO C90". With one constness argument,
   sometimes we input 'const', sometimes we have to input empty.

   If action is Fetch, we may do Malloc/Free in the routine. It is costly but the expectation is that this case is really rare.
 */
#define DEF_Action(action,Type,BS,EQ,FILTER,CType,Cvoid)               \
  static PetscErrorCode CPPJoin4(action##AndInsert,Type,BS,EQ)(PetscInt count,const PetscInt *idx,PetscSFPack link,PetscSFPackOpt opt,void *unpacked,Cvoid *packed) \
  {                                                                                                          \
    PetscErrorCode ierr;                                                                                     \
    Type           *u = (Type*)unpacked,*u2;                                                                 \
    CType          *p = (CType*)packed,*p2;                                                                  \
    PetscInt       i,j,k,l,r,step,bs=link->bs;                                                               \
    const PetscInt *idx2,M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */    \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      FILTER(Type *v);                                                                                       \
      FILTER(ierr = PetscMalloc1(count*MBS,&v);CHKERRQ(ierr));                                               \
      FILTER(ierr = PetscArraycpy(v,u,count*MBS);CHKERRQ(ierr));                                             \
             ierr = PetscArraycpy(u,p,count*MBS);CHKERRQ(ierr);                                              \
      FILTER(ierr = PetscArraycpy(p,v,count*MBS);CHKERRQ(ierr));                                             \
      FILTER(ierr = PetscFree(v);CHKERRQ(ierr));                                                             \
    } else if (!opt) { /* No optimizations available */                                                      \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
            FILTER(Type t                = u[idx[i]*MBS+j*BS+k]);                                            \
                   u[idx[i]*MBS+j*BS+k]  = p[i*MBS+j*BS+k];                                                  \
            FILTER(p[i*MBS+j*BS+k]       = t);                                                               \
          }                                                                                                  \
    } else {                                                                                                 \
      for (r=0; r<opt->n; r++) {                                                                             \
        p2 = p + opt->offset[r]*MBS;                                                                         \
        if (opt->type[r] == PETSCSF_PACKOPT_NONE) {                                                          \
          idx2 = idx + opt->offset[r];                                                                       \
          for (i=0; i<opt->offset[r+1]-opt->offset[r]; i++)                                                  \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++) {                                                                         \
                FILTER(Type t                = u[idx2[i]*MBS+j*BS+k]);                                       \
                       u[idx2[i]*MBS+j*BS+k] = p2[i*MBS+j*BS+k];                                             \
                FILTER(p2[i*MBS+j*BS+k]      = t);                                                           \
              }                                                                                              \
        } else if (opt->type[r] == PETSCSF_PACKOPT_MULTICOPY) {                                              \
          FILTER(Type *v);                                                                                   \
          FILTER(ierr = PetscMalloc1((opt->offset[r+1]-opt->offset[r])*MBS,&v);CHKERRQ(ierr)); /* max buf */ \
          for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) { /* i-th piece */                       \
            u2 = u + idx[opt->copy_start[i]]*MBS;                                                            \
            l  = opt->copy_length[i]*MBS;                                                                    \
            FILTER(ierr = PetscArraycpy(v,u2,l);CHKERRQ(ierr));                                              \
                   ierr = PetscArraycpy(u2,p2,l);CHKERRQ(ierr);                                              \
            FILTER(ierr = PetscArraycpy(p2,v,l);CHKERRQ(ierr));                                              \
            p2 += l;                                                                                         \
          }                                                                                                  \
          FILTER(ierr = PetscFree(v);CHKERRQ(ierr));                                                         \
        } else if (opt->type[r] == PETSCSF_PACKOPT_STRIDE) {                                                 \
          u2   = u + idx[opt->offset[r]]*MBS;                                                                \
          step = opt->stride_step[r];                                                                        \
          for (i=0; i<opt->stride_n[r]; i++)                                                                 \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++) {                                                                         \
                FILTER(Type t                = u2[i*step*MBS+j*BS+k]);                                       \
                       u2[i*step*MBS+j*BS+k] = p2[i*MBS+j*BS+k];                                             \
                FILTER(p2[i*MBS+j*BS+k]      = t);                                                           \
              }                                                                                              \
        } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unknown SFPack optimzation type %D",opt->type[r]);   \
      }                                                                                                      \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

/* DEF_ActionAndOp - macro defining a Unpack(Fetch)AndOp routine. Op can not be Insert, Maxloc or Minloc

   Arguments:
  +action     Unpack or Fetch
  .opname     Name of the Op, such as Add, Mult, LAND, etc.
  .Type       Type of the data
  .BS         Block size for vectorization
  .EQ         (bs == BS) ? 1 : 0. EQ is a compile-time const.
  .op         Operator for the op, such as +, *, &&, ||, PetscMax, PetscMin, etc.
  .APPLY      Macro defining application of the op. Could be BINARY_OP, FUNCTION_OP, LXOR_OP or PAIRTYPE_OP
  .FILTER     Macro defining what to do with a statement, either EXECUTE or IGNORE
  .CType      Type with or without the const qualifier, i.e., const Type or Type
  -Cvoid      void with or without the const qualifier, i.e., const void or void
 */
#define DEF_ActionAndOp(action,opname,Type,BS,EQ,op,APPLY,FILTER,CType,Cvoid) \
  static PetscErrorCode CPPJoin4(action##And##opname,Type,BS,EQ)(PetscInt count,const PetscInt *idx,PetscSFPack link,PetscSFPackOpt opt,void *unpacked,Cvoid *packed) \
  {                                                                                                          \
    Type           *u = (Type*)unpacked,*u2,t;                                                               \
    CType          *p = (CType*)packed,*p2;                                                                  \
    PetscInt       i,j,k,l,r,step,bs=link->bs;                                                               \
    const PetscInt *idx2,M = (EQ) ? 1 : bs/BS; /* If EQ, then M=1 enables compiler's const-propagation */    \
    const PetscInt MBS = M*BS; /* MBS=bs. We turn MBS into a compile time const when EQ=1. */                \
    PetscFunctionBegin;                                                                                      \
    if (!idx) {                                                                                              \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
            t    = u[i*MBS+j*BS+k];                                                                          \
            APPLY (u[i*MBS+j*BS+k],t,op,p[i*MBS+j*BS+k]);                                                    \
            FILTER(p[i*MBS+j*BS+k] = t);                                                                     \
          }                                                                                                  \
    } else if (!opt) { /* No optimizations available */                                                      \
      for (i=0; i<count; i++)                                                                                \
        for (j=0; j<M; j++)                                                                                  \
          for (k=0; k<BS; k++) {                                                                             \
              t    = u[idx[i]*MBS+j*BS+k];                                                                   \
              APPLY (u[idx[i]*MBS+j*BS+k],t,op,p[i*MBS+j*BS+k]);                                             \
              FILTER(p[i*MBS+j*BS+k] = t);                                                                   \
          }                                                                                                  \
    } else {                                                                                                 \
      for (r=0; r<opt->n; r++) {                                                                             \
        p2 = p + opt->offset[r]*MBS;                                                                         \
        if (opt->type[r] == PETSCSF_PACKOPT_NONE) {                                                          \
          idx2 = idx + opt->offset[r];                                                                       \
          for (i=0; i<opt->offset[r+1]-opt->offset[r]; i++)                                                  \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++) {                                                                         \
                t    = u[idx2[i]*MBS+j*BS+k];                                                                \
                APPLY (u[idx2[i]*MBS+j*BS+k],t,op,p2[i*MBS+j*BS+k]);                                         \
                FILTER(p2[i*MBS+j*BS+k] = t);                                                                \
              }                                                                                              \
        } else if (opt->type[r] == PETSCSF_PACKOPT_MULTICOPY) {                                              \
          for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) { /* i-th piece */                       \
            u2 = u + idx[opt->copy_start[i]]*MBS;                                                            \
            l  = opt->copy_length[i]*MBS;                                                                    \
            for (j=0; j<l; j++) {                                                                            \
              t    = u2[j];                                                                                  \
              APPLY (u2[j],t,op,p2[j]);                                                                      \
              FILTER(p2[j] = t);                                                                             \
            }                                                                                                \
            p2 += l;                                                                                         \
          }                                                                                                  \
        } else if (opt->type[r] == PETSCSF_PACKOPT_STRIDE) {                                                 \
          u2   = u + idx[opt->offset[r]]*MBS;                                                                \
          step = opt->stride_step[r];                                                                        \
          for (i=0; i<opt->stride_n[r]; i++)                                                                 \
            for (j=0; j<M; j++)                                                                              \
              for (k=0; k<BS; k++) {                                                                         \
                t    = u2[i*step*MBS+j*BS+k];                                                                \
                APPLY (u2[i*step*MBS+j*BS+k],t,op,p2[i*MBS+j*BS+k]);                                         \
                FILTER(p2[i*MBS+j*BS+k] = t);                                                                \
              }                                                                                              \
        } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unknown SFPack optimzation type %D",opt->type[r]);   \
      }                                                                                                      \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

/* DEF_ActionAndXloc - macro defining a Unpack(Fetch)AndMaxloc(Minloc) routine

   Arguments:
  +Action     Unpack or Fetch
  .locname    Max or Min
  .type1      Type of the first data in a pair type
  .type2      Type of the second data in a pair type, usually PetscMPIInt for MPI ranks.
  .op         > or <
  .FILTER     Macro defining what to do with a statement, either EXECUTE or IGNORE
  .CType      Type with or without the const qualifier, i.e., const PairType(Type1,Type2) or PairType(Type1,Type2)
  -Cvoid      void with or without the const qualifier, i.e., const void or void
 */
#define DEF_ActionAndXloc(action,locname,Type1,Type2,op,FILTER,CType,Cvoid) \
  static PetscErrorCode CPPJoin4(action##And##locname##loc,PairType(Type1,Type2),1,1)(PetscInt count,const PetscInt *idx,PetscSFPack link,PetscSFPackOpt opt,void *unpacked,Cvoid *packed) { \
    PairType(Type1,Type2) *u = (PairType(Type1,Type2)*)unpacked;                                             \
    CType                 *p = (CType*)packed;                                                               \
    PetscInt              i,j;                                                                               \
    for (i=0; i<count; i++) {                                                                                \
      FILTER(PairType(Type1,Type2) v);                                                                       \
      j = idx? idx[i] : i;                                                                                   \
      FILTER(v = u[j]);                                                                                      \
      if (p[i].a op u[j].a) {                                                                                \
        u[j] = p[i];                                                                                         \
      } else if (p[i].a == u[j].a) {                                                                         \
        u[j].b = PetscMin(u[j].b,p[i].b); /* Minimal rank. Ref MPI MAXLOC */                                 \
      }                                                                                                      \
      FILTER(p[i] = v);                                                                                      \
    }                                                                                                        \
    PetscFunctionReturn(0);                                                                                  \
  }

/* Pack, Unpack/Fetch ops */
#define DEF_Pack(Type,BS,EQ)                                                                   \
  DEF_PackFunc(Type,BS,EQ)                                                                     \
  DEF_Action(Unpack,Type,BS,EQ,IGNORE,const Type,const void)                                   \
  DEF_Action(Fetch, Type,BS,EQ,EXECUTE,Type,void)                                              \
  static void CPPJoin4(PackInit_Pack,Type,BS,EQ)(PetscSFPack link) {                           \
    link->h_Pack            = CPPJoin4(Pack,           Type,BS,EQ);                            \
    link->h_UnpackAndInsert = CPPJoin4(UnpackAndInsert,Type,BS,EQ);                            \
    link->h_FetchAndInsert  = CPPJoin4(FetchAndInsert, Type,BS,EQ);                            \
  }

/* Add, Mult ops */
#define DEF_Add(Type,BS,EQ)                                                                    \
  DEF_ActionAndOp(Unpack,Add, Type,BS,EQ,+,BINARY_OP,IGNORE,const Type,const void)             \
  DEF_ActionAndOp(Unpack,Mult,Type,BS,EQ,*,BINARY_OP,IGNORE,const Type,const void)             \
  DEF_ActionAndOp(Fetch, Add, Type,BS,EQ,+,BINARY_OP,EXECUTE,Type,void)                        \
  DEF_ActionAndOp(Fetch, Mult,Type,BS,EQ,*,BINARY_OP,EXECUTE,Type,void)                        \
  static void CPPJoin4(PackInit_Add,Type,BS,EQ)(PetscSFPack link) {                            \
    link->h_UnpackAndAdd    = CPPJoin4(UnpackAndAdd,   Type,BS,EQ);                            \
    link->h_UnpackAndMult   = CPPJoin4(UnpackAndMult,  Type,BS,EQ);                            \
    link->h_FetchAndAdd     = CPPJoin4(FetchAndAdd,    Type,BS,EQ);                            \
    link->h_FetchAndMult    = CPPJoin4(FetchAndMult,   Type,BS,EQ);                            \
  }

/* Max, Min ops */
#define DEF_Cmp(Type,BS,EQ)                                                                    \
  DEF_ActionAndOp(Unpack,Max,Type,BS,EQ,PetscMax,FUNCTION_OP,IGNORE,const Type,const void)     \
  DEF_ActionAndOp(Unpack,Min,Type,BS,EQ,PetscMin,FUNCTION_OP,IGNORE,const Type,const void)     \
  DEF_ActionAndOp(Fetch, Max,Type,BS,EQ,PetscMax,FUNCTION_OP,EXECUTE,Type,void)                \
  DEF_ActionAndOp(Fetch, Min,Type,BS,EQ,PetscMin,FUNCTION_OP,EXECUTE,Type,void)                \
  static void CPPJoin4(PackInit_Compare,Type,BS,EQ)(PetscSFPack link) {                        \
    link->h_UnpackAndMax    = CPPJoin4(UnpackAndMax,   Type,BS,EQ);                            \
    link->h_UnpackAndMin    = CPPJoin4(UnpackAndMin,   Type,BS,EQ);                            \
    link->h_FetchAndMax     = CPPJoin4(FetchAndMax ,   Type,BS,EQ);                            \
    link->h_FetchAndMin     = CPPJoin4(FetchAndMin ,   Type,BS,EQ);                            \
  }

/* Logical ops.
  The operator in LXOR_OP should be empty but is &. It is not used. Put here to avoid
  the compilation warning "empty macro arguments are undefined in ISO C90"
 */
#define DEF_Log(Type,BS,EQ)                                                                    \
  DEF_ActionAndOp(Unpack,LAND,Type,BS,EQ,&&,BINARY_OP,IGNORE,const Type,const void)            \
  DEF_ActionAndOp(Unpack,LOR, Type,BS,EQ,||,BINARY_OP,IGNORE,const Type,const void)            \
  DEF_ActionAndOp(Unpack,LXOR,Type,BS,EQ,&, LXOR_OP,  IGNORE,const Type,const void)            \
  DEF_ActionAndOp(Fetch, LAND,Type,BS,EQ,&&,BINARY_OP,EXECUTE,Type,void)                       \
  DEF_ActionAndOp(Fetch, LOR, Type,BS,EQ,||,BINARY_OP,EXECUTE,Type,void)                       \
  DEF_ActionAndOp(Fetch, LXOR,Type,BS,EQ,&, LXOR_OP,  EXECUTE,Type,void)                       \
  static void CPPJoin4(PackInit_Logical,Type,BS,EQ)(PetscSFPack link) {                        \
    link->h_UnpackAndLAND   = CPPJoin4(UnpackAndLAND,Type,BS,EQ);                              \
    link->h_UnpackAndLOR    = CPPJoin4(UnpackAndLOR, Type,BS,EQ);                              \
    link->h_UnpackAndLXOR   = CPPJoin4(UnpackAndLXOR,Type,BS,EQ);                              \
    link->h_FetchAndLAND    = CPPJoin4(FetchAndLAND, Type,BS,EQ);                              \
    link->h_FetchAndLOR     = CPPJoin4(FetchAndLOR,  Type,BS,EQ);                              \
    link->h_FetchAndLXOR    = CPPJoin4(FetchAndLXOR, Type,BS,EQ);                              \
  }

/* Bitwise ops */
#define DEF_Bit(Type,BS,EQ)                                                                    \
  DEF_ActionAndOp(Unpack,BAND,Type,BS,EQ,&,BINARY_OP,IGNORE,const Type,const void)             \
  DEF_ActionAndOp(Unpack,BOR, Type,BS,EQ,|,BINARY_OP,IGNORE,const Type,const void)             \
  DEF_ActionAndOp(Unpack,BXOR,Type,BS,EQ,^,BINARY_OP,IGNORE,const Type,const void)             \
  DEF_ActionAndOp(Fetch, BAND,Type,BS,EQ,&,BINARY_OP,EXECUTE,Type,void)                        \
  DEF_ActionAndOp(Fetch, BOR, Type,BS,EQ,|,BINARY_OP,EXECUTE,Type,void)                        \
  DEF_ActionAndOp(Fetch, BXOR,Type,BS,EQ,^,BINARY_OP,EXECUTE,Type,void)                        \
  static void CPPJoin4(PackInit_Bitwise,Type,BS,EQ)(PetscSFPack link) {                        \
    link->h_UnpackAndBAND   = CPPJoin4(UnpackAndBAND,Type,BS,EQ);                              \
    link->h_UnpackAndBOR    = CPPJoin4(UnpackAndBOR, Type,BS,EQ);                              \
    link->h_UnpackAndBXOR   = CPPJoin4(UnpackAndBXOR,Type,BS,EQ);                              \
    link->h_FetchAndBAND    = CPPJoin4(FetchAndBAND, Type,BS,EQ);                              \
    link->h_FetchAndBOR     = CPPJoin4(FetchAndBOR,  Type,BS,EQ);                              \
    link->h_FetchAndBXOR    = CPPJoin4(FetchAndBXOR, Type,BS,EQ);                              \
  }

/* Maxloc, Minloc */
#define DEF_Xloc(Type1,Type2)                                                                  \
  DEF_ActionAndXloc(Unpack,Max,Type1,Type2,>,IGNORE,const PairType(Type1,Type2),const void)    \
  DEF_ActionAndXloc(Unpack,Min,Type1,Type2,<,IGNORE,const PairType(Type1,Type2),const void)    \
  DEF_ActionAndXloc(Fetch, Max,Type1,Type2,>,EXECUTE,PairType(Type1,Type2),void)               \
  DEF_ActionAndXloc(Fetch, Min,Type1,Type2,<,EXECUTE,PairType(Type1,Type2),void)               \
  static void CPPJoin3(PackInit_Xloc,Type1,Type2)(PetscSFPack link) {                          \
    link->h_UnpackAndMaxloc = CPPJoin4(UnpackAndMaxloc,PairType(Type1,Type2),1,1);             \
    link->h_UnpackAndMinloc = CPPJoin4(UnpackAndMinloc,PairType(Type1,Type2),1,1);             \
    link->h_FetchAndMaxloc  = CPPJoin4(FetchAndMaxloc, PairType(Type1,Type2),1,1);             \
    link->h_FetchAndMinloc  = CPPJoin4(FetchAndMinloc, PairType(Type1,Type2),1,1);             \
  }

#define DEF_IntegerType(Type,BS,EQ)                                                            \
  DEF_Pack(Type,BS,EQ)                                                                         \
  DEF_Add(Type,BS,EQ)                                                                          \
  DEF_Cmp(Type,BS,EQ)                                                                          \
  DEF_Log(Type,BS,EQ)                                                                          \
  DEF_Bit(Type,BS,EQ)                                                                          \
  static void CPPJoin4(PackInit_IntegerType,Type,BS,EQ)(PetscSFPack link) {                    \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                                  \
    CPPJoin4(PackInit_Add,Type,BS,EQ)(link);                                                   \
    CPPJoin4(PackInit_Compare,Type,BS,EQ)(link);                                               \
    CPPJoin4(PackInit_Logical,Type,BS,EQ)(link);                                               \
    CPPJoin4(PackInit_Bitwise,Type,BS,EQ)(link);                                               \
  }

#define DEF_RealType(Type,BS,EQ)                                                               \
  DEF_Pack(Type,BS,EQ)                                                                         \
  DEF_Add(Type,BS,EQ)                                                                          \
  DEF_Cmp(Type,BS,EQ)                                                                          \
  static void CPPJoin4(PackInit_RealType,Type,BS,EQ)(PetscSFPack link) {                       \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                                  \
    CPPJoin4(PackInit_Add,Type,BS,EQ)(link);                                                   \
    CPPJoin4(PackInit_Compare,Type,BS,EQ)(link);                                               \
  }

#if defined(PETSC_HAVE_COMPLEX)
#define DEF_ComplexType(Type,BS,EQ)                                                            \
  DEF_Pack(Type,BS,EQ)                                                                         \
  DEF_Add(Type,BS,EQ)                                                                          \
  static void CPPJoin4(PackInit_ComplexType,Type,BS,EQ)(PetscSFPack link) {                    \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                                  \
    CPPJoin4(PackInit_Add,Type,BS,EQ)(link);                                                   \
  }
#endif

#define DEF_DumbType(Type,BS,EQ)                                                               \
  DEF_Pack(Type,BS,EQ)                                                                         \
  static void CPPJoin4(PackInit_DumbType,Type,BS,EQ)(PetscSFPack link) {                       \
    CPPJoin4(PackInit_Pack,Type,BS,EQ)(link);                                                  \
  }

/* Maxloc, Minloc */
#define DEF_PairType(Type1,Type2)                                                              \
  typedef struct {Type1 a; Type2 b;} PairType(Type1,Type2);                                    \
  DEF_Pack(PairType(Type1,Type2),1,1)                                                          \
  DEF_Xloc(Type1,Type2)                                                                        \
  static void CPPJoin3(PackInit_PairType,Type1,Type2)(PetscSFPack link) {                      \
    CPPJoin4(PackInit_Pack,PairType(Type1,Type2),1,1)(link);                                   \
    CPPJoin3(PackInit_Xloc,Type1,Type2)(link);                                                 \
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

DEF_PairType(int,int)
DEF_PairType(PetscInt,PetscInt)

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

PetscErrorCode PetscSFPackGetInUse(PetscSF sf,MPI_Datatype unit,const void *rkey,const void *lkey,PetscCopyMode cmode,PetscSFPack *mylink)
{
  PetscErrorCode    ierr;
  PetscSFPack       link,*p;
  PetscSF_Basic     *bas=(PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (p=&bas->inuse; (link=*p); p=&link->next) {
    PetscBool match;
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match && (rkey == link->rkey) && (lkey == link->lkey)) {
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

PetscErrorCode PetscSFPackReclaim(PetscSF sf,PetscSFPack *link)
{
  PetscSF_Basic     *bas=(PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  (*link)->rkey = NULL;
  (*link)->lkey = NULL;
  (*link)->next = bas->avail;
  bas->avail    = *link;
  *link         = NULL;
  PetscFunctionReturn(0);
}

/* Destroy all links, i.e., PetscSFPacks in the linked list, usually named 'avail' */
PetscErrorCode PetscSFPackDestoryAvailable(PetscSFPack *avail)
{
  PetscErrorCode    ierr;
  PetscSFPack       link=*avail,next;
  PetscInt          i;

  PetscFunctionBegin;
  for (; link; link=next) {
    next = link->next;
    if (!link->isbuiltin) {ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);}
    for (i=0; i<(link->nrootreqs+link->nleafreqs)*4; i++) { /* Persistent reqs must be freed. */
      if (link->reqs[i] != MPI_REQUEST_NULL) {ierr = MPI_Request_free(&link->reqs[i]);CHKERRQ(ierr);}
    }
    ierr = PetscFree(link->reqs);CHKERRQ(ierr);
    ierr = PetscFreeWithMemType(PETSC_MEMTYPE_HOST,link->rootbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
    ierr = PetscFreeWithMemType(PETSC_MEMTYPE_HOST,link->leafbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);
    ierr = PetscFreeWithMemType(PETSC_MEMTYPE_HOST,link->selfbuf[PETSC_MEMTYPE_HOST]);CHKERRQ(ierr);

#if defined(PETSC_HAVE_CUDA)
    ierr = PetscFreeWithMemType(PETSC_MEMTYPE_DEVICE,link->rootbuf[PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr = PetscFreeWithMemType(PETSC_MEMTYPE_DEVICE,link->leafbuf[PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr = PetscFreeWithMemType(PETSC_MEMTYPE_DEVICE,link->selfbuf[PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    if (link->stream) {cudaError_t err =  cudaStreamDestroy(link->stream);CHKERRCUDA(err); link->stream = NULL;}
#endif
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  *avail = NULL;
  PetscFunctionReturn(0);
}

/* Error out on unsupported overlapped communications */
PetscErrorCode PetscSFPackSetErrorOnUnsupportedOverlap(PetscSF sf,MPI_Datatype unit,const void *rkey,const void *lkey)
{
  PetscErrorCode    ierr;
  PetscSFPack       link,*p;
  PetscSF_Basic     *bas=(PetscSF_Basic*)sf->data;
  PetscBool         match;

  PetscFunctionBegin;
  /* Look up links in use and error out if there is a match. When both rootdata and leafdata are NULL, ignore
     the potential overlapping since this process does not participate in communication. Overlapping is harmless.
  */
  if (rkey || lkey) {
    for (p=&bas->inuse; (link=*p); p=&link->next) {
      ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
      if (match && (rkey == link->rkey) && (lkey == link->lkey)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for overlapped PetscSF communications with the same SF, rootdata(%p), leafdata(%p) and data type. You can undo the overlap to avoid the error.",rkey,lkey);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFPackSetUp_Host(PetscSF sf,PetscSFPack link,MPI_Datatype unit)
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
  link->isbuiltin = (combiner == MPI_COMBINER_NAMED) ? PETSC_TRUE : PETSC_FALSE;
  link->bs = 1; /* default */

  if (is2Int) {
    PackInit_PairType_int_int(link);
    link->bs        = 1;
    link->unitbytes = 2*sizeof(int);
    link->basicunit = MPI_2INT;
  } else if (is2PetscInt) { /* TODO: when is2PetscInt and nPetscInt=2, we don't know which path to take. The two paths support different ops. */
    PackInit_PairType_PetscInt_PetscInt(link);
    link->bs        = 1;
    link->unitbytes = 2*sizeof(PetscInt);
    link->basicunit = MPIU_2INT;
  } else if (nPetscReal) {
    if      (nPetscReal == 8) PackInit_RealType_PetscReal_8_1(link); else if (nPetscReal%8 == 0) PackInit_RealType_PetscReal_8_0(link);
    else if (nPetscReal == 4) PackInit_RealType_PetscReal_4_1(link); else if (nPetscReal%4 == 0) PackInit_RealType_PetscReal_4_0(link);
    else if (nPetscReal == 2) PackInit_RealType_PetscReal_2_1(link); else if (nPetscReal%2 == 0) PackInit_RealType_PetscReal_2_0(link);
    else if (nPetscReal == 1) PackInit_RealType_PetscReal_1_1(link); else if (nPetscReal%1 == 0) PackInit_RealType_PetscReal_1_0(link);
    link->bs        = nPetscReal;
    link->unitbytes = nPetscReal*sizeof(PetscReal);
    link->basicunit = MPIU_REAL;
  } else if (nPetscInt) {
    if      (nPetscInt == 8) PackInit_IntegerType_PetscInt_8_1(link); else if (nPetscInt%8 == 0) PackInit_IntegerType_PetscInt_8_0(link);
    else if (nPetscInt == 4) PackInit_IntegerType_PetscInt_4_1(link); else if (nPetscInt%4 == 0) PackInit_IntegerType_PetscInt_4_0(link);
    else if (nPetscInt == 2) PackInit_IntegerType_PetscInt_2_1(link); else if (nPetscInt%2 == 0) PackInit_IntegerType_PetscInt_2_0(link);
    else if (nPetscInt == 1) PackInit_IntegerType_PetscInt_1_1(link); else if (nPetscInt%1 == 0) PackInit_IntegerType_PetscInt_1_0(link);
    link->bs        = nPetscInt;
    link->unitbytes = nPetscInt*sizeof(PetscInt);
    link->basicunit = MPIU_INT;
#if defined(PETSC_USE_64BIT_INDICES)
  } else if (nInt) {
    if      (nInt == 8) PackInit_IntegerType_int_8_1(link); else if (nInt%8 == 0) PackInit_IntegerType_int_8_0(link);
    else if (nInt == 4) PackInit_IntegerType_int_4_1(link); else if (nInt%4 == 0) PackInit_IntegerType_int_4_0(link);
    else if (nInt == 2) PackInit_IntegerType_int_2_1(link); else if (nInt%2 == 0) PackInit_IntegerType_int_2_0(link);
    else if (nInt == 1) PackInit_IntegerType_int_1_1(link); else if (nInt%1 == 0) PackInit_IntegerType_int_1_0(link);
    link->bs        = nInt;
    link->unitbytes = nInt*sizeof(int);
    link->basicunit = MPI_INT;
#endif
  } else if (nSignedChar) {
    if      (nSignedChar == 8) PackInit_IntegerType_SignedChar_8_1(link); else if (nSignedChar%8 == 0) PackInit_IntegerType_SignedChar_8_0(link);
    else if (nSignedChar == 4) PackInit_IntegerType_SignedChar_4_1(link); else if (nSignedChar%4 == 0) PackInit_IntegerType_SignedChar_4_0(link);
    else if (nSignedChar == 2) PackInit_IntegerType_SignedChar_2_1(link); else if (nSignedChar%2 == 0) PackInit_IntegerType_SignedChar_2_0(link);
    else if (nSignedChar == 1) PackInit_IntegerType_SignedChar_1_1(link); else if (nSignedChar%1 == 0) PackInit_IntegerType_SignedChar_1_0(link);
    link->bs        = nSignedChar;
    link->unitbytes = nSignedChar*sizeof(SignedChar);
    link->basicunit = MPI_SIGNED_CHAR;
  }  else if (nUnsignedChar) {
    if      (nUnsignedChar == 8) PackInit_IntegerType_UnsignedChar_8_1(link); else if (nUnsignedChar%8 == 0) PackInit_IntegerType_UnsignedChar_8_0(link);
    else if (nUnsignedChar == 4) PackInit_IntegerType_UnsignedChar_4_1(link); else if (nUnsignedChar%4 == 0) PackInit_IntegerType_UnsignedChar_4_0(link);
    else if (nUnsignedChar == 2) PackInit_IntegerType_UnsignedChar_2_1(link); else if (nUnsignedChar%2 == 0) PackInit_IntegerType_UnsignedChar_2_0(link);
    else if (nUnsignedChar == 1) PackInit_IntegerType_UnsignedChar_1_1(link); else if (nUnsignedChar%1 == 0) PackInit_IntegerType_UnsignedChar_1_0(link);
    link->bs        = nUnsignedChar;
    link->unitbytes = nUnsignedChar*sizeof(UnsignedChar);
    link->basicunit = MPI_UNSIGNED_CHAR;
#if defined(PETSC_HAVE_COMPLEX)
  } else if (nPetscComplex) {
    if      (nPetscComplex == 8) PackInit_ComplexType_PetscComplex_8_1(link); else if (nPetscComplex%8 == 0) PackInit_ComplexType_PetscComplex_8_0(link);
    else if (nPetscComplex == 4) PackInit_ComplexType_PetscComplex_4_1(link); else if (nPetscComplex%4 == 0) PackInit_ComplexType_PetscComplex_4_0(link);
    else if (nPetscComplex == 2) PackInit_ComplexType_PetscComplex_2_1(link); else if (nPetscComplex%2 == 0) PackInit_ComplexType_PetscComplex_2_0(link);
    else if (nPetscComplex == 1) PackInit_ComplexType_PetscComplex_1_1(link); else if (nPetscComplex%1 == 0) PackInit_ComplexType_PetscComplex_1_0(link);
    link->bs        = nPetscComplex;
    link->unitbytes = nPetscComplex*sizeof(PetscComplex);
    link->basicunit = MPIU_COMPLEX;
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
  }

  if (link->isbuiltin) link->unit = unit; /* builtin datatypes are common. Make it fast */
  else {ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFPackGetUnpackAndOp(PetscSFPack link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**UnpackAndOp)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*))
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
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %D",mtype);

  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFPackGetFetchAndOp(PetscSFPack link,PetscMemType mtype,MPI_Op op,PetscBool atomic,PetscErrorCode (**FetchAndOp)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,void*))
{
  PetscFunctionBegin;
  *FetchAndOp = NULL;
  if (mtype == PETSC_MEMTYPE_HOST) {
    if (op == MPIU_REPLACE)                   *FetchAndOp = link->h_FetchAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *FetchAndOp = link->h_FetchAndAdd;
    else if (op == MPI_MAX || op == MPIU_MAX) *FetchAndOp = link->h_FetchAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *FetchAndOp = link->h_FetchAndMin;
    else if (op == MPI_MAXLOC)                *FetchAndOp = link->h_FetchAndMaxloc;
    else if (op == MPI_MINLOC)                *FetchAndOp = link->h_FetchAndMinloc;
    else if (op == MPI_PROD)                  *FetchAndOp = link->h_FetchAndMult;
    else if (op == MPI_LAND)                  *FetchAndOp = link->h_FetchAndLAND;
    else if (op == MPI_BAND)                  *FetchAndOp = link->h_FetchAndBAND;
    else if (op == MPI_LOR)                   *FetchAndOp = link->h_FetchAndLOR;
    else if (op == MPI_BOR)                   *FetchAndOp = link->h_FetchAndBOR;
    else if (op == MPI_LXOR)                  *FetchAndOp = link->h_FetchAndLXOR;
    else if (op == MPI_BXOR)                  *FetchAndOp = link->h_FetchAndBXOR;
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op");
  }
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE && !atomic) {
    if (op == MPIU_REPLACE)                   *FetchAndOp = link->d_FetchAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *FetchAndOp = link->d_FetchAndAdd;
    else if (op == MPI_MAX || op == MPIU_MAX) *FetchAndOp = link->d_FetchAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *FetchAndOp = link->d_FetchAndMin;
    else if (op == MPI_MAXLOC)                *FetchAndOp = link->d_FetchAndMaxloc;
    else if (op == MPI_MINLOC)                *FetchAndOp = link->d_FetchAndMinloc;
    else if (op == MPI_PROD)                  *FetchAndOp = link->d_FetchAndMult;
    else if (op == MPI_LAND)                  *FetchAndOp = link->d_FetchAndLAND;
    else if (op == MPI_BAND)                  *FetchAndOp = link->d_FetchAndBAND;
    else if (op == MPI_LOR)                   *FetchAndOp = link->d_FetchAndLOR;
    else if (op == MPI_BOR)                   *FetchAndOp = link->d_FetchAndBOR;
    else if (op == MPI_LXOR)                  *FetchAndOp = link->d_FetchAndLXOR;
    else if (op == MPI_BXOR)                  *FetchAndOp = link->d_FetchAndBXOR;
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op");
  } else if (mtype == PETSC_MEMTYPE_DEVICE && atomic) {
    if (op == MPIU_REPLACE)                   *FetchAndOp = link->da_FetchAndInsert;
    else if (op == MPI_SUM || op == MPIU_SUM) *FetchAndOp = link->da_FetchAndAdd;
    else if (op == MPI_MAX || op == MPIU_MAX) *FetchAndOp = link->da_FetchAndMax;
    else if (op == MPI_MIN || op == MPIU_MIN) *FetchAndOp = link->da_FetchAndMin;
    else if (op == MPI_MAXLOC)                *FetchAndOp = link->da_FetchAndMaxloc;
    else if (op == MPI_MINLOC)                *FetchAndOp = link->da_FetchAndMinloc;
    else if (op == MPI_PROD)                  *FetchAndOp = link->da_FetchAndMult;
    else if (op == MPI_LAND)                  *FetchAndOp = link->da_FetchAndLAND;
    else if (op == MPI_BAND)                  *FetchAndOp = link->da_FetchAndBAND;
    else if (op == MPI_LOR)                   *FetchAndOp = link->da_FetchAndLOR;
    else if (op == MPI_BOR)                   *FetchAndOp = link->da_FetchAndBOR;
    else if (op == MPI_LXOR)                  *FetchAndOp = link->da_FetchAndLXOR;
    else if (op == MPI_BXOR)                  *FetchAndOp = link->da_FetchAndBXOR;
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for MPI_Op");
  }
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %D",mtype);
  PetscFunctionReturn(0);
}

/*
  Create pack/unpack optimization plans based on indice patterns available

   Input Parameters:
  +  n       - Number of target ranks
  .  offset  - [n+1] For the i-th rank, its associated indices are idx[offset[i], offset[i+1]). offset[0] needs not to be 0.
  -  idx     - [*]   Array storing indices

   Output Parameters:
  +  opt    - Optimization plans. Maybe NULL if no optimization can be built.
*/
PetscErrorCode PetscSFPackOptCreate(PetscInt n,const PetscInt *offset,const PetscInt *idx,PetscSFPackOpt *out)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,n_copies,tot_copies=0,step;
  PetscBool      strided,optimized=PETSC_FALSE;
  PetscSFPackOpt opt;

  PetscFunctionBegin;
  if (!n) {
    *out = NULL;
    PetscFunctionReturn(0);
  }

  ierr = PetscCalloc1(1,&opt);CHKERRQ(ierr);
  ierr = PetscCalloc3(n,&opt->type,n+1,&opt->offset,n+1,&opt->copy_offset);CHKERRQ(ierr);
  ierr = PetscArraycpy(opt->offset,offset,n+1);CHKERRQ(ierr);
  if (offset[0]) {for (i=0; i<n+1; i++) opt->offset[i] -= offset[0];} /* Zero-base offset[]. Note the packing routine is Pack(count, idx[], ...*/

  opt->n = n;

  /* Check if the indices are piece-wise contiguous (if yes, we can optimize a packing with mulitple memcpy's ) */
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

PetscErrorCode PetscSFPackOptDestory(PetscSFPackOpt *out)
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
