
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

/*
 * MPI_Reduce_local is not really useful because it can't handle sparse data and it vectorizes "in the wrong direction",
 * therefore we pack data types manually. This file defines packing routines for the standard data types.
 */

#define CPPJoin2_exp(a,b)     a ## b
#define CPPJoin2(a,b)         CPPJoin2_exp(a,b)
#define CPPJoin3_exp_(a,b,c)  a ## b ## _ ## c
#define CPPJoin3_(a,b,c)      CPPJoin3_exp_(a,b,c)

#define EXECUTE(statement)    statement /* no braces since the statement might declare a variable; braces impose an unwanted scope */
#define IGNORE(statement)     do {} while(0)

#define BINARY_OP(r,s,op,t)   do {(r) = (s) op (t);  } while(0)  /* binary ops in the middle such as +, *, && etc. */
#define FUNCTION_OP(r,s,op,t) do {(r) = op((s),(t)); } while(0)  /* ops like a function, such as PetscMax, PetscMin */
#define LXOR_OP(r,s,op,t)     do {(r) = (!s) != (!t);} while(0)  /* logical exclusive OR */
#define PAIRTYPE_OP(r,s,op,t) do {(r).a = (s).a op (t).a; (r).b = (s).b op (t).b;} while(0)

#define BlockType(type,count) CPPJoin3_(_blocktype_,type,count) /* typename for struct {type v[count];} */
#define PairType(type1,type2) CPPJoin3_(_pairtype_,type1,type2) /* typename for struct {type1 a; type2 b;} */

/* DEF_PackFunc - macro defining a Pack routine

   Arguments of the macro:
   +type      Type of the basic data in an entry, i.e., int, PetscInt, PetscReal etc. It is not the type of an entry.
   -BS        Block size for vectorization. It is a factor of bs.

   Arguments of the Pack routine:
   +n         Number of entries to pack. Each entry is of type 'unit'. Here the unit is the arg used in calls like PetscSFBcastBegin(sf,unit,..).
              If idx is not NULL, then n also indicates the number of indices in idx[]
   .bs        Number of basic types in an entry. Ex. if unit is MPI_2INT, then bs=2 and the basic type is int
   .idx       Indices of entries. NULL means contiguous indices [0,n)
   .r         Do packing for the r-th destination process
   .opt       Pack optimization plans. NULL means no plan.
   .unpacked  Address of the unpacked data
   -packed    Address of the packed data
 */
#define DEF_PackFunc(type,BS) \
  static PetscErrorCode CPPJoin3_(Pack_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,PetscInt r,PetscSFPackOpt opt,const void *unpacked,void *packed) { \
    PetscErrorCode ierr;                                                                                   \
    const type     *u = (const type*)unpacked,*u2;                                                         \
    type           *p = (type*)packed;                                                                     \
    PetscInt       i,j,k,l,step;                                                                           \
    PetscFunctionBegin;                                                                                    \
    if (!idx) {  /* idx[] is contiguous */                                                                 \
      ierr = PetscArraycpy(p,u,bs*n);CHKERRQ(ierr);                                                        \
    } else if (!opt || !opt->optimized[r]) { /* idx[] is not optimized*/                                   \
      for (i=0; i<n; i++)                                                                                  \
        for (j=0; j<bs; j+=BS)                                                                             \
          for (k=j; k<j+BS; k++)                                                                           \
            p[i*bs+k] = u[idx[i]*bs+k];                                                                    \
    } else { /* idx[] is optimized*/                                                                       \
      if (opt->copy_offset[r] != opt->copy_offset[r+1]) { /* idx[] is piece-wise contiguous */             \
        for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) {                                        \
          l    = opt->copy_length[i]*bs; /* length in types */                                             \
          u2   = u + opt->copy_start[i]*bs;                                                                \
          ierr = PetscArraycpy(p,u2,l);CHKERRQ(ierr);                                                      \
          p   += l;                                                                                        \
        }                                                                                                  \
      } else { /* idx[] is strided */                                                                      \
        u   += opt->stride_first[r]*bs;                                                                    \
        step = opt->stride_step[r];                                                                        \
        for (i=0; i<opt->stride_n[r]; i++)                                                                 \
          for (j=0; j<bs; j++)                                                                             \
            p[i*bs+j] = u[i*step*bs+j];                                                                    \
      }                                                                                                    \
    }                                                                                                      \
    PetscFunctionReturn(0);                                                                                \
  }

/* DEF_Action - macro defining a Unpack(Fetch)AndInsert routine

   Arguments:
  +action     Unpack or Fetch
  .type       Type of the data
  .BS         Block size for vectorization
  .FILTER     Macro defining what to do with a statement, either EXECUTE or IGNORE
  .ctype      Type with or without the const qualifier, i.e., const type or type
  .cvoid      void with or without the const qualifier, i.e., const void or void

  Notes:
   This macro is not combined with DEF_ActionAndOp because we want to use memcpy in this macro.
   The two arguments ctype and cvoid are used (instead of one constness argument), because we want to
   get rid of compilation warning "empty macro arguments are undefined in ISO C90". With one constness argument,
   sometimes we input 'const', sometimes we have to input empty.
 */
#define DEF_Action(action,type,BS,FILTER,ctype,cvoid)               \
  static PetscErrorCode CPPJoin3_(action##AndInsert_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,PetscInt r,PetscSFPackOpt opt,void *unpacked,cvoid *packed) { \
    PetscErrorCode ierr;                                                                                   \
    type           *u = (type*)unpacked,*u2;                                                               \
    ctype          *p = (ctype*)packed;                                                                    \
    PetscInt       i,j,k,l,step;                                                                           \
    PetscFunctionBegin;                                                                                    \
    if (!idx) {  /* idx[] is contiguous */                                                                 \
      FILTER(type *v);                                                                                     \
      FILTER(ierr = PetscMalloc1(bs*n,&v);CHKERRQ(ierr));                                                  \
      FILTER(ierr = PetscArraycpy(v,u,bs*n);CHKERRQ(ierr));                                                \
             ierr = PetscArraycpy(u,p,bs*n);CHKERRQ(ierr);                                                 \
      FILTER(ierr = PetscArraycpy(p,v,bs*n);CHKERRQ(ierr));                                                \
      FILTER(ierr = PetscFree(v);CHKERRQ(ierr));                                                           \
    } else if (!opt || !opt->optimized[r]) { /* idx[] is not optimized*/                                   \
      for (i=0; i<n; i++) {                                                                                \
        for (j=0; j<bs; j+=BS) {                                                                           \
          for (k=j; k<j+BS; k++) {                                                                         \
            FILTER(type t = u[idx[i]*bs+k]);                                                               \
            u[idx[i]*bs+k] = p[i*bs+k];                                                                    \
            FILTER(p[i*bs+k] = t);                                                                         \
          }                                                                                                \
        }                                                                                                  \
      }                                                                                                    \
    } else { /* idx[] is optimized*/                                                                       \
      if (opt->copy_offset[r] != opt->copy_offset[r+1]) { /* idx[] is piece-wise contiguous */             \
        FILTER(type *v);                                                                                   \
        FILTER(ierr = PetscMalloc1(bs*n,&v);CHKERRQ(ierr)); /* maximal buffer  */                          \
        for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) { /* i-th piece */                       \
          l  = opt->copy_length[i]*bs; /* length in types */                                               \
          u2 = u + opt->copy_start[i]*bs;                                                                  \
          FILTER(ierr = PetscArraycpy(v,u2,l);CHKERRQ(ierr));                                              \
                 ierr = PetscArraycpy(u2,p,l);CHKERRQ(ierr);                                               \
          FILTER(ierr = PetscArraycpy(p,v,l);CHKERRQ(ierr));                                               \
          p += l;                                                                                          \
        }                                                                                                  \
        FILTER(ierr = PetscFree(v);CHKERRQ(ierr));                                                         \
      } else { /* idx[] is strided */                                                                      \
        u   += opt->stride_first[r]*bs;                                                                    \
        step = opt->stride_step[r];                                                                        \
        for (i=0; i<opt->stride_n[r]; i++)                                                                 \
          for (j=0; j<bs; j++) {                                                                           \
            FILTER(type t = u[i*step*bs+j]);                                                               \
            u[i*step*bs+j] = p[i*bs+j];                                                                    \
            FILTER(p[i*bs+j] = t);                                                                         \
          }                                                                                                \
      }                                                                                                    \
    }                                                                                                      \
    PetscFunctionReturn(0);                                                                                \
  }

/* DEF_ActionAndOp - macro defining a Unpack(Fetch)AndOp routine. Op can not be Insert, Maxloc or Minloc

   Arguments:
  +action     Unpack or Fetch
  .opname     Name of the Op, such as Add, Mult, LAND, etc.
  .type       Type of the data
  .BS         Block size for vectorization
  .op         Operator for the op, such as +, *, &&, ||, PetscMax, PetscMin, etc.
  .APPLY      Macro defining application of the op. Could be BINARY_OP, FUNCTION_OP, LXOR_OP or PAIRTYPE_OP
  .FILTER     Macro defining what to do with a statement, either EXECUTE or IGNORE
  .ctype      Type with or without the const qualifier, i.e., const type or type
  -cvoid      void with or without the const qualifier, i.e., const void or void
 */
#define DEF_ActionAndOp(action,opname,type,BS,op,APPLY,FILTER,ctype,cvoid) \
  static PetscErrorCode CPPJoin3_(action##And##opname##_,type,BS)(PetscInt n,PetscInt bs,const PetscInt *idx,PetscInt r,PetscSFPackOpt opt,void *unpacked,cvoid *packed) { \
    type     *u = (type*)unpacked,*u2,t;                                                                   \
    ctype    *p = (ctype*)packed;                                                                          \
    PetscInt i,j,k,l,step;                                                                                 \
    PetscFunctionBegin;                                                                                    \
    if (!idx) {  /* idx[] is contiguous */                                                                 \
      for (i=0; i<n*bs; i++) {                                                                             \
        t = u[i];                                                                                          \
        APPLY(u[i],t,op,p[i]);                                                                             \
        FILTER(p[i] = t);                                                                                  \
      }                                                                                                    \
    } else if (!opt || !opt->optimized[r]) { /* idx[] is not optimized*/                                   \
      for (i=0; i<n; i++) {                                                                                \
        for (j=0; j<bs; j+=BS) {                                                                           \
          for (k=j; k<j+BS; k++) {                                                                         \
            t = u[idx[i]*bs+k];                                                                            \
            APPLY(u[idx[i]*bs+k],t,op,p[i*bs+k]);                                                          \
            FILTER(p[i*bs+k] = t);                                                                         \
          }                                                                                                \
        }                                                                                                  \
      }                                                                                                    \
    } else { /* idx[] is optimized*/                                                                       \
      if (opt->copy_offset[r] != opt->copy_offset[r+1]) { /* idx[] is piece-wise contiguous */             \
        for (i=opt->copy_offset[r]; i<opt->copy_offset[r+1]; i++) { /* i-th piece */                       \
          l  = opt->copy_length[i]*bs; /* length in types */                                               \
          u2 = u + opt->copy_start[i]*bs;                                                                  \
          for (j=0; j<l; j++) {                                                                            \
            t = u2[j];                                                                                     \
            APPLY(u2[j],t,op,p[j]);                                                                        \
            FILTER(p[j] = t);                                                                              \
          }                                                                                                \
          p += l;                                                                                          \
        }                                                                                                  \
      } else { /* idx[] is strided */                                                                      \
        u   += opt->stride_first[r]*bs;                                                                    \
        step = opt->stride_step[r];                                                                        \
        for (i=0; i<opt->stride_n[r]; i++)                                                                 \
          for (j=0; j<bs; j++) {                                                                           \
            t = u[i*step*bs+j];                                                                            \
            APPLY(u[i*step*bs+j],t,op,p[i*bs+j]);                                                          \
            FILTER(p[i*bs+j] = t);                                                                         \
          }                                                                                                \
      }                                                                                                    \
    }                                                                                                      \
    PetscFunctionReturn(0);                                                                                \
  }

/* DEF_ActionAndXloc - macro defining a Unpack(Fetch)AndMaxloc(Minloc) routine

   Arguments:
  +Action     Unpack or Fetch
  .locname    Max or Min
  .type1      Type of the first data in a pair type
  .type2      Type of the second data in a pair type, usually PetscMPIInt for MPI ranks.
  .op         > or <
  .FILTER     Macro defining what to do with a statement, either EXECUTE or IGNORE
  .ctype      Type with or without the const qualifier, i.e., const PairType(type1,type2) or PairType(type1,type2)
  -cvoid      void with or without the const qualifier, i.e., const void or void
 */
#define DEF_ActionAndXloc(action,locname,type1,type2,op,FILTER,ctype,cvoid) \
  static PetscErrorCode CPPJoin3_(action##And##locname##loc_,PairType(type1,type2),1)(PetscInt n,PetscInt bs,const PetscInt *idx,PetscInt r,PetscSFPackOpt opt,void *unpacked,cvoid *packed) { \
    PairType(type1,type2) *u = (PairType(type1,type2)*)unpacked;                                           \
    ctype                 *p = (ctype*)packed;                                                             \
    PetscInt              i;                                                                               \
    for (i=0; i<n; i++) {                                                                                  \
      PetscInt j = idx[i];                                                                                 \
      FILTER(PairType(type1,type2) v = u[j]);                                                              \
      if (p[i].a op u[j].a) {                                                                              \
        u[j] = p[i];                                                                                       \
      } else if (p[i].a == u[j].a) {                                                                       \
        u[j].b = PetscMin(u[j].b,p[i].b); /* Minimal rank. Ref MPI MAXLOC */                               \
      }                                                                                                    \
      FILTER(p[i] = v);                                                                                    \
    }                                                                                                      \
    PetscFunctionReturn(0);                                                                                \
  }


/* Pack/unpack/fetch ops for all types */
#define DEF_PackNoInit(type,BS)                                                         \
  DEF_PackFunc(type,BS)                                                                 \
  DEF_Action(Unpack,type,BS,IGNORE,const type,const void)                               \
  DEF_Action(Fetch, type,BS,EXECUTE,type,void)                                          \


/* Extra addition ops for types supporting them */
#define DEF_PackAddNoInit(type,BS)                                                      \
  DEF_PackNoInit(type,BS)                                                               \
  DEF_ActionAndOp(Unpack,Add, type,BS,+,BINARY_OP,IGNORE,const type,const void)         \
  DEF_ActionAndOp(Unpack,Mult,type,BS,*,BINARY_OP,IGNORE,const type,const void)         \
  DEF_ActionAndOp(Fetch, Add, type,BS,+,BINARY_OP,EXECUTE,type,void)                    \
  DEF_ActionAndOp(Fetch, Mult,type,BS,*,BINARY_OP,EXECUTE,type,void)

/* Basic types */
#define DEF_Pack(type,BS)                                                               \
  DEF_PackAddNoInit(type,BS)                                                            \
  static void CPPJoin3_(PackInit_,type,BS)(PetscSFPack link) {                          \
    link->Pack            = CPPJoin3_(Pack_,           type,BS);                        \
    link->UnpackAndInsert = CPPJoin3_(UnpackAndInsert_,type,BS);                        \
    link->UnpackAndAdd    = CPPJoin3_(UnpackAndAdd_,   type,BS);                        \
    link->UnpackAndMult   = CPPJoin3_(UnpackAndMult_,  type,BS);                        \
    link->FetchAndInsert  = CPPJoin3_(FetchAndInsert_, type,BS);                        \
    link->FetchAndAdd     = CPPJoin3_(FetchAndAdd_,    type,BS);                        \
    link->FetchAndMult    = CPPJoin3_(FetchAndMult_,   type,BS);                        \
    link->unitbytes       = sizeof(type);                                               \
  }

/* Comparable types */
#define DEF_PackCmp(type)                                                               \
  DEF_PackAddNoInit(type,1)                                                             \
  DEF_ActionAndOp(Unpack,Max,type,1,PetscMax,FUNCTION_OP,IGNORE,const type,const void)  \
  DEF_ActionAndOp(Unpack,Min,type,1,PetscMin,FUNCTION_OP,IGNORE,const type,const void)  \
  DEF_ActionAndOp(Fetch, Max,type,1,PetscMax,FUNCTION_OP,EXECUTE,type,void)             \
  DEF_ActionAndOp(Fetch, Min,type,1,PetscMin,FUNCTION_OP,EXECUTE,type,void)             \
  static void CPPJoin2(PackInit_,type)(PetscSFPack link) {                              \
    link->Pack            = CPPJoin3_(Pack_,           type,1);                         \
    link->UnpackAndInsert = CPPJoin3_(UnpackAndInsert_,type,1);                         \
    link->UnpackAndAdd    = CPPJoin3_(UnpackAndAdd_,   type,1);                         \
    link->UnpackAndMult   = CPPJoin3_(UnpackAndMult_,  type,1);                         \
    link->UnpackAndMax    = CPPJoin3_(UnpackAndMax_,   type,1);                         \
    link->UnpackAndMin    = CPPJoin3_(UnpackAndMin_,   type,1);                         \
    link->FetchAndInsert  = CPPJoin3_(FetchAndInsert_, type,1);                         \
    link->FetchAndAdd     = CPPJoin3_(FetchAndAdd_ ,   type,1);                         \
    link->FetchAndMult    = CPPJoin3_(FetchAndMult_,   type,1);                         \
    link->FetchAndMax     = CPPJoin3_(FetchAndMax_ ,   type,1);                         \
    link->FetchAndMin     = CPPJoin3_(FetchAndMin_ ,   type,1);                         \
    link->unitbytes       = sizeof(type);                                               \
  }

/* Logical Types */
/* The operator in LXOR_OP should be empty but is &. It is not used. Put here to avoid
   the compilation warning "empty macro arguments are undefined in ISO C90"
 */
#define DEF_PackLog(type)                                                               \
  DEF_ActionAndOp(Unpack,LAND,type,1,&&,BINARY_OP,IGNORE,const type,const void)         \
  DEF_ActionAndOp(Unpack,LOR, type,1,||,BINARY_OP,IGNORE,const type,const void)         \
  DEF_ActionAndOp(Unpack,LXOR,type,1,&, LXOR_OP,  IGNORE,const type,const void)         \
  DEF_ActionAndOp(Fetch, LAND,type,1,&&,BINARY_OP,EXECUTE,type,void)                    \
  DEF_ActionAndOp(Fetch, LOR, type,1,||,BINARY_OP,EXECUTE,type,void)                    \
  DEF_ActionAndOp(Fetch, LXOR,type,1,&, LXOR_OP,  EXECUTE,type,void)                    \
  static void CPPJoin2(PackInit_Logical_,type)(PetscSFPack link) {                      \
    link->UnpackAndLAND   = CPPJoin3_(UnpackAndLAND_,type,1);                           \
    link->UnpackAndLOR    = CPPJoin3_(UnpackAndLOR_, type,1);                           \
    link->UnpackAndLXOR   = CPPJoin3_(UnpackAndLXOR_,type,1);                           \
    link->FetchAndLAND    = CPPJoin3_(FetchAndLAND_, type,1);                           \
    link->FetchAndLOR     = CPPJoin3_(FetchAndLOR_,  type,1);                           \
    link->FetchAndLXOR    = CPPJoin3_(FetchAndLXOR_, type,1);                           \
  }


/* Bitwise Types */
#define DEF_PackBit(type)                                                               \
  DEF_ActionAndOp(Unpack,BAND,type,1,&,BINARY_OP,IGNORE,const type,const void)          \
  DEF_ActionAndOp(Unpack,BOR, type,1,|,BINARY_OP,IGNORE,const type,const void)          \
  DEF_ActionAndOp(Unpack,BXOR,type,1,^,BINARY_OP,IGNORE,const type,const void)          \
  DEF_ActionAndOp(Fetch, BAND,type,1,&,BINARY_OP,EXECUTE,type,void)                     \
  DEF_ActionAndOp(Fetch, BOR, type,1,|,BINARY_OP,EXECUTE,type,void)                     \
  DEF_ActionAndOp(Fetch, BXOR,type,1,^,BINARY_OP,EXECUTE,type,void)                     \
  static void CPPJoin2(PackInit_Bitwise_,type)(PetscSFPack link) {                      \
    link->UnpackAndBAND   = CPPJoin3_(UnpackAndBAND_,type,1);                           \
    link->UnpackAndBOR    = CPPJoin3_(UnpackAndBOR_, type,1);                           \
    link->UnpackAndBXOR   = CPPJoin3_(UnpackAndBXOR_,type,1);                           \
    link->FetchAndBAND    = CPPJoin3_(FetchAndBAND_, type,1);                           \
    link->FetchAndBOR     = CPPJoin3_(FetchAndBOR_,  type,1);                           \
    link->FetchAndBXOR    = CPPJoin3_(FetchAndBXOR_, type,1);                           \
  }


/* Pair types */
#define DEF_PackPair(type1,type2)                                                                                   \
  typedef struct {type1 a; type2 b;} PairType(type1,type2);                                                         \
  DEF_PackFunc(PairType(type1,type2),1)                                                                             \
  DEF_Action(Unpack,PairType(type1,type2),1,IGNORE,const PairType(type1,type2),const void)                          \
  DEF_Action(Fetch, PairType(type1,type2),1,EXECUTE,PairType(type1,type2),void)                                     \
  DEF_ActionAndOp(Unpack,Add,PairType(type1,type2),1,+,PAIRTYPE_OP,IGNORE,const PairType(type1,type2),const void)   \
  DEF_ActionAndOp(Fetch, Add,PairType(type1,type2),1,+,PAIRTYPE_OP,EXECUTE,PairType(type1,type2),void)              \
  DEF_ActionAndXloc(Unpack,Max,type1,type2,>,IGNORE,const PairType(type1,type2),const void)                         \
  DEF_ActionAndXloc(Unpack,Min,type1,type2,<,IGNORE,const PairType(type1,type2),const void)                         \
  DEF_ActionAndXloc(Fetch, Max,type1,type2,>,EXECUTE,PairType(type1,type2),void)                                    \
  DEF_ActionAndXloc(Fetch, Min,type1,type2,<,EXECUTE,PairType(type1,type2),void)                                    \
  static void CPPJoin3_(PackInit_,type1,type2)(PetscSFPack link) {                                                  \
    link->Pack            = CPPJoin3_(Pack_,           PairType(type1,type2),1);                                    \
    link->UnpackAndInsert = CPPJoin3_(UnpackAndInsert_,PairType(type1,type2),1);                                    \
    link->UnpackAndAdd    = CPPJoin3_(UnpackAndAdd_,   PairType(type1,type2),1);                                    \
    link->UnpackAndMaxloc = CPPJoin3_(UnpackAndMaxloc_,PairType(type1,type2),1);                                    \
    link->UnpackAndMinloc = CPPJoin3_(UnpackAndMinloc_,PairType(type1,type2),1);                                    \
    link->FetchAndInsert  = CPPJoin3_(FetchAndInsert_, PairType(type1,type2),1);                                    \
    link->FetchAndAdd     = CPPJoin3_(FetchAndAdd_,    PairType(type1,type2),1);                                    \
    link->FetchAndMaxloc  = CPPJoin3_(FetchAndMaxloc_, PairType(type1,type2),1);                                    \
    link->FetchAndMinloc  = CPPJoin3_(FetchAndMinloc_, PairType(type1,type2),1);                                    \
    link->unitbytes       = sizeof(PairType(type1,type2));                                                          \
  }


/* Currently only dumb blocks of data */
#define DEF_Block(type,count)                                                           \
  typedef struct {type v[count];} BlockType(type,count);                                \
  DEF_PackNoInit(BlockType(type,count),1)                                               \
  static void CPPJoin3_(PackInit_block_,type,count)(PetscSFPack link) {                 \
    link->Pack            = CPPJoin3_(Pack_,           BlockType(type,count),1);        \
    link->UnpackAndInsert = CPPJoin3_(UnpackAndInsert_,BlockType(type,count),1);        \
    link->FetchAndInsert  = CPPJoin3_(FetchAndInsert_, BlockType(type,count),1);        \
    link->unitbytes       = sizeof(BlockType(type,count));                              \
  }

/* The typedef is used to get a typename without space that CPPJoin can handle */
typedef signed char SignedChar;
typedef unsigned char UnsignedChar;

DEF_PackCmp(SignedChar)
DEF_PackBit(SignedChar)
DEF_PackLog(SignedChar)
DEF_PackCmp(UnsignedChar)
DEF_PackBit(UnsignedChar)
DEF_PackLog(UnsignedChar)
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
DEF_Block(int,4)
DEF_Block(int,8)
DEF_Block(char,1)
DEF_Block(char,2)
DEF_Block(char,4)

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

PetscErrorCode PetscSFPackSetupType(PetscSFPack link,MPI_Datatype unit)
{
  PetscErrorCode ierr;
  PetscBool      isInt,isPetscInt,isPetscReal,is2Int,is2PetscInt,isSignedChar,isUnsignedChar;
  PetscInt       nPetscIntContig,nPetscRealContig;
  PetscMPIInt    ni,na,nd,combiner;
#if defined(PETSC_HAVE_COMPLEX)
  PetscBool      isPetscComplex;
  PetscInt       nPetscComplexContig;
#endif

  PetscFunctionBegin;
  ierr = MPIPetsc_Type_compare(unit,MPI_SIGNED_CHAR,&isSignedChar);CHKERRQ(ierr);
  ierr = MPIPetsc_Type_compare(unit,MPI_UNSIGNED_CHAR,&isUnsignedChar);CHKERRQ(ierr);
  /* MPI_CHAR is treated below as a dumb block type that does not support reduction according to MPI standard */
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
  ierr = MPI_Type_get_envelope(unit,&ni,&na,&nd,&combiner);CHKERRQ(ierr);
  link->isbuiltin = (combiner == MPI_COMBINER_NAMED) ? PETSC_TRUE : PETSC_FALSE;
  link->bs = 1;

  if (isSignedChar) {PackInit_SignedChar(link); PackInit_Logical_SignedChar(link); PackInit_Bitwise_SignedChar(link); link->basicunit = MPI_SIGNED_CHAR;}
  else if (isUnsignedChar) {PackInit_UnsignedChar(link); PackInit_Logical_UnsignedChar(link); PackInit_Bitwise_UnsignedChar(link); link->basicunit = MPI_UNSIGNED_CHAR;}
  else if (isInt) {PackInit_int(link); PackInit_Logical_int(link); PackInit_Bitwise_int(link); link->basicunit = MPI_INT;}
  else if (isPetscInt) {PackInit_PetscInt(link); PackInit_Logical_PetscInt(link); PackInit_Bitwise_PetscInt(link); link->basicunit = MPIU_INT;}
  else if (isPetscReal) {PackInit_PetscReal(link); PackInit_Logical_PetscReal(link); link->basicunit = MPIU_REAL;}
#if defined(PETSC_HAVE_COMPLEX)
  else if (isPetscComplex) {PackInit_PetscComplex_1(link); link->basicunit = MPIU_COMPLEX;}
#endif
  else if (is2Int) {PackInit_int_int(link); link->basicunit = MPI_2INT;}
  else if (is2PetscInt) {PackInit_PetscInt_PetscInt(link); link->basicunit = MPIU_2INT;}
  else if (nPetscIntContig) {
    if (nPetscIntContig%7 == 0) PackInit_PetscInt_7(link);
    else if (nPetscIntContig%5 == 0) PackInit_PetscInt_5(link);
    else if (nPetscIntContig%4 == 0) PackInit_PetscInt_4(link);
    else if (nPetscIntContig%3 == 0) PackInit_PetscInt_3(link);
    else if (nPetscIntContig%2 == 0) PackInit_PetscInt_2(link);
    else PackInit_PetscInt(link);
    link->bs = nPetscIntContig;
    link->unitbytes *= nPetscIntContig;
    link->basicunit = MPIU_INT;
  } else if (nPetscRealContig) {
    if (nPetscRealContig%7 == 0) PackInit_PetscReal_7(link);
    else if (nPetscRealContig%5 == 0) PackInit_PetscReal_5(link);
    else if (nPetscRealContig%4 == 0) PackInit_PetscReal_4(link);
    else if (nPetscRealContig%3 == 0) PackInit_PetscReal_3(link);
    else if (nPetscRealContig%2 == 0) PackInit_PetscReal_2(link);
    else PackInit_PetscReal(link);
    link->bs = nPetscRealContig;
    link->unitbytes *= nPetscRealContig;
    link->basicunit = MPIU_REAL;
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
    link->basicunit = MPIU_COMPLEX;
#endif
  } else {
    MPI_Aint lb,bytes;
    ierr = MPI_Type_get_extent(unit,&lb,&bytes);CHKERRQ(ierr);
    if (lb != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
    if (bytes % sizeof(int)) { /* If the type size is not multiple of int */
      if      (bytes%4 == 0) {PackInit_block_char_4(link); link->bs = bytes/4;} /* Note the basic type is char[4] */
      else if (bytes%2 == 0) {PackInit_block_char_2(link); link->bs = bytes/2;}
      else                   {PackInit_block_char_1(link); link->bs = bytes/1;}
      link->unitbytes = bytes;
      link->basicunit = MPI_CHAR;
    } else {
      PetscInt nInt = bytes / sizeof(int);
      if      (nInt%8 == 0)  {PackInit_block_int_8(link);  link->bs = nInt/8;} /* Note the basic type is int[8] */
      else if (nInt%4 == 0)  {PackInit_block_int_4(link);  link->bs = nInt/4;}
      else if (nInt%2 == 0)  {PackInit_block_int_2(link);  link->bs = nInt/2;}
      else                   {PackInit_block_int_1(link);  link->bs = nInt/1;}
      link->unitbytes = bytes;
      link->basicunit = MPI_INT;
    }
  }
  if (link->isbuiltin) link->unit = unit; /* builtin datatypes are common. Make it fast */
  else {ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFPackGetUnpackAndOp(PetscSF sf,PetscSFPack link,MPI_Op op,PetscErrorCode (**UnpackAndOp)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*))
{
  PetscFunctionBegin;
  *UnpackAndOp = NULL;
  if (op == MPIU_REPLACE) *UnpackAndOp = link->UnpackAndInsert;
  else if (op == MPI_SUM || op == MPIU_SUM) *UnpackAndOp = link->UnpackAndAdd;
  else if (op == MPI_PROD) *UnpackAndOp = link->UnpackAndMult;
  else if (op == MPI_MAX || op == MPIU_MAX) *UnpackAndOp = link->UnpackAndMax;
  else if (op == MPI_MIN || op == MPIU_MIN) *UnpackAndOp = link->UnpackAndMin;
  else if (op == MPI_LAND)   *UnpackAndOp = link->UnpackAndLAND;
  else if (op == MPI_BAND)   *UnpackAndOp = link->UnpackAndBAND;
  else if (op == MPI_LOR)    *UnpackAndOp = link->UnpackAndLOR;
  else if (op == MPI_BOR)    *UnpackAndOp = link->UnpackAndBOR;
  else if (op == MPI_LXOR)   *UnpackAndOp = link->UnpackAndLXOR;
  else if (op == MPI_BXOR)   *UnpackAndOp = link->UnpackAndBXOR;
  else if (op == MPI_MAXLOC) *UnpackAndOp = link->UnpackAndMaxloc;
  else if (op == MPI_MINLOC) *UnpackAndOp = link->UnpackAndMinloc;
  else *UnpackAndOp = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFPackGetFetchAndOp(PetscSF sf,PetscSFPack link,MPI_Op op,PetscErrorCode (**FetchAndOp)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*))
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

/*
  Setup pack/unpack optimization plans based on indice patterns available

   Input Parameters:
  +  n       - number of target processors
  .  offset  - [n+1] for the i-th processor, its associated indices are idx[offset[i], offset[i+1])
  -  idx     - [] array storing indices. Its length is offset[n+1]

   Output Parameters:
  +  opt    - the optimization
*/
PetscErrorCode PetscSFPackSetupOptimization(PetscInt n,const PetscInt *offset,const PetscInt *idx,PetscSFPackOpt *out)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,n_copies,tot_copies=0,step;
  PetscBool      strided,has_strided=PETSC_FALSE,has_optimized=PETSC_FALSE;
  PetscSFPackOpt opt;

  PetscFunctionBegin;
  ierr = PetscCalloc1(1,&opt);CHKERRQ(ierr);
  ierr = PetscCalloc2(n,&opt->optimized,n+1,&opt->copy_offset);CHKERRQ(ierr);

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
      opt->optimized[i] = PETSC_TRUE;
      has_optimized     = PETSC_TRUE;
      tot_copies       += n_copies;
    }
  }

  /* Setup memcpy plan for each contiguous piece */
  k    = 0; /* k-th copy */
  ierr = PetscMalloc2(tot_copies,&opt->copy_start,tot_copies,&opt->copy_length);CHKERRQ(ierr);
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    if (opt->optimized[i]) {
      n_copies           = 1;
      opt->copy_start[k] = idx[offset[i]];
      for (j=offset[i]; j<offset[i+1]-1; j++) {
        if (idx[j]+1 != idx[j+1]) { /* meet end of a copy (and next copy must exist) */
          n_copies++;
          opt->copy_start[k+1] = idx[j+1];
          opt->copy_length[k]  = idx[j] - opt->copy_start[k] + 1;
          k++;
        }
      }
      /* Set copy length of the last copy for this target */
      opt->copy_length[k] = idx[j] - opt->copy_start[k] + 1;
      k++;
    }
    /* Set offset for next target. When optimized[i]=false, copy_offsets[i]=copy_offsets[i+1] */
    opt->copy_offset[i+1] = k;
  }

  /* Last chance! If the indices do not have long contiguous pieces, are they strided? */
  ierr = PetscMalloc3(n,&opt->stride_first,n,&opt->stride_step,n,&opt->stride_n);CHKERRQ(ierr);
  for (i=0; i<n; i++) { /* for each remote */
    if (!opt->optimized[i] && (offset[i+1] - offset[i]) >= 16) { /* few indices (<16) are not worth striding */
      strided = PETSC_TRUE;
      step    = idx[offset[i]+1] - idx[offset[i]];
      for (j=offset[i]; j<offset[i+1]-1; j++) {
        if (idx[j]+step != idx[j+1]) { strided = PETSC_FALSE; break; }
      }
      if (strided) {
        opt->optimized[i]    = PETSC_TRUE;
        opt->stride_first[i] = idx[offset[i]];
        opt->stride_step[i]  = step;
        opt->stride_n[i]     = offset[i+1] - offset[i];
        has_strided          = PETSC_TRUE;
        has_optimized        = PETSC_TRUE;
      }
    }
  }
  /* If no target has been stride-optimized or optimized, free related arrays to save memory */
  if (!has_strided) {ierr = PetscFree3(opt->stride_first,opt->stride_step,opt->stride_n);CHKERRQ(ierr);}
  if (!has_optimized) {
    ierr = PetscFree2(opt->optimized,opt->copy_offset);CHKERRQ(ierr);
    ierr = PetscFree2(opt->copy_start,opt->copy_length);CHKERRQ(ierr);
    ierr = PetscFree(opt);CHKERRQ(ierr);
    *out = NULL;
  } else *out = opt;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFPackDestoryOptimization(PetscSFPackOpt *out)
{
  PetscErrorCode ierr;
  PetscSFPackOpt opt = *out;

  PetscFunctionBegin;
  if (opt) {
    ierr = PetscFree2(opt->optimized,opt->copy_offset);CHKERRQ(ierr);
    ierr = PetscFree2(opt->copy_start,opt->copy_length);CHKERRQ(ierr);
    ierr = PetscFree3(opt->stride_first,opt->stride_step,opt->stride_n);CHKERRQ(ierr);
    ierr = PetscFree(opt);CHKERRQ(ierr);
    *out = NULL;
  }
  PetscFunctionReturn(0);
}
