#if !defined(__TAODEF_H)
#define __TAODEF_H

#include "petsc/finclude/petscts.h"

#define Tao PetscFortranAddr
#define TaoLineSearch PetscFortranAddr
#define TaoConvergedReason PetscEnum
#define TaoType character*(80)
#define TaoLineSearchType character*(80)

#define TAOLMVM     "lmvm"
#define TAONLS      "nls"
#define TAONTR      "ntr"
#define TAONTL      "ntl"
#define TAOCG       "cg"
#define TAOTRON     "tron"
#define TAOOWLQN    "owlqn"
#define TAOBMRM     "bmrm"
#define TAOBLMVM    "blmvm"
#define TAOBNCG     "bncg"
#define TAOBNLS     "bnls"
#define TAOBNTR     "bntr"
#define TAOBNTL     "bntl"
#define TAOBQNKLS   "bqnkls"
#define TAOBQNKTR   "bqnktr"
#define TAOBQNKTL   "bqnktl"
#define TAOQBNLS    "bqnls"
#define TAOBRGN     "brgn"
#define TAOBQPIP    "bqpip"
#define TAOGPCG     "gpcg"
#define TAONM       "nm"
#define TAOPOUNDERS "pounders"
#define TAOLCL      "lcl"
#define TAOSSILS    "ssils"
#define TAOSSFLS    "ssfls"
#define TAOASILS    "asils"
#define TAOASFLS    "asfls"
#define TAOIPM      "ipm"
#define TAOFDTEST   "test"

#endif
