#ifndef TAO_DIR

typedef PetscObject Tao;
static PetscClassId TAO_CLASSID = 0;
#define TaoInitializePackage() PetscClassIdRegister("TAO",&TAO_CLASSID)

#else

#include "tao.h"
#include "tao-private/taosolver_impl.h"

#define TAO_CLASSID TAOSOLVER_CLASSID
typedef TaoSolver Tao;
typedef TaoSolverType TaoType;
#define TAOLMVM     "lmvm"
#define TAONLS      "nls"
#define TAONTR      "ntr"
#define TAONTL      "ntl"
#define TAOCG       "cg"
#define TAOTRON     "tron"
#define TAOOWLQN    "owlqn"
#define TAOBMRM     "bmrm"
#define TAOBLMVM    "blmvm"
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
#define TAOTEST     "test"

typedef TaoSolverTerminationReason TaoConvergedReason;
#define TaoSetConvergedReason TaoSetTerminationReason
#define TaoGetConvergedReason TaoGetTerminationReason

#undef __FUNCT__
#define __FUNCT__ "TaoGetConstraintTolerances"
PetscErrorCode TaoGetConstraintTolerances(TaoSolver tao, PetscReal *catol, PetscReal *crtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  if (catol) *catol=tao->catol;
  if (crtol) *crtol=tao->crtol;
  PetscFunctionReturn(0);
}

#endif
