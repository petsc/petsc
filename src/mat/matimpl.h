
#if !defined(__MATIMPL)
#define __MATIMPL
#include "ptscimpl.h"
#include "mat.h"

#define MAT_COOKIE 0x404040

struct _MatOps {
  int       (*insert)(), (*insertadd)(),
            (*view)(), (*getrow)(), (*restorerow)(),
            (*mult)(),(*multadd)(),(*multtrans)(),(*multtransadd)(),
            (*solve)(),(*solveadd)(),(*solvetrans)(),(*solvetransadd)(),
            (*lufactor)(),(*chfactor)(),
            (*relax)(),(*relaxforward)(),(*relaxback)(),
            (*trans)();
  int       (*NZ)(),(*memory)(),(*equal)();
  int       (*copy)();
  int       (*getdiag)(),(*scale)(),(*norm)(),
            (*bassembly)(),(*eassembly)();
};

struct _Mat {
  PETSCHEADER
  struct _MatOps *ops;
  void           *data;
};

#endif


