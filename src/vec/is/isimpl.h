
/*
    Index sets for scatter-gather type operations in vectors
and matrices. 

   Eventually ther may be operations like union, difference etc.
for now we define only the shell and what we absolutely need.
*/

#if !defined(_INDEX)
#define _INDEX
#include "ptscimpl.h"
#include "is.h"

#define IS_COOKIE 0x13131313

struct _ISOps {
  int  (*size)(),(*localsize)(),(*position)();
  int  (*indices)();
  int  (*restoreindices)();
};

struct _IS {
  PETSCHEADER
  struct _ISOps *ops;
  void          *data;
};


#define GENERALSEQUENTIAL 1

#endif
