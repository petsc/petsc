/* 
   This should not be included in users code.
*/

#ifndef __VECIMPL 
#define __VECIMPL
#include "ptscimpl.h"
#include "is.h"
#include "vec.h"

#define VEC_COOKIE 0x101010

 
struct _VeOps {
  int  (*create_vector)();    /* Eeturns single vector */
  int  (*obtain_vectors)();  /* Returns array of vectors */
  int  (*release_vectors)();   /* Free array of vectors */
  int  (*dot)(),                   /* z = x^H * y */
       (*mdot)(),                  /*   z[j] = x dot y[j] */
       (*norm)(),                  /* z = sqrt(x^H * x) */
       (*max)(),                   /* z = max(|x|); idx = index of sup(|x|) */
       (*asum)(),                  /*  z = sum |x| */
       (*tdot)(),                  /* x'*y */
       (*mtdot)(),                 /*   z[j] = x dot y[j] */
       (*scal)(),                  /*  x = alpha * x   */
       (*copy)(),                  /*  y = x */
       (*set)(),                   /*  y = alpha  */
       (*swap)(),                  /* exchange x and y */
       (*axpy)(),                  /*  y = y + alpha * x */
       (*maxpy)(),                 /*   y    = y + alpha[j] x[j] */
       (*aypx)(),                  /*  y = x + alpha * y */
       (*waxpy)(),                 /*  w = y + alpha * x */
       (*pmult)(),                 /*  w = x .* y */
       (*pdiv)(),                  /*  w = x ./ y */
       (*scatteraddbegin)(),       /* y[ix[i]] += x[i] */
       (*scatteraddend)(),        
       (*scatterbegin)(),
       (*scatterend)(),
       (*addvalues)(),
       (*insertvalues)(),
       (*beginassm)(),
       (*endassm)(),
       (*getarray)(),              /* returns pointer to actual values*/
       (*view)();
  int  (*getsize)(),(*localsize)();
};

/* Vector types */
#define SEQVECTOR               1
#define SEQCOMPLEXVECTOR        3 

struct _Vec {
  PETSCHEADER
  struct _VeOps *ops;
  void          *data;
};

/* Default obtain and release vectors; can be used by any implementation */
int     Veiobtain_vectors();
int     Veirelease_vectors();
int     VeiDestroyVector();

#endif
