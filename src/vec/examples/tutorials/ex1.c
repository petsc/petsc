
/*
      Example demonstrating some features of the vectors directory.
*/
#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "cplxutil.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int worker(argc,argv)
int  argc;
char **argv;
{
  int      n = 20, ierr;
  Scalar   one, two, three, dots[3],dot;
  double   norm;
  Vec      x,y,w,*z;
  FILE     *fd = stdout;
  char     filename[1024];
 
  SET(one,1.0,0.0); SET(two,2.0,0.0); SET(three,3.0,0.0);
  SYArgGetInt(&argc,argv,0,"-n",&n);
  if (SYArgGetString(&argc,argv,0,"-f",filename,1024)) {
    ierr = PISfopen(0,filename,"w",&fd); CHKERR(ierr);
  }

  /* create a default serial double precision vector */
  ierr = VecCreateInitialVector(n,argc,argv,&x); CHKERR(ierr);
  ierr = VecCreate(x,&y); CHKERR(ierr);
  ierr = VecCreate(x,&w); CHKERR(ierr);
  ierr = VecGetVecs(x,3,&z); CHKERR(ierr); 
  ierr = VecSet(&one,x);CHKERR(ierr);
  ierr = VecSet(&two,y);CHKERR(ierr);
  ierr = VecSet(&one,z[0]); CHKERR(ierr);
  ierr = VecSet(&two,z[1]); CHKERR(ierr);
  ierr = VecSet(&three,z[2]);CHKERR(ierr);
  
  ierr = VecDot(x,x,&dot); CHKERR(ierr);
  PISfprintf(0,fd,"Vector length %d\n",(int) REALPART(dot));
  ierr = VecMDot(3,x,z,dots); CHKERR(ierr);
  PISfprintf(0,fd,"Vector length %d %d %d\n",
                          (int)REALPART(dots[0]),
                          (int)REALPART(dots[1]),
                          (int)REALPART(dots[2]));

  PISfprintf(0,fd,"All other values should be near zero\n");
  ierr = VecScale(&two,x);CHKERR(ierr);
  ierr = VecNorm(x,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VeScale %g\n",norm-2.0*sqrt((double) n));

  ierr = VecCopy(x,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VeCopy  %g\n",norm-2.0*sqrt((double) n));

  ierr = VecAXPY(&three,x,y);CHKERR(ierr);
  ierr = VecNorm(y,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VecAXPY %g\n",norm-8.0*sqrt((double) n));

  ierr = VecAYPX(&two,x,y);CHKERR(ierr);
  ierr = VecNorm(y,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VecAXPY %g\n",norm-18.0*sqrt((double) n));

  ierr = VecSwap(x,y);CHKERR(ierr);
  ierr = VecNorm(y,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VecSwap  %g\n",norm-2.0*sqrt((double) n));
  ierr = VecNorm(x,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VecSwap  %g\n",norm-18.0*sqrt((double) n));

  ierr = VecWAXPY(&two,x,y,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VecWAXPY %g\n",norm-38.*sqrt((double) n));

  ierr = VecPMult(y,x,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm);CHKERR(ierr); 
  PISfprintf(0,fd,"VecPMult %g\n",norm-36.0*sqrt((double) n));

  ierr = VecPDiv(x,y,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm); CHKERR(ierr);
  PISfprintf(0,fd,"VecPDiv  %g\n",norm-9.0*sqrt((double) n));
  
  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);
  ierr = VecDestroy(w);CHKERR(ierr);
  ierr = VecFreeVecs(z,3);CHKERR(ierr);
  if (fd != stdout) PISfclose(0,fd);
  return 0;
}
 
