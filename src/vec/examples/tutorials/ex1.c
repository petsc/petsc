

/*
      Example demonstrating some features of the vectors directory.
*/
#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int worker(int argc,char **argv)
{
  int      n = 20, ierr;
  Scalar   one = 1.0, two = 2.0, three = 3.0, dots[3],dot;
  double   norm;
  Vec      x,y,w,*z;
  FILE     *fd = stdout;
 
  MPI_Init(&argc,&argv);
  SYArgGetInt(&argc,argv,0,"-n",&n);

  /* create a vector */
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
  ierr = VecMDot(3,x,z,dots); CHKERR(ierr);
#if defined(PETSC_COMPLEX)
  fprintf(fd,"Vector length %d\n",(int) real(dot));
  fprintf(fd,"Vector length %d %d %d\n",(int)real(dots[0]),(int)real(dots[1]),
                          (int)real(dots[2]));
#else
  fprintf(fd,"Vector length %d\n",(int) dot);
  fprintf(fd,"Vector length %d %d %d\n",(int)dots[0],(int)dots[1],
                          (int)dots[2]);
#endif

  fprintf(fd,"All other values should be near zero\n");
  ierr = VecScale(&two,x);CHKERR(ierr);
  ierr = VecNorm(x,&norm); CHKERR(ierr);
  fprintf(fd,"VecScale %g\n",norm-2.0*sqrt((double) n));

  ierr = VecCopy(x,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm); CHKERR(ierr);
  fprintf(fd,"VecCopy  %g\n",norm-2.0*sqrt((double) n));

  ierr = VecAXPY(&three,x,y);CHKERR(ierr);
  ierr = VecNorm(y,&norm); CHKERR(ierr);
  fprintf(fd,"VecAXPY %g\n",norm-8.0*sqrt((double) n));

  ierr = VecAYPX(&two,x,y);CHKERR(ierr);
  ierr = VecNorm(y,&norm); CHKERR(ierr);
  fprintf(fd,"VecAXPY %g\n",norm-18.0*sqrt((double) n));

  ierr = VecSwap(x,y);CHKERR(ierr);
  ierr = VecNorm(y,&norm); CHKERR(ierr);
  fprintf(fd,"VecSwap  %g\n",norm-2.0*sqrt((double) n));
  ierr = VecNorm(x,&norm); CHKERR(ierr);
  fprintf(fd,"VecSwap  %g\n",norm-18.0*sqrt((double) n));

  ierr = VecWAXPY(&two,x,y,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm); CHKERR(ierr);
  fprintf(fd,"VecWAXPY %g\n",norm-38.*sqrt((double) n));

  ierr = VecPMult(y,x,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm);CHKERR(ierr); 
  fprintf(fd,"VecPMult %g\n",norm-36.0*sqrt((double) n));

  ierr = VecPDiv(x,y,w);CHKERR(ierr);
  ierr = VecNorm(w,&norm); CHKERR(ierr);
  fprintf(fd,"VecPDiv  %g\n",norm-9.0*sqrt((double) n));
  
  ierr = VecDestroy(x);CHKERR(ierr);
  ierr = VecDestroy(y);CHKERR(ierr);
  ierr = VecDestroy(w);CHKERR(ierr);
  ierr = VecFreeVecs(z,3);CHKERR(ierr);

  MPI_Finalize();
  return 0;
}
 
