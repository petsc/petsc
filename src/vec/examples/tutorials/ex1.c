

static char help[] = "Tests various vector routines\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int      n = 20, ierr;
  Scalar   one = 1.0, two = 2.0, three = 3.0, dots[3],dot;
  double   norm;
  Vec      x,y,w,*z;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-n",&n);

  /* create a vector */
  ierr = VecCreateInitialVector(MPI_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecCreate(x,&y); CHKERRA(ierr);
  ierr = VecCreate(x,&w); CHKERRA(ierr);
  ierr = VecGetVecs(x,3,&z); CHKERRA(ierr); 
  ierr = VecSet(&one,x);CHKERRA(ierr);
  ierr = VecSet(&two,y);CHKERRA(ierr);
  ierr = VecSet(&one,z[0]); CHKERRA(ierr);
  ierr = VecSet(&two,z[1]); CHKERRA(ierr);
  ierr = VecSet(&three,z[2]);CHKERRA(ierr);
  
  ierr = VecDot(x,x,&dot); CHKERRA(ierr);
  ierr = VecMDot(3,x,z,dots); CHKERRA(ierr);
#if defined(PETSC_COMPLEX)
  MPE_printf(MPI_COMM_WORLD,"Vector length %d\n",(int) real(dot));
  MPE_printf(MPI_COMM_WORLD,"Vector length %d %d %d\n",(int)real(dots[0]),
                             (int)real(dots[1]),(int)real(dots[2]));
#else
  MPE_printf(MPI_COMM_WORLD,"Vector length %d\n",(int) dot);
  MPE_printf(MPI_COMM_WORLD,"Vector length %d %d %d\n",(int)dots[0],
                             (int)dots[1],(int)dots[2]);
#endif

  MPE_printf(MPI_COMM_WORLD,"All other values should be near zero\n");
  ierr = VecScale(&two,x);CHKERRA(ierr);
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecScale %g\n",norm-2.0*sqrt((double) n));

  ierr = VecCopy(x,w);CHKERRA(ierr);
  ierr = VecNorm(w,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecCopy  %g\n",norm-2.0*sqrt((double) n));

  ierr = VecAXPY(&three,x,y);CHKERRA(ierr);
  ierr = VecNorm(y,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecAXPY %g\n",norm-8.0*sqrt((double) n));

  ierr = VecAYPX(&two,x,y);CHKERRA(ierr);
  ierr = VecNorm(y,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecAXPY %g\n",norm-18.0*sqrt((double) n));

  ierr = VecSwap(x,y);CHKERRA(ierr);
  ierr = VecNorm(y,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecSwap  %g\n",norm-2.0*sqrt((double) n));
  ierr = VecNorm(x,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecSwap  %g\n",norm-18.0*sqrt((double) n));

  ierr = VecWAXPY(&two,x,y,w);CHKERRA(ierr);
  ierr = VecNorm(w,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecWAXPY %g\n",norm-38.*sqrt((double) n));

  ierr = VecPMult(y,x,w);CHKERRA(ierr);
  ierr = VecNorm(w,&norm);CHKERRA(ierr); 
  MPE_printf(MPI_COMM_WORLD,"VecPMult %g\n",norm-36.0*sqrt((double) n));

  ierr = VecPDiv(x,y,w);CHKERRA(ierr);
  ierr = VecNorm(w,&norm); CHKERRA(ierr);
  MPE_printf(MPI_COMM_WORLD,"VecPDiv  %g\n",norm-9.0*sqrt((double) n));
  
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = VecDestroy(w);CHKERRA(ierr);
  ierr = VecFreeVecs(z,3);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
