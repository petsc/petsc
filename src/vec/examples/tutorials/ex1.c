#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.29 1996/01/12 22:05:06 bsmith Exp curfman $";
#endif

static char help[] = "Tests various vector routines\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int      n = 20, ierr,flg;
  Scalar   one = 1.0, two = 2.0, three = 3.0, dots[3],dot;
  double   norm,v;
  Vec      x,y,w,*z;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  /* create a vector */
  ierr = VecCreate(MPI_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);
  ierr = VecDuplicate(x,&w); CHKERRA(ierr);
  ierr = VecDuplicateVecs(x,3,&z); CHKERRA(ierr); 
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);
  ierr = VecSet(&one,z[0]); CHKERRA(ierr);
  ierr = VecSet(&two,z[1]); CHKERRA(ierr);
  ierr = VecSet(&three,z[2]); CHKERRA(ierr);

  /* Test whether vector has been corrupted (just to demonstrate this
     routine) not needed in most application codes. */
  ierr = VecValidVector(x,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,"Corrupted vector.");
  
  ierr = VecDot(x,x,&dot); CHKERRA(ierr);
  ierr = VecMDot(3,x,z,dots); CHKERRA(ierr);
#if defined(PETSC_COMPLEX)
  MPIU_printf(MPI_COMM_WORLD,"Vector length %d\n", int (real(dot)));
  MPIU_printf(MPI_COMM_WORLD,"Vector length %d %d %d\n",(int)real(dots[0]),
                             (int)real(dots[1]),(int)real(dots[2]));
#else
  MPIU_printf(MPI_COMM_WORLD,"Vector length %d\n",(int) dot);
  MPIU_printf(MPI_COMM_WORLD,"Vector length %d %d %d\n",(int)dots[0],
                             (int)dots[1],(int)dots[2]);
#endif

  MPIU_printf(MPI_COMM_WORLD,"All other values should be near zero\n");
  ierr = VecScale(&two,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecScale %g\n",v);

  ierr = VecCopy(x,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr);
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecCopy  %g\n",v);

  ierr = VecAXPY(&three,x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  v = norm-8.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecAXPY %g\n",v);

  ierr = VecAYPX(&two,x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  v = norm-18.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecAXPY %g\n",v);

  ierr = VecSwap(x,y); CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm); CHKERRA(ierr);
  v = norm-2.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecSwap  %g\n",v);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  v = norm-18.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecSwap  %g\n",v);

  ierr = VecWAXPY(&two,x,y,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr);
  v = norm-38.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecWAXPY %g\n",v);

  ierr = VecPMult(y,x,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr); 
  v = norm-36.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecPMult %g\n",v);

  ierr = VecPDiv(x,y,w); CHKERRA(ierr);
  ierr = VecNorm(w,NORM_2,&norm); CHKERRA(ierr);
  v = norm-9.0*sqrt((double) n); if (v > -1.e-10 && v < 1.e-10) v = 0.0; 
  MPIU_printf(MPI_COMM_WORLD,"VecPDiv  %g\n",v);
  
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  ierr = VecDestroy(w); CHKERRA(ierr);
  ierr = VecDestroyVecs(z,3); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
