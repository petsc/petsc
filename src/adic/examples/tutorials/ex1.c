#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.1 1997/03/20 23:12:06 bsmith Exp bsmith $";
#endif

static char help[] = "Demonstrates using ADIC to compute a derivative.\n\n";

/*T
   Concepts: Vectors^Using basic vector routines;
   Routines: VecCreate(); VecDuplicate(); VecSet();
   Routines: VecScale(); VecNorm(); 
   Routines: VecDestroy(); 
   Processors: n
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines
     sys.h    - system routines
*/

#include "vec.h"
#include <math.h>

extern int Function(Vec,Scalar,double*);

int main(int argc,char **argv)
{
  Vec            x;               /* vectors */
  double         output;
  InactiveDouble grad[1];
  int            n = 20, ierr, flg;
  Scalar         input;

  PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  input = 2.0;

  AD_Init();
  AD_SetIndep(input);
  AD_SetIndepDone();


  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate(), the vector format (currently parallel
     or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
  */
  ierr = VecCreate(PETSC_COMM_WORLD,n,&x); CHKERRA(ierr);

  ierr = Function(x,input,&output); CHKERRA(ierr);
  
  
  AD_ExtractGrad(grad,output);

  printf("Function %g grad %g sqrt(n) %g \n",output,grad[0],sqrt((double)n));

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x); CHKERRA(ierr);
  AD_Final();
  PetscFinalize();
  return 0;
}
 
int Function(Vec x,Scalar input,double *output)
{
  int      ierr;
  Scalar   one = 1.0;

  ierr = VecSet(&one,x); CHKERRQ(ierr);
  ierr = VecScale(&input,x); CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_2,output); CHKERRQ(ierr);
  return 0;
}




