#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.1 1997/04/08 04:03:11 bsmith Exp bsmith $";
#endif

static char help[] = "Demonstrates using ADIC to compute a derivative.\n\n";

/*T
   Concepts: Vectors^Using basic vector routines;
   Processors: n
T*/

/* 
  Include "vec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines
     sys.h    - system routines
*/

#include "vec.h"
#include "vec.ad.h"
#include "petscadic.h"

extern int Function(Vec,Vec);
extern int ad_Function(ad_Vec,ad_Vec);

int main(int argc,char **argv)
{
  Vec               x,y,z,Az;               /* vectors */
  ad_Vec            ad_x,ad_y;
  int               n = 20, ierr, flg;
  Scalar            one = 1.0, onethird = 1.0/3.0;
  Mat               grad,igrad;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ad_PetscInitialize(&argc,&argv,(char*)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  
  ad_AD_Init();


  /*
     Evaluate the function itself 
  */
  ierr = VecCreate(PETSC_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&y); CHKERRA(ierr);

  ierr = VecSet(x,one); CHKERRA(ierr);

  ierr = Function(x,y); CHKERRA(ierr);
  VecView(y,0);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  /* ----------------------------------------------------------*/
  /*
       Evaluate the differentiated function
  */
  ierr = ad_VecCreate(PETSC_COMM_WORLD,n,&ad_x); CHKERRA(ierr);
  ierr = ad_VecSetFromOptions(ad_x); CHKERRA(ierr);
  ierr = ad_VecDuplicate(ad_x,&ad_y); CHKERRA(ierr);

  ierr = ad_Function(ad_x,ad_y); CHKERRA(ierr);
  ad_VecView(ad_y,0);
  
  ierr = ad_VecDestroy(ad_x); CHKERRA(ierr);
  ierr = ad_VecDestroy(ad_y); CHKERRA(ierr);

  ad_AD_Final();
  PetscFinalize();
  ad_PetscFinalize();
  return 0;
}
 



