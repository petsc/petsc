#include <petscsnes.h>
#include <petscdmda.h>
#include <../src/snes/impls/vi/viimpl.h>
#include <math.h>  /* for cos() sin(0), and atan() */

static  char help[]=
"This example is an implementation of the journal bearing problem from TAO package\n\
(src/bound/examples/tutorials/jbearing.c). This example is based on \n\
the problem DPJB from the MINPACK-2 test suite.  This pressure journal \n\
bearing problem is an example of elliptic variational problem defined over \n\
a two dimensional rectangle.  By discretizing the domain into triangular \n\
elements, the pressure surrounding the journal bearing is defined as the \n\
minimum of a quadratic function whose variables are bounded below by zero.\n";

typedef struct {
  /* problem parameters */
  PetscReal      ecc;          /* test problem parameter */
  PetscReal      b;            /* A dimension of journal bearing */
  PetscInt       nx,ny;        /* discretization in x, y directions */
  DM             da;           /* distributed array data structure */
  Mat            A;            /* Quadratic Objective term */
  Vec            B;            /* Linear Objective term */
} AppCtx;

/* User-defined routines */
static PetscReal p(PetscReal xi, PetscReal ecc);
PetscErrorCode FormGradient(SNES, Vec, Vec,void *);
PetscErrorCode FormHessian(SNES,Vec,Mat *, Mat *, MatStructure *, void *);
PetscErrorCode ComputeB(AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode      info;               /* used to check for functions returning nonzeros */
  Vec                 x;                  /* variables vector */
  Vec                 xl,xu;              /* lower and upper bound on variables */
  PetscBool           flg;              /* A return variable when checking for user options */
  SNESConvergedReason reason;
  AppCtx              user;               /* user-defined work context */
  SNES                snes;
  Vec                 r;
  PetscReal         zero=0.0,thnd=1000;

  
  /* Initialize PETSC */
  PetscInitialize( &argc, &argv,(char *)0,help );

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example does not work for scalar type complex\n");
#endif

  /* Set the default values for the problem parameters */
  user.nx = 50; user.ny = 50; user.ecc = 0.1; user.b = 10.0;

  /* Check for any command line arguments that override defaults */
  info = PetscOptionsGetReal(PETSC_NULL,"-ecc",&user.ecc,&flg); CHKERRQ(info);
  info = PetscOptionsGetReal(PETSC_NULL,"-b",&user.b,&flg); CHKERRQ(info);

  /*
     A two dimensional distributed array will help define this problem,
     which derives from an elliptic PDE on two dimensional domain.  From
     the distributed array, Create the vectors.
  */
  info = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-50,-50,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(info);
  info = DMDAGetInfo(user.da,PETSC_IGNORE,&user.nx,&user.ny,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(info);

  PetscPrintf(PETSC_COMM_WORLD,"\n---- Journal Bearing Problem -----\n");
  PetscPrintf(PETSC_COMM_WORLD,"mx: %d,  my: %d,  ecc: %4.3f, b:%3.1f \n",
              user.nx,user.ny,user.ecc,user.b);
  /*
     Extract global and local vectors from DA; the vector user.B is
     used solely as work space for the evaluation of the function, 
     gradient, and Hessian.  Duplicate for remaining vectors that are 
     the same types.
  */
  info = DMCreateGlobalVector(user.da,&x); CHKERRQ(info); /* Solution */
  info = VecDuplicate(x,&user.B); CHKERRQ(info); /* Linear objective */
  info = VecDuplicate(x,&r);CHKERRQ(info);

  /*  Create matrix user.A to store quadratic, Create a local ordering scheme. */
  info = DMCreateMatrix(user.da,MATAIJ,&user.A);CHKERRQ(info);
  
  /* User defined function -- compute linear term of quadratic */
  info = ComputeB(&user); CHKERRQ(info);

  /* Create nonlinear solver context */
  info = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(info);

  /*  Set function evaluation and Jacobian evaluation  routines */
  info = SNESSetFunction(snes,r,FormGradient,&user);CHKERRQ(info);
  info = SNESSetJacobian(snes,user.A,user.A,FormHessian,&user);CHKERRQ(info);

  /* Set the initial solution guess */
  info = VecSet(x, zero); CHKERRQ(info);

  info = SNESSetFromOptions(snes);CHKERRQ(info);

  /* Set variable bounds */
  info = VecDuplicate(x,&xl);CHKERRQ(info);
  info = VecDuplicate(x,&xu);CHKERRQ(info);
  info = VecSet(xl,zero);CHKERRQ(info);
  info = VecSet(xu,thnd);CHKERRQ(info);
  info = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(info);

  /* Solve the application */
  info = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(info);

  info = SNESGetConvergedReason(snes,&reason); CHKERRQ(info);
  if (reason <= 0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"The SNESVI solver did not converge, adjust some parameters, or check the function evaluation routines\n");
  }
  
  /* Free memory */
  info = VecDestroy(&x); CHKERRQ(info); 
  info = VecDestroy(&xl);CHKERRQ(info);
  info = VecDestroy(&xu);CHKERRQ(info);
  info = VecDestroy(&r);CHKERRQ(info);
  info = MatDestroy(&user.A); CHKERRQ(info);
  info = VecDestroy(&user.B); CHKERRQ(info); 
  info = DMDestroy(&user.da); CHKERRQ(info);
  info = SNESDestroy(&snes);CHKERRQ(info);

  info = PetscFinalize();

  return 0;
}

static PetscReal p(PetscReal xi, PetscReal ecc)
{ 
  PetscReal t=1.0+ecc*cos(xi);
  return(t*t*t);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeB"
PetscErrorCode ComputeB(AppCtx* user)
{
  PetscErrorCode info;
  PetscInt       i,j;
  PetscInt       nx,ny,xs,xm,ys,ym;
  PetscReal      two=2.0, pi=4.0*atan(1.0);
  PetscReal      hx,hy,ehxhy;
  PetscReal      temp;
  PetscReal      ecc=user->ecc;
  PetscReal      **b;

  PetscFunctionBegin;
  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  ehxhy = ecc*hx*hy;

  /* Get pointer to local vector data */
  info = DMDAVecGetArray(user->da,user->B, &b); CHKERRQ(info);

  info = DMDAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);

  /* Compute the linear term in the objective function */  
  for (i=xs; i<xs+xm; i++){
    temp=sin((i+1)*hx);
    for (j=ys; j<ys+ym; j++){
      b[j][i] =  - ehxhy*temp;
    }
  }
  /* Restore vectors */
  info = DMDAVecRestoreArray(user->da,user->B,&b);CHKERRQ(info);
  info = PetscLogFlops(5*xm*ym+3*xm); CHKERRQ(info);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormGradient"
PetscErrorCode FormGradient(SNES snes, Vec X, Vec G,void *ctx)
{
  AppCtx*        user=(AppCtx*)ctx;
  PetscErrorCode info;
  PetscInt       i,j,k,kk;
  PetscInt       row[5],col[5];
  PetscInt       nx,ny,xs,xm,ys,ym;
  PetscReal      one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal      hx,hy,hxhy,hxhx,hyhy;
  PetscReal      xi,v[5];
  PetscReal      ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal      vmiddle, vup, vdown, vleft, vright;
  PetscReal      tt;
  PetscReal      **x,**g;
  PetscReal      zero=0.0;
  Vec            localX;

  PetscFunctionBegin;
  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  info = VecSet(G, zero); CHKERRQ(info);
  
  /* Get local vector */
  info = DMGetLocalVector(user->da,&localX);CHKERRQ(info);
  /* Get ghoist points */
  info = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(info);
  info = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(info);
  /* Get pointer to vector data */
  info = DMDAVecGetArray(user->da,localX,&x);CHKERRQ(info);
  info = DMDAVecGetArray(user->da,G,&g);CHKERRQ(info);

  info = DMDAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(info);

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);

    for (j=ys; j<ys+ym; j++){
      
       v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;
       
       k=0;
       if (j > 0){ 
	 v[k]=vdown; row[k] = i; col[k] = j-1; k++;
       }
       
       if (i > 0){
	 v[k]= vleft; row[k] = i-1; col[k] = j; k++;
       }

       v[k]= vmiddle; row[k] = i; col[k] = j; k++;
       
       if (i+1 < nx){
	 v[k]= vright; row[k] = i+1; col[k] = j; k++;
       }
       
       if (j+1 < ny){
	 v[k]= vup; row[k] = i; col[k] = j+1; k++;
       }
       tt=0;
       for (kk=0;kk<k;kk++){
	 tt+=v[kk]*x[col[kk]][row[kk]];
       }
       g[j][i] = tt;

     }

  }

  /* Restore vectors */
  info = DMDAVecRestoreArray(user->da,localX, &x); CHKERRQ(info);
  info = DMDAVecRestoreArray(user->da,G, &g); CHKERRQ(info);
  info = DMRestoreLocalVector(user->da,&localX);CHKERRQ(info);

  info = VecAXPY(G, one, user->B); CHKERRQ(info);

  info = PetscLogFlops((91 + 10*ym) * xm); CHKERRQ(info);
  PetscFunctionReturn(0);

}



#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/* 
   FormHessian computes the quadratic term in the quadratic objective function 
   Notice that the objective function in this problem is quadratic (therefore a constant
   hessian).  If using a nonquadratic solver, then you might want to reconsider this function
*/
PetscErrorCode FormHessian(SNES snes,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr)
{
  AppCtx*        user=(AppCtx*)ptr;
  PetscErrorCode info;
  PetscInt       i,j,k;
  MatStencil     row,col[5];
  PetscInt       nx,ny,xs,xm,ys,ym;
  PetscReal      one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal      hx,hy,hxhy,hxhx,hyhy;
  PetscReal      xi,v[5];
  PetscReal      ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal      vmiddle, vup, vdown, vleft, vright;
  Mat            hes=*H;
  PetscBool      assembled;
  PetscReal      **x;
  Vec            localX;

  PetscFunctionBegin;
  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  info = MatAssembled(hes,&assembled); CHKERRQ(info);
  if (assembled){info = MatZeroEntries(hes);  CHKERRQ(info);}
  *flg=SAME_NONZERO_PATTERN;

  /* Get local vector */
  info = DMGetLocalVector(user->da,&localX);CHKERRQ(info);
  /* Get ghost points */
  info = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(info);
  info = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(info);
 
  /* Get pointers to vector data */
  info = DMDAVecGetArray(user->da,localX, &x); CHKERRQ(info);

  info = DMDAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(info);

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);
    v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;

    for (j=ys; j<ys+ym; j++){
      k=0;
      row.i = i; row.j = j;
      if (j > 0){ 
	v[k]=vdown; col[k].i=i;col[k].j = j-1; k++;
      }
       
      if (i > 0){
	v[k]= vleft; col[k].i= i-1; col[k].j = j;k++;
      }

      v[k]= vmiddle; col[k].i=i; col[k].j = j;k++;
       
      if (i+1 < nx){
	v[k]= vright; col[k].i = i+1; col[k].j = j; k++;
      }
       
      if (j+1 < ny){
	v[k]= vup; col[k].i = i; col[k].j = j+1; k++;
      }
      info = MatSetValuesStencil(hes,1,&row,k,col,v,INSERT_VALUES); CHKERRQ(info);
    }
  }

  info = MatAssemblyBegin(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  info = DMDAVecRestoreArray(user->da,localX,&x);CHKERRQ(info);
  info = MatAssemblyEnd(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  info = DMRestoreLocalVector(user->da,&localX);CHKERRQ(info);

  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  info = MatSetOption(hes,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(info);
  info = MatSetOption(hes,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(info);

  info = PetscLogFlops(9*xm*ym+49*xm); CHKERRQ(info);

  PetscFunctionReturn(0);
}
