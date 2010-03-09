
static char help[] = "Tests DA wirebasket interpolation.\n\n";

#include "petscda.h"

extern PetscErrorCode DAGetWireBasketInterpolation(DA,Mat,MatReuse,Mat*);
extern PetscErrorCode ComputeMatrix(DA,Mat);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DA             da;
  Mat            Aglobal,P;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,3,5,5,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATAIJ,&Aglobal);CHKERRQ(ierr);
  ierr = ComputeMatrix(da,Aglobal);CHKERRQ(ierr);
  ierr = DAGetWireBasketInterpolation(da,Aglobal,MAT_INITIAL_MATRIX,&P);CHKERRQ(ierr);
  ierr = MatView(P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = MatDestroy(Aglobal);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DA da,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,cnt;
  PetscScalar    v[7],Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy,val;
  MatStencil     row,col[7];

  ierr = DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  
#define Endpoint(a,b) (a == 0 || a == (b-1))

  for (k=zs; k<zs+zm; k++){
    for (j=ys; j<ys+ym; j++){
      for(i=xs; i<xs+xm; i++){
        row.i = i; row.j = j; row.k = k;
        cnt   = 0;
        val   = 0;
	if (k != 0) { val += v[cnt] = -HxHydHz;col[cnt].i = i; col[cnt].j = j; col[cnt].k = k-1;cnt++;}
	if (j != 0) { val += v[cnt] = -HxHzdHy;col[cnt].i = i; col[cnt].j = j-1; col[cnt].k = k;cnt++;}
	if (i != 0) { val += v[cnt] = -HyHzdHx;col[cnt].i = i-1; col[cnt].j = j; col[cnt].k = k;cnt++;}
	if (i != mx-1) {val += v[cnt] = -HyHzdHx;col[cnt].i = i+1; col[cnt].j = j; col[cnt].k = k;cnt++;}
	if (j != my-1) {val += v[cnt] = -HxHzdHy;col[cnt].i = i; col[cnt].j = j+1; col[cnt].k = k;cnt++;}
	if (k != mz-1) {val += v[cnt] = -HxHydHz;col[cnt].i = i; col[cnt].j = j; col[cnt].k = k+1;cnt++;}
        v[cnt] = -val;col[cnt].i = row.i; col[cnt].j = row.j; col[cnt].k = row.k;cnt++;
        ierr = MatSetValuesStencil(B,1,&row,cnt,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}



