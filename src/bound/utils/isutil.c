#include "petscvec.h"
#include "petscis.h"
#include "tao.h"


#undef __FUNCT__
#define __FUNCT__ "VecWhichEqual"
PetscErrorCode VecWhichEqual(Vec Vec1, Vec Vec2, IS * S)
{
  /* 
     Create an index set containing the indices of
     the vectors Vec1 and Vec2 with identical elements.
  */
  PetscErrorCode    ierr;
  PetscInt i,n_same=0;
  PetscInt n,low,high,low2,high2;
  PetscInt    *same;
  PetscReal *v1,*v2;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1); 
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2); 
  PetscCheckSameComm(Vec1,1,Vec2,2);


  ierr = VecGetOwnershipRange(Vec1, &low, &high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Vec2, &low2, &high2); CHKERRQ(ierr);

  if ( low != low2 || high != high2 )
    SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");

  ierr = VecGetLocalSize(Vec1,&n); CHKERRQ(ierr);

  if (n>0){
    
    if (Vec1 == Vec2){
      ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
      v2=v1;
    } else {
      ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
      ierr = VecGetArray(Vec2,&v2); CHKERRQ(ierr);
    }

    ierr = PetscMalloc( n*sizeof(PetscInt),&same ); CHKERRQ(ierr);
    
    for (i=0; i<n; i++){
      if (v1[i] == v2[i]) {same[n_same]=low+i; n_same++;}
    }
    
    if (Vec1 == Vec2){
      ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
      ierr = VecRestoreArray(Vec2,&v2); CHKERRQ(ierr);
    }

  } else {

    n_same = 0; same=NULL;

  }

  ierr = PetscObjectGetComm((PetscObject)Vec1,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_same,same,PETSC_COPY_VALUES,S);CHKERRQ(ierr);

  if (same) {
    ierr = PetscFree(same); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWhichLessThan"
PetscErrorCode VecWhichLessThan(Vec Vec1, Vec Vec2, IS * S)
{
  int ierr;
  PetscInt i;
  PetscInt n,low,high,low2,high2,n_lt=0;
  PetscInt *lt;
  PetscReal *v1,*v2;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1); 
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2); 
  PetscCheckSameComm(Vec1,1,Vec2,2);

  ierr = VecGetOwnershipRange(Vec1, &low, &high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Vec2, &low2, &high2); CHKERRQ(ierr);

  if ( low != low2 || high != high2 )
    SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");

  ierr = VecGetLocalSize(Vec1,&n); CHKERRQ(ierr);

  if (n>0){

    if (Vec1 == Vec2){
      ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
      v2=v1;
    } else {
      ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
      ierr = VecGetArray(Vec2,&v2); CHKERRQ(ierr);
    }
    ierr = PetscMalloc( n*sizeof(PetscInt),&lt ); CHKERRQ(ierr);
    
    for (i=0; i<n; i++){
      if (v1[i] < v2[i]) {lt[n_lt]=high+i; n_lt++;}
    }

    if (Vec1 == Vec2){
      ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
      ierr = VecRestoreArray(Vec2,&v2); CHKERRQ(ierr);
    }
      
  } else {
    n_lt=0; lt=NULL;
  }

  ierr = PetscObjectGetComm((PetscObject)Vec1,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_lt,lt,PETSC_COPY_VALUES,S);CHKERRQ(ierr);

  if (lt) {
    ierr = PetscFree(lt); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWhichGreaterThan"
PetscErrorCode VecWhichGreaterThan(Vec Vec1, Vec Vec2, IS * S)
{
  int    ierr;
  PetscInt n,low,high,low2,high2,n_gt=0,i;
  PetscInt    *gt=NULL;
  PetscReal *v1,*v2;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1); 
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2); 
  PetscCheckSameComm(Vec1,1,Vec2,2);

  ierr = VecGetOwnershipRange(Vec1, &low, &high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Vec2, &low2, &high2); CHKERRQ(ierr);

  if ( low != low2 || high != high2 )
    SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");

  ierr = VecGetLocalSize(Vec1,&n); CHKERRQ(ierr);

  if (n>0){

    if (Vec1 == Vec2){
      ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
      v2=v1;
    } else {
      ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
      ierr = VecGetArray(Vec2,&v2); CHKERRQ(ierr);
    }    

    ierr = PetscMalloc( n*sizeof(PetscInt), &gt ); CHKERRQ(ierr);
    
    for (i=0; i<n; i++){
      if (v1[i] > v2[i]) {gt[n_gt]=low+i; n_gt++;}
    }

    if (Vec1 == Vec2){
      ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
      ierr = VecRestoreArray(Vec2,&v2); CHKERRQ(ierr);
    }
    
  } else{
    
    n_gt=0; gt=NULL;

  }

  ierr = PetscObjectGetComm((PetscObject)Vec1,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_gt,gt,PETSC_COPY_VALUES,S);CHKERRQ(ierr);

  if (gt) {
    ierr = PetscFree(gt); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWhichBetween"
PetscErrorCode VecWhichBetween(Vec VecLow, Vec V, Vec VecHigh, IS * S)
{
  /* 
     Creates an index set with the indices of V whose 
     elements are stictly between the corresponding elements 
     of the vector VecLow and the Vector VecHigh
  */
  PetscErrorCode ierr;
  PetscInt n,low,high,low2,high2,low3,high3,n_vm=0;
  PetscInt *vm,i;
  PetscReal *v1,*v2,*vmiddle;
  MPI_Comm comm;

  PetscValidHeaderSpecific(V,VEC_CLASSID,2); 
  PetscCheckSameComm(V,2,VecLow,1); PetscCheckSameComm(V,2,VecHigh,3);

  ierr = VecGetOwnershipRange(VecLow, &low, &high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(VecHigh, &low2, &high2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(V, &low3, &high3); CHKERRQ(ierr);

  if ( low!=low2 || high!=high2 || low!=low3 || high!=high3 )
    SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");

  ierr = VecGetLocalSize(VecLow,&n); CHKERRQ(ierr);

  if (n>0){

    ierr = VecGetArray(VecLow,&v1); CHKERRQ(ierr);
    if (VecLow != VecHigh){
      ierr = VecGetArray(VecHigh,&v2); CHKERRQ(ierr);
    } else {
      v2=v1;
    }
    if ( V != VecLow && V != VecHigh){
      ierr = VecGetArray(V,&vmiddle); CHKERRQ(ierr);
    } else if ( V==VecLow ){
      vmiddle=v1;
    } else {
      vmiddle =v2;
    }

    ierr = PetscMalloc( n*sizeof(PetscInt), &vm ); CHKERRQ(ierr);
    
    for (i=0; i<n; i++){
      if (v1[i] < vmiddle[i] && vmiddle[i] < v2[i]) {vm[n_vm]=low+i; n_vm++;}
    }

    ierr = VecRestoreArray(VecLow,&v1); CHKERRQ(ierr);
    if (VecLow != VecHigh){
      ierr = VecRestoreArray(VecHigh,&v2); CHKERRQ(ierr);
    }
    if ( V != VecLow && V != VecHigh){
      ierr = VecRestoreArray(V,&vmiddle); CHKERRQ(ierr);
    }

  } else {

    n_vm=0; vm=NULL;

  }

  ierr = PetscObjectGetComm((PetscObject)V,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_vm,vm,PETSC_COPY_VALUES,S);CHKERRQ(ierr);

  if (vm) {
    ierr = PetscFree(vm); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecWhichBetweenOrEqual"
PetscErrorCode VecWhichBetweenOrEqual(Vec VecLow, Vec V, Vec VecHigh, IS * S)
{
  /* 
     Creates an index set with the indices of V whose 
     elements are stictly between the corresponding elements 
     of the vector VecLow and the Vector VecHigh
  */
  PetscErrorCode ierr;
  PetscInt n,low,high,low2,high2,low3,high3,n_vm=0,i;
  PetscInt *vm;
  PetscReal *v1,*v2,*vmiddle;
  MPI_Comm comm;

  PetscValidHeaderSpecific(V,VEC_CLASSID,2); 
  PetscCheckSameComm(V,2,VecLow,1); PetscCheckSameComm(V,2,VecHigh,3);

  ierr = VecGetOwnershipRange(VecLow, &low, &high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(VecHigh, &low2, &high2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(V, &low3, &high3); CHKERRQ(ierr);

  if ( low!=low2 || high!=high2 || low!=low3 || high!=high3 )
    SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");

  ierr = VecGetLocalSize(VecLow,&n); CHKERRQ(ierr);

  if (n>0){

    ierr = VecGetArray(VecLow,&v1); CHKERRQ(ierr);
    if (VecLow != VecHigh){
      ierr = VecGetArray(VecHigh,&v2); CHKERRQ(ierr);
    } else {
      v2=v1;
    }
    if ( V != VecLow && V != VecHigh){
      ierr = VecGetArray(V,&vmiddle); CHKERRQ(ierr);
    } else if ( V==VecLow ){
      vmiddle=v1;
    } else {
      vmiddle =v2;
    }

    ierr = PetscMalloc( n*sizeof(PetscInt), &vm ); CHKERRQ(ierr);
    
    for (i=0; i<n; i++){
      if (v1[i] <= vmiddle[i] && vmiddle[i] <= v2[i]) {vm[n_vm]=low+i; n_vm++;}
    }

    ierr = VecRestoreArray(VecLow,&v1); CHKERRQ(ierr);
    if (VecLow != VecHigh){
      ierr = VecRestoreArray(VecHigh,&v2); CHKERRQ(ierr);
    }
    if ( V != VecLow && V != VecHigh){
      ierr = VecRestoreArray(V,&vmiddle); CHKERRQ(ierr);
    }

  } else {

    n_vm=0; vm=NULL;

  }

  ierr = PetscObjectGetComm((PetscObject)V,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_vm,vm,PETSC_COPY_VALUES,S);CHKERRQ(ierr);

  if (vm) {
    ierr = PetscFree(vm); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetSubVec"
PetscErrorCode VecGetSubVec(Vec vfull, IS is, Vec *vreduced) 
{
    PetscErrorCode ierr;
    PetscInt nfull,nreduced,nreduced_local,rlow,rhigh,flow,fhigh,nfull_local;
    IS ident;
    const VecType vtype;
    VecScatter scatter;
    MPI_Comm comm;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
    PetscValidHeaderSpecific(is,IS_CLASSID,2);
	
    ierr = VecGetSize(vfull, &nfull); CHKERRQ(ierr);
    ierr = ISGetSize(is, &nreduced); CHKERRQ(ierr);

    if (nreduced == nfull) {
	ierr = VecDuplicate(vfull,vreduced); CHKERRQ(ierr);
	ierr = VecCopy(vfull,*vreduced); CHKERRQ(ierr);
    } else {
      
	ierr = VecGetType(vfull,&vtype); CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(vfull,&flow,&fhigh); CHKERRQ(ierr);
	nfull_local = fhigh - flow;
	ierr = ISGetLocalSize(is,&nreduced_local); CHKERRQ(ierr);
	ierr = PetscObjectGetComm((PetscObject)vfull,&comm); CHKERRQ(ierr);
	if (*vreduced) {
	  ierr = VecDestroy(vreduced); CHKERRQ(ierr);
	}
	ierr = VecCreate(comm,vreduced); CHKERRQ(ierr);
	ierr = VecSetType(*vreduced,vtype); CHKERRQ(ierr);
	ierr = VecSetSizes(*vreduced,nreduced_local,nreduced); CHKERRQ(ierr);
	
	ierr = VecGetOwnershipRange(*vreduced,&rlow,&rhigh); CHKERRQ(ierr);
	
	ierr = ISCreateStride(comm,nreduced_local,rlow,1,&ident); CHKERRQ(ierr);
	ierr = VecScatterCreate(vfull,is,*vreduced,ident,&scatter); CHKERRQ(ierr);
	ierr = VecScatterBegin(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(scatter,vfull,*vreduced,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRQ(ierr);
	ierr = ISDestroy(&ident); CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "VecReducedXPY"
PetscErrorCode VecReducedXPY(Vec vfull, Vec vreduced, IS is)
{
    VecScatter scatter;
    IS ident;
    PetscInt nfull,nreduced,rlow,rhigh;
    MPI_Comm comm;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
    PetscValidHeaderSpecific(vreduced,VEC_CLASSID,2);
    PetscValidHeaderSpecific(is,IS_CLASSID,3);
    ierr = VecGetSize(vfull,&nfull); CHKERRQ(ierr);
    ierr = VecGetSize(vreduced,&nreduced); CHKERRQ(ierr);
    
    if (nfull == nreduced) {
	ierr = VecAXPY(vfull,1.0,vreduced); CHKERRQ(ierr);
    } else {
	ierr = PetscObjectGetComm((PetscObject)vfull,&comm); CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(vreduced,&rlow,&rhigh); CHKERRQ(ierr);
	ierr = ISCreateStride(comm,rhigh-rlow,rlow,1,&ident); CHKERRQ(ierr);
	ierr = VecScatterCreate(vreduced,ident,vfull,is,&scatter); CHKERRQ(ierr);
	ierr = VecScatterBegin(scatter,vreduced,vfull,ADD_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(scatter,vreduced,vfull,ADD_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRQ(ierr);
	ierr = ISDestroy(&ident); CHKERRQ(ierr);
    }
    
    PetscFunctionReturn(0);
}



