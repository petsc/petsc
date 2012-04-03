#include "tao.h" /*I "tao.h" I*/
#include "petsc-private/matimpl.h"
#include "src/matrix/submatfree.h"
#include "tao_util.h" /*I "tao_util.h" I*/


#undef __FUNCT__
#define __FUNCT__ "VecWhichEqual"
/*@
  VecWhichEqual - Creates an index set containing the indices 
  where the vectors Vec1 and Vec2 have identical elements.
  
  Collective on S

  Input Parameters:
. Vec1, Vec2 - the two vectors to compare

  OutputParameter:
. S - The index set containing the indices i where vec1[i] == vec2[i]

  Level: advanced
@*/
PetscErrorCode VecWhichEqual(Vec Vec1, Vec Vec2, IS * S)
{
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
/*@
  VecWhichLessThan - Creates an index set containing the indices 
  where the vectors Vec1 < Vec2
  
  Collective on S

  Input Parameters:
. Vec1, Vec2 - the two vectors to compare

  OutputParameter:
. S - The index set containing the indices i where vec1[i] < vec2[i]

  Level: advanced
@*/
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
/*@ 
  VecWhichGreaterThan - Creates an index set containing the indices 
  where the vectors Vec1 > Vec2
  
  Collective on S

  Input Parameters:
. Vec1, Vec2 - the two vectors to compare

  OutputParameter:
. S - The index set containing the indices i where vec1[i] > vec2[i]

  Level: advanced
@*/
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
/*@
  VecWhichBetween - Creates an index set containing the indices 
  where  VecLow < V < VecHigh
  
  Collective on S

  Input Parameters:
+ VecLow - lower bound
. V - Vector to compare
- VecHigh - higher bound

  OutputParameter:
. S - The index set containing the indices i where veclow[i] < v[i] < vechigh[i]

  Level: advanced
@*/
PetscErrorCode VecWhichBetween(Vec VecLow, Vec V, Vec VecHigh, IS *S)
{

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
/*@  
  VecWhichBetweenOrEqual - Creates an index set containing the indices 
  where  VecLow <= V <= VecHigh
  
  Collective on S

  Input Parameters:
+ VecLow - lower bound
. V - Vector to compare
- VecHigh - higher bound

  OutputParameter:
. S - The index set containing the indices i where veclow[i] <= v[i] <= vechigh[i]

  Level: advanced
@*/

PetscErrorCode VecWhichBetweenOrEqual(Vec VecLow, Vec V, Vec VecHigh, IS * S)
{
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
/*@ 
  VecGetSubVec - Gets a subvector using the IS

  Input Parameters:
+ vfull - the full matrix 
. is - the index set for the subvector
. reduced_type - the method TAO is using for subsetting (TAO_SUBSET_SUBVEC, TAO_SUBSET_MASK,  TAO_SUBSET_MATRIXFREE)
- maskvalue - the value to set the unused vector elements to (for TAO_SUBSET_MASK or TAO_SUBSET_MATRIXFREE)


  Output Parameters:
. vreduced - the subvector

  Note:
  maskvalue should usually be 0.0, unless a pointwise divide will be used.
@*/
PetscErrorCode VecGetSubVec(Vec vfull, IS is, PetscInt reduced_type, PetscReal maskvalue, Vec *vreduced) 
{
    PetscErrorCode ierr;
    PetscInt nfull,nreduced,nreduced_local,rlow,rhigh,flow,fhigh;
    PetscInt i,nlocal;
    PetscReal *fv,*rv;
    const PetscInt *s;
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

      ierr = VecDestroy(vreduced); CHKERRQ(ierr);
      ierr = VecDuplicate(vfull,vreduced); CHKERRQ(ierr);
      ierr = VecCopy(vfull,*vreduced); CHKERRQ(ierr);

    } else { 
     
	switch (reduced_type) {
	case TAO_SUBSET_SUBVEC:
	  ierr = VecGetType(vfull,&vtype); CHKERRQ(ierr);
	  ierr = VecGetOwnershipRange(vfull,&flow,&fhigh); CHKERRQ(ierr);
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
	  break;

	case TAO_SUBSET_MASK:
	case TAO_SUBSET_MATRIXFREE:
	  /* vr[i] = vf[i]   if i in is
             vr[i] = 0       otherwise */
	  if (*vreduced == PETSC_NULL) {
	    ierr = VecDuplicate(vfull,vreduced); CHKERRQ(ierr);
	  }

	  ierr = VecSet(*vreduced,maskvalue); CHKERRQ(ierr);
	  ierr = ISGetLocalSize(is,&nlocal); CHKERRQ(ierr);
	  ierr = VecGetOwnershipRange(vfull,&flow,&fhigh); CHKERRQ(ierr);
	  ierr = VecGetArray(vfull,&fv); CHKERRQ(ierr);
	  ierr = VecGetArray(*vreduced,&rv); CHKERRQ(ierr);
	  ierr = ISGetIndices(is,&s); CHKERRQ(ierr);
	  if (nlocal > (fhigh-flow)) {
	    SETERRQ2(PETSC_COMM_WORLD,1,"IS local size %d > Vec local size %d",nlocal,fhigh-flow);
	  }
	  for (i=0;i<nlocal;i++) {
	    rv[s[i]-flow] = fv[s[i]-flow];
	  }
	  ierr = ISRestoreIndices(is,&s); CHKERRQ(ierr);
	  ierr = VecRestoreArray(vfull,&fv); CHKERRQ(ierr);
	  ierr = VecRestoreArray(*vreduced,&rv); CHKERRQ(ierr);
	  break;
	}
    }
    PetscFunctionReturn(0);
    
}

#undef __FUNCT__
#define __FUNCT__ "VecReducedXPY"
/*@ 
  VecReducedXPY - Adds a reduced vector to the appropriate elements of a full-space vector.

  Input Parameters:
+ vfull - the full-space vector
. vreduced - the reduced-space vector
- is - the index set for the reduced space

  Output Parameters:
. vfull - the sum of the full-space vector and reduced-space vector
@*/
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
    
    if (nfull == nreduced) { /* Also takes care of masked vectors */
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


#undef __FUNCT__  
#define __FUNCT__ "ISCreateComplement"
/*@
   ISCreateComplement - Creates the complement of the the index set

   Collective on IS

   Input Parameter:
+  S -  a PETSc IS
-  V - the reference vector space

   Output Parameter:
.  T -  the complement of S


.seealso ISCreateGeneral()

   Level: advanced
@*/
PetscErrorCode ISCreateComplement(IS S, Vec V, IS *T){
  PetscErrorCode ierr;
  PetscInt i,nis,nloc,high,low,n=0;
  const PetscInt *s;
  PetscInt *tt,*ss;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,IS_CLASSID,1); 
  PetscValidHeaderSpecific(V,VEC_CLASSID,2); 

  ierr = VecGetOwnershipRange(V,&low,&high); CHKERRQ(ierr);
  ierr = VecGetLocalSize(V,&nloc); CHKERRQ(ierr);
  ierr = ISGetLocalSize(S,&nis); CHKERRQ(ierr);
  ierr = ISGetIndices(S, &s); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt),&tt ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt),&ss ); CHKERRQ(ierr);

  for (i=low; i<high; i++){ tt[i-low]=i; }

  for (i=0; i<nis; i++){ tt[s[i]-low] = -2; }
  
  for (i=0; i<nloc; i++){
    if (tt[i]>-1){ ss[n]=tt[i]; n++; }
  }

  ierr = ISRestoreIndices(S, &s); CHKERRQ(ierr);
  
  ierr = PetscObjectGetComm((PetscObject)S,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,ss,PETSC_COPY_VALUES,T);CHKERRQ(ierr);
  
  if (tt) {
    ierr = PetscFree(tt); CHKERRQ(ierr);
  }
  if (ss) {
    ierr = PetscFree(ss); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecISSetToConstant"
/*@
   VecISSetToConstant - Sets the elements of a vector, specified by an index set, to a constant

   Input Parameter:
+  S -  a PETSc IS
.  c - the constant
-  V - a Vec

.seealso VecSet()

   Level: advanced
@*/
PetscErrorCode VecISSetToConstant(IS S, PetscReal c, Vec V){
  PetscErrorCode ierr;
  PetscInt nloc,low,high,i;
  const PetscInt *s;
  PetscReal *v;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,VEC_CLASSID,3); 
  PetscValidHeaderSpecific(S,IS_CLASSID,1); 
  PetscValidType(V,3);
  PetscCheckSameComm(V,3,S,1);

  ierr = VecGetOwnershipRange(V, &low, &high); CHKERRQ(ierr);
  ierr = ISGetLocalSize(S,&nloc);CHKERRQ(ierr);

  ierr = ISGetIndices(S, &s); CHKERRQ(ierr);
  ierr = VecGetArray(V,&v); CHKERRQ(ierr);
  for (i=0; i<nloc; i++){
    v[s[i]-low] = c;
  }
  
  ierr = ISRestoreIndices(S, &s); CHKERRQ(ierr);
  ierr = VecRestoreArray(V,&v); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMat"
/*@ 
  MatGetSubMat - Gets a submatrix using the IS

  Input Parameters:
+ M - the full matrix (n x n)
. is - the index set for the submatrix (both row and column index sets need to be the same)
. v1 - work vector of dimension n, needed for TAO_SUBSET_MASK option
- subset_type - the method TAO is using for subsetting (TAO_SUBSET_SUBVEC, TAO_SUBSET_MASK,
  TAO_SUBSET_MATRIXFREE)

  Output Parameters:
. Msub - the submatrix
@*/
PetscErrorCode MatGetSubMat(Mat M, IS is, Vec v1, TaoSubsetType subset_type, Mat *Msub)
{
  PetscErrorCode ierr;
  IS iscomp;
  PetscBool flg;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(M,MAT_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  if (*Msub) {
    ierr = MatDestroy(Msub); CHKERRQ(ierr);
  }
  switch (subset_type) {
    case TAO_SUBSET_SUBVEC:
      ierr = MatGetSubMatrix(M, is, is, MAT_INITIAL_MATRIX, Msub); CHKERRQ(ierr);
      break;

    case TAO_SUBSET_MASK:
      /* Get Reduced Hessian 
	 Msub[i,j] = M[i,j] if i,j in Free_Local or i==j
	 Msub[i,j] = 0      if i!=j and i or j not in Free_Local
      */
      ierr = PetscOptionsBool("-different_submatrix","use separate hessian matrix when computing submatrices","TaoSubsetType",PETSC_FALSE,&flg,PETSC_NULL);
      if (flg == PETSC_TRUE) {
	ierr = MatDuplicate(M, MAT_COPY_VALUES, Msub); CHKERRQ(ierr);
      } else {
	/* Act on hessian directly (default) */
	ierr = PetscObjectReference((PetscObject)M); CHKERRQ(ierr);
	*Msub = M;
      }
      /* Save the diagonal to temporary vector */
      ierr = MatGetDiagonal(*Msub,v1); CHKERRQ(ierr);
    
      /* Zero out rows and columns */
      ierr = ISCreateComplement(is,v1,&iscomp); CHKERRQ(ierr);

      /* Use v1 instead of 0 here because of PETSc bug */
      ierr = MatZeroRowsColumnsIS(*Msub,iscomp,1.0,v1,v1); CHKERRQ(ierr);

      ierr = ISDestroy(&iscomp); CHKERRQ(ierr);
      break;
    case TAO_SUBSET_MATRIXFREE:
      ierr = ISCreateComplement(is,v1,&iscomp); CHKERRQ(ierr);
      ierr = MatCreateSubMatrixFree(M,iscomp,iscomp,Msub); CHKERRQ(ierr);
      ierr = ISDestroy(&iscomp); CHKERRQ(ierr);
      break;
  }      
  PetscFunctionReturn(0);
}
