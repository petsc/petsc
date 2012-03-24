
#include <petsc-private/matimpl.h>               /*I "petscmat.h" I*/

/* Logging support */
PetscClassId  MAT_PARTITIONING_CLASSID;

/*
   Simplest partitioning, keeps the current partitioning.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningApply_Current" 
static PetscErrorCode MatPartitioningApply_Current(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscInt       m;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)part)->comm,&size);CHKERRQ(ierr);
  if (part->n != size) {
    const char *prefix;
    ierr = PetscObjectGetOptionsPrefix((PetscObject)part,&prefix);CHKERRQ(ierr);
    SETERRQ1(((PetscObject)part)->comm,PETSC_ERR_SUP,"This is the DEFAULT NO-OP partitioner, it currently only supports one domain per processor\nuse -%smat_partitioning_type parmetis or chaco or ptscotch for more than one subdomain per processor",prefix?prefix:"");
  }
  ierr = MPI_Comm_rank(((PetscObject)part)->comm,&rank);CHKERRQ(ierr);

  ierr = MatGetLocalSize(part->adj,&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)part)->comm,m,rank,0,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningApply_Square" 
static PetscErrorCode MatPartitioningApply_Square(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscInt       cell,n,N,p,rstart,rend,*color;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)part)->comm,&size);CHKERRQ(ierr);
  if (part->n != size) SETERRQ(((PetscObject)part)->comm,PETSC_ERR_SUP,"Currently only supports one domain per processor");
  p = (PetscInt)sqrt((double)part->n);
  if (p*p != part->n) SETERRQ(((PetscObject)part)->comm,PETSC_ERR_SUP,"Square partitioning requires \"perfect square\" number of domains");

  ierr = MatGetSize(part->adj,&N,PETSC_NULL);CHKERRQ(ierr);
  n = (PetscInt)sqrt((double)N);
  if (n*n != N) SETERRQ(((PetscObject)part)->comm,PETSC_ERR_SUP,"Square partitioning requires square domain");
  if (n%p != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Square partitioning requires p to divide n");
  ierr = MatGetOwnershipRange(part->adj,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc((rend-rstart)*sizeof(PetscInt),&color);CHKERRQ(ierr);
  /* for (int cell=rstart; cell<rend; cell++) { color[cell-rstart] = ((cell%n) < (n/2)) + 2 * ((cell/n) < (n/2)); } */
  for (cell=rstart; cell<rend; cell++) {
    color[cell-rstart] = ((cell%n) / (n/p)) + p * ((cell/n) / (n/p));
  }
  ierr = ISCreateGeneral(((PetscObject)part)->comm,rend-rstart,color,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN  
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningCreate_Current" 
PetscErrorCode  MatPartitioningCreate_Current(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Current;
  part->ops->view    = 0;
  part->ops->destroy = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN  
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningCreate_Square" 
PetscErrorCode  MatPartitioningCreate_Square(MatPartitioning part)
{
  PetscFunctionBegin;
  part->ops->apply   = MatPartitioningApply_Square;
  part->ops->view    = 0;
  part->ops->destroy = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ===========================================================================================*/

PetscFList MatPartitioningList = 0;
PetscBool  MatPartitioningRegisterAllCalled = PETSC_FALSE;


#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningRegister" 
PetscErrorCode  MatPartitioningRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(MatPartitioning))
{
  PetscErrorCode ierr;
  char fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatPartitioningList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningRegisterDestroy" 
/*@C
   MatPartitioningRegisterDestroy - Frees the list of partitioning routines.

  Not Collective

  Level: developer

.keywords: matrix, register, destroy

.seealso: MatPartitioningRegisterDynamic(), MatPartitioningRegisterAll()
@*/
PetscErrorCode  MatPartitioningRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatPartitioningRegisterAllCalled = PETSC_FALSE;
  ierr = PetscFListDestroy(&MatPartitioningList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningGetType"
/*@C
   MatPartitioningGetType - Gets the Partitioning method type and name (as a string) 
        from the partitioning context.

   Not collective

   Input Parameter:
.  partitioning - the partitioning context

   Output Parameter:
.  type - partitioner type

   Level: intermediate

   Not Collective

.keywords: Partitioning, get, method, name, type
@*/
PetscErrorCode  MatPartitioningGetType(MatPartitioning partitioning,const MatPartitioningType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(partitioning,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)partitioning)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetNParts"
/*@C
   MatPartitioningSetNParts - Set how many partitions need to be created;
        by default this is one per processor. Certain partitioning schemes may
        in fact only support that option.

   Not collective

   Input Parameter:
.  partitioning - the partitioning context
.  n - the number of partitions

   Level: intermediate

   Not Collective

.keywords: Partitioning, set

.seealso: MatPartitioningCreate(), MatPartitioningApply()
@*/
PetscErrorCode  MatPartitioningSetNParts(MatPartitioning part,PetscInt n)
{
  PetscFunctionBegin;
  part->n = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningApply" 
/*@
   MatPartitioningApply - Gets a partitioning for a matrix.

   Collective on Mat

   Input Parameters:
.  matp - the matrix partitioning object

   Output Parameters:
.   partitioning - the partitioning. For each local node this tells the processor
                   number that that node is assigned to.

   Options Database Keys:
   To specify the partitioning through the options database, use one of
   the following 
$    -mat_partitioning_type parmetis, -mat_partitioning current
   To see the partitioning result
$    -mat_partitioning_view

   Level: beginner

   The user can define additional partitionings; see MatPartitioningRegisterDynamic().

.keywords: matrix, get, partitioning

.seealso:  MatPartitioningRegisterDynamic(), MatPartitioningCreate(),
           MatPartitioningDestroy(), MatPartitioningSetAdjacency(), ISPartitioningToNumbering(),
           ISPartitioningCount()
@*/
PetscErrorCode  MatPartitioningApply(MatPartitioning matp,IS *partitioning)
{
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(partitioning,2);
  if (!matp->adj->assembled) SETERRQ(((PetscObject)matp)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (matp->adj->factortype) SETERRQ(((PetscObject)matp)->comm,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  if (!matp->ops->apply) SETERRQ(((PetscObject)matp)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set type with MatPartitioningSetFromOptions() or MatPartitioningSetType()");
  ierr = PetscLogEventBegin(MAT_Partitioning,matp,0,0,0);CHKERRQ(ierr);
  ierr = (*matp->ops->apply)(matp,partitioning);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Partitioning,matp,0,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_partitioning_view",&flag,PETSC_NULL);CHKERRQ(ierr);
  if (flag) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(((PetscObject)matp)->comm,&viewer);CHKERRQ(ierr);
    ierr = MatPartitioningView(matp,viewer);CHKERRQ(ierr);
    ierr = ISView(*partitioning,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetAdjacency"
/*@
   MatPartitioningSetAdjacency - Sets the adjacency graph (matrix) of the thing to be
      partitioned.

   Collective on MatPartitioning and Mat

   Input Parameters:
+  part - the partitioning context
-  adj - the adjacency matrix

   Level: beginner

.keywords: Partitioning, adjacency

.seealso: MatPartitioningCreate()
@*/
PetscErrorCode  MatPartitioningSetAdjacency(MatPartitioning part,Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidHeaderSpecific(adj,MAT_CLASSID,2);
  part->adj = adj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningDestroy"
/*@
   MatPartitioningDestroy - Destroys the partitioning context.

   Collective on Partitioning

   Input Parameters:
.  part - the partitioning context

   Level: beginner

.keywords: Partitioning, destroy, context

.seealso: MatPartitioningCreate()
@*/
PetscErrorCode  MatPartitioningDestroy(MatPartitioning *part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*part),MAT_PARTITIONING_CLASSID,1);
  if (--((PetscObject)(*part))->refct > 0) {*part = 0; PetscFunctionReturn(0);}

  if ((*part)->ops->destroy) {
    ierr = (*(*part)->ops->destroy)((*part));CHKERRQ(ierr);
  }
  ierr = PetscFree((*part)->vertex_weights);CHKERRQ(ierr);
  ierr = PetscFree((*part)->part_weights);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetVertexWeights"
/*@C
   MatPartitioningSetVertexWeights - Sets the weights for vertices for a partitioning.

   Logically Collective on Partitioning

   Input Parameters:
+  part - the partitioning context
-  weights - the weights, on each process this array must have the same size as the number of local rows

   Level: beginner

   Notes:
      The array weights is freed by PETSc so the user should not free the array. In C/C++
   the array must be obtained with a call to PetscMalloc(), not malloc().

.keywords: Partitioning, destroy, context

.seealso: MatPartitioningCreate(), MatPartitioningSetType(), MatPartitioningSetPartitionWeights()
@*/
PetscErrorCode  MatPartitioningSetVertexWeights(MatPartitioning part,const PetscInt weights[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);

  ierr = PetscFree(part->vertex_weights);CHKERRQ(ierr);
  part->vertex_weights = (PetscInt*)weights;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetPartitionWeights"
/*@C
   MatPartitioningSetPartitionWeights - Sets the weights for each partition.

   Logically Collective on Partitioning

   Input Parameters:
+  part - the partitioning context
-  weights - An array of size nparts that is used to specify the fraction of
             vertex weight that should be distributed to each sub-domain for
             the balance constraint. If all of the sub-domains are to be of
             the same size, then each of the nparts elements should be set
             to a value of 1/nparts. Note that the sum of all of the weights
             should be one.

   Level: beginner

   Notes:
      The array weights is freed by PETSc so the user should not free the array. In C/C++
   the array must be obtained with a call to PetscMalloc(), not malloc().

.keywords: Partitioning, destroy, context

.seealso: MatPartitioningCreate(), MatPartitioningSetType(), MatPartitioningSetVertexWeights()
@*/
PetscErrorCode  MatPartitioningSetPartitionWeights(MatPartitioning part,const PetscReal weights[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);

  ierr = PetscFree(part->part_weights);CHKERRQ(ierr);
  part->part_weights = (PetscReal*)weights;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningCreate"
/*@
   MatPartitioningCreate - Creates a partitioning context.

   Collective on MPI_Comm

   Input Parameter:
.   comm - MPI communicator 

   Output Parameter:
.  newp - location to put the context

   Level: beginner

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningApply(), MatPartitioningDestroy(),
          MatPartitioningSetAdjacency()

@*/
PetscErrorCode  MatPartitioningCreate(MPI_Comm comm,MatPartitioning *newp)
{
  MatPartitioning part;
  PetscErrorCode  ierr;
  PetscMPIInt     size;

  PetscFunctionBegin;
  *newp          = 0;

#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = MatInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(part,_p_MatPartitioning,struct _MatPartitioningOps,MAT_PARTITIONING_CLASSID,-1,"MatPartitioning","Matrix/graph partitioning","MatOrderings",comm,MatPartitioningDestroy,
                    MatPartitioningView);CHKERRQ(ierr);
  part->vertex_weights = PETSC_NULL;
  part->part_weights   = PETSC_NULL;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  part->n = (PetscInt)size;

  *newp = part;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningView"
/*@C 
   MatPartitioningView - Prints the partitioning data structure.

   Collective on MatPartitioning

   Input Parameters:
.  part - the partitioning context
.  viewer - optional visualization context

   Level: intermediate

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open alternative visualization contexts with
.     PetscViewerASCIIOpen() - output to a specified file

.keywords: Partitioning, view

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  MatPartitioningView(MatPartitioning part,PetscViewer viewer)
{
  PetscErrorCode            ierr;
  PetscBool                 iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)part)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(part,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)part,viewer,"MatPartitioning Object");CHKERRQ(ierr);
    if (part->vertex_weights) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using vertex weights\n");CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this MatParitioning",((PetscObject)viewer)->type_name);
  }

  if (part->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*part->ops->view)(part,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetType"
/*@C
   MatPartitioningSetType - Sets the type of partitioner to use

   Collective on MatPartitioning

   Input Parameter:
.  part - the partitioning context.
.  type - a known method

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

   Level: intermediate

.keywords: partitioning, set, method, type

.seealso: MatPartitioningCreate(), MatPartitioningApply(), MatPartitioningType

@*/
PetscErrorCode  MatPartitioningSetType(MatPartitioning part,const MatPartitioningType type)
{
  PetscErrorCode ierr,(*r)(MatPartitioning);
  PetscBool  match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)part,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (part->setupcalled) {
    ierr =  (*part->ops->destroy)(part);CHKERRQ(ierr);
    part->ops->destroy = PETSC_NULL;
    part->data        = 0;
    part->setupcalled = 0;
  }

  ierr =  PetscFListFind(MatPartitioningList,((PetscObject)part)->comm,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(((PetscObject)part)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown partitioning type %s",type);

  part->ops->destroy      = (PetscErrorCode (*)(MatPartitioning)) 0;
  part->ops->view         = (PetscErrorCode (*)(MatPartitioning,PetscViewer)) 0;
  ierr = (*r)(part);CHKERRQ(ierr);

  ierr = PetscFree(((PetscObject)part)->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&((PetscObject)part)->type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPartitioningSetFromOptions"
/*@
   MatPartitioningSetFromOptions - Sets various partitioning options from the 
        options database.

   Collective on MatPartitioning

   Input Parameter:
.  part - the partitioning context.

   Options Database Command:
$  -mat_partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

   Level: beginner

.keywords: partitioning, set, method, type
@*/
PetscErrorCode  MatPartitioningSetFromOptions(MatPartitioning part)
{
  PetscErrorCode ierr;
  PetscBool  flag;
  char       type[256];
  const char *def;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)part);CHKERRQ(ierr);
    if (!((PetscObject)part)->type_name) {
#if defined(PETSC_HAVE_PARMETIS)
      def = MATPARTITIONINGPARMETIS;
#else
      def = MATPARTITIONINGCURRENT;
#endif
    } else {
      def = ((PetscObject)part)->type_name;
    }
    ierr = PetscOptionsList("-mat_partitioning_type","Type of partitioner","MatPartitioningSetType",MatPartitioningList,def,type,256,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatPartitioningSetType(part,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)part)->type_name) {
      ierr = MatPartitioningSetType(part,def);CHKERRQ(ierr);
    }

    if (part->ops->setfromoptions) {
      ierr = (*part->ops->setfromoptions)(part);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






