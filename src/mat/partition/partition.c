
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: partition.c,v 1.10 1998/03/23 21:22:00 bsmith Exp bsmith $";
#endif
 

#include "petsc.h"
#include "src/mat/matimpl.h"               /*I "mat.h" I*/


/*
   Simplest partitioning, keeps the current partitioning.
*/
#undef __FUNC__  
#define __FUNC__ "PartitioningApply_Current" 
static int PartitioningApply_Current(Partitioning part, IS *partitioning)
{
  int   ierr,m,rank,size;

  PetscFunctionBegin;
  MPI_Comm_size(part->comm,&size);
  if (part->n != size) {
    SETERRQ(PETSC_ERR_SUP,1,"Currently only supports one domain per processor");
  }
  MPI_Comm_rank(part->comm,&rank);

  ierr = MatGetLocalSize(part->adj,&m,PETSC_NULL); CHKERRQ(ierr);
  ierr = ISCreateStride(part->comm,m,rank,0,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ "PartitioningCreate_Current" 
int PartitioningCreate_Current(Partitioning part)
{
  PetscFunctionBegin;
  part->apply   = PartitioningApply_Current;
  part->view    = 0;
  part->destroy = 0;
  part->type    = PARTITIONING_CURRENT;
  PetscFunctionReturn(0);
}

/* ===========================================================================================*/

#include "src/sys/nreg.h"
#include "sys.h"

static NRList *__PartitioningList = 0;
int PartitioningRegisterAllCalled = 0;

#undef __FUNC__  
#define __FUNC__ "PartitioningRegister" 
/*@C
   PartitioningRegister - Adds a new sparse matrix partitioning to the 
   matrix package. 

   Input Parameters:
.  name - name of partitioning (for example PARTITIONING_CURRENT) or PARTITIONING_NEW
.  sname -  corresponding string for name
.  order - routine that does partitioning

   Output Parameters:
.  oname - number associated with the partitioning (for example PARTITIONING_CURRENT)

.keywords: matrix, partitioning, register

.seealso: PartitioningRegisterDestroy(), PartitioningRegisterAll()
@*/
int PartitioningRegister(PartitioningType name,PartitioningType *oname,char *sname,int (*part)(Partitioning))
{
  int         ierr;
  static int  numberregistered = 0;

  PetscFunctionBegin;
  if (!__PartitioningList) {
    ierr = NRCreate(&__PartitioningList); CHKERRQ(ierr);
  }

  if (name == PARTITIONING_NEW) {
    name = (PartitioningType) ((int) PARTITIONING_NEW + numberregistered++);
  }
  if (oname) *oname = name;
  ierr = NRRegister(__PartitioningList,(int)name,sname,(int (*)(void*))part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningRegisterDestroy" 
/*@C
   PartitioningRegisterDestroy - Frees the list of partitioning routines.

.keywords: matrix, register, destroy

.seealso: PartitioningRegister(), PartitioningRegisterAll()
@*/
int PartitioningRegisterDestroy(void)
{
  PetscFunctionBegin;
  if (__PartitioningList) {
    NRDestroy( __PartitioningList );
    __PartitioningList = 0;
  }
  PartitioningRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningGetType"
/*@C
   PartitioningGetType - Gets the Partitioning method type and name (as a string) 
        from the partitioning context.

   Input Parameter:
.  Partitioning - the partitioning context

   Output Parameter:
.  meth - partitioner type (or use PETSC_NULL)
.  name - name of partitioner (or use PETSC_NULL)

.keywords: Partitioning, get, method, name, type
@*/
int PartitioningGetType(Partitioning partitioning,PartitioningType *meth,char **name)
{
  int ierr;

  PetscFunctionBegin;
  if (!__PartitioningList) {ierr = PartitioningRegisterAll(); CHKERRQ(ierr);}
  if (meth) *meth = (PartitioningType) partitioning->type;
  if (name)  *name = NRFindName( __PartitioningList, (int)partitioning->type );
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningApply" 
/*@C
   PartitioningApply - Gets a partitioning for a matrix.

   Input Parameters:
.  matp - the matrix partitioning object

   Output Parameters:
.   partitioning - the partitioning

   Options Database Keys:
   To specify the partitioning through the options database, use one of
   the following 
$    -partitioning_type parmetis, -mat_partitioning current
   To see the partitioning result
$    -partitioning_view

   The user can define additional partitionings; see PartitioningRegister().

.keywords: matrix, get, partitioning

.seealso:  PartitioningGetTypeFromOptions(), PartitioningRegister()
@*/
int PartitioningApply(Partitioning matp,IS *partitioning)
{
  int         ierr,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(matp,PARTITIONING_COOKIE);
  if (!matp->adj->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (matp->adj->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  PLogEventBegin(MAT_Partitioning,matp,0,0,0); 
  ierr = (*matp->apply)(matp,partitioning);CHKERRQ(ierr);
  PLogEventEnd(MAT_Partitioning,matp,0,0,0); 

  ierr = OptionsHasName(PETSC_NULL,"-partitioning_view",&flag);
  if (flag) {
    ierr = PartitioningView(matp,VIEWER_STDOUT_(matp->comm));CHKERRQ(ierr);
    ierr = ISView(*partitioning,VIEWER_STDOUT_(matp->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ "PartitioningSetAdjacency"
/*@C
   PartitioningSetAdjacency - Sets the adjacency graph (matrix) of the thing to be
      partitioned.

   Input Parameters:
.  part - the partitioning context
.  adj - the adjacency matrix

.keywords: Partitioning, adjacency

.seealso: PartitioningCreate()
@*/
int PartitioningSetAdjacency(Partitioning part,Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,PARTITIONING_COOKIE);
  PetscValidHeaderSpecific(adj,MAT_COOKIE);
  part->adj = adj;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningDestroy"
/*@C
   PartitioningDestroy - Destroys the partitioning context.

   Input Parameters:
.  part - the partitioning context

.keywords: Partitioning, destroy, context

.seealso: PartitioningCreate()
@*/
int PartitioningDestroy(Partitioning part)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,PARTITIONING_COOKIE);
  if (--part->refct > 0) PetscFunctionReturn(0);

  if (part->destroy) {
    ierr = (*part->destroy)(part);CHKERRQ(ierr);
  }
  PLogObjectDestroy(part);
  PetscHeaderDestroy(part);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningCreate"
/*@C
   PartitioningCreate - Creates a partitioning context.

   Input Parameter:
.   comm - MPI communicator 

   Output Parameter:
.  newp - location to put the context


.keywords: Partitioning, create, context

.seealso: PartitioningSetUp(), PartitioningApply(), PartitioningDestroy()
@*/
int PartitioningCreate(MPI_Comm comm,Partitioning *newp)
{
  Partitioning     part;
  PartitioningType initialtype = PARTITIONING_CURRENT;
  int              ierr;

  PetscFunctionBegin;
  *newp          = 0;

  PetscHeaderCreate(part,_p_Partitioning,int,PARTITIONING_COOKIE,initialtype,comm,PartitioningDestroy,
                    PartitioningView);
  PLogObjectCreate(part);
  part->type               = -1;
  MPI_Comm_size(comm,&part->n);

  /* this violates rule about seperating abstract from implementions*/
  ierr = PartitioningSetType(part,initialtype);CHKERRQ(ierr);
  *newp = part;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningView"
/*@ 
   PartitioningView - Prints the partitioning data structure.

   Input Parameters:
.  part - the partitioning context
.  viewer - optional visualization context

   Note:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: Partitioning, view

.seealso: ViewerFileOpenASCII()
@*/
int PartitioningView(Partitioning  part,Viewer viewer)
{
  ViewerType  vtype;
  int         ierr;
  char        *name;
  FILE        *fd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,PARTITIONING_COOKIE);
  if (viewer) {PetscValidHeader(viewer);} 
  else { viewer = VIEWER_STDOUT_SELF;}

  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    ierr = PartitioningGetType(part,PETSC_NULL,&name); CHKERRQ(ierr);
    PetscFPrintf(part->comm,fd,"Partitioning Object: %s\n",name);
  }

  if (part->view) {
    ierr = (*part->view)(part,viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningPrintHelp"
/*@ 
   PartitioningPrintHelp - Prints all options to the partitioning object.

   Input Parameters:
.  part - the partitioning context

.keywords: Partitioning, help

.seealso: 
@*/
int PartitioningPrintHelp(Partitioning  part)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,PARTITIONING_COOKIE);
  if (!PartitioningRegisterAllCalled) {ierr = PartitioningRegisterAll(); CHKERRQ(ierr);}

  (*PetscHelpPrintf)(part->comm,"Partitioning options ----------------------------------------------\n");
  NRPrintTypes(part->comm,stdout,part->prefix,"partitioning_type",__PartitioningList);

  if (part->printhelp) {
    ierr = (*part->printhelp)(part);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningSetType"
/*@
   PartitioningSetType - Sets the type of partitioner to use

   Input Parameter:
.  part - the partitioning context.
.  type - a known method

   Options Database Command:
$  -partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

.keywords: partitioning, set, method, type
@*/
int PartitioningSetType(Partitioning part,PartitioningType type)
{
  int ierr,(*r)(Partitioning);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,PARTITIONING_COOKIE);
  if (part->type == (int) type) PetscFunctionReturn(0);

  if (part->setupcalled) {
    if (part->destroy) ierr =  (*part->destroy)(part);
    else {if (part->data) PetscFree(part->data);}
    part->data        = 0;
    part->setupcalled = 0;
  }
  /* Get the function pointers for the method requested */
  if (!PartitioningRegisterAllCalled) {ierr = PartitioningRegisterAll(); CHKERRQ(ierr);}
  r =  (int (*)(Partitioning))NRFindRoutine( __PartitioningList, (int)type, (char *)0 );
  if (!r) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown type");}
  if (part->data) PetscFree(part->data);

  part->destroy      = ( int (*)(Partitioning )) 0;
  part->view         = ( int (*)(Partitioning,Viewer) ) 0;
  ierr = (*r)(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PartitioningSetFromOptions"
/*@
   PartitioningSetFromOptions - Sets various partitioing options from the 
        options database.

   Input Parameter:
.  part - the partitioning context.

   Options Database Command:
$  -partitioning_type  <type>
$      Use -help for a list of available methods
$      (for instance, parmetis)

.keywords: partitioning, set, method, type
@*/
int PartitioningSetFromOptions(Partitioning part)
{
  int              ierr,flag;
  PartitioningType type;

  PetscFunctionBegin;

  if (!__PartitioningList) {ierr = PartitioningRegisterAll();CHKERRQ(ierr);}
  ierr = NRGetTypeFromOptions(part->prefix,"-partitioning_type",__PartitioningList,&type,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PartitioningSetType(part,type);CHKERRQ(ierr);
  }
  if (part->setfromoptions) {
    ierr = (*part->setfromoptions)(part);CHKERRQ(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-help",&flag); CHKERRQ(ierr);
  if (flag) {
    ierr = PartitioningPrintHelp(part);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}






