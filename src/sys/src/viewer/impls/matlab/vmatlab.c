/* $Id: matlab.c,v 1.17 2001/08/06 21:14:26 bsmith Exp $ #include "petsc.h" */

#include "src/sys/src/viewer/viewerimpl.h"
#include "mat.h"

typedef struct {
  MATFile               *ep;
  int                   rank;
  PetscViewerMatlabType btype;
} PetscViewer_Matlab;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabPutArray"
/*@C
    PetscViewerMatlabPutArray - Puts an array into the Matlab viewer.

      Not collective: only processor zero saves the array

    Input Parameters:
+    mfile - the viewer
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

     Notes: Only writes array values on processor 0.

@*/
int PetscViewerMatlabPutArray(PetscViewer mfile,int m,int n,PetscScalar *array,char *name)
{
  int                ierr;
  PetscViewer_Matlab *ml = (PetscViewer_Matlab*)mfile->data; 
  mxArray            *mat;
  
  PetscFunctionBegin;  
  if (ml->rank) PetscFunctionReturn(0);
  PetscLogInfo(0,"Putting Matlab array %s\n",name);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  matPutVariable(ml->ep,name,mat);

  PetscLogInfo(0,"Put Matlab array %s\n",name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabPutVariable"
int PetscViewerMatlabPutVariable(PetscViewer viewer,const char* name,void* mat)
{
  PetscFunctionBegin;
  PetscViewer_Matlab *ml = (PetscViewer_Matlab*)viewer->data; 
  matPutVariable(ml->ep,name,(mxArray*)mat);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabGetArray"
/*@C
    PetscViewerMatlabGetArray - Gets a variable from a Matlab viewer into an array

    Not Collective; only processor zero reads in the array

    Input Parameters:
+    mfile - the Matlab file viewer
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

     Notes: Only reads in array values on processor 0.

@*/
int PetscViewerMatlabGetArray(PetscViewer mfile,int m,int n,PetscScalar *array,char *name)
{
  int                ierr;
  PetscViewer_Matlab *ml = (PetscViewer_Matlab*)mfile->data; 
  mxArray            *mat;
  
  PetscFunctionBegin;  
  if (ml->rank) PetscFunctionReturn(0);
  PetscLogInfo(0,"Getting Matlab array %s\n",name);
  mat  = matGetVariable(ml->ep,name);
  if (!mat) SETERRQ1(1,"Unable to get array %s from matlab",name);
  ierr = PetscMemcpy(array,mxGetPr(mat),m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscLogInfo(0,"Got Matlab array %s\n",name);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabSetType_Matlab" 
int PetscViewerMatlabSetType_Matlab(PetscViewer viewer,PetscViewerMatlabType type)
{
  PetscViewer_Matlab *vmatlab = (PetscViewer_Matlab*)viewer->data;

  PetscFunctionBegin;
  vmatlab->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
        Actually opens the file 
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSetFilename_Matlab" 
int PetscViewerSetFilename_Matlab(PetscViewer viewer,const char name[])
{
  PetscViewer_Matlab    *vmatlab = (PetscViewer_Matlab*)viewer->data;
  PetscViewerMatlabType type = vmatlab->btype;

  PetscFunctionBegin;
  if (type == (PetscViewerMatlabType) -1) {
    SETERRQ(1,"Must call PetscViewerMatlabSetType() before PetscViewerSetFilename()");
  }

  /* only first processor opens file */
  if (!vmatlab->rank){
    if (type == PETSC_MATLAB_RDONLY){
      vmatlab->ep = matOpen(name,"r");
    }
    if (type == PETSC_MATLAB_CREATE || type == PETSC_MATLAB_WRONLY) {
      vmatlab->ep = matOpen(name,"w");
    } else {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown file type");
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_Matlab"
int PetscViewerCreate_Matlab(PetscViewer viewer)
{
  int                ierr;
  PetscViewer_Matlab *e;

  PetscFunctionBegin;
  ierr = PetscNew(PetscViewer_Matlab,&e);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(viewer->comm,&e->rank);CHKERRQ(ierr);
  e->btype = (PetscViewerMatlabType)-1;
  viewer->data = (void*) e;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerSetFilename_C",
                                    "PetscViewerSetFilename_Matlab",
                                     PetscViewerSetFilename_Matlab);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerMatlabSetType_C",
                                    "PetscViewerMatlabSetType_Matlab",
                                     PetscViewerMatlabSetType_Matlab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Matlab"
int PetscViewerDestroy_Matlab(PetscViewer v)
{
  PetscViewer_Matlab *vf = (PetscViewer_Matlab*)v->data; 
  PetscFunctionBegin;
  if (vf->ep) matClose(vf->ep);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabOpen" 
/*@C
   PetscViewerMatlabOpen - Opens a Matlab .mat file for input or output.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  name - name of file 
-  type - type of file
$    PETSC_MATLAB_CREATE - create new file for Matlab output
$    PETSC_MATLAB_RDONLY - open existing file for Matlab input
$    PETSC_MATLAB_WRONLY - open existing file for Matlab output

   Output Parameter:
.  binv - PetscViewer for Matlab input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

    For writing files it only opens the file on processor 0 in the communicator.
    For readable files it opens the file on all nodes that have the file. If 
    node 0 does not have the file it generates an error even if other nodes
    do have the file.

   Concepts: Matlab .mat files
   Concepts: PetscViewerMatlab^creating

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad()
@*/
int PetscViewerMatlabOpen(MPI_Comm comm,const char name[],PetscViewerMatlabType type,PetscViewer *binv)
{
  int ierr;
  
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,binv);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*binv,PETSC_VIEWER_MATLAB);CHKERRQ(ierr);
  ierr = PetscViewerMatlabSetType(*binv,type);CHKERRQ(ierr);
  ierr = PetscViewerSetFilename(*binv,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMatlabSetType" 
/*@C
     PetscViewerMatlabSetType - Sets the type of matlab file to be open

    Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; must be a Matlab PetscViewer
-  type - type of file
$    PETSC_MATLAB_CREATE - create new file for matlab output
$    PETSC_MATLAB_RDONLY - open existing file for matlab input
$    PETSC_MATLAB_WRONLY - open existing file for matlab output

  Level: advanced

.seealso: PetscViewerCreate(), PetscViewerSetType(), PetscViewerMatlabOpen()

@*/
int PetscViewerMatlabSetType(PetscViewer viewer,PetscViewerMatlabType type)
{
  int ierr,(*f)(PetscViewer,PetscViewerMatlabType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"PetscViewerMatlabSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static int Petsc_Viewer_Matlab_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_MATLAB_"  
/*@C
     PETSC_VIEWER_MATLAB_ - Creates a Matlab PetscViewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the Matlab PetscViewer
    
     Level: intermediate

   Options Database Keys:
$    -viewer_matlab_filename <name>

   Environmental variables:
-   PETSC_VIEWER_MATLAB_FILENAME

     Notes:
     Unlike almost all other PETSc routines, PETSC_VIEWER_MATLAB_ does not return 
     an error code.  The matlab PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_MATLAB_(comm));

.seealso: PETSC_VIEWER_MATLAB_WORLD, PETSC_VIEWER_MATLAB_SELF, PetscViewerMatlabOpen(), PetscViewerCreate(),
          PetscViewerDestroy()
@*/
PetscViewer PETSC_VIEWER_MATLAB_(MPI_Comm comm)
{
  int         ierr;
  PetscTruth  flg;
  PetscViewer viewer;
  char        fname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (Petsc_Viewer_Matlab_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Matlab_keyval,0);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Matlab_keyval,(void **)&viewer,(int *)&flg);
  if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(comm,"PETSC_VIEWER_MATLAB_FILENAME",fname,PETSC_MAX_PATH_LEN,&flg);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    if (!flg) {
      ierr = PetscStrcpy(fname,"matlaboutput.mat");
      if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    }
    ierr = PetscViewerMatlabOpen(comm,fname,PETSC_MATLAB_CREATE,&viewer); 
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_STDOUT_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Matlab_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_MATLAB_",__FILE__,__SDIR__,1,1," "); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}





