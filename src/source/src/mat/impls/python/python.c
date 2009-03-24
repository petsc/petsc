/* -------------------------------------------------------------------------- */

#include "private/matimpl.h"
#include "src/inline/python.h"

/* -------------------------------------------------------------------------- */

#define MATPYTHON "python"

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetContext(Mat,void*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonGetContext(Mat,void**);
PETSC_EXTERN_CXX_END

/* -------------------------------------------------------------------------- */

typedef struct {
  PyObject *self;
  char     *pyname;
  PetscTruth  scale,shift;
  PetscScalar vscale,vshift;
} Mat_Py;

/* -------------------------------------------------------------------------- */

#define Mat_Py_Self(mat) (((Mat_Py*)(mat)->data)->self)

#define MAT_PYTHON_CALL_HEAD(mat, PyMethod)	\
  PETSC_PYTHON_CALL_HEAD(Mat_Py_Self(mat), PyMethod)
#define MAT_PYTHON_CALL_JUMP(mat, LABEL)	\
  PETSC_PYTHON_CALL_JUMP(LABEL)
#define MAT_PYTHON_CALL_BODY(mat, ARGS)		\
  PETSC_PYTHON_CALL_BODY(ARGS)
#define MAT_PYTHON_CALL_TAIL(mat, PyMethod)	\
  PETSC_PYTHON_CALL_TAIL()

#define MAT_PYTHON_CALL(mat, PyMethod, ARGS)	\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);		\
  MAT_PYTHON_CALL_BODY(mat, ARGS);		\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)		\
/**/

#define MAT_PYTHON_CALL_MAYBE(mat, PyMethod, ARGS, LABEL)	\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);				\
  MAT_PYTHON_CALL_JUMP(mat, LABEL);				\
  MAT_PYTHON_CALL_BODY(mat, ARGS);				\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)				\
/**/

#define MAT_PYTHON_CALL_NOARGS(mat, PyMethod)				\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);					\
  MAT_PYTHON_CALL_BODY(mat, ("", NULL));				\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)					\
/**/

#define MAT_PYTHON_MATARG(mat) ("O&",PyPetscMat_New,mat)

#define MAT_PYTHON_CALL_MATARG(mat, PyMethod)				\
  MAT_PYTHON_CALL_HEAD(mat, PyMethod);					\
  MAT_PYTHON_CALL_BODY(mat, ("O&", PyPetscMat_New, mat));		\
  MAT_PYTHON_CALL_TAIL(mat, PyMethod)					\
/**/

#define MAT_PYTHON_SETERRSUP(mat, PyMethod)				\
  SETERRQ1(PETSC_ERR_SUP,"method %s() not implemented",PyMethod);	\
  PetscFunctionReturn(PETSC_ERR_SUP)					\
/**/

/* -------------------------------------------------------------------------- */
#if 0
static int MatPythonHasOperation(Mat mat, const char operation[])
{
  Mat_Py   *py   = (Mat_Py *) mat->data;
  PyObject *attr = NULL;
  if (py->self == NULL || py->self == Py_None) return 0;
  attr = PyObject_GetAttrString(py->self, operation);
  if      (attr == NULL)    { PyErr_Clear();   return 0; }
  else if (attr == Py_None) { Py_DecRef(attr); return 0; }
  else                      { Py_DecRef(attr); return 1; }
  return 1;
}

#undef __FUNCT__  
#define __FUNCT__ "MatPythonFillOperations"
static PetscErrorCode MatPythonFillOperations(Mat mat)
{
  PetscFunctionBegin;
  if (MatPythonHasOperation(mat, "multTranspose"))
    mat->ops->multtranspose = MatMultTransopse_Python;
  PetscFunctionReturn(0);
}
#endif
/* -------------------------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatPythonSetType_PYTHON"
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetType_PYTHON(Mat mat,const char pyname[])
{
  PyObject       *self = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* create the Python object from module/class/function  */
  ierr = PetscCreatePythonObject(pyname,&self);CHKERRQ(ierr);
  /* set the created Python object in Mat context */
  ierr = MatPythonSetContext(mat,self);Py_DecRef(self);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Python"
static PetscErrorCode MatDestroy_Python(Mat mat)
{
  Mat_Py         *py   = (Mat_Py *)mat->data;
  PyObject       *self = py->self;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (Py_IsInitialized()) {
    MAT_PYTHON_CALL_NOARGS(mat, "destroy");
    py->self = NULL; Py_DecRef(self);
  }
  ierr = PetscStrfree(py->pyname);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatPythonSetType_C",
				    "",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatSetFromOptions_Python"
static PetscErrorCode MatSetFromOptions_Python(Mat mat)
{
  Mat_Py         *py = (Mat_Py*)mat->data;
  char           pyname[2*PETSC_MAX_PATH_LEN+3];
  PetscTruth     flg = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Mat Python options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mat_python_type","Python [package.]module[.{class|function}]",
			    "MatPythonSetType",py->pyname,pyname,sizeof(pyname),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg && pyname[0]) {
    ierr = PetscStrcmp(py->pyname,pyname,&flg);CHKERRQ(ierr);
    if (!flg) { ierr = MatPythonSetType_PYTHON(mat,pyname);CHKERRQ(ierr); }
  }
  MAT_PYTHON_CALL_MATARG(mat, "setFromOptions");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_Python"
static PetscErrorCode MatView_Python(Mat mat,PetscViewer viewer)
{
  Mat_Py         *py = (Mat_Py*)mat->data;
  PetscTruth     isascii,isstring;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    const char* pyname = py->pyname  ? py->pyname  : "no yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  Python: %s\n", pyname);CHKERRQ(ierr);
  }
  if (isstring) {
    const char* pyname = py->pyname  ? py->pyname  : "<unknown>";
    ierr = PetscViewerStringSPrintf(viewer,"%s",pyname);CHKERRQ(ierr);
  }
  MAT_PYTHON_CALL(mat, "view", ("O&O&",
				PyPetscMat_New,     mat,
				PyPetscViewer_New,  viewer));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_Python"
static PetscErrorCode MatSetOption_Python(Mat mat,MatOption op,PetscTruth flag)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL(mat, "setOption", ("O&ii",
				     PyPetscMat_New, mat,
				     (int)           op,
				     (int)           flag));
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_(2,3,3) || PETSC_VERSION_(2,3,2)
static PetscErrorCode MatSetOption_Python_old(Mat mat,MatOption op)
{ return MatSetOption_Python(mat,op,PETSC_TRUE); }
#define MatSetOption_Python MatSetOption_Python_old
#endif

#if PETSC_VERSION_(2,3,2)
#define PetscGetMap(o, m) (&(o)->m)
#define PetscSetUpMap(o, m) PetscMapInitialize((o)->comm,&(o)->m)
#elif PETSC_VERSION_(2,3,3)
#define PetscGetMap(o, m) (&(o)->m)
#define PetscSetUpMap(o, m) PetscMapSetUp(&(o)->m)
#else
#define PetscGetMap(o, m) ((o)->m)
#define PetscSetUpMap(o, m) PetscMapSetUp((o)->m)
#endif

#if PETSC_VERSION_(2,3,2)
#define PetscMapSetBlockSize(map,bs) ((map)->bs=(bs), 0)
#endif

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_Python"
static PetscErrorCode MatSetBlockSize_Python(Mat mat, PetscInt bs)
{
  PetscMap       *rmap = PetscGetMap(mat,rmap);
  PetscMap       *cmap = PetscGetMap(mat,cmap);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscMapSetBlockSize(rmap,bs);CHKERRQ(ierr);
  ierr = PetscMapSetBlockSize(cmap,bs);CHKERRQ(ierr);
  ierr = PetscSetUpMap(mat,rmap);CHKERRQ(ierr);
  ierr = PetscSetUpMap(mat,cmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatSetUpPreallocation_Python"
static PetscErrorCode MatSetUpPreallocation_Python(Mat mat)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscMap       *rmap = PetscGetMap(mat,rmap);
  PetscMap       *cmap = PetscGetMap(mat,cmap);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* MatDestroy() calls MatPreallocated() !!! */
  if (!Py_IsInitialized()) PetscFunctionReturn(0);
  /* setup row and columns maps */
  if (rmap->bs == -1) rmap->bs = 1;
  if (cmap->bs == -1) cmap->bs = 1;
  ierr = PetscSetUpMap(mat,rmap);CHKERRQ(ierr);
  ierr = PetscSetUpMap(mat,cmap);CHKERRQ(ierr);
  /* try to load Python code if not yet done */
  if (py->self == NULL || py->self == Py_None) {
    char       pyname[2*PETSC_MAX_PATH_LEN+3];
    PetscTruth flag = PETSC_FALSE;
    ierr = PetscOptionsGetString(((PetscObject)mat)->prefix,"-mat_python_type",
				 pyname,sizeof(pyname),&flag);CHKERRQ(ierr);
    if (flag && pyname[0]==0) flag = PETSC_FALSE;
    if (flag) { ierr = MatPythonSetType_PYTHON(mat,pyname);CHKERRQ(ierr); }
  }
  if (!py->self) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Python context not set, call one of \n"
	    " * MatPythonSetType(mat,\"[package.]module.class\")\n"
	    " * MatSetFromOptions(mat) and pass option -mat_python_type [package.]module.class");
  }
  /* */
  MAT_PYTHON_CALL_MATARG(mat, "setUp");
  mat->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_Python"
static PetscErrorCode MatZeroEntries_Python(Mat mat)
{
  Mat_Py *py = (Mat_Py *) mat->data;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "zeroEntries", 
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  py->scale = PETSC_FALSE; py->vscale = 1;
  py->shift = PETSC_FALSE; py->vshift = 0;
  PetscFunctionReturn(0);
 notimplemented: /* MatZeroEntries */
  MAT_PYTHON_SETERRSUP(mat, "zeroEntries");
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_Python"
static PetscErrorCode MatScale_Python(Mat mat,PetscScalar a)
{
  Mat_Py *py = (Mat_Py *) mat->data;
#if defined(PETSC_USE_COMPLEX)
  Py_complex ca;
#endif
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ca.real = PetscRealPart(a);
  ca.imag = PetscImaginaryPart(a);
#endif
#if defined(PETSC_USE_COMPLEX)
  MAT_PYTHON_CALL_MAYBE(mat, "scale", ("O&D",
				       PyPetscMat_New, mat,
				       ca),
			scale);
#else
  MAT_PYTHON_CALL_MAYBE(mat, "scale", ("O&d",
				       PyPetscMat_New, mat,
				       (double)        a),
			scale);
#endif
  py->scale  = PETSC_FALSE;
  py->vscale = 1;
  PetscFunctionReturn(0);
 scale: /* MatScale */
  py->scale  = PETSC_TRUE;
  py->vscale *= a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatShift_Python"
static PetscErrorCode MatShift_Python(Mat mat,PetscScalar a)
{
  Mat_Py *py = (Mat_Py *) mat->data;
#if defined(PETSC_USE_COMPLEX)
  Py_complex ca;
#endif
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ca.real = PetscRealPart(a);
  ca.imag = PetscImaginaryPart(a);
#endif
#if defined(PETSC_USE_COMPLEX)
  MAT_PYTHON_CALL_MAYBE(mat, "shift", ("O&D",
				       PyPetscMat_New, mat,
				       ca),
			shift);
#else
  MAT_PYTHON_CALL_MAYBE(mat, "shift", ("O&d",
				       PyPetscMat_New, mat,
				       (double)        a),
			shift);
#endif
  py->shift  = PETSC_FALSE;
  py->vshift = 0;
  PetscFunctionReturn(0);
 shift: /* MatShift */
  py->shift  = PETSC_TRUE;
  py->vshift += a;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_Python"
static PetscErrorCode MatAssemblyBegin_Python(Mat mat,MatAssemblyType type)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL(mat, "assemblyBegin", ("O&i",
					 PyPetscMat_New, mat,
					 (int)           type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Python"
static PetscErrorCode MatAssemblyEnd_Python(Mat mat,MatAssemblyType type)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscFunctionBegin;
  MAT_PYTHON_CALL(mat, "assemblyEnd", ("O&i",
				       PyPetscMat_New, mat,
				       (int)           type));
  if (type == MAT_FINAL_ASSEMBLY) {
    py->scale = PETSC_FALSE; py->vscale = 1;
    py->shift = PETSC_FALSE; py->vshift = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Python"
static PetscErrorCode MatMult_Python(Mat mat,Vec x,Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "mult", ("O&O&O&",
				      PyPetscMat_New, mat,
				      PyPetscVec_New, x,
				      PyPetscVec_New, y),
			notimplemented);
  /* shift and scale */
  if (py->shift && py->scale) {
    ierr = VecAXPBY(y,py->vshift,py->vscale,x);CHKERRQ(ierr);
  } else if (py->scale) {
    ierr = VecScale(y,py->vscale);CHKERRQ(ierr);
  } else if (py->shift) {
    ierr = VecAXPY(y,py->vshift,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
 notimplemented: /* MatMult */
  MAT_PYTHON_SETERRSUP(mat, "mult");
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Python"
static PetscErrorCode MatMultAdd_Python(Mat mat,Vec x,Vec v,Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto multadd;
  MAT_PYTHON_CALL_MAYBE(mat, "multAdd", ("O&O&O&O&",
					 PyPetscMat_New, mat,
					 PyPetscVec_New, x,
					 PyPetscVec_New, v,
					 PyPetscVec_New, y),
			multadd);
  PetscFunctionReturn(0);
 multadd: /* MatMultAdd */
  ierr = MatMult_Python(mat,x,y);CHKERRQ(ierr);
  ierr = VecAXPY(y, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Python"
static PetscErrorCode MatMultTranspose_Python(Mat mat,Vec x, Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscTruth     symmset,symmknown;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "multTranspose", ("O&O&O&",
					       PyPetscMat_New, mat,
					       PyPetscVec_New, x,
					       PyPetscVec_New, y),
			mult);
  /* shift and scale */
  if (py->shift && py->scale) {
    ierr = VecAXPBY(y,py->vshift,py->vscale,x);CHKERRQ(ierr);
  } else if (py->scale) {
    ierr = VecScale(y,py->vscale);CHKERRQ(ierr);
  } else if (py->shift) {
    ierr = VecAXPY(y,py->vshift,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
 mult:
  ierr = MatIsSymmetricKnown(mat,&symmset,&symmknown);CHKERRQ(ierr);
  if (!symmset || !symmknown) goto notimplemented;
  ierr = MatMult_Python(mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 notimplemented: /* MatMultTranspose */
  MAT_PYTHON_SETERRSUP(mat, "multTranspose");
}


#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Python"
static PetscErrorCode MatMultTransposeAdd_Python(Mat mat,Vec x,Vec v,Vec y)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto multtransposeadd;
  MAT_PYTHON_CALL_MAYBE(mat, "multTransposeAdd", ("O&O&O&O&",
						  PyPetscMat_New, mat,
						  PyPetscVec_New, x,
						  PyPetscVec_New, v,
						  PyPetscVec_New, y),
			multtransposeadd);
  PetscFunctionReturn(0);
 multtransposeadd: /* MatMultTransposeAdd */
  ierr = MatMultTranspose_Python(mat,x,y);CHKERRQ(ierr);
  ierr = VecAXPY(y, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_Python"
static PetscErrorCode MatSolve_Python(Mat mat,Vec b,Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscScalar    one = 1;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*  shift */
  if (py->shift) goto notimplemented;
  MAT_PYTHON_CALL_MAYBE(mat, "solve", ("O&O&O&",
				       PyPetscMat_New, mat,
				       PyPetscVec_New, b,
				       PyPetscVec_New, x),
			notimplemented);
  /*  scale */
  if (py->scale) { ierr = VecScale(x,one/py->vscale);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
 notimplemented: /* MatSolve */
  MAT_PYTHON_SETERRSUP(mat, "solve");
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveAdd_Python"
static PetscErrorCode MatSolveAdd_Python(Mat mat,Vec b,Vec v,Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto solveadd;
  MAT_PYTHON_CALL_MAYBE(mat, "solveAdd", ("O&O&O&O&",
					  PyPetscMat_New, mat,
					  PyPetscVec_New, b,
					  PyPetscVec_New, v,
					  PyPetscVec_New, x),
			solveadd);
  PetscFunctionReturn(0);
 solveadd: /* MatSolveAdd */
  ierr = MatSolve_Python(mat,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_Python"
static PetscErrorCode MatSolveTranspose_Python(Mat mat,Vec b, Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscScalar    one = 1;
  PetscTruth     symmset,symmknown;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /*  shift */
  if (py->shift) goto notimplemented;
  MAT_PYTHON_CALL_MAYBE(mat, "solveTranspose", ("O&O&O&",
						PyPetscMat_New, mat,
						PyPetscVec_New, b,
						PyPetscVec_New, x),
			solve);
  /* scale */
  if (py->scale) { ierr = VecScale(x,one/py->vscale);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
 solve:
  ierr = MatIsSymmetricKnown(mat,&symmset,&symmknown);CHKERRQ(ierr);
  if (!symmset || !symmknown) goto notimplemented;
  ierr = MatSolve_Python(mat,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 notimplemented: /* MatSolveTranspose */
  MAT_PYTHON_SETERRSUP(mat, "solveTranspose");
}


#undef __FUNCT__
#define __FUNCT__ "MatSolveTransposeAdd_Python"
static PetscErrorCode MatSolveTransposeAdd_Python(Mat mat,Vec b,Vec v,Vec x)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (py->scale || py->shift) goto solvetransposeadd;
  MAT_PYTHON_CALL_MAYBE(mat, "solveTransposeAdd", ("O&O&O&O&",
						   PyPetscMat_New, mat,
						   PyPetscVec_New, b,
						   PyPetscVec_New, v,
						   PyPetscVec_New, x),
			solvetransposeadd);
  PetscFunctionReturn(0);
 solvetransposeadd: /* MatSolveTransposeAdd */
  ierr = MatSolveTranspose_Python(mat,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x, 1, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Python"
static PetscErrorCode MatGetDiagonal_Python(Mat mat,Vec v)
{
  Mat_Py         *py = (Mat_Py *) mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "getDiagonal", ("O&O&",
					     PyPetscMat_New, mat,
					     PyPetscVec_New, v),
			notimplemented);
  if (py->scale) { ierr = VecScale(v,py->vscale);CHKERRQ(ierr); }
  if (py->shift) { ierr = VecShift(v,py->vshift);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
 notimplemented: /* MatGetDiagonal */
  MAT_PYTHON_SETERRSUP(mat, "getDiagonal");
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Python"
static PetscErrorCode MatDiagonalSet_Python(Mat mat,Vec v,InsertMode im)
{
  Mat_Py *py = (Mat_Py *) mat->data;
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "setDiagonal", ("O&O&i",
					     PyPetscMat_New, mat,
					     PyPetscVec_New, v,
					     (int)           im),
			notimplemented);
  if (im != ADD_VALUES) py->shift = PETSC_FALSE; py->vshift = 0;
  PetscFunctionReturn(0);
 notimplemented:  /* MatDiagonalSet */
  MAT_PYTHON_SETERRSUP(mat, "setDiagonal");
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_Python"
static PetscErrorCode MatDiagonalScale_Python(Mat mat,Vec l, Vec r)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "diagonalScale", ("O&O&O&",
					       PyPetscMat_New, mat,
					       PyPetscVec_New, l,
					       PyPetscVec_New, r),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatDiagonalScale */
  MAT_PYTHON_SETERRSUP(mat, "diagonalScale");
}

#undef  __FUNCT__
#define __FUNCT__ "MatRealPart_Python"
static PetscErrorCode MatRealPart_Python(Mat mat)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "realPart",
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatRealPart */
  MAT_PYTHON_SETERRSUP(mat, "realPart");
}

#undef  __FUNCT__
#define __FUNCT__ "MatImaginaryPart_Python"
static PetscErrorCode MatImaginaryPart_Python(Mat mat)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "imagPart", 
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatImaginaryPart */
  MAT_PYTHON_SETERRSUP(mat, "imagPart");
}

#undef  __FUNCT__
#define __FUNCT__ "MatImaginaryPart_Python"
static PetscErrorCode MatConjugate_Python(Mat mat)
{
  PetscFunctionBegin;
  MAT_PYTHON_CALL_MAYBE(mat, "conjugate", 
			MAT_PYTHON_MATARG(mat),
			notimplemented);
  PetscFunctionReturn(0);
 notimplemented: /* MatConjugate */
  MAT_PYTHON_SETERRSUP(mat, "conjugate");
}

/* -------------------------------------------------------------------------- */

#if PETSC_VERSION_(2,3,3) || PETSC_VERSION_(2,3,2)
#define MAT_FACTOR_NONE 0
#endif

/*MC
   MATPYTHON - .

   Level: intermediate

   Contributed by Lisandro Dalcin <dalcinl at gmail dot com>

.seealso:  Mat, MatCreate(), MatSetType(), MatType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_Python"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_Python(Mat mat)
{
  Mat_Py      *py;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = Petsc4PyInitialize();CHKERRQ(ierr);

  ierr = PetscNew(Mat_Py,&py);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(mat,sizeof(Mat_Py));CHKERRQ(ierr);
  mat->data  = (void*)py;

  /* Python */
  py->self      = NULL;
  py->pyname  = NULL;
  py->scale     = PETSC_FALSE;
  py->vscale    = 1;
  py->shift     = PETSC_FALSE;
  py->vshift    = 0;

  /* PETSc */
  mat->ops->destroy              = MatDestroy_Python;
  mat->ops->view                 = MatView_Python;
  mat->ops->setfromoptions       = MatSetFromOptions_Python;

  mat->ops->setoption            = MatSetOption_Python;
  mat->ops->setblocksize         = MatSetBlockSize_Python;
  mat->ops->setuppreallocation   = MatSetUpPreallocation_Python;

  mat->ops->zeroentries          = MatZeroEntries_Python;
  mat->ops->scale                = MatScale_Python;
  mat->ops->shift                = MatShift_Python;
  mat->ops->assemblybegin        = MatAssemblyBegin_Python;
  mat->ops->assemblyend          = MatAssemblyEnd_Python;

  mat->ops->mult                 = MatMult_Python;
  mat->ops->multtranspose        = MatMultTranspose_Python;
  mat->ops->multadd              = MatMultAdd_Python;
  mat->ops->multtransposeadd     = MatMultTransposeAdd_Python;

  mat->ops->solve                = MatSolve_Python;
  mat->ops->solvetranspose       = MatSolveTranspose_Python;
  mat->ops->solveadd             = MatSolveAdd_Python;
  mat->ops->solvetransposeadd    = MatSolveTransposeAdd_Python;

  mat->ops->getdiagonal          = MatGetDiagonal_Python;
  mat->ops->diagonalset          = MatDiagonalSet_Python;
  mat->ops->diagonalscale        = MatDiagonalScale_Python;

  mat->ops->realpart             = MatRealPart_Python;
  mat->ops->imaginarypart        = MatImaginaryPart_Python;
  mat->ops->conjugate            = MatConjugate_Python;

  mat->factor       = MAT_FACTOR_NONE;
  mat->assembled    = PETSC_TRUE;
  mat->preallocated = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)mat,MATPYTHON);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,
				    "MatPythonSetType_C","MatPythonSetType_PYTHON",
				    (PetscVoidFunction)MatPythonSetType_PYTHON);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatPythonGetContext"
/*@
   MatPythonGetContext - .

   Input Parameter:
.  mat - Mat object

   Output Parameter:
.  ctx - Python context

   Level: beginner

.keywords: Mat, matrix, create

.seealso: Mat, MatCreate(), MAtSetType(), MATPYTHON
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonGetContext(Mat mat,void **ctx)
{
  Mat_Py         *py;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(ctx,2);
  *ctx = NULL;
  ierr = PetscTypeCompare((PetscObject)mat,MATPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (Mat_Py *) mat->data;
  *ctx = (void *) py->self;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPythonSetContext"
/*@
   MatPythonSetContext - .

   Collective on PC

   Input Parameters:
.  pc - PC context
.  ctx - Python context

   Level: beginner

.keywords: Mat, matrix, create

.seealso: Mat, MatCreate(), MAtSetType(), MATPYTHON
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetContext(Mat mat,void *ctx)
{
  Mat_Py          *py;
  PyObject       *old, *self = (PyObject *) ctx;
  PetscTruth     ispython;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (ctx) PetscValidPointer(ctx,2);
  ierr = PetscTypeCompare((PetscObject)mat,MATPYTHON,&ispython);CHKERRQ(ierr);
  if (!ispython) PetscFunctionReturn(0);
  py = (Mat_Py *) mat->data;
  /* do nothing if contexts are the same */
  if (self == Py_None) self = NULL;
  if (py->self == self) PetscFunctionReturn(0);
  /* del previous Python context in the Mat object */
  MAT_PYTHON_CALL_NOARGS(mat, "destroy");
  old = py->self; py->self = NULL; Py_DecRef(old);
  /* set current Python context in the Mat object  */
  py->self = self; Py_IncRef(py->self);
  ierr = PetscStrfree(py->pyname);CHKERRQ(ierr);
  ierr = PetscPythonGetFullName(py->self,&py->pyname);CHKERRQ(ierr);
  MAT_PYTHON_CALL_MATARG(mat, "create");
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#if PETSC_VERSION_(2,3,3) || PETSC_VERSION_(2,3,2)

PETSC_EXTERN_CXX_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetType(Mat,const char[]);
PETSC_EXTERN_CXX_END

#undef __FUNCT__
#define __FUNCT__ "MatPythonSetType"
/*@C
   MatPythonSetType - Initalize a Mat object implemented in Python.

   Collective on Mat

   Input Parameter:
+  mat - the matrix (Mat) object.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -mat_python_type <pyname>

   Level: intermediate

.keywords: Mat, Python

.seealso: MATPYTHON, MatCreatePython(), PetscPythonInitialize()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPythonSetType(Mat mat,const char pyname[])
{
  PetscErrorCode (*f)(Mat, const char[]) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidCharPointer(pyname,2);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatPythonSetType_C",
				  (PetscVoidFunction*)&f);CHKERRQ(ierr);
  if (f) {ierr = (*f)(mat,pyname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#endif

/* -------------------------------------------------------------------------- */
