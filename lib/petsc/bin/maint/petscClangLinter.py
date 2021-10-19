#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os,sys,enum
import multiprocessing as mp
import multiprocessing.queues
try:
  import clang.cindex as clx
  import petscClangLinterUtil
except ModuleNotFoundError as mnfe:
  if mnfe.name == "clang":
    raise RuntimeError("Must run e.g. 'python -m pip install clang' to use linter") from mnfe
  elif mnfe.name == "petscClangLinterUtil":
    raise RuntimeError("Must run the linter from ${PETSC_DIR}/lib/petsc/bin/maint/") from mnfe

"""
clang.cindex.TranslationUnit does not have all latest flags, but we prefix
with P_ just in case

see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
"""
P_CXTranslationUnit_None                                 = 0x0
P_CXTranslationUnit_DetailedPreprocessingRecord          = 0x01
P_CXTranslationUnit_Incomplete                           = 0x02
P_CXTranslationUnit_PrecompiledPreamble                  = 0x04
P_CXTranslationUnit_CacheCompletionResults               = 0x08
P_CXTranslationUnit_ForSerialization                     = 0x10
P_CXTranslationUnit_SkipFunctionBodies                   = 0x40
P_CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = 0x80
P_CXTranslationUnit_CreatePreambleOnFirstParse           = 0x100
P_CXTranslationUnit_KeepGoing                            = 0x200
P_CXTranslationUnit_SingleFileParse                      = 0x400
P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble    = 0x800
P_CXTranslationUnit_IncludeAttributedTypes               = 0x1000
P_CXTranslationUnit_VisitImplicitAttributes              = 0x2000
P_CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles     = 0x4000
P_CXTranslationUnit_RetainExcludedConditionalBlocks      = 0x8000

# clang options used for parsing files
baseClangOptions = (P_CXTranslationUnit_PrecompiledPreamble |
                    P_CXTranslationUnit_SkipFunctionBodies |
                    P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble)

# clang options for creating the precompiled megaheader
basePCHClangOptions  = (P_CXTranslationUnit_CreatePreambleOnFirstParse |
                        P_CXTranslationUnit_Incomplete |
                        P_CXTranslationUnit_ForSerialization |
                        P_CXTranslationUnit_KeepGoing)

# Cursors that may be attached to function-like usage
funcCallCursors = {clx.CursorKind.FUNCTION_DECL,clx.CursorKind.CALL_EXPR}

# Cursors that indicate change of logical scope
scopeCursors    = {clx.CursorKind.COMPOUND_STMT}

# Cursors that may be attached to mathemateical operations or types
mathCursors     = {clx.CursorKind.INTEGER_LITERAL,clx.CursorKind.UNARY_OPERATOR,clx.CursorKind.BINARY_OPERATOR}

# Cursors that contain base literal types
literalCursors  = {clx.CursorKind.INTEGER_LITERAL,clx.CursorKind.STRING_LITERAL}

# Cursors that may be attached to casting
castCursors     = {clx.CursorKind.CSTYLE_CAST_EXPR,clx.CursorKind.CXX_STATIC_CAST_EXPR,clx.CursorKind.CXX_DYNAMIC_CAST_EXPR,clx.CursorKind.CXX_REINTERPRET_CAST_EXPR,clx.CursorKind.CXX_CONST_CAST_EXPR,clx.CursorKind.CXX_FUNCTIONAL_CAST_EXPR}

# Cursors that may be attached when types are converted
convertCursors  = castCursors|{clx.CursorKind.UNEXPOSED_EXPR}

varTokens       = {clx.TokenKind.IDENTIFIER}

functionTypes   = {clx.TypeKind.FUNCTIONPROTO,clx.TypeKind.FUNCTIONNOPROTO}

# General Array types, note this doesn't contain the pointer type since that is usually handled
# differently
arrayTypes      = {clx.TypeKind.INCOMPLETEARRAY,clx.TypeKind.CONSTANTARRAY,clx.TypeKind.VARIABLEARRAY}

# Specific types
enumTypes   = {clx.TypeKind.ENUM}
# because PetscBool is an enum...
boolTypes   = enumTypes|{clx.TypeKind.BOOL}
charTypes   = {clx.TypeKind.CHAR_S,clx.TypeKind.UCHAR}
mpiIntTypes = {clx.TypeKind.INT}
intTypes    = enumTypes|mpiIntTypes|{clx.TypeKind.USHORT,clx.TypeKind.SHORT,clx.TypeKind.UINT,clx.TypeKind.LONG,clx.TypeKind.LONGLONG,clx.TypeKind.ULONGLONG}
realTypes   = {clx.TypeKind.FLOAT,clx.TypeKind.DOUBLE,clx.TypeKind.LONGDOUBLE,clx.TypeKind.FLOAT128}
scalarTypes = realTypes|{clx.TypeKind.COMPLEX}

"""
Adding new classes
------------------

You must register new instances of PETSc classes in the classIdMap which expects its
contents to be in the form:

"CaseSensitiveNameOfPrivateStruct *" : "CaseSensitiveNameOfCorrespondingClassId",

See below for examples.

* please add your new class in alphabetical order and preserve the alignment! *

The automated way to do it (in emacs) is to slap it in the first entry then highlight
the the contents (i.e. excluding "classIdMap = {" and the closing "}") and do:

1. M-x sort-fields RET
2. M-x align-regexp RET : RET
"""
classIdMap = {
  "_p_AO *"                     : "AO_CLASSID",
  "_p_Characteristic *"         : "CHARACTERISTIC_CLASSID",
  "_p_DM *"                     : "DM_CLASSID",
  "_p_DMAdaptor *"              : "DM_CLASSID",
  "_p_DMField *"                : "DMFIELD_CLASSID",
  "_p_DMKSP *"                  : "DMKSP_CLASSID",
  "_p_DMLabel *"                : "DMLABEL_CLASSID",
  "_p_DMPlexTransform *"        : "DMPLEXTRANSFORM_CLASSID",
  "_p_DMSNES *"                 : "DMSNES_CLASSID",
  "_p_DMTS *"                   : "DMTS_CLASSID",
  "_p_IS *"                     : "IS_CLASSID",
  "_p_ISLocalToGlobalMapping *" : "IS_LTOGM_CLASSID",
  "_p_KSP *"                    : "KSP_CLASSID",
  "_p_KSPGuess *"               : "KSPGUESS_CLASSID",
  "_p_LineSearch *"             : "SNESLINESEARCH_CLASSID",
  "_p_Mat *"                    : "MAT_CLASSID",
  "_p_MatCoarsen *"             : "MAT_COARSEN_CLASSID",
  "_p_MatColoring *"            : "MAT_COLORING_CLASSID",
  "_p_MatFDColoring *"          : "MAT_FDCOLORING_CLASSID",
  "_p_MatMFFD *"                : "MATMFFD_CLASSID",
  "_p_MatNullSpace *"           : "MAT_NULLSPACE_CLASSID",
  "_p_MatPartitioning *"        : "MAT_PARTITIONING_CLASSID",
  "_p_MatTransposeColoring *"   : "MAT_TRANSPOSECOLORING_CLASSID",
  "_p_PC *"                     : "PC_CLASSID",
  "_p_PF *"                     : "PF_CLASSID",
  "_p_PetscContainer *"         : "PETSC_CONTAINER_CLASSID",
  "_p_PetscConvEst *"           : "PETSC_OBJECT_CLASSID",
  "_p_PetscDS *"                : "PETSCDS_CLASSID",
  "_p_PetscDraw *"              : "PETSC_DRAW_CLASSID",
  "_p_PetscDrawAxis *"          : "PETSC_DRAWAXIS_CLASSID",
  "_p_PetscDrawBar *"           : "PETSC_DRAWBAR_CLASSID",
  "_p_PetscDrawHG *"            : "PETSC_DRAWHG_CLASSID",
  "_p_PetscDrawLG *"            : "PETSC_DRAWLG_CLASSID",
  "_p_PetscDrawSP *"            : "PETSC_DRAWSP_CLASSID",
  "_p_PetscDualSpace *"         : "PETSCDUALSPACE_CLASSID",
  "_p_PetscFE *"                : "PETSCFE_CLASSID",
  "_p_PetscFV *"                : "PETSCFV_CLASSID",
  "_p_PetscLimiter *"           : "PETSCLIMITER_CLASSID",
  "_p_PetscPartitioner *"       : "PETSCPARTITIONER_CLASSID",
  "_p_PetscQuadrature *"        : "PETSCQUADRATURE_CLASSID",
  "_p_PetscRandom *"            : "PETSC_RANDOM_CLASSID",
  "_p_PetscSF *"                : "PETSCSF_CLASSID",
  "_p_PetscSection *"           : "PETSC_SECTION_CLASSID",
  "_p_PetscSectionSym *"        : "PETSC_SECTION_SYM_CLASSID",
  "_p_PetscSpace *"             : "PETSCSPACE_CLASSID",
  "_p_PetscViewer *"            : "PETSC_VIEWER_CLASSID",
  "_p_PetscWeakForm *"          : "PETSCWEAKFORM_CLASSID",
  "_p_SNES *"                   : "SNES_CLASSID",
  "_p_TS *"                     : "TS_CLASSID",
  "_p_TSAdapt *"                : "TSADAPT_CLASSID",
  "_p_TSGLLEAdapt *"            : "TSGLLEADAPT_CLASSID",
  "_p_TSTrajectory *"           : "TSTRAJECTORY_CLASSID",
  "_p_Tao *"                    : "TAO_CLASSID",
  "_p_TaoLineSearch *"          : "TAOLINESEARCH_CLASSID",
  "_p_Vec *"                    : "VEC_CLASSID",
  "_p_VecTagger *"              : "VEC_TAGGER_CLASSID",
}

# directory names to exclude from processing, case sensitive
excludeDirNames     = {"tests","tutorials","f90-mod","f90-src","f90-custom","output","input","python","fsrc","ftn-auto","ftn-custom","f2003-src","ftn-kernels","benchmarks","docs"}
# directory suffixes to exclude from processing, case sensitive
excludeDirSuffixes  = (".dSYM",)
# file extensions to process, case sensitve
allowFileExtensions = (".c",".cpp",".cxx",".cu",".cc")

class QueueSignal(enum.IntEnum):
  __doc__="""
  Various signals to indicate return type on the data queue from child processes
  """
  WARNING      = enum.auto()
  UNIFIED_DIFF = enum.auto()
  ERRORS_LEFT  = enum.auto()
  ERRORS_FIXED = enum.auto()
  EXIT_QUEUE   = enum.auto()

class ParsingError(Exception):
  __doc__="""
  Mostly to just have a custom "something went wrong when trying to perform a check" to except
  for rather than using a built-in type. These are errors that are meant to be caught and logged
  rather than stopping execution alltogether.

  This should make it so that actual errors aren't hidden.
  """
  pass

class PetscCursor(object):
  __doc__="""
  A utility wrapper around clang.cindex.Cursor that makes retrieving certain useful properties (such as demangled names) from a cursor easier.
  Also provides a host of utility functions that get and (optionally format) the source code around a particular cursor. As it is a wrapper any
  operation done on a clang Cursor may be performed directly on a PetscCursor (although this object does not pass the isinstance() check).

  See __getattr__ below for more info.
  """
  @staticmethod
  def errorViewFromCursor(cursor):
    __doc__="""
    Something has gone wrong, and we try to extract as much information from the cursor as
    possible for the exception. Nothing is guaranteed to be useful here.
    """
    name = cursor.displayname
    kind = cursor.kind
    loc  = cursor.location
    try:
      fname  = loc.file.name
    except AttributeError:
      fname  = "UNKNOWN_FILE"
    locStr   = ":".join([fname,str(loc.column),str(loc.line)])
    # Does not yet raise exception so we can call it here
    typename = PetscCursor.getTypenameFromCursor(cursor)
    srcStr   = PetscCursor.getFormattedSourceFromCursor(cursor,nboth=2)
    return "'{}' of kind '{}' of type '{}' at {}:\n{}".format(name,kind,typename,locStr,srcStr)

  @staticmethod
  def getNameFromCursor(cursor):
    __doc__="""
    Try to convert **&(PetscObject)obj[i]+73 to obj
    """
    name = None
    if cursor.spelling:
      name = cursor.spelling
    elif cursor.kind in mathCursors:
      if cursor.kind  == clx.CursorKind.BINARY_OPERATOR:
        # we arbitrarily use the first token here since we assume that it is the important
        # one.
        operands = [c for c in cursor.get_children()]
        # its certainly funky when a binary operation doesn't have a binary system of
        # operands
        assert len(operands) == 2, "Found {} operands for binary operator when only expecting 2 for cursor {}".format(len(operands),PetscCursor.errorViewFromCursor(cursor))
        name = operands[0].spelling
      else:
        # just a plain old number or unary operator
        name = "".join(t.spelling for t in cursor.get_tokens())
    elif cursor.kind in castCursors:
      # Need to extract the castee from the caster
      castee = [c for c in cursor.get_children() if c.kind == clx.CursorKind.UNEXPOSED_EXPR]
      # If we don't have 1 symbol left then we're in trouble, as we probably didn't
      # pick the right cursors above
      assert len(castee) == 1, "Cannot determine castee from the caster for cursor {}".format(PetscCursor.errorViewFromCursor(cursor))
      # Easer to do some mild recursion to figure out the naming for us than duplicate
      # the code. Perhaps this should have some sort of recursion check
      name = PetscCursor.getNameFromCursor(castee[0])
    elif (cursor.type.get_canonical().kind == clx.TypeKind.POINTER) or (cursor.kind == clx.CursorKind.UNEXPOSED_EXPR):
      pointees = []
      if cursor.type.get_pointee().kind  == clx.TypeKind.CHAR_S:
        # For some reason preprocessor macros that contain strings don't propagate
        # their spelling up to the primary cursor, so we need to plumb through
        # the various sub-cursors to find it.
        pointees = [c for c in cursor.walk_preorder() if c.kind in literalCursors]
      elif clx.CursorKind.ARRAY_SUBSCRIPT_EXPR in {c.kind for c in cursor.get_children()}:
        # in the form of obj[i], so we try and weed out the iterator variable
        pointees = [c for c in cursor.walk_preorder() if c.type.get_canonical().kind in arrayTypes]
        if not pointees:
          # wasn't a pure array, so we try pointer
          pointees = [c for c in cursor.walk_preorder() if c.type.kind == clx.TypeKind.POINTER]
      pointees = list({p.spelling: p for p in pointees}.values())
      if len(pointees) > 1:
        # sometimes array subscripts can creep in
        pointees = [c for c in pointees if c.kind not in mathCursors]
      if len(pointees) == 1:
        name = PetscCursor.getNameFromCursor(pointees[0])
    if not name:
      # Catchall last attempt, we become the very thing we swore to destroy and parse the
      # tokens ourselves
      tokenList = [t for t in cursor.get_tokens() if t.kind in varTokens]
      # Remove iterator variables
      tokenList = [t for t in tokenList if t.cursor.kind not in mathCursors]
      # removes all cursors that have duplicate spelling
      tokenList = list({t.spelling: t for t in tokenList}.values())
      if len(tokenList) != 1:
        # For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
        # and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
        # children, no name, no tokens, and a completely incorrect SourceLocation.
        # They are for all intents and purposes uncheckable :)
        srcstr = PetscCursor.getRawSourceFromCursor(cursor)
        errstr = PetscCursor.errorViewFromCursor(cursor)
        if "PETSC_HASH" in srcstr:
          if "_MAP" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_MAP for cursor {}".format(errstr))
          elif "_SET" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_SET for cursor {}".format(errstr))
          else:
            raise RuntimeError("Unhandled unparsable PETSC_HASH_XXX for cursor {}".format(errstr))
        elif "PetscKernel_" in srcstr:
          raise ParsingError("Encountered unparsable PetscKernel_XXX for cursor {}".format(errstr))
        elif ("PetscOptions" in srcstr) or ("PetscObjectOptions" in srcstr):
          raise ParsingError("Encountered unparsable Petsc[Object]OptionsBegin for cursor {}".format(errstr))
        else:
          raise RuntimeError("Unexpected number of tokens ({}) for cursor {}".format(len(tokenList),errstr))
      name = tokenList[0].spelling
      assert name, "Cannot determine name of symbol from cursor {}".format(PetscCursor.errorViewFromCursor(cursor))
    return name

  @staticmethod
  def getRawNameFromCursor(cursor):
    __doc__="""
    if getNameFromCursor tries to convert **&(PetscObject)obj[i]+73 to obj then this function tries to extract **&(PetscObject)obj[i]+73
    in the cleanest way possible
    """
    name = "".join(t.spelling for t in cursor.get_tokens())
    if not name:
      try:
        # now we try for the formatted name
        name = PetscCursor.getNameFromCursor(cursor)
      except ParsingError:
        srcstr = PetscCursor.getRawSourceFromCursor(cursor)
        errstr = PetscCursor.errorViewFromCursor(cursor)
        if "PETSC_HASH" in srcstr:
          if "_MAP" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_MAP for cursor {}".format(errstr))
          elif "_SET" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_SET for cursor {}".format(errstr))
          else:
            raise RuntimeError("Unhandled unparsable PETSC_HASH_XXX for cursor {}".format(errstr))
        elif "PetscKernel_" in srcstr:
          raise ParsingError("Encountered unparsable PetscKernel_XXX for cursor {}".format(errstr))
        elif ("PetscOptions" in srcstr) or ("PetscObjectOptions" in srcstr):
          raise ParsingError("Encountered unparsable Petsc[Object]OptionsBegin for cursor {}".format(errstr))
        else:
          raise RuntimeError("Could not determine useful name for cursor {}".format(errstr))
    return name

  @staticmethod
  def getTypenameFromCursor(cursor):
    __doc__="""
    Try to get the most canonical type from a cursor so DM -> _p_DM *
    """
    if cursor.type.get_pointee().spelling:
      ctemp = cursor.type.get_pointee()
      if ctemp.get_canonical().spelling:
        typename = ctemp.get_canonical().spelling
      else:
        typename = ctemp.spelling
    elif cursor.type.get_canonical().spelling:
      typename = cursor.type.get_canonical().spelling
    else:
      typename = cursor.type.spelling
    return typename

  @staticmethod
  def getDerivedTypenameFromCursor(cursor):
    __doc__="""
    Get the least canonical type form a cursor so DM -> DM
    """
    return cursor.type.spelling

  @staticmethod
  def getRawSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,trim=False):
    return petscClangLinterUtil.getRawSourceFromCursor(cursor,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,trim=trim)

  def getRawSource(self,nbefore=0,nafter=0,nboth=0,trim=False):
    return petscClangLinterUtil.getRawSourceFromCursor(self,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,trim=trim)

  @staticmethod
  def getFormattedSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,view=False):
    return petscClangLinterUtil.getFormattedSourceFromCursor(cursor,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,view=view)

  def getFormattedSource(self,nbefore=0,nafter=0,nboth=0,view=False):
    return petscClangLinterUtil.getFormattedSourceFromCursor(self,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,view=view)

  @staticmethod
  def getFormattedLocationStringFromCursor(cursor):
    loc = cursor.location
    return ":".join([loc.file.name,str(loc.column),str(loc.line)])

  def getFormattedLocationString(self):
    loc = self.location
    return ":".join([loc.file.name,str(loc.column),str(loc.line)])

  @staticmethod
  def viewAstFromCursor(cursor):
    return print("\n".join(petscClangLinterUtil.viewAstFromCursor(cursor)))

  def viewAst(self):
    return self.viewAstFromCursor(self)

  @staticmethod
  def findCursorReferencesFromCursor(cursor):
    __doc__="""
    Brute force find and collect all references in a file that pertain to a particular
    cursor. Essentially refers to finding every reference to the symbol that the cursor
    represents, so this function is only useful for first-class symbols (i.e. variables,
    functions)
    """
    import ctypes

    foundCursors  = []
    callbackProto = ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,clx.Cursor,clx.SourceRange)

    class CXCursorAndRangeVisitor(ctypes.Structure):
      # see https://clang.llvm.org/doxygen/structCXCursorAndRangeVisitor.html
      #
      # typedef struct CXCursorAndRangeVisitor {
      #   void *context;
      #   enum CXVisitorResult (*visit)(void *context, CXCursor, CXSourceRange);
      # } CXCursorAndRangeVisitor;
      #
      # Note this is not a  strictly accurate recreation, as this struct expects a
      # (void *) but since C lets anything be a (void *) we can pass in a (PyObject *)
      _fields_ = [("context",ctypes.py_object),("visit",callbackProto)]

      @staticmethod
      def callBack(ctx,cursor,srcRange):
        # convert to py_object then take value of the pointer, i.e. the original class
        origCursor = ctypes.cast(ctx,ctypes.py_object).value
        # The "cursor" returned here is actually just a CXCursor, not the real
        # clx.Cursor that we lead python to believe in our function prototype. Luckily we
        # have all we need to remake the python object from scratch
        cursor = clx.Cursor.from_location(origCursor.translation_unit,srcRange.start)
        try:
          cursor = PetscCursor(cursor)
          foundCursors.append(cursor)
        except ParsingError:
          pass
        except RuntimeError as re:
          string = "Full error full error message below:"
          print('='*30,"CXCursorAndRangeVisitor Error",'='*30)
          print("It is possible that this is a false positive! E.g. some 'unexpected number of tokens' errors are due to macro instantiation locations being misattributed.\n",string,"\n","-"*len(string),"\n",re,sep="")
          print('='*30,"CXCursorAndRangeVisitor End Error",'='*26)
        return 1 # continue

    if not hasattr(clx.conf.lib,"clang_findReferencesInFile"):
      item = ("clang_findReferencesInFile",[clx.Cursor,clx.File,CXCursorAndRangeVisitor],ctypes.c_uint)
      clx.register_function(clx.conf.lib,item,False)

    pyCtx      = ctypes.py_object(cursor) # pyCtx = (PyObject *)cursor;
    callBack   = callbackProto(CXCursorAndRangeVisitor.callBack)
    cxCallback = CXCursorAndRangeVisitor(pyCtx,callBack)
    clx.conf.lib.clang_findReferencesInFile(cursor._PetscCursor__cursor,cursor.location.file,cxCallback)
    return foundCursors

  def findCursorReferences(self):
    return PetscCursor.findCursorReferencesFromCursor(self)

  def __init__(self,cursor,idx=-12345):
    assert isinstance(cursor,(clx.Cursor,PetscCursor))
    if isinstance(cursor,PetscCursor):
      self.__cursor        = cursor._PetscCursor__cursor
      self.name            = cursor.name
      self.typename        = cursor.typename
      self.derivedtypename = cursor.derivedtypename
      self.argidx          = cursor.argidx if idx == -12345 else idx
    else:
      self.__cursor        = cursor
      self.name            = self.getNameFromCursor(cursor)
      self.typename        = self.getTypenameFromCursor(cursor)
      self.derivedtypename = self.getDerivedTypenameFromCursor(cursor)
      self.argidx          = idx
    return

  def __getattr__(self,attr):
    __doc__="""
    Allows us to essentialy fake being a clang cursor, if __getattribute__ fails
    (i.e. the value wasn't found in self), then we try the cursor. So we can do things
    like self.translation_unit, but keep all of our variables out of the cursors
    namespace
    """
    return getattr(self.__cursor,attr)

  def __str__(self):
    locStr = self.getFormattedLocationString()
    srcStr = self.getFormattedSource(nboth=2)
    return "{}\n'{}' of derived type '{}', canonical type '{}'\n{}\n".format(locStr,self.name,self.derivedtypename,self.typename,srcStr)

class SourceFix(object):
  def __init__(self,filename,src,startline,begin,end,value):
    self.filename  = filename
    self.src       = src
    self.startLine = startline
    assert self.startLine >= 1, "startline {} < 1".format(self.startLine)
    assert end > begin, "end {} <= begin {}, ill-formed source fix".format(end,begin)
    self.begins    = [begin]
    self.ends      = [end]
    value, replace = str(value),self.src[begin:end]
    # this is an error, since previous detection should not have created a fix
    assert value != replace, "trying to replace {} with itself".format(replace)
    self.replace   = [replace]
    self.deltas    = [value]
    self.fixed     = None
    self.fixDepth  = 0
    return

  @classmethod
  def fromCursor(cls,cursor,value):
    fname     = cursor.location.file.name
    src       = PetscCursor.getRawSourceFromCursor(cursor)
    startline = cursor.extent.start.line
    begin,end = cursor.extent.start.column-1,cursor.extent.end.column-1
    return cls(fname,src,startline,begin,end,value)

  def appendFix(self,fix):
    assert isinstance(fix,SourceFix)
    assert self.src == fix.src, "Cannot combine fixes that do not share identical source!"
    self.begins.extend(fix.begins)
    self.ends.extend(fix.ends)
    self.replace.extend(fix.replace)
    self.deltas.extend(fix.deltas)
    return

  def collapse(self):
    __doc__="""
    Collapses a list of fixes and produces a fixed src line.
    Fixes probably should not overwrite each other (for now), so we error out, but this
    is arguably a completely valid case. I just have not seen an example of it that I
    can use to debug with yet.
    """
    if self.fixDepth == len(self.deltas): # already collapsed, no need to do it again
      assert self.fixed, "Fix depth {} = number of deltas {} but no fixed string exists".format(self.fixDepth,len(self.deltas))
      return
    idxDelta = 0
    newSrc   = self.src
    for begin,end,replace,delta in zip(self.begins,self.ends,self.replace,self.deltas):
      assert replace in newSrc, "Target replacement '{}' not in src '{}' anymore, fix no longer relevant".format(replace,newSrc)
      if (begin+idxDelta < 0) or (end+idxDelta > len(newSrc)):
        raise RuntimeError("Idx out of bounds of src, fix not viable")
      newSrcTemp = newSrc[:begin+idxDelta]+delta+newSrc[end+idxDelta:]
      idxDelta   = len(newSrcTemp)-len(newSrc)
      newSrc     = newSrcTemp
    self.fixDepth = len(self.deltas)
    self.fixed    = newSrc
    return

  @staticmethod
  def fastUnifiedDiff(listA,listB,fromfile="",tofile="",fromfiledate="",tofiledate="",n=0,lineterm="\n"):
    __doc__="""
    Optimized version of difflib.unified_diff. difflib.SequenceMatcher is unbelievably slow but we can aggresively cut corners since we know the general location of all of the differences. This function only really serves to format the changes into the unified diff format.
    """
    import difflib,itertools
    def formatRangeUnified(pre,start,stop):
      __doc__="""
      Convert range to the 'ed' format
      """
      start += pre
      stop += pre
      # Per the diff spec at http://www.unix.org/single_unix_specification/
      beginning = max(start,1)# lines start numbering with one
      length = stop-start
      if length == 1:
        return "{}".format(beginning)
      if not length:
        beginning -= 1        # empty ranges begin at line just before the range
      return "{},{}".format(beginning,length)

    if not fromfiledate or not tofiledate:
      import datetime
      rn = datetime.datetime.now().ctime()
      if not fromfiledate:
        fromfiledate = rn
      if not tofiledate:
        tofiledate   = rn
    fromdate = "\t{}".format(fromfiledate)
    todate   = "\t{}".format(tofiledate)
    yield "--- {}{}{}".format(fromfile,fromdate,lineterm)
    yield "+++ {}{}{}".format(tofile,todate,lineterm)
    deletes = {"replace","delete"}
    inserts = {"replace","insert"}

    # find consecutive streaks of values, do this by taking the difference between a value
    # and its index. If the values are consecutive val-idx(val) will be equal.
    for _,g in itertools.groupby(enumerate(val for _,val in listA),lambda x: x[0]-x[1]):
      groupIdxs = list(g)
      lineStart = min(l for _,l in groupIdxs)
      groupA,groupB = [listA[i][0] for i,_ in groupIdxs],[listB[i][0] for i,_ in groupIdxs if listB[i][0]]
      for group in difflib.SequenceMatcher(a=groupA,b=groupB).get_grouped_opcodes(n):
        first,last  = group[0],group[-1]
        file1_range = formatRangeUnified(lineStart,first[1],last[2])
        file2_range = formatRangeUnified(lineStart,first[3],last[4])
        yield "@@ -{} +{} @@{}".format(file1_range,file2_range,lineterm)

        for tag,i1,i2,j1,j2 in group:
          if tag == "equal":
            for line in groupA[i1:i2]:
              yield " "+line
            continue
          if tag in deletes:
            for line in groupA[i1:i2]:
              yield "-"+line
          if tag in inserts:
            for line in groupB[j1:j2]:
                yield "+"+line

class PetscLinter(object):
  def __init__(self,compilerFlags,clangOptions=baseClangOptions,prefix="[ROOT]",verbose=False,werror=False,lock=None):
    self.flags      = compilerFlags
    self.clangOpts  = clangOptions
    self.prefix     = prefix
    self.verbose    = verbose
    self.werror     = werror
    self.lock       = lock
    self.errPrefix  = " ".join([prefix,85*"-"])
    self.warnPrefix = " ".join([prefix,85*"%"])
    self.errors     = []
    self.warnings   = []
    # This can actually just be a straight list, since each linter object only ever
    # handles a single file, but use dict nonetheless
    self.patches    = {}
    self.index      = clx.Index.create()
    return

  def __str__(self):
    prefixStr = "Prefix:        '{}'".format(self.prefix)
    flagStr   = "Compiler Flags: {}".format(self.flags)
    clangStr  = "Clang Options:  {}".format(self.clangOpts)
    lockStr   = "Lock:           {}".format(self.lock!=None)
    showStr   = "Verbose:        {}".format(self.verbose)
    printList = [prefixStr,flagStr,clangStr,lockStr,showStr]
    errorStr  = self.getAllErrors()
    if errorStr: printList.append(errorStr)
    warnStr   = self.getAllWarnings(joinToString=True)
    if warnStr: printList.append(warnStr)
    return "\n".join(printList)

  def __enter__(self):
    return self

  def __exit__(self,excType,*args):
    if not excType:
      if self.verbose:
        self.__print(self.getAllWarnings(joinToString=True))
      self.__print(self.getAllErrors())
    return

  def __print(self,*args,**kwargs):
    args = tuple(a for a in args if a)
    if not args and not kwargs:
      return
    if self.lock:
      with self.lock:
        print(*args,**kwargs)
    else:
      print(*args,**kwargs)
    return

  @staticmethod
  def findFunctionCallExpr(tu,functionNames):
    __doc__="""
    Finds all function call expressions in container functionNames.

    Note that if a particular function call is not 100% correctly defined (i.e. would the
    file actually compile) then it will not be picked up by clang AST.

    Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
    built as if you are about to compile it, so macros are handled before any real
    function definitions in the AST, making it impossible to map a macro invocation to
    its 'parent' function.
    """
    class Scope(object):
      __doc__="""
      Scope encompasses both the logical and lexical reach of a callsite, and is used to
      determine if two function calls may be occur in chronological order. Scopes may be
      approximated by incrementing or decrementing a counter every time a pair of '{}' are
      encountered however it is not that simple. In practice they behave almost identically
      to sets. Every relation between scopes may be formed by the following axioms.

      - Scope A is said to be greater than scope B if one is able to get to scope B from scope A
      e.g.:
      { // scope A
        { // scope B < scope A
          ...
        }
      }
      - Scope A is said to be equivalent to scope B if and only if they are the same object.
      e.g.:
      { // scope A and scope B
        ...
      }

      One notable exception are switch-case statements. Here every 'case' label acts as its
      own scope, regardless of whether a "break" is inserted i.e.:

      switch (cond) { // scope A
      case 1: // scope B begin
        ...
        break; // scope B end
      case 2: // scope C begin
        ...
      case 2:// scope C end, scope D begin
        ...
        break; // scope D end
      }

      Semantics here are weird, as:
      - scope B, C, D < scope A
      - scope B != scope C != scope D
      """
      __slots__ = ("gen","super","children")

      def __init__(self,superScope=None):
        if superScope:
          assert isinstance(superScope,Scope)
          self.gen      = superScope.gen+1
        else:
          self.gen      = 0
        self.super      = superScope
        self.children   = []
        return

      def __str__(self):
        return "gen {} id {}".format(self.gen,id(self))

      def __lt__(self,other):
        assert isinstance(other,Scope)
        return not (self >= other)

      def __gt__(self,other):
        assert isinstance(other,Scope)
        return self.isChildOf(other)

      def __le__(self,other):
        assert isinstance(other,Scope)
        return not (self > other)

      def __ge__(self,other):
        assert isinstance(other,Scope)
        return (self > other) or (self == other)

      def __eq__(self,other):
        if other is not None:
          assert isinstance(other,Scope)
          return id(self) == id(other)
        return False

      def __ne__(self,other):
        return not (self == other)

      def sub(self):
        __doc__="""spawn sub-scope"""
        child = Scope(self)
        self.children.append(child)
        return child

      def isParentOf(self,other):
        __doc__="""self is parent of other"""
        if self == other:
          return False
        for child in self.children:
          if (other == child) or child.isParentOf(other):
            return True
        return False

      def isChildOf(self,other):
        __doc__="""self is child of other, or other is parent of self"""
        return other.isParentOf(self)

    def walkScopeSwitch(parent,scope):
      __doc__="""
      special treatment for switch-case since the AST setup for it is mind-boggingly stupid.
      The first node after a case statement is listed as the cases *child* whereas every other
      node (including the break!!) is the cases *sibling*
      """
      # in case we get here from a scope decrease within a case
      caseScope = scope
      for child in parent.get_children():
        if child.kind == clx.CursorKind.CASE_STMT:
          # create a new scope every time we encounter a case, this is now for all intents
          # and purposes the 'scope' going forward. We don't overwrite the original scope
          # since we still need each case scope to be the previous scopes sibling
          caseScope = scope.sub()
          yield from walkScope(child,caseScope)
        elif child.kind == clx.CursorKind.CALL_EXPR:
          if child.spelling in functionNames:
            yield (child,possibleParent,caseScope)
        elif child.kind in scopeCursors:
          yield from walkScopeSwitch(child,caseScope.sub())

    def walkScope(parent,scope=Scope()):
      __doc__="""
      walk the tree determining the scope of a node. here 'scope' refers not only
      to lexical scope but also to logical scope, see Scope object above
      """
      for child in parent.get_children():
        if child.kind == clx.CursorKind.SWITCH_STMT:
          # switch-case statements require special treatment, we skip to the compound
          # statement
          switchChildren = [c for c in child.get_children() if c.kind == clx.CursorKind.COMPOUND_STMT]
          assert len(switchChildren) == 1, "Switch statement has multiple '{' operators?"
          yield from walkScopeSwitch(switchChildren[0],scope.sub())
        elif child.kind == clx.CursorKind.CALL_EXPR:
          if child.spelling in functionNames:
            yield (child,possibleParent,scope)
        elif child.kind in scopeCursors:
          # scope has descreased
          yield from walkScope(child,scope.sub())
        else:
          # same scope
          yield from walkScope(child,scope)

    cursor,filename = tu.cursor,tu.cursor.spelling
    for possibleParent in cursor.get_children():
      # getting filename is for some reason stupidly expensive, so we do this check first
      if possibleParent.kind not in funcCallCursors: continue
      try:
        if possibleParent.location.file.name != filename: continue
      except AttributeError:
        # possibleParent.location.file is None
        continue
      # if we've gotten this far we have found a function definition
      yield from walkScope(possibleParent)

  def clear(self):
    self.errors   = []
    self.warnings = []
    self.patches  = {}
    return

  def parse(self,filename):
    if self.verbose:
      self.__print(self.prefix,"Processing file     ",filename)
    tu = self.index.parse(filename,args=self.flags,options=self.clangOpts)
    if tu.diagnostics and self.verbose:
      diags = {" ".join([self.prefix,d]) for d in map(str,tu.diagnostics)}
      self.__print("\n".join(diags))
    self.processRemoveDuplicates(filename,tu)
    return

  def getArgumentCursors(self,funcCursor):
    return tuple(PetscCursor(a,i+1) for i,a in enumerate(funcCursor.get_arguments()))

  def process(self,filename,tu):
    for func,parent,_ in self.findFunctionCallExpr(tu,checkFunctionMap.keys()):
      try:
        checkFunctionMap[func.spelling](self,func,parent)
      except ParsingError as pe:
        self.addWarning(filename,str(pe))
    return

  def processRemoveDuplicates(self,filename,tu):
    processedFuncs = {}
    for func,parent,scope in self.findFunctionCallExpr(tu,set(checkFunctionMap.keys())):
      try:
        checkFunctionMap[func.spelling](self,func,parent)
      except ParsingError as pe:
        self.addWarning(filename,str(pe))
      func  = PetscCursor(func)
      pname = PetscCursor.getNameFromCursor(parent)
      try:
        processedFuncs[pname].append((func,scope))
      except KeyError:
        processedFuncs[pname] = [(func,scope)]
    for pname,functionList in processedFuncs.items():
      seen = {}
      for func,scope in functionList:
        try:
          combo = tuple([func.displayname]+[PetscCursor.getRawNameFromCursor(a) for a in func.get_arguments()])
        except ParsingError:
          continue
        if combo not in seen:
          seen[combo] = (func,scope)
        elif scope >= seen[combo][1]:
          seenStart = seen[combo][0].extent.start.line
          fname     = func.location.file.name
          src       = func.getRawSource()
          startline = func.extent.start.line
          begin,end = 0,len(src)
          patch     = SourceFix(fname,src,startline,begin,end,"")
          self.addErrorFromCursor(func,"Duplicate function found previous identical usage:\n\n{}".format(seen[combo][0].getFormattedSource(nbefore=2,nafter=startline-seenStart)),patch=patch)
    return

  def addErrorFromCursor(self,cursor,errorMessage,patch=None):
    errMess = "".join(["\nERROR {}: ".format(len(self.errors)),str(cursor),"\n",errorMessage])
    self.errors.append((errMess,patch != None))
    try:
      self.patches[patch.filename].append(patch)
    except KeyError:
      self.patches[patch.filename] = [patch]
      return
    except AttributeError:
      # patch = None, return
      return
    # check if this is a compound error, i.e. an additional error on the same line
    # in which case we need to combine with previous patch
    for prevPatch in self.patches[patch.filename][:-1]:
      if prevPatch.startLine == patch.startLine:
        # remove ourselves from the list
        patch = self.patches[patch.filename].pop()
        # this should now be the previous patch on the same line, so we combine with it
        prevPatch.appendFix(patch)
        break
    return

  def getAllErrors(self):
    errLeftStr,errFixedStr = "",""
    errLeft,errFixed       = [],[]
    for err,fixed in self.errors:
      if fixed:
        errFixed.append(err)
      else:
        errLeft.append(err)
    if errLeft:
      errLeftStr = "\n".join([self.errPrefix,"\n".join(errLeft)[1:],self.errPrefix])
    if errFixed:
      errFixedStr = "\n".join([self.errPrefix,"\n".join(errFixed)[1:],self.errPrefix])
    return errLeftStr,errFixedStr

  def addWarning(self,filename,warnMsg):
    if self.werror:
      self.addErrorFromCursor(filename,warnMsg)
    else:
      try:
        if warnMsg in self.warnings[-1][1]:
          # we just had the exact same warning, we can ignore it. This happens very often
          # for warnings occurring deep within a macro
          return
      except IndexError:
        pass
      warnStr = "".join(["\nWARNING {}: ".format(len(self.warnings)),warnMsg])
      self.warnings.append((filename,warnStr))
    return

  def addWarningFromCursor(self,locCursor,warnMsg):
    if self.werror:
      self.addErrorFromCursor(locCursor,warnMsg)
    else:
      warnPrefix = str(locCursor)
      warnFile   = locCursor.location.file.name
      warnStr    = "".join(["\nWARNING {}: ".format(len(self.warnings)),warnPrefix,"\n",warnMsg])
      self.warnings.append((warnFile,warnStr))
    return

  def getAllWarnings(self,joinToString=False):
    if joinToString:
      if len(self.warnings):
        warnings = "\n".join([self.warnPrefix,"\n".join(s for _,s in self.warnings)[1:],self.warnPrefix])
      else:
        warnings = ""
    else:
      warnings = self.warnings
    return warnings

  def coalescePatches(self):
    combinedPatches = []
    for filename,patches in self.patches.items():
      for p in patches:
        p.collapse()
      srcList   = [(p.src,p.startLine) for p in patches]
      fixedList = [(p.fixed,p.startLine) for p in patches]
      unified   = "".join(SourceFix.fastUnifiedDiff(srcList,fixedList,fromfile=filename,tofile=filename))
      combinedPatches.append((filename,unified))
    return combinedPatches

class WorkerPool(mp.queues.JoinableQueue):
  def __init__(self,numWorkers=-1,timeout=2,verbose=False,prefix="[ROOT]",**kwargs):
    if numWorkers < 0:
      numWorkers = max(mp.cpu_count()-1,1)
    super().__init__(3*numWorkers,**kwargs,ctx=mp.get_context())
    if numWorkers in {0,1}:
      print(prefix,"Number of worker processes ({}) too small, disabling multiprocessing".format(numWorkers))
      self.parallel    = False
      self.errorQueue  = None
      self.returnQueue = None
      self.lock        = None
    else:
      print(prefix,"Number of worker processes ({}) sufficient, enabling multiprocessing".format(numWorkers))
      self.parallel    = True
      self.errorQueue  = mp.Queue()
      self.returnQueue = mp.Queue()
      self.lock        = mp.Lock()
    self.workers     = []
    self.numWorkers  = numWorkers
    self.timeout     = timeout
    self.verbose     = verbose
    self.prefix      = prefix
    self.warnings    = []
    self.errorsLeft  = []
    self.errorsFixed = []
    self.patches     = []
    return

  def setup(self,compilerFlags,clangLib=None,clangOptions=baseClangOptions,werror=False):
    if clangLib is None:
      assert clx.conf.loaded, "Must initialize libClang first"
      clangLib = clx.conf.get_filename()
    if self.parallel:
      workerArgs = (clangLib,checkFunctionMap,classIdMap,compilerFlags,clangOptions,self.verbose,werror,self.errorQueue,self.returnQueue,self,self.lock)
      for i in range(self.numWorkers):
        workerName = "[{}]".format(i)
        worker     = mp.Process(target=queueMain,args=workerArgs,name=workerName,daemon=True)
        worker.start()
        self.workers.append(worker)
    else:
      self.linter = PetscLinter(compilerFlags,clangOptions=clangOptions,prefix=self.prefix,verbose=self.verbose,werror=werror)
    return

  def walk(self,srcLoc,excludeDirs=excludeDirNames,excludeDirSuff=excludeDirSuffixes,allowFileSuff=allowFileExtensions):
    if os.path.isfile(srcLoc):
      self.put(srcLoc)
    else:
      for root,dirs,files in os.walk(srcLoc):
        if self.verbose: print(self.prefix,"Processing directory",root)
        dirs[:] = [d for d in dirs if d not in excludeDirs]
        dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuff)]
        files   = [os.path.join(root,f) for f in files if f.endswith(allowFileSuff)]
        for filename in files:
          self.put(filename)
    return

  def put(self,filename,*args):
    if self.parallel:
      import queue
      # continuously put files onto the queue, if the queue is full we block for
      # queueTimeout seconds and if we still cannot insert to the queue we check
      # children for errors. If no errors are found we try again.
      while True:
        try:
          super().put(filename,True,self.timeout)
          break # only get here if put is successful
        except queue.Full:
          # we don't want to join here since a child may have encountered an error!
          self.check()
    else:
      self.linter.parse(filename)
      self.patches.extend(self.linter.coalescePatches())
      errLeft,errFixed = self.linter.getAllErrors()
      self.errorsLeft.append(errLeft)
      self.errorsFixed.append(errFixed)
      self.warnings.append(self.linter.getAllWarnings())
      self.linter.clear()
    return

  def check(self,join=False):
    if self.parallel:
      stopMultiproc = False
      # join here to colocate error messages if needs be
      if join: self.join()
      while not self.errorQueue.empty():
        # while this does get recreated for every error, we do not want to needlessly
        # reinitialize it when no errors exist. If we get to this point however we no longer
        # care about performance as we are about to crash everything.
        errBars   = "".join(["[ERROR]",85*"-","[ERROR]\n"])
        errBars   = [errBars,errBars]
        exception = self.errorQueue.get()
        try:
          errMess = str(exception).join(errBars)
        except:
          errMess = exception
        print(errMess)
        stopMultiproc = True
      if stopMultiproc:
        raise RuntimeError("Error in child process detected")
    return

  def finalize(self):
    if self.parallel:
      self.check(join=True)
      self.errorQueue.close()
      # send stop-signal to child processes
      for _ in range(self.numWorkers):
        self.put(QueueSignal.EXIT_QUEUE)
      while not self.returnQueue.empty():
        signal,returnData = self.returnQueue.get()
        if signal == QueueSignal.ERRORS_LEFT:
          self.errorsLeft.append(returnData)
        elif signal == QueueSignal.ERRORS_FIXED:
          self.errorsFixed.append(returnData)
        elif signal == QueueSignal.UNIFIED_DIFF:
          self.patches.extend(returnData)
        elif signal == QueueSignal.WARNING:
          self.warnings.append(returnData)
        else:
          raise ValueError("Unknown data returned by returnQueue {}, {}".format(signal,returnData))
      self.returnQueue.close()
      self.close()
      for worker in self.workers:
        # spin until every child exits, we need to do this otherwise the final summary
        # output is totally garbled
        worker.join()
        if sys.version_info >= (3,7):
          worker.close()
    self.errorsLeft  = [e for e in self.errorsLeft if e] # remove any None's
    self.errorsFixed = [e for e in self.errorsFixed if e]
    self.warnings    = [w for w in self.warnings if w]
    self.patches     = [p for p in self.patches if p]
    return self.warnings,self.errorsLeft,self.errorsFixed,self.patches


"""Generic test and utility functions"""
def alwaysTrue(*args,**kwargs):
  return True

def alwaysFalse(*args,**kwargs):
  return False

def addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName):
  __doc__="""
  shorthand for extracting a fix from a function cursor
  """
  call = [c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO]
  assert len(call) == 1
  fix = SourceFix.fromCursor(call[0],validFuncName)
  linter.addErrorFromCursor(obj,"Incorrect use of {}(), use {}() instead".format(funcCursor.displayname,validFuncName),patch=fix)
  return

def convertToCorrectPetscValidLogicalCollectiveXXX(linter,obj,objType,**kwargs):
  __doc__="""
  Try to glean the correct PetscValidLogicalCollectiveXXX from the type, used as a failure hook in the validlogicalcollective checks.
  """
  validFuncName = None
  objTypeKind   = objType.kind
  if objTypeKind in scalarTypes:
    if "PetscReal" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveReal"
    elif "PetscScalar" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveScalar"
  elif objTypeKind in enumTypes:
    if "PetscBool" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveBool"
    else:
      validFuncName = "PetscValidLogicalCollectiveEnum"
  elif objTypeKind in intTypes:
    if "PetscInt" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveInt"
    elif "PetscMPIInt" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveMPIInt"
  if validFuncName:
    funcCursor = kwargs["funcCursor"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName)
    return True
  return False

def convertToCorrectPetscValidXXXPointer(linter,obj,objType,**kwargs):
  __doc__="""
  Try to glean the correct PetscValidLogicalXXXPointer from the type, used as a failure hook in the validpointer checks.
  """
  validFuncName = None
  objTypeKind   = objType.kind
  if (objTypeKind == clx.TypeKind.RECORD) or (objTypeKind == clx.TypeKind.VOID):
    # pointer to struct or void pointer, use PetscValidPointer() instead
    validFuncName = "PetscValidPointer"
  elif objTypeKind in charTypes:
    validFuncName = "PetscValidCharPointer"
  elif objTypeKind in scalarTypes:
    if "PetscReal" in obj.derivedtypename:
      validFuncName = "PetscValidRealPointer"
    elif "PetscScalar" in obj.derivedtypename:
      validFuncName = "PetscValidScalarPointer"
  elif objTypeKind in enumTypes:
    if "PetscBool" in obj.derivedtypename:
      validFuncName = "PetscValidBoolPointer"
  elif objTypeKind in intTypes:
    if ("PetscInt" in obj.derivedtypename) or ("PetscMPIInt" in obj.derivedtypename):
      validFuncName = "PetscValidIntPointer"
  if validFuncName:
    funcCursor = kwargs["funcCursor"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName)
    return True
  return False

def checkIsPetscScalarAndNotPetscReal(linter,obj,objType,**kwargs):
  __doc__="""
  Used as a success hook, since a scalar may (depending on how petsc was configured) pass the type check for reals, so we must double check the name
  """
  if "PetscScalar" not in obj.derivedtypename:
    funcCursor = kwargs["funcCursor"]
    if "PetscReal" in obj.derivedtypename:
      validFunc = kwargs["validFunc"]
      addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
    else:
      linter.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscScalars".format(funcName=funcCursor.displayname))
  return True

def checkIsPetscRealAndNotPetscScalar(linter,obj,objType,**kwargs):
  if "PetscReal" not in obj.derivedtypename:
    funcCursor = kwargs["funcCursor"]
    if "PetscScalar" in obj.derivedtypename:
      validFunc = kwargs["validFunc"]
      addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
    else:
      linter.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscReals".format(funcName=funcCursor.displayname))
  return True

def checkIntIsNotPetscBool(linter,obj,objType,**kwargs):
  if "PetscBool" in obj.derivedtypename:
    funcCursor,validFunc = kwargs["funcCursor"],kwargs["validFunc"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
  return True

def checkMPIIntIsNotPetscInt(linter,obj,objType,**kwargs):
  if "PetscInt" in obj.derivedtypename:
    funcCursor,validFunc = kwargs["funcCursor"],kwargs["validFunc"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
  return True

def checkIsPetscBool(linter,obj,objType,**kwargs):
  if ("PetscBool" not in obj.derivedtypename) and ("bool" not in obj.typename):
    funcCursor = kwargs["funcCursor"]
    linter.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscBool or bool".format(funcName=funcCursor.displayname))
  return True

def checkIsPetscObject(linter,obj):
  __doc__="""
  Returns True if obj is a valid PetscObject, otherwise False. Automatically adds the error to the linter. Raises RuntimeError if obj is a PetscObject that isn't registered in the classIdMap.
  """
  if not obj.typename.startswith("_p_"):
    linter.addErrorFromCursor(obj,"Non-PETSc type when PETSc object expected.")
    return False
  elif obj.typename not in classIdMap:
    # Raise exception here since this isn't a bad source, moreso a failure of
    # this script since it should know about all petsc classes
    errorMessage = "{}\nUnknown or invalid PETSc class '{}'. If you are introducing a new class, you must register it with this linter! See {} and search for 'Adding new classes' for more information\n".format(obj,obj.derivedtypename,osResolvePath(__file__))
    raise RuntimeError(errorMessage)
  validObject = True
  pObjType    = obj.type.get_canonical().get_pointee()
  # Must have a struct here, e.g. _p_Vec
  assert pObjType.kind == clx.TypeKind.RECORD,"Symbol does not appear to be a struct!"
  objFields = [f for f in pObjType.get_fields()]
  if len(objFields) >= 2:
    if (PetscCursor.getTypenameFromCursor(objFields[0]) != "_p_PetscObject"):
      validObject = False
  else:
    validObject = False
  if not validObject:
    objDecl = PetscCursor(pObjType.get_declaration())
    if len(objFields) == 0:
      linter.addWarningFromCursor(obj,"Object '{}' is prefixed with '_p_' to indicate it is a PetscObject but cannot determine fields. Likely the header containing definition of the object is in a nonstandard place:\n\n{}\n{}".format(objDecl.typename,objDecl.getFormattedLocationString(),objDecl.getFormattedSource(nafter=2)))
    else:
      linter.addErrorFromCursor(obj,"Object '{}' is prefixed with '_p_' to indicate it is a PetscObject but its definition is missing a PETSCHEADER as the first struct member:\n\n{}\n{}".format(objDecl.typename,objDecl.getFormattedLocationString(),objDecl.getFormattedSource(nafter=2)))
  return validObject

def checkMatchingClassid(linter,obj,objClassid):
  __doc__="""
  Does the classid match the particular PETSc type
  """
  checkIsPetscObject(linter,obj)
  expectedClassid = classIdMap[obj.typename]
  if objClassid.name != expectedClassid:
    fix = SourceFix.fromCursor(objClassid,expectedClassid)
    linter.addErrorFromCursor(obj,"Classid doesn't match. Expected '{}' found '{}'".format(expectedClassid,objClassid.name),patch=fix)
  return

def checkTraceableToParentArgs(obj,parentArgNames):
  __doc__="""
  Try and see if the cursor can be linked to parent function arguments. If it can be successfully linked return the index of the matched object otherwise raises ParsingError.

  myFunction(barType bar)
  ...
  fooType foo = bar->baz;
  macro(foo,barIdx);
  /* or */
  macro(bar->baz,barIdx);
  /* or */
  initFooFromBar(bar,&foo);
  macro(foo,barIdx);
  """
  potentialParents = []
  defCursor        = obj.get_definition()
  if defCursor:
    assert defCursor.location != obj.location, "Object has definition cursor, yet the cursor did not move. This should be handled!"
    if defCursor.kind == clx.CursorKind.VAR_DECL:
      # found definition, so were in business
      # Parents here is an odd choice of words since on the very same line I loop
      # over children, but then again clangs AST has an odd semantic for parents/children
      convertOrDereferenceCursors = convertCursors|{clx.CursorKind.UNARY_OPERATOR}
      for defChild in defCursor.get_children():
        if defChild.kind in convertOrDereferenceCursors:
          potentialParentsTemp = [child for child in defChild.walk_preorder() if child.kind == clx.CursorKind.DECL_REF_EXPR]
          # Weed out any self-references
          potentialParentsTemp = [parent for parent in potentialParentsTemp if parent.spelling != defCursor.spelling]
          potentialParents.extend(potentialParentsTemp)
    elif defCursor.kind == clx.CursorKind.FIELD_DECL:
      # we have deduced that the original cursor may refer to a struct member
      # reference, so we go back and see if indeed this is the case
      for memberChild in obj.get_children():
        if memberChild.kind == clx.CursorKind.MEMBER_REF_EXPR:
          potentialParentsTemp = [c for c in memberChild.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR]
          potentialParentsTemp = [parent for parent in potentialParentsTemp if parent.spelling != memberChild.spelling]
          potentialParents.extend(potentialParentsTemp)
  elif obj.kind in convertCursors:
    curs = [PetscCursor(c,obj.argidx) for c in obj.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR]
    if len(curs) > 1:
      curs = [c for c in curs if c.displayname == obj.name]
    assert len(curs) == 1, "Could not uniquely determine base cursor from conversion cursor {}".format(obj)
    obj = curs[0]
    # for cases with casting + struct member reference:
    #
    # macro((type *)bar->baz,barIdx);
    #
    # the object "name" will (rightly) refer to 'baz', but since this is an inline
    # "definition" it doesn't show up in get_definition(), thus we check here
    potentialParents.append(obj)
  if not potentialParents:
    # this is the if-all-else-fails approach, first we search the __entire__ file for
    # references to the cursor. Once we have some matches we take the earliest one
    # as this one is in theory where the current cursor is instantiated. Then we select
    # the best match for the possible instantiating cursor and recursively call this
    # function. This section stops when the cursor definition is of type PARM_DECL (i.e.
    # defined as the function parameter to the parent function).
    refsAll = obj.findCursorReferences()
    # don't care about uses of object __after__ the macro, and don't want to pick up
    # the actual macro location either
    refsAll = [r for r in refsAll if r.location.line < obj.location.line]
    # we just tried those and they didn't work, also more importantly weeds out the
    # instantiation line if this is an intermediate cursor in a recursive call to this
    # function
    argRefs = [r for r in refsAll if r.kind not in {clx.CursorKind.VAR_DECL,clx.CursorKind.FIELD_DECL}]
    if not len(argRefs):
      # it's not traceable to a function argument, so maybe its a global static variable
      if len([r for r in refsAll if r.storage_class in {clx.StorageClass.STATIC}]):
        # a global variable is not a function argumment, so this is unhandleable
        raise ParsingError("PETSC_CLANG_STATIC_ANALYZER_IGNORE")

    assert len(argRefs), "Could not determine the origin of cursor {}".format(obj)
    # take the first, as this is the earliest
    firstRef  = argRefs[0]
    tu,loc    = firstRef.translation_unit,firstRef.location
    srcLen    = len(firstRef.getRawSource())
    # why the following song and dance? Because you cannot walk the AST backwards, and
    # in the case that the current cursor is in a function call we need to access
    # our co-arguments to the function, i.e. "adjacent" branches since they should link
    # to (or be) in the parent functions argument list. So we have to
    # essentially reparse this line to be able to start from the top.
    lineStart = clx.SourceLocation.from_position(tu,loc.file,loc.line,1)
    lineEnd   = clx.SourceLocation.from_position(tu,loc.file,loc.line,srcLen+1)
    lineRange = clx.SourceRange.from_locations(lineStart,lineEnd)
    tGroup    = list(clx.TokenGroup.get_tokens(tu,lineRange))
    funcProto = [i for i,t in enumerate(tGroup) if t.cursor.type.get_canonical().kind in functionTypes]
    if funcProto:
      import itertools

      assert len(funcProto) == 1, "Could not determine unique function prototype from {} for provenance of {}".format("".join([t.spelling for t in tGroup]),obj)
      idx        = funcProto[0]
      lambdaExpr = lambda t: (t.spelling != ")") and t.kind in varTokens
      iterator   = map(lambda x: x.cursor,itertools.takewhile(lambdaExpr,tGroup[idx+2:]))
    # we now have completely different cursor selected, so we recursively call this
    # function
    else:
      # not a function call, must be an assignment statement, meaning we should now
      # assert that the current obj is being assigned to
      assert PetscCursor.getNameFromCursor(tGroup[0].cursor) == obj.name
      # find the binary operator, it will contain the most comprehensive AST
      eqLoc    = list(map(lambda x: x.spelling,tGroup)).index("=")
      iterator = tGroup[eqLoc].cursor.walk_preorder()
      iterator = [c for c in iterator if c.kind == clx.CursorKind.DECL_REF_EXPR]
    altCursor = [c for c in iterator if PetscCursor.getNameFromCursor(c) != obj.name]
    potentialParents.extend(altCursor)
  if not potentialParents:
    raise ParsingError
  # arguably at this point anything other than len(potentialParents) should be 1,
  # and anything else can be considered a failure of this routine (therefore a RTE)
  # as it should be able to detect the definition.
  assert len(potentialParents) == 1, "Cannot determine a unique definition cursor for object"
  # If >1 cursor, probably a bug since we should have weeded something out
  parent = potentialParents[0]
  if parent.get_definition().kind == clx.CursorKind.PARM_DECL:
    name = PetscCursor.getNameFromCursor(parent)
    try:
      loc  = parentArgNames.index(name)
    except ValueError as ve:
      # name isn't in the parent arguments, so we raise parsing error from it
      raise ParsingError from ve
  else:
    parent = PetscCursor(parent,obj.argidx)
    # deeper into the rabbit hole
    loc = checkTraceableToParentArgs(parent,parentArgNames)
  return loc

def checkMatchingArgNum(linter,obj,idx,parentArgs):
  __doc__="""
  Is the Arg # correct w.r.t. the function arguments
  """
  if idx.canonical.kind not in mathCursors:
    # sometimes it is impossible to tell if the index is correct so this is a warning not
    # an error. For example in the case of a loop:
    # for (i = 0; i < n; ++i) PetscValidIntPointer(arr+i,i);
    linter.addWarningFromCursor(idx,"Index value is of unexpected type '{}'".format(idx.canonical.kind))
    return
  try:
    idxNum = int(idx.name)
  except ValueError:
    linter.addWarningFromCursor(idx,"Potential argument mismatch, could not determine integer value")
    return
  parentArgNames = tuple(s.name for s in parentArgs)
  try:
    matchLoc = parentArgNames.index(obj.name)
  except ValueError:
    try:
      matchLoc = checkTraceableToParentArgs(obj,parentArgNames)
    except ParsingError as pe:
      # If the parent arguments don't contain the symbol and we couldn't determine a
      # definition then we cannot check for correct numbering, so we cannot do
      # anything here but emit a warning
      if "PETSC_CLANG_STATIC_ANALYZER_IGNORE" in pe.args:
        return
      if len(parentArgs):
        parentFunc = PetscCursor(parentArgs[0].semantic_parent)
        parentFuncName = parentFunc.name+"()"
        parentFuncSrc  = parentFunc.getFormattedSource()
      else:
        # parent function has no arguments (very likely that "obj" is a global variable)
        parentFuncName = "UNKNOWN FUNCTION"
        parentFuncSrc  = "  <could not determine parent function signature from arguments>"
      linter.addWarningFromCursor(obj,"Cannot determine index correctness, parent function '{}' seemingly does not contain the object:\n\n{}".format(parentFuncName,parentFuncSrc))
      return
  if idxNum != parentArgs[matchLoc].argidx:
    errMess = "Argument number doesn't match for '{}'. Found '{}' expected '{}' from\n\n{}".format(obj.name,str(idxNum),str(parentArgs[matchLoc].argidx),parentArgs[matchLoc].getFormattedSource())
    fix = SourceFix.fromCursor(idx,parentArgs[matchLoc].argidx)
    linter.addErrorFromCursor(idx,errMess,patch=fix)
  return

def checkMatchingSpecificType(linter,obj,expectedTypeKinds,pointer,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=alwaysFalse,**kwargs):
  __doc__="""
  Checks that obj is of a particular kind, for example char. Can optionally handle pointers too.

  Nonstandard arguments:

  expectedTypeKinds            - the base type that you want obj to be, e.g. clx.TypeKind.ENUM
                                 for PetscBool
  pointer                      - should obj be a pointer to your type?
  unexpectedNotPointerFunction - pointer is TRUE, the object matches the base type but IS NOT
                                 a pointer
  unexpectedPointerFunction    - pointer is FALSE, the object matches the base type but IS a
                                 pointer
  successFunction              - the object matches the type and pointer specification
  failureFunction              - the object does NOT match the base type

  The hooks must return whether they handled the failure, this can mean either determining
  that the object was correct all along, or that a more helpful error message was logged
  and/or that a fix was created.
  """
  objType = obj.canonical.type.get_canonical()
  if pointer:
    if objType.kind in expectedTypeKinds:
      if not unexpectedNotPointerFunction(linter,obj,objType,**kwargs):
        linter.addErrorFromCursor(obj,"Object of clang type {} is not a pointer. Expected pointer of one of the following types: {}".format(objType.kind,expectedTypeKinds))
      return
    if objType.kind == clx.TypeKind.INCOMPLETEARRAY:
      objType = objType.element_type
      # get rid of any nested array types
      while objType.kind in arrayTypes:
        objType = objType.element_type
    if objType.kind == clx.TypeKind.POINTER:
      objType = objType.get_pointee()
      # get rid of any nested pointer types
      while objType.kind == clx.TypeKind.POINTER:
        objType = objType.get_pointee()
  else:
    if objType.kind in arrayTypes or objType.kind == clx.TypeKind.POINTER:
      if not unexpectedPointerFunction(linter,obj,objType,**kwargs):
        linter.addErrorFromCursor(obj,"Object of clang type {} is a pointer when it should not be".format(objType.kind))
      return
  if objType.kind in expectedTypeKinds:
    handled = successFunction(linter,obj,objType,**kwargs)
    if not handled:
      errorMessage = "{}\nType checker successfully matched object of type {} to (one of) expected types:\n- {}\n\nBut user supplied on-successful-match hook '{}' returned non-truthy value '{}' indicating unhandled error!".format(obj,objType.kind,'\n- '.join(map(str,expectedTypeKinds)),successFunction,handled,expectedTypeKinds,objType.kind)
      raise RuntimeError(errorMessage)
  else:
    if not failureFunction(linter,obj,objType,**kwargs):
      linter.addErrorFromCursor(obj,"Object of clang type {} is not in expected types: {}".format(objType.kind,expectedTypeKinds))
  return


"""Specific 'driver' function to test a particular macro archetype"""
def checkObjIdxGenericN(linter,func,parent):
  __doc__="""
  For generic checks where the form is func(obj1,idx1,...,objN,idxN)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  for obj,idx in zip(funcArgs[::2],funcArgs[1::2]):
    checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidHeaderSpecificType(linter,func,parent):
  __doc__="""
  Specific check for PetscValidHeaderSpecificType(obj,classid,idx,type)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  # Don't need the type
  obj,classid,idx,_ = funcArgs
  checkMatchingClassid(linter,obj,classid)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidHeaderSpecific(linter,func,parent):
  __doc__="""
  Specific check for PetscValidHeaderSpecific(obj,classid,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  obj,classid,idx = funcArgs
  checkMatchingClassid(linter,obj,classid)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidPointerAndType(linter,func,parent,expectedTypes,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=convertToCorrectPetscValidXXXPointer,**kwargs):
  __doc__="""
  Generic check for PetscValidXXXPointer(obj,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  obj,idx = funcArgs
  checkMatchingSpecificType(linter,obj,expectedTypes,True,
                            unexpectedNotPointerFunction=unexpectedNotPointerFunction,
                            unexpectedPointerFunction=unexpectedPointerFunction,
                            successFunction=successFunction,
                            failureFunction=failureFunction,
                            funcCursor=func,
                            **kwargs)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidCharPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidCharPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,charTypes)
  return

def checkPetscValidIntPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidIntPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,intTypes,successFunction=checkIntIsNotPetscBool,validFunc="PetscValidBoolPointer")
  return

def checkPetscValidBoolPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidBoolPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,boolTypes,successFunction=checkIsPetscBool)
  return

def checkPetscValidScalarPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidScalarPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,scalarTypes,successFunction=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidRealPointer")
  return

def checkPetscValidRealPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidRealPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,realTypes,successFunction=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidScalarPointer")
  return

def checkPetscValidLogicalCollective(linter,func,parent,expectedTypes,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=convertToCorrectPetscValidLogicalCollectiveXXX,**kwargs):
  __doc__="""
  Generic check for PetscValidLogicalCollectiveXXX(pobj,obj,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  # dont need the petsc object, nothing to check there
  _,obj,idx = funcArgs
  checkMatchingSpecificType(linter,obj,expectedTypes,False,
                            unexpectedNotPointerFunction=unexpectedNotPointerFunction,
                            unexpectedPointerFunction=unexpectedPointerFunction,
                            successFunction=successFunction,
                            failureFunction=failureFunction,
                            funcCursor=func,
                            **kwargs)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidLogicalCollectiveScalar(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveScalar(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,scalarTypes,successFunction=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidLogicalCollectiveReal")
  return

def checkPetscValidLogicalCollectiveReal(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveReal(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,realTypes,successFunction=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidLogicalCollectiveScalar")
  return

def checkPetscValidLogicalCollectiveInt(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveInt(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,intTypes,successFunction=checkIntIsNotPetscBool,validFunc="PetscValidLogicalCollectiveBool")
  return

def checkPetscValidLogicalCollectiveMPIInt(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveMPIInt(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,mpiIntTypes,successFunction=checkMPIIntIsNotPetscInt,validFunc="PetscValidLogicalCollectiveInt")
  return

def checkPetscValidLogicalCollectiveBool(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveBool(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,boolTypes,successFunction=checkIsPetscBool)
  return

def checkPetscValidLogicalCollectiveEnum(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveEnum(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,enumTypes)
  return


checkFunctionMap = {
  "PetscValidHeaderSpecificType"       : checkPetscValidHeaderSpecificType,
  "PetscValidHeaderSpecific"           : checkPetscValidHeaderSpecific,
  "PetscValidHeader"                   : checkObjIdxGenericN,
  "PetscValidPointer"                  : checkObjIdxGenericN,
  "PetscValidCharPointer"              : checkPetscValidCharPointer,
  "PetscValidIntPointer"               : checkPetscValidIntPointer,
  "PetscValidBoolPointer"              : checkPetscValidBoolPointer,
  "PetscValidScalarPointer"            : checkPetscValidScalarPointer,
  "PetscValidRealPointer"              : checkPetscValidRealPointer,
  "PetscCheckSameType"                 : checkObjIdxGenericN,
  "PetscValidType"                     : checkObjIdxGenericN,
  "PetscCheckSameComm"                 : checkObjIdxGenericN,
  "PetscCheckSameTypeAndComm"          : checkObjIdxGenericN,
  "PetscValidLogicalCollectiveScalar"  : checkPetscValidLogicalCollectiveScalar,
  "PetscValidLogicalCollectiveReal"    : checkPetscValidLogicalCollectiveReal,
  "PetscValidLogicalCollectiveInt"     : checkPetscValidLogicalCollectiveInt,
  "PetscValidLogicalCollectiveMPIInt"  : checkPetscValidLogicalCollectiveMPIInt,
  "PetscValidLogicalCollectiveBool"    : checkPetscValidLogicalCollectiveBool,
  "PetscValidLogicalCollectiveEnum"    : checkPetscValidLogicalCollectiveEnum,
  "VecNestCheckCompatible2"            : checkObjIdxGenericN,
  "VecNestCheckCompatible3"            : checkObjIdxGenericN,
  "MatCheckPreallocated"               : checkObjIdxGenericN,
  "MatCheckProduect"                   : checkObjIdxGenericN,
  "MatCheckSameLocalSize"              : checkObjIdxGenericN,
  "MatCheckSameSize"                   : checkObjIdxGenericN,
  "PetscValidDevice"                   : checkObjIdxGenericN,
  "PetscCheckCompatibleDevices"        : checkObjIdxGenericN,
  "PetscValidDeviceContext"            : checkObjIdxGenericN,
  "PetscCheckCompatibleDeviceContexts" : checkObjIdxGenericN,
}

"""Utility and pre-check setup"""
def osResolvePath(path):
  __doc__="""
  Fully resolve a path, expanding any shell variables, the home variable and making an absolute path
  """
  if path:
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
  return path

def osRemoveSilent(filename):
  __doc__="""
  Silently remove a file, suppressing error if the file does not exist
  """
  try:
    os.remove(filename)
  except OSError as ose:
    import errno
    if ose.errno != errno.ENOENT: # no such file or directory
      raise # re-raise exception if a different error occurred
  return

def subprocessRun(*args,**kwargs):
  __doc__="""
  lightweight wrapper to hoist the ugly version check out of the regular code
  """
  import subprocess,sys

  if sys.version_info >= (3,7):
    output = subprocess.run(*args,**kwargs)
  else:
    if kwargs.pop("capture_output",None):
      kwargs.setdefault("stdout",subprocess.PIPE)
      kwargs.setdefault("stderr",subprocess.PIPE)
    output = subprocess.run(*args,**kwargs)
  return output

def tryToFindLibclangDir():
  __doc__="""
  Crudely tries to find libclang directory first using ctypes.util.find_library(), then llvm-config, and then finally checks a few places on macos
  """
  import ctypes.util

  llvmLibDir = ctypes.util.find_library("clang")
  if not llvmLibDir:
    try:
      output = subprocessRun(["llvm-config","--libdir"],capture_output=True,universal_newlines=True,check=True)
      llvmLibDir = output.stdout.strip()
    except FileNotFoundError:
      # FileNotFoundError: [Errno 2] No such file or directory: 'llvm-config'
      # try to find llvmLibDir by hand
      import platform

      if platform.system().lower() == "darwin":
        try:
          output = subprocessRun(["xcode-select","-p"],capture_output=True,universal_newlines=True,check=True)
          xcodeDir = output.stdout.strip()
          if xcodeDir == "/Applications/Xcode.app/Contents/Developer": # default Xcode path
            llvmLibDir = os.path.join(xcodeDir,"Toolchains","XcodeDefault.xctoolchain","usr","lib")
          elif xcodeDir == "/Library/Developer/CommandLineTools":      # CLT path
            llvmLibDir = os.path.join(xcodeDir,"usr","lib")
        except FileNotFoundError:
          # FileNotFoundError: [Errno 2] No such file or directory: 'xcode-select'
          pass
  return llvmLibDir

def initializeLibclang(clangDir=None,clangLib=None):
  __doc__="""
  Set the required library file or directory path to initialize libclang
  """
  if not clx.conf.loaded:
    clx.conf.set_compatibility_check(True)
    if clangLib:
      clangLib = osResolvePath(clangLib)
      clx.conf.set_library_file(clangLib)
    elif clangDir:
      clangDir = osResolvePath(clangDir)
      clx.conf.set_library_path(clangDir)
    else:
      raise RuntimeError("Must supply either clang directory path or clang library path")
  return

def filterCheckFunctionMap(filterChecks):
  __doc__="""
  Remove checks from checkFunctionMap if they are not in filterChecks
  """
  if filterChecks:
    global checkFunctionMap

    # note the list, this makes a copy of the keys allowing us to delete entries "in place"
    for key in list(checkFunctionMap.keys()):
      if key not in filterChecks:
        del checkFunctionMap[key]
  return

def getPetscExtraIncludes(petscDir,petscArch):
  import re

  # keep these separate, since ORDER MATTERS HERE. Imagine that for example the
  # mpiInclude dir has copies of old petsc headers, you don't want these to come first
  # in the include search path and hence override those found in petsc/include.

  # You might be thinking that seems suspiciously specific, but I was this close to filing
  # a bug report for python believing that cdll.load() was not deterministic...
  petscIncludes = []
  mpiIncludes   = []
  cxxflags      = []
  with open(os.path.join(petscDir,petscArch,"lib","petsc","conf","petscvariables"),"r") as pv:
    ccinc  = re.compile("^PETSC_CC_INCLUDES\s*=")
    mpiinc = re.compile("^MPI_INCLUDE\s*=")
    shoinc = re.compile("^MPICC_SHOW\s*=")
    cxxflg = re.compile("^CXX_FLAGS\s*=")
    line   = pv.readline()
    while line:
      if ccinc.search(line):
        petscIncludes.append(line.split("=",1)[1])
      elif mpiinc.search(line) or shoinc.search(line):
        mpiIncludes.append(line.split("=",1)[1])
      elif cxxflg.search(line):
        cxxflags.append(line.split("=",1)[1])
      line = pv.readline()
  cxxflags      = [l.strip().split(" ") for l in cxxflags if l]
  cxxflags      = [flag for flags in cxxflags for flag in flags if flag.startswith("-std=")]
  cxxflags      = [cxxflags[-1]] if cxxflags else [] # take only the last one
  extraIncludes = [l.strip().split(" ") for l in petscIncludes+mpiIncludes if l]
  extraIncludes = [item for sublist in extraIncludes for item in sublist if item.startswith("-I")]
  seen          = set()
  extraIncludes = [item for item in extraIncludes if not item in seen and not seen.add(item)]
  return cxxflags+extraIncludes

def getClangSysIncludes():
  __doc__="""
  Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.
  """
  output = subprocessRun(["clang","-E","-x","c++","/dev/null","-v"],capture_output=True,check=True,universal_newlines=True)
  # goes to stderr because of /dev/null
  includes = output.stderr.split("#include <...> search starts here:\n")[1]
  includes = includes.split("End of search list.")[0].replace("(framework directory)","")
  includes = includes.split("\n")
  includes = ["-I"+os.path.abspath(i.strip()) for i in includes if i]
  return includes

def buildCompilerFlags(petscDir,petscArch,extraCompilerFlags=[],verbose=False,printPrefix="[ROOT]"):
  __doc__="""
  build the baseline set of compiler flags, these are passed to all translation unit parse attempts
  """
  miscFlags        = ["-D","PETSC_CLANG_STATIC_ANALYZER","-x","c++","-Wno-nullability-completeness"]
  sysincludes      = getClangSysIncludes()
  petscIncludes    = getPetscExtraIncludes(petscDir,petscArch)
  compilerFlags    = sysincludes+miscFlags+petscIncludes+extraCompilerFlags
  if verbose: print("\n".join([printPrefix+" Compile flags:",*compilerFlags]))
  return compilerFlags

def buildPrecompiledHeader(petscDir,compilerFlags,extraHeaderIncludes=[],verbose=False,printPrefix="[ROOT]",pchClangOptions=basePCHClangOptions):
  __doc__="""
  create a precompiled header from petsc.h, and all of the private headers, this not only saves a lot of time, but is critical to finding struct definitions. Header contents are not parsed during the actual linting, since this balloons the parsing time as libclang provides no builtin auto header-precompilation like the normal compiler does.

  Including petsc.h first should define almost everything we need so no side effects from including headers in the wrong order below.
  """
  index             = clx.Index.create()
  precompiledHeader = os.path.join(petscDir,"include","petsc_ast_precompile.pch")
  megaHeaderLines   = [("petsc.h","#include <petsc.h>")]
  privateDirName    = os.path.join(petscDir,"include","petsc","private")
  # build a megaheader from every header in private first
  for headerFile in os.listdir(privateDirName):
    if headerFile.endswith((".h",".hpp")):
      megaHeaderLines.append((headerFile,"#include <petsc/private/{}>".format(headerFile)))
  while True:
    # loop until we get a completely clean compilation, any problematic headers are simply
    # discarded
    megaHeader = "\n".join(hfi for _,hfi in megaHeaderLines)+"\n"  # extra newline for last line
    tu = index.parse("megaHeader.hpp",args=compilerFlags,unsaved_files=[("megaHeader.hpp",megaHeader)],options=pchClangOptions)
    diags = {}
    for diag in tu.diagnostics:
      try:
        filename = diag.location.file.name
      except AttributeError:
        continue
      basename,filename = os.path.split(filename)
      if filename not in diags:
        # save the problematic header name as well as its path (a surprise tool that will
        # help us later)
        diags[filename] = (basename,diag)
    for dirname,diag in tuple(diags.values()):
      # the reason this is done twice is because as usual libclang hides
      # everything in children. Suppose you have a primary header A (which might be
      # include/petsc/private/headerA.h), header B and header C. Header B and C are in
      # unknown locations and all we know is that Header A includes B which includes C.
      #
      # Now suppose header C is missing, meaning that Header A needs to be removed.
      # libclang isn't gonna tell you that without some elbow grease since that would be
      # far too easy. Instead it raises the error about header B, so we need to link it
      # back to header A.
      if dirname != privateDirName:
        # problematic header is NOT in include/petsc/private, so we have a header B on our
        # hands
        for child in diag.children:
          # child of header B here is header A not header C
          try:
            filename = child.location.file.name
          except AttributeError:
            continue
          # filter out our fake header
          if filename != "megaHeader.hpp":
            # this will be include/petsc/private, headerA.h
            basename,filename = os.path.split(filename)
            if filename not in diags:
              diags[filename] = (basename,diag)
    if diags:
      diagerrs = "\n"+"\n".join(str(d) for _,d in diags.values())
      print(printPrefix,"Included header has errors, removing",diagerrs)
      megaHeaderLines = [(hdr,hfi) for hdr,hfi in megaHeaderLines if hdr not in diags]
    else:
      break
  if extraHeaderIncludes:
    # now include the other headers but this time immediately crash on errors, let the
    # user figure out their own busted header files
    megaHeader = megaHeader+"\n".join(extraHeaderIncludes)
    if verbose:
      print("\n".join([printPrefix+" Mega header:",megaHeader]))
    tu = index.parse("megaHeader.hpp",args=compilerFlags,unsaved_files=[("megaHeader.hpp",megaHeader)],options=pchClangOptions)
    if tu.diagnostics:
      print("\n".join(map(str,tu.diagnostics)))
      raise clx.LibclangError("\n\nWarnings or errors generated when creating the precompiled header. This usually means that the provided libclang setup is faulty. If you used the auto-detection mechanism to find libclang then perhaps try specifying the location directly.")
  elif verbose:
    print("\n".join([printPrefix+" Mega header:",megaHeader]))
  osRemoveSilent(precompiledHeader)
  tu.save(precompiledHeader)
  compilerFlags.extend(["-include-pch",precompiledHeader])
  if verbose:
    print(printPrefix,"Saving precompiled header",precompiledHeader)
  return precompiledHeader


"""Main functions for root and queue processes"""
def testMain(petscDir,srcDir,outputDir,patches,replace=False,verbose=False):
  import glob,itertools,difflib

  class TestException(Exception):
    pass

  if not patches:
    raise RuntimeError("outputDir {} provided but no patches generated".format(outputDir))
  returncode = 0
  patchError = {}
  patches    = dict(patches)
  fileList   = []
  for ext in ('c','cxx','cpp','cc','CC'):
    fileList.extend(glob.glob("".join([srcDir,os.path.sep,"*."+ext])))
  for testFile in fileList:
    basename   = os.path.basename(os.path.splitext(testFile)[0])
    outputFile = os.path.join(outputDir,basename+".patch")
    shortName  = testFile.replace(petscDir+os.path.sep,"")

    print("\tTEST   ",shortName)
    try:
      try:
        patch = patches[testFile]
      except KeyError:
        raise TestException("File had no corresponding patch: '{}'\n".format(testFile))
      if replace:
        print("\tREPLACE",shortName)
        patch = "".join(patch.splitlines(True)[2:])
        with open(outputFile,"w") as fd:
          fd.write(patch)
        continue
      elif not os.path.exists(outputFile):
        raise TestException("File had no corresponding output: '{}'\n".format(testFile))

      with open(outputFile,"r") as fd:
        fileLines  = fd.readlines()
        # skip header lines containing date, the output files shouldn't contain them
        patchLines = patch.splitlines(True)[2:]
        diffs      = list(difflib.unified_diff(fileLines,patchLines,n=0))
        if diffs:
          raise TestException("".join(diffs))
      print("\tOK     ",shortName)
    except TestException as te:
      print("\tNOT OK ",shortName)
      patchError[testFile] = str(te)
  if patchError:
    returncode = 21
    errBars    = "".join(["[ERROR]",85*"-","[ERROR]"])
    errBars    = [errBars+"\n",errBars]
    for errFile in patchError:
      print(patchError[errFile].join(errBars))
  return returncode

def queueMain(clangLib,checkFunctionMapU,classIdMapU,compilerFlags,clangOptions,verbose,werror,errorQueue,returnQueue,fileQueue,lock):
  __doc__="""
  main function for worker processes in the queue, does pretty much the same thing the main process would do in their place
  """
  def updateGlobals(updatedCheckFunctionMap,updatedClassIdMap):
    global checkFunctionMap,classIdMap # in a function so the "globalness" doesn't leak
    checkFunctionMap = updatedCheckFunctionMap
    classIdMap       = updatedClassIdMap
    return

  def lockPrint(*args,**kwargs):
    if verbose:
      with lock:
        print(*args,**kwargs)
    return

  # in case errors are thrown before setup is complete
  errorPrefix = "[UNKNOWN_CHILD]"
  filename    = "QUEUE SETUP"
  try:
    updateGlobals(checkFunctionMapU,classIdMapU)
    proc        = mp.current_process().name
    printPrefix = proc+" --"[:len("[ROOT]")-len(proc)]
    errorPrefix = " ".join([printPrefix,"Exception detected while processing"])
    lockPrint(printPrefix,15*"=","Performing setup",15*"=")
    initializeLibclang(clangLib=clangLib)
    linter = PetscLinter(compilerFlags,clangOptions=clangOptions,prefix=printPrefix,verbose=verbose,werror=werror,lock=lock)
    lockPrint(printPrefix,15*"=","Entering queue  ",15*"=")
    while True:
      filename = fileQueue.get()
      if filename == QueueSignal.EXIT_QUEUE:
        fileQueue.task_done()
        break
      linter.parse(filename)
      returnQueue.put((QueueSignal.UNIFIED_DIFF,linter.coalescePatches()))
      errLeft,errFixed = linter.getAllErrors()
      returnQueue.put((QueueSignal.ERRORS_LEFT ,errLeft))
      returnQueue.put((QueueSignal.ERRORS_FIXED,errFixed))
      returnQueue.put((QueueSignal.WARNING     ,linter.getAllWarnings()))
      linter.clear()
      fileQueue.task_done()
    lockPrint(printPrefix,15*"=","Exiting queue   ",15*"=")
  except:
    try:
      # attempt to send the traceback back to parent
      import traceback
      preamble = " ".join([errorPrefix,filename])
      errorQueue.put("\n".join([preamble,traceback.format_exc()]))
    except:
      # if this fails then I guess we really are screwed
      errorQueue.put("[UNKNOWN CHILD] UNKNOWN ERROR")
    finally:
      try:
        # in case we had any work from the queue we need to release it but only after
        # putting our exception on the queue
        fileQueue.task_done()
      except ValueError:
        # task_done() called more times than get(), means we threw before getting the
        # filename
        pass
  errorQueue.close()
  returnQueue.close()
  return

def main(petscDir,petscArch,srcDir=None,clangDir=None,clangLib=None,verbose=False,workers=-1,checkFunctionFilter=None,patchDir=None,applyPatches=False,extraCompilerFlags=[],extraHeaderIncludes=[],testDir=None,replaceTests=False,werror=False):
  __doc__="""
  entry point for linter

  Positional arguments:
  petscDir  -- $PETSC_DIR
  petscArch -- $PETSC_ARCH

  Keyword arguments:
  srcDir              -- alternative directory to use as src root (default: $PETSC_DIR/src)
  clangDir            -- directory containing libclang.[so|dylib|dll] (default: None)
  clangLib            -- direct path to libclang.[so|dylib|dll], overrrides clangDir if set (default: None)
  verbose             -- display debugging statements (default: False)
  workers             -- number of processes for multiprocessing, -1 is number of system CPU's-1, 0 or 1 for serial computation (default: -1)
  checkFunctionFilter -- list of function names as strings to only check for, none == all of them. For example ["PetscValidPointer","PetscValidHeaderSpecific"] (default: None)
  patchDir            -- directory to store patches if they are generated (default: $PETSC_DIR/petscLintPatches)
  applyPatches        -- automatically apply patch files to source if they are generated (default: False)
  extraCompilerFlags  -- list of extra compiler flags to append to petsc and system flags. For example ["-I/my/non/standard/include","-Wsome_warning"] (default: None)
  extraHeaderIncludes -- list of #include statements to append to the precompiled mega-header, these must be in the include search path. Use extraCompilerFlags to make any other search path additions. For example ["#include <slepc/private/epsimpl.h>"] (default: None)
  testDir             -- directory containing test output to compare patches against, use special keyword '__at_src__' to use srcDir/output (default: None)
  replaceTests        -- replace output files in testDir with patches generated (default: False)
  werror              -- treat all linter-generated warnings as errors (default: False)
  """

  # pre-processing setup
  if applyPatches and testDir:
    raise RuntimeError("Test directory and apply patches are both non-zero. It is probably not a good idea to apply patches over the test directory!")
  initializeLibclang(clangDir=clangDir,clangLib=clangLib)
  petscDir = osResolvePath(petscDir)
  if srcDir is None:
    srcDir = os.path.join(petscDir,"src")
  else:
    srcDir = osResolvePath(srcDir)
  if patchDir is None:
    patchDir = os.path.join(petscDir,"petscLintPatches")
  else:
    patchDir = osResolvePath(patchDir)
  if testDir == "__at_src__":
    testDir = os.path.join(srcDir,"output")

  rootPrintPrefix   = "[ROOT]"
  compilerFlags     = buildCompilerFlags(petscDir,petscArch,extraCompilerFlags=extraCompilerFlags,verbose=verbose)
  precompiledHeader = buildPrecompiledHeader(petscDir,compilerFlags,extraHeaderIncludes=extraHeaderIncludes,verbose=verbose)
  filterCheckFunctionMap(checkFunctionFilter)

  pool = WorkerPool(numWorkers=workers,verbose=verbose)
  pool.setup(compilerFlags,werror=werror)
  pool.walk(srcDir)
  warnings,errorsLeft,errorsFixed,patches = pool.finalize()
  if verbose: print(rootPrintPrefix,"Deleting precompiled header",precompiledHeader)
  osRemoveSilent(precompiledHeader)
  if testDir is not None:
    return testMain(petscDir,srcDir,testDir,patches,replace=replaceTests,verbose=verbose)
  if patches:
    import time

    try:
      os.mkdir(patchDir)
    except FileExistsError:
      pass
    manglePostfix = "".join(["_",str(int(time.time())),".patch"])
    for filename,patch in patches:
      filename    = filename.replace(srcDir,"").replace(os.path.sep,"_")[1:]
      mangledFile = os.path.splitext(filename)[0]+manglePostfix
      mangledFile = os.path.join(patchDir,mangledFile)
      if verbose: print(rootPrintPrefix,"Writing patch to file",mangledFile)
      with open(mangledFile,"w") as fd:
        fd.write(patch)
    if applyPatches:
      import glob

      if verbose: print(rootPrintPrefix,"Applying patches from patch directory",patchDir)
      rootDir   = "".join(["-d",os.path.abspath(os.path.sep)])
      patchGlob = "".join([patchDir,os.path.sep,"*",manglePostfix])
      for patchFile in glob.iglob(patchGlob):
        if verbose: print(rootPrintPrefix,"Applying patch",patchFile)
        output = subprocessRun(["patch",rootDir,"-p0","--unified","-i",patchFile],check=True,universal_newlines=True,capture_output=True)
        if verbose: print(output.stdout)
  returnCode = 0
  if warnings and verbose:
    print("\n"+rootPrintPrefix,30*"=","Found warnings      ",33*"=")
    print("\n".join(s for tup in warnings for _,s in tup))
    print(rootPrintPrefix,30*"=","End warnings        ",33*"=")
  if errorsFixed and verbose:
    print("\n"+rootPrintPrefix,30*"=","Fixed Errors        ",33*"=")
    print("\n".join(errorsFixed))
    print(rootPrintPrefix,30*"=","End fixed errors    ",33*"=")
  if errorsLeft:
    print("\n"+rootPrintPrefix,30*"=","Unfixable Errors    ",33*"=")
    print("\n".join(errorsLeft))
    print(rootPrintPrefix,30*"=","End unfixable errors",33*"=")
    returnCode = 11
    print("Some errors or warnings could not be automatically corrected via the patch files, see above")
  elif patches:
    if applyPatches:
      print("\nAll errors or warnings successfully patched")
    else:
      rootDir = "".join(["-d",os.path.abspath(os.path.sep)])
      print("\nAll errors fixable via patch files written to",patchDir)
      patchGlob = "*".join([patchDir+os.path.sep,manglePostfix])
      print("Apply manually using:\n\tpatch {} -p0 --unified -i {}".format(rootDir,patchGlob))
      returnCode = 12
  return returnCode


if __name__ == "__main__":
  import argparse
  def str2bool(v):
    if isinstance(v,bool):
      return v
    v = v.lower()
    if v in {"yes","true","t","y","1"}:
      return True
    elif v in {"no","false","f","n","0",""}:
      return False
    else:
      raise argparse.ArgumentTypeError("Boolean value expected, got '{}'".format(v))

  clangDir = tryToFindLibclangDir()
  try:
    petscDir      = os.environ["PETSC_DIR"]
    defaultSrcDir = os.path.join(petscDir,"src")
  except KeyError:
    petscDir      = None
    defaultSrcDir = "$PETSC_DIR/src"
  try:
    petscArch = os.environ["PETSC_ARCH"]
  except KeyError:
    petscArch = None

  parser = argparse.ArgumentParser(description="set options for clang static analysis tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  grouplibclang = parser.add_argument_group(title="libclang location settings")
  group = grouplibclang.add_mutually_exclusive_group(required=False)
  group.add_argument("--clang_dir",nargs="?",help="directory containing libclang.[so|dylib|dll], if not given attempts to automatically detect it via llvm-config",default=clangDir,dest="clangdir")
  group.add_argument("--clang_lib",nargs="?",help="direct location of libclang.[so|dylib|dll], overrides clang directory if set",dest="clanglib")
  grouppetsc = parser.add_argument_group(title="petsc location settings")
  grouppetsc.add_argument("--PETSC_DIR",required=False,default=petscDir,help="if this option is unused defaults to environment variable $PETSC_DIR",dest="petscdir")
  grouppetsc.add_argument("--PETSC_ARCH",required=False,default=petscArch,help="if this option is unused defaults to environment variable $PETSC_ARCH",dest="petscarch")
  parser.add_argument("-s","--src-dir",required=False,default=defaultSrcDir,help="Alternate base directory of source tree (e.g. $SLEPC_DIR/src)",dest="src")
  parser.add_argument("-v","--verbose",required=False,type=str2bool,nargs="?",const=True,default=False,help="verbose progress printed to screen")
  filterFuncChoices = ", ".join(list(checkFunctionMap.keys()))
  parser.add_argument("-f","--functions",required=False,nargs="+",choices=list(checkFunctionMap.keys()),metavar="FUNCTIONNAME",help="filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: "+filterFuncChoices,dest="funcs")
  parser.add_argument("-j","--jobs",required=False,type=int,const=-1,default=-1,nargs="?",help="number of multiprocessing jobs, -1 means number of processors on machine")
  parser.add_argument("-p","--patch-dir",required=False,help="directory to store patches in if they are generated, defaults to SRC_DIR/../petscLintPatches",dest="patchdir")
  parser.add_argument("-a","--apply-patches",required=False,type=str2bool,nargs="?",const=True,default=False,help="automatically apply patches that are saved to file",dest="apply")
  parser.add_argument("--CXXFLAGS",required=False,nargs="+",default=[],help="extra flags to pass to CXX compiler",dest="cxxflags")
  parser.add_argument("--test",required=False,nargs="?",const="__at_src__",help="test the linter for correctness. Optionally provide a directory containing the files against which to compare patches, defaults to SRC_DIR/output if no argument is given. The files of correct patches must be in the format [path_from_src_dir_to_testFileName].out")
  parser.add_argument("--replace",required=False,type=str2bool,nargs="?",const=True,default=False,help="replace output files in test directory with patches generated")
  parser.add_argument("--werror",required=False,type=str2bool,nargs="?",const=True,default=False,help="treat all warnings as errors")
  args = parser.parse_args()

  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.clanglib:
    args.clangdir = None

  if args.src == "$PETSC_DIR/src":
    args.src = os.path.join(petscDir,"src")

  ret = main(args.petscdir,args.petscarch,srcDir=args.src,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,workers=args.jobs,checkFunctionFilter=args.funcs,patchDir=args.patchdir,applyPatches=args.apply,extraCompilerFlags=args.cxxflags,testDir=args.test,replaceTests=args.replace,werror=args.werror)
  sys.exit(ret)
