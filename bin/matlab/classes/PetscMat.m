classdef PetscMat < PetscObject
%
%    PetscMat - Holds a PETSc sparse matrix or linear operator
%
%   Creation:
%     A = PetscMat;
%       A.SetType('seqaij');
%       A.SetSizes(3,3);
%     
%     A = PetscMat(speye(3,3));
%
%   If A is a PetscMat then a = A(:,:) returns the MATLAB version of the sparse matrix
%       and A(:,:) = a; assigns the sparse matrix values into the PETScMat
%       you CANNOT yet use syntax like A(1,2) = 1.0
%
%   Indexing into PETSc Vecs and Mats from Matlab starts with index of 1, NOT 0 like 
%     everywhere else in PETSc, but Shri felt MATLAB users could not handle the 0 index
%
  properties (Constant)
    FLUSH_ASSEMBLY=1;
    FINAL_ASSEMBLY=0;

    SAME_NONZERO_PATTERN=0;
    DIFFERENT_NONZERO_PATTERN=1;
    SAME_PRECONDITIONER=2;
    SUBSET_NONZERO_PATTERN=3;
  end
  methods
    function obj = PetscMat(array,flg)
      if (nargin > 1) 
        %  PetscMat(pid,'pobj') uses an already existing PETSc Mat object
        obj.pobj = array;
      elseif (nargin == 1)
        if (~issparse(array)) 
          error('When creating PetscMat from Matlab matrix the Matlab matrix must be sparse');
        end
        comm = PETSC_COMM_SELF();
        [err,obj.pobj] = calllib('libpetsc', 'MatCreate',comm ,0);PetscCHKERRQ(err);
        err = calllib('libpetsc', 'MatSetType', obj.pobj,'seqaij');PetscCHKERRQ(err);
        err  = calllib('libpetsc', 'MatSeqAIJFromMatlab',array',obj.pobj);PetscCHKERRQ(err);
      else 
        comm = PETSC_COMM_SELF();
        [err,obj.pobj] = calllib('libpetsc', 'MatCreate', comm,0);PetscCHKERRQ(err);
      end
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'MatSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'MatSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'MatSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetSizes(obj,m,n,M,N)
      if (nargin == 3) 
        M = Petsc.DECIDE;
        N = Petsc.DECIDE;
      end
      err = calllib('libpetsc', 'MatSetSizes', obj.pobj,m,n,M,N);PetscCHKERRQ(err);
    end
    function [m,n,err] = GetSize(obj)
      m = 0;
      n = 0;
      [err,m,n] = calllib('libpetsc', 'MatGetLocalSize', obj.pobj,m,n);PetscCHKERRQ(err);
    end
    function err = SetValues(obj,idx,idy,values,insertmode)
      idx = idx - 1;
      idy = idy - 1;
      if (nargin < 5) 
        insertmode = Petsc.INSERT_VALUES;
      end
      err = calllib('libpetsc', 'MatSetValues', obj.pobj,length(idx),idx,length(idy),idy,values,insertmode);PetscCHKERRQ(err);
    end
    function err = AssemblyBegin(obj,mode)
      err = calllib('libpetsc', 'MatAssemblyBegin', obj.pobj,mode);PetscCHKERRQ(err);
    end
    function err = AssemblyEnd(obj,mode)
      err = calllib('libpetsc', 'MatAssemblyEnd', obj.pobj,mode);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'MatView', obj.pobj,0);PetscCHKERRQ(err);
      else 
        err = calllib('libpetsc', 'MatView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Load(obj,viewer)
      err = calllib('libpetsc', 'MatLoad', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'MatDestroy', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetValuesStencil(obj,row,col,values,insertmode)
      if (nargin < 5) 
        insertmode = Petsc.INSERT_VALUES;
      end
      ndim = isfield(row,'i') + isfield(row,'j') + isfield(row,'k'); 
      nrow = length(row);
      ncol = length(col);
      if (ndim == 1)  %% 1D DM
	for (m=1:nrow)
            row(m).i = row(m).i - 1;
        end
        for (m = 1:ncol)
             col(m).i = col(m).i - 1;
        end
      elseif (ndim == 2)  %% 2D DM
        for (m = 1:nrow)
          row(m).i = row(m).i - 1;
          row(m).j = row(m).j - 1;
        end
        for (m = 1:ncol)
          col(m).i = col(m).i - 1;
          col(m).j = col(m).j - 1;
        end 
      elseif (ndim == 3)  %% 3D DM
        for (m = 1:nrow)
          row(m).i = row(m).i - 1;
          row(m).j = row(m).j - 1;
          row(m).k = row(m).k - 1;
        end
        for (m = 1:ncol)
          col(m).i = col(m).i - 1;
          col(m).j = col(m).j - 1;
          col(m).k = col(m).k - 1;
        end 
      end
      err = calllib('libpetsc','MatSetValuesStencil',obj.pobj,nrow,row,ncol,col,values,insertmode);PetscCHKERRQ(err);  
    end
%
%   The following overload a = x(:)
%
    function varargout = subsref(obj,S)
      %  Matlab design of subsref is MORONIC
      %  To overload () it automatically overloads . which is used
      %  for method calls so we need to force the method calls to use
      %  the 'regular' subsref
      if (S(1).type == '.')
        [varargout{1:nargout}] = builtin('subsref', obj, S);
      else
        [A] = calllib('libpetsc', 'MatSeqAIJToMatlab', obj.pobj);
        varargout = {A}';
      end
    end
%
%   The following overload x(:,:) = a
%
    function obj = subsasgn(obj,S,value)
      if (S(1).type ~= '.')
        err  = calllib('libpetsc', 'MatSeqAIJFromMatlab',value',obj.pobj);PetscCHKERRQ(err);
      end
    end

  end
end

 
