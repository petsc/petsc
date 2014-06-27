classdef PetscVec < PetscObject
%
%   PetscVec - Holds field variables, right hand sides to linear systems etc.
%
%   Creation:
%     v = PetscVec;
%       v.SetType('seq');
%       v.SetSizes(3);
%     
%     v = PetscVec([1 2 3]);
%
%   If v is a PetscVec then a = v(:) returns a MATLAB array of the vector
%       and v(:) = a; assigns the array values in a into the vector. 
%       v(1:3) = [2.0 2. 3.0]; also work
%
%   Indexing into PETSc Vecs and Mats from MATLAB starts with index of 1, NOT 0 like 
%     everywhere else in PETSc, but Shri felt MATLAB users could not handle 0.
%
  properties
    VecFromDM=0;
    DM = [];
  end
  methods
    function [obj] = SetVecfromDM(obj,value)
      obj.VecFromDM = value;
    end
    function [obj] = SetDM(obj,DM)
        obj.DM = DM;
    end
    function obj = PetscVec(array,flg)
      if (nargin > 1) 
        %  PetscVec(pid,'pobj') uses an already existing PETSc Vec object
        obj.pobj = array;
        return
      end
      comm = PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libpetsc', 'VecCreate',comm ,0);PetscCHKERRQ(err);
      if (nargin > 0) 
        % Vec(array) creates a Vec initialized with the given array
        err = calllib('libpetsc', 'VecSetType', obj.pobj,'seq');PetscCHKERRQ(err);
        err = calllib('libpetsc', 'VecSetSizes', obj.pobj,length(array),length(array));PetscCHKERRQ(err);
        idx = 0:length(array)-1;
        err = calllib('libpetsc', 'VecSetValues', obj.pobj,length(idx),idx,array,Petsc.INSERT_VALUES);PetscCHKERRQ(err);
        err = calllib('libpetsc', 'VecAssemblyBegin', obj.pobj);PetscCHKERRQ(err);
        err = calllib('libpetsc', 'VecAssemblyEnd', obj.pobj);PetscCHKERRQ(err);
      end
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'VecSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'VecSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetSizes(obj,m,n)
      if (nargin == 2) 
        n = Petsc.DECIDE;
      end
      err = calllib('libpetsc', 'VecSetSizes', obj.pobj,m,n);PetscCHKERRQ(err);
    end
    function [n,err] = GetSize(obj)
      n = 0;
      [err,n] = calllib('libpetsc', 'VecGetLocalSize', obj.pobj,n);PetscCHKERRQ(err);
    end
    function err = SetValues(obj,idx,values,insertmode)
      if (ischar(idx)) % assume it is ':' 
        [n,err] = GetSize(obj);
        idx = 1:n;
      end
      if (nargin < 3) 
        values = idx;
      end
      if (nargin  < 4) 
        insertmode = Petsc.INSERT_VALUES;
      end
      idx = idx - 1;
      err = calllib('libpetsc', 'VecSetValues', obj.pobj,length(idx),idx,values,insertmode);PetscCHKERRQ(err);
    end
    function [values,err] = GetValues(obj,idx)
      if (ischar(idx)) % assume it is ':' 
        [n,err] = GetSize(obj);
        idx = 1:n;
      end
      idx = idx - 1;
      values = zeros(1,length(idx));
      [err,idx,values] = calllib('libpetsc', 'VecGetValues', obj.pobj,length(idx),idx,values);PetscCHKERRQ(err);
      values = values'; % Want to return a column vector since that is more natural in MATLAB
    end
    function err = AssemblyBegin(obj)
      err = calllib('libpetsc', 'VecAssemblyBegin', obj.pobj);PetscCHKERRQ(err);
    end
    function err = AssemblyEnd(obj)
      err = calllib('libpetsc', 'VecAssemblyEnd', obj.pobj);PetscCHKERRQ(err);
    end
    function [vec,err] = Duplicate(obj)
      [err,pid] = calllib('libpetsc', 'VecDuplicate', obj.pobj,0);PetscCHKERRQ(err);
      vec = PetscVec(pid,'pobj');
    end
    function err = Copy(obj,v)
      err = calllib('libpetsc', 'VecCopy', obj.pobj,v.pobj);PetscCHKERRQ(err);
    end
    function err = Set(obj,v)
      err = calllib('libpetsc', 'VecSet', obj.pobj,v);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'VecView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'VecView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'VecDestroy', obj.pobj);PetscCHKERRQ(err);
    end
%
%   The following overload a = x(idx)
%
    function varargout = subsref(obj,S)
      %  MATLAB design of subsref is MORONIC
      %  To overload () it automatically overloads . which is used
      %  for method calls so we need to force the method calls to use
      %  the 'regular' subsref
      if (S(1).type == '.')
        [varargout{1:nargout}] = builtin('subsref', obj, S);
      else
        if (obj.VecFromDM)
          varargout = {obj.GetValues_DM(S)};
        else
          varargout = {obj.GetValues(S.subs{1})};
        end
      end
    end
    function varargout = GetValues_DM(obj,S)
      M = obj.DM.M; N = obj.DM.N; P = obj.DM.P;
      ndim = obj.DM.ndim;
      if (ndim == 1)  %% 1D DM
        idx = S.subs{1};
      elseif (ndim == 2)  %% 2D DM
        idx = M*(S.subs{2}-1) + S.subs{1};  
      elseif (ndim == 3)  %% 3D DM
        idx = N*M(S.subs{3}-1) + M*(S.subs{2}-1) + S.subs{1};
      end
      varargout = {obj.GetValues(idx)};      
    end
%
%   The following overload x(idx) = a
%
    function obj = subsasgn(obj,S,value)
      if (S(1).type ~= '.')
        if (obj.VecFromDM)
          [obj] = obj.SetValues_DM(S,value);
        else
          obj.SetValues(S.subs{1},value);
        end
      end
    end
    function [obj] = SetValues_DM(obj,S,value)
      M = obj.DM.M; N = obj.DM.N; P = obj.DM.P;
      ndim = obj.DM.ndim;
      if (ndim == 1)  %% 1D DM
        idx = S.subs{1};
      elseif (ndim == 2)  %% 2D DM
        idx = M*(S.subs{2}-1) + S.subs{1};  
      elseif (ndim == 3)  %% 3D DM
        idx = N*M(S.subs{3}-1) + M*(S.subs{2}-1) + S.subs{1};
      end
      obj.SetValues(idx,value);      
    end
  end
end
