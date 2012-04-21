classdef PetscIS < PetscObject
%
%   PetscIS - PETSc object representing a collection of integers used for indexing
%
%   Creation:
%     is  = PetscIS();
%     is.SetType("general");
%     is.GeneralSetIndices(indices);
%
%     is  = PetscIS(indices);
% 
  methods
    function obj = PetscIS(indices,flag)
      if (nargin > 1) 
        %  PetscIS(pid,'pobj') uses an already existing PETSc IS object
        obj.pobj = indices;
        return
      end
      comm = PETSC_COMM_SELF();
      if (nargin == 1) 
        [err,obj.pobj] = calllib('libpetsc', 'ISCreate',comm ,0);PetscCHKERRQ(err);
        err = calllib('libpetsc', 'ISSetType', obj.pobj,'general');PetscCHKERRQ(err);
        indices = indices-1;
        err = calllib('libpetsc', 'ISGeneralSetIndices', obj.pobj,length(indices),indices,Petsc.COPY_VALUES);PetscCHKERRQ(err);
      else
        [err,obj.pobj] = calllib('libpetsc', 'ISCreate',comm ,0);PetscCHKERRQ(err);
      end
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'ISSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function [n,err] = GetSize(obj)
      n = 0;
      [err,n] = calllib('libpetsc', 'ISGetSize', obj.pobj,n);PetscCHKERRQ(err);
    end
    function err = GeneralSetIndices(obj,indices)  
      indices = indices - 1;  
      err = calllib('libpetsc', 'ISGeneralSetIndices', obj.pobj,length(indices),indices,Petsc.COPY_VALUES);PetscCHKERRQ(err);
    end
    function [indices,err] = GetIndices(obj,idx)
        [n,err] = GetSize(obj);
        indices = zeros(n,1);
        [err,indices] = calllib('libpetsc','ISGetIndicesCopy',obj.pobj,indices);
        indices = indices + 1;
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'ISView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'ISView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'ISDestroy', obj.pobj);PetscCHKERRQ(err);
    end
    %
%   The following overload a = x(idx)
    function varargout = subsref(obj,S)
      if (S(1).type == '.')
        [varargout{1:nargout}] = builtin('subsref', obj, S);
      else  
        varargout = {obj.GetIndices(S.subs{1})};
      end
    end
  end
end

 

