classdef PetscKSP < PetscObject
%
%   PetscKSP - a PETSc linear solver object
%
%   Creation:
%     ksp = PetscKSP;
%       ksp.SetType('gmres');
%       ksp.SetOperators(A,A,PetscMat.SAME_NONZERO_PATTERN);
%       ksp.SetFromOptions;
%
  methods
    function obj = PetscKSP(pid,flag)
      if (nargin > 1) 
        %  PetscKSP(pid,'pobj') uses an already existing PETSc KSP object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libpetsc', 'KSPCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'KSPSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'KSPSetDM', obj.pobj,da.pobj);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'KSPSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'KSPSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj,b,x)
      if (nargin == 1) 
        b = 0;
        x = 0;
      end
      if (b ~= 0) 
        b = b.pobj;
      end
      if (x ~= 0)
        x = x.pobj;
      end
      err = calllib('libpetsc', 'KSPSolve', obj.pobj,b,x);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,A,B,pattern)
      if (nargin == 2) 
        B = A;
        pattern = PetscMat.SAME_NONZERO_PATTERN;
      end
      err = calllib('libpetsc', 'KSPSetOperators', obj.pobj,A.pobj,B.pobj,pattern);PetscCHKERRQ(err);
    end
    function [x,err] = GetSolution(obj)
      [err,pid] = calllib('libpetsc', 'KSPGetSolution', obj.pobj,0);PetscCHKERRQ(err);
      x = PetscVec(pid,'pobj');
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'KSPView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'KSPView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function [pc,err] = GetPC(obj)
      [err,pid] = calllib('libpetsc', 'KSPGetPC', obj.pobj,0);PetscCHKERRQ(err);
      pc = PetscPC(pid,'pobj');
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'KSPDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
