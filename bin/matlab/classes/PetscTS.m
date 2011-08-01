classdef PetscTS < PetscObject
%
%  PetscTS - Manages time-integration
%
%  Creation:
%    ts = PetscTS;
%      ts.SetType('gl');
%      ts.SetFunction(ts,funcname);
%      ts.SetJacobian(ts,A,B,funcname);
%      ts.SetFromOptions;
%
  properties (Constant)
    LINEAR=0;
    NONLINEAR=1;
  end
  methods
    function obj = PetscTS(pid,flg)
      if (nargin > 1) 
        %  PetscTS(pid,'pobj') uses an already existing PETSc TS object
        obj.pobj = pid;
        return
      end
      comm = PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libpetsc', 'TSCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'TSSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetProblemType(obj,t)
      err = calllib('libpetsc', 'TSSetProblemType', obj.pobj,t);PetscCHKERRQ(err);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'TSSetDM', obj.pobj,da.pobj);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'TSSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'TSSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function [err,ftime] = Solve(obj,x)
      ftime=0.0;
      [err,ftime] = calllib('libpetsc', 'TSSolve', obj.pobj,x.pobj,ftime);PetscCHKERRQ(err);
    end
    function err = SetFunction(obj,func,arg)
      if (nargin < 3) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSSetFunctionMatlab', obj.pobj,func,arg);PetscCHKERRQ(err);
    end
    function err = SetJacobian(obj,A,B,func,arg)
      if (nargin < 5) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSSetJacobianMatlab', obj.pobj,A.pobj,B.pobj,func,arg);PetscCHKERRQ(err);
    end
    function err = MonitorSet(obj,func,arg)
      if (nargin < 3) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSMonitorSetMatlab', obj.pobj,func,arg);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'TSView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'TSView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'TSDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end
