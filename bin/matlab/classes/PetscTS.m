classdef PetscTS < PetscObject
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
      [err,obj.pobj] = calllib('libpetsc', 'TSCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'TSSetType', obj.pobj,name);
    end
    function err = SetProblemType(obj,t)
      err = calllib('libpetsc', 'TSSetProblemType', obj.pobj,t);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'TSSetDM', obj.pobj,da.pobj);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'TSSetFromOptions', obj.pobj);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'TSSetUp', obj.pobj);
    end
    function err = Solve(obj,x)
      err = calllib('libpetsc', 'TSSolve', obj.pobj,x.pobj);
    end
    function err = SetFunction(obj,func,arg)
      if (nargin < 3) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSSetFunctionMatlab', obj.pobj,func,arg);
    end
    function err = SetJacobian(obj,A,B,func,arg)
      if (nargin < 5) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSSetJacobianMatlab', obj.pobj,A.pobj,B.pobj,func,arg);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'TSView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'TSDestroy', obj.pobj);
    end
  end
end

 
