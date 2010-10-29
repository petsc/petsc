classdef PetscTS < PetscObject
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
      if (nargin < 2) 
        err = calllib('libpetsc', 'TSSolve', obj.pobj,0);
      else
        err = calllib('libpetsc', 'TSSolve', obj.pobj,x.pboj);
      end
    end
    function err = SetIFunction(obj,func,arg)
      if (nargin < 3) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSSetIFunctionMatlab', obj.pobj,func,arg);
    end
    function err = SetIJacobian(obj,A,B,func,arg)
      if (nargin < 5) 
        arg = 0;
      end
      err = calllib('libpetsc', 'TSSetIJacobianMatlab', obj.pobj,A.pobj,B.pobj,func,arg);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'TSView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'TSDestroy', obj.pobj);
    end
  end
end

 
