classdef PetscSNES < PetscObject
  methods
    function obj = PetscSNES(pid,flg)
      if (nargin > 1) 
        %  PetscSNES(pid,'pobj') uses an already existing PETSc SNES object
        obj.pobj = pid;
        return
      end
      [err,obj.pobj] = calllib('libpetsc', 'SNESCreate', PETSC_COMM_SELF,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'SNESSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'SNESSetDM', obj.pobj,da.pobj);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'SNESSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'SNESSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj,b,x)
      if (nargin < 3) 
        err = calllib('libpetsc', 'SNESSolve', obj.pobj,0,b.pobj);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'SNESSolve', obj.pobj,b.pboj,x.pobj);PetscCHKERRQ(err);
      end
    end
    function err = VISetVariableBounds(obj,xl,xb)
        for i=1:length(xl(:))
        if isinf(xl(i)) xl(i) = -1.0e20; end
        if isinf(xb(i)) xb(i) = 1.0e20;end
      end
      err = calllib('libpetsc', 'SNESVISetVariableBounds', obj.pobj,xl.pobj,xb.pobj);PetscCHKERRQ(err);
    end
    function err = SetFunction(obj,f,func,arg)
      if (nargin < 4) 
        arg = 0;
      end
      err = calllib('libpetsc', 'SNESSetFunctionMatlab', obj.pobj,f.pobj,func,arg);PetscCHKERRQ(err);
    end
    function err = SetJacobian(obj,A,B,func,arg)
      if (nargin < 5) 
        arg = 0;
      end
      err = calllib('libpetsc', 'SNESSetJacobianMatlab', obj.pobj,A.pobj,B.pobj,func,arg);PetscCHKERRQ(err);
    end
    function err = MonitorSet(obj,func,arg)
      if (nargin < 3) 
        arg = 0;
      end
      err = calllib('libpetsc', 'SNESMonitorSetMatlab', obj.pobj,func,arg);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libpetsc', 'SNESView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libpetsc', 'SNESView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'SNESDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
