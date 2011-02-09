import os
import sys
import subprocess
import signal
import string

class Example:
    def __init__(self,example,nprocs=1,options="",method=None,tags=[],name=""):
        self.example=example
        self.nprocs=nprocs
        self.options=options.split(" ")
        self.method=method
        self.tags=tags
        self.name=name

    def executableName(self,version=1):
        if version==1:
            return self.example
        elif version==2:
            return 'test_'+self.example
        else:
            return "Bad TAO version (%d)" % version

    def buildCommand(self,version=2):
        if version ==1 or version==2:
            return ['make', self.executableName(version)]

        
    def runCommand(self,version=2):
        if version==1:
            c = [os.path.join('.',self.example)]
        elif version==2:
            c = [os.path.join('.','test_%s'%(self.example))]
        else:
            return "Bad TAO version (%d)" % version
        c.extend(self.options)
        c.extend(['-tao_method','tao_'+self.method])
        return c

    def getTao1Directory(self):
        section=""
        for t in ["unconstrained","bound"]:
            if self.hasTag(t):
                section=t
                break
        if t:
            return os.path.join("src",t,"examples","tutorials")
        if self.hasTag("meshrefinement"):
            return os.path.join("src","petsctao","gridapplication","examples")
            

    def hasTag(self,tag):
        return (tag in self.tags)

class ExampleList:
    def __init__(self):
        self.list = []

    def get_children(self,pid):
        p = subprocess.Popen('ps --no-heading -o pid --ppid %d' % pid, shell = True,
              stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = p.communicate()
        pidlist = []
        for child in [int(p) for p in stdout.split()]:
            pidlist.append(child)
            pidlist.extend(self.get_children(child))
        return pidlist

    def execute(self,command, cwd = None, echo=False,timeout = 60):
        class Alarm(Exception):
            pass
        def alarm_handler(sgnl, frame):
            raise Alarm
        if (echo):
            sys.stdout.write(string.join(command," ")+"\n")
        sys.stdout.flush()
        try:
            p = subprocess.Popen(command, shell = False, cwd = cwd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        except OSError, e:
            sys.stderr.write("Could not execute '%s':\n  %s\n" % (string.join(command),str(e)))
            return (-1,"","")
        if timeout > 0:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timeout)
        try:
            stdout, stderr = p.communicate()
            if timeout > 0:
                signal.alarm(0)
        except Alarm:
            pids = [p.pid]
            pids.extend(self.get_children(p.pid))
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            return -9, p.stdout.read(), 'ERROR: %s timed out after %d seconds\n' % (string.join(command," "),timeout)
        return p.returncode, stdout, stderr





    def add(self,example):
        if example not in self.list:
            self.list.append(example)
    
    def withTag(self,tag):
        l = []
        for e in self.list:
            if e.hasTag(tag):
                l.append(e)
        return l

class TaoExamples(ExampleList):
    def __init__(self):
        self.list = []

        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_smonitor",method="ntr",tags=["rosenbrock","single","unconstrained","c","ntr"],name="rosenbrockntr"))

        # Check gradients and hessians of examples
        for n in ["minsurf1","eptorsion1","rosenbrock1","rosenbrock1f"]:
            self.add(Example(example=n,nprocs=1,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name=n+"_fd_test"))
        for n in ["plate2","plate2f","eptorsion2","eptorsion2f"]:
            self.add(Example(example=n,nprocs=2,options="-tao_fd_test_gradient -tao_fd_test_hessian",method="fd_test",tags=["fd_test"],name=n+"_fd_test"))
            
            
                 
        # Unconstrained
        self.add(Example(example="minsurf1",nprocs=1,options="-tao_smonitor -mx 10 -my 8",method="nls",tags=["minsurf","single","unconstrained","c","nls"],name="minsurf1"))
        self.add(Example(example="minsurf2",nprocs=1,options="-tao_smonitor -mx 10 -my 8",method="lmvm",tags=["minsurf","single","unconstrained","dm","c","lmvm"],name="minsurf2"))
        self.add(Example(example="minsurf2",nprocs=2,options="-tao_smonitor",method="nls",tags=["minsurf","multiprocessor","unconstrained","dm","c","nls"],name="minsurf2_2"))
        self.add(Example(example="minsurf2",nprocs=3,options="-tao_smonitor -mx 10 -my 10 -tao_cg_type fr",method="cg",tags=["minsurf","multiprocessor","unconstrained","dm","c","cg"],name="minsurf2_3"))
        self.add(Example(example="minsurf2",nprocs=2,options="-tao_smonitor -my 6 -my 8",method="ntr",tags=["minsurf","multiprocessor","unconstrained","dm","c","ntr"],name="minsurf2_4"))
        self.add(Example(example="minsurf2",nprocs=3,options="-tao_smonitor -my 23 -my 17",method="nls",tags=["minsurf","multiprocessor","unconstrained","dm","c","nls"],name="minsurf2_5"))
        self.add(Example(example="minsurf2",nprocs=1,options="-tao_smonitor -mx 4 -my 20 -random 2",method="ntr",tags=["minsurf","single","unconstrained","dm","c","ntr"],name="minsurf2_6"))

        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","c","lmvm"],name="rosenbrock1"))
        self.add(Example(example="rosenbrock1",nprocs=1,options="-tao_smonitor -tao_frtol 0 -tao_fatol 0",method="pounder",tags=["rosenbrock","single","unconstrained","c","pounder"],name="rosenbrock1_2"))
        self.add(Example(example="rosenbrock1f",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["rosenbrock","single","unconstrained","fortran","lmvm"],name="rosenbrock1f"))

        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["eptorsion","single","unconstrained","c","lmvm"],name="eptorsion1"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","c","nls"],name="eptorsion1_2"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor -tao_cg_type prp",method="cg",tags=["eptorsion","single","unconstrained","c","cg"],name="eptorsion1_3"))
        self.add(Example(example="eptorsion1",nprocs=1,options="-tao_smonitor",method="ntr",tags=["eptorsion","single","unconstrained","c","ntr"],name="eptorsion1_4"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","c","nls","dm"],name="eptorsion2"))
        self.add(Example(example="eptorsion2",nprocs=2,options="-tao_smonitor",method="nls",tags=["eptorsion","multiprocessor","unconstrained","c","nls","dm"],name="eptorsion2_2"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="ntr",tags=["eptorsion","multiprocessor","unconstrained","c","ntr","dm"],name="eptorsion2_3"))
        self.add(Example(example="eptorsion2",nprocs=3,options="-tao_smonitor -mx 16 -my 16",method="ntr",tags=["eptorsion","multiprocessor","unconstrained","c","ntr","dm"],name="eptorsion2_4"))
        self.add(Example(example="eptorsion2",nprocs=1,options="-tao_smonitor",method="lmvm",tags=["eptorsion","single","unconstrained","c","lmvm","dm"],name="eptorsion2_5"))
        self.add(Example(example="eptorsion2",nprocs=3,options="-tao_smonitor -mx 16 -my 16",method="lmvm",tags=["eptorsion","multiprocessor","unconstrained","c","lmvm","dm"],name="eptorsion2_5"))
        self.add(Example(example="eptorsion2f",nprocs=1,options="-tao_smonitor",method="nls",tags=["eptorsion","single","unconstrained","fortran","nls","dm"],name="eptorsion2f"))
        self.add(Example(example="eptorsion2f",nprocs=2,options="-tao_smonitor",method="nls",tags=["eptorsion","multiprocessor","unconstrained","fortran","nls","dm"],name="eptorsion2f_2"))
        self.add(Example(example="eptorsion2f",nprocs=3,options="-tao_smonitor -mx 16 -my 16",method="lmvm",tags=["eptorsion","multiprocessor","unconstrained","fortran","lmvm","dm"],name="eptorsion2f_3"))

        # Bound constrained
        self.add(Example(example="plate2",nprocs=1,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="tron",tags=["bound","plate","single","c","tron","dm"],name="plate2"))
        self.add(Example(example="plate2",nprocs=2,options="-tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3",method="blmvm",tags=["bound","plate","multiprocessor","c","blmvm","dm"],name="plate2_2"))
        self.add(Example(example="plate2",nprocs=3,options="-tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1",method="tron",tags=["bound","plate","multiprocessor","c","tron","dm"],name="plate2_3"))
        self.add(Example(example="plate2f",nprocs=1,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="blmvm",tags=["bound","plate","single","fortran","tron","dm"],name="plate2f"))
        self.add(Example(example="plate2f",nprocs=2,options="-tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2",method="blmvm",tags=["bound","plate","multiprocessor","fortran","blmvm","dm"],name="plate2f_2"))

        self.add(Example(example="jbearing2",nprocs=1,options="-tao_smonitor -mx 8 -my 12",method="tron",tags=["bound","jbearing","single","c","tron","dm"],name="jbearing2"))
        self.add(Example(example="jbearing2",nprocs=2,options="-tao_smonitor -mx 50 -my 50 -ecc 0.99",method="gpcg",tags=["bound","jbearing","multiprocessor","c","gpcg","dm"],name="jbearing2_2"))
        self.add(Example(example="jbearing2",nprocs=2,options="-tao_smonitor -mx 10 -my 16 -ecc 0.9",method="bqpip",tags=["bound","jbearing","multiprocessor","c","bqpip","dm"],name="jbearing2_3"))
        

        # Least squares
        self.add(Example(example="chwirut1",nprocs=1,options="-tao_smonitor",method="pounders -tao_fatol 0 -tao_frtol 0",tags=["leastsquares","chwirut","single","c","pounders"],name="chwirut1"))

        
        

                
