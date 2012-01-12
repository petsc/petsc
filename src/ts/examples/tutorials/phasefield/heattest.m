
% Runs the heat equation solver for several mesh spacing and determines the order of accuracy of the method with respect to time

u = cell(5,1);
for i=1:5
%  system(['./heat -allen-cahn -kappa .001  -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason   -ts_type cn  -da_refine ' int2str(i+4) ' -ts_final_time 10 -ts_dt 1.e-3 -snes_atol 1.e-25 -snes_rtol 1.e-25'])
  system(['./biharmonic -ts_monitor -snes_monitor  -snes_rtol 1.e-10 -pc_type lu -vi -snes_converged_reason   -ts_type cn    -da_refine ' int2str(i+4) '   -kappa .00001 -ts_dt 2.96046e-06 -cahn-hillard -ts_final_time 1. -ts_max_steps 300'])
  ii = 2^(i-1);
         ut = PetscBinaryRead('binaryoutput','precision','float128');
  u{i} = ut(1:ii:(288*ii));
end

ustar = (4*u{5} - u{4})/3;

for i=1:5
  norm(ustar - u{i})
end

   
