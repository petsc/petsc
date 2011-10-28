
% Runs the heat equation solver for several mesh spacing and determines the order of accuracy of the method with respect to time

u = cell(5,1);
for i=1:5
  system(['./heat -ts_monitor -snes_monitor  -pc_type lu   -snes_converged_reason  -allen-cahn -kappa .001   -ts_type cn  -da_refine ' int2str(i+4) ' -ts_final_time 1.e-4 -ts_dt .125e-5 -snes_atol 1.e-25 -snes_rtol 1.e-25'])
  ii = 2^(i-1);
  ut = PetscBinaryRead('binaryoutput','precision','float128');
  u{i} = ut(1:ii:(288*ii));
end

ustar = (2*u{5} - u{4})/1;

for i=1:5
  norm(ustar - u{i})
end

   
