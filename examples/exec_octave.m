#!/usr/bin/octave -qf

a = randn(100,1000);
b = randn(1000,100);
for i=1:30000
  disp(i);
  c=a*b;
endfor
%printf("Elapsed time: %.4f seconds", elapsed);
