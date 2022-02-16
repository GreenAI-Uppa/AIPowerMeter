#! /usr/bin/Rscript
niter=500
for(n in 1:niter){
    if((n%%10)==0){
        print(sprintf("%i iterations over %i",n,niter))
        }
    A<-matrix(rnorm(100000),nrow=100)
    B<-matrix(rnorm(100000),nrow=1000)
    C<-A %*% B
    }
print("done")
