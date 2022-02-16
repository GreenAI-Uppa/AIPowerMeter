#! /usr/bin/Rscript
for(n in 1:500){
    if((n%%10)==0){
        print(sprintf("%i iterations over %i",n,1000))
        }
    A<-matrix(rnorm(100000),nrow=100)
    B<-matrix(rnorm(100000),nrow=1000)
    C<-A %*% B
    }
print("done")
