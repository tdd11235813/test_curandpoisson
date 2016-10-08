# http://stats.stackexchange.com/questions/1174/how-can-i-test-if-given-samples-are-taken-from-a-poisson-distribution

set.seed(1234)
#x.poi<-rpois(n=2000,lambda=2.5) # a vector of random variables from the Poisson distr.
x.poi <- as.numeric(read.table("dump.csv",header=F)[,1])

hist(x.poi,main="Poisson distribution")
lambda.est <- mean(x.poi) ## estimate of parameter lambda
(tab.os<-table(x.poi)) ## table with empirical frequencies
lambda.est

freq.os<-vector()
for(i in 1: length(tab.os)) freq.os[i]<-tab.os[[i]]  ## vector of emprical frequencies
freq.ex<-(dpois(0:max(x.poi),lambda=lambda.est)*200) ## vector of fitted (expected) frequencies

acc <- mean(abs(freq.os-trunc(freq.ex))) ## absolute goodness of fit index acc
acc/mean(freq.os)*100 ## relative (percent) goodness of fit index

h <- hist(x.poi ,breaks=length(tab.os))
xhist <- c(min(h$breaks),h$breaks)
yhist <- c(0,h$density,0)
xfit <- min(x.poi):max(x.poi)
yfit <- dpois(xfit,lambda=lambda.est)
plot(xhist,yhist,type="s",ylim=c(0,max(yhist,yfit)), main="Poisson density and histogram")
lines(xfit,yfit, col="red")

                                        #Perform the chi-square goodness of fit test
                                        #In case of count data we can use goodfit() included in vcd package
library(vcd) ## loading vcd package
gf <- goodfit(x.poi,type= "poisson",method= "MinChisq")
summary(gf)


png(filename="test.png",width=800,height=600)
plot(gf,main="Count data vs Poisson distribution")
dev.off()
