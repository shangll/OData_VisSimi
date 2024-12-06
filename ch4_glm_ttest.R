setwd("\\\\CNAS.RU.NL\\U759254\\Documents\\DCC\\ch4_LSTM\\res_alex")

p.crit <- 0.05
cond.vec <- c("within","between")
# corr.tag <- "max"
# df.glm <- read.csv("glm_simi_max.csv")
corr.tag <- "mean"
df.glm <- read.csv("glm_simi_mean.csv")

library(rstatix)

layer.vec <- c()
setsize.vec <- c()
exp.vec <- c()
t.vec.cate <- c()
t.vec.simi <- c()
t.vec.inter <- c()
sig.vec.cate <- c()
sig.vec.simi <- c()
sig.vec.inter <- c()

for (k in unique(df.glm$layer)){
  for (n in c(1,2,4,8)){
    for (exp.tag in c("exp1b","exp2")){
      layer.vec <- append(layer.vec,k)
      setsize.vec <- append(setsize.vec,n)
      exp.vec <- append(exp.vec,exp.tag)
      
      for (h in c("cate","simi","inter")){
        df.tmp <- df.glm[(df.glm$layer==k)&
                           (df.glm$exp==exp.tag)&
                           (df.glm$setsize==n)&
                           (df.glm$cond==h),]
        t.res <- t.test(df.tmp$coeff,mu=0,alternative="two.sided")
        t <- round(t.res$statistic,3)
        num <- t.res$parameter
        p <- round(t.res$p.value,3)
        d.res <- df.tmp %>% cohens_d(coeff~1,mu=0)
        d <- d.res$effsize
        res <- sprintf("t(%d)=%0.3f,p=%0.3f,d=%0.3f",num,t,p,d)
        
        if (h=="cate"){
          t.vec.cate <- append(t.vec.cate,res)
          if (p<p.crit){
            sig.vec.cate <- append(sig.vec.cate,"*")
          }else{
            sig.vec.cate <- append(sig.vec.cate,"")
          }
        }else if (h=="simi"){
          t.vec.simi <- append(t.vec.simi,res)
          if (p<p.crit){
            sig.vec.simi <- append(sig.vec.simi,"*")
          }else{
            sig.vec.simi <- append(sig.vec.simi,"")
          }
        }else{
          t.vec.inter <- append(t.vec.inter,res)
          if (p<p.crit){
            sig.vec.inter <- append(sig.vec.inter,"*")
          }else{
            sig.vec.inter <- append(sig.vec.inter,"")
          }
        }
      }
    }
  }
}

glm.ttest <- data.frame(
  layer=layer.vec,setsize=setsize.vec,exp=exp.vec,
  category=t.vec.cate,sig.cate=sig.vec.cate,
  similarity=t.vec.simi,sig.simi=sig.vec.simi,
  interaction=t.vec.inter,sig.inter=sig.vec.inter)
write.csv(
  glm.ttest,file=sprintf("glm_ttest_%s.csv",corr.tag),
  row.names=F)



