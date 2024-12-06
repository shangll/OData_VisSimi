setwd("\\\\CNAS.RU.NL\\U759254\\Documents\\DCC\\ch4_LSTM\\res_alex")

cond.vec <- c("within","between")
layer.vec <- c("conv_1","conv_2","conv_3","conv_4","conv_5",
               "fc_6","fc_7","fc_8")
# simi.tag <- "simi_mean"
# exp1b.simi <- read.csv("exp1b_simi_mean.csv")
# exp2.simi <- read.csv("exp2_simi_mean.csv")
simi.tag <- "simi_max"
exp1b.simi <- read.csv("exp1b_simi_max.csv")
exp2.simi <- read.csv("exp2_simi_max.csv")

# 1. Modelling
#
get_coeff <- function(exp.data.mean,y.var){
  data.coeff <- data.frame()
  coeff_lm_list <- c()
  coeff_log_list <- c()
  r2_lm_list <- c()
  r2_log_list <- c()
  subjs <- c()
  conds <- c()
  layers <- c()
  
  subj.vec <- unique(exp.data.mean$subj)
  for (h in layer.vec){
    for (n in subj.vec){
      exp.data.mean.subj <- subset(exp.data.mean,(layer==h)&(subj==n))
      
      for (k in cond.vec){
        data.subj <- subset(
          exp.data.mean.subj,(subj==n)&(cond==k))
        if (y.var=="rt"){
          lm_model <- lm(formula=rt~setsize,data.subj)
          log_model <- lm(formula=rt~log(setsize,2),data.subj)
        }else{
          lm_model <- lm(formula=simi_mean~setsize,data.subj)
          log_model <- lm(formula=simi_mean~log(setsize,0.5),data.subj)
        }
        res_lm <- summary(lm_model)
        res_log <- summary(log_model)
        coeff_lm <- coef(res_lm)
        coeff_log <- coef(res_log)
        
        coeff_lm_list <- append(coeff_lm_list,coeff_lm[2,1])
        coeff_log_list <- append(coeff_log_list,coeff_log[2,1])
        r2_lm_list <- append(r2_lm_list,res_lm$r.squared)
        r2_log_list <- append(r2_log_list,res_log$r.squared)
        subjs <- append(subjs,n)
        conds <- append(conds,k)
        layers <- append(layers,h)
      }
    }
  }
  
  exp.coeff <- data.frame(
    layer=layers,
    subj=subjs,
    cond=conds,
    lm=coeff_lm_list,
    log=coeff_log_list)
  data.coeff <- rbind(data.coeff,exp.coeff)
  
  return(data.coeff)
}

exp1b.coeff <- get_coeff(exp1b.simi,"rt")
simi.coeff <- get_coeff(exp1b.simi,"simi")
exp1b.coeff$simi_lm <- simi.coeff$lm
exp1b.coeff$simi_log <- simi.coeff$log
library(dplyr)
rename(
  exp1b.coeff,c(rt_lm=lm,rt_log=log)) -> exp1b.coeff

exp2.coeff <- get_coeff(exp2.simi,"rt")
simi.coeff <- get_coeff(exp2.simi,"simi")
exp2.coeff$simi_lm <- simi.coeff$lm
exp2.coeff$simi_log <- simi.coeff$log
rename(
  exp2.coeff,c(rt_lm=lm,rt_log=log)) -> exp2.coeff

library(psych)
for (h in layer.vec){
  for (cate in cond.vec){
    print(cate)
    
    res1 <- corr.test(
      exp1b.coeff[(exp1b.coeff$layer==h)&(exp1b.coeff$cond==cate),"rt_log"],
      exp1b.coeff[(exp1b.coeff$layer==h)&(exp1b.coeff$cond==cate),"simi_log"],
      method="spearman")
    print(sprintf("%s (LTM)",h))
    print(sprintf("r = %0.3f, p = %0.3f",res1$r,res1$p))
    print("--- --- --- --- --- ---")
    
    res2 <- corr.test(
      exp2.coeff[(exp2.coeff$layer==h)&(exp2.coeff$cond==cate),"rt_log"],
      exp2.coeff[(exp2.coeff$layer==h)&(exp2.coeff$cond==cate),"simi_log"],
      method="spearman")
    print(sprintf("%s (STM)",h))
    print(sprintf("r = %0.3f, p = %0.3f",res2$r,res2$p))
    
    print("--- * --- * --- * --- * --- * ---")
  }
}


for (h in layer.vec){
  for (cate in cond.vec){
    print(cate)
    
    res1 <- corr.test(
      exp1b.coeff[(exp1b.coeff$layer==h)&(exp1b.coeff$cond==cate),"rt_lm"],
      exp1b.coeff[(exp1b.coeff$layer==h)&(exp1b.coeff$cond==cate),"simi_lm"],
      method="spearman")
    print(sprintf("%s (LTM)",h))
    print(sprintf("r = %0.3f, p = %0.3f",res1$r,res1$p))
    print("--- --- --- --- --- ---")
    
    res2 <- corr.test(
      exp2.coeff[(exp2.coeff$layer==h)&(exp2.coeff$cond==cate),"rt_lm"],
      exp2.coeff[(exp2.coeff$layer==h)&(exp2.coeff$cond==cate),"simi_lm"],
      method="spearman")
    print(sprintf("%s (STM)",h))
    print(sprintf("r = %0.3f, p = %0.3f",res2$r,res2$p))
    
    print("--- * --- * --- * --- * --- * ---")
  }
}




















