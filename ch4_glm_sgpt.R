library(rstatix)


setwd("\\\\CNAS.RU.NL\\U759254\\Documents\\DCC\\ch4_LSTM\\res_sgpt")
p.crit <- 0.05
size.vec <- c(1,2,4,8)
cate.vec <- c("within","between")

dat <- read.csv("glm_rt-simi_sgpt.csv")
dat$grp <- paste(dat$exp,dat$subj,sep="")
head(dat)


for (corr_tag in c("mean","max")){
  dat.temp <- subset(
    dat,(corr==corr_tag))
  
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    within=setsize,between=exp,
    type=3,effect.size="pes",detailed=TRUE)
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  print("--- --- --- --- --- ---")
  
  one.way <- dat.temp %>%
    group_by(exp) %>%
    anova_test(
      dv=coeff,wid=subj,within=setsize,
      type=3,effect.size="pes",detailed=TRUE) %>%
    get_anova_table() %>%
    adjust_pvalue(method="bonferroni")
  print(get_anova_table(one.way))
  print("--- --- --- --- --- ---")

}

for (corr_tag in c("mean","max")){
  for (exp_tag in c("exp1b","exp2")){
    dat.temp <- subset(
      dat,(corr==corr_tag)&(exp==exp_tag))
    
    res.aov <- anova_test(
      data=dat.temp,dv=coeff,wid=subj,
      within=setsize,type=3,
      effect.size="pes",detailed=TRUE)
    
    print(sprintf("%s %s",corr_tag,exp_tag))
    print(get_anova_table(res.aov))
    print("--- --- --- --- --- ---")
    
    pwc <- dat.temp %>%
      pairwise_t_test(coeff~setsize,paired=TRUE,
                      p.adjust.method="bonferroni")
    print(pwc)
    print("--- --- --- --- --- ---")
  }
}