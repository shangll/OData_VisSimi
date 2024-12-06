library(word2vec)
library(rstatix)



setwd("\\\\CNAS.RU.NL\\U759254\\Documents\\DCC\\ch4_LSTM\\res_w2v")
dat.all <- read.csv("glm_rt-simi_w2v.csv")
dat.all$grp <- paste(dat.all$exp,dat.all$subj,sep="")


for (corr.tag in c("mean","max")){
  dat <- subset(dat.all,corr==corr.tag)
  print(corr.tag)
  
  dat %>%
    anova_test(
      dv=coeff,wid=grp,within=setsize,between=exp,
      effect.size="pes") -> res.aov
  
  print(get_anova_table(res.aov,correction=c("auto")))
}

