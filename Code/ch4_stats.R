library(rstatix)

setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_all")
p_crit <- 0.05
sizes <- c(1,2,4,8)
vars <- c("exp","subj","cond","setsize","rt","acc")
cond_list <- c("within","between")
exp.tags <- c("exp1b","exp2")


# 1. data loaded

# 1.1 mean data
#
exp.mean <- read.csv("exp_mean_all.csv")
exp.mean.distr <- subset(exp.mean,cond!="target")
exp.mean.distr$grp <- paste(exp.mean.distr$exp,exp.mean.distr$subj,sep="")



# 2. anova
#

# ACC
# 3-way
aov.res <- anova_test(
  data=exp.mean.distr,dv=acc,wid=grp,within=c(setsize,cond),
  between="exp",type=3,effect.size="pes",detailed=TRUE)
print("ACC")
print(get_anova_table(aov.res))
# 2-way
exp.mean.distr %>%
  group_by(exp) %>%
  anova_test(
    dv=acc,wid=subj,within=c(setsize,cond),
    effect.size="pes") -> aov.2way
print(get_anova_table(aov.2way,correction=c("auto")))
# ACC in MSS 8
exp.mean.distr.8 <- subset(exp.mean.distr,setsize==8)
for (exp.tag in exp.tags){
  dat <- subset(exp.mean.distr.8,exp==exp.tag)
  stat.test <- dat %>% 
    t_test(acc~cond,paired=TRUE) %>%
    add_significance()
  cohens_d(acc~cond,data=dat,paired=TRUE) -> d
  # d
  print(var)
  print(sprintf(
    "t(%0.3f) = %0.3f, p = %0.3f, d = %0.3f",
    stat.test$df,stat.test$statistic,
    stat.test$p,d[1,4]))
}

#
exp.mean.distr %>%
  group_by(setsize,cond) %>%
  anova_test(
    dv=acc,wid=grp,between=exp,
    effect.size="pes") -> aov.1way
print(get_anova_table(aov.1way))


# RT
# 3-way
aov.res <- anova_test(
  data=exp.mean.distr,dv=rt,wid=grp,within=c(setsize,cond),
  between="exp",type=3,effect.size="pes",detailed=TRUE)
print("RT")
print(get_anova_table(aov.res))
# 2-way
exp.mean.distr %>%
  group_by(exp) %>%
  anova_test(
    dv=rt,wid=subj,within=c(setsize,cond),
    effect.size="pes") -> aov.2way
print(get_anova_table(aov.2way,correction=c("auto")))
#
exp.mean.distr %>%
  group_by(setsize,cond) %>%
  anova_test(
    dv=rt,wid=grp,between=exp,
    effect.size="pes") -> aov.1way
print(get_anova_table(aov.1way))
#
exp.mean.distr %>%
  group_by(exp,cond) %>%
  anova_test(
    dv=rt,wid=subj,within=setsize,
    effect.size="pes") -> aov.1way
print(get_anova_table(aov.1way))



# 3. Fit
# 3.1 predicting

for (exp.tag in exp.tags){
  exp.data <- subset(exp.mean.distr,exp==exp.tag)
  
  print(exp.tag)
  print("----------------------------")
  
  subj_list <- unique(exp.data$subj)
  w_obs_rt <- subset(exp.data,(setsize==8)&(cond=="within"))$rt
  b_obs_rt <- subset(exp.data,(setsize==8)&(cond=="between"))$rt
  w_8_lm <- subset(exp.data,(setsize==8)&(cond=="within"))$lm
  w_8_log <- subset(exp.data,(setsize==8)&(cond=="within"))$log
  b_8_lm <- subset(exp.data,(setsize==8)&(cond=="between"))$lm
  b_8_log <- subset(exp.data,(setsize==8)&(cond=="between"))$log
  
  # t_val <- t.test(w_8_lm,w_obs_rt,paired=TRUE,alternative="greater")
  t_val <- t.test(w_8_lm,w_obs_rt,paired=TRUE)
  # t_val
  dat <- data.frame(mss8=c(w_8_lm,w_obs_rt),
                    pred=rep(c("lm","rt"),each=length(subj_list)))
  dat %>% cohens_d(mss8~pred,paired=TRUE) -> d
  # d
  print("within: lm")
  print(sprintf("t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f",
                t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))
  # ----------------------------
  t_val <- t.test(w_8_log,w_obs_rt,paired=TRUE)
  # t_val
  dat <- data.frame(mss8=c(w_8_log,w_obs_rt),
                    pred=rep(c("log","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
  # d
  print("within: log2")
  print(sprintf("t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f",
                t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))
  # ----------------------------
  t_val <- t.test(b_8_lm,b_obs_rt,paired=TRUE)
  # t_val
  dat <- data.frame(mss8=c(b_8_lm,b_obs_rt),
                    pred=rep(c("lm","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
  # d
  print("bertween: lm")
  print(sprintf("t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f",
                t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))
  # ----------------------------
  t_val <- t.test(b_8_log,b_obs_rt,paired=TRUE)
  # t_val
  dat <- data.frame(mss8=c(b_8_log,b_obs_rt),
                    pred=rep(c("log","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
  # d
  print("between: log2")
  print(sprintf("t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f",
                t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))
}

for (exp.tag in exp.tags){
  exp.data <- subset(exp.mean.distr,exp==exp.tag)
  subj_list <- unique(exp.data$subj)
  
  w_obs_rt <- subset(exp.data,(setsize==8)&(cond=="within"))$rt
  b_obs_rt <- subset(exp.data,(setsize==8)&(cond=="between"))$rt
  w_8_lm <- subset(exp.data,(setsize==8)&(cond=="within"))$lm
  w_8_log <- subset(exp.data,(setsize==8)&(cond=="within"))$log
  b_8_lm <- subset(exp.data,(setsize==8)&(cond=="between"))$lm
  b_8_log <- subset(exp.data,(setsize==8)&(cond=="between"))$log
  
  w_resid_lm <- abs(w_obs_rt-w_8_lm)
  w_resid_log <- abs(w_obs_rt-w_8_log)
  b_resid_lm <- abs(b_obs_rt-b_8_lm)
  b_resid_log <- abs(b_obs_rt-b_8_log)
  
  print(exp.tag)
  # ----------------------------
  print("within")
  t_val <- t.test(w_resid_log,w_resid_lm,paired=TRUE)
  # t_val
  dat <- data.frame(resids=c(w_resid_log,w_resid_lm),
                    pred=rep(c("log","lm"),each=length(subj_list)))
  d <- dat %>% cohens_d(resids~pred,paired=TRUE)
  # d
  print(sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))
  # ----------------------------
  print("between")
  t_val <- t.test(b_resid_log,b_resid_lm,paired=TRUE)
  # t_val
  dat <- data.frame(resids=c(b_resid_log,b_resid_lm),
                    pred=rep(c("log","lm"),each=length(subj_list)))
  d <- dat %>% cohens_d(resids~pred,paired=TRUE)
  # d
  print(sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))
  #
}




# 3.2 Modelling
#
coeff_lm_list <- c()
coeff_log_list <- c()
r2_lm_list <- c()
r2_log_list <- c()
exp_list <- c()
subjs <- c()
conds <- c()

for (exp.tag in exp.tags){
  exp.data <- subset(exp.mean.distr,exp==exp.tag)
  subj_list <- unique(exp.data$subj)
  
  for (n in subj_list){
    exp.data.mean.subj <- subset(exp.data,(subj==n))
    
    for (k in cond_list){
      data.subj <- subset(
        exp.data.mean.subj,(subj==n)&(cond==k))
      lm_model <- lm(formula=rt~setsize,data.subj)
      log_model <- lm(formula=rt~log(setsize,2),data.subj)
      res_lm <- summary(lm_model)
      res_log <- summary(log_model)
      coeff_lm <- coef(res_lm)
      coeff_log <- coef(res_log)
      
      coeff_lm_list <- append(coeff_lm_list,coeff_lm[2,1])
      coeff_log_list <- append(coeff_log_list,coeff_log[2,1])
      r2_lm_list <- append(r2_lm_list,res_lm$r.squared)
      r2_log_list <- append(r2_log_list,res_log$r.squared)
      exp_list <- append(exp_list,exp.tag)
      subjs <- append(subjs,n)
      conds <- append(conds,k)
    }
  }
}

exp.coeff <- data.frame(
  exp=exp_list,
  subj=subjs,
  cond=conds,
  lm=coeff_lm_list,
  log=coeff_log_list)
exp.coeff$grp <- paste(exp.coeff$exp,exp.coeff$subj,sep="")
# 2-way
exp.coeff %>%
  anova_test(
    dv=log,wid=grp,between=exp,within=cond,
    effect.size="pes") -> aov.2way
print(get_anova_table(aov.2way))
#
exp.coeff %>%
  group_by(cond) %>%
  anova_test(
    dv=log,wid=grp,between=exp,
    effect.size="pes") -> aov.1way
print(get_anova_table(aov.1way))
#
exp.coeff %>%
  group_by(exp) %>%
  anova_test(
    dv=log,wid=grp,within=cond,
    effect.size="pes") -> aov.1way
print(get_anova_table(aov.1way),correction="bonferroni")




















# ##################################################################
# check bias at individual level
#

expMean.err <- data.frame()
for (exp.tag in c("exp1b","exp2")){
  exp.err <- subset(
    expAll.delNA,(expAll.delNA$exp==exp.tag)&(expAll.delNA$acc==0))
  exp.err.mean <- aggregate(
    x=exp.err["rt"],by=list(subj=exp.err$subj),
    mean)
  exp.err.mean$acc <- 0
  exp.err.mean$exp <- exp.tag
  expMean.err <- rbind(expMean.err,exp.err.mean)
  
  exp.err <- subset(
    expAll.delNA,(expAll.delNA$exp==exp.tag)&(expAll.delNA$acc==1))
  exp.err.mean <- aggregate(
    x=exp.err["rt"],by=list(subj=exp.err$subj),
    mean)
  exp.err.mean$acc <- 1
  exp.err.mean$exp <- exp.tag
  expMean.err <- rbind(expMean.err,exp.err.mean)
}

# 2.1.1 RT in error trials should be faster than correct
expMean.err1b <- subset(expMean.err,exp=="exp1b")
t_val <- t.test(expMean.err1b[expMean.err1b$acc==0,]$rt,
                expMean.err1b[expMean.err1b$acc==1,]$rt)
t_val
expMean.err2 <- subset(expMean.err,exp=="exp2")
t_val <- t.test(expMean.err2[expMean.err2$acc==0,]$rt,
                expMean.err2[expMean.err2$acc==1,]$rt,
                alternative="greater")
t_val

library(dplyr)
library(ggplot2)
library(gridExtra)
#
# plot
expMean.err %>%
  group_by(exp,acc)%>%
  summarise(rt=mean(rt,na.rm=TRUE)) -> exp.speed.mean
exp.speed.mean$acc <- as.factor(exp.speed.mean$acc)
windows()
plt.speed <- ggplot(data=exp.speed.mean,
                    aes(x=acc,y=rt,group=exp))+
  geom_line(aes(color=exp),size=1.5)+
  geom_point(aes(color=exp),size=2)+
  scale_color_manual(values=c('#0066b2','#ec1c24'))+
  theme(legend.position=c(0,1),
        legend.justification=c("left","top"))
plt.speed
ggsave(
  file=paste("ch4_stats_speed.png"),
  plot=plt.speed,width=12,height=9)

# 2.1.2 anova
expMean0 <- data.frame()
for (exp.tag in c("exp1b","exp2")){
  exp.err <- subset(
    expAll.delNA,expAll.delNA$exp==exp.tag)
  exp.err.mean <- aggregate(
    x=exp.err["rt"],by=list(subj=exp.err$subj,
                            acc=exp.err$acc,
                            trialType=exp.err$trialType),
    mean)
  exp.err.mean$exp <- exp.tag
  expMean0 <- rbind(expMean0,exp.err.mean)
}
#
expMean0$acc <- as.factor(expMean0$acc)
for (exp.tag in c("exp1b","exp2")){
  exp.err <- subset(
    expMean0,exp==exp.tag)
  aov.res <- anova_test(
    data=exp.err,dv=rt,wid=subj,
    within=trialType,between=acc,
    type=3,effect.size="pes",detailed=TRUE)
  print(exp.tag)
  print(get_anova_table(aov.res))
  print("--- --- --- --- --- ---")
  
  exp.err %>%
    group_by(trialType) %>%
    anova_test(dv=rt,wid=subj,between=acc,effect.size="pes") %>%
    get_anova_table() %>%
    adjust_pvalue(method="bonferroni") -> one.way
  print(get_anova_table(one.way))
  print("--- --- --- --- --- ---")
  
  exp.err %>%
    group_by(acc) %>%
    anova_test(dv=rt,wid=subj,within=trialType,effect.size="pes") %>%
    get_anova_table() %>%
    adjust_pvalue(method="bonferroni") -> one.way
  print(get_anova_table(one.way))
  print("--- * --- * --- * --- * --- * ---")
}

# plot
expAll.delNA %>%
  group_by(exp,trialType,acc)%>%
  summarise(rt=mean(rt,na.rm=TRUE)) -> exp.bias.mean
exp.bias.mean$acc <- as.factor(exp.bias.mean$acc)
windows()
plt.bias <- ggplot(data=exp.bias.mean,
                   aes(x=trialType,y=rt,group=acc))+
  geom_line(aes(color=acc),size=1)+
  geom_point(aes(color=acc),size=2)+
  # scale_color_manual(values=c('#0066b2','#ec1c24','#84BADB'))+
  facet_wrap(~exp,ncol=2)+
  theme(legend.position=c(0,1),
        legend.justification=c("left","top"))
plt.bias
ggsave(
  file=paste("ch4_stats_bias.png"),
  plot=plt.bias,width=12,height=9)
#
windows()
plt.bias <- ggplot(data=exp.bias.mean,
                   aes(x=acc,y=rt,group=trialType))+
  geom_line(aes(color=trialType),size=1)+
  geom_point(aes(color=trialType),size=2)+
  # scale_color_manual(values=c('#0066b2','#ec1c24','#84BADB'))+
  facet_wrap(~exp,ncol=2)+
  theme(legend.position=c(0,1),
        legend.justification=c("left","top"))
plt.bias
ggsave(
  file=paste("ch4_stats_bias2.png"),
  plot=plt.bias,width=12,height=9)



# 2.2 ANOVA for RT & ACC (group level)
#

for(exp.tag in c("exp1b","exp2")){
  print(exp.tag)
  exp.data.mean.distr <- subset(
    expMean.distr,exp==exp.tag)
  
  # RT
  aov.res <- anova_test(
    data=exp.data.mean.distr,dv=rt,wid=subj,
    within=c(setsize,cond),
    type=3,effect.size="pes",detailed=TRUE)
  print("RT")
  print(get_anova_table(aov.res))
  print("--- --- --- --- --- ---")
  
  # ACC
  aov.res <- anova_test(
    data=exp.data.mean.distr,dv=acc,wid=subj,
    within=c(setsize,cond),
    type=3,effect.size="pes",detailed=TRUE)
  print("ACC")
  print(get_anova_table(aov.res))
  print("--- * --- * --- * --- * --- * ---")
}
#
expMean.grp <- aggregate(
  x=expMean[c("rt","acc")],by=list(
    exp=expMean$exp,
    cond=expMean$cond,
    setsize=expMean$setsize),
  mean)
library(cowplot)
windows()
plt.aov.rt <- ggplot(data=expMean.grp,
                   aes(x=setsize,y=rt,group=cond))+
  geom_line(aes(color=cond),size=1)+
  geom_point(aes(color=cond),size=2)+
  scale_color_manual(values=c('#0066b2','#ec1c24','#84BADB'))+
  facet_wrap(~exp,ncol=2)+
  theme(legend.position=c(0,1),
        legend.justification=c("left","top"))
plt.aov.acc <- ggplot(data=expMean.grp,
                      aes(x=setsize,y=acc,group=cond))+
  geom_line(aes(color=cond),size=1)+
  geom_point(aes(color=cond),size=2)+
  scale_color_manual(values=c('#0066b2','#ec1c24','#84BADB'))+
  facet_wrap(~exp,ncol=2)+
  theme(legend.position="none")
plt.aov <- plot_grid(plt.aov.rt,plt.aov.acc,ncol=1)
plt.aov
ggsave(
  file=paste("ch4_stats_aov.png"),
  plot=plt.aov,width=12,height=12)

# 2.3 compare category effects
#
expMean.distr$grp <- c(rep(1:30,times=8),rep(31:60,times=8))
aov.res <- anova_test(
  data=expMean.distr,dv=rt,wid=grp,
  within=c(setsize,cond),between=exp,
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(aov.res)
# simple 2-way
expMean.distr %>%
  group_by(exp) %>%
  anova_test(dv=rt,wid=grp,within=c(cond,setsize),effect.size="pes") -> two.way
get_anova_table(two.way)
# simple 2-way
expMean.distr %>%
  group_by(setsize) %>%
  anova_test(dv=rt,wid=grp,within=cond,between=exp,effect.size="pes") -> two.way
get_anova_table(two.way)
# simple 2-way
expMean.distr %>%
  group_by(cond) %>%
  anova_test(dv=rt,wid=grp,within=setsize,between=exp,effect.size="pes") -> two.way
get_anova_table(two.way)
# simple 1-way
expMean.distr %>%
  group_by(setsize,cond) %>%
  anova_test(dv=rt,wid=grp,between=exp,effect.size="pes") %>%
  get_anova_table() %>%
  adjust_pvalue(method="bonferroni") -> one.way
get_anova_table(one.way)


# 3. Linear or Log 2

cond_list <- c("target","within","between")
data.coeff <- data.frame()
  
for (exp.tag in c("exp1b","exp2")){
  exp.data.mean <- subset(expMean,exp==exp.tag)
  print(exp.tag)
  print("Prediction")
  
  # 3.1 predicting
  #
  subj_list <- unique(exp.data.mean$subj)
  exp.data.mean$setsize <- as.numeric(as.character(exp.data.mean$setsize))
  exp.data.mean$setsize <- as.numeric(as.character(exp.data.mean$setsize))
  t_8_lm <- c()
  t_8_log <- c()
  w_8_lm <- c()
  b_8_lm <- c()
  w_8_log <- c()
  b_8_log <- c()
  
  for (n in subj_list){
    for (k in cond_list){
      trainData <- subset(
        exp.data.mean,(subj==n)&(setsize!=8)&(cond==k))
      testData <- data.frame(setsize=c(8))
      
      lm_pred <- lm(formula=rt~setsize,trainData)
      log_pred <- lm(formula=rt~log(setsize,2),trainData)
      lm_8 <- predict(lm_pred,newdata=testData)
      log_8 <- predict(log_pred,newdata=testData)
      if (k=="target"){
        t_8_lm <- append(t_8_lm,lm_8[[1]])
        t_8_log <- append(t_8_log,log_8[[1]])
      }
      else if (k=="within"){
        w_8_lm <- append(w_8_lm,lm_8[[1]])
        w_8_log <- append(w_8_log,log_8[[1]])
      }else{
        b_8_lm <- append(b_8_lm,lm_8[[1]])
        b_8_log <- append(b_8_log,log_8[[1]])
      }
    }
  }
  t_obs <- subset(exp.data.mean,(setsize==8)&(cond=="target"))
  w_obs <- subset(exp.data.mean,(setsize==8)&(cond=="within"))
  b_obs <- subset(exp.data.mean,(setsize==8)&(cond=="between"))
  t_obs_rt <- t_obs$rt
  w_obs_rt <- w_obs$rt
  b_obs_rt <- b_obs$rt
  
  #
  print('target: lm')
  t_val <- t.test(t_8_lm,t_obs_rt,alternative="greater")
  dat <- data.frame(mss8=c(t_8_lm,t_obs_rt),
                    pred=rep(c("lm","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--- --- --- --- --- ---")
  #
  print('target: log2')
  t_val <- t.test(t_8_log,t_obs_rt)
  t_val
  dat <- data.frame(mss8=c(t_8_log,t_obs_rt),
                    pred=rep(c("log","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--- --- --- --- --- ---")
  
  #
  print('within: lm')
  t_val <- t.test(w_8_lm,w_obs_rt,alternative="greater")
  dat <- data.frame(mss8=c(w_8_lm,w_obs_rt),
                    pred=rep(c("lm","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--- --- --- --- --- ---")
  #
  print('within: log2')
  t_val <- t.test(w_8_log,w_obs_rt)
  t_val
  dat <- data.frame(mss8=c(w_8_log,w_obs_rt),
                    pred=rep(c("log","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--- --- --- --- --- ---")
  
  #
  print('between: lm')
  # t_val <- t.test(b_8_lm,b_obs_rt,alternative="greater")
  t_val <- t.test(b_8_lm,b_obs_rt)
  dat <- data.frame(mss8=c(b_8_lm,b_obs_rt),
                    pred=rep(c("lm","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--- --- --- --- --- ---")
  #
  print('between: log2')
  t_val <- t.test(b_8_lm,b_obs_rt,alternative="greater")
  t_val <- t.test(b_8_log,b_obs_rt)
  dat <- data.frame(mss8=c(b_8_log,b_obs_rt),
                    pred=rep(c("log","rt"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--- * --- * --- * --- * --- * ---")
  
  # 3.2 Modelling
  #
  coeff_lm_list <- c()
  coeff_log_list <- c()
  r2_lm_list <- c()
  r2_log_list <- c()
  exp_list <- c()
  subjs <- c()
  conds <- c()
  
  for (n in subj_list){
    exp.data.mean.subj <- subset(exp.data.mean,(subj==n))
    
    for (k in cond_list){
      data.subj <- subset(
        exp.data.mean.subj,(subj==n)&(cond==k))
      lm_model <- lm(formula=rt~setsize,data.subj)
      log_model <- lm(formula=rt~log(setsize,2),data.subj)
      res_lm <- summary(lm_model)
      res_log <- summary(log_model)
      coeff_lm <- coef(res_lm)
      coeff_log <- coef(res_log)
      
      coeff_lm_list <- append(coeff_lm_list,coeff_lm[2,1])
      coeff_log_list <- append(coeff_log_list,coeff_log[2,1])
      r2_lm_list <- append(r2_lm_list,res_lm$r.squared)
      r2_log_list <- append(r2_log_list,res_log$r.squared)
      exp_list <- append(exp_list,exp.tag)
      subjs <- append(subjs,n)
      conds <- append(conds,k)
    }
  }
  
  exp.coeff <- data.frame(
    exp=exp_list,
    subj=subjs,
    cond=conds,
    lm=coeff_lm_list,
    log=coeff_log_list)
  data.coeff <- rbind(data.coeff,exp.coeff)
}

# 2-way
data.coeff$grp <- c(rep(1:30,each=3),rep(31:60,each=3))
data.coeff.distr <- subset(data.coeff,cond!="target")
aov.res <- anova_test(
  data=data.coeff.distr,dv=log,wid=grp,
  within=cond,between=exp,
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(aov.res)
# simple
data.coeff.distr %>%
  group_by(cond) %>%
  anova_test(dv=log,wid=grp,between=exp,effect.size="pes") %>%
  get_anova_table() %>%
  adjust_pvalue(method="bonferroni") -> one.way
get_anova_table(one.way)











