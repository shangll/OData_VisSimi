library(rstatix)
library(ez)
library(bruceR)


# setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_w2v")
# w2v_cate <- read.csv("w2v_simi_cate.csv")
# w2v_cate$w2v_mean <- as.numeric(w2v_cate$simi)
# w2v_cate$cate <- as.factor(w2v_cate$cate)
# t_val <- t.test(
#   w2v_cate[w2v_cate$cate=="within","simi"],
#   w2v_cate[w2v_cate$cate=="between","simi"],paired=TRUE)
# # t_val
# cohens_d(w2v_cate$w2v_mean,w2v_cate$cate,paired=TRUE) -> d
# # d
# print(sprintf(
#   "t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f",
#   t_val$statistic,t_val$parameter,t_val$p.value,d[1,4]))


p.crit <- 0.05
size.vec <- c(1,2,4,8)
cate.vec <- c("within","between")
layer.vec <- c(
  "conv_1","conv_2","conv_3","conv_4","conv_5",
  "fc_6","fc_7","fc_8")
exp.vec <- c("exp1b","exp2")
options(scipen=100,digits=4)



# ------------------------------------------------------------------------------


# 
setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_alex")
# v
glm_v <- read.csv("glm_rt-simi_3layers.csv")

# v w/o c
glm_v_c <- read.csv("glm_rt-2avg_3layers.csv")
# c w/o v
glm_c_v <- read.csv("glm_resid-cate_3layers.csv")

# c
glm_c <- read.csv("glm_rt-cate_3layers.csv")

#
setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_w2v")
# v w/o s
glm_v_s <- read.csv("glm_resid-simi_3layers.csv")
# v w/o c&s
glm_v_cs <- read.csv("glm_resid-2avg_3layers_v.csv")
# s w/o v
glm_s_v <- read.csv("glm_resid-w2v_3layers.csv")
# s w/o c
glm_s_c <- read.csv("glm_rt-s-2avg_3layers.csv")
# s w/o c&v
glm_s_cv <- read.csv("glm_resid-2avg_3layers_s.csv")
# v w/o c&s
glm_v_cs <- read.csv("glm_resid-s-2avg_3layers_v.csv")
# s
glm_s <- read.csv("glm_rt-w2v_3layers.csv")

# c w/o s
glm_c_s <- read.csv("glm_resid-c_3layers.csv")


#
# dat <- glm_c_v
# dat <- glm_s_v
#
# dat <- glm_v
dat <- glm_v_c
# dat <- glm_v_s
# dat <- glm_v_cs
head(dat)
dat$layer <- as.factor(dat$layer)
dat$setsize <- as.factor(dat$setsize)


# 1.1 2-way: layer*setsize in each exp
for (corr_tag in c("mean","max")){
  for (exp_tag in exp.vec){
    
    dat.temp <- subset(
      dat,(corr==corr_tag)&(exp==exp_tag))
    
    # 2 way
    aov.2way <- anova_test(
      data=dat.temp,dv=coeff,wid=subj,
      within=c(setsize,layer),
      effect.size="pes")
    # aov.2way <- ezANOVA(
    #   data=dat.temp,dv=coeff,wid=subj,
    #   within=.(setsize,layer),detailed=F)
    
    print(sprintf("%s %s",corr_tag,exp_tag))
    print(get_anova_table(aov.2way))
    
    
    # group by setsize
    aov.1way <- dat.temp %>%
      group_by(setsize) %>%
      anova_test(
        dv=coeff,wid=subj,within=layer,
        effect.size="pes") %>%
      get_anova_table() %>%
      adjust_pvalue(method="bonferroni")
    print("group setsize")
    print(aov.1way)
    
    # multi-compar
    pwc <- dat.temp %>%
      group_by(setsize) %>%
      pairwise_t_test(
        coeff~layer,paired=TRUE,
        p.adjust.method="bonferroni")
    print(pwc,n=999)
    
    print("--- --- --- --- --- ---")
    print("")
    
    # group by layer
    aov.1way <- dat.temp %>%
      group_by(layer) %>%
      anova_test(
        dv=coeff,wid=subj,within=setsize,
        effect.size="pes") %>%
      get_anova_table() %>%
      adjust_pvalue(method="bonferroni")
    print("group by layer")
    print(aov.1way)
    
    # multi-compar
    pwc <- dat.temp %>%
      group_by(layer) %>%
      pairwise_t_test(
        coeff~setsize,paired=TRUE,
        p.adjust.method="bonferroni")
    print(pwc,n=999)
    
    print("--- --- --- --- --- ---")
    print("")
    
    
    MANOVA(data=dat.temp,subID="subj", 
           dv="coeff",within=c("layer")) %>%
      EMMEANS("layer")
    print("--- --- --- --- --- ---")
  }
}



#
#
#
# dat.ref <- glm_c
# dat.ref.coeff <- c(dat.ref$coeff,dat.ref$coeff)
# minus.coeff <- dat.ref.coeff-dat$coeff

dat.ref <- glm_s
minus.coeff <- dat.ref$coeff


dat.minus <- data.frame(
  corr=dat$corr,subj=dat$subj,setsize=as.factor(dat$setsize),
  exp=dat$exp,layer=as.factor(dat$layer),
  coeff=minus.coeff)

for (corr_tag in c("mean","max")){
  for (exp_tag in exp.vec){
    
    dat.temp <- subset(
      dat.minus,(corr==corr_tag)&(exp==exp_tag))
    
    # 2 way
    aov.2way <- anova_test(
      data=dat.temp,dv=coeff,wid=subj,
      within=c(setsize,layer),
      effect.size="pes")
    
    print(sprintf("%s %s",corr_tag,exp_tag))
    print(get_anova_table(aov.2way))
    
    print("--- --- --- --- --- ---")
    print("")
    
    # group by setsize
    aov.1way <- dat.temp %>%
      group_by(setsize) %>%
      anova_test(
        dv=coeff,wid=subj,within=layer,
        effect.size="pes") %>%
      get_anova_table() %>%
      adjust_pvalue(method="bonferroni")
    print("group setsize")
    print(aov.1way)
    
    # multi-compar
    pwc <- dat.temp %>%
      group_by(setsize) %>%
      pairwise_t_test(
        coeff~layer,paired=TRUE,
        p.adjust.method="bonferroni")
    print(pwc,n=999)
    
    print("--- --- --- --- --- ---")
    print("")
    
    # group by layer
    aov.1way <- dat.temp %>%
      group_by(layer) %>%
      anova_test(
        dv=coeff,wid=subj,within=setsize,
        effect.size="pes") %>%
      get_anova_table() %>%
      adjust_pvalue(method="bonferroni")
    print("group by layer")
    print(aov.1way)
    
    # multi-compar
    pwc <- dat.temp %>%
      group_by(layer) %>%
      pairwise_t_test(
        coeff~setsize,paired=TRUE,
        p.adjust.method="bonferroni")
    print(pwc,n=999)
    
    print("--- --- --- --- --- ---")
    print("")
  }
}


tag <- 0
# 1.2 exp*layer*setsize

dat$grp <- paste(dat$exp,dat$subj,sep="")
for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat,(corr==corr_tag))
  
  # 3 way
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    between=exp,within=c(setsize,layer),
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  
  print("--- --- --- --- --- ---")
  print("")
  
  
  
  if (tag==1){
    # group by layer
    aov.2way <- dat.temp %>%
      group_by(layer) %>%
      anova_test(
        dv=coeff,wid=grp,between=exp,
        within=setsize,effect.size="pes")
    print("group by layer")
    print(get_anova_table(aov.2way))
    
    # group by setsize
    aov.2way <- dat.temp %>%
      group_by(setsize) %>%
      anova_test(
        dv=coeff,wid=grp,between=exp,
        within=layer,effect.size="pes")
    print("group by setsize")
    print(get_anova_table(aov.2way))
    
    # group by exp
    aov.2way <- dat.temp %>%
      group_by(exp) %>%
      anova_test(
        dv=coeff,wid=grp,within=c(layer,setsize),
        effect.size="pes")
    print("group by exp")
    print(get_anova_table(aov.2way))
    
    
    # group by exp & setsize
    aov.1way <- dat.temp %>%
      group_by(exp,setsize) %>%
      anova_test(
        dv=coeff,wid=grp,within=layer,effect.size="pes") %>%
      get_anova_table() %>%
      adjust_pvalue(method="bonferroni")
    print("group by exp and setsize")
    print(aov.1way)
    
    # group by exp & layer
    aov.1way <- dat.temp %>%
      group_by(exp,layer) %>%
      anova_test(
        dv=coeff,wid=grp,within=setsize,effect.size="pes") %>%
      get_anova_table() %>%
      adjust_pvalue(method="bonferroni")
    print("group by exp and layer")
    print(aov.1way)
  }
}

# ------------------------------------------------------------------------------



setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_alex")
# v
dat_v <- read.csv("glm_rt-simi.csv")
dat_v$eff <- "v"
dat_v$grp <- paste(dat_v$exp,dat_v$subj,sep="")
# v w/o c
dat_v_c <- read.csv("glm_rt-2avg.csv")
dat_v_c$eff <- "v w/o c"
dat_v_c$grp <- paste(dat_v_c$exp,dat_v_c$subj,sep="")
# v w/o s
dat_v_s <- read.csv("glm_resid-simi.csv")
dat_v_s$eff <- "v w/o s"
dat_v_s$grp <- paste(dat_v_s$exp,dat_v_s$subj,sep="")
# c
dat_c <- read.csv("glm_rt-cate.csv")
dat_c$eff <- "c"
dat_c$grp <- paste(dat_c$exp,dat_c$subj,sep="")
# c w/o v
dat_c_v <- read.csv("glm_resid-cate.csv")
dat_c_v$eff <- "c w/o v"
dat_c_v$grp <- paste(dat_c_v$exp,dat_c_v$subj,sep="")


setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_w2v")
# c w/o s
dat_c_s <- read.csv("glm_resid-c.csv")
dat_c_s$eff <- "c w/o s"
dat_c_s$grp <- paste(dat_c_s$exp,dat_c_s$subj,sep="")
# v w/o c&s
dat_v_cs <- read.csv("glm_resid-2avg_v.csv")
dat_v_cs$eff <- "v w/o c&s"
dat_v_cs$grp <- paste(dat_v_cs$exp,dat_v_cs$subj,sep="")
# s
dat_s <- read.csv("glm_rt-w2v.csv")
dat_s$eff <- "s"
dat_s$grp <- paste(dat_s$exp,dat_s$subj,sep="")
# s w/o c
dat_s_c <- read.csv("glm_rt-s-2avg.csv")
dat_s_c$eff <- "s w/o c"
dat_s_c$grp <- paste(dat_s_c$exp,dat_s_c$subj,sep="")
# s w/o v
dat_s_v <- read.csv("glm_resid-w2v.csv")
dat_s_v$eff <- "s w/o v"
dat_s_v$grp <- paste(dat_s_v$exp,dat_s_v$subj,sep="")
# s w/o c&v
dat_s_cv <- read.csv("glm_resid-2avg_s.csv")
dat_s_cv$eff <- "s w/o c&v"
dat_s_cv$grp <- paste(dat_s_cv$exp,dat_s_cv$subj,sep="")
# c w/o v&s
dat_c_vs <- read.csv("glm_resid-c_vs.csv")
dat_c_vs$eff <- "c w/o v&s"
dat_c_vs$grp <- paste(dat_c_vs$exp,dat_c_vs$subj,sep="")

dat.all <- rbind(
  dat_c_v,dat_s_v,dat_s_c,dat_v,dat_v_c,dat_v_s,
  dat_v_cs,dat_s_cv,dat_c_vs,dat_s,dat_c,dat_c_s)
head(dat.all)
tail(dat.all)
dat <- subset(dat.all,layer=="fc_6")
dat$exp <- as.factor(dat$exp)
dat$eff <- as.factor(dat$eff)
dat$setsize <- as.factor(dat$setsize)
head(dat)
tail(dat)



# --- --- ---

# eff_tag <- "v"
# eff_tag <- "v w/o c"
# eff_tag <- "v w/o s"
# eff_tag <- "v w/o c&s"
# eff_tag <- "c w/o v"
eff_tag <- "c w/o s"
# eff_tag <- "s w/o c"
# eff_tag <- "s w/o v"
# eff_tag <- "s w/o c&v"
# eff_tag <- "c w/o v&s"
# eff_tag <- "c"
# eff_tag <- "s"

for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat,(corr==corr_tag)&(eff==eff_tag))
  
  # 2 way: task*set size
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    between=exp,within=setsize,
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  
  # multi-compar
  pwc <- dat.temp %>%
    group_by(exp) %>%
    pairwise_t_test(
      coeff~setsize,paired=TRUE,
      p.adjust.method="bonferroni")
  print(pwc,n=999)
  
}

# --- --- ---
dat.all.layers <- rbind(
  dat_c_v,dat_s_v,dat_v,dat_v_c,
  dat_v_s,dat_v_cs,dat_s_cv,dat_c_vs)
dat <- subset(dat.all.layers,layer=="fc_6")

for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat,(corr==corr_tag))
  
  # 3 way
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    between=exp,within=c(setsize,eff),
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))

}

#
# setsize*eff
dat.avg <- aggregate(
  dat$coeff,by=list(dat$setsize,dat$eff,dat$grp,dat$corr),mean)
dat.avg <- dat.avg %>% 
  rename(
    setsize=Group.1,eff=Group.2,grp=Group.3,corr=Group.4,coeff=x)
for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat.avg,(corr==corr_tag))
  
  # 2-way
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    within=c(setsize,eff),
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  
  aov.1way <- dat.temp %>%
    group_by(setsize) %>%
    anova_test(
      dv=coeff,wid=grp,within=eff,
      effect.size="pes") %>%
    get_anova_table() %>%
    adjust_pvalue(method="bonferroni")
  print("group by setsize")
  print(aov.1way)
  
  # multi-compar
  pwc <- dat.temp %>%
    group_by(setsize) %>%
    pairwise_t_test(
      coeff~eff,paired=TRUE,
      p.adjust.method="bonferroni")
  print(pwc,n=999)
  
  aov.1way <- dat.temp %>%
    group_by(eff) %>%
    anova_test(
      dv=coeff,wid=grp,within=setsize,
      effect.size="pes")%>%
    get_anova_table() %>%
    adjust_pvalue(method="bonferroni")
  print("group by eff")
  print(get_anova_table(aov.1way))
  
  # multi-compar
  pwc <- dat.temp %>%
    group_by(eff) %>%
    pairwise_t_test(
      coeff~setsize,paired=TRUE,
      p.adjust.method="bonferroni")
  print(pwc,n=999)
  
  print("--- --- --- --- --- ---")
  print("")
}

# collaspe set size
# exp*eff
dat.avg.allsimi <- aggregate(
  dat$coeff,by=list(dat$eff,dat$exp,dat$grp,dat$corr),mean)
dat.avg.allsimi <- dat.avg.allsimi %>% 
  rename(
    eff=Group.1,exp=Group.2,grp=Group.3,corr=Group.4,coeff=x)
dat.avg <- subset(
  dat.avg.allsimi,((eff=='v w/o c&s')|
    (eff=='s w/o c&v')|
    (eff=='c w/o v&s')))
for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat.avg,(corr==corr_tag))
  
  # 2-way
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    between=exp,within=eff,
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  
  # 1-way
  aov.1way <- dat.temp %>%
    group_by(exp) %>%
    anova_test(
      dv=coeff,wid=grp,within=eff,
      effect.size="pes") %>%
    get_anova_table() %>%
    adjust_pvalue(method="bonferroni")
  print("group by exp")
  print(aov.1way)
  # multi-compar
  pwc <- dat.temp %>%
    group_by(exp) %>%
    pairwise_t_test(
      coeff~eff,paired=TRUE,
      p.adjust.method="bonferroni")
  print(pwc,n=999)
  
  # 1-way
  aov.1way <- dat.temp %>%
    group_by(eff) %>%
    anova_test(
      dv=coeff,wid=grp,between=exp,
      effect.size="pes") %>%
    get_anova_table()
  print("group by eff")
  print(aov.1way)
  # multi-compar
  pwc <- dat.temp %>%
    group_by(eff) %>%
    pairwise_t_test(
      coeff~exp,paired=F,
      p.adjust.method="bonferroni")
  print(pwc,n=999)
  
  
  # # main
  # # exp
  # MANOVA(data=dat.temp,subID="grp", 
  #        dv="coeff",between=c("exp")) %>%
  #   EMMEANS("exp")
  # 
  # # eff
  # MANOVA(data=dat.temp,subID="grp", 
  #        dv="coeff",within=c("eff")) %>%
  #   EMMEANS("eff")
  
  
  print("--- --- --- --- --- ---")
  print("")
}


# setsize*exp
dat.avg <- aggregate(
  dat$coeff,by=list(dat$setsize,dat$exp,dat$grp,dat$corr),mean)
dat.avg <- dat.avg %>% 
  rename(
    setsize=Group.1,exp=Group.2,grp=Group.3,corr=Group.4,coeff=x)
for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat.avg,(corr==corr_tag))
  
  # 2-way
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    between=exp,within=setsize,
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  
  
  # main
  # setsize
  MANOVA(data=dat.temp,subID="grp", 
         dv="coeff",within=c("setsize")) %>%
    EMMEANS("setsize")
  
  # exp
  MANOVA(data=dat.temp,subID="grp", 
         dv="coeff",between=c("exp")) %>%
    EMMEANS("exp")
  
  
  print("--- --- --- --- --- ---")
  print("")
}


for (corr_tag in c("mean","max")){
  dat.temp <- subset(dat,(corr==corr_tag)&((eff=="c w/o v")|(eff=="s w/o v")))
  
  # 2 way: task*set size
  res.aov <- anova_test(
    data=dat.temp,dv=coeff,wid=grp,
    between=exp,within=c(setsize,eff),
    effect.size="pes")
  
  print(sprintf("%s",corr_tag))
  print(get_anova_table(res.aov))
  
  # multi-compar
  pwc <- dat.temp %>%
    group_by(exp) %>%
    pairwise_t_test(
      coeff~setsize,paired=TRUE,
      p.adjust.method="bonferroni")
  print(pwc,n=999)
  
}









