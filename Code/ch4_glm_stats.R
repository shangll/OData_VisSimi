library(rstatix)
library(ez)
library(bruceR)
library(coin)
library(EnvStats)
library(Deducer)
library(boot)

p.crit <- 0.05
size.vec <- c(1,2,4,8)
cate.vec <- c("within","between")
layer.vec <- c(
  "conv_1","conv_2","conv_3","conv_4","conv_5",
  "fc_6","fc_7","fc_8")
simi.vec <- c(
  "v","v w/o c","v w/o s","v w/o c&s",
  "s","s w/o v","s w/o c","s w/o v&c",
  "c","c w/o v","c w/o s","c w/o v&s")
semi.vec <- c("v w/o c&s","s w/o v&c","c w/o v&s")
exp.vec <- c("exp1b","exp2")
options(scipen=100,digits=4)
setwd("D:\\shang\\OneDrive - Radboud Universiteit\\RU-Drive\\DCC\\ch4_LSTM\\res_all")



# ------------------------------------------------------------------------------


#
dat.all <- read.csv("glm_data.csv")
dat6 <- subset(
  dat.all,(layer=="fc_6")&
    (corr=="mean")&
    (simi %in% simi.vec))

dat6$setsize <- as.factor(dat6$setsize)
dat6$exp <- as.factor(dat6$exp)

# layer 6: task*setsize
dat6$simi <- as.factor(dat6$simi)
for (var in simi.vec){
  dat <- subset(dat6,(simi==var))
  aov.2way <- anova_test(
    data=dat,dv=coeff,wid=subj,
    between=exp,within=setsize,
    effect.size="pes",detailed=TRUE)
  
  print(var)
  print(get_anova_table(aov.2way,correction= "GG"))
  
  print("--- --- --- --- --- ---")
}
#
#
#

dat.uniq.simi <- subset(
  dat6,simi %in% semi.vec)
dat.simi.avg <- dat.uniq.simi %>%
  group_by(exp,subj,simi) %>%
  summarise(mean_coeff=mean(coeff),
            .groups = 'drop') %>%
  as.data.frame()
aov.2way <- anova_test(
  data=dat.simi.avg,dv=mean_coeff,wid=subj,
  between=exp,within=simi,
  effect.size="pes")
get_anova_table(aov.2way,correction= "GG")

for (var in semi.vec){
  dat <- subset(dat.simi.avg,simi==var)
  dat$exp <- as.factor(dat$exp)
  dat$mean_coeff <- as.numeric(dat$mean_coeff)
  
  stat.test <- dat %>% 
    t_test(mean_coeff~exp,paired=FALSE) %>%
    add_significance()
  cohens_d(mean_coeff~exp,data=dat) -> d
  # d
  print(var)
  print(sprintf(
    "t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f",
    stat.test$statistic,stat.test$df,
    stat.test$p,d[1,4]))
}


#
#
#
corr_dat <- read.csv("corr_dat_sub.csv")
# nrow(corr_dat)
# t test
corr_dat$cond <- factor(corr_dat$cond,levels=cate.vec)
for (lyr in layer.vec){
  dat <- subset(corr_dat,layer==lyr)
  
  res <- independence_test(
    img_corr~cond,data=dat,alternative="greater",
    distribution=approximate(nresample=1000))
  t_value <- statistic(res)
  p_value <- pvalue(res)
  print(
    sprintf("t = %0.3f, p = %0.3f",t_value,p_value))
}
dat <- subset(corr_dat,layer=="conv_1")
res <- independence_test(
  w2v_corr~cond,data=dat,alternative="greater",
  distribution=approximate(nresample=1000))
t_value <- statistic(res)
p_value <- pvalue(res)
print(
  sprintf("t = %0.3f, p = %0.3f",t_value,p_value))



# for (lyr in layer.vec){
#   dat <- subset(corr_dat,layer==lyr)
#   res <- perm.t.test(
#     x=dat[dat$cond=="within",]$img_corr,
#     y=dat[dat$cond=="between",]$img_corr,
#     alternative="greater",
#     midp=TRUE,
#     B=10000)
#   print(res)
# }
# #
# dat <- subset(corr_dat,layer=="conv_1")
# res <- perm.t.test(
#   x=dat[dat$cond=="within",]$w2v_corr,
#   y=dat[dat$cond=="between",]$w2v_corr,
#   alternative="greater",
#   midp=TRUE,
#   B=10000)
# print(res)
# #
# for (lyr in layer.vec){
#   dat.raw <- subset(corr_dat,layer==lyr)
#   dat <- dat.raw %>%
#     mutate(cond=factor(cond,ordered=TRUE))
#   oneway_test(
#     img_corr~cond,data=dat)
#   set.seed(123)
#   res <- oneway_test(
#     img_corr~cond,data=dat,alternative="greater",
#     distribution=approximate(nresample=10000))
#   print(res)
# }
# #
# dat.raw <- subset(corr_dat,layer=="conv_1")
# dat <- dat.raw %>%
#   mutate(cond=factor(cond,ordered=TRUE))
# oneway_test(
#   w2v_corr~cond,data=dat)
# set.seed(123)
# res <- oneway_test(
#   w2v_corr~cond,data=dat,alternative="greater",
#   distribution=approximate(nresample=10000))
# print(res)
#



for (lyr in layer.vec){
  dat <- subset(corr_dat,layer==lyr)
  perm_out <- twoSamplePermutationTestLocation(
    x=dat[dat$cond=="within",]$img_corr,
    y=dat[dat$cond=="between",]$img_corr,
    fcn="mean",
    alternative="greater",
    mu1.minus.mu2=0,
    paired=FALSE,
    exact=FALSE,
    n.permutations=10000,
    seed=123)
  print(perm_out)
}
#
perm_out <- twoSamplePermutationTestLocation(
  x=corr_dat[
    (corr_dat$cond=="within")&
      (corr_dat$layer=="conv_1"),]$w2v_corr,
  y=corr_dat[
    (corr_dat$cond=="between")&
      (corr_dat$layer=="conv_1"),]$w2v_corr,
  fcn="mean",
  alternative="greater",
  mu1.minus.mu2=0,
  paired=FALSE,
  exact=FALSE,
  n.permutations=10000,
  seed=123)
print(perm_out)



#
#
# correlation
perm_cor <- function(data,indices){
  cor(data[indices,1],data[indices,2],method="spearman")
}

for (var in cate.vec){
  print(var)
  
  for (lyr in layer.vec){
    dat.lyr <- subset(corr_dat,layer==lyr)
    x=dat.lyr[dat.lyr$cond==var,]$img_corr
    y=dat.lyr[dat.lyr$cond==var,]$w2v_corr
    dat <- data.frame(x,y)
    
    res <- boot(dat,perm_cor,R=1000)
    rho_value <- res$t0
    rho_boot <- res$t
    p_value <- mean(abs(rho_boot)>=abs(rho_value))
    
    print(lyr)
    print(
      sprintf("rho = %0.3f, p = %0.3f",rho_value,p_value))
    # cat("Spearman rho:",rho_value,"\n")
    # cat("Permutation p-value:",p_value,"\n")
  }
  print("--- --- ---")
}


for (lyr in layer.vec){
  dat.lyr <- subset(corr_dat,layer==lyr)
  x=dat.lyr$img_corr
  y=dat.lyr$w2v_corr
  dat <- data.frame(x,y)
  
  res <- boot(dat,perm_cor,R=1000)
  rho_value <- res$t0
  rho_boot <- res$t
  p_value <- mean(abs(rho_boot)>=abs(rho_value))
  
  print(lyr)
  print(
    sprintf("rho = %0.3f, p = %0.3f",rho_value,p_value))
  # cat("Spearman rho:",rho_value,"\n")
  # cat("Permutation p-value:",p_value,"\n")
}


