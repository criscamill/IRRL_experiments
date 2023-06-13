
 rm(list=ls(all=TRUE))
 dir = 'C:/Users/crist/Documents/Ecomp - POLI/Mestrado/Implementations/Cart-pole/'
 setwd(dir)
 library(ggplot2)
 library(Cairo)
 library(stringi)
 library(zoo)

######################################
# Likelihood analisys reward
######################################

 files_5 <- list.files(paste(dir,'experiment_IRL/reward',sep=''))
 
 likelihood <- c(0.0, 0.1, 0.3, 0.5, 0.7)
 exploration <- c(2, 1, 0.5)

 param_5 <- data.frame(matrix(nrow=length(likelihood)*length(exploration), ncol=4))
 for(i in 1:length(exploration)){
   for(j in 1:length(likelihood)){
     param_5[length(likelihood)*(i-1) + j, 1] <- length(likelihood)*(i-1) + j
     param_5[length(likelihood)*(i-1) + j, 2] <- paste('IRL-CP-',length(likelihood)*(i-1) + j, 
		'-reward.txt', sep='')
     param_5[length(likelihood)*(i-1) + j, -c(1,2)] <- c(likelihood[j], exploration[i])
   }
 }

 param_exp <- param_5[param_5[,4] == 1,]
 data_5 <- data.frame(Episodes=NA, reward=NA, maximum=NA, minimum=NA, 
	Inf_band=NA, Sup_band=NA, experiment=NA)
 data_m_5 <- data.frame(Episodes=NA, reward=NA, sample=NA, experiment=NA)

 for( d in 1:nrow(param_exp)){
    data_rew <- read.csv(paste('experiment_IRL/reward/',param_exp[d,2],sep=''),sep=' ',header=FALSE)

    n <- ncol(data_rew[,-1])
    m <- 30
    reward <- apply(data_rew[,-1],1,mean,na.rm=TRUE)
    reward_m <- rollmean(reward, m)
    quart <- t(apply(data_rew[,-1],1,quantile,c(0.25, 0.5,0.975),na.rm=TRUE))
    sd_rew <- apply(data_rew[,-1],1,sd,na.rm=TRUE)
    Inf_band <- reward - qnorm(0.975)*sd_rew/sqrt(n)
    Sup_band <- sapply(reward + qnorm(0.975)*sd_rew/sqrt(n),min,1)
    maximum <- apply(data_rew[,-1],1,max,na.rm=TRUE)
    minimum <- apply(data_rew[,-1],1,min,na.rm=TRUE)

    data_plot <- data.frame(Episodes=data_rew[,1], reward, maximum, minimum, Inf_band, Sup_band)
    data_5 <- rbind(data_5,cbind(data_plot,experiment=param_exp[d,1]))
    sam <- rep(NA, length(reward_m))
    sam[seq(1,length(reward_m),by=100)] <- reward_m[seq(1,length(reward_m),by=100)]
    data_plot_m <- data.frame(Episodes=1:length(reward_m), reward=reward_m, sample=sam)
    data_m_5 <- rbind(data_m_5,cbind(data_plot_m,experiment=param_exp[d,1]))
 }

 data_5 <- data_5[-1,]
 data_m_5 <- data_m_5[-1,]

 temp_sup <- c()
 temp_inf <- c()

 for(i in 1:length(levels(factor(data_5$experiment)))){
    Sup_band_m <- data_5[data_5$experiment==levels(factor(data_5$experiment))[i],colnames(data_5)=="Sup_band"]
    Sup_band_m <- Sup_band_m[m:length(Sup_band_m)]
    temp_sup <- c(temp_sup, Sup_band_m)
   
    Inf_band_m <- data_5[data_5$experiment==levels(factor(data_5$experiment))[i],colnames(data_5)=="Inf_band"]
    Inf_band_m <- Inf_band_m[m:length(Inf_band_m)]
    temp_inf <- c(temp_inf, Inf_band_m)
 }

 data_m_5$Sup_band <- temp_sup
 data_m_5$Inf_band <- temp_inf

 legend_text <- as.expression(lapply(param_exp[,3], function(d) {
    bquote(italic(L)==.(d))
 }))
 legend_text[1] = 'Autonomous RL'

 title_text <- expression(list(sigma[x] == 1, sigma[j] == 1, mu[j] == 5, gamma == 0.9, alpha[theta] == '0.001', alpha[upsilon] == '0.0001'))
 text_size <- 20

 CairoPDF("experiment_IRL/graph/interactive_smth_s1_RL_new.pdf", 12, 9, bg="transparent", pointsize=12)
 graph <- ggplot(data_m_5, aes(x=Episodes, y=reward, color=factor(experiment),
		linetype=factor(experiment))) + theme_bw() + 
      geom_line() + geom_point(aes(y=sample), shape=18, size=4) +
      geom_ribbon(aes(x=Episodes, ymin=Inf_band, ymax=Sup_band, fill = factor(experiment)),
                      alpha=0.1, show.legend=FALSE, inherit.aes=FALSE) +
	scale_fill_manual(
		values = c("#000000","#FF0000","#00FF00","#0000FF","#F000F0"),
		labels = legend_text) +  
	scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "solid", "solid", "solid"),
		labels = legend_text) + 
	scale_colour_manual(
		values = c("#000000","#FF0000","#00FF00","#0000FF","#F000F0"),
		labels = legend_text) +
	theme(legend.justification=c(1,0), legend.position=c(0.97,0.05), legend.text.align=0,
		plot.title = element_text(size = text_size), 
            plot.subtitle = element_text(size = text_size),
            plot.margin = margin(5.5, 30, 5.5, 5.5, "pt"),
		axis.title = element_text(size = text_size),
            axis.text = element_text(size = text_size),
		legend.text = element_text(size = text_size)) +
	labs(title='Average reward using actor-critic IRL',subtitle=title_text,
      	y ='Average Reward', x='Episodes', color='', linetype='') +
      scale_x_continuous(expand = c(0, 0), limits = c(0,1500))
 print(graph)
 dev.off()

######################################
# Likelihood analisys steps
######################################

 files_5 <- list.files(paste(dir,'experiment_IRL/step',sep=''))
 
 likelihood <- c(0.0, 0.1, 0.3, 0.5, 0.7)
 exploration <- c(2, 1, 0.5)

 param_5 <- data.frame(matrix(nrow=length(likelihood)*length(exploration), ncol=4))
 for(i in 1:length(exploration)){
   for(j in 1:length(likelihood)){
     param_5[length(likelihood)*(i-1) + j, 1] <- length(likelihood)*(i-1) + j
     param_5[length(likelihood)*(i-1) + j, 2] <- paste('IRL-CP-',length(likelihood)*(i-1) + j, 
		'-step.txt', sep='')
     param_5[length(likelihood)*(i-1) + j, -c(1,2)] <- c(likelihood[j], exploration[i])
   }
 }

 param_exp <- param_5[param_5[,4] == 1,]
 data_5 <- data.frame(Episodes=NA, reward=NA, maximum=NA, minimum=NA, 
	Inf_band=NA, Sup_band=NA, experiment=NA)
 data_m_5 <- data.frame(Episodes=NA, reward=NA, sample=NA, experiment=NA)

 for( d in 1:nrow(param_exp)){
    data_rew <- read.csv(paste('experiment_IRL/step/',param_exp[d,2],sep=''),sep=' ',header=FALSE)

    n <- ncol(data_rew[,-1])
    m <- 30
    reward <- apply(data_rew[,-1],1,mean,na.rm=TRUE)
    reward_m <- rollmean(reward, m)
    quart <- t(apply(data_rew[,-1],1,quantile,c(0.25, 0.5,0.975),na.rm=TRUE))
    sd_rew <- apply(data_rew[,-1],1,sd,na.rm=TRUE)
    Inf_band <- reward - qnorm(0.975)*sd_rew/sqrt(n)
    Sup_band <- sapply(reward + qnorm(0.975)*sd_rew/sqrt(n),min,400)
    maximum <- apply(data_rew[,-1],1,max,na.rm=TRUE)
    minimum <- apply(data_rew[,-1],1,min,na.rm=TRUE)

    data_plot <- data.frame(Episodes=data_rew[,1], reward, maximum, minimum, Inf_band, Sup_band)
    data_5 <- rbind(data_5,cbind(data_plot,experiment=param_exp[d,1]))
    sam <- rep(NA, length(reward_m))
    sam[seq(1,length(reward_m),by=100)] <- reward_m[seq(1,length(reward_m),by=100)]
    data_plot_m <- data.frame(Episodes=1:length(reward_m), reward=reward_m, sample=sam)
    data_m_5 <- rbind(data_m_5,cbind(data_plot_m,experiment=param_exp[d,1]))
 }

 data_5 <- data_5[-1,]
 data_m_5 <- data_m_5[-1,]

 temp_sup <- c()
 temp_inf <- c()

 for(i in 1:length(levels(factor(data_5$experiment)))){
    Sup_band_m <- data_5[data_5$experiment==levels(factor(data_5$experiment))[i],colnames(data_5)=="Sup_band"]
    Sup_band_m <- Sup_band_m[m:length(Sup_band_m)]
    temp_sup <- c(temp_sup, Sup_band_m)
   
    Inf_band_m <- data_5[data_5$experiment==levels(factor(data_5$experiment))[i],colnames(data_5)=="Inf_band"]
    Inf_band_m <- Inf_band_m[m:length(Inf_band_m)]
    temp_inf <- c(temp_inf, Inf_band_m)
 }

 data_m_5$Sup_band <- temp_sup
 data_m_5$Inf_band <- temp_inf

 legend_text <- as.expression(lapply(param_exp[,3], function(d) {
    bquote(italic(L)==.(d))
 }))
 legend_text[1] = 'Autonomous RL'

 title_text <- expression(list(sigma[x] == 1, sigma[j] == 1, mu[j] == 5, gamma == 0.9, alpha[theta] == '0.001', alpha[upsilon] == '0.0001'))
 text_size <- 20

 CairoPDF("experiment_IRL/graph/interactive_smth_step_s1_RL_new.pdf", 12, 9, bg="transparent", pointsize=12)
 graph <- ggplot(data_m_5, aes(x=Episodes, y=reward, color=factor(experiment),
		linetype=factor(experiment))) + theme_bw() + 
      geom_line() + geom_point(aes(y=sample), shape=18, size=4) +
      geom_ribbon(aes(x=Episodes, ymin=Inf_band, ymax=Sup_band, fill = factor(experiment)),
                      alpha=0.1, show.legend=FALSE, inherit.aes=FALSE) +
	scale_fill_manual(
		values = c("#000000","#FF0000","#00FF00","#0000FF","#F000F0"),
		labels = legend_text) +  
	scale_linetype_manual(values = c("solid", "solid", "solid", "solid", "solid", "solid", "solid"),
		labels = legend_text) + 
	scale_colour_manual(
		values = c("#000000","#FF0000","#00FF00","#0000FF","#F000F0"),
		labels = legend_text) +
	theme(legend.justification=c(1,0), legend.position=c(0.97,0.05), legend.text.align=0,
		plot.title = element_text(size = text_size), 
            plot.subtitle = element_text(size = text_size),
            plot.margin = margin(5.5, 30, 5.5, 5.5, "pt"),
		axis.title = element_text(size = text_size),
            axis.text = element_text(size = text_size),
		legend.text = element_text(size = text_size)) +
	labs(title='Average steps using actor-critic IRL',subtitle=title_text,
      	y ='Steps', x='Episodes', color='', linetype='') +
      scale_x_continuous(expand = c(0, 0), limits = c(0,1500))
 print(graph)
 dev.off()


