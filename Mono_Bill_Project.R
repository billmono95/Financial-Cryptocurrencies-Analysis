
# Clear the environment
rm(list=ls())

#install.packages("readr")
#install.packages("neuralnet")
#install.packages("moments")
#install.packages("factoextra")
#install.packages("rattle")

#install.packages("randomForest")
#install.packages("rattle")
# Call libraries
library(caret)
library(readxl)
library(readr)
library(corrplot)
library(ppcor)
library(qgraph)
library(igraph)
library(nnet)
library(neuralnet)
library(Hmisc)
library(pastecs)
library(cluster)
library(moments)
library(factoextra)
library(rpart) 
library(partykit) 
library(rattle) 
library(rpart.plot) 
library(ggplot2) 
library(randomForest) 

#############################          Dataset             ##############################################



Dataset1_Cryptos <- read_csv("Dataset1_Cryptos.csv", 
                                   col_types = cols(Date = col_character()))

Extension_2020  <- read_csv("Crypto_data_extension_2020.csv", 
                                                            col_types = cols(Date = col_character()))

Commodities_Indexes <- read_csv("Commodities_Indexes.csv", 
                                     col_types = cols(Date = col_character(), 
                                                      HSI = col_double(), N225 = col_double()))

Commodities_Indexes <- Commodities_Indexes[,2:9] 

crypto_data_series_complete <- rbind(Dataset1_Cryptos,Extension_2020)

data <- merge(crypto_data_series_complete,Commodities_Indexes,by="Date")

dim(crypto_data_series_complete)
dim(data)


data$Date<- as.Date(data$Date, "%Y-%m-%d")


######################## plot of the trend of variables over time  ###############################


ggplot(data = data, x = Date) +
  geom_line(aes(x = Date, y = ETH, col = "ETH")) +
  geom_line(aes(x = Date, y = BTC, col = "BTC")) +
  geom_line(aes(x = Date, y = XRP, col = "XRP")) +
  geom_line(aes(x = Date, y = LTC, col = "LTC")) +
  geom_line(aes(x = Date, y = LTC, col = "XLM")) +
  scale_colour_discrete(name = "Cryptocurrencies") +  
  theme_bw() +
  labs(y = "Values ($)", x = "Date")

ggplot(data = data, x = Date) +
  geom_line(aes(x = Date, y = Gold, col = "Gold")) +
  geom_line(aes(x = Date, y = BTC, col = "BTC")) +
  geom_line(aes(x = Date, y = Palladium, col = "Palladium")) +
  theme_bw() + scale_color_brewer(palette="Paired", name = "BTC and Commodities")+
  #scale_colour_discrete(name = "BTC and Commodities") +
  labs(y = "Values ($)", x = "Date")

ggplot(data = data, x = Date) +
  geom_line(aes(x = Date, y = HSI, col = "HSI")) +
  geom_line(aes(x = Date, y = BTC, col = "BTC")) +
  geom_line(aes(x = Date, y = N225, col = "N225")) +
  geom_line(aes(x = Date, y = SP500, col = "SP500")) +
  geom_line(aes(x = Date, y = NVIDIA, col = "NVIDIA")) +
  geom_line(aes(x = Date, y = AMD, col = "AMD")) +
  theme_bw() + scale_color_brewer(palette="Paired",  name = "BTC and Indexes stock") +
  labs(y = "Value ($)", x = "Date")



summary(data[-1])

################# Boxplot and removal of the outlier ################################

for (i in 2:ncol(data)) {
  boxplot(data[,i], main = colnames(data)[i])
}


boxplot(data[,2:13], col ='blue')

rmOutlier <- function(x){
  low  <- quantile(x, 0.05, na.rm = T) #replace outlier
  high <- quantile(x, 0.95, na.rm = T) 
  out <- ifelse(x > high, high,ifelse(x < low, low, x)) 
  out }


clean      <- sapply(data[,2:13], rmOutlier)
data.clean <- cbind(clean, data[,2:13])



#data2 <- as.data.frame(sapply(data[,-1], function(x) diff(log(x), lag=1)))
data2<-as.data.frame(sapply(data[,2:13], function(x) diff(log(x), lag=1)))
data2 <- as.data.frame(cbind(data[2:nrow(data),1],data2))
names(data2)[1] <- "Date"
boxplot(data2)



#################### correlation and partial correlation ###############################à




correlations <- cor(data2[,-1])


corrplot(correlations, method="number", tl.col = "blue", bg = "transparent")



p_correlations <- pcor(data2[,-1])
pcor_mat       <- p_correlations$estimate
colnames(pcor_mat) <- c(colnames(correlations))
rownames(pcor_mat) <- c(colnames(correlations))


corrplot(pcor_mat, method="number", tl.col = "blue", bg = "transparent")


#####################################     Linear regression     ############################################


r_squared <- function(fitted_values, observed_values){
  SSE = sum((fitted_values - observed_values)^2)
  SST = sum( (observed_values - mean(observed_values) )^2)
  R2 = 1 - SSE/SST
  return(R2)
  }

rmse <- function(fitted_values, observed_values){
  return(sqrt(mean((fitted_values - observed_values)^2)))
}

set.seed(42)

# Train dataset 80% of the original

lenghtWindowSlice = round(nrow(data2)*0.8)
timeSlices <- createTimeSlices(y = data2$BTC, 
                               initialWindow = lenghtWindowSlice, 
                               fixedWindow = FALSE)
trainSlices <- timeSlices[[1]][[1]]

data_train <- data2[trainSlices,]
data_test <- data2[-trainSlices,]

model <- lm(BTC ~  Palladium + Gold + HSI + N225 + 
              SP500 + NVIDIA + AMD, data = data_train[,-1])
summary(model)

#Evaluate model on test data
fit_out <- predict(model, data_test[,-1])

#Accuracy measures
results<- cbind.data.frame(data_test[,1], data_test$BTC, fit_out)
colnames(results)<-c('Date','Observed','Fitted')

RMSE <- rmse(results$Fitted,results$Observed)
MSE <- (RMSE)^2
R2 <- r_squared(results$Fitted,results$Observed)




############### stepwise regression #################

f_stat <- function(m,m_full,n_obs){
  
  SSE  <- sum(m$residuals^2)
  SSE_full <- sum(m_full$residuals^2)
  
  p <- length(coefficients(m))-1 
  k <- length(coefficients(m_full)) - length(coefficients(m)) 
  
  f_stat_num <- (SSE-SSE_full)/k
  f_stat_den <- SSE_full/(n_obs-p-k-1)
  
  f_stat     <- f_stat_num/f_stat_den
  
  f_pvalue   <- 1-pf(f_stat, df1=k, df2=n_obs-p-k-1)
  return(list(f_stat, f_pvalue))
}

set.seed(42)

## Perform stepwise forward and backward regression and compare with the initial regression

model_no_predictors <- lm(BTC ~ 1, data = data_train[,-1])
summary(model_no_predictors)

model_step_f <- step(model, direction='forward')
summary(model_step_f)

model_step_b <- step(model, direction='backward')
summary(model_step_b)

## Both strategy perform stepwise and backward
model_step_both   <- step(model, direction='both')
summary(model_step_both)

## F-statistic and anova in order discover if the stepwiesed models 
n_observations= nrow(data_train)
F_stat <- f_stat(model_step_both,model, n_observations)
F_stat_value = F_stat[[1]]
F_stat_pvalue = F_stat[[2]]


anova(model,model_step_both)
anova(model_no_predictors,model_step_both)
####

## Accuracy comparison beetwen different stepwise methods

fit_step_f <- predict(model_step_f, data_test[,-1])
fit_step_b <- predict(model_step_b, data_test[,-1])
fit_step_both <- predict(model_step_both, data_test[,-1])

results <- cbind.data.frame(results,fit_step_f,fit_step_b,fit_step_both)
colnames(results)<-c('Date','Observed','Fitted','Fitted_forward','Fitted_backward','Fitted_both')

RMSE_step_f <- rmse(results$Fitted_forward,results$Observed)
MSE_step_f <- (RMSE_step_f)^2
R2_step_f <- r_squared(results$Fitted_forward,results$Observed)

model_comparison_df <- cbind(model_comparison_df,c(MSE_step_f,RMSE_step_f,R2_step_f))

RMSE_step_b <- rmse(results$Fitted_backward,results$Observed)
MSE_step_b <- (RMSE_step_b)^2
R2_step_b <- r_squared(results$Fitted_backward,results$Observed)

model_comparison_df <- cbind(model_comparison_df,c(MSE_step_b,RMSE_step_b,R2_step_b))

RMSE_step_both <- rmse(results$Fitted_both,results$Observed)
MSE_step_both <- (RMSE_step_both)^2
R2_step_both <- r_squared(results$Fitted_both,results$Observed)

model_comparison_df <- cbind(model_comparison_df,c(MSE_step_both,RMSE_step_both,R2_step_both))

colnames(model_comparison_df) <- c('Full_model','Stepwise_forward','Stepwise_backward','Stepwise_both')



##################################### Neural network     ############################################



cor_network <- cor_auto(data2[,-1])


Graph_1 <- qgraph(cor_network, graph = "cor", layout = "spring", edge.width=0.5)
summary(Graph_1) # provides a summary of the network (number of edges)



# Partial correlation network
Graph_2 <- qgraph(cor_network, graph= "pcor", layout = "spring", edge.width=1)
summary(Graph_2)

# Taking into account only statistically significant correlation

Graph_3 <- qgraph(cor_network, graph = "pcor", layout = "spring", edge.width=1, threshold = "sig",
                  sampleSize = nrow(data), alpha = 0.05)
summary(Graph_3)

# Investigate the centrality measures of the graphs 
centralities_Graph1 <- centrality(Graph_1)
centralities_Graph2 <- centrality(Graph_2)
centralities_Graph3 <- centrality(Graph_3)


# Plotting the centrality measures
centralityPlot(Graph_1, include =c("Strength", "Closeness"))


# Compare the two networks
centralityPlot(GGM = list(correlation = Graph_1, partial_correlation = Graph_3)
               ,include =c("Strength", "Closeness"))


##################################### NEURAL NETWORK MODELS ############################################


formula <- BTC ~ .

# Let's fit a neural network on the training dataset. 
set.seed(42)
nn <- neuralnet(formula,
                data = data_train[,-1],
                hidden = c(5, 5,2), 
                linear.output = TRUE,  # the default error option for linear functions is SSE (no need to specify it)
                lifesign = "minimal")


# Plot the neural network 
plot(nn, rep = NULL, x.entry = NULL, x.out = NULL,
     radius = 0.15, arrow.length = 0.2, intercept = TRUE,
     intercept.factor = 0.4, information = TRUE, information.pos = 0.1,
     col.entry.synapse = "blue", col.entry = "blue",
     col.hidden = "black", col.hidden.synapse = "black",
     col.out = "red", col.out.synapse = "red",
     col.intercept = "green", fontsize = 12, dimension = 6,
     show.weights = TRUE)


set.seed(42)
nn_pred <- compute(nn, data_test[,-1])

# Extract results
predicted_nn <- nn_pred$net.result

results <- cbind.data.frame(results,predicted_nn)
colnames(results)<-c('Date','Observed','Fitted','Fitted_forward',
                     'Fitted_backward','Fitted_both','Fitted_nn')

RMSE_nn<- rmse(predicted_nn,results$Observed)
MSE_nn <- (RMSE_nn)^2
R2_nn <- r_squared(predicted_nn,results$Observed)

model_comparison_df <- cbind(model_comparison_df,c(MSE_nn,RMSE_nn,R2_nn))

colnames(model_comparison_df) <- c('Full_model','Stepwise_forward','Stepwise_backward',
                                   'Stepwise_both','Neural_network')


####################################################################################################
##################################### CLUSTER ANALYSIS #############################################
####################################################################################################


data.scale <- data.frame(scale(data2[,-1]))

set.seed(42)
km_fit <- kmeans(data.scale,3)

data_clus <- as.data.frame(cbind(data2, as.factor(km_fit$cluster)))
colnames(data_clus)[14] <- 'Cluster'

cl_table <- table(data_clus$Cluster)

barplot(cl_table, beside = TRUE,
        legend.text = c('Cluster 1', 'Cluster 2', 'Cluster 3'),
        xlab = 'Cluster', ylab = 'Occurrences', col = rainbow(3),
        args.legend = list(x = "topleft"))

ggplot(data_clus, aes(data_clus$Date)) + 
  geom_line(aes(y = BTC, colour = Cluster))+
  labs(y = "Returns ($)", x = "Date") +
  ggtitle('BTC returns\n(Cluster groups)') + theme(plot.title = element_text(hjust = 0.5))+ 


set.seed(42)
fviz_nbclust(data.scale, kmeans, method = "wss",k.max = 6)


####################################################################################################
################################ Hierarchical clustering ###########################################
####################################################################################################


correlation <- cor(data2[,-1])
dist <- 2-2*correlation
dist <- as.dist(dist)

# Hierarchical clustering
hir_fit <- hclust(dist)


# Plot the dendogram 
plot(hir_fit, labels=colnames(data2[,-1]), main='Cryptocurrencies,Indexes and Commodities')

groups <- cutree(hir_fit, k=4)
rect.hclust(hir_fit, k=4, border="red") 

####################################################################################################
####################################### TREE MODELS ############################################
####################################################################################################



set.seed(42)

# Recursive tree
fit_rtree <- rpart(BTC ~ ., data=data_train[,-1])

printcp(fit_rtree)

plot(fit_rtree, margin = 0.2)
text(fit_rtree, cex=0.8) 

predict_rtree <- predict(fit_rtree,data_test[,-1])

# Conditional inference tree
fit_ctree <- ctree(BTC ~ ., data=data_train[,-1])
plot(fit_ctree)

predict_ctree <- predict(fit_ctree,data_test[,-1])

##### Random forest #####

set.seed(42)

fit_rf <- randomForest(BTC ~ ., data = data_train[,-1], na.action=na.roughfix)
plot(fit_rf, main="Errors and number of trees")

#Plot variable importance
varImpPlot(fit_rf, main="Random Forest: Variable Importance")

predict_rf <- predict(fit_rf, newdata = data_test[,-1])


##### Update results dataframe ######

results <- cbind.data.frame(results,predict_rtree,predict_ctree,predict_rf)
colnames(results)<-c('Date','Observed','Fitted','Fitted_forward',
                     'Fitted_backward','Fitted_both','Fitted_nn',
                     'Fitted_rtree','Fitted_ctree','Fitted_rf')

# R-Squared Value recursive tree
RMSE_rtree <- rmse(results$Fitted_rtree,results$Observed)
MSE_rtree <- (RMSE_rtree)^2 
R2_rtree <- r_squared(results$Fitted_rtree,results$Observed)

# R-Squared conditional tree
RMSE_ctree <- rmse(results$Fitted_ctree,results$Observed)
MSE_ctree <- (RMSE_rtree)^2 
R2_ctree <- r_squared(results$Fitted_ctree,results$Observed)

# R-Squared Value for random forest
RMSE_rf <- rmse(predict_rf,results$Observed)
MSE_rf <- RMSE_rf^2 
R2_rf <- r_squared(predict_rf,results$Observed)


model_comparison_df <- cbind(model_comparison_df,c(MSE_rtree,RMSE_rtree,R2_rtree))
model_comparison_df <- cbind(model_comparison_df,c(MSE_ctree,RMSE_ctree,R2_ctree))
model_comparison_df <- cbind(model_comparison_df,c(MSE_rf,RMSE_rf,R2_rf))

colnames(model_comparison_df) <- c('Full_model','Stepwise_forward','Stepwise_backward',
                                   'Stepwise_both','Neural_network',
                                   'Recursive_tree','Conditional_tree','Random_forest')











