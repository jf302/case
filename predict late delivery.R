### Load the packages
install.packages('dplyr')
install.packages('plyr')
install.packages("partykit")
install.packages("libcoin")
install.packages("randomForest")
install.packages("tree")
install.packages("ROSE")
install.packages('rpart')
install.packages('rattle')
install.packages('fastDummies')
install.packages('DAAG')
install.packages('party')
install.packages('mlbench')
install.packages('caret')
install.packages('pROC')
install.packages('precrec')
install.packages("glmnet")
install.packages("cvAUC")
install.packages('rfUtilities')
install.packages("RColorBrewer")
install.packages("formattable")
install.packages("devtools")
library(tidyverse)
library(formattable)
library(RColorBrewer)
library(glmnet)
library(plotROC)
library('plyr')
library('dplyr')
library('tidyr')
library('tree')
library(randomForest)
library(libcoin)
library(partykit)
library(ROSE)
library(rpart)
library(rattle)
library('fastDummies')
library(DAAG)
library(party)
library(mlbench)
library(caret)
library(pROC)
library(precrec)
library(cvAUC)
library(rfUtilities)
library(mclust)
library(corrplot)
library(devtools)
install_github("vqv/ggbiplot")
library(ggbiplot)
library(GGally)
library(corrplot)
library(tree)

### Load the dataset 
data <- read.csv("DataCoSupplyChainDataset.csv")


#B2C
#################################
### Data Preparation

### I. Show summary of the dataset: 
summary(data)
### II. Clean the dataset
### 1) Remove variables that do not provide values for prediction 
data <- data[,names(data) %in% c("Late_delivery_risk", "Category.Name", "Customer.Segment", "Market", "Order.Region", "Shipping.Mode")]
summary(data)
### 2) No missing values were discovered in any of the variables
sum(is.na(data))
### 3) Remove the duplicate records from the dataset
data <- data[!duplicated(data),]
summary(data)
### 4) Reduce variable categories
data$Category.Name[data$Category.Name!=c("Cleats", "Men's Footwear", "Women's Apparel","Indoor/Outdoor Games")] = "Others"
data$Order.Region[!data$Order.Region==c("Central America", "Western Europe", "South America")] = "Others"
### 5) Customer segmentation: B2C
dataC <- data[data$Customer.Segment=="Consumer", ]
dataC <- dataC[,!names(dataC) %in% "Customer.Segment"]
### 6) Construct dummy variables for non-numeric variables
Dum_data <- dummy_cols(dataC, select_columns = c("Category.Name", "Market", "Order.Region", "Shipping.Mode"),
                       remove_selected_columns = TRUE)
Dum_data <- Dum_data[,!names(Dum_data) 
                     %in% c("Category.Name_Cleats", 
                            "Market_LATAM",
                            "Order.Region_Central America", 
                            "Shipping.Mode_Standard Class")]
summary(Dum_data)
### data is now ready for use in further analysis
set.seed(42)

#################################
### Data Visualization
### 1) correlation with category name
Category.Name <- c("Late_delivery_risk" ,"Category.Name_Indoor/Outdoor Games","Category.Name_Men's Footwear", "Category.Name_Others", "Category.Name_Women's Apparel")
CorMatrix1 <- cor(Dum_data[,Category.Name])
corrplot(CorMatrix1, method = "square")

### 2) correlation with market
panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE) 
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "steelblue", ...)
}


# function to call to put correlations on the upper panels,
panel.cor <- function(x, y, digits = 2, prefix = "", ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- cor(x, y)
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  text(0.5, 0.5, txt, cex = 1.5)
}

# scatterplot matrix here:
pairs(Dum_data[,c(1,6,7,8,9)], upper.panel=panel.smooth, lower.panel=panel.cor, diag.panel=panel.hist, pch=20)

### 3) correlation with order region
panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE) 
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "steelblue", ...)
}


# function to call to put correlations on the upper panels,
panel.cor <- function(x, y, digits = 2, prefix = "", ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- cor(x, y)
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  text(0.5, 0.5, txt, cex = 1.5)
}

# scatterplot matrix here:
pairs(Dum_data[,c(1,11,12,13)], upper.panel=panel.smooth, lower.panel=panel.cor, diag.panel=panel.hist, pch=20)

### 4) correlation with shipping mode
Shipping.Mode <- c("Late_delivery_risk" ,"Shipping.Mode_First Class","Shipping.Mode_Same Day", "Shipping.Mode_Second Class")
CorMatrix2 <- cor(Dum_data[,Shipping.Mode])
corrplot(CorMatrix2, method = "square")

#################################
### Modeling
colnames(Dum_data) <- c("Latedeliveryrisk", "CategoryNameIndoorOutdoorGames", "CategoryNameMensFootwear", "CategoryNameOthers", "CategoryNameWomensApparel",
                        "MarketAfrica","MarketEurope","MarketPacificAsia", "MarketUSCA", 
                        "OrderRegionOthers", "OrderRegionSouthAmerica", "OrderRegionWesternEurope",
                        "ShippingModeFirstClass", "ShippingModeSameDay", "ShippingModeSecondClass")
### 1) Logistic regression
### Constructed a simple logistic regression model with Latedeliveryrisk as the dependent 
### variable and all the other variables as the independent variables.
log_reg <- glm(Latedeliveryrisk~., data=Dum_data, family="binomial")
summary(log_reg)
log_reg.pred <- predict(log_reg,Dum_data[,!names(Dum_data) %in% c("Latedeliveryrisk")],type = "response")
### R2: 0.03645674
R2(Dum_data$Latedeliveryrisk, log_reg.pred)
### Area under the curve: 0.6003192
AUC(log_reg.pred, Dum_data$Latedeliveryrisk)
rocplot <- ggplot(Dum_data,aes(m = log_reg.pred, d = Latedeliveryrisk))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 
### 2）Logistic regression with interaction
### Constructed a logistic regression model with Risk_Flag as the dependent 
### variable and all the other variables as the independent variables. 
### Interaction terms were created by combining each pair of variables.
log_interact_reg <- glm(Latedeliveryrisk~.^2, data=Dum_data, family="binomial")
summary(log_interact_reg)
log_interact_reg.pred <- predict(log_interact_reg,Dum_data[,!names(Dum_data) %in% c("Latedeliveryrisk")],type = "response")
### R2: 0.04736877
R2(Dum_data$Latedeliveryrisk, log_interact_reg.pred)
### Area under the curve: 0.6134905
AUC(log_interact_reg.pred, Dum_data$Latedeliveryrisk)
rocplot <- ggplot(Dum_data,aes(m = log_interact_reg.pred, d = Latedeliveryrisk))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 
### 3) Classification tree
tree1 <- tree(Latedeliveryrisk ~ ., data = Dum_data)
tree.pred <- predict(tree1,Dum_data[,!names(Dum_data) %in% c("Latedeliveryrisk")],type = "vector")
### R2: 0.03116563
R2(Dum_data$Latedeliveryrisk, tree.pred)
### Area under the curve: 0.5719464
AUC(tree.pred, Dum_data$Latedeliveryrisk)
rocplot <- ggplot(Dum_data,aes(m = tree.pred, d = Latedeliveryrisk))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 
### 4) Random forest
#rf  <- randomForest(formula = Latedeliveryrisk ~ ., data = Dum_data, mtry = 10, ntree = 10, nodesize = 10, importance = TRUE)
#predrf <- predict(rf, newdata = DATANEW[-train,!(names(DATANEW)%in%c("price"))])
# We have so few values so that it is not enough for the random forest to create unique trees.
### 5) K-NN

#################################
### Evaluation
### K Fold Cross Validation
### Create a vector of fold memberships (random order)
n<- nrow(Dum_data)
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

### Create an empty dataframe of results
OOS <- data.frame( logistic=rep(NA,nfold),logistic.interaction=rep(NA,nfold),tree=rep(NA,nfold),null=rep(NA,nfold)) 

### Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## fit the regressions and null model
  logistic <-glm(Latedeliveryrisk==1~., data=Dum_data, subset=train, family="binomial")
  logistic.interaction <-glm(Latedeliveryrisk==1~.^2, data=Dum_data, subset=train, family="binomial")
  tree1 <- tree(Latedeliveryrisk ~ ., data = Dum_data)
  null <-glm(Latedeliveryrisk==1~., data=Dum_data, subset=train,family="binomial")
  ## get predictions: type=response
  pred.logistic <- predict(logistic, newdata=Dum_data[-train,], type="response")
  pred.logistic.interaction <- predict(logistic.interaction, newdata=Dum_data[-train,], type="response")
  tree.pred <- predict(tree1,Dum_data[-train,!(names(Dum_data)%in%c("Latedeliveryrisk"))],type = "vector")
  pred.null <- predict(null, newdata=Dum_data[-train,], type="response")
  
  ## calculate and log AUC
  # Logistic
  OOS$logistic[k] <- AUC(pred.logistic,Dum_data$Latedeliveryrisk[-train]==1)
  OOS$logistic[k]
  # Logistic Interaction
  OOS$logistic.interaction[k] <- AUC(pred.logistic.interaction,Dum_data$Latedeliveryrisk[-train]==1)
  OOS$logistic.interaction[k]
  #Classification Tree
  OOS$tree[k] <- AUC(tree.pred,Dum_data$Latedeliveryrisk[-train]==1)
  OOS$tree[k]
  # Null
  OOS$null[k] <- AUC(pred.null, Dum_data$Latedeliveryrisk[-train]==1)
  OOS$null[k]
  
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

### List the mean of the results stored in the dataframe OOS
colMeans(OOS)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0),
        ylab= bquote( "Out of Sample " ~ AUC), xlab="Fold", names.arg = c(1:5))

### K Fold Cross Validation
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
### create an empty dataframe of results
OOS <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 

### Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## fit the two regressions and null model
  logistic <-glm(Latedeliveryrisk==1~., data=Dum_data, subset=train, family="binomial")
  logistic.interaction <-glm(Latedeliveryrisk==1~.^2, data=Dum_data, subset=train, family="binomial")
  tree <- tree(Latedeliveryrisk ~ ., data = Dum_data, subset=train)
  null <-glm(Latedeliveryrisk==1~., data=Dum_data, subset=train, family="binomial")
  ## get predictions: type=response so we have probabilities
  pred.logistic <- predict(logistic2, newdata=Dum_data[-train,], type="response")
  pred.logistic.interaction <- predict(logistic.interaction, newdata=Dum_data[-train,], type="response")
  pred.tree <- predict(tree, newdata=Dum_data[-train,!(names(Dum_data)%in%c("Latedeliveryrisk"))],type = "vector")
  pred.null <- predict(null, newdata=Dum_data[-train,], type="response")
  ## calculate and log R2
  # Logistic Interaction
  OOS$logistic.interaction[k] <- R2(Dum_data$Latedeliveryrisk[-train]==1, pred.logistic.interaction)
  OOS$logistic.interaction[k]
  # Logistic
  OOS$logistic[k] <- R2(Dum_data$Latedeliveryrisk[-train]==1, pred.logistic)
  OOS$logistic[k]
  # Tree
  OOS$tree[k] <- R2(Dum_data$Latedeliveryrisk[-train]==1, pred.tree)
  OOS$tree[k]
  #Null
  OOS$null[k] <- R2(Dum_data$Latedeliveryrisk[-train]==1, pred.null)
  OOS$null[k]
  #Null Model guess
  sum(y=Dum_data$Latedeliveryrisk[-train]==1)/length(train)
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

barplot(colMeans(OOS), las=2,xpd=FALSE , xlab="", ylim=c(0,0.05), ylab = bquote( "Average Out of Sample "~R^2))

#B2B
#################################
### Data Preparation

### I. Show summary of the dataset: 
summary(data)
### II. Clean the dataset
### 1) Remove variables that do not provide values for prediction 
data <- data[,names(data) %in% c("Late_delivery_risk", "Category.Name", "Customer.Segment", "Market", "Order.Region", "Shipping.Mode")]
summary(data)
### 2) No missing values were discovered in any of the variables
sum(is.na(data))
### 3) Remove the duplicate records from the dataset
data <- data[!duplicated(data),]
summary(data)
### 4) Reduce variable categories
data$Category.Name[!data$Category.Name==c("Cleats", "Men's Footwear", "Women's Apparel","Indoor/Outdoor Games")] = "Others"
data$Order.Region[!data$Order.Region==c("Central America", "Western Europe", "South America")] = "Others"
### 5) Customer segmentation: B2B
dataB <- data[data$Customer.Segment==c("Corporate", "Home Office"), ]
dataB <- dataB[,!names(dataB) %in% "Customer.Segment"]
### 6) Construct dummy variables for non-numeric variables
Dum_data2 <- dummy_cols(dataB, select_columns = c("Category.Name", "Market", "Order.Region", "Shipping.Mode"),
                       remove_selected_columns = TRUE)
Dum_data2 <- Dum_data2[,!names(Dum_data2) 
                     %in% c("Category.Name_Cleats", 
                            "Market_LATAM",
                            "Order.Region_Central America", 
                            "Shipping.Mode_Standard Class")]
summary(Dum_data2)
### data is now ready for use in further analysis
set.seed(42)

#################################
### Modelling
colnames(Dum_data2) <- c("Latedeliveryrisk", "CategoryNameIndoorOutdoorGames", "CategoryNameMensFootwear", "CategoryNameOthers", "CategoryNameWomensApparel",
                        "MarketAfrica","MarketEurope","MarketPacificAsia", "MarketUSCA", 
                        "OrderRegionOthers", "OrderRegionSouthAmerica", "OrderRegionWesternEurope",
                        "ShippingModeFirstClass", "ShippingModeSameDay", "ShippingModeSecondClass")
### 1) Logistic regression
### Constructed a simple logistic regression model with Late_delivery_risk as the dependent 
### variable and all the other variables as the independent variables.
log_reg2 <- glm(Latedeliveryrisk~., data=Dum_data2, family="binomial")
summary(log_reg2)
log_reg.pred2 <- predict(log_reg2,Dum_data2[,!names(Dum_data2) %in% c("Latedeliveryrisk")],type = "response")
### R2: 0.05680148
R2(Dum_data2$Latedeliveryrisk, log_reg.pred2)
### Area under the curve: 0.6232797
AUC(log_reg.pred2, Dum_data2$Latedeliveryrisk)
rocplot2 <- ggplot(Dum_data2,aes(m = log_reg.pred2, d = Latedeliveryrisk))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot2 + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 
### 2）Logistic regression with interaction
### Constructed a logistic regression model with Risk_Flag as the dependent 
### variable and all the other variables as the independent variables. 
### Interaction terms were created by combining each pair of variables.
log_interact_reg2 <- glm(Latedeliveryrisk~.^2, data=Dum_data2, family="binomial")
summary(log_interact_reg2)
log_interact_reg.pred2 <- predict(log_interact_reg2,Dum_data2[,!names(Dum_data2) %in% c("Latedeliveryrisk")],type = "response")
### R2: 0.06734806
R2(Dum_data2$Latedeliveryrisk, log_interact_reg.pred2)
### Area under the curve: 0.6399369
AUC(log_interact_reg.pred2, Dum_data2$Latedeliveryrisk)
rocplot2 <- ggplot(Dum_data2,aes(m = log_interact_reg.pred2, d = Latedeliveryrisk))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot2 + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 
### 3) Classification tree
tree2 <- tree(Latedeliveryrisk ~ ., data = Dum_data2)
tree.pred2 <- predict(tree1,Dum_data2[,!names(Dum_data2) %in% c("Latedeliveryrisk")],type = "vector")
### R2: 0.03116563
R2(Dum_data2$Latedeliveryrisk, tree.pred)
### Area under the curve: 0.5719464
AUC(tree.pred, Dum_data2$Latedeliveryrisk)
rocplot2 <- ggplot(Dum_data2,aes(m = tree.pred, d = Latedeliveryrisk))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot2 + style_roc(theme = theme_grey) + geom_rocci(fill="pink") 
### 4) Random forest
### 5) K-NN

#################################
### Evaluation
### K Fold Cross Validation
### Create a vector of fold memberships (random order)
n2 <- nrow(Dum_data2)
nfold2 <- 5
foldid2 <- rep(1:nfold2,each=ceiling(n/nfold2))[sample(1:n)]

### Create an empty dataframe of results
OOS2 <- data.frame( logistic=rep(NA,nfold),logistic.interaction=rep(NA,nfold),tree=rep(NA,nfold),null=rep(NA,nfold)) 
### Use a for loop to run through the nfold trails
for(k in 1:nfold2){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## fit the regressions and null model
  logistic2 <-glm(Latedeliveryrisk==1~., data=Dum_data2, subset=train, family="binomial")
  logistic.interaction2 <-glm(Latedeliveryrisk==1~.^2, data=Dum_data2, subset=train, family="binomial")
  tree2 <- tree(Latedeliveryrisk ~ ., data = Dum_data2)
  null2 <-glm(Latedeliveryrisk==1~., data=Dum_data2, subset=train, family="binomial")
  ## get predictions: type=response
  pred.logistic2 <- predict(logistic2, newdata=Dum_data2[-train,], type="response")
  pred.logistic.interaction2 <- predict(logistic.interaction2, newdata=Dum_data2[-train,], type="response")
  tree.pred2 <- predict(tree2,Dum_data2[-train,!(names(Dum_data2)%in%c("Latedeliveryrisk"))],type = "vector")
  pred.null2 <- predict(null2, newdata=Dum_data2[-train,], type="response")
  
  ## calculate and log AUC
  # Logistic
  OOS2$logistic[k] <- AUC(pred.logistic2,Dum_data2$Latedeliveryrisk[-train]==1)
  OOS2$logistic[k]
  # Logistic Interaction
  OOS2$logistic.interaction[k] <- AUC(pred.logistic.interaction2,Dum_data2$Latedeliveryrisk[-train]==1)
  OOS2$logistic.interaction[k]
  #Classification Tree
  OOS2$tree[k] <- AUC(tree.pred2,Dum_data2$Latedeliveryrisk[-train]==1)
  OOS2$tree[k]
  # Null
  OOS2$null[k] <- AUC(pred.null2, Dum_data2$Latedeliveryrisk[-train]==1)
  OOS2$null[k]
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

### List the mean of the results stored in the dataframe OOS
colMeans(OOS2)
m.OOS2 <- as.matrix(OOS2)
rownames(m.OOS2) <- c(1:nfold2)
barplot(t(as.matrix(OOS2)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0),
        ylab= bquote( "Out of Sample " ~ AUC), xlab="Fold", names.arg = c(1:5))


### K Fold Cross Validation
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
### create an empty dataframe of results
OOS2 <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 

### Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ## fit the two regressions and null model
  logistic2 <-glm(Latedeliveryrisk==1~., data=Dum_data2, subset=train, family="binomial")
  logistic.interaction2 <-glm(Latedeliveryrisk==1~.^2, data=Dum_data2, subset=train, family="binomial")
  tree2 <- tree(Latedeliveryrisk ~ ., data = Dum_data2, subset=train)
  null2 <-glm(Latedeliveryrisk==1~., data=Dum_data2, subset=train, family="binomial")
  ## get predictions: type=response so we have probabilities
  pred.logistic2 <- predict(logistic2, newdata=Dum_data2[-train,], type="response")
  pred.logistic.interaction2 <- predict(logistic.interaction2, newdata=Dum_data2[-train,], type="response")
  pred.tree2 <- predict(tree2, newdata=Dum_data2[-train,!(names(Dum_data2)%in%c("Latedeliveryrisk"))],type = "vector")
  pred.null2 <- predict(null2, newdata=Dum_data2[-train,], type="response")
  ## calculate and log R2
  # Logistic Interaction
  OOS2$logistic.interaction[k] <- R2(Dum_data2$Latedeliveryrisk[-train]==1, pred.logistic.interaction2)
  OOS2$logistic.interaction[k]
  # Logistic
  OOS2$logistic[k] <- R2(Dum_data2$Latedeliveryrisk[-train]==1, pred.logistic2)
  OOS2$logistic[k]
  # Tree
  OOS2$tree[k] <- R2(Dum_data2$Latedeliveryrisk[-train]==1, pred.tree2)
  OOS2$tree[k]
  #Null
  OOS2$null[k] <- R2(Dum_data2$Latedeliveryrisk[-train]==1, pred.null2)
  OOS2$null[k]
  #Null Model guess
  sum(y=Dum_data2$Latedeliveryrisk[-train]==1)/length(train)
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

barplot(colMeans(OOS2), las=2,xpd=FALSE , xlab="", ylim=c(0,0.05), ylab = bquote( "Average Out of Sample "~R^2))
