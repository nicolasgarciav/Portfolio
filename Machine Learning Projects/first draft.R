library(MLTools)
library(fpp2)
library(ggplot2)
library(readxl)
library(lmtest)  #contains coeftest function
library(tseries) #contains adf.test function
library(forecast)
library(TSA)
library(Hmisc) 
library(tidyselect)
library(dplyr)


#-------------------------------------------------------------------------------
# SARIMA MODEL
#-------------------------------------------------------------------------------

# Load the data
fdata <- read.table('UnemploymentSpain.dat', header = TRUE)
fdata

# Convert to ts
y <- fdata
y <- ts(y$TOTAL,start = 2001, frequency = 12)

y <- y/1000000

autoplot(y)

# Dividing between test and train
y.Tr <- subset(y, end = length(y)-5*12)
y.Ts <- subset(y, start = length(y)-5*12)

autoplot(y.Tr)
autoplot(y.Ts)
# D-F test to check whether the ts is stationary or not, if we can reject the H0
#(p-value smaller than 0.05, ts is stationary)
adf.test(y.Tr,alternative = 'stationary') 
#p-value = 0.9385, therefore ts is non-stationary


ndiffs(y.Tr) #Si vale 1, entonces necesitamos hacer 1 diferenciación

# If differencing is needed
Dy <- diff(y.Tr,differences = 1)

plot(Dy)

# At this point we no longer have a non-stationary ts (p-value = 0.01)
adf.test(y.Tr, alternative = 'stationary', k = 0)


# If ACF decreases slowly, we must differentiate at least once
ggtsdisplay(y.Tr) # Slow decaying ACF suggestive of a non stationary ts


# MODEL 1
arima.fit1 <- Arima(y.Tr, 
                    order=c(1,1,1))
summary(arima.fit1) # summary of training errors and estimated coefficients
coeftest(arima.fit1) # statistical significance of estimated coefficients
autoplot(arima.fit1) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit1, bins = 100)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit1),lag.max = 100)


# MODEL 2
arima.fit2 <- Arima(y.Tr, 
                    order=c(2,0,1))
summary(arima.fit2) # summary of training errors and estimated coefficients
coeftest(arima.fit2) # statistical significance of estimated coefficients
autoplot(arima.fit2) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit2, bins = 100)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit2),lag.max = 100)

# MODEL 3
arima.fit3 <- Arima(y.Tr, 
                    order=c(2,0,1))
summary(arima.fit3) # summary of training errors and estimated coefficients
coeftest(arima.fit3) # statistical significance of estimated coefficients
autoplot(arima.fit3) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit3, bins = 100)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit3),lag.max = 100)



# MODEL 4
arima.fit4 <- Arima(y.Tr,
                    order=c(2,1,1),
                    seasonal = c(1,0,1))
summary(arima.fit4) # summary of training errors and estimated coefficients
coeftest(arima.fit4) # statistical significance of estimated coefficients
autoplot(arima.fit4) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit4, bins = 100)
# If residuals are not white noise, change order of ARMA
ggtsdisplay(residuals(arima.fit4),lag.max = 100)


# Check fitted
autoplot(y.Tr, series = "Real")+
  forecast::autolayer(arima.fit4$fitted, series = "Fitted")

# Perform future forecast
y_est <- forecast(arima.fit4, h = 2)
autoplot(y_est)
y_est

## Validation error for h = 1 --------------------------------------------------
# Obtain the forecast in validation for horizon = 1 using the trained parameters of the model
y.TV.est <- y*NA
for (i in seq(length(y.Tr)+1, length(y), 1)){# loop for validation period
  y.TV.est[i] <- forecast(subset(y,end=i-1), # y series up to sample i 
                          model = arima.fit4,   # Model trained (Also valid for exponential smoothing models)
                          h=1)$mean             # h is the forecast horizon
}

#Plot series and forecast 
autoplot(y)+
  forecast::autolayer(y.TV.est)

#Compute validation errors
accuracy(y.TV.est,y)


aic_sarima <- AIC(arima.fit4)
summary(arima.fit4)

sarima_rmse = 0.0414112 

mape_sarima <- mean(abs((y - arima.fit4$fitted)/y)) * 100
mape_sarima
#-------------------------------------------------------------------------------
# DYNAMIC REGRESSION MODEL
#-------------------------------------------------------------------------------
# Dummy variable for Covid
fdata_c <- read.table('UnemploymentSpain.dat', header = TRUE)

fdata_c$COVID <- rep(0,dim(fdata_c)[1])
fdata_c$COVID[231:250] <- 1

fdata_ts_c <- fdata_c
y_ts <- ts(fdata_ts_c,frequency = 12, start = 2001)
TOTAL <- y_ts[,2]/1000000
COVID <- y_ts[,3]
autoplot(TOTAL)
autoplot(COVID)
auto.arima(TOTAL,xreg = COVID)

## Identification and fitting process ------------------------------------------
#### Fit initial FT model with large s
# This arima function belongs to the TSA package
TF.fit2 <- arima(TOTAL,
                order=c(0,1,2), # el 1 de c(1,0,0) siempre es asi para la primera prueba
                seasonal = list(order=c(0,1,0),period=12),
                xtransf = COVID,
                transfer = list(c(1,1)), #List with (r,s) orders
                include.mean = TRUE,
                method="ML")
summary(TF.fit2) # summary of training errors and estimated coefficients
coeftest(TF.fit2) # statistical significance of estimated coefficients
# Check regression error to see the need of differentiation
TF.RegressionError.plot(TOTAL,COVID,TF.fit2,lag.max = 100)
#NOTE: If this regression error is not stationary in variance,boxcox should be applied to input and output series.CheckResiduals.ICAI(TF.fit, bins = 100)
# Check numerator coefficients of explanatory variable
TF.Identification.plot(COVID,TF.fit2)

TF.fit2$fitted <- fitted(TF.fit2)
TF.fit2$fitted <- na.omit(TF.fit2$fitted)

# Check fitted
autoplot(TOTAL, series = "Real")+
  forecast::autolayer(fitted(TF.fit2), series = "Fitted")

# Perform future forecast
dynamic_est <- forecast(TF.fit2, h = 1)
autoplot(dynamic_est)
dynamic_est

aic_dynamic <- AIC(TF.fit2)
sarima_values <- fitted(TF.fit2)


dynamic_rmse = 0.02803065 
mape_dynamic <- mean(abs((y - TF.fit2$fitted)/y)) * 100
mape_dynamic
#-------------------------------------------------------------------------------
# COMPARSION BETWEEN BOTH MODELS
#-------------------------------------------------------------------------------

# AIC comp
aic_value <- c(aic_dynamic, aic_sarima)
aic_model <- c('Dynamic Regression', 'SARIMA')


aic_comp <- data.frame(aic_value,row.names = aic_model)
aic_comp

ggplot(data = aic_comp) +
  geom_col(mapping = aes(y = aic_value, x = aic_model, fill = aic_model))

# RMSE
rmse_value <- c(dynamic_rmse, sarima_rmse)
rmse_model <- c('Dynamic Regression', 'SARIMA')

rmse_comp <- data.frame(rmse_value,row.names = rmse_model)
rmse_comp

ggplot(data = rmse_comp) +
  geom_col(mapping = aes(y = rmse_value, x = rmse_model, fill = rmse_model))

# MAPE
MAPE_value <- c(mape_dynamic, mape_sarima)
MAPE_model <- c('Dynamic Regression', 'SARIMA')

MAPE_comp <- data.frame(MAPE_value,row.names = MAPE_model)
MAPE_comp

ggplot(data = MAPE_comp) +
  geom_col(mapping = aes(y = MAPE_value, x = MAPE_model, fill = MAPE_model))
  