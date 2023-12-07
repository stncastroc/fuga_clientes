#Librerias
library(mlbench)
library(ggplot2)
library(neuralnet)
library(dplyr)
library(caret)
library(nnet)
library(corrplot)
library(pROC)

#Cargado de la base de datos
clients <- read.csv("churn-analysis.csv", sep = ";")
clients

#Análisis exploratorio
clients %>% head
clients %>% str
clients %>% summary

##Separamos variable objetivo y de variables explicativas X
y <- clients$churn
## Verificamos
y %>% str
y %>% summary

### Para las variables explicativas eliminamos las variables churn, phone.number y state
### Considero que la variable phone.number no aporta a la implementación del modelo, ya que es un
### identificador único de cada cliente. También, a pesar que es diferente, la variable area.code
### puede explicar infomación similar a la variable state, siendo la variable area.code ya un valor 
### numérico con el que preferimos trabajar. Por lo que en este primero intento, eliminaremos state
X <- clients %>% select(-c(churn,phone.number,state))
X %>% str
### Extraemos aquí las variables numéricas para un posterior analisis boxplot
numeric_X <- X %>%
  select_if(is.numeric)
## Verificamos
numeric_X %>% str

## Seleccionamos y convertimos las variables categóricas a numéricas, ya que de ésta manera trabajaremos mejor con una RN.
## además, las almacenamos en una nueva variable. 
cat_X <- X %>%
  select(international.plan, voice.mail.plan) %>%
  mutate(international.plan = ifelse(international.plan == "yes", 1, 0),
         voice.mail.plan = ifelse(voice.mail.plan == "yes", 1, 0))
## Volvemos a verificar
cat_X %>% str

## Haremos lo mismo para la variable objetivo churn. Aplicamos directamente ifelse, ya que mutate solo es para DF
y <- ifelse(y == "True", 1, 0)
## Verificamos una vez más
y %>% str
y %>% summary

# Correlaciones 
## Analizaremos la correlación entre las variables numéricas. Esto lo
## hacemos a través de la correlación de Pearson. Recordemos que:
#El coeficiente de correlación de Pearson es un valor que varía entre -1 y 1, donde:
#1 significa una correlación positiva perfecta (ambas variables aumentan juntas en la misma proporción).
#-1 significa una correlación negativa perfecta (una variable aumenta mientras la otra disminuye en la misma proporción).
#0 significa que no hay correlación lineal entre las variables.
cor_matrix <- cor(numeric_X)
print(cor_matrix)
corrplot(cor_matrix, method = "color")
title(main = "Matriz de Correlación de Variables Numéricas")

## Eliminamos algunas de las variables altamente correlacionadas (0.999) para evitar multicolinealidad en el modelo.
numeric_X <- numeric_X %>% select(-c(total.day.minutes, total.eve.minutes, total.night.minutes, total.intl.minutes))
## Verificamos
numeric_X %>% str 

## Continuamos con el análisis, veamos si existen valores atípicos graficamente en un boxplot
boxplot(numeric_X, main = "Diagrama de caja y bigote", boxwex = 0.6, cex.axis = 0.8)

## Notamos que las variables se encuentran en escalas distintas,
## Y ya que aplicaremos un modelo de RN, con gradiente de descenso, 
## este puede verse beneficiado de la normalizacion, por lo que normalizaremos los datos
## Además, la normalización puede reducir la influencia de valores atípicos y extremos en tus datos.
# Normalización
numeric_X <- numeric_X %>% mutate_if(is.numeric,scale)
numeric_X %>% str

## Y volvemos a visualizar el boxplot
boxplot(numeric_X, main="Diagrama de caja y bigote")

## Encontramos gran cantidad de valores atipicos en nuestros datos
## estos valores pueden influir negativamente en los pesos de  nuestra red 
## neuronal, e incluso pueden provocar sobreajuste del modelo. Sin embargo, 
## probaremos para ver como se desarrolla el entrenamiento sin manejar estos
## valores en este momento. 
## Además La normalización puede reducir la influencia de valores atípicos y extremos en tus datos.

# Ahora unimos nuestros todos nuestros datos y verificamos
X <- cbind(numeric_X, cat_X)
X %>% str

df_preproc <- X %>% mutate(y=y)
df_preproc %>% head
df_preproc %>% str
df_preproc %>% summary

# Verificar duplicados en df_preproc
duplicated_rows <- duplicated(df_preproc)
duplicated_indices <- which(duplicated_rows)

if (length(duplicated_indices) == 0) {
  cat("No se encontraron datos duplicados en df_preproc.\n")
} else {
  cat("Se encontraron datos duplicados en las filas:", paste(duplicated_indices, collapse = ", "), "\n")
}

# Revisar datos faltantes
sum(is.na(df_preproc))
# Revisar que variables tienen datos faltantes
nombres <- names(df_preproc)
sum_name <- NULL
name <- NULL
for (i in 1:length(nombres)){
  name[i] = nombres[i]
  sum_name[i] = sum(is.na(df_preproc[,nombres[i]]))
}
nulls <- data.frame(name,sum_name)
colnames(nulls) <- c("nombre_columna","cantidad_nulos")
nulls

### Como se observa en el resultado, no tenemos variables con datos faltantes.

#Revisemos los porcentajes de informacion
df_preproc %>% group_by(y)%>% summarise(conteo=n(), porcentaje = conteo/nrow(df_preproc)*100)

# Separar conjunto de entrenamiento y prueba (80 - 20) (Recomendado)
idx <- sample(1:nrow(df_preproc),
              round(nrow(df_preproc)*0.8))
train <- df_preproc[idx,]
test <- df_preproc[-idx,]

# Validar el balance de la variable objetivo 
### (Que tenga sentido con los porcentajes calculados antes)
train %>% group_by(y) %>% 
  summarise(n=n(),
            porcentaje=n/nrow(train)*100)
test %>% group_by(y) %>%
  summarise(n=n(),
            porcentaje=n/nrow(test)*100)

### Tiene sentido, mantenemos el mismo balance. 

# Entrenamiento de la red con la función neuralnet
#?neuralnet
fit <- neuralnet(y~.,
                 data=train,
                 hidden=c(2),         
                 err.fct = "sse",     
                 act.fct = "logistic", 
                 linear.output=FALSE,
                 stepmax=100000)     

#El numero de capas ocultas se determina por ensayo y error, partir siempre de menos a mas.
# GRÁFICA DE LA RED
plot(fit)
# FUNCIÓN DE ERROR
fit$err.fct
# FUNCIÓN DE ACTIVACIÓN
fit$act.fct
# PESOS INICIALES
fit$startweights
# PESOS FINALES
fit$weights
# RESULTADOS DEL CONJUNTO DE ENTRENAMIENTO
fit$net.result
# PESOS GENERALIZADOS
fit$generalized.weights
# MATRIZ DE RESULTADOS
fit$result.matrix

# PREDICCIONES
y_pred <- predict(fit, newdata = test, type = "response")
y_pred <- as.factor(ifelse(y_pred > 0.5, 1, 0))

levels(y_pred)
levels(test$y)
test$y <- factor(test$y, levels = levels(y_pred))

# MÉTRICAS
conf <- caret::confusionMatrix(y_pred,test$y)
conf$table
conf$overall["Accuracy"]
conf$byClass

# Curva ROC
y_probs <- predict(fit, newdata = test, type = "response")
y_numeric <- as.numeric(as.character(test$y))
y_probs_numeric <- as.numeric(as.character(y_probs))
roc_obj <- roc(y_numeric, y_probs_numeric)
plot(roc_obj, main = "Curva ROC")
print(roc_obj)
# Guardar el modelo en un archivo .rds
saveRDS(fit, "modelo_entrenado_ac_9370315.rds")
## Para cargar 
# modelo_cargado <- readRDS("modelo_entrenado.rds")

######################## FIT 2
fit2 <- neuralnet(y~.,
                 data=train,
                 hidden=c(4),         
                 err.fct = "sse",     
                 act.fct = "logistic", 
                 linear.output=FALSE,
                 stepmax=100000)   
plot(fit2)

y_pred2 <- predict(fit2, newdata = test, type = "response")
y_pred2 <- as.factor(ifelse(y_pred2 > 0.5, 1, 0))
# MÉTRICAS
conf2 <- caret::confusionMatrix(y_pred2,test$y)
conf2$table
conf2$overall["Accuracy"]
conf2$byClass

# Curva ROC
y_probs2 <- predict(fit2, newdata = test, type = "response")
y_probs_numeric2 <- as.numeric(as.character(y_probs2))
roc_obj2 <- roc(y_numeric, y_probs_numeric2)
plot(roc_obj2, main = "Curva ROC")
print(roc_obj2)


######################## FIT 3
fit3 <- neuralnet(y~.,
                  data=train,
                  hidden=c(2,2),         
                  err.fct = "sse",     
                  act.fct = "logistic", 
                  linear.output=FALSE,
                  stepmax=100000)   
plot(fit3)

y_pred3 <- predict(fit3, newdata = test, type = "response")
y_pred3 <- as.factor(ifelse(y_pred3 > 0.5, 1, 0))
# MÉTRICAS
conf3 <- caret::confusionMatrix(y_pred3,test$y)
conf3$table
conf3$overall["Accuracy"]
conf3$byClass

# Curva ROC
y_probs3 <- predict(fit3, newdata = test, type = "response")
y_probs_numeric3 <- as.numeric(as.character(y_probs3))
roc_obj3 <- roc(y_numeric, y_probs_numeric3)
plot(roc_obj3, main = "Curva ROC")
print(roc_obj3)

######################## FIT 4
fit4 <- neuralnet(y~.,
                  data=train,
                  hidden=c(3,3),         
                  err.fct = "sse",     
                  act.fct = "logistic", 
                  linear.output=FALSE,
                  stepmax=300000)   
plot(fit4)

y_pred4 <- predict(fit4, newdata = test, type = "response")
y_pred4 <- as.factor(ifelse(y_pred4 > 0.5, 1, 0))
# MÉTRICAS
conf4 <- caret::confusionMatrix(y_pred4,test$y)
conf4$table
conf4$overall["Accuracy"]
conf4$byClass

# Curva ROC
y_probs4 <- predict(fit4, newdata = test, type = "response")
y_probs_numeric4 <- as.numeric(as.character(y_probs4))
roc_obj4 <- roc(y_numeric, y_probs_numeric4)
plot(roc_obj4, main = "Curva ROC")
print(roc_obj4)

######################## FIT 5
fit5 <- neuralnet(y~.,
                  data=train,
                  hidden=c(3),         
                  err.fct = "sse",     
                  act.fct = "logistic", 
                  linear.output=FALSE,
                  stepmax=100000)   
plot(fit5)

y_pred5 <- predict(fit5, newdata = test, type = "response")
y_pred5 <- as.factor(ifelse(y_pred5 > 0.5, 1, 0))
# MÉTRICAS
conf5 <- caret::confusionMatrix(y_pred5,test$y)
conf5$table
conf5$overall["Accuracy"]
conf5$byClass

# Curva ROC
y_probs5 <- predict(fit5, newdata = test, type = "response")
y_probs_numeric5 <- as.numeric(as.character(y_probs5))
roc_obj5 <- roc(y_numeric, y_probs_numeric5)
plot(roc_obj5, main = "Curva ROC")
print(roc_obj5)
