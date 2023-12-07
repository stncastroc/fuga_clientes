# fuga_clientes
Modelo para predicción de fuga de clientes en empresa de telecomunicaciones. 
Proyecto final diplomado Diplomado Data Science, Machine Learning e Inteligencia Artificial, Deep Learning ofrecido por la Pontificia Universidad Católica de Chile.
Lenguaje: R

## Problemática 
Las empresas de telecomunicaciones se enfrentan a un problema con la fuga de sus clientes a empresas competidoras. Es por esto, que la empresa que otorga los datos busca implementar un modelo de clasificación que les permita identificar a aquellos clientes que se fugarán, de manera que puedan gestionar un mecanismo de retención/fidelización de aquellos clientes.

## Descripción de datos
El conjunto de datos con el que se trabaja, de nombre “churn-analysis” y proporcionado por la Universidad, corresponde a información de clientes de cierta empresa de telecomunicaciones. El conjunto contienen 3333 registros y encontramos los siguientes atributos:

* state: Región del usuario.
* area.code: Código de área.
* phone.number: Número telefónico.
* international.plan: Plan internacional (yes o no).
* voice.mail.plan: Plan con correo de voz (yes o no).
* number.vmail.messages: Cantidad de mensajes virtuales posee.
* total.day.minutes: Cantidad de minutos diarios.
* total.day.calls: Cantidad de llamadas diarias.
* total.day.charge: Cantidad del costo diario.
* total.eve.minutes: Cantidad de minutos en la tarde.
* total.eve.calls: Cantidad de llamadas en la tarde.
* total.eve.charge: Cantidad de costo en la tarde.
* total.night.minutes: Cantidad de minutos en la noche.
* total.night.calls: Cantidad de llamadas en la noche.
* total.night.charge: Cantidad de costo en la noche.
* total.intl.minutes: Cantidad de minutos internacionales.
* total.intl.calls: Cantidad de llamadas internacionales.
* total.intl.charge: Cantidad de costo internacionales.
* customer.service.calls: Cantidad de llamados a la mesa de ayuda.
* churn: Fuga del cliente (True o False).

## Desarrollo 
Para dar solución a este problema se consideran las siguientes etapas:

* Análisis exploratorio: Se presenta un análisis exploratorio de los datos entregados.
* Evaluación de datos faltantes y/o atípicos: Al evaluar los datos, se implementan técnicas adecuadas para el tratamiento de datos faltantes y/o atípicos.
* Modelo propuesto: Se implementa un modelo en base a los requerimientos necesarios. En este caso, se opta por un modelo de Redes neuronales artificiales (RN).
* Evaluar métricas de validación: Se evalúa la capacidad predictiva del modelo, en particular, la métrica de accuracy y el área bajo la curva.
* Conclusión: Se entregan conclusiones generales en base al contexto del problema.

Un informe completo de la metodología y resultados del proyecto se encuentra en formato PDF aquí: [Predicción de la fuga de
clientes en una empresa de telecomunicaciones](https://drive.google.com/file/d/19h1uxrfBD32vdW44sYrG_Go8Vo7brtKG/view?usp=sharing)

## Resultados
Los resultados estádisticos del modelo final obtenido son los que se presentan a continuación. Se optó por escoger un modelo pequeño compuesto de una capa oculta y 4 neuronas. 

### Métricas del modelo

* Verdaderos positivos (TP): 66
* Falsos positivos (FP): 35
* Verdaderos negativos (TN): 553
* Falsos Negativos (FN): 13
* Accuracy: 0.928
* Recall: 0.977
* Specificity: 0.653
* Precision: 0.940
* AUC (ROC): 0.889

### Curva ROC
![Curva Roc](/images/curvaRoc.jpg)

### Diagrama Red Neuronal 
![Diagrama Red Neuronal](/images/diagramaRed.jpg)


