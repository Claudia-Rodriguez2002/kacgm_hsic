# Reunión preparación 24/07

### Normalización MSE y HSIC

##### MSE

**NMSE = MSE / var(Y).** Esto se materializa en que el NMSE = 1 cuando el modelo estima Y=media. Este caso es malo porque sin entrenamiento ni averiguar relaciones causales, puede estimar que Y=media. Mejor estimación: NMSE = 0 error. No está acotado entre 0 y 1 porque el modelo puede estimar peor que NMSE=1, caso malo.

##### HSIC

**NHSIC = HSIC / sqrt(hsic(xx) * hsic(yy))**. Dividir entre el máximo. Acotado entre 0 y 1 siempre.



### Early stop

- **Arreglo**: Guardado de los parámetros que e**ntregan la mejor funcion de loss**. En vez de antes, que entregaba la última función de loss cuando paraba de entrenar.

- **Enseñar figuras en Funcionamiento early stopping.ipynb** de cómo se detiene debido a los datos de test (dentro de model.fit()) no superando consecutivamente best_loss
  
  

### Experimentos

### Cambios

- Normalizar varianza entre differentes distribuciones de prueba

- Ruido gamma k=2 porque k=1 es una exponencial tb. k=2 da una distribucion asimetrica parecida a una gaussiana pero con más acumulacion en un lateral.

- pruebas para N=100,300,500

##### Experimento 1: suma

- 4 ruidos, media de 10 experimentos

##### Experimento 2: multiplicativo

- 2 ruidos, media de 10 experimentos

##### Experimento 3: dependencia

- 2 ruidos, media de 10 experimentos

- para valorar dependencia de otra variable aleatoria se hacen dos experimentos
  
  - Audio: 
    
    N(1,1) para X>0.5, N(-1,1) para X<0.5. 
    
    Es suficiente para valorar dependencia de X? En realidad sería lo mismo que poner una distribucion igual que X para probar si la red entrena bien este modelo. **Preguntar**
  
  - dependencia en varianza: 
    
    eps_y = var_y * N(0,1)
    
    var_y = 0.4 + 0.8 * abs(X). *No abs(var_x) como te dije*
    
    Lo mismo para var_z pero dependiente en X e Y. 



### Resultados generales y preguntas

- Poca mejora

- Varianza muy alta y no mucha correlacion entre differentes experimentos

- qué concluir?





### Intervenciones

Entrenamiento para N=2000. Intervienes en que Y=1, en vez de la fórmula que lo describe respecto a su padre X.

##### Resultados

Los resultados dan 11 histogramas (1 por beta), que representan la distrucion de Z al intervenir en Y con Y=1: P(Z / do(Y=1)). En cada plot se ve un histograma superpuesto a otro, que compara la distribucion de Z cuando intervienes con Y=1 con la distribución original de los datos sintéticos (ground-truth). Idealmente deberían ser iguales, es decir, cuanto más se parezca la intervencional al ground-truth, quiere decir que el modelo es más robusto para predecir intervenciones sobre las q no se ha entrenado.

##### Preguntas

- Tiene sentido? Creo que sí porque se ve claramente una gausiana en cada plot en la que algunas se parecen más o menos dependiendo de beta. 

- Es conveniente que saques las ecuaciones finales simbólicas que la red crea? 

- Es conveniente medir el **Average Treatment Effect**(ATE)?
