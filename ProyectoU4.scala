//Contenido del proyecto
//    Objectivo: Comparacion del rendimiento siguientes algoritmos de machine learning

//                                 ~~~~~~Limpieza de datos~~~~~
//Importación para la creacion de una sesion spark
import org.apache.spark.sql.SparkSession
//Lo siguiente se aplica para evitar errores ya que nos va marcando cada uno
// y los va categorizando.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//importación de las librerías a utilizar
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType};

// Se crea la variable Spark para crear una sesion y permitira
//dar lectura al archivo csv .
val spark = SparkSession.builder().getOrCreate()

//division de dataset en columnas gracias al la delimitacion del ";" a partir de la lectura del csv a utilizar.
val df = spark.read.option("header", true).option("inferSchema", "true").option("delimiter", ";").csv("bank-full.csv")

//se pivotea la insercion del dataset para que valide los "yes" como 1 ó 0 en caso contrario
val df2 = when($"y".contains("yes"), 1.0).otherwise(0.0)
//se vuelve a pivotear los resultados insertandolos en la columna "y" nuevamente
val df3 = df.withColumn("y", df2)

// pruebas para análisis y visualización del dataset
// df3.printSchema
// df3.show(3)
// df3.columns

//cracion de variables donde se redefinen las columnas en el dataset df3
//para mostrarlo con una sola variable
// val colnames = df3.columns
// //para cuestiones de estetica nos arroja el primer renglon con sus respectivos encabezados
// val firstrow = df3.head(1)(0)
// println("\n")
// println("Example data row")
// for(ind <- Range(1, colnames.length)){
//     println(colnames(ind))
//     println(firstrow(ind))
//     println("\n")
// }

//importaciones que nos ayudaran para transformar los valores en datos binarios
//para la ejecucion.
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoderEstimator}
import org.apache.spark.ml.linalg.Vectors

//Todas las columnas que contengan STRING seran modificadas utilizando el StringIndexer
//Creando un vector indice que convertira estos datos para poderlos manipular, asi que los compacta.
val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex").fit(df3)
val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex").fit(df3)
val eduIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex").fit(df3)
val defaultIndexer = new StringIndexer().setInputCol("default").setOutputCol("defaultIndex").fit(df3)
val housingIndexer = new StringIndexer().setInputCol("housing").setOutputCol("housingIndex").fit(df3)
val loanIndexer = new StringIndexer().setInputCol("loan").setOutputCol("loanIndex").fit(df3)
val contactIndexer = new StringIndexer().setInputCol("contact").setOutputCol("contactIndex").fit(df3)
val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex").fit(df3)

val jobIndexed = jobIndexer.transform(df3)
val maritalIndexed = maritalIndexer.transform(df3)
val eduIndexed = eduIndexer.transform(df3)
val defaultIndexed = defaultIndexer.transform(df3)
val housingIndexed = housingIndexer.transform(df3)
val loanIndexed = loanIndexer.transform(df3)
val contactIndexed = contactIndexer.transform(df3)
val monthIndexed = monthIndexer.transform(df3)

val Encoder = new OneHotEncoderEstimator().setInputCols(Array("jobIndex", "maritalIndex", "educationIndex", "defaultIndex", "housingIndex", "loanIndex", "contactIndex", "monthIndex")).setOutputCols(Array("jobVec", "maritalVec", "educationVec", "defaultVec", "housingVec", "loanVec", "contactVec", "monthVec"))

val assembler = (new VectorAssembler().setInputCols(Array("age","duration", "balance","day","campaign", "previous", "jobVec", "maritalVec", "educationVec", "defaultVec", "housingVec", "loanVec", "contactVec", "monthVec")).setOutputCol("features"))

val Array(training, test) = df3.randomSplit(Array(0.7, 0.3), seed = 12345)

//////////////////////////////    ~~~~Logistic Regresion~~~~     ///////////////////////////////
//Importacion de librerias a utilizar para la corrida del codigo llamando funciones
//y la libreria de la regresion logistica..
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline

//Se declara la variable log_Re para utilizar la regresion logistica en la columna "Y"
//con un maximo de 10 iteraciones.
val log_Re = new LogisticRegression().setLabelCol("y").setMaxIter(10)

//Si le damos un parametro para el setMaxIter y el setRegParam no nos arroja una matriz de confusion correcta
//val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

//utilizamos el pipeline(embudo) para tomar en un arreglo los indexer creados
val pipelinelog_Re = new Pipeline().setStages(Array(jobIndexer,maritalIndexer, eduIndexer, defaultIndexer, housingIndexer, loanIndexer, contactIndexer, monthIndexer, Encoder, assembler,log_Re))

//se realiza el modelo de entrenamiento y se transforman los datos para arrojarlos como resultados de pruebas
val modellog_Re = pipelinelog_Re.fit(training)
val resultslog_Re = modellog_Re.transform(test)

//Variable de prediccion y mediciones.
val predictionAndLabelslog_Re = resultslog_Re.select($"prediction",$"y").as[(Double, Double)].rdd

import org.apache.spark.mllib.evaluation.MulticlassMetrics
val metricsLog_Re = new MulticlassMetrics(predictionAndLabelslog_Re)

//Matriz de confusion donde nos indicara cuales fueron los falsos positivos y falsos negativos.
println("Matriz de confusion: ")
println(metricsLog_Re.confusionMatrix)

//Resultado de la Exactitud
println("Exactitud")
println(metricsLog_Re.accuracy)

////////////////////////////    ~~~~~~~ SVM ~~~~~~~~     ///////////////////////////////
import org.apache.spark.ml.classification.LinearSVC

//Debido a que ya hemos precargado el dataset con el que se va a trabajar
//Se evita la redundancia de informacion y la siguiente linea de codigo
//val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
val l_SVM = new LinearSVC().setLabelCol("y").setMaxIter(10)

val pipelinel_SVM = new Pipeline().setStages(Array(jobIndexer,maritalIndexer, eduIndexer, defaultIndexer, housingIndexer, loanIndexer, contactIndexer, monthIndexer, Encoder, assembler, l_SVM))

val modell_SVM = pipelinel_SVM.fit(training)
val resultsl_SVM = modell_SVM.transform(test)

val predictionAndLabelsl_SVM = resultsl_SVM.select($"prediction",$"y").as[(Double, Double)].rdd

val metricsl_SVM = new MulticlassMetrics(predictionAndLabelsl_SVM)

//println("Confusion matrix:")
println("Matriz de confusion: ")
println(metricsl_SVM.confusionMatrix)
println("Exactitud: ")
println(metricsl_SVM.accuracy)
///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////    ~~~~~~~ MultilayerPerceptron ~~~~~~~~     ///////////////////////////////
//Fue necesaria otra limpieza de datos ya que persistia error en los entrenamientos tanto de multilayer Perceptron
//Ni en arbol de desiciones.
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType};

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", true).option("inferSchema", "true").option("delimiter", ";").csv("bank-full.csv")

//Ver Estructura de los datos.
// df.printSchema
// df.show(3)
// df.head
// df.columns

//Eliminar posibles solumnas con datos nulos.
val data = df.na.drop()

//Reemplazar los datos de tipo String a numéricos.
import org.apache.spark.ml.feature.StringIndexer

val indexer_age = new StringIndexer().setInputCol("age").setOutputCol("InAge")
val indexer_job = new StringIndexer().setInputCol("job").setOutputCol("InJob")
val indexer_marital = new StringIndexer().setInputCol("marital").setOutputCol("InMarital")
val indexer_education = new StringIndexer().setInputCol("education").setOutputCol("InEducation")
val indexer_default = new StringIndexer().setInputCol("default").setOutputCol("InDefault")
val indexer_balance = new StringIndexer().setInputCol("balance").setOutputCol("InBalance")
val indexer_housing = new StringIndexer().setInputCol("housing").setOutputCol("InHousing")
val indexer_loan = new StringIndexer().setInputCol("loan").setOutputCol("InLoan")
val indexer_contact = new StringIndexer().setInputCol("contact").setOutputCol("InContact")
val indexer_day = new StringIndexer().setInputCol("day").setOutputCol("InDay")
val indexer_month = new StringIndexer().setInputCol("month").setOutputCol("InMonth")
val indexer_duration = new StringIndexer().setInputCol("duration").setOutputCol("InDuration")
val indexer_campaign = new StringIndexer().setInputCol("campaign").setOutputCol("InCampaign")
val indexer_pdays = new StringIndexer().setInputCol("pdays").setOutputCol("InPdays")
val indexer_previous = new StringIndexer().setInputCol("previous").setOutputCol("InPrevious")
val indexer_poutcome = new StringIndexer().setInputCol("poutcome").setOutputCol("InPoutcome")
val indexer_y = new StringIndexer().setInputCol("y").setOutputCol("label")

val indexed = indexer_job.fit(data).transform(data)
val data2 = indexed
val indexed = indexer_marital.fit(data2).transform(data2)
val data3 = indexed
val indexed = indexer_education.fit(data3).transform(data3)
val data4 = indexed
val indexed = indexer_default.fit(data4).transform(data4)
val data5 = indexed
val indexed = indexer_housing.fit(data5).transform(data5)
val data6 = indexed
val indexed = indexer_loan.fit(data6).transform(data6)
val data7 = indexed
val indexed = indexer_contact.fit(data7).transform(data7)
val data8 = indexed
val indexed = indexer_month.fit(data8).transform(data8)
val data9 = indexed
val indexed = indexer_poutcome.fit(data9).transform(data9)
val data10 = indexed
val indexed = indexer_y.fit(data10).transform(data10)
val data11 = indexed
val indexed = indexer_age.fit(data11).transform(data11)
val data12 = indexed
val indexed = indexer_balance.fit(data12).transform(data12)
val data13 = indexed
val indexed = indexer_day.fit(data13).transform(data13)
val data14 = indexed
val indexed = indexer_duration.fit(data14).transform(data14)
val data15 = indexed
val indexed = indexer_campaign.fit(data15).transform(data15)
val data16 = indexed
val indexed = indexer_pdays.fit(data16).transform(data16)
val data17 = indexed
val indexed = indexer_previous.fit(data17).transform(data17)

val df2 = indexed
df2.show()

val df3 = df2.select($"label", $"InAge", $"InJob", $"InMarital", $"InEducation", $"InDefault", $"InBalance", $"InHousing",
$"InLoan", $"InContact", $"InDay", $"InMonth", $"InDuration", $"InCampaign", $"InPdays", $"InPrevious", $"InPoutcome")

///////////////////hasta obtenemos la tabla con datos numericos ////////////////////////
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("InAge", "InJob", "InMarital", "InEducation", "InDefault", "InBalance", "InHousing",
"InLoan", "InContact", "InDay", "InMonth", "InDuration", "InCampaign", "InPdays", "InPrevious", "InPoutcome")).setOutputCol("features")

val output = assembler.transform(df3)

output.select("label", "features").show(false)


val mlp = output.select("label", "features")
mlp.show(false)
mlp.printSchema

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row

// Load training data
//val data = MLUtils.loadLibSVMFile(sc, "bank-full.csv").toDF()
//val mlpc = new MLUtils().setLabelCol("y").setMaxIter(10).setRegParam(0.1)


// Load training data
//val data = output
//data.show(2)
// Load training data
// Split the data into train and test
val splits = mlp.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
val layers = Array[Int](16, 2, 2, 5)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// train the model
val model = trainer.fit(train)

val result = model.transform(test)

val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


//////////////////////////////    ~~~~ Arbol de Desiciones ~~~~     ///////////////////////////////
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)// features with > 4 distinct values are treated as continuous.  .fit(data)
// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = df.randomSplit(Array(0.6, 0.4))
// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)
// Make predictions.
val predictions = model.transform(testData)
// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
