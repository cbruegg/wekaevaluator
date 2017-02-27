package com.cbruegg

import weka.attributeSelection.CfsSubsetEval
import weka.attributeSelection.GreedyStepwise
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.lazy.IBk
import weka.classifiers.meta.AttributeSelectedClassifier
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
import weka.core.Instance
import weka.core.Instances
import weka.core.converters.ConverterUtils
import java.io.File
import java.util.*
import kotlin.concurrent.thread

enum class ClassifierMode {
    RF, MULTIPLE_RF, ALL
}

sealed class ValidateMode(val useFeatureSelection: Boolean, val classifierMode: ClassifierMode, val evalConvergence: Boolean) {
    class RandomCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: ClassifierMode, evalConvergence: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)

    class UserCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: ClassifierMode, evalConvergence: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)

    class PerUserCrossValidation(useFeatureSelection: Boolean,
                                 useAllClassifiers: ClassifierMode,
                                 evalConvergence: Boolean,
                                 val personal: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)

    class PersonalRandomCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: ClassifierMode, evalConvergence: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)
}

private const val FLAG_RANDOM_CROSS_VALIDATION = "--random-cross-validation"
private const val FLAG_USE_FEATURE_SELECTION = "--feature-selection"
private const val FLAG_USE_ALL_CLASSIFIERS = "--use-all-classifiers"
private const val FLAG_VARY_RF_PARAMS = "--vary-rf-params"
private const val FLAG_PERSONAL_MODEL = "--personal"
private const val FLAG_EVAL_PER_USER = "--eval-per-user"
private const val FLAG_ACCURACY_CONVERGENCE = "--eval-convergence"
private const val FLAG_PREDICT_USER = "--predict-user"
private val random = Random(0)

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Usage: java -jar xxx.jar <trainingset> <flags>")
        println("$FLAG_RANDOM_CROSS_VALIDATION will make cross validation be performed with random instances being taken out. Otherwise users will be taken out.")
        println("$FLAG_USE_FEATURE_SELECTION will perform initial feature selection.")
        println("$FLAG_USE_ALL_CLASSIFIERS will use all classifiers instead of just RF.")
        println("$FLAG_PERSONAL_MODEL will perform cross-validation on models for one user only. Cannot be used in conjunction with $FLAG_RANDOM_CROSS_VALIDATION")
        println("$FLAG_VARY_RF_PARAMS will use multiple RF models and vary their parameters. Cannot be used in conjunction with $FLAG_USE_ALL_CLASSIFIERS")
        println("$FLAG_ACCURACY_CONVERGENCE will evaluate the model with 1 to N users.")
        println("$FLAG_EVAL_PER_USER will evaluate a model for each user and print its results.")
        println("$FLAG_PREDICT_USER will cross-validate RF-models predicting the username. Can only be used with a trainingset file, not multiple files in a directory.")
        return
    }

    val input = File(args[0])
    val useFeatureSelection = FLAG_USE_FEATURE_SELECTION in args
    val useAllClassifiers = FLAG_USE_ALL_CLASSIFIERS in args
    val varyRfParams = FLAG_VARY_RF_PARAMS in args
    val evalConvergence = FLAG_ACCURACY_CONVERGENCE in args
    val evalPerUser = FLAG_EVAL_PER_USER in args
    val personal = FLAG_PERSONAL_MODEL in args
    val classifierMode =
            if (useAllClassifiers) ClassifierMode.ALL
            else if (varyRfParams) ClassifierMode.MULTIPLE_RF
            else ClassifierMode.RF

    if (FLAG_PREDICT_USER in args) {
        if (input.isDirectory) {
            input.listFiles()
                    .forEach(::evaluateUserPrediction)
        } else {
            evaluateUserPrediction(input)
        }

        return
    }

    val validateMode =
            if (evalPerUser) {
                ValidateMode.PerUserCrossValidation(useFeatureSelection, classifierMode, evalConvergence, personal)
            } else if (FLAG_RANDOM_CROSS_VALIDATION in args) {
                ValidateMode.RandomCrossValidation(useFeatureSelection, classifierMode, evalConvergence)
            } else if (personal) {
                ValidateMode.PersonalRandomCrossValidation(useFeatureSelection, classifierMode, evalConvergence)
            } else ValidateMode.UserCrossValidation(useFeatureSelection, classifierMode, evalConvergence)


    if (input.isDirectory) {
        val results = Collections.synchronizedMap(mutableMapOf<File, String>())
        input.listFiles()
                .filter { it.absolutePath.endsWith(".csv") }
                .map {
                    thread {
                        results[it] = evaluate(it, validateMode)
                    }
                }.forEach(Thread::join)
        results.entries.sortedBy { it.key.path }.map { it.value }.forEach(::println)
    } else {
        println(evaluate(input, validateMode))
    }
}

fun Classifier.toAttributeSelectedClassifier() = AttributeSelectedClassifier().apply {
    classifier = this@toAttributeSelectedClassifier
    evaluator = CfsSubsetEval()
    search = GreedyStepwise().apply {
        searchBackwards = true
    }
}

private fun evaluateUserPrediction(file: File) {
    val fullDataset = loadDataFromFile(file, classAttr = "username")
    fullDataset.deleteAttributeAt(fullDataset.attribute("sampleClass").index())

    val model = RandomForest().apply {
        numIterations = 50
        numFeatures = 10 // -K 10 (Number of attributes to randomly investigate)
        maxDepth = 25
    }

    val eval = Evaluation(fullDataset).apply {
        crossValidateModel(model, fullDataset, 10, Random(1))
    }

    println("=== Results of username prediction with file $file ===")
    println(eval.toSummaryString("", false))
    println("=== Confusion Matrix ===")
    println(eval.toMatrixString(""))
}

private fun Instances.extractUsers(): List<String> = map { it.stringValue(attribute("username")) }.distinct()

private fun evaluate(input: File, validateMode: ValidateMode): String {
    val resultsByModel = mutableMapOf<String, String>()
    val threads = mutableListOf<Thread>()


    for ((description, baseModel) in models(validateMode.classifierMode)) {
        val model = if (validateMode.useFeatureSelection)
            baseModel.toAttributeSelectedClassifier()
        else baseModel

        threads += thread {
            fun eval(data: Instances, validateMode: ValidateMode): Evaluation {
                val users = data.extractUsers()
                val usernameAttrIndex = data.attribute("username").index()
                val userByInstance = data.associate {
                    Arrays.hashCode(it.numericalValues()) to it.stringValue(usernameAttrIndex)
                }
                return when (validateMode) {
                    is ValidateMode.PersonalRandomCrossValidation -> {
                        Evaluation(data).apply {
                            for (user in users) {
                                val dataFromUser = Instances(data).apply {
                                    for (i in indices.reversed()) {
                                        val instanceUser = userByInstance[Arrays.hashCode(this[i].numericalValues())]
                                        if (instanceUser != user) {
                                            removeAt(i)
                                        }
                                    }
                                }
                                model.buildClassifier(dataFromUser)
                                crossValidateModel(model, dataFromUser, 10, Random(1))
                            }
                        }
                    }
                    is ValidateMode.RandomCrossValidation -> {
                        data.deleteAttributeAt(usernameAttrIndex)
                        model.buildClassifier(data)
                        Evaluation(data).apply {
                            crossValidateModel(model, data, 10, Random(1))
                        }
                    }
                    is ValidateMode.UserCrossValidation -> {
                        data.deleteAttributeAt(usernameAttrIndex)
                        model.buildClassifier(data)

                        Evaluation(data).apply {
                            crossValidateModel(model, data, emptyArray(), Random(1)) {
                                users.indexOf(userByInstance[Arrays.hashCode(it.numericalValues())]!!)
                            }
                        }
                    }
                    is ValidateMode.PerUserCrossValidation -> throw IllegalArgumentException()
                }
            }

            val fullDataset = loadDataFromFile(input)
            val users = fullDataset.extractUsers()
            val usernameAttrIndex = fullDataset.attribute("username").index()
            val userByInstance = fullDataset.associate {
                Arrays.hashCode(it.numericalValues()) to it.stringValue(usernameAttrIndex)
            }

            resultsByModel[description] =
                    if (validateMode is ValidateMode.PerUserCrossValidation) {
                        val personal = validateMode.personal
                        users
                                .map { user ->
                                    val dataWithoutUser = Instances(fullDataset).apply {
                                        for (i in indices.reversed()) {
                                            val instanceUser = userByInstance[Arrays.hashCode(this[i].numericalValues())]
                                            if (instanceUser == user) {
                                                removeAt(i)
                                            }
                                        }
                                    }
                                    val dataFromUser = Instances(fullDataset).apply {
                                        for (i in indices.reversed()) {
                                            val instanceUser = userByInstance[Arrays.hashCode(this[i].numericalValues())]
                                            if (instanceUser != user) {
                                                removeAt(i)
                                            }
                                        }
                                    }

                                    val eval = if (personal) {
                                        Evaluation(dataFromUser).apply {
                                            crossValidateModel(model, dataFromUser, 10, Random(1))
                                        }
                                    } else {
                                        model.buildClassifier(dataWithoutUser)
                                        Evaluation(dataWithoutUser).apply {
                                            evaluateModel(model, dataFromUser)
                                        }
                                    }

                                    user to eval
                                }
                                .map { userToEvaled ->
                                    """
|+++ TRAINING $description for user ${userToEvaled.first} +++
|=== Results of $description ===
|${userToEvaled.second.toSummaryString("", false)}
|=== Confusion Matrix of $description ===
|${userToEvaled.second.toMatrixString("")}
|""".trimMargin()
                                }
                                .joinToString(separator = "\n")
                    } else if (validateMode.evalConvergence) {
                        val reducedDatasets = fullDataset.generateSubsets(users, userByInstance,
                                howManyMaxPerSize = 10, random = random)
                        val datasetsEvals = reducedDatasets.map { eval(it.second, validateMode).pctCorrect() to it.first }
                        val accuraciesBySize = datasetsEvals.groupBy { it.second }
                        val avgAccuraciesBySize = accuraciesBySize.mapValues { it.value.asSequence().map { it.first }.average() }
                        val accuracyTable = avgAccuraciesBySize.entries.joinToString(separator = "\n") {
                            "${it.key},${it.value}"
                        }
                        """
|+++ Evaluating convergence with $description +++
|subset_size,avg_pct_correct""".trimMargin().trim() + "\n" + accuracyTable
                    } else {
                        val evaled = eval(fullDataset, validateMode)
                        """
|+++ TRAINING $description +++
|=== Results of $description ===
|${evaled.toSummaryString("", false)}
|=== Confusion Matrix of $description ===
|${evaled.toMatrixString("")}
|""".trimMargin()
                    }
        }
    }
    threads.forEach(Thread::join)
    return resultsByModel.entries.sortedBy {
        it.key
    }.map {
        it.value
    }.joinToString(separator = "", prefix = "Now evaluating file $input.\n")
}

/**
 * Generate subsets of the complete dataset to evaluate the convergence
 * of the accuracy depending on the amount of data we have. This
 * method will generate [Instances] with 2, ..., N users, [howManyMaxPerSize] each at max.
 *
 * @param [userByInstance] val instanceUser = userByInstance[Arrays.hashCode(instance.numericalValues())]
 * @return seq of userCount to sublists
 */
fun Instances.generateSubsets(users: List<String>,
                              userByInstance: Map<Int, String>,
                              howManyMaxPerSize: Int,
                              random: Random) =
        (2..users.size)
                .asSequence()
                .flatMap {
                    users.randomSubLists(
                            howMany = Math.min(howManyMaxPerSize, binomialCoefficient(n = users.size, k = it).toInt()),
                            ofSize = it,
                            random = random)
                }
                .map { usersToKeep ->
                    usersToKeep.size to Instances(this).apply {
                        for (i in indices.reversed()) {
                            val instanceUser = userByInstance[Arrays.hashCode(this[i].numericalValues())]
                            if (instanceUser !in usersToKeep) {
                                removeAt(i)
                            }
                        }
                    }
                }

fun <T> List<T>.randomSubLists(howMany: Int, ofSize: Int, random: Random): Sequence<List<T>> {
    return if (ofSize == size) {
        return sequenceOf(this)
    } else generateSequence {
        random.intSequence(indices).distinct().take(ofSize).map { this[it] }.toList()
    }.distinct().take(howMany)
}

fun Random.intSequence(range: IntRange) = generateSequence { range.start + nextInt(range.endInclusive - range.start) }

fun Instance.numericalValues(): DoubleArray = (0 until numAttributes())
        .filter { attribute(it).isNumeric }
        .map { value(it) }
        .toDoubleArray()

private fun loadDataFromFile(input: File, classAttr: String = "sampleClass"): Instances {
    val data = ConverterUtils.getLoaderForFile(input).apply {
        setSource(input)
    }.dataSet.apply {
        randomize(Random())
        setClass(attribute(classAttr))
    }
    return data
}

fun models(classifierMode: ClassifierMode): List<Pair<String, Classifier>> {
    val all = listOf<Pair<String, Classifier>>(
            "RF" to RandomForest().apply {
                numIterations = 50
                numFeatures = 10 // -K 10 (Number of attributes to randomly investigate)
                maxDepth = 25
            },
            "J48" to J48(),
            "IB3" to IBk().apply {
                knn = 3
            },
            "NB" to NaiveBayes(),
            "MLP" to MultilayerPerceptron()
    )
    val rfs = listOf<Pair<String, Classifier>>(
            "RF1_default" to RandomForest().apply {
                numIterations = 50
                numFeatures = 10 // -K 10 (Number of attributes to randomly investigate)
                maxDepth = 25
            },
            "RF2" to RandomForest().apply {
                numIterations = 200 // Give it time
                numFeatures = Int.MAX_VALUE // No limit, consider all attributes at all nodes
                maxDepth = 0 // No limit
            },
            "RF3" to RandomForest().apply {
                numIterations = 200 // Give it time
                numFeatures = Int.MAX_VALUE // No limit, consider all attributes at all nodes
                maxDepth = 50 // Like above, but limit depth (avoid overfitting)
            },
            "RF4" to RandomForest().apply {
                numIterations = 100
                numFeatures = 0 // Will use a default value
                maxDepth = 50
            },
            "RF5" to RandomForest().apply {
                numIterations = 200 // Give it time
                numFeatures = Int.MAX_VALUE // No limit, consider all attributes at all nodes
                maxDepth = 50 // No limit
            },
            "RF6" to RandomForest().apply {
                numIterations = 150
                numFeatures = 0 // Will use a default value
                maxDepth = 50
            },
            "RF7" to RandomForest().apply {
                numIterations = 100
                numFeatures = 0 // Will use a default value
                maxDepth = 75
            },
            "RF8" to RandomForest().apply {
                numIterations = 75
                numFeatures = 0 // Will use a default value
                maxDepth = 50
            },
            "RF9" to RandomForest().apply {
                numIterations = 100
                numFeatures = 0 // Will use a default value
                maxDepth = 30
            },
            "RF10" to RandomForest().apply {
                numIterations = 75
                numFeatures = Int.MAX_VALUE
                maxDepth = 50
            }
    )

    return when (classifierMode) {
        ClassifierMode.RF -> all.filter { it.first == "RF" }
        ClassifierMode.MULTIPLE_RF -> rfs
        ClassifierMode.ALL -> all
    }

}