package com.cbruegg

import weka.attributeSelection.CfsSubsetEval
import weka.attributeSelection.GreedyStepwise
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.meta.AttributeSelectedClassifier
import weka.classifiers.trees.RandomForest
import weka.core.Instances
import java.io.File
import java.util.*
import kotlin.concurrent.thread

enum class ClassifierMode {
    RF, MULTIPLE_RF, ALL
}

sealed class ValidateMode(val useFeatureSelection: Boolean, val classifierMode: ClassifierMode, val evalConvergence: Boolean) {
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
        println("$FLAG_USE_FEATURE_SELECTION will perform initial feature selection.")
        println("$FLAG_USE_ALL_CLASSIFIERS will use all classifiers instead of just RF.")
        println("$FLAG_PERSONAL_MODEL will perform cross-validation on models for one user only.")
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

    // Quick and dirty username prediction implementation
    if (FLAG_PREDICT_USER in args) {
        if (input.isDirectory) {
            input.listFiles()
                    .filter { it.absolutePath.endsWith(".csv") }
                    .forEach(::evaluateUserPrediction)
        } else {
            evaluateUserPrediction(input)
        }

        return
    }

    // Ok, so this isn't username prediction..
    val validateMode =
            if (evalPerUser) {
                ValidateMode.PerUserCrossValidation(useFeatureSelection, classifierMode, evalConvergence, personal)
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

/**
 * Make this classifier a classifier with initial feature selection.
 */
fun Classifier.toAttributeSelectedClassifier() = AttributeSelectedClassifier().apply {
    classifier = this@toAttributeSelectedClassifier
    evaluator = CfsSubsetEval()
    search = GreedyStepwise().apply {
        searchBackwards = true
    }
}

/**
 * Evaluate the username prediction capabilities.
 */
private fun evaluateUserPrediction(file: File) {
    val fullDataset = loadDataFromFile(file, classAttr = "username")
    fullDataset.deleteAttributeAt(fullDataset.attribute("sampleClass").index())

    val classifier = RandomForest().apply {
        numIterations = 50
        numFeatures = 10 // -K 10 (Number of attributes to randomly investigate)
        maxDepth = 25
    }

    val eval = Evaluation(fullDataset).apply {
        crossValidateModel(classifier, fullDataset, 10, Random(1))
    }

    println("=== Results of username prediction with file $file ===")
    println(eval.toSummaryString("", false))
    println("=== Confusion Matrix ===")
    println(eval.toMatrixString(""))
}

private fun evaluate(input: File, validateMode: ValidateMode): String {
    val resultsByClassifier = mutableMapOf<String, String>()
    val threads = mutableListOf<Thread>()

    for ((description, baseClassifier) in classifiers(validateMode.classifierMode)) {
        val classifier = if (validateMode.useFeatureSelection)
            baseClassifier.toAttributeSelectedClassifier()
        else baseClassifier

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
                                classifier.buildClassifier(dataFromUser)
                                crossValidateModel(classifier, dataFromUser, 10, Random(1))
                            }
                        }
                    }
                    is ValidateMode.UserCrossValidation -> {
                        data.deleteAttributeAt(usernameAttrIndex)
                        classifier.buildClassifier(data)

                        Evaluation(data).apply {
                            crossValidateModel(classifier, data, emptyArray(), Random(1)) {
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
            val userByInstanceHash = fullDataset.associate {
                Arrays.hashCode(it.numericalValues()) to it.stringValue(usernameAttrIndex)
            }

            resultsByClassifier[description] =
                    if (validateMode is ValidateMode.PerUserCrossValidation) {
                        performPerUserCrossValidation(description, fullDataset, classifier, userByInstanceHash, users, validateMode)
                    } else if (validateMode.evalConvergence) {
                        val reducedDatasets = fullDataset.generateSubsets(users, userByInstanceHash,
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
    return resultsByClassifier.entries.sortedBy {
        it.key
    }.map {
        it.value
    }.joinToString(separator = "", prefix = "Now evaluating file $input.\n")
}

private fun performPerUserCrossValidation(description: String,
                                          fullDataset: Instances,
                                          classifier: Classifier,
                                          userByInstanceHash: Map<Int, String>,
                                          users: List<String>,
                                          validateMode: ValidateMode.PerUserCrossValidation): String {
    return users
            .map { user ->
                val dataWithoutUser = Instances(fullDataset).apply {
                    for (i in indices.reversed()) {
                        val instanceUser = userByInstanceHash[Arrays.hashCode(this[i].numericalValues())]
                        if (instanceUser == user) {
                            removeAt(i)
                        }
                    }
                }
                val dataFromUser = Instances(fullDataset).apply {
                    for (i in indices.reversed()) {
                        val instanceUser = userByInstanceHash[Arrays.hashCode(this[i].numericalValues())]
                        if (instanceUser != user) {
                            removeAt(i)
                        }
                    }
                }

                val eval = if (validateMode.personal) {
                    Evaluation(dataFromUser).apply {
                        crossValidateModel(classifier, dataFromUser, 10, Random(1))
                    }
                } else {
                    classifier.buildClassifier(dataWithoutUser)
                    Evaluation(dataWithoutUser).apply {
                        evaluateModel(classifier, dataFromUser)
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
}