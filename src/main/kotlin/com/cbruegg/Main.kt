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

sealed class ValidateMode(val useFeatureSelection: Boolean, val useAllClassifiers: Boolean, val evalConvergence: Boolean) {
    class RandomCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: Boolean, evalConvergence: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)

    class UserCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: Boolean, evalConvergence: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)

    class PersonalRandomCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: Boolean, evalConvergence: Boolean) :
            ValidateMode(useFeatureSelection, useAllClassifiers, evalConvergence)
}

private const val FLAG_RANDOM_CROSS_VALIDATION = "--random-cross-validation"
private const val FLAG_USE_FEATURE_SELECTION = "--feature-selection"
private const val FLAG_USE_ALL_CLASSIFIERS = "--use-all-classifiers"
private const val FLAG_PERSONAL_MODEL = "--personal"
private const val FLAG_ACCURACY_CONVERGENCE = "--eval-convergence"

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Usage: java -jar xxx.jar <trainingset> <flags>")
        println("$FLAG_RANDOM_CROSS_VALIDATION will make cross validation be performed with random instances being taken out. Otherwise users will be taken out.")
        println("$FLAG_USE_FEATURE_SELECTION will perform initial feature selection.")
        println("$FLAG_USE_ALL_CLASSIFIERS will use all classifiers instead of just RF.")
        println("$FLAG_PERSONAL_MODEL will perform cross-validation on models for one user only. Cannot be used in conjunction with $FLAG_RANDOM_CROSS_VALIDATION")
        println("$FLAG_ACCURACY_CONVERGENCE will evaluate the model with 1 to N users.")
        return
    }

    val input = File(args[0])
    val useFeatureSelection = FLAG_USE_FEATURE_SELECTION in args
    val useAllClassifiers = FLAG_USE_ALL_CLASSIFIERS in args
    val evalConvergence = FLAG_ACCURACY_CONVERGENCE in args
    val validateMode = if (FLAG_RANDOM_CROSS_VALIDATION in args) {
        ValidateMode.RandomCrossValidation(useFeatureSelection, useAllClassifiers, evalConvergence)
    } else if (FLAG_PERSONAL_MODEL in args) {
        ValidateMode.PersonalRandomCrossValidation(useFeatureSelection, useAllClassifiers, evalConvergence)
    } else ValidateMode.UserCrossValidation(useFeatureSelection, useAllClassifiers, evalConvergence)

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

private fun evaluate(input: File, validateMode: ValidateMode): String {
    val resultsByModel = mutableMapOf<String, String>()
    val threads = mutableListOf<Thread>()
    for ((description, baseModel) in models(validateMode.useAllClassifiers)) {
        val model = if (validateMode.useFeatureSelection)
            baseModel.toAttributeSelectedClassifier()
        else baseModel

        threads += thread {
            val fullDataset = loadDataFromFile(input)

            val usernameAttrIndex = fullDataset.attribute("username").index()
            val users = fullDataset.attribute("username")
                    .enumerateValues()
                    .asSequence()
                    .distinct()
                    .filterIsInstance<String>()
                    .toList()
            val userByInstance = fullDataset.associate {
                Arrays.hashCode(it.numericalValues()) to it.stringValue(usernameAttrIndex)
            }

            fun eval(data: Instances) = when (validateMode) {
                is ValidateMode.PersonalRandomCrossValidation -> {
                    Evaluation(data).apply {
                        for (user in users) {
                            val filteredData = Instances(data).apply {
                                for (i in indices.reversed()) {
                                    val instanceUser = userByInstance[Arrays.hashCode(this[i].numericalValues())]
                                    if (instanceUser != user) {
                                        removeAt(i)
                                    }
                                }
                            }
                            model.buildClassifier(filteredData)
                            crossValidateModel(model, filteredData, 10, Random(1))
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
            }

            val reducedDatasets =
                    if (validateMode.evalConvergence) generateSubsets(fullDataset, users, userByInstance)
                    else sequenceOf(fullDataset)
            val datasetsEvals = reducedDatasets.map(::eval)

            resultsByModel[description] = datasetsEvals.mapIndexed { i, eval ->
                """
            |+++ TRAINING $description ${if (validateMode.evalConvergence) "with subset of size ${i + 1} " else ""}+++
            |=== Results of $description ===
            |${eval.toSummaryString("", false)}
            |=== Confusion Matrix of $description ===
            |${eval.toMatrixString("")}
            |""".trimMargin()
            }.joinToString(separator = "\n")
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
 * method will generate [Instances] with users {u1, u2}, ... {u1, ..., un}.
 *
 * @param [userByInstance] val instanceUser = userByInstance[Arrays.hashCode(instance.numericalValues())]
 */
fun generateSubsets(data: Instances, users: List<String>, userByInstance: Map<Int, String>) =
        (2..users.size)
                .asSequence()
                .map { users.subList(0, it) }
                .map { usersToKeep ->
                    Instances(data).apply {
                        for (i in indices.reversed()) {
                            val instanceUser = userByInstance[Arrays.hashCode(this[i].numericalValues())]
                            if (instanceUser !in usersToKeep) {
                                removeAt(i)
                            }
                        }
                    }
                }

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

fun models(useAll: Boolean) = listOf<Pair<String, Classifier>>(
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
).let { if (useAll) it else it.filter { it.first == "RF" } }