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

sealed class ValidateMode(val useFeatureSelection: Boolean, val useAllClassifiers: Boolean) {
   class RandomCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: Boolean) : ValidateMode(useFeatureSelection, useAllClassifiers)
   class UserCrossValidation(useFeatureSelection: Boolean, useAllClassifiers: Boolean) : ValidateMode(useFeatureSelection, useAllClassifiers)
}

private const val FLAG_RANDOM_CROSS_VALIDATION = "--random-cross-validation"
private const val FLAG_USE_FEATURE_SELECTION = "--feature-selection"
private const val FLAG_USE_ALL_CLASSIFIERS = "--use-all-classifiers"

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Usage: java -jar xxx.jar <trainingset> [$FLAG_RANDOM_CROSS_VALIDATION] [$FLAG_USE_FEATURE_SELECTION] [$FLAG_USE_ALL_CLASSIFIERS]")
        println("$FLAG_RANDOM_CROSS_VALIDATION will make cross validation be performed with random instances being taken out. Otherwise users will be taken out.")
        println("$FLAG_USE_FEATURE_SELECTION will perform initial feature selection.")
        println("$FLAG_USE_ALL_CLASSIFIERS will use all classifiers instead of just RF.")
        return
    }

    val input = File(args[0])
    val useFeatureSelection = FLAG_USE_FEATURE_SELECTION in args
    val useAllClassifiers = FLAG_USE_ALL_CLASSIFIERS in args
    val validateMode = if (FLAG_RANDOM_CROSS_VALIDATION in args) {
        ValidateMode.RandomCrossValidation(useFeatureSelection, useAllClassifiers)
    } else ValidateMode.UserCrossValidation(useFeatureSelection, useAllClassifiers)

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
            val data = loadDataFromFile(input)

            val usernameAttrIndex = data.attribute("username").index()
            val eval = when (validateMode) {
                is ValidateMode.RandomCrossValidation -> {
                    data.deleteAttributeAt(usernameAttrIndex)
                    model.buildClassifier(data)
                    Evaluation(data).apply {
                        crossValidateModel(model, data, 10, Random(1))
                    }
                }
                is ValidateMode.UserCrossValidation -> {
                    val users = data.attribute("username")
                            .enumerateValues()
                            .asSequence()
                            .distinct()
                            .filterIsInstance<String>()
                            .toList()
                    val userByInstance = data.associate {
                        Arrays.hashCode(it.numericalValues()) to it.stringValue(usernameAttrIndex)
                    }
                    data.deleteAttributeAt(usernameAttrIndex)
                    model.buildClassifier(data)

                    Evaluation(data).apply {
                        crossValidateModel(model, data, emptyArray(), Random(1)) {
                            users.indexOf(userByInstance[Arrays.hashCode(it.numericalValues())]!!)
                        }
                    }
                }
            }

            resultsByModel[description] = """
            |+++ TRAINING $description +++
            |=== Results of $description ===
            |${eval.toSummaryString("", false)}
            |=== Confusion Matrix of $description ===
            |${eval.toMatrixString("")}
            |""".trimMargin()
        }
    }
    threads.forEach(Thread::join)
    return resultsByModel.entries.sortedBy {
        it.key
    }.map {
        it.value
    }.joinToString(separator = "", prefix = "Now evaluating file $input.\n")
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

// TODO These models will need some fine-tuning
// TODO Personal model evaluation
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