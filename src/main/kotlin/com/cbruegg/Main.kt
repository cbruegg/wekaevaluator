package com.cbruegg

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.lazy.IBk
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
import weka.core.Instance
import weka.core.Instances
import weka.core.converters.ConverterUtils
import java.io.File
import java.util.*
import kotlin.concurrent.thread

sealed class ValidateMode {
    object RandomCrossValidation : ValidateMode()
    object UserCrossValidation : ValidateMode()
    class ValidationAgainst(val testFile: File) : ValidateMode()
}

private const val FLAG_USER_VALIDATION = "--uservalidation"

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Usage: java -jar xxx.jar <trainingset> <optional testset>")
        println("If no testset is provided, cross validation is used.")
        println("OR: java -jar xxx.jar <trainingset> $FLAG_USER_VALIDATION")
        println("In this case, cross validation will be performed with users being taken out.")
        return
    }

    val input = File(args[0])
    val validateMode = if (1 in args.indices) {
        if (args[1] == FLAG_USER_VALIDATION) {
            ValidateMode.UserCrossValidation
        } else {
            ValidateMode.ValidationAgainst(File(args[1]))
        }
    } else ValidateMode.RandomCrossValidation

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

private fun evaluate(input: File, validateMode: ValidateMode): String {
    val resultsByModel = mutableMapOf<String, String>()
    val threads = mutableListOf<Thread>()
    for ((description, model) in models()) {
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
                is ValidateMode.ValidationAgainst -> {
                    data.deleteAttributeAt(usernameAttrIndex)
                    val testFile = validateMode.testFile
                    val testSet = loadDataFromFile(testFile)
                    testSet.deleteAttributeAt(usernameAttrIndex)
                    model.buildClassifier(data)
                    Evaluation(data).apply {
                        evaluateModel(model, testSet)
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
fun models() = listOf<Pair<String, Classifier>>(
        "RF" to RandomForest().apply {
            // -K 0
            numIterations = 100
            numFeatures = 100
        },
        "J48" to J48(),
        "IB3" to IBk().apply {
            knn = 3
        },
        "NB" to NaiveBayes()
//        "MLP" to MultilayerPerceptron()
)
