package com.cbruegg

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.lazy.IBk
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
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
    val test = if (1 in args.indices) File(args[1]) else null
    val validateMode = if (1 in args.indices) {
        if (args[1] == FLAG_USER_VALIDATION) {
            ValidateMode.UserCrossValidation
        } else {
            ValidateMode.ValidationAgainst(File(args[1]))
        }
    } else ValidateMode.RandomCrossValidation

    if (input.isDirectory) {
        input.listFiles().filter { it.endsWith(".csv") }.forEach { evaluate(it, validateMode) }
    } else {
        evaluate(input, validateMode)
    }
}

private fun evaluate(input: File, validateMode: ValidateMode) {
    println("Now evaluating file $input.")

    val resultsByModel = mutableMapOf<String, String>()
    val threads = mutableListOf<Thread>()
    for ((description, model) in models()) {
        threads += thread {
            val data = ConverterUtils.getLoaderForFile(input).apply {
                setSource(input)
            }.dataSet.apply {
                randomize(Random())
                setClass(attribute("sampleClass"))
            }
            val eval = Evaluation(data)

            model.buildClassifier(data)

            val usernameAttrIndex = data.attribute("username").index()
            when (validateMode) {
                is ValidateMode.RandomCrossValidation -> {
                    data.deleteAttributeAt(usernameAttrIndex)
                    eval.crossValidateModel(model, data, 10, Random(1))
                }
                is ValidateMode.UserCrossValidation -> {
                    val users = data.attribute("username")
                            .enumerateValues()
                            .asSequence()
                            .distinct()
                            .filterIsInstance<String>()
                    val bestUser = users.maxBy {
                        evaluateWithoutUser(data, eval, model, it)
                        eval.pctCorrect()
                    } ?: throw IllegalArgumentException("Dataset is empty or contains no users.")
                    // Restore evaluation of best user
                    evaluateWithoutUser(data, eval, model, bestUser)
                }
                is ValidateMode.ValidationAgainst -> {
                    data.deleteAttributeAt(usernameAttrIndex)
                    val testFile = validateMode.testFile
                    val testSet = ConverterUtils.getLoaderForFile(testFile).apply {
                        setSource(testFile)
                    }.dataSet.apply {
                        randomize(Random())
                        setClass(attribute("sampleClass"))
                    }
                    eval.evaluateModel(model, testSet)
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
    resultsByModel.entries.sortedBy { it.key }.map { it.value }.forEach(::print)
}

private fun evaluateWithoutUser(data: Instances, eval: Evaluation, model: Classifier, user: String) {
    val usernameAttrIndex = data.attribute("username").index()

    val dataCopy = Instances(data)
    dataCopy.removeAll { it.stringValue(usernameAttrIndex) == user }
    dataCopy.deleteAttributeAt(usernameAttrIndex)

    val testSet = Instances(data)
    testSet.retainAll { it.stringValue(usernameAttrIndex) == user }
    testSet.deleteAttributeAt(usernameAttrIndex)

    eval.evaluateModel(model, testSet)
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