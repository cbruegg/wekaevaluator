package com.cbruegg

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.lazy.IBk
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
import weka.core.converters.ConverterUtils
import java.io.File
import java.util.*
import kotlin.concurrent.thread

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Usage: java -jar xxx.jar <trainingset> <optional testset>")
        println("If no testset is provided, cross validation is used.")
        return
    }

    val input = File(args[0])
    val test = if (1 in args.indices) File(args[1]) else null

    if (input.isDirectory) {
        input.listFiles().filter { it.endsWith(".csv") }.forEach { evaluate(it, test) }
    } else {
        evaluate(input, test)
    }
}

private fun evaluate(input: File, testFile: File?) {
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
            if (testFile == null) {
                eval.crossValidateModel(model, data, 10, Random(1))
            } else {
                val testSet = ConverterUtils.getLoaderForFile(testFile).apply {
                    setSource(testFile)
                }.dataSet.apply {
                    randomize(Random())
                    setClass(attribute("sampleClass"))
                }
                eval.evaluateModel(model, testSet)
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