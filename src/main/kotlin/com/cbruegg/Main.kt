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
    val input = File(args[0])

    if (input.isDirectory) {
        input.listFiles().forEach(::evaluate)
    } else {
        evaluate(input)
    }
}

private fun evaluate(input: File) {
    println("Now evaluation file $input.")

    val resultsByModel = mutableMapOf<String, String>()
    val threads = mutableListOf<Thread>()
    for ((description, model) in models()) {
        threads += thread {
            val data = ConverterUtils.getLoaderForFile(input).apply {
                setSource(input)
            }.dataSet
            data.randomize(Random())
            data.setClass(data.attribute("sampleClass"))
            val eval = Evaluation(data)

            model.buildClassifier(data)
            eval.crossValidateModel(model, data, 10, Random(1))

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