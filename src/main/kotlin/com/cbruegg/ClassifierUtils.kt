package com.cbruegg

import weka.attributeSelection.CfsSubsetEval
import weka.attributeSelection.GreedyStepwise
import weka.classifiers.Classifier
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.lazy.IBk
import weka.classifiers.meta.AttributeSelectedClassifier
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest

fun classifiers(classifierMode: ClassifierMode): List<Pair<String, Classifier>> {
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