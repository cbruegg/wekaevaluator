package com.cbruegg

import weka.core.Instance
import weka.core.Instances
import weka.core.converters.ConverterUtils
import java.io.File
import java.util.*

/**
 * Return a [DoubleArray] containing only
 * the numeric attribute values of this instance.
 */
fun Instance.numericalValues(): DoubleArray = (0 until numAttributes())
        .filter { attribute(it).isNumeric }
        .map { value(it) }
        .toDoubleArray()

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

fun loadDataFromFile(input: File, classAttr: String = "sampleClass"): Instances {
    val data = ConverterUtils.getLoaderForFile(input).apply {
        setSource(input)
    }.dataSet.apply {
        randomize(Random())
        setClass(attribute(classAttr))
    }
    return data
}

/**
 * Get the set of usernames from the dataset.
 */
fun Instances.extractUsers(): List<String> = map { it.stringValue(attribute("username")) }.distinct()