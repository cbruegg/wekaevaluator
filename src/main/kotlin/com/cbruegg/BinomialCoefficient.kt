package com.cbruegg

fun binomialCoefficient(n: Int, k: Int) =
        Math.round((1..k).asSequence().map { (n + 1 - it) / it.toDouble() }.product())

fun Sequence<Double>.product() = reduce { l, r -> l * r }