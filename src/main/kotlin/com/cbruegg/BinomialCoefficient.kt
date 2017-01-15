package com.cbruegg

fun binomialCoefficient(n: Int, k: Int) =
        (1..k).asSequence().map { (n + 1 - it) / it.toLong() }.product()

fun Sequence<Long>.product() = reduce { l, r -> l * r }